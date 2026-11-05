// Chapter 14 — Spectrum. A progressive spectral path tracer running in WebGPU compute.
//
// Light transport is split by light source so nothing is counted twice:
//  • the collimated white beam (a delta light that camera paths can never hit) is traced from the light:
//    single-wavelength photons through Sellmeier glass with exact Fresnel / TIR, deposited into a caustic
//    light-map atlas over every diffuse surface (atomic fixed-point XYZ) and splatted onto the image as single
//    scattering in the haze (light tracing with camera connections);
//  • the dim gallery lights are handled by camera paths: hero-wavelength spectral path tracing with NEE + MIS,
//    Russian roulette, GGX polished floor, and the light-map gathered at diffuse lobes (so the rainbow is seen
//    through glass and in the floor's reflection).
// Progressive running average + variance-guided à-trous denoiser (normal / plane-distance / luminance stops).

import * as THREE from 'three/webgpu';
import {
	Fn, wgslFn, wgsl, uniform, instancedArray, instanceIndex, uint, int, float, vec2, vec3, vec4, If, Loop, Break, Return,
	atomicAdd, atomicLoad, atomicStore, texture, textureStore, uvec2, screenUV, mix, max, sqrt, exp, abs, dot, pow, clamp,
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { buildWGSL, UNIFORMS, LM_SCALE, SCR_SCALE } from './prism/wgsl.js';
import { PRESETS, ROOM, buildAtlas } from './prism/geometry.js';
import { GLASSES, LIGHTS, blackbodyNorm, blackbodyXYZ, linesXYZ, LINES, sellmeier } from './prism/spectral.js';

const TIERS = [
	{ maxPix: 76000, spp: 1, bounces: 5, photons: 1 << 14, lmDensity: 34, maxD: 1, steps: 7, atrous: 4 },
	{ maxPix: 400000, spp: 1, bounces: 6, photons: 1 << 17, lmDensity: 70, maxD: 2, steps: 9, atrous: 4 },
	{ maxPix: 1000000, spp: 2, bounces: 8, photons: 1 << 19, lmDensity: 105, maxD: 2, steps: 10, atrous: 5 },
];

const LUMA = vec3( 0.2126, 0.7152, 0.0722 );

export default {
	id: 'prism',
	title: 'Spectrum',
	subtitle: 'A spectral path tracer: one beam of white light, Sellmeier glass and Fresnel’s equations — every rainbow here is computed wavelength by wavelength.',

	async create( ctx ) {

		const { renderer, camera, controls, params } = ctx;
		const tier = Math.max( 0, Math.min( 2, ctx.tier ) );
		const cfg = { ...TIERS[ tier ] };
		if ( params.has( 'spp' ) ) cfg.spp = Math.max( 1, parseInt( params.get( 'spp' ) ) || 1 );
		if ( params.has( 'photons' ) ) cfg.photons = Math.max( 1024, parseInt( params.get( 'photons' ) ) || cfg.photons );

		const scene = new THREE.Scene();

		// ── resolution & capacity ──
		const TW = Math.ceil( Math.sqrt( cfg.maxPix * 2.4 ) ), TH = TW;
		const CAP = cfg.maxPix;
		const res = { W: 1, H: 1 };
		function pickRes( w, h ) {

			const s = Math.min( 1, Math.sqrt( cfg.maxPix / Math.max( 1, w * h ) ) );
			let W = Math.max( 16, Math.floor( w * s ) ), H = Math.max( 16, Math.floor( h * s ) );
			if ( W > TW ) {

				H = Math.floor( H * TW / W ); W = TW;

			}

			if ( H > TH ) {

				W = Math.floor( W * TH / H ); H = TH;

			}

			while ( W * H > CAP ) {

				W --; H = Math.floor( W * h / w );

			}

			res.W = W; res.H = H;

		}

		pickRes( ctx.size.x, ctx.size.y );

		// ── light-map atlas ──
		const atlas = buildAtlas( cfg.lmDensity );
		const LMN = atlas.W * atlas.H;
		const src = buildWGSL( atlas, { maxDiffusePhoton: cfg.maxD } );

		// ── uniforms ──
		const U = {};
		for ( const k of UNIFORMS ) U[ k ] = uniform( new THREE.Vector4() );
		const uWi = uniform( res.W, 'uint' );
		const uNPix = uniform( res.W * res.H, 'uint' );
		const uFrame = uniform( 0, 'uint' ); // accumulated frames since reset
		const uSeed = uniform( 0, 'uint' );
		const uBlend = uniform( 1 ); // 1 / (n + 1)
		const uNf = uniform( 1 ); // n + 1 as float
		const uLmAlpha = uniform( 1 );
		const uFade = uniform( 0 );
		const uExposure = uniform( 1 );
		const uPixAngle = uniform( 0.002 );
		const uSigL = uniform( 4.0 );
		const uClamp = uniform( 40 );
		const uUV = uniform( new THREE.Vector2( 1, 1 ) );

		// ── buffers ──
		const accBuf = instancedArray( CAP, 'vec4' ); // rgb mean, mean of luminance²
		const gN = instancedArray( CAP, 'vec4' ); // primary normal, distance
		const gP = instancedArray( CAP, 'vec4' ); // primary position, material
		const dA = instancedArray( CAP, 'vec4' );
		const dB = instancedArray( CAP, 'vec4' );
		const scr = instancedArray( CAP * 3, 'uint' ).toAtomic(); // per-frame haze splats (XYZ fixed point)
		const lmAtom = instancedArray( LMN * 3, 'uint' ).toAtomic(); // per-frame photon deposits
		const lmAcc = instancedArray( LMN, 'vec4' ); // running mean irradiance (XYZ)
		const buffers = [ accBuf, gN, gP, dA, dB, scr, lmAtom, lmAcc ];

		const lmTex = new THREE.StorageTexture( atlas.W, atlas.H );
		lmTex.type = THREE.HalfFloatType;
		lmTex.minFilter = lmTex.magFilter = THREE.LinearFilter;
		lmTex.generateMipmaps = false;
		const outTex = new THREE.StorageTexture( TW, TH );
		outTex.type = THREE.HalfFloatType;
		outTex.minFilter = outTex.magFilter = THREE.LinearFilter;
		outTex.generateMipmaps = false;

		// ── WGSL functions ──
		const lib = wgsl( src.lib );
		const eyeFn = wgslFn( src.eye, [ lib ] );
		const photonFn = wgslFn( src.photon, [ lib ] );
		const uArgs = () => {

			const o = {};
			for ( const k of UNIFORMS ) o[ k ] = U[ k ];
			return o;

		};

		// ── photon (light) tracing ──
		const photonK = Fn( () => {

			const seed = instanceIndex.add( uSeed ).toVar();
			const ro = vec4( 0 ).toVar(), rd = vec4( 0 ).toVar();
			const st = vec4( 0, 0, 1, 0 ).toVar();
			const spl = vec4( - 1 ).toVar(), sps = vec4( - 1 ).toVar();
			Loop( cfg.steps, () => {

				const alive = photonFn( { ro, rd, st, seed, spl, sps, ...uArgs() } ).toVar();
				If( spl.x.greaterThanEqual( 0 ), () => {

					const b = uint( spl.x ).mul( 3 );
					atomicAdd( lmAtom.element( b ), uint( spl.y ) );
					atomicAdd( lmAtom.element( b.add( 1 ) ), uint( spl.z ) );
					atomicAdd( lmAtom.element( b.add( 2 ) ), uint( spl.w ) );

				} );
				If( sps.x.greaterThanEqual( 0 ), () => {

					const b = uint( sps.x ).mul( 3 );
					atomicAdd( scr.element( b ), uint( sps.y ) );
					atomicAdd( scr.element( b.add( 1 ) ), uint( sps.z ) );
					atomicAdd( scr.element( b.add( 2 ) ), uint( sps.w ) );

				} );
				If( alive.lessThan( 0.5 ), () => {

					Break();

				} );

			} );

		} )().compute( cfg.photons ).setName( 'prismPhotons' );

		// ── light-map resolve: fixed point → running mean → half-float atlas texture ──
		const lmK = Fn( () => {

			const i = instanceIndex;
			const b = i.mul( 3 );
			const x = float( atomicLoad( lmAtom.element( b ) ) ).toVar();
			const y = float( atomicLoad( lmAtom.element( b.add( 1 ) ) ) ).toVar();
			const z = float( atomicLoad( lmAtom.element( b.add( 2 ) ) ) ).toVar();
			atomicStore( lmAtom.element( b ), uint( 0 ) );
			atomicStore( lmAtom.element( b.add( 1 ) ), uint( 0 ) );
			atomicStore( lmAtom.element( b.add( 2 ) ), uint( 0 ) );
			const a = lmAcc.element( i );
			const v = mix( a.xyz, vec3( x, y, z ).mul( 1 / LM_SCALE ), uLmAlpha ).toVar();
			a.assign( vec4( v, 1 ) );
			textureStore( lmTex, uvec2( i.mod( uint( atlas.W ) ), i.div( uint( atlas.W ) ) ), vec4( v, 1 ) );

		} )().compute( LMN ).setName( 'prismLightmap' );

		// ── camera paths + accumulation ──
		const XYZ2RGB = ( c ) => vec3(
			dot( c, vec3( 3.2404542, - 1.5371385, - 0.4985314 ) ),
			dot( c, vec3( - 0.9692660, 1.8760108, 0.0415560 ) ),
			dot( c, vec3( 0.0556434, - 0.2040259, 1.0572252 ) ) );

		const eyeK = Fn( () => {

			const i = instanceIndex;
			If( i.greaterThanEqual( uNPix ), () => {

				Return();

			} );
			const px = float( i.mod( uWi ) ), py = float( i.div( uWi ) );
			const gd = vec4( 0, 1, 0, 1e4 ).toVar(), gp = vec4( 0 ).toVar();
			const seed = i.mul( uint( 1664525 ) ).add( uFrame.mul( uint( 1013904223 ) ) ).add( uSeed ).toVar();
			const lmNode = texture( lmTex );
			const c = eyeFn( { pix: vec2( px, py ), nidx: uFrame, seed, gd, gp, lm: lmNode, lms: lmNode, ...uArgs() } ).toVar();
			const b = i.mul( 3 );
			const mx = float( atomicLoad( scr.element( b ) ) ).toVar();
			const my = float( atomicLoad( scr.element( b.add( 1 ) ) ) ).toVar();
			const mz = float( atomicLoad( scr.element( b.add( 2 ) ) ) ).toVar();
			atomicStore( scr.element( b ), uint( 0 ) );
			atomicStore( scr.element( b.add( 1 ) ), uint( 0 ) );
			atomicStore( scr.element( b.add( 2 ) ), uint( 0 ) );
			const smp = c.xyz.add( XYZ2RGB( vec3( mx, my, mz ).mul( 1 / SCR_SCALE ) ) ).toVar();
			// firefly clamp (luminance)
			const l = dot( smp, LUMA ).toVar();
			If( l.greaterThan( uClamp ), () => {

				smp.mulAssign( uClamp.div( l ) );
				l.assign( uClamp );

			} );
			const a = accBuf.element( i );
			a.assign( mix( a, vec4( smp, l.mul( l ) ), uBlend ) );
			gN.element( i ).assign( gd );
			gP.element( i ).assign( gp );

		} )().compute( CAP ).setName( 'prismEye' );

		// ── denoiser ──
		const pixXY = () => {

			const i = instanceIndex;
			return [ int( i.mod( uWi ) ), int( i.div( uWi ) ) ];

		};

		const tapIndex = ( x, y ) => uint( y.mul( int( uWi ) ).add( x ) );
		const clampX = ( x ) => clamp( x, int( 0 ), int( uWi ).sub( 1 ) );
		const clampY = ( y ) => clamp( y, int( 0 ), int( uNPix.div( uWi ) ).sub( 1 ) );

		// variance estimate: temporal (moments) once history exists, spatial 3×3 before
		const prepK = Fn( () => {

			const i = instanceIndex;
			If( i.greaterThanEqual( uNPix ), () => {

				Return();

			} );
			const [ px, py ] = pixXY();
			const a = accBuf.element( i ).toVar();
			const l = dot( a.xyz, LUMA );
			const vt = max( a.w.sub( l.mul( l ) ), 0 ).div( uNf );
			const s1 = float( 0 ).toVar(), s2 = float( 0 ).toVar();
			for ( let dy = - 1; dy <= 1; dy ++ ) for ( let dx = - 1; dx <= 1; dx ++ ) {

				const q = accBuf.element( tapIndex( clampX( px.add( int( dx ) ) ), clampY( py.add( int( dy ) ) ) ) );
				const lq = dot( q.xyz, LUMA );
				s1.addAssign( lq ); s2.addAssign( lq.mul( lq ) );

			}

			const m = s1.div( 9 );
			const vs = max( s2.div( 9 ).sub( m.mul( m ) ), 0 ).div( uNf );
			const k = clamp( uNf.sub( 1 ).div( 4 ), 0, 1 );
			dA.element( i ).assign( vec4( a.xyz, mix( vs, vt, k ) ) );

		} )().compute( CAP ).setName( 'prismVar' );

		const atrousK = ( srcB, dstB, step ) => Fn( () => {

			const i = instanceIndex;
			If( i.greaterThanEqual( uNPix ), () => {

				Return();

			} );
			const [ px, py ] = pixXY();
			const c0 = srcB.element( i ).toVar();
			const n0 = gN.element( i ).toVar();
			const p0 = gP.element( i ).toVar();
			const l0 = dot( c0.xyz, LUMA );
			const sig = sqrt( max( c0.w, 0 ) ).mul( uSigL ).add( 1e-3 ).toVar();
			const foot = n0.w.mul( uPixAngle ).mul( step * 1.5 ).add( 2e-3 ).toVar();
			const sumC = vec3( 0 ).toVar(), sumW = float( 0 ).toVar(), sumV = float( 0 ).toVar();
			const h = [ 0.25, 0.5, 0.25 ];
			for ( let dy = - 1; dy <= 1; dy ++ ) for ( let dx = - 1; dx <= 1; dx ++ ) {

				const k = h[ dx + 1 ] * h[ dy + 1 ];
				if ( dx === 0 && dy === 0 ) {

					sumC.addAssign( c0.xyz.mul( k ) ); sumW.addAssign( k ); sumV.addAssign( c0.w.mul( k * k ) );
					continue;

				}

				const j = tapIndex( clampX( px.add( int( dx * step ) ) ), clampY( py.add( int( dy * step ) ) ) );
				const cq = srcB.element( j ).toVar();
				const nq = gN.element( j );
				const pq = gP.element( j );
				const wn = pow( max( dot( n0.xyz, nq.xyz ), 0 ), 64 );
				const wp = exp( abs( dot( n0.xyz, pq.xyz.sub( p0.xyz ) ) ).div( foot ).negate() );
				const wl = exp( abs( dot( cq.xyz, LUMA ).sub( l0 ) ).div( sig ).negate() );
				const w = wn.mul( wp ).mul( wl ).mul( k ).toVar();
				sumC.addAssign( cq.xyz.mul( w ) ); sumW.addAssign( w ); sumV.addAssign( cq.w.mul( w ).mul( w ) );

			}

			dstB.element( i ).assign( vec4( sumC.div( sumW ), sumV.div( sumW.mul( sumW ) ) ) );

		} )().compute( CAP ).setName( 'prismAtrous' + step );

		const atrous = [];
		for ( let k = 0; k < cfg.atrous; k ++ ) atrous.push( atrousK( k % 2 === 0 ? dA : dB, k % 2 === 0 ? dB : dA, 1 << k ) );
		const finalOf = ( k ) => ( k === 0 ? dA : ( k % 2 === 1 ? dB : dA ) );

		const outK = ( denB ) => Fn( () => {

			const i = instanceIndex;
			If( i.greaterThanEqual( uNPix ), () => {

				Return();

			} );
			const raw = accBuf.element( i ).xyz;
			const dn = denB.element( i ).xyz;
			const c = mix( dn, raw, uFade ).mul( uExposure ).toVar();
			// gamut map: spectral colours fall outside sRGB — desaturate toward luminance until non-negative
			const y = max( dot( c, LUMA ), 0 ).toVar();
			const mn = c.x.min( c.y ).min( c.z );
			If( mn.lessThan( 0 ), () => {

				const t = y.div( max( y.sub( mn ), 1e-6 ) );
				c.assign( vec3( y ).add( c.sub( y ).mul( t ) ) );

			} );
			textureStore( outTex, uvec2( i.mod( uWi ), i.div( uWi ) ), vec4( max( c, 0 ), 1 ) );

		} )().compute( CAP ).setName( 'prismOut' );

		const outDenoised = outK( finalOf( cfg.atrous ) );
		const outRaw = outK( dA );

		// ── settings / GUI ──
		const S = {
			preset: params.get( 'preset' ) && PRESETS[ params.get( 'preset' ) ] ? params.get( 'preset' ) : 'prism',
			glass: 'SF11 dense flint',
			dispersion: 1.0,
			light: 0,
			haze: 1.0,
			bounces: cfg.bounces,
			spp: cfg.spp,
			denoise: true,
			exposure: 1.0,
			ambient: 1.0,
		};

		const state = {
			yaw: 0, pitch: 0, beamYaw: 0, frame: 0, lmFrame: 0, total: 0, seed: 1,
			dirtyScene: true, camKey: '', lastStats: 0, dt: 1 / 60, photonsTotal: 0,
		};

		function applyPreset( name, moveCam ) {

			const P = PRESETS[ name ];
			state.yaw = P.yaw; state.pitch = P.pitch; state.beamYaw = 0;
			if ( name === 'gem' && S.glass !== 'diamond' ) S.glass = 'diamond';
			if ( name !== 'gem' && S.glass === 'diamond' ) S.glass = 'SF11 dense flint';
			if ( moveCam ) {

				camera.position.set( ...P.cam.pos );
				controls.target.set( ...P.cam.target );
				controls.update();

			}

			state.dirtyScene = true;
			gui.controllersRecursive().forEach( ( c ) => c.updateDisplay() );

		}

		const gui = ctx.gui;
		gui.add( S, 'preset', Object.keys( PRESETS ) ).name( 'scene' ).onChange( ( v ) => applyPreset( v, true ) );
		gui.add( S, 'glass', Object.keys( GLASSES ) ).name( 'glass' ).onChange( () => state.dirtyScene = true );
		gui.add( S, 'dispersion', 0, 3, 0.01 ).name( 'dispersion ×' ).onChange( () => state.dirtyScene = true );
		gui.add( S, 'light', Object.fromEntries( LIGHTS.map( ( l, i ) => [ l.name, i ] ) ) ).name( 'light (click)' ).onChange( () => state.dirtyScene = true );
		gui.add( S, 'haze', 0, 3, 0.01 ).name( 'haze density' ).onChange( () => state.dirtyCam = true );
		gui.add( S, 'bounces', 2, 12, 1 ).name( 'bounces' ).onChange( () => state.dirtyCam = true );
		gui.add( S, 'spp', 1, 8, 1 ).name( 'samples / frame' ).onChange( () => state.dirtyCam = true );
		gui.add( S, 'denoise' ).name( 'denoiser' );
		gui.add( S, 'exposure', 0.1, 4, 0.01 ).name( 'exposure' );
		gui.add( S, 'ambient', 0, 4, 0.01 ).name( 'gallery lights' ).onChange( () => state.dirtyCam = true );

		// ── camera ──
		camera.fov = 44; camera.near = 0.05; camera.far = 100; camera.updateProjectionMatrix();
		controls.enableDamping = true;
		controls.minDistance = 1.0; controls.maxDistance = 6.5;
		controls.maxPolarAngle = Math.PI * 0.53;
		controls.minAzimuthAngle = - 1.1; controls.maxAzimuthAngle = 1.1;
		controls.autoRotate = false;
		applyPreset( S.preset, true );

		// ── pointer: drag on the glass rotates it (or sweeps the beam), click cycles the light ──
		const dom = renderer.domElement;
		const drag = { active: false, x: 0, y: 0, sx: 0, sy: 0, t: 0, moved: 0 };
		const ray = new THREE.Raycaster();
		const v2 = new THREE.Vector2();
		function hitsObject( e ) {

			const rect = dom.getBoundingClientRect();
			v2.set( ( ( e.clientX - rect.left ) / rect.width ) * 2 - 1, - ( ( e.clientY - rect.top ) / rect.height ) * 2 + 1 );
			ray.setFromCamera( v2, camera );
			const P = PRESETS[ S.preset ];
			const c = new THREE.Vector3( ...P.objPos );
			if ( S.preset === 'prism' ) c.y += 0.23;
			if ( S.preset === 'spheres' ) c.set( 0, 0.3, 0.2 );
			return ray.ray.distanceSqToPoint( c ) < P.bound * P.bound;

		}

		const onDown = ( e ) => {

			drag.sx = drag.x = e.clientX; drag.sy = drag.y = e.clientY; drag.t = performance.now(); drag.moved = 0;
			if ( e.button === 0 && hitsObject( e ) ) {

				drag.active = true;
				controls.enabled = false;

			}

		};

		const onMove = ( e ) => {

			drag.moved = Math.max( drag.moved, Math.hypot( e.clientX - drag.sx, e.clientY - drag.sy ) );
			if ( ! drag.active ) return;
			const dx = ( e.clientX - drag.x ) / Math.max( 300, dom.clientWidth ) * Math.PI * 1.2;
			const dy = ( e.clientY - drag.y ) / Math.max( 300, dom.clientHeight ) * Math.PI * 1.2;
			drag.x = e.clientX; drag.y = e.clientY;
			if ( S.preset === 'spheres' ) state.beamYaw += dx * 0.5;
			else {

				state.yaw += dx;
				if ( S.preset === 'gem' ) state.pitch = Math.max( - 1.4, Math.min( 1.4, state.pitch + dy ) );

			}

			state.dirtyScene = true;

		};

		const onUp = ( e ) => {

			if ( drag.moved < 5 && performance.now() - drag.t < 400 && e.button === 0 ) {

				S.light = ( S.light + 1 ) % LIGHTS.length;
				state.dirtyScene = true;
				gui.controllersRecursive().forEach( ( c ) => c.updateDisplay() );

			}

			if ( drag.active ) {

				drag.active = false;
				controls.enabled = true;

			}

		};

		dom.addEventListener( 'pointerdown', onDown, { capture: true } );
		addEventListener( 'pointermove', onMove );
		addEventListener( 'pointerup', onUp );

		// ── uniform upload ──
		const tmpM = new THREE.Matrix4(), tmpE = new THREE.Euler();
		function sceneUniforms() {

			const P = PRESETS[ S.preset ];
			// object rotation (object → world), rows into o0..o2, translation in .w
			tmpE.set( state.pitch, state.yaw, 0, 'YXZ' );
			tmpM.makeRotationFromEuler( tmpE );
			const e = tmpM.elements; // column-major
			U.o0.value.set( e[ 0 ], e[ 4 ], e[ 8 ], P.objPos[ 0 ] );
			U.o1.value.set( e[ 1 ], e[ 5 ], e[ 9 ], P.objPos[ 1 ] );
			U.o2.value.set( e[ 2 ], e[ 6 ], e[ 10 ], P.objPos[ 2 ] );
			U.s0.value.set( P.id, P.plinth ? 1 : 0, P.objPos[ 0 ], P.objPos[ 2 ] );

			const G = GLASSES[ S.glass ];
			U.g0.value.set( G.B[ 0 ], G.B[ 1 ], G.B[ 2 ], S.dispersion );
			U.g1.value.set( G.C[ 0 ], G.C[ 1 ], G.C[ 2 ], sellmeier( G, 587.6 ) );

			// beam
			const D = new THREE.Vector3( ...P.beam.dir ).normalize();
			if ( state.beamYaw ) D.applyAxisAngle( new THREE.Vector3( 0, 1, 0 ), state.beamYaw );
			const T = new THREE.Vector3( ...P.beam.target );
			// back-trace to the wall the beam enters through
			let s = 1e9;
			const lo = [ ROOM.x0, 0, ROOM.z0 ], hi = [ ROOM.x1, ROOM.y1, ROOM.z1 ];
			const tA = T.toArray(), dA2 = D.toArray();
			for ( let a = 0; a < 3; a ++ ) {

				if ( Math.abs( dA2[ a ] ) < 1e-6 ) continue;
				const w = dA2[ a ] > 0 ? lo[ a ] : hi[ a ];
				const t = ( tA[ a ] - w ) / dA2[ a ];
				if ( t > 0 ) s = Math.min( s, t );

			}

			const O = T.clone().addScaledVector( D, - ( s - 0.04 ) );
			const Ux = new THREE.Vector3().crossVectors( D, new THREE.Vector3( 0, 1, 0 ) );
			if ( Ux.lengthSq() < 1e-6 ) Ux.set( 1, 0, 0 );
			Ux.normalize();
			U.b0.value.set( O.x, O.y, O.z, P.beam.hw );
			U.b1.value.set( D.x, D.y, D.z, P.beam.hh );
			const L = LIGHTS[ S.light ];
			U.b2.value.set( Ux.x, Ux.y, Ux.z, L.mode );
			const area = 4 * P.beam.hw * P.beam.hh;
			const power = 6.0 * area / 0.0033 * ( S.preset === 'prism' ? 1 : S.preset === 'gem' ? 0.9 : 0.07 );
			U.b3.value.set( blackbodyNorm( L.T ), power / cfg.photons, 0.06 * S.haze, 0.35 );
			const xyz = L.mode === 0 ? blackbodyXYZ( L.T ) : linesXYZ( L.mode === 1 ? LINES.hydrogen : LINES.mercury );
			const glow = 60;
			U.m0.value.set( xyz[ 0 ] * glow, xyz[ 1 ] * glow, xyz[ 2 ] * glow, L.T );

		}

		function frameUniforms() {

			const Tamb = 3300;
			U.a0.value.set( 6.0 * S.ambient, S.bounces, S.spp, 0 );
			U.a1.value.set( Tamb, blackbodyNorm( Tamb ), 0.16, 0.09 );
			// keep haze in sync (it is in b3.z)
			U.b3.value.z = 0.06 * S.haze;

		}

		function cameraUniforms() {

			camera.updateMatrixWorld();
			const e = camera.matrixWorld.elements;
			const tanY = Math.tan( THREE.MathUtils.degToRad( camera.fov ) * 0.5 );
			const tanX = tanY * res.W / res.H;
			const fw = new THREE.Vector3( - e[ 8 ], - e[ 9 ], - e[ 10 ] ).normalize();
			U.c0.value.set( e[ 12 ], e[ 13 ], e[ 14 ], tanY );
			U.c1.value.set( e[ 0 ], e[ 1 ], e[ 2 ], tanX );
			U.c2.value.set( e[ 4 ], e[ 5 ], e[ 6 ], res.W );
			U.c3.value.set( fw.x, fw.y, fw.z, res.H );
			uPixAngle.value = 2 * tanY / res.H;
			const key = [ e[ 12 ], e[ 13 ], e[ 14 ], fw.x, fw.y, fw.z, tanY, res.W, res.H ].map( ( v ) => v.toFixed( 5 ) ).join( ',' );
			const changed = key !== state.camKey;
			state.camKey = key;
			return changed;

		}

		function applyRes() {

			uWi.value = res.W; uNPix.value = res.W * res.H;
			uUV.value.set( res.W / TW, res.H / TH );

		}

		applyRes();

		// keep the camera inside the gallery
		function clampCamera() {

			const p = camera.position;
			p.x = Math.min( ROOM.x1 - 0.15, Math.max( ROOM.x0 + 0.15, p.x ) );
			p.y = Math.min( ROOM.y1 - 0.15, Math.max( 0.12, p.y ) );
			p.z = Math.min( ROOM.z1 - 0.15, Math.max( ROOM.z0 + 0.15, p.z ) );

		}

		// warm-up compile (async) so the first frame doesn't stall the shell
		sceneUniforms(); frameUniforms(); cameraUniforms();
		const kernels = [ photonK, lmK, eyeK, prepK, ...atrous, outDenoised, outRaw ];
		for ( const k of kernels ) await renderer.computeAsync( k );
		state.dirtyScene = true; state.camKey = '';

		let bloomNode = null;
		let fpsT = 0, fpsN = 0, fps = 0;

		return {
			scene,
			post() {

				const col = texture( outTex, screenUV.mul( uUV ) );
				bloomNode = bloom( col, 0.55, 0.35, 0.85 );
				return vec4( col.rgb.add( bloomNode.rgb ), 1 );

			},
			update( dt ) {

				fpsT += dt; fpsN ++;
				if ( fpsT > 0.5 ) {

					fps = fpsN / fpsT; fpsT = 0; fpsN = 0;

				}

				clampCamera();
				let resetCam = cameraUniforms() || state.dirtyCam;
				state.dirtyCam = false;
				if ( state.dirtyScene ) {

					sceneUniforms();
					state.lmFrame = 0;
					state.dirtyScene = false;
					resetCam = true;

				}

				frameUniforms();
				if ( resetCam ) state.frame = 0;

				state.seed = ( state.seed + 0x9E3779B9 ) >>> 0;
				uSeed.value = state.seed;
				uFrame.value = state.frame;
				uBlend.value = 1 / ( state.frame + 1 );
				uNf.value = state.frame + 1;
				uLmAlpha.value = 1 / ( state.lmFrame + 1 );
				const n = ( state.frame + 1 ) * S.spp;
				uFade.value = S.denoise ? Math.min( 1, Math.max( 0, ( n - 24 ) / 900 ) ) : 1;
				uExposure.value = S.exposure;

				renderer.compute( photonK );
				renderer.compute( lmK );
				renderer.compute( eyeK );
				renderer.compute( prepK );
				if ( S.denoise ) {

					for ( const k of atrous ) renderer.compute( k );
					renderer.compute( outDenoised );

				} else renderer.compute( outRaw );

				state.frame ++; state.lmFrame ++;
				state.photonsTotal += cfg.photons;

				const now = performance.now();
				if ( now - state.lastStats > 500 ) {

					state.lastStats = now;
					const rays = res.W * res.H * S.spp * ( S.bounces * 0.6 * 2 ) + cfg.photons * 5;
					ctx.setStats( [
						[ 'spp', ( state.frame * S.spp ).toLocaleString() ],
						[ 'photons', ( state.lmFrame * cfg.photons / 1e6 ).toFixed( 1 ) + ' M' ],
						[ 'bounces', S.bounces ],
						[ 'rays/s', ( rays * fps / 1e6 ).toFixed( 0 ) + ' M' ],
						[ 'trace res', `${res.W}×${res.H}` ],
					] );

				}

			},
			resize( w, h ) {

				const oW = res.W, oH = res.H;
				pickRes( w, h );
				if ( oW !== res.W || oH !== res.H ) {

					applyRes();
					state.camKey = '';

				}

			},
			onKey( e ) {

				if ( e.key === 'l' || e.key === 'L' ) {

					S.light = ( S.light + 1 ) % LIGHTS.length; state.dirtyScene = true;

				}

			},
			dispose() {

				dom.removeEventListener( 'pointerdown', onDown, { capture: true } );
				removeEventListener( 'pointermove', onMove );
				removeEventListener( 'pointerup', onUp );
				controls.enabled = true;
				for ( const k of kernels ) k.dispose();
				bloomNode?.dispose();
				lmTex.dispose(); outTex.dispose();
				for ( const b of buffers ) {

					try {

						renderer._attributes?.delete( b.value );

					} catch ( e ) { /* already gone */ }

				}

			},
		};

	},
};
