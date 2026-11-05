// Chapter 02 — Stellar Nursery
// GPU-generated emission nebula: compute-shader density field (domain-warped fBm + ridged noise sculpted into
// pillar SDFs) in a 3D storage texture, a compute-precomputed column-density (shadow) volume toward the ionising
// cluster and user-ignited stars, and a half-res jittered raymarch (HG phase, Beer–Lambert, multiple-scattering
// octaves, recombination-line emission) with temporal accumulation. Stars are extinguished per-star on the GPU.
import * as THREE from 'three/webgpu';
import { uniform, vec3, vec4, texture, screenUV, float, Fn, mix, dot, max, pow, clamp } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { NebulaVolume, BOX_MIN, BOX_SIZE, MAX_USER_STARS } from './nebula/volume.js';
import { buildMarchMaterial, buildResolveMaterial } from './nebula/march.js';
import { buildStars } from './nebula/stars.js';

const TIERS = {
	0: { res: [ 64, 40, 28 ], detailRes: 32, lightSteps: 24, userSteps: 12, steps: 48, slab: 0, scale: 0.5 },
	1: { res: [ 144, 96, 64 ], detailRes: 48, lightSteps: 48, userSteps: 24, steps: 96, slab: 4, scale: 0.5 },
	2: { res: [ 208, 136, 96 ], detailRes: 64, lightSteps: 64, userSteps: 32, steps: 144, slab: 8, scale: 0.5 },
};

const INFLUENCE = 4.0; // world-units reach of a user star's light volume
const BUBBLE_MAX = 2.3;

export default {
	id: 'nebula',
	title: 'Stellar Nursery',
	subtitle: 'Ultraviolet light from a newborn cluster ionises and erodes towering columns of molecular gas — a compute-generated density field, ray-marched with self-shadowing, scattering and recombination-line emission.',

	async create( ctx ) {

		const { renderer, camera, controls, pointer } = ctx;
		const tier = Math.max( 0, Math.min( 2, ctx.tier | 0 ) );
		const cfg = { ...TIERS[ tier ] };
		const [ nx, ny, nz ] = cfg.res;
		const scene = new THREE.Scene();
		const disposables = [];

		// ───────── uniforms ─────────
		const U = {
			density: uniform( 1.0 ), extinction: uniform( 2.0 ), emission: uniform( 1.0 ), light: uniform( 1.0 ),
			palette: uniform( 1.0 ), steps: uniform( cfg.steps, 'int' ), frame: uniform( 0 ), time: uniform( 0 ),
			drift: uniform( 0 ), detail: uniform( 1.0 ), detailScale: uniform( 0.36 ), detailOff: uniform( new THREE.Vector3() ),
			jets: uniform( 1.0 ), userOn: uniform( 0 ), userCol: uniform( new THREE.Color( 0.75, 0.85, 1.0 ) ),
			invProj: uniform( new THREE.Matrix4() ), camWorld: uniform( new THREE.Matrix4() ), camPos: uniform( new THREE.Vector3() ),
			prevViewProj: uniform( new THREE.Matrix4() ), texel: uniform( new THREE.Vector2( 1, 1 ) ), blend: uniform( 1.0 ),
			pickO: uniform( new THREE.Vector3() ), pickD: uniform( new THREE.Vector3( 0, 0, - 1 ) ),
			starPos: [], starPar: [],
			starGain: uniform( 1.0 ), pxScale: uniform( 0.001 ), pxRef: uniform( 1.0 ), bgStars: uniform( 1.0 ),
		};
		for ( let k = 0; k < MAX_USER_STARS; k ++ ) {

			U.starPos.push( uniform( new THREE.Vector4( 0, - 1000, 0, 0.01 ) ) );
			U.starPar.push( uniform( new THREE.Vector4( 0, 0, INFLUENCE, 0 ) ) );

		}

		// ───────── volume + compute kernels ─────────
		const vol = new NebulaVolume( renderer, cfg, U );
		const offD = uniform( new THREE.Vector3() ), offL = uniform( new THREE.Vector3() );
		const full = [ nx / 4, ny / 4, nz / 4 ];
		const kDens = vol.makeDensityKernel( offD, full );
		const kLight = vol.makeLightKernel( offL, full );
		const kDetail = vol.makeDetailKernel();
		vol.kernels = { kDens, kLight, kDetail };

		await renderer.computeAsync( kDetail );
		await renderer.computeAsync( kDens );
		await renderer.computeAsync( kLight );

		// local (bubble) kernels, compiled lazily on first ignition
		const voxel = new THREE.Vector3( BOX_SIZE.x / nx, BOX_SIZE.y / ny, BOX_SIZE.z / nz );
		const half = new THREE.Vector3( Math.ceil( INFLUENCE / voxel.x / 4 ) * 4, Math.ceil( INFLUENCE / voxel.y / 4 ) * 4, Math.ceil( INFLUENCE / voxel.z / 4 ) * 4 );
		const localDispatch = [ half.x / 2, half.y / 2, half.z / 2 ];
		const RD = BUBBLE_MAX * 1.3;
		const halfD = new THREE.Vector3( Math.ceil( RD / voxel.x / 4 ) * 4, Math.ceil( RD / voxel.y / 4 ) * 4, Math.ceil( RD / voxel.z / 4 ) * 4 );
		const offLD = uniform( new THREE.Vector3() ), offLL = uniform( new THREE.Vector3() );
		let kLocalD = null, kLocalL = null, kPick = null;

		// ───────── stars ─────────
		const stars = buildStars( ctx, vol, U );
		scene.add( stars.sprite );

		// ───────── raymarch + temporal resolve (half res) ─────────
		const rtOpts = { type: THREE.HalfFloatType, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
		const rtMarch = new THREE.RenderTarget( 2, 2, rtOpts );
		const rtResolve = new THREE.RenderTarget( 2, 2, rtOpts );
		const rtHist = new THREE.RenderTarget( 2, 2, rtOpts );
		const marchMat = buildMarchMaterial( vol, U );
		const resolveMat = buildResolveMaterial( texture( rtMarch.texture ), texture( rtHist.texture ), U );
		const qMarch = new THREE.QuadMesh( marchMat );
		const qResolve = new THREE.QuadMesh( resolveMat );
		disposables.push( marchMat, resolveMat, rtMarch, rtResolve, rtHist );
		let histValid = false;

		// ───────── camera ─────────
		camera.fov = 40;
		camera.near = 0.05;
		camera.far = 3000;
		camera.updateProjectionMatrix();
		camera.position.set( 1.0, - 2.6, 15.0 );
		controls.target.set( 0.2, - 0.4, - 1.6 );
		controls.minDistance = 4;
		controls.maxDistance = 40;
		controls.enablePan = true;
		controls.update();

		let interacting = false, idle = 10, driftT = 0;
		const onStart = () => {

			interacting = true; idle = 0;

		};
		const onEnd = () => {

			interacting = false;

		};
		controls.addEventListener( 'start', onStart );
		controls.addEventListener( 'end', onEnd );
		const sph = new THREE.Spherical(), tmpV = new THREE.Vector3();
		const driftPose = ( t ) => ( {
			az: 0.34 * Math.sin( t * 2 * Math.PI / 80 ),
			el: 0.06 * Math.sin( t * 2 * Math.PI / 57 + 1.3 ),
			rad: 1 - 0.14 * Math.sin( t * 2 * Math.PI / 97 ),
		} );

		// ───────── GUI ─────────
		const state = { palette: 'Hubble SHO', drift: 1.0, paletteTarget: 1 };
		const g = ctx.gui;
		g.add( U.density, 'value', 0.2, 3, 0.01 ).name( 'density' );
		g.add( U.extinction, 'value', 0.3, 6, 0.01 ).name( 'extinction' );
		g.add( U.emission, 'value', 0, 4, 0.01 ).name( 'emission' );
		g.add( U.light, 'value', 0, 4, 0.01 ).name( 'cluster light' );
		g.add( state, 'palette', [ 'Hubble SHO', 'Natural' ] ).name( 'palette' ).onChange( ( v ) => state.paletteTarget = v === 'Natural' ? 0 : 1 );
		g.add( U.steps, 'value', 24, 256, 1 ).name( 'march steps' );
		g.add( state, 'drift', 0, 4, 0.01 ).name( 'gas drift' );
		const actions = { clear: () => clearStars() };
		g.add( actions, 'clear' ).name( 'clear ignited stars' );

		if ( ctx.params.get( 'pal' ) === 'natural' ) {

			state.palette = 'Natural'; state.paletteTarget = 0; U.palette.value = 0;

		}

		// ───────── user stars (press & hold to ignite) ─────────
		const slots = [];
		for ( let k = 0; k < MAX_USER_STARS; k ++ ) slots.push( { active: false, pending: false, pos: new THREE.Vector3(), age: 0, energy: 0, held: false, born: 0 } );
		const localQueue = [];
		let press = null, nextSlot = 0, picking = false;

		const ensureLocalKernels = () => {

			if ( ! kLocalD ) {

				kLocalD = vol.makeDensityKernel( offLD, [ halfD.x / 2, halfD.y / 2, halfD.z / 2 ] );
				kLocalL = vol.makeLightKernel( offLL, localDispatch );
				kPick = vol.makePickKernel();
				vol.kernels.kLocalD = kLocalD; vol.kernels.kLocalL = kLocalL; vol.kernels.kPick = kPick;

			}

		};

		const ignite = ( ray ) => {

			ensureLocalKernels();
			picking = true;
			const k = slots.findIndex( ( s ) => ! s.active && ! s.pending ) >= 0 ? slots.findIndex( ( s ) => ! s.active && ! s.pending ) : nextSlot;
			nextSlot = ( k + 1 ) % MAX_USER_STARS;
			const s = slots[ k ];
			if ( s.active ) localQueue.push( s.pos.clone() ); // old bubble heals
			s.active = false; s.pending = true; s.held = true;
			U.starPar[ k ].value.set( 0, 0, INFLUENCE, 0 );
			U.pickO.value.copy( ray.origin );
			U.pickD.value.copy( ray.direction ).normalize();
			renderer.compute( kPick );
			renderer.getArrayBufferAsync( vol.pickBuf.value ).then( ( buf ) => {

				picking = false;
				if ( disposed ) return;
				const f = new Float32Array( buf );
				s.pending = false;
				if ( f[ 3 ] <= 0 ) return;
				s.active = true; s.age = 0; s.energy = 0.05; s.born = clockT;
				s.pos.set( f[ 0 ], f[ 1 ], f[ 2 ] );
				s.held = press !== null && press.slot === k;
				stars.setUserStar( k, s.pos );

			} ).catch( () => {

				picking = false; s.pending = false;

			} );
			return k;

		};

		const clearStars = () => {

			for ( let k = 0; k < MAX_USER_STARS; k ++ ) {

				slots[ k ].active = false; slots[ k ].pending = false;
				U.starPos[ k ].value.set( 0, - 1000, 0, 0.01 );
				U.starPar[ k ].value.set( 0, 0, INFLUENCE, 0 );
				stars.setUserStar( k, new THREE.Vector3( 0, - 5000, 0 ) );

			}

			localQueue.length = 0;
			offD.value.set( 0, 0, 0 ); offL.value.set( 0, 0, 0 );
			renderer.compute( kDens, full );
			renderer.compute( kLight, full );

		};

		// Spitzer-law D-type ionisation front: R(t) = Rs (1 + 7 t / 4 ts)^(4/7)
		const bubbleR = ( age, energy ) => Math.min( BUBBLE_MAX, 0.22 * Math.pow( 1 + 7 * age / ( 4 * 0.35 ), 4 / 7 ) ) * ( 0.55 + 0.45 * energy );

		// debug/capture: ?ignite=x,y (ndc) ignites a star at frame 4
		const igniteParam = ctx.params.get( 'ignite' );
		let frameNo = 0, clockT = 0, statT = 0, disposed = false;
		let sweepD = 0, sweepL = Math.floor( nz / 2 / 4 ) * 4;
		const viewProj = new THREE.Matrix4();
		const raycaster = new THREE.Raycaster();

		const resize = ( w, h, dpr ) => {

			const W = Math.max( 2, Math.round( w * dpr * cfg.scale ) ), H = Math.max( 2, Math.round( h * dpr * cfg.scale ) );
			rtMarch.setSize( W, H ); rtResolve.setSize( W, H ); rtHist.setSize( W, H );
			U.texel.value.set( 1 / W, 1 / H );
			const Hp = h * dpr;
			U.pxScale.value = 2 * Math.tan( THREE.MathUtils.degToRad( camera.fov ) / 2 ) / Hp;
			U.pxRef.value = Math.max( 0.55, Hp / 1080 );
			histValid = false;

		};

		resize( ctx.size.x, ctx.size.y, renderer.getPixelRatio() );

		let bloomNode = null;

		return {
			scene,
			camera,

			post( _ctx, scenePass ) {

				const starsTex = scenePass.getTextureNode( 'output' );
				const neb = texture( rtResolve.texture, screenUV );
				const grade = Fn( () => {

					const c = neb.rgb.toVar();
					// gentle saturation lift before AgX (which desaturates highlights)
					const l = dot( c, vec3( 0.2126, 0.7152, 0.0722 ) );
					c.assign( max( mix( vec3( l ), c, 1.18 ), vec3( 0 ) ) );
					return c;

				} )();
				const col = vec4( starsTex.rgb.add( grade ), 1.0 );
				bloomNode = bloom( col, 0.4, 0.55, 0.0 );
				return col.add( bloomNode );

			},

			update( dt, t ) {

				clockT = t;
				frameNo ++;

				// ── camera drift (incremental, so user orbiting is never fought)
				if ( ! interacting && ! pointer.down ) idle += dt;
				const w = THREE.MathUtils.smoothstep( idle, 2.5, 6.0 );
				if ( w > 0 ) {

					const a = driftPose( driftT );
					driftT += dt * w;
					const b = driftPose( driftT );
					tmpV.subVectors( camera.position, controls.target );
					sph.setFromVector3( tmpV );
					sph.theta += b.az - a.az;
					sph.phi = THREE.MathUtils.clamp( sph.phi - ( b.el - a.el ), 0.2, Math.PI - 0.2 );
					sph.radius *= b.rad / a.rad;
					tmpV.setFromSpherical( sph );
					camera.position.copy( controls.target ).add( tmpV );
					camera.lookAt( controls.target );

				}

				camera.updateMatrixWorld();
				U.camPos.value.copy( camera.position );
				U.camWorld.value.copy( camera.matrixWorld );
				U.invProj.value.copy( camera.projectionMatrixInverse );
				U.time.value = t;
				U.frame.value = frameNo % 4096;

				// palette cross-fade
				U.palette.value += ( state.paletteTarget - U.palette.value ) * Math.min( 1, dt * 3 );

				// slow advection of the sub-voxel detail + large-scale warp evolution
				U.detailOff.value.set( t * 0.010, t * 0.006, - t * 0.008 ).multiplyScalar( state.drift );
				U.drift.value += dt * state.drift;

				// ── pointer: press & hold (without dragging) ignites a star where the ray meets the cloud
				if ( pointer.down && pointer.button === 0 ) {

					if ( ! press ) press = { t: 0, x: pointer.ndc.x, y: pointer.ndc.y, moved: false, slot: - 1 };
					press.t += dt;
					if ( Math.hypot( pointer.ndc.x - press.x, pointer.ndc.y - press.y ) > 0.03 ) press.moved = true;
					if ( ! press.moved && press.t > 0.22 && press.slot < 0 && ! picking ) press.slot = ignite( pointer.ray );

				} else if ( press ) {

					if ( press.slot >= 0 ) slots[ press.slot ].held = false;
					press = null;

				}

				if ( igniteParam && frameNo === 4 ) {

					const [ px, py ] = igniteParam.split( ',' ).map( Number );
					raycaster.setFromCamera( new THREE.Vector2( px, py ), camera );
					const k = ignite( raycaster.ray );
					slots[ k ].held = false;
					slots[ k ].autoEnergy = true;

				}

				// ── user star evolution
				let localThisFrame = null;
				for ( let k = 0; k < MAX_USER_STARS; k ++ ) {

					const s = slots[ k ];
					if ( ! s.active ) continue;
					if ( s.held ) s.energy = Math.min( 1, s.energy + dt * 0.7 );
					else s.energy = Math.min( 1, Math.max( s.energy, s.autoEnergy ? 1 : 0.35 ) );
					const ageRate = ctx.shot ? 6 : 1; // captures see the bubble grown
					s.age += dt * ageRate * ( 0.3 + 0.7 * s.energy );
					const R = bubbleR( s.age, s.energy );
					const flicker = 1 + 0.06 * Math.sin( t * 13.0 + k ) * Math.exp( - s.age * 0.3 );
					U.starPos[ k ].value.set( s.pos.x, s.pos.y, s.pos.z, R );
					U.starPar[ k ].value.set( ( 0.6 + 2.6 * s.energy ) * flicker, Math.min( 1, s.energy * 3 ), INFLUENCE, 0 );
					if ( ! localThisFrame && ( s.age < 14 || s.held ) ) localThisFrame = s.pos;

				}

				U.userOn.value = slots.some( ( s ) => s.active ) ? 1 : 0;

				// ── volume maintenance: time-sliced global sweep + local bubble updates
				if ( cfg.slab > 0 && state.drift > 0 ) {

					offD.value.set( 0, 0, sweepD );
					renderer.compute( kDens, [ nx / 4, ny / 4, cfg.slab / 4 ] );
					offL.value.set( 0, 0, sweepL );
					renderer.compute( kLight, [ nx / 4, ny / 4, cfg.slab / 4 ] );
					sweepD = ( sweepD + cfg.slab ) % nz;
					sweepL = ( sweepL + cfg.slab ) % nz;

				}

				const lp = localThisFrame || localQueue.shift();
				if ( lp && kLocalD && ( tier > 0 || frameNo % 2 === 0 ) ) {

					const sv = tmpV.copy( lp ).sub( BOX_MIN ).divide( voxel ).floor();
					offLD.value.copy( sv ).sub( halfD );
					offLL.value.copy( sv ).sub( half );
					renderer.compute( kLocalD, [ halfD.x / 2, halfD.y / 2, halfD.z / 2 ] );
					renderer.compute( kLocalL, localDispatch );

				}

				// ── per-star extinction (camera → star column density)
				renderer.compute( stars.tauKernel );

				// ── raymarch + temporal resolve
				const prev = renderer.getRenderTarget();
				U.blend.value = histValid ? ( ctx.shot ? 0.2 : 0.12 ) : 1.0;
				renderer.setRenderTarget( rtMarch );
				qMarch.render( renderer );
				renderer.setRenderTarget( rtResolve );
				qResolve.render( renderer );
				renderer.setRenderTarget( prev );
				renderer.copyTextureToTexture( rtResolve.texture, rtHist.texture );
				histValid = true;

				viewProj.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse );
				U.prevViewProj.value.copy( viewProj );

				statT -= dt;
				if ( statT <= 0 ) {

					statT = 0.5;
					ctx.setStats( [
						[ 'volume', `${nx}×${ny}×${nz}` ],
						[ 'march', `${rtMarch.width}×${rtMarch.height} · ${U.steps.value} st` ],
						[ 'light', `${cfg.lightSteps} st${cfg.slab ? ' · sliced' : ''}` ],
						[ 'stars', stars.count.toLocaleString() ],
						[ 'ignited', `${slots.filter( ( s ) => s.active ).length}/${MAX_USER_STARS}` ],
					] );

				}

			},

			resize,

			onKey( e ) {

				if ( e.key === 'c' || e.key === 'C' ) clearStars();
				if ( e.key === 'p' || e.key === 'P' ) {

					state.paletteTarget = 1 - state.paletteTarget;
					state.palette = state.paletteTarget ? 'Hubble SHO' : 'Natural';
					g.controllersRecursive().forEach( ( c ) => c.updateDisplay() );

				}

			},

			dispose() {

				disposed = true;
				controls.removeEventListener( 'start', onStart );
				controls.removeEventListener( 'end', onEnd );
				bloomNode?.dispose?.();
				qMarch.dispose?.();
				qResolve.dispose?.();
				for ( const d of disposables ) d.dispose();
				stars.dispose();
				vol.dispose();

			},
		};

	},
};
