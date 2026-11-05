// Chapter 02 — Cosmic Web
// ΛCDM structure formation in a periodic box: Gaussian random field (BBKS spectrum) → Zel'dovich ICs → GPU Particle-Mesh
// gravity (fixed-point CIC atomics, 3D Stockham FFT Poisson solve in compute shaders, KDK leapfrog in comoving coordinates).
// Rendering follows the classic simulation-image recipe: particles are splatted with an adaptive (SPH-like) kernel
// h ∝ ρ^(-1/3) into a float accumulator holding projected mass and mass-weighted log-density; a post pass maps
// log(projected density) to brightness and the density index to hue (indigo voids → violet/blue filaments → gold halos).
import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, vec4, uv, varying, positionWorld, cameraPosition, texture3D, pass, Loop,
	exp, log, pow, max, min, abs, mix, smoothstep, clamp, select, screenCoordinate, fract, sin, dot, length,
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { Cosmology } from './cosmicweb/cosmology.js';
import { ParticleMesh, MAX_SEEDS } from './cosmicweb/pm.js';

const BOX = 4; // world size of the periodic box
const L_BOX = 160; // Mpc/h
const Z_START = 50;
const A_START = 1 / ( 1 + Z_START );
const STORY = 48; // seconds from z=50 to z=0 at speed 1
const HOLD = 9; // seconds at z=0 before the next universe

// a(u) — front-loaded so the web is visible within a few seconds, then lingers where structure is rich
const aOfU = ( u ) => A_START + ( 1 - A_START ) * Math.pow( Math.min( 1, Math.max( 0, u ) ), 0.75 );
const uOfA = ( a ) => Math.pow( Math.max( 0, ( a - A_START ) / ( 1 - A_START ) ), 1 / 0.75 );

// mass-weighted colour index from local density (ρ / ρ̄)
const colourIndex = ( r ) => clamp( log( max( r, 1e-3 ) ).mul( 0.4343 ).add( 0.7 ).div( 3.0 ), 0, 1 );

export default {
	id: 'cosmicweb',
	title: 'Cosmic Web',
	subtitle: 'Dark matter in a periodic ΛCDM box, pulled by gravity on a GPU particle-mesh from a near-uniform Gaussian field at z = 50 into filaments, sheets, voids and halos.',

	async create( ctx ) {

		const { renderer, camera, controls, params, tier } = ctx;
		const LOG = params.has( 'cwlog' ) ? ( ...m ) => console.warn( '[cw]', ( performance.now() / 1000 ).toFixed( 2 ), ...m ) : () => {};
		LOG( 'create' );

		// ───────────────────────── sizes per tier ─────────────────────────
		const pow2 = ( v, d ) => {

			const n = parseInt( v );
			return [ 16, 32, 64, 128 ].includes( n ) ? n : d;

		};

		const NL = pow2( params.get( 'np' ), tier === 0 ? 32 : tier === 1 ? 64 : 128 );
		const N = pow2( params.get( 'mesh' ), tier === 0 ? 32 : tier === 1 ? 64 : 128 );
		const NP = NL * NL * NL;
		const maxRunSteps = tier === 0 ? 2 : 4;
		const MAX_DLNA = 0.035; // max Δ ln a per PM step during normal play
		const SEEK_DLNA = 0.045; // during rewind/replay

		const cosmo = new Cosmology( { Om: 0.3, OL: 0.7, h: 0.7 } );
		const pm = new ParticleMesh( renderer, { N, NL, L: L_BOX } );
		pm.u.gamma.value = cosmo.gamma;

		const S = {
			sigma8: 0.85, ns: 0.96, speed: 1, size: 1, view: 'particles', tile: 'auto', loop: true, seedAmp: 5,
		};

		// ───────────────────────── simulation state ─────────────────────────
		const st = {
			seed: ( parseInt( params.get( 'seed' ) ) || ( ( Math.random() * 1e9 ) >>> 0 ) ) >>> 0,
			a: A_START, aP: A_START, tau: 0, phase: 'run', seekTarget: 1, seekPerFrame: 2, hold: 0, steps: 0,
			seeds: [], fade: 0, fadeTarget: 1, paused: false, pendingRestart: null,
		};

		const setGreen = ( a ) => {

			pm.u.green.value = - 1.5 * cosmo.Om / a / pm.N3;

		};

		const DBG = ( k ) => params.has( k );
		function regenerate() {

			if ( DBG( 'cwnoic' ) ) return;

			pm.u.seed.value = st.seed;
			pm.u.amp.value = cosmo.normalise( S.sigma8, S.ns );
			pm.u.ns.value = S.ns;
			pm.u.seedR.value = 6 / ( L_BOX / N ); // 6 Mpc/h
			pm.setSeeds( st.seeds, S.seedAmp );
			st.a = A_START; st.aP = A_START; st.steps = 0;
			pm.u.D0.value = cosmo.D( A_START );
			pm.u.P0.value = cosmo.zaMomentumFactor( A_START );
			renderer.compute( pm.icList );
			// refresh the density field (colours, volume) without moving anything
			setGreen( st.a ); pm.u.kick.value = 0; pm.u.drift.value = 0;
			renderer.compute( pm.stepList );

		}

		function step( aNext ) {

			if ( DBG( 'cwnostep' ) ) {

				st.a = aNext; return;

			}

			const a = st.a;
			const aMid = Math.sqrt( a * aNext );
			pm.u.kick.value = cosmo.kick( st.aP, aMid );
			pm.u.drift.value = cosmo.drift( a, aNext );
			setGreen( a );
			renderer.compute( pm.stepList );
			st.a = aNext; st.aP = aMid; st.steps ++;

		}

		// advance towards aTarget with at most nMax steps of at most dMax in ln a
		function advance( aTarget, nMax, dMax ) {

			let n = 0;
			while ( st.a < aTarget - 1e-7 && n < nMax ) {

				const next = Math.min( aTarget, st.a * Math.exp( dMax ) );
				step( next );
				n ++;

			}

			return n;

		}

		function seekTo( aTarget, seconds = 1.2 ) {

			aTarget = Math.min( 1, Math.max( A_START, aTarget ) );
			regenerate();
			const total = Math.ceil( Math.log( aTarget / A_START ) / SEEK_DLNA );
			st.seekTarget = aTarget;
			st.seekPerFrame = Math.max( 1, Math.min( tier === 0 ? 3 : 10, Math.ceil( total / ( seconds * 60 ) ) ) );
			st.phase = aTarget > A_START + 1e-6 ? 'seek' : 'run';
			st.tau = uOfA( aTarget ) * STORY;

		}

		function newUniverse( keepSeed = false ) {

			if ( ! keepSeed ) st.seed = ( ( Math.random() * 4294967295 ) >>> 0 ) || 1;
			st.seeds.length = 0;
			markers.forEach( ( m ) => m.life = 0 );
			regenerate();
			st.tau = 0; st.phase = 'run'; st.hold = 0;
			st.fade = 0;

		}

		// ───────────────────────── accumulation scene (data, not colour) ─────────────────────────
		// rgb = ( Σ m·K , Σ m·K·c , Σ m·K·c² ) — projected mass and its mass-weighted colour index
		const scene = new THREE.Scene();
		const accBlend = ( m ) => {

			m.transparent = true;
			m.depthWrite = false; m.depthTest = false;
			m.blending = THREE.CustomBlending;
			m.blendEquation = THREE.AddEquation; m.blendEquationAlpha = THREE.AddEquation;
			m.blendSrc = THREE.OneFactor; m.blendDst = THREE.OneFactor;
			m.blendSrcAlpha = THREE.OneFactor; m.blendDstAlpha = THREE.OneFactor;

		};

		const U = {
			size: uniform( 1 ),
			partGain: uniform( 1 ),
			volGain: uniform( 0.12 ),
			tileGain: uniform( 0 ),
			extent: uniform( 1 ),
			exposure: uniform( 1.6 ),
			fade: uniform( 0 ),
			boxAlpha: uniform( 0.3 ),
		};

		// particles: adaptive kernel h = size · spacing · ρ^(-1/3), flux per particle constant (mass conserving splat)
		const spacing = BOX / NL;
		const flux = BOX * BOX / NP; // so that a uniform universe projects to Σ = 1
		const pAttr = pm.pos.toAttribute();
		const pmat = new THREE.SpriteNodeMaterial();
		accBlend( pmat );
		const rhoP = pAttr.w.max( 0.02 );
		const hNode = clamp( pow( rhoP, - 1 / 3 ), 0.35, 2.2 ).mul( spacing * 0.5 ).mul( U.size ); // Gaussian σ (world)
		pmat.positionNode = pAttr.xyz.mul( BOX / N ).sub( BOX / 2 );
		pmat.scaleNode = hNode.mul( 6 ); // quad spans ±3σ
		const vSig = varying( hNode, 'vSig' );
		const vC = varying( colourIndex( rhoP ), 'vC' );
		pmat.fragmentNode = Fn( () => {

			const r2 = uv().sub( 0.5 ).mul( 2 ).dot( uv().sub( 0.5 ).mul( 2 ) );
			const k = exp( r2.mul( - 4.5 ) ).mul( r2.oneMinus().max( 0 ) ); // σ = 1/3 of the quad half-size
			const w = k.mul( U.partGain.mul( flux / ( 2 * Math.PI ) ) ).div( vSig.mul( vSig ) ).mul( 1.1 );
			return vec4( w, w.mul( vC ), w.mul( vC ).mul( vC ), 0 );

		} )();
		const sprites = new THREE.Sprite( pmat );
		sprites.count = NP;
		sprites.frustumCulled = false;
		sprites.renderOrder = 1;
		scene.add( sprites );

		// volume: emission-only ray march through the CIC density 3D texture (periodic → free 3×3×3 tiling)
		const K_STEPS = tier === 0 ? 40 : 72;
		const vmat = new THREE.MeshBasicNodeMaterial( { side: THREE.BackSide } );
		accBlend( vmat );
		vmat.fragmentNode = Fn( () => {

			const ro = cameraPosition;
			const rd = positionWorld.sub( ro ).normalize().toVar();
			const hb = U.extent.mul( BOX / 2 );
			const inv = vec3( 1 ).div( rd );
			const tA = vec3( hb ).negate().sub( ro ).mul( inv ), tB = vec3( hb ).sub( ro ).mul( inv );
			const tmn = min( tA, tB ), tmx = max( tA, tB );
			const t0 = max( max( tmn.x, tmn.y ), max( tmn.z, 0 ) ).toVar();
			const t1 = min( min( tmx.x, tmx.y ), tmx.z ).toVar();
			const dt = max( t1.sub( t0 ), 0 ).div( K_STEPS ).toVar();
			const jit = fract( sin( dot( screenCoordinate.xy, vec2( 12.9898, 78.233 ) ) ).mul( 43758.5453 ) );
			const acc = vec3( 0 ).toVar();
			Loop( K_STEPS, ( { i } ) => {

				const p = ro.add( rd.mul( t0.add( float( i ).add( jit ).mul( dt ) ) ) ).toVar();
				const r = texture3D( pm.volTex, p.div( BOX ).add( 0.5 ) ).level( 0 ).r.toVar();
				const inside = max( max( abs( p.x ), abs( p.y ) ), abs( p.z ) ).lessThan( BOX / 2 );
				const m = select( inside, U.volGain, U.tileGain );
				const c = colourIndex( r );
				const w = r.mul( m );
				acc.addAssign( vec3( w, w.mul( c ), w.mul( c ).mul( c ) ) );

			} );

			return vec4( acc.mul( dt.div( BOX ) ), 0 );

		} )();
		const vol = new THREE.Mesh( new THREE.BoxGeometry( BOX, BOX, BOX ), vmat );
		vol.frustumCulled = false;
		vol.renderOrder = 0;
		scene.add( vol );
		if ( DBG( 'cwnovol' ) ) vol.visible = false;

		// ───────────────────────── overlay scene (true colour) ─────────────────────────
		const overlay = new THREE.Scene();
		const boxMat = new THREE.LineBasicNodeMaterial( { transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending } );
		boxMat.colorNode = vec3( 0.35, 0.45, 1.0 );
		boxMat.opacityNode = U.boxAlpha;
		const boxGeo = new THREE.EdgesGeometry( new THREE.BoxGeometry( BOX, BOX, BOX ) );
		const boxLines = new THREE.LineSegments( boxGeo, boxMat );
		overlay.add( boxLines );

		const markers = [];
		for ( let i = 0; i < MAX_SEEDS; i ++ ) {

			const op = uniform( 0 ), grow = uniform( 0 );
			const m = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending } );
			m.colorNode = Fn( () => {

				const r = length( uv().sub( 0.5 ) ).mul( 2 );
				const ring = exp( r.sub( grow.mul( 0.7 ).add( 0.2 ) ).div( 0.06 ).pow2().negate() );
				const core = exp( r.mul( r ).mul( - 60 ) );
				return vec3( 1.4, 1.0, 0.55 ).mul( ring.mul( 0.8 ).add( core.mul( 2.0 ) ) ).mul( op );

			} )();
			const sp = new THREE.Sprite( m );
			sp.scale.setScalar( 0.9 );
			sp.visible = false;
			overlay.add( sp );
			markers.push( { sp, m, op, grow, life: 0 } );

		}

		// ───────────────────────── post: projected density → colour ─────────────────────────
		const floatOK = renderer.hasFeature?.( 'float32-blendable' );
		let accPass, overlayPass, bloomNode;

		const palette = Fn( ( [ c ] ) => {

			const c0 = vec3( 0.045, 0.03, 0.20 ); // deep indigo (voids)
			const c1 = vec3( 0.20, 0.07, 0.62 ); // violet
			const c2 = vec3( 0.16, 0.36, 1.25 ); // electric blue (filaments)
			const c3 = vec3( 0.75, 0.55, 1.35 ); // lavender transition
			const c4 = vec3( 1.9, 1.15, 0.38 ); // gold (halos)
			const c5 = vec3( 2.2, 1.9, 1.35 ); // white-gold cores
			const col = mix( c0, c1, smoothstep( 0.0, 0.25, c ) ).toVar();
			col.assign( mix( col, c2, smoothstep( 0.22, 0.45, c ) ) );
			col.assign( mix( col, c3, smoothstep( 0.5, 0.66, c ) ) );
			col.assign( mix( col, c4, smoothstep( 0.62, 0.8, c ) ) );
			col.assign( mix( col, c5, smoothstep( 0.82, 1.0, c ) ) );
			return col;

		} );

		// ───────────────────────── GUI ─────────────────────────
		const gui = ctx.gui;
		const actions = {
			newUniverse: () => newUniverse( false ),
			replay: () => {

				st.seeds.length = 0; newUniverse( true );

			},
		};
		gui.add( actions, 'newUniverse' ).name( 'new universe (seed)' );
		gui.add( S, 'sigma8', 0.4, 1.4, 0.01 ).name( 'amplitude σ8' ).onFinishChange( () => seekTo( st.a, 0.8 ) );
		gui.add( S, 'ns', 0.6, 1.4, 0.01 ).name( 'spectral index n' ).onFinishChange( () => seekTo( st.a, 0.8 ) );
		gui.add( S, 'speed', 0, 3, 0.05 ).name( 'expansion speed' );
		gui.add( S, 'view', [ 'particles', 'volume', 'both' ] ).name( 'show' );
		gui.add( S, 'size', 0.4, 3, 0.05 ).name( 'particle size' );
		gui.add( S, 'tile', [ 'auto', 'on', 'off' ] ).name( 'tile 3×3×3' );
		gui.add( S, 'seedAmp', 0, 12, 0.1 ).name( 'seed strength δ' );
		gui.add( S, 'loop' ).name( 'loop at z = 0' );

		// ───────────────────────── pointer: plant a primordial seed ─────────────────────────
		// A click adds a Gaussian peak to the *initial* density field at the clicked point (ray ∩ box, mid-chord),
		// then the universe is re-run from z = 50 to the current epoch: a protocluster grows there and the web reorganises.
		const el = renderer.domElement;
		const ray = new THREE.Raycaster();
		const ndc = new THREE.Vector2();
		let down = null;
		let userMoved = false;
		const onDown = ( e ) => {

			down = { x: e.clientX, y: e.clientY, t: performance.now() };
			userMoved = true;

		};

		const onUp = ( e ) => {

			if ( ! down ) return;
			const moved = Math.hypot( e.clientX - down.x, e.clientY - down.y );
			const dtc = performance.now() - down.t;
			down = null;
			if ( moved > 6 || dtc > 450 || e.button !== 0 ) return;
			const rect = el.getBoundingClientRect();
			ndc.set( ( ( e.clientX - rect.left ) / rect.width ) * 2 - 1, - ( ( e.clientY - rect.top ) / rect.height ) * 2 + 1 );
			ray.setFromCamera( ndc, camera );
			plantAtRay( ray.ray );

		};

		function plantAtRay( r ) {

			const o = r.origin, d = r.direction, hb = BOX / 2;
			let t0 = - Infinity, t1 = Infinity;
			for ( const ax of [ 'x', 'y', 'z' ] ) {

				const inv = 1 / d[ ax ];
				let ta = ( - hb - o[ ax ] ) * inv, tb = ( hb - o[ ax ] ) * inv;
				if ( ta > tb ) [ ta, tb ] = [ tb, ta ];
				t0 = Math.max( t0, ta ); t1 = Math.min( t1, tb );

			}

			if ( t1 < Math.max( t0, 0 ) ) return false; // missed the box
			const t = ( Math.max( t0, 0 ) + t1 ) / 2;
			const p = r.at( t, new THREE.Vector3() );
			plant( p );
			return true;

		}

		function plant( p ) {

			if ( st.seeds.length >= MAX_SEEDS ) st.seeds.shift();
			st.seeds.push( { x: ( p.x / BOX + 0.5 ) * N, y: ( p.y / BOX + 0.5 ) * N, z: ( p.z / BOX + 0.5 ) * N, w: 1, world: p.clone() } );
			const mk = markers.find( ( m ) => m.life <= 0 ) || markers[ 0 ];
			mk.sp.position.copy( p ); mk.life = 1; mk.sp.visible = true;
			const target = st.phase === 'seek' ? st.seekTarget : st.a;
			seekTo( Math.max( target, 0.12 ), 1.4 );
			st.hold = 0;

		}

		el.addEventListener( 'pointerdown', onDown );
		el.addEventListener( 'pointerup', onUp );

		// ───────────────────────── camera ─────────────────────────
		camera.position.set( 7.2, 3.6, 5.6 );
		controls.target.set( 0, 0, 0 );
		controls.minDistance = 1.2;
		controls.maxDistance = 40;
		controls.autoRotate = true;
		controls.autoRotateSpeed = 0.35;
		controls.update();

		// ───────────────────────── start ─────────────────────────
		LOG( 'pm built' );
		regenerate();
		LOG( 'regenerate submitted' );
		await renderer.computeAsync( pm.densityList );
		LOG( 'ic done' );
		const zt = params.has( 'zt' ) ? parseFloat( params.get( 'zt' ) ) : ( ctx.shot && ! params.has( 'warp' ) ? 1.2 : null );
		if ( zt !== null && isFinite( zt ) ) {

			// debug / capture: jump straight to redshift zt
			const aT = 1 / ( 1 + Math.max( 0, zt ) );
			advance( aT, 100000, SEEK_DLNA );
			st.tau = uOfA( st.a ) * STORY;

		}

		const warp = parseInt( params.get( 'warp' ) || '0' );
		for ( let i = 0; i < warp; i ++ ) {

			st.tau += 1 / 60;
			advance( aOfU( st.tau / STORY ), maxRunSteps, MAX_DLNA );

		}

		st.fade = ctx.shot ? 1 : 0;
		await renderer.computeAsync( pm.densityList );
		LOG( 'warp done', st.steps, st.a );

		let statT = 1;
		const sizeTxt = `${NL}³ = ${NP.toLocaleString()}`;

		return {
			scene,

			post( c, scenePass ) {

				const cam = camera;
				accPass = pass( scene, cam, { type: floatOK ? THREE.FloatType : THREE.HalfFloatType, depthBuffer: false } );
				overlayPass = pass( overlay, cam, { depthBuffer: false } );
				const acc = accPass.getTextureNode( 'output' );
				const mapped = Fn( () => {

					const s = acc.r.max( 1e-7 ).toVar();
					const cbar = clamp( acc.g.div( s ), 0, 1 ).toVar();
					const lum = log( s.mul( U.exposure ).add( 1 ) ).toVar();
					const b = pow( lum, 1.55 ).mul( 0.55 );
					// faint deep-space floor so voids read as indigo rather than black
					const col = palette( cbar ).mul( b ).add( vec3( 0.006, 0.004, 0.018 ) );
					return vec4( col.mul( U.fade ), 1 );

				} )();
				if ( DBG( 'cwsimplepost' ) ) return vec4( acc.rgb, 1 );
				bloomNode = bloom( mapped, 0.85, 0.55, 0.55 );
				return vec4( mapped.rgb.add( bloomNode.rgb ).add( overlayPass.getTextureNode( 'output' ).rgb ), 1 );

			},

			update( dt, t ) {

				// ── timeline ──
				if ( ! st.paused ) {

					if ( st.phase === 'seek' ) {

						advance( st.seekTarget, st.seekPerFrame, SEEK_DLNA );
						if ( st.a >= st.seekTarget - 1e-6 ) st.phase = 'run';

					} else if ( st.a < 1 - 1e-6 ) {

						st.tau += dt * S.speed;
						advance( aOfU( st.tau / STORY ), maxRunSteps, MAX_DLNA );
						st.hold = 0;

					} else {

						st.hold += dt;
						if ( S.loop && st.hold > HOLD && st.pendingRestart === null ) st.pendingRestart = 0;

					}

				}

				// fade in/out (restart crossfade)
				if ( st.pendingRestart !== null ) {

					st.fade = Math.max( 0, st.fade - dt * 1.4 );
					if ( st.fade <= 0 ) {

						st.pendingRestart = null;
						newUniverse( false );

					}

				} else {

					st.fade = Math.min( 1, st.fade + dt * 0.9 );

				}

				U.fade.value = st.fade;

				// ── look ──
				const a = st.a;
				U.size.value = S.size;
				U.boxAlpha.value = 0.05 + 0.32 * ( 1 - smooth( 0.08, 0.45, a ) );
				const dist = camera.position.distanceTo( controls.target );
				const tileMix = S.tile === 'on' ? 1 : S.tile === 'off' ? 0 : smooth( 9, 16, dist );
				const vg = { particles: [ 1, 0.1 ], volume: [ 0, 1.0 ], both: [ 0.75, 0.45 ] }[ S.view ];
				U.partGain.value = vg[ 0 ];
				U.volGain.value = vg[ 1 ];
				U.tileGain.value = tileMix * 0.9;
				U.extent.value = tileMix > 0.001 ? 3 : 1;
				vol.scale.setScalar( U.extent.value );
				sprites.visible = vg[ 0 ] > 0 && ! DBG( 'cwnosprite' );
				// exposure rises gently as contrast grows, keeps early epochs readable
				U.exposure.value = 1.6 + 1.4 * ( 1 - smooth( 0.05, 0.4, a ) );

				// markers
				for ( const m of markers ) {

					if ( m.life > 0 ) {

						m.life -= dt / 3.5;
						m.op.value = Math.max( 0, Math.min( 1, m.life * 2.5 ) ) * 0.9;
						m.grow.value = 1 - Math.max( 0, m.life );
						m.sp.visible = m.life > 0;

					}

				}

				// cinematic idle dolly until the user takes over
				if ( ! userMoved ) {

					const off = camera.position.clone().sub( controls.target );
					const want = 9.6 - 2.2 * smooth( 0, 40, t );
					off.setLength( off.length() + ( want - off.length() ) * Math.min( 1, dt * 0.5 ) );
					camera.position.copy( controls.target ).add( off );

				}

				// ── stats ──
				statT += dt;
				if ( statT > 0.5 ) {

					statT = 0;
					const z = 1 / a - 1;
					ctx.setStats( [
						[ 'particles', sizeTxt ],
						[ 'mesh', `${N}³ FFT` ],
						[ 'z', z < 0.005 ? '0.00' : z.toFixed( 2 ) ],
						[ 'a(t)', a.toFixed( 3 ) ],
						[ 'D(a)', cosmo.D( a ).toFixed( 3 ) ],
						[ 'PM steps', String( st.steps ) + ( st.phase === 'seek' ? ' ⟲' : '' ) ],
					] );

				}

			},

			onKey( e ) {

				if ( e.key === 'n' || e.key === 'N' ) newUniverse( false );
				if ( e.key === 'r' || e.key === 'R' ) {

					st.seeds.length = 0; newUniverse( true );

				}

				if ( e.key === ' ' ) st.paused = ! st.paused;
				if ( e.key === 'v' || e.key === 'V' ) {

					S.view = { particles: 'volume', volume: 'both', both: 'particles' }[ S.view ];
					gui.controllersRecursive().forEach( ( c ) => c.updateDisplay() );

				}

			},

			dispose() {

				el.removeEventListener( 'pointerdown', onDown );
				el.removeEventListener( 'pointerup', onUp );
				controls.autoRotate = false;
				pmat.dispose(); vmat.dispose(); boxMat.dispose();
				vol.geometry.dispose(); boxGeo.dispose(); // (Sprite geometry is shared by three.js — never dispose it)
				markers.forEach( ( m ) => m.m.dispose() );
				accPass?.dispose(); overlayPass?.dispose(); bloomNode?.dispose();
				pm.dispose();
				for ( const b of [ pm.grid, pm.C, pm.Dk, pm.rho, pm.F, pm.pos, pm.vel ] ) {

					try {

						renderer._attributes?.delete( b.value );

					} catch ( err ) { /* already gone */ }

				}

			},
		};

	},
};

function smooth( e0, e1, x ) {

	const t = Math.min( 1, Math.max( 0, ( x - e0 ) / ( e1 - e0 ) ) );
	return t * t * ( 3 - 2 * t );

}
