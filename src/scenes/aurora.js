// Chapter 12 — Aurora.
//
// A ground observer on a fjord at 69.6°N under a physically-motivated aurora borealis:
//   · up to 2 M electrons integrated on the GPU with the Boris pusher in an Earth dipole field (+ local perturbations),
//     slowing down in a two-scale-height thermosphere and depositing energy by fixed-point atomicAdd into an
//     observer-centric 3D emission grid (azimuth × altitude × range), decaying each frame;
//   · injection along curtain sheets folded by a slowly evolving Kelvin–Helmholtz-like displacement field;
//   · altitude-dependent O 557.7 / O 630.0 / N₂⁺ 427.8 / N₂ 1P emission, raymarched from the ground,
//     with extinction; stars, Milky Way and airglow; snowy fjord lit by the aurora's GPU-integrated irradiance,
//     mirrored in still water.
// Pointer: left click/drag injects a substorm where you point; right-drag / wheel / ↑↓ look around (look up for the corona).
import * as THREE from 'three/webgpu';
import { uniform, vec4 } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { AuroraSim } from './aurora/sim.js';
import { createEmission, R_E } from './aurora/model.js';
import { createSky } from './aurora/sky.js';
import { createLandscape } from './aurora/terrain.js';
import { height } from './aurora/height.js';

const TIERS = [
	{ NX: 256, NY: 48, NZ: 64, electrons: 16384, maxElectrons: 32768, presets: [ 8192, 16384, 32768 ], nSub: 4, probeSamples: 24, maxSteps: 56, maxStepsRefl: 40, stepScale: 1.25, auroraScale: 0.3, reflScale: 0.3, mwW: 512, mwOct: 3 },
	{ NX: 384, NY: 72, NZ: 112, electrons: 262144, maxElectrons: 524288, presets: [ 131072, 262144, 524288 ], nSub: 3, probeSamples: 48, maxSteps: 110, maxStepsRefl: 70, stepScale: 1.1, auroraScale: 0.5, reflScale: 0.4, mwW: 2048, mwOct: 5 },
	{ NX: 512, NY: 96, NZ: 160, electrons: 1048576, maxElectrons: 2097152, presets: [ 262144, 524288, 1048576, 2097152 ], nSub: 3, probeSamples: 64, maxSteps: 170, maxStepsRefl: 110, stepScale: 1.0, auroraScale: 0.5, reflScale: 0.5, mwW: 2048, mwOct: 5 },
];

const TAU_FAST = 0.55; // s — prompt + O(¹S) (0.7 s in nature)
const TAU_RED = 5.0; // s — O(¹D) (110 s in nature, compressed)
const GAIN_REF = 60 * 1048576; // fixed-point scale × reference electron count
const VOL_NORM = 0.012;
const D2R = Math.PI / 180;

export default {
	id: 'aurora',
	title: 'Aurora',
	subtitle: 'Solar-wind electrons spiral down Earth\'s magnetic field and set oxygen and nitrogen glowing in folded curtains a hundred kilometres above a frozen fjord.',

	async create( ctx ) {

		const { renderer, params } = ctx;
		const DBG = params.has( 'adebug' );
		const T0 = performance.now();
		const log = ( m ) => DBG && console.warn( `[aurora ${( ( performance.now() - T0 ) / 1000 ).toFixed( 1 )}s] ${m}` );
		const tier = Math.max( 0, Math.min( 2, ctx.tier ) );
		const cfg = { ...TIERS[ tier ] };
		if ( params.has( 'electrons' ) ) cfg.electrons = Math.min( cfg.maxElectrons, parseInt( params.get( 'electrons' ) ) );

		const scene = new THREE.Scene();
		const camera = new THREE.PerspectiveCamera( 60, ctx.size.x / ctx.size.y, 0.003, 160 );
		ctx.controls.enabled = false;
		const eyeY = height( 0, 0 ) + 0.0018;
		camera.position.set( 0, eyeY, 0 );

		// ── render targets (reduced resolution, HDR) ──
		const rtOpts = { type: THREE.HalfFloatType, depthBuffer: false };
		const auroraRT = new THREE.RenderTarget( 1, 1, rtOpts );
		auroraRT.texture.minFilter = auroraRT.texture.magFilter = THREE.LinearFilter;
		const reflRT = new THREE.RenderTarget( 1, 1, { type: THREE.HalfFloatType, depthBuffer: true } );
		reflRT.texture.minFilter = reflRT.texture.magFilter = THREE.LinearFilter;

		// ── simulation + sky + landscape ──
		const emission = createEmission();
		log( 'start' );
		const sim = new AuroraSim( renderer, cfg );
		sim.buildProbe( emission.emit );
		const sky = createSky( renderer, sim, emission, cfg, auroraRT );
		const land = createLandscape( { tier, sim, sky, reflRT, eyeY } );
		scene.add( sky.skyMain, sky.skyRefl, land.group );
		log( 'built' );

		const virtualCam = camera.clone();

		// ── state / GUI ──
		const S = {
			solarWind: 0.6, activity: 0.8, speed: 1.0, realism: 0.15, exposure: 1.0, reflection: true,
			electrons: cfg.electrons,
		};
		const uExposure = uniform( S.exposure );
		const gui = ctx.gui;
		gui.add( S, 'solarWind', 0, 1, 0.01 ).name( 'solar wind' );
		gui.add( S, 'activity', 0, 1.5, 0.01 ).name( 'curtain activity' );
		gui.add( S, 'speed', 0, 3, 0.01 ).name( 'time scale' );
		gui.add( S, 'realism', 0, 1, 0.01 ).name( 'vivid ↔ naked eye' ).onChange( ( v ) => emission.setRealism( v ) );
		gui.add( S, 'exposure', 0.2, 4, 0.01 ).name( 'exposure' ).onChange( ( v ) => uExposure.value = v );
		gui.add( S, 'reflection' ).name( 'water reflection' ).onChange( ( v ) => land.u.reflOn.value = v ? 1 : 0 );
		const presetNames = Object.fromEntries( cfg.presets.map( ( n ) => [ n >= 1048576 ? ( n / 1048576 ) + 'M' : ( n / 1024 ) + 'k', n ] ) );
		gui.add( S, 'electrons', presetNames ).name( 'electrons' ).onChange( ( v ) => sim.setCount( v ) );
		const actions = {
			substorm: () => triggerSubstorm( - 12 + Math.random() * 24, 28 + Math.random() * 20 ),
			corona: () => {

				look.coronaMode = ! look.coronaMode;
				look.tPitch = look.coronaMode ? 78 : 12; look.tYaw = 0;

			},
		};
		gui.add( actions, 'substorm' ).name( 'trigger substorm' );
		gui.add( actions, 'corona' ).name( 'look up / back (corona)' );
		emission.setRealism( S.realism );
		sim.setCount( S.electrons );

		// ── camera: fixed observer, yaw/pitch with gentle sway ──
		const look = { yaw: 0, pitch: 12, tYaw: 0, tPitch: 12, coronaMode: false, dragging: false, lastX: 0, lastY: 0, idle: 0 };
		const dom = renderer.domElement;
		const touches = new Map();
		const onDown = ( e ) => {

			touches.set( e.pointerId, e );
			if ( e.button === 2 || e.button === 1 || e.shiftKey || e.altKey || touches.size >= 2 ) {

				look.dragging = true; look.lastX = e.clientX; look.lastY = e.clientY;

			}

		};

		const onMove = ( e ) => {

			if ( touches.has( e.pointerId ) ) touches.set( e.pointerId, e );
			if ( ! look.dragging ) return;
			const k = camera.fov / dom.clientHeight;
			look.tYaw -= ( e.clientX - look.lastX ) * k * ( touches.size >= 2 ? 0.5 : 1 );
			look.tPitch += ( e.clientY - look.lastY ) * k * ( touches.size >= 2 ? 0.5 : 1 );
			look.lastX = e.clientX; look.lastY = e.clientY;
			clampLook(); look.idle = 0;

		};

		const onUp = ( e ) => {

			touches.delete( e.pointerId );
			if ( touches.size === 0 ) look.dragging = false;

		};

		const onWheel = ( e ) => {

			look.tPitch -= e.deltaY * 0.03; clampLook(); look.idle = 0;
			e.preventDefault();

		};

		const onCtx = ( e ) => e.preventDefault();
		const clampLook = () => {

			look.tPitch = Math.max( - 4, Math.min( 84, look.tPitch ) );
			look.tYaw = Math.max( - 75, Math.min( 75, look.tYaw ) );

		};

		dom.addEventListener( 'pointerdown', onDown );
		addEventListener( 'pointermove', onMove );
		addEventListener( 'pointerup', onUp );
		addEventListener( 'pointercancel', onUp );
		dom.addEventListener( 'wheel', onWheel, { passive: false } );
		dom.addEventListener( 'contextmenu', onCtx );

		// ── substorms ──
		const blobs = [ 0, 1, 2, 3 ].map( () => ( { cx: 0, cz: - 300, age: 99, held: false, phase: 0 } ) );
		let flash = 0, activeBlob = null, lastInject = { x: 1e9, z: 1e9 };
		const footprint = ( d ) => {

			// where a view ray reaches the 110 km lower border (with Earth curvature); the mirror image works too
			const dy = Math.max( Math.abs( d.y ), 0.03 );
			const dh = Math.max( Math.hypot( d.x, d.z ), 1e-4 );
			const h = 110, q = 2 * R_E * h + h * h, Rdy = R_E * dy;
			const s = q / ( Math.sqrt( Rdy * Rdy + q ) + Rdy );
			const rho = Math.min( 950, R_E * Math.atan2( s * dh, R_E + s * dy ) );
			const phi = Math.atan2( d.x, - d.z );
			return { x: rho * Math.sin( phi ), z: - rho * Math.cos( phi ) };

		};

		const newBlob = ( x, z ) => {

			let b = blobs[ 0 ];
			for ( const c of blobs ) if ( c.age > b.age ) b = c;
			Object.assign( b, { cx: x, cz: z, age: 0, held: false, phase: Math.random() * 6.28 } );
			flash = 1;
			return b;

		};

		const dirFromAzEl = ( az, el ) => new THREE.Vector3( Math.sin( az * D2R ) * Math.cos( el * D2R ), Math.sin( el * D2R ), - Math.cos( az * D2R ) * Math.cos( el * D2R ) );
		const triggerSubstorm = ( az, el ) => {

			const f = footprint( dirFromAzEl( az, el ) );
			newBlob( f.x, f.z );

		};

		const blobStrength = ( b ) => {

			const a = b.age;
			if ( a > 40 ) return 0;
			const onset = Math.min( 1, a / 0.5 ) ** 2;
			const peak = 1 + 0.8 * Math.exp( - ( ( a - 1.5 ) ** 2 ) / 2 );
			const decay = a < 5 ? 1 : Math.exp( - ( a - 5 ) / 7 );
			return onset * peak * decay;

		};

		const updateSubstorms = ( dt, simDt ) => {

			const p = ctx.pointer;
			const injecting = p.down && p.button === 0 && ! look.dragging && touches.size < 2 && ! ctx.shot;
			if ( injecting ) {

				const f = footprint( p.ray.direction );
				if ( ! activeBlob ) {

					activeBlob = newBlob( f.x, f.z );
					lastInject = f;

				} else if ( Math.hypot( f.x - lastInject.x, f.z - lastInject.z ) > 120 ) {

					// dragging paints new surges along the path
					activeBlob.held = false;
					activeBlob = newBlob( f.x, f.z );
					lastInject = f;

				}

				activeBlob.held = true;
				activeBlob.cx += ( f.x - activeBlob.cx ) * Math.min( 1, dt * 6 );
				activeBlob.cz += ( f.z - activeBlob.cz ) * Math.min( 1, dt * 6 );
				look.idle = 0;

			} else if ( activeBlob ) {

				activeBlob.held = false; activeBlob = null;

			}

			let total = 0;
			const st = [];
			blobs.forEach( ( b, k ) => {

				b.age += simDt;
				if ( b.held ) b.age = Math.min( b.age, 2.2 );
				const s = blobStrength( b );
				const R = 35 + 190 * ( 1 - Math.exp( - b.age / 2.8 ) );
				const E = 2.5 + 8.5 * Math.exp( - b.age / 9 );
				sim.subA.array[ k ].set( b.cx, b.cz, R, b.phase - b.age * 0.45 );
				sim.subB.array[ k ].set( s, E, b.age, b.phase );
				st.push( s ); total += s;

			} );
			let acc = 0;
			const cum = st.map( ( s ) => ( acc += s / Math.max( total, 1e-6 ) ) );
			sim.u.subCum.value.set( cum[ 0 ], cum[ 1 ], cum[ 2 ], 1.01 );
			sim.u.pSub.value = Math.min( 0.6, total * 0.3 );
			flash *= Math.exp( - dt / 2.0 );
			return total;

		};

		// ── activity → physics parameters ──
		let simTime = 0, subLevel = 0, kp = 0;
		const applyActivity = () => {

			const sw = S.solarWind, act = S.activity;
			const D = 470 - 320 * sw; // the oval moves equatorward (toward the observer) as the solar wind strengthens
			const subE = Math.min( 1, subLevel );
			sim.arcs.array[ 0 ].set( D, 3.2 + 5.5 * sw + 2 * subE, 1.0, 0.3 );
			sim.arcs.array[ 1 ].set( D + 115, 2.4 + 3 * sw, 0.85, 2.1 );
			sim.arcs.array[ 2 ].set( D + 290, 1.8 + 2 * sw, 0.7, 4.4 );
			const w0 = 1, w1 = 0.35 + 0.6 * sw, w2 = 0.3;
			const ws = w0 + w1 + w2;
			sim.u.arcCum.value.set( w0 / ws, ( w0 + w1 ) / ws, 1 );
			sim.u.diffD.value = D - 25;
			sim.u.act.value = act;
			const t = simTime;
			const boost = 1 + 0.6 * subE;
			sim.u.fold.value.set(
				act * ( 0.45 + 0.35 * Math.sin( 0.045 * t ) ) * boost,
				act * ( 0.9 + 0.6 * Math.sin( 0.07 * t + 1 ) ) * boost,
				act * ( 1.1 + 0.6 * Math.sin( 0.11 * t + 2 ) ) * boost );
			sim.u.drift.value.set( 2.5 * Math.sin( 0.03 * t ), 0, 0.8 );
			kp = Math.max( 0, Math.min( 9, 0.6 + 6.2 * sw + 2.6 * Math.min( 1, subLevel * 0.9 ) + 0.4 * ( act - 0.8 ) ) );

		};

		const flux = () => ( 0.35 + 1.25 * S.solarWind ) * ( 1 + 0.9 * flash ) * ( 1 + 0.35 * Math.min( 1, subLevel ) );

		const setDecay = ( dt, tauF, tauR ) => {

			sim.u.decay.value = Math.exp( - dt / tauF );
			sim.u.kRed.value = 1 - Math.exp( - dt / tauR );

		};

		// ── prewarm: start already mid-display, with an onset in progress ──
		await sim.init();
		log( 'sim init' );
		await sky.init();
		log( 'sky init' );
		triggerSubstorm( - 16, 30 );
		blobs.forEach( ( b ) => b.age === 0 && ( b.age = 0.6 ) );
		{

			const nPre = tier === 0 ? 14 : 40;
			const dtp = 1 / 30;
			for ( let i = 0; i < nPre; i ++ ) {

				const k = i / nPre;
				simTime += dtp;
				subLevel = updateSubstorms( 0, dtp * 0.5 );
				applyActivity();
				sim.u.time.value = simTime;
				sim.u.frame.value = i + 1;
				sim.u.dt.value = dtp / cfg.nSub;
				// shortened time constants (and matching gain) converge to the steady state in a few steps
				const tf = THREE.MathUtils.lerp( 0.12, TAU_FAST, k * k );
				setDecay( dtp, tf, THREE.MathUtils.lerp( 0.25, TAU_RED, k * k ) );
				sim.u.gain.value = GAIN_REF / sim.count * flux() * ( TAU_FAST / tf );
				sim.u.lightK.value = i === 0 ? 1 : 0.3;
				sim.u.volNorm.value = VOL_NORM;
				sim.step();
				if ( DBG && ( i < 2 || i === nPre - 1 ) ) {

					await renderer.computeAsync( sim.probeResolve );
					log( 'prewarm ' + i );

				}

			}

		}

		sim.u.lightK.value = 0.08;
		land.u.reflOn.value = S.reflection ? 1 : 0;

		// ── post: exposure + bloom (shell adds grade + AgX) ──
		let bloomNode = null;
		const post = ( _ctx, scenePass ) => {

			const c = scenePass.getTextureNode( 'output' );
			const ce = vec4( c.rgb.mul( uExposure ), 1 );
			bloomNode = bloom( ce, 0.55, 0.55, 0.06 );
			return ce.add( bloomNode );

		};

		// ── per-frame ──
		const size = new THREE.Vector2();
		let frame = 100, statT = 0;
		const resize = () => {

			renderer.getDrawingBufferSize( size );
			auroraRT.setSize( Math.max( 1, Math.round( size.x * cfg.auroraScale ) ), Math.max( 1, Math.round( size.y * cfg.auroraScale ) ) );
			reflRT.setSize( Math.max( 1, Math.round( size.x * cfg.reflScale ) ), Math.max( 1, Math.round( size.y * cfg.reflScale ) ) );
			const fovR = camera.fov * D2R;
			sky.u.pixMain.value = fovR / size.y;
			sky.u.pixRefl.value = fovR / reflRT.height;
			land.u.pix.value = fovR / size.y;

		};

		resize();

		const _v = new THREE.Vector3(), _t = new THREE.Vector3();
		const updateVirtualCamera = () => {

			// planar mirror about y = 0 (same construction as ReflectorNode, sampled with screenUV.flipX())
			camera.updateMatrixWorld();
			virtualCam.copy( camera );
			const cp = camera.position;
			virtualCam.position.set( cp.x, - cp.y, cp.z );
			_v.set( 0, 0, - 1 ).applyQuaternion( camera.quaternion ).add( cp );
			_t.set( _v.x, - _v.y, _v.z );
			_v.set( 0, 1, 0 ).applyQuaternion( camera.quaternion );
			virtualCam.up.set( _v.x, - _v.y, _v.z );
			virtualCam.lookAt( _t );
			virtualCam.updateMatrixWorld();
			virtualCam.projectionMatrix.copy( camera.projectionMatrix );
			virtualCam.projectionMatrixInverse.copy( camera.projectionMatrixInverse );

		};

		const kpStr = ( k ) => {

			const n = Math.round( k * 3 ) / 3;
			const base = Math.round( n );
			const frac = n - base;
			return frac > 0.1 ? `${base}+` : frac < - 0.1 ? `${base}−` : `${base}o`;

		};

		const update = ( dt, t ) => {

			frame ++;
			if ( DBG && frame < 110 ) log( 'frame ' + frame );
			const dts = Math.min( dt, 1 / 20 );
			simTime += dts * S.speed * ( 0.55 + 0.55 * S.activity );
			const subDt = dts * Math.max( 0.25, S.speed );
			subLevel = updateSubstorms( dts, subDt );
			applyActivity();

			// idle: a substorm now and then, so an unattended screen keeps breathing
			look.idle += dts;
			if ( ! ctx.shot && look.idle > 38 ) {

				look.idle = 0;
				triggerSubstorm( - 35 + Math.random() * 70, 18 + Math.random() * 30 );

			}

			// sim uniforms
			sim.u.time.value = simTime;
			sim.u.frame.value = frame;
			sim.u.dt.value = Math.min( dt, 1 / 30 ) * Math.max( 0.3, Math.min( S.speed, 1.5 ) ) / cfg.nSub;
			setDecay( dts * Math.max( 0.3, S.speed ), TAU_FAST, TAU_RED );
			sim.u.gain.value = GAIN_REF / sim.count * flux() * Math.min( 1.5, Math.max( 0.3, S.speed ) );
			sim.u.volNorm.value = VOL_NORM;
			sim.step();

			// camera
			const sw = Math.sin( t * 0.21 ) * 0.6 + Math.sin( t * 0.077 + 1 ) * 1.4;
			const k = 1 - Math.exp( - dts * 3.5 );
			look.yaw += ( look.tYaw - look.yaw ) * k;
			look.pitch += ( look.tPitch - look.pitch ) * k;
			camera.rotation.set( ( look.pitch + Math.sin( t * 0.13 ) * 0.5 ) * D2R, ( - look.yaw - sw ) * D2R, Math.sin( t * 0.09 ) * 0.25 * D2R, 'YXZ' );
			camera.updateMatrixWorld();

			// sky / landscape uniforms
			sky.u.time.value = t;
			sky.u.frame.value = frame % 64;
			sky.setSidereal( 0.35 + t * 0.0012 );
			land.u.time.value = t;
			land.u.cabinGlow.value = 1;

			const prevRT = renderer.getRenderTarget();
			// 1) mirrored scene for the water
			if ( S.reflection ) {

				updateVirtualCamera();
				land.water.visible = false; sky.skyMain.visible = false; sky.skyRefl.visible = true;
				renderer.setRenderTarget( reflRT );
				renderer.render( scene, virtualCam );
				land.water.visible = true; sky.skyMain.visible = true; sky.skyRefl.visible = false;

			}

			// 2) aurora raymarch at reduced resolution for the main view
			sky.u.invProj.value.copy( camera.projectionMatrixInverse );
			sky.u.camWorld.value.copy( camera.matrixWorld );
			renderer.setRenderTarget( auroraRT );
			sky.quad.render( renderer );
			renderer.setRenderTarget( prevRT );

			statT += dt;
			if ( statT > 0.5 ) {

				statT = 0;
				ctx.setStats( [
					[ 'electrons', sim.count.toLocaleString() ],
					[ 'volume', `${cfg.NX}×${cfg.NY}×${cfg.NZ}` ],
					[ 'Kp', kpStr( kp ) ],
				] );

			}

		};

		const onKey = ( e ) => {

			if ( e.key === 'ArrowUp' || e.key === 'w' ) {

				look.tPitch += 6; clampLook();

			} else if ( e.key === 'ArrowDown' || e.key === 's' ) {

				look.tPitch -= 6; clampLook();

			} else if ( e.key === 'c' || e.key === 'C' ) actions.corona();
			else if ( e.key === ' ' || e.key === 'b' ) actions.substorm();

		};

		return {
			scene, camera, post, update, onKey,
			resize() {

				resize();

			},
			dispose() {

				dom.removeEventListener( 'pointerdown', onDown );
				removeEventListener( 'pointermove', onMove );
				removeEventListener( 'pointerup', onUp );
				removeEventListener( 'pointercancel', onUp );
				dom.removeEventListener( 'wheel', onWheel );
				dom.removeEventListener( 'contextmenu', onCtx );
				bloomNode?.dispose();
				sim.dispose(); sky.dispose(); land.dispose();
				auroraRT.dispose(); reflRT.dispose();

			},
		};

	},
};
