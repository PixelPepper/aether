// Chapter 06 — Atmosphere.
// A young Earth-like planet seen from orbit at the day/night terminator. Hillaire 2020 sky/atmosphere LUTs
// (transmittance, multiple scattering, sky-view, aerial-perspective froxels) are rebuilt by compute every
// frame; the planet surface, Perlin–Worley cloud shell and sun are ray-traced in one full-screen pass.
import * as THREE from 'three/webgpu';
import { uniform, vec4, min } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { createAtmosphere, RB, RT } from './atmosphere/atmo.js';
import { createPlanet } from './atmosphere/planet.js';

const DEG = Math.PI / 180;

const TIERS = [
	// tier 0: SwiftShader / headless — correct but lean
	{ mapW: 512, mapH: 256, bakeOct: 0.6, cloudN: 32, skyW: 96, skyH: 54, skySteps: 16, apW: 32, apH: 18, apD: 24, apSub: 2, skyPixelSteps: 14, cloudSteps: 12, lightSteps: 2, detailOctaves: 3, cloudDetail: false },
	// tier 1: mobile
	{ mapW: 2048, mapH: 1024, bakeOct: 1, cloudN: 64, skyW: 192, skyH: 108, skySteps: 24, apW: 48, apH: 27, apD: 32, apSub: 2, skyPixelSteps: 24, cloudSteps: 20, lightSteps: 3, detailOctaves: 4, cloudDetail: false },
	// tier 2: desktop discrete
	{ mapW: 4096, mapH: 2048, bakeOct: 1, cloudN: 128, skyW: 192, skyH: 108, skySteps: 30, apW: 96, apH: 54, apD: 48, apSub: 3, skyPixelSteps: 40, cloudSteps: 40, lightSteps: 4, detailOctaves: 5, cloudDetail: true },
];

export default {
	id: 'atmosphere',
	title: 'Atmosphere',
	subtitle: 'Rayleigh, Mie and ozone light transport through 100 km of air — transmittance, multiple-scattering, sky-view and aerial-perspective tables recomputed on the GPU every frame, over a planet whose clouds, oceans and cities are all procedural.',
	async create( ctx ) {

		const { renderer, camera, controls } = ctx;
		const cfg = { ...TIERS[ Math.max( 0, Math.min( 2, ctx.tier ) ) ], debugView: ctx.params.has( 'atmview' ) };
		const scene = new THREE.Scene();
		scene.background = new THREE.Color( 0x000000 );

		controls.enabled = false;
		camera.fov = 42;
		camera.near = 1;
		camera.far = 1e5;
		camera.updateProjectionMatrix();

		// ───────────── uniforms (km, planet-centred) ─────────────
		const U = {
			sunDir: uniform( new THREE.Vector3( 0, 0, 1 ) ),
			camPos: uniform( new THREE.Vector3( 0, RB + 500, 0 ) ),
			camFwd: uniform( new THREE.Vector3( 0, 0, - 1 ) ),
			camRight: uniform( new THREE.Vector3( 1, 0, 0 ) ),
			camUp: uniform( new THREE.Vector3( 0, 1, 0 ) ),
			tanHalf: uniform( new THREE.Vector2( 0.5, 0.4 ) ),
			pixelAngle: uniform( 0.001 ),
			viewR: uniform( RB + 1 ),
			viewSunZenith: uniform( 0 ),
			apDmax: uniform( 1200 ),
			rayScale: uniform( 1 ),
			mieScale: uniform( 1 ),
			ozoneScale: uniform( 1 ),
			mieG: uniform( 0.8 ),
			coverage: uniform( 0.52 ),
			cloudDensity: uniform( 1.0 ),
			exposure: uniform( 10 ),
			planetRot: uniform( 0 ),
			cloudRot: uniform( 0 ),
			cloudTime: uniform( 0 ),
			bump: uniform( 5.0 ),
			cityLights: uniform( 1.4 ),
			airglow: uniform( 2.2e-5 ),
		};

		const atmo = createAtmosphere( U, cfg );
		const planet = createPlanet( U, atmo, cfg );
		scene.add( planet.mesh );

		// one-off procedural bakes
		const dbg = ctx.params.has( 'atmdbg' );
		const t0 = performance.now();
		const mark = async ( label ) => {

			if ( ! dbg ) return;
			await renderer.backend.device?.queue.onSubmittedWorkDone();
			console.warn( `[atm] ${label} ${( performance.now() - t0 ).toFixed( 0 )} ms` );

		};

		await renderer.computeAsync( planet.bakeCompute );
		await mark( 'bake planet map' );
		await renderer.computeAsync( planet.cloudNoiseCompute );
		await mark( 'cloud noise volume' );
		await renderer.computeAsync( atmo.transCompute );
		await renderer.computeAsync( atmo.msCompute );
		await renderer.computeAsync( atmo.irrCompute );
		await mark( 'transmittance + multi-scattering + irradiance LUTs' );

		// ───────────── camera rig: anchor point on the sphere + altitude + heading ─────────────
		const ORBIT_ALT = 520;
		const rig = {
			anchor: new THREE.Vector3(), // unit vector under the camera
			heading: new THREE.Vector3(), // tangent at anchor
			alt: ORBIT_ALT,
			altTarget: ORBIT_ALT,
			anchorTarget: null,
			roll: - 9 * DEG,
			sway: 0,
		};
		const lat0 = 16 * DEG, lon0 = 0.6;
		rig.anchor.set( Math.cos( lat0 ) * Math.cos( lon0 ), Math.sin( lat0 ), Math.cos( lat0 ) * Math.sin( lon0 ) );
		const east = new THREE.Vector3( 0, 1, 0 ).cross( rig.anchor ).normalize().negate();
		rig.heading.copy( east ).applyAxisAngle( rig.anchor, 12 * DEG );

		// sun placed "just behind the limb", slightly right of the view heading
		const sun = { dir: new THREE.Vector3(), hour: 0, decl: 0, base: new THREE.Vector3() };
		{

			const dip = Math.acos( RB / ( RB + ORBIT_ALT ) );
			const e = - dip + 0.25 * DEG; // a sliver of the disk clears the hard limb, deep in the red twilight layer
			const a = 13 * DEG;
			const right = new THREE.Vector3().crossVectors( rig.heading, rig.anchor ).normalize();
			sun.base.copy( rig.heading ).multiplyScalar( Math.cos( a ) ).addScaledVector( right, Math.sin( a ) ).multiplyScalar( Math.cos( e ) ).addScaledVector( rig.anchor, Math.sin( e ) ).normalize();

		}

		const params = { timeOfDay: 0, altitude: ORBIT_ALT, returnToOrbit: () => flyTo( null ) };

		const updateSun = () => {

			// time of day = rotation of the sun about the planet axis; declination = tilt toward a pole
			const d = sun.base.clone().applyAxisAngle( new THREE.Vector3( 0, 1, 0 ), ( params.timeOfDay + sun.hour ) * DEG );
			const axis = new THREE.Vector3( 0, 1, 0 ).cross( d ).normalize();
			if ( axis.lengthSq() > 1e-6 ) d.applyAxisAngle( axis, - sun.decl * DEG );
			sun.dir.copy( d.normalize() );
			U.sunDir.value.copy( sun.dir );

		};

		updateSun();

		// ───────────── LUT bookkeeping ─────────────
		let lutDirty = false;
		let frameN = 0;
		const markDirty = () => ( lutDirty = true );

		// ───────────── GUI ─────────────
		const gui = ctx.gui;
		gui.add( params, 'timeOfDay', - 180, 180, 0.1 ).name( 'sun angle (°)' ).onChange( updateSun ).listen();
		gui.add( U.rayScale, 'value', 0, 4, 0.01 ).name( 'Rayleigh scale' ).onChange( markDirty );
		gui.add( U.mieScale, 'value', 0, 10, 0.01 ).name( 'Mie scale' ).onChange( markDirty );
		gui.add( U.coverage, 'value', 0, 1, 0.01 ).name( 'cloud coverage' );
		gui.add( U.cloudDensity, 'value', 0.05, 3, 0.01 ).name( 'cloud density' );
		gui.add( U.exposure, 'value', 1, 40, 0.1 ).name( 'exposure' );
		gui.add( params, 'altitude', 4, 3000, 1 ).name( 'altitude (km)' ).onChange( ( v ) => ( rig.altTarget = v ) ).listen();
		gui.add( params, 'returnToOrbit' ).name( 'return to orbit' );

		// ───────────── pointer: drag = move the sun, click = fly down to that point ─────────────
		const el = renderer.domElement;
		const drag = { active: false, x: 0, y: 0, moved: 0, t: 0 };
		const onDown = ( e ) => {

			drag.active = true; drag.x = e.clientX; drag.y = e.clientY; drag.moved = 0; drag.t = performance.now();

		};

		const onMove = ( e ) => {

			if ( ! drag.active ) return;
			const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
			drag.x = e.clientX; drag.y = e.clientY;
			drag.moved += Math.abs( dx ) + Math.abs( dy );
			const k = 110 / Math.max( 300, ctx.size.x );
			params.timeOfDay = THREE.MathUtils.euclideanModulo( params.timeOfDay + dx * k + 180, 360 ) - 180;
			sun.decl = THREE.MathUtils.clamp( sun.decl + dy * k * 0.6, - 40, 40 );
			updateSun();

		};

		const onUp = ( e ) => {

			if ( ! drag.active ) return;
			drag.active = false;
			if ( drag.moved < 6 && performance.now() - drag.t < 400 ) pick( e.clientX, e.clientY );

		};

		const onWheel = ( e ) => {

			rig.altTarget = THREE.MathUtils.clamp( rig.altTarget * Math.exp( e.deltaY * 0.0012 ), 4, 3000 );
			params.altitude = Math.round( rig.altTarget );

		};

		el.addEventListener( 'pointerdown', onDown );
		window.addEventListener( 'pointermove', onMove );
		window.addEventListener( 'pointerup', onUp );
		el.addEventListener( 'wheel', onWheel, { passive: true } );

		const basis = { fwd: new THREE.Vector3(), right: new THREE.Vector3(), up: new THREE.Vector3() };
		const tmp = new THREE.Vector3();

		function rayFromScreen( cx, cy, out ) {

			const nx = ( cx / ctx.size.x ) * 2 - 1, ny = 1 - ( cy / ctx.size.y ) * 2;
			return out.copy( basis.fwd ).addScaledVector( basis.right, nx * U.tanHalf.value.x ).addScaledVector( basis.up, ny * U.tanHalf.value.y ).normalize();

		}

		function pick( cx, cy ) {

			const rd = rayFromScreen( cx, cy, new THREE.Vector3() );
			const ro = U.camPos.value;
			const b = ro.dot( rd );
			const perp = tmp.copy( ro ).addScaledVector( rd, - b );
			const disc = RB * RB - perp.lengthSq();
			if ( disc < 0 || - b - Math.sqrt( disc ) < 0 ) {

				flyTo( null );
				return;

			}

			const hit = ro.clone().addScaledVector( rd, - b - Math.sqrt( disc ) );
			flyTo( hit.normalize() );

		}

		function flyTo( dir ) {

			if ( dir ) {

				rig.anchorTarget = dir.clone();
				rig.altTarget = 9;

			} else {

				rig.anchorTarget = null;
				rig.altTarget = ORBIT_ALT;

			}

			params.altitude = Math.round( rig.altTarget );

		}

		const onKey = ( e ) => {

			if ( e.key === 'o' || e.key === 'O' ) flyTo( null );

		};

		// ───────────── per-frame camera ─────────────
		const updateCamera = ( dt, t ) => {

			// altitude: exponential approach in log space (orbit ↔ 9 km in ~6 s)
			const la = Math.log( rig.alt ), lt = Math.log( rig.altTarget );
			rig.alt = Math.exp( la + ( lt - la ) * ( 1 - Math.exp( - dt * 0.7 ) ) );
			if ( rig.anchorTarget ) {

				const ang = rig.anchor.angleTo( rig.anchorTarget );
				if ( ang > 1e-5 ) {

					const step = ang * ( 1 - Math.exp( - dt * 0.9 ) );
					const axis = tmp.crossVectors( rig.anchor, rig.anchorTarget ).normalize();
					rig.anchor.applyAxisAngle( axis, step ).normalize();
					rig.heading.applyAxisAngle( axis, step );

				}

			}

			// keep heading tangent; near the ground, swing it toward the sun's azimuth so the descent ends on a sunrise
			rig.heading.addScaledVector( rig.anchor, - rig.heading.dot( rig.anchor ) ).normalize();
			const low = 1 - THREE.MathUtils.smoothstep( Math.log( rig.alt ), Math.log( 15 ), Math.log( 300 ) );
			if ( low > 0 ) {

				const sunH = tmp.copy( sun.dir ).addScaledVector( rig.anchor, - sun.dir.dot( rig.anchor ) );
				if ( sunH.lengthSq() > 1e-6 ) {

					sunH.normalize();
					rig.heading.lerp( sunH, Math.min( 1, dt * 0.8 * low ) ).normalize();

				}

			}

			const r = RB + rig.alt;
			const dip = Math.acos( RB / r );
			// horizon sits ~9° above frame centre in orbit, ~6° below it near the ground (looking up into the sky)
			const above = THREE.MathUtils.lerp( 8.5, - 5.5, low ) * DEG;
			rig.sway = Math.sin( t * 0.07 ) * 1.2 * DEG;
			const pitch = dip + above;
			const up = rig.anchor;
			const fwd = basis.fwd.copy( rig.heading ).applyAxisAngle( up, rig.sway ).multiplyScalar( Math.cos( pitch ) ).addScaledVector( up, - Math.sin( pitch ) ).normalize();
			const right = basis.right.crossVectors( fwd, up ).normalize();
			const cup = basis.up.crossVectors( right, fwd ).normalize();
			const roll = rig.roll * ( 1 - low * 0.7 );
			right.applyAxisAngle( fwd, roll );
			cup.applyAxisAngle( fwd, roll );

			U.camPos.value.copy( up ).multiplyScalar( r );
			U.camFwd.value.copy( fwd );
			U.camRight.value.copy( right );
			U.camUp.value.copy( cup );
			const ty = Math.tan( camera.fov * DEG * 0.5 );
			U.tanHalf.value.set( ty * camera.aspect, ty );
			const hPx = Math.max( 1, ctx.size.y * renderer.getPixelRatio() );
			U.pixelAngle.value = ( 2 * ty ) / hPx;

			// keep the shared camera in sync (km units) for anything that reads it
			camera.position.copy( U.camPos.value );
			camera.up.copy( cup );
			camera.lookAt( tmp.copy( camera.position ).add( fwd ) );
			camera.updateMatrixWorld();

			// sky-view LUT viewpoint (clamped inside the atmosphere) and AP depth range
			const viewR = THREE.MathUtils.clamp( r, RB + 0.02, RT - 0.02 );
			U.viewR.value = viewR;
			U.viewSunZenith.value = up.dot( sun.dir );
			const horizon = Math.sqrt( Math.max( r * r - RB * RB, 0 ) );
			U.apDmax.value = r > RT ? Math.sqrt( RT * RT - RB * RB ) + 60 : Math.min( horizon + 450, 1400 );

		};

		// ───────────── stats ─────────────
		const sz = atmo.sizes;
		let statT = 1;
		const stats = () => {

			const sunElev = Math.asin( THREE.MathUtils.clamp( rig.anchor.dot( sun.dir ), - 1, 1 ) ) / DEG;
			ctx.setStats( [
				[ 'LUTs', `T ${sz.TW}×${sz.TH} · MS ${sz.MS}² · sky ${sz.SW}×${sz.SH}` ],
				[ 'froxels', `${sz.AW}×${sz.AH}×${sz.AD}` ],
				[ 'clouds', `${cfg.cloudN}³ PW · ${cfg.cloudSteps} steps` ],
				[ 'altitude', `${rig.alt < 20 ? rig.alt.toFixed( 1 ) : Math.round( rig.alt )} km` ],
				[ 'sun elev.', `${sunElev.toFixed( 1 )}°` ],
			] );

		};

		const post = ( ctx2, scenePass ) => {

			const color = scenePass.getTextureNode( 'output' );
			const b = bloom( min( color, vec4( 60 ) ), 0.55, 0.65, 0.9 );
			return color.add( b );

		};

		return {
			scene,
			post,
			update( dt, t ) {

				// slow orbital drift: the ground and (slightly faster) the cloud deck turn beneath a fixed sun/camera
				const orbitF = THREE.MathUtils.smoothstep( rig.alt, 20, 300 );
				U.planetRot.value += dt * 0.0016 * orbitF;
				U.cloudRot.value = U.planetRot.value + t * 0.00025;
				U.cloudTime.value = t * 0.002;

				updateCamera( dt, t );

				if ( lutDirty ) {

					renderer.compute( atmo.transCompute );
					renderer.compute( atmo.msCompute );
					renderer.compute( atmo.irrCompute );
					lutDirty = false;

				}

				if ( rig.alt < RT - RB + 1 ) renderer.compute( atmo.skyCompute );
				renderer.compute( atmo.apCompute );

				if ( dbg && frameN < 4 ) {

					const n = frameN;
					renderer.backend.device?.queue.onSubmittedWorkDone().then( () => console.warn( `[atm] frame ${n} computes done ${( performance.now() - t0 ).toFixed( 0 )} ms` ) );

				}

				frameN ++;
				statT += dt;
				if ( statT > 0.5 ) {

					statT = 0;
					stats();

				}

			},
			onKey,
			dispose() {

				el.removeEventListener( 'pointerdown', onDown );
				window.removeEventListener( 'pointermove', onMove );
				window.removeEventListener( 'pointerup', onUp );
				el.removeEventListener( 'wheel', onWheel );
				atmo.dispose();
				planet.dispose();
				scene.remove( planet.mesh );

			},
		};

	},
};
