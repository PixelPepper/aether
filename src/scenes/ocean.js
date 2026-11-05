// Chapter 04 — Primordial Sea.
// A real-time 3D MLS-MPM fluid (fixed-point atomic P2G, APIC G2P, Tait EOS + viscosity) inside a glass orb,
// rendered with screen-space fluid rendering (sphere depth → bilateral smoothing → normals, refraction,
// Beer–Lambert absorption, Fresnel reflection of a procedural nebula) and shear-triggered bioluminescence.
import * as THREE from 'three/webgpu';
import { uniform, convertToTexture } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { MPMFluid } from './ocean/mpm.js';
import { FluidRenderer } from './ocean/ssfr.js';
import { bakeEnvironment, ORB_C, ORB_R } from './ocean/env.js';

const PRESETS = {
	'low · 16k · 32³': { count: 16000, grid: 32, sub: 2 },
	'medium · 55k · 48³': { count: 55000, grid: 48, sub: 2 },
	'high · 180k · 64³': { count: 180000, grid: 64, sub: 3 },
	'ultra · 260k · 72³': { count: 260000, grid: 72, sub: 3 },
};
const TIER_PRESET = [ 'low · 16k · 32³', 'medium · 55k · 48³', 'high · 180k · 64³' ];

export default {
	id: 'ocean',
	title: 'Primordial Sea',
	subtitle: 'A material-point-method ocean: particles trade momentum through a grid each substep, and the strained water flashes like plankton.',

	async create( ctx ) {

		const { renderer, camera, controls, pointer } = ctx;
		const DBG = ctx.params.has( 'odebug' );
		let dbgT = performance.now();
		const dbg = async ( label ) => {

			if ( ! DBG ) return;
			await renderer.backend.device?.queue.onSubmittedWorkDone();
			const now = performance.now();
			console.log( `[ocean] ${label}: ${( now - dbgT ).toFixed( 0 )} ms` );
			dbgT = now;

		};

		const scene = new THREE.Scene(); // everything is drawn by our own passes + the analytic composite

		// ── shared uniforms (survive simulation rebuilds) ──
		const U = {
			// sim
			rest: uniform( 5 ), stiffK: uniform( 0.25 ), pressureMax: uniform( 6 ), mu: uniform( 0.05 ),
			gravity: uniform( new THREE.Vector3( 0, - 0.003, 0 ) ), maxSpeed: uniform( 0.8 ),
			glowLo: uniform( 0.12 ), glowHi: uniform( 0.45 ), glowDecay: uniform( 0.99 ),
			mirror: uniform( 0 ), blobStart: uniform( 0, 'int' ), blobCenter: uniform( new THREE.Vector3() ), blobVel: uniform( new THREE.Vector3() ),
			ptrOrigin: uniform( new THREE.Vector3() ), ptrDir: uniform( new THREE.Vector3( 0, 0, - 1 ) ), ptrVel: uniform( new THREE.Vector3() ),
			ptrRadius: uniform( 3 ), ptrStrength: uniform( 0 ), ptrRange: uniform( new THREE.Vector2() ),
			// render
			glow: uniform( 1 ), absorb: uniform( 1 ), radiusMul: uniform( 0.78 ), smoothing: uniform( 3.0 ),
			pixAngle: uniform( 0.002 ), time: uniform( 0 ), pool: uniform( 0.5 ),
		};

		const params = {
			preset: TIER_PRESET[ Math.min( 2, Math.max( 0, ctx.tier ) ) ],
			viscosity: 0.25,
			stiffness: 1.0,
			gravity: 1.0,
			tilt: 0,
			tide: 0.5,
			glow: 1.0,
			autoDrops: true,
		};

		// ── environment + fluid renderer ──
		const envRT = bakeEnvironment( renderer, ctx.tier === 0 ? 1024 : 2048 );
		await dbg( 'env bake' );
		const fluid = new FluidRenderer( renderer, camera, U, envRT, {
			fluidScale: ctx.tier === 2 ? 1 : 1,
			thickScale: 0.5,
			iterations: ctx.tier === 0 ? 2 : 3,
		} );

		const dbSize = new THREE.Vector2();
		const syncSize = () => {

			renderer.getDrawingBufferSize( dbSize );
			// cap the fluid buffers on very large canvases (they are smooth anyway)
			fluid.fluidScale = Math.min( 1, 2200 / Math.max( 1, dbSize.x ) );
			fluid.setSize( dbSize.x, dbSize.y );

		};

		syncSize();

		// ── simulation ──
		let sim = null, substeps = 2, stepsPerSec = 120;
		const build = async ( name ) => {

			const pr = PRESETS[ name ] || PRESETS[ TIER_PRESET[ 0 ] ];
			sim?.dispose();
			sim = new MPMFluid( renderer, { count: pr.count, grid: pr.grid, U } );
			substeps = pr.sub;
			stepsPerSec = 60 * substeps;
			fluid.setSim( sim );
			applyParams();
			await renderer.computeAsync( sim.stepList ); // compile + first step
			statsDirty = true;

		};

		let statsDirty = true;
		const applyParams = () => {

			if ( ! sim ) return;
			const rho = sim.restDensity;
			U.rest.value = rho;
			// sound speed ≈ 0.5 cells/step at stiffness 1 (c² = kγ/ρ0, γ = 5)
			U.stiffK.value = 0.05 * rho * params.stiffness;
			U.pressureMax.value = 40 * U.stiffK.value;
			U.mu.value = params.viscosity * 0.12 * rho * 0.6;
			const g = params.gravity * 3.1 * sim.radius / ( stepsPerSec * stepsPerSec );
			gMag = g;
			const vc = Math.sqrt( Math.max( 1e-6, 3.1 * sim.radius / ( stepsPerSec * stepsPerSec ) ) * sim.radius );
			U.glowLo.value = 0.42 * vc;
			U.glowHi.value = 1.5 * vc;
			U.glowDecay.value = Math.exp( - 1 / ( stepsPerSec * 0.75 ) );
			U.maxSpeed.value = 0.8;
			U.ptrRadius.value = sim.radius * 0.16;
			U.glow.value = params.glow;

		};

		let gMag = 0.003;
		await build( params.preset );
		await dbg( 'sim build + compile' );

		// ── camera ──
		camera.fov = 40;
		camera.updateProjectionMatrix();
		const target = new THREE.Vector3( 0, 0.92, 0 );
		const az = THREE.MathUtils.degToRad( 38 ), el = THREE.MathUtils.degToRad( 23 ), dist = 4.7;
		camera.position.set( Math.cos( el ) * Math.sin( az ), Math.sin( el ), Math.cos( el ) * Math.cos( az ) ).multiplyScalar( dist ).add( target );
		controls.target.copy( target );
		controls.minDistance = 2.4;
		controls.maxDistance = 10;
		controls.maxPolarAngle = THREE.MathUtils.degToRad( 86 );
		controls.autoRotate = true;
		controls.autoRotateSpeed = 0.35;
		controls.update();

		// ── GUI ──
		const gui = ctx.gui;
		gui.add( params, 'preset', Object.keys( PRESETS ) ).name( 'particles (rebuild)' ).onChange( ( v ) => build( v ) );
		gui.add( params, 'viscosity', 0, 1, 0.01 ).onChange( applyParams );
		gui.add( params, 'stiffness', 0.3, 2, 0.01 ).onChange( applyParams );
		gui.add( params, 'gravity', 0, 2, 0.01 ).onChange( applyParams );
		gui.add( params, 'tilt', - 35, 35, 0.5 ).name( 'gravity tilt °' );
		gui.add( params, 'tide', 0, 1.5, 0.01 ).name( 'tidal sway' );
		gui.add( params, 'glow', 0, 3, 0.01 ).name( 'bioluminescence' ).onChange( applyParams );
		gui.add( params, 'autoDrops' ).name( 'idle rain' );
		const actions = {
			dam: () => resetDam(),
			blob: () => dropRandom(),
		};
		gui.add( actions, 'dam' ).name( 'dam break  [R]' );
		gui.add( actions, 'blob' ).name( 'drop blob  [B]' );

		// ── interaction helpers ──
		let mirror = false;
		const resetDam = () => {

			mirror = ! mirror;
			sim.reset( mirror );
			lastInteract = clockT;

		};

		const tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3();
		const toGrid = ( wp, out ) => out.copy( wp ).sub( ORB_C ).multiplyScalar( sim.radius / ORB_R ).addScalar( sim.center );
		const dropAtGrid = ( gp ) => {

			const v = new THREE.Vector3( ( Math.random() - 0.5 ) * 0.04, - 0.12 * sim.radius / 29 * 3 / substeps, ( Math.random() - 0.5 ) * 0.04 );
			sim.dropBlob( gp, v );
			pulse = Math.min( 2.0, pulse + 1.0 );

		};

		const dropRandom = () => {

			const a = Math.random() * Math.PI * 2, r = Math.random() * 0.35 * sim.radius;
			tmp.set( sim.center + Math.cos( a ) * r, sim.center + 0.42 * sim.radius, sim.center + Math.sin( a ) * r );
			dropAtGrid( tmp );
			lastInteract = clockT - 2; // count as ambient, not as user interaction

		};

		// ray in grid space, intersect the container sphere
		const gRay = { o: new THREE.Vector3(), d: new THREE.Vector3(), t0: 0, t1: 0, hit: false };
		const castRay = () => {

			toGrid( pointer.ray.origin, gRay.o );
			gRay.d.copy( pointer.ray.direction ).normalize();
			const oc = tmp2.copy( gRay.o ).subScalar( sim.center );
			const b = oc.dot( gRay.d ), c = oc.lengthSq() - sim.radius * sim.radius;
			const h = b * b - c;
			gRay.hit = h > 0 && - b + Math.sqrt( h ) > 0;
			if ( gRay.hit ) {

				const s = Math.sqrt( h );
				gRay.t0 = Math.max( 0, - b - s ); gRay.t1 = - b + s;

			}

		};

		let prevDown = false, dragging = false, downT = 0, moved = 0, lastInteract = - 100, clockT = 0, nextDrop = 0.75, pulse = 0;
		let glowLevel = 0.6;
		const ptrVelW = new THREE.Vector3(), camRight = new THREE.Vector3(), camUp = new THREE.Vector3();
		let accum = 0, statT = 0;

		const onKey = ( e ) => {

			if ( e.key === 'r' || e.key === 'R' ) resetDam();
			if ( e.key === 'b' || e.key === 'B' ) dropRandom();

		};

		return {
			scene,
			camera,

			post( _ctx, _scenePass ) {

				const comp = convertToTexture( fluid.compositeNode() );
				const bl = bloom( comp, 0.75, 0.55, 0.35 );
				this._bloom = bl;
				return comp.add( bl );

			},

			update( dt, t ) {

				const fi = ( this._fi = ( this._fi || 0 ) + 1 );
				const mark = ( label ) => {

					if ( ! DBG ) return;
					renderer.backend.device?.queue.onSubmittedWorkDone().then( () => {

						const now = performance.now();
						console.log( `[ocean] f${fi} ${label} +${( now - dbgT ).toFixed( 0 )} ms` );
						dbgT = now;

					} );

				};

				mark( 'prev post/composite' );

				clockT = t;
				U.time.value = t;
				if ( ! sim ) return;

				// ── pointer ──
				castRay();
				const hover = pointer.inside && gRay.hit;
				if ( ! pointer.down ) {

					controls.enabled = ! hover;
					if ( prevDown && dragging && t - downT < 0.35 && moved < 0.03 ) {

						// click → drop a blob where the ray meets the upper half of the orb
						const tt = THREE.MathUtils.lerp( gRay.t0, gRay.t1, 0.3 );
						tmp.copy( gRay.d ).multiplyScalar( tt ).add( gRay.o );
						tmp.y = Math.max( tmp.y, sim.center + 0.3 * sim.radius );
						dropAtGrid( tmp );

					}

					dragging = false;

				} else if ( ! prevDown && hover ) {

					dragging = true; downT = t; moved = 0;

				}

				prevDown = pointer.down;

				if ( dragging ) {

					moved += pointer.velocity.length() * dt;
					// ndc velocity → world velocity on a plane through the orb centre
					const d = camera.position.distanceTo( ORB_C );
					const hh = d * Math.tan( THREE.MathUtils.degToRad( camera.fov ) * 0.5 );
					camRight.setFromMatrixColumn( camera.matrixWorld, 0 );
					camUp.setFromMatrixColumn( camera.matrixWorld, 1 );
					tmp.copy( camRight ).multiplyScalar( pointer.velocity.x * hh * camera.aspect ).addScaledVector( camUp, pointer.velocity.y * hh );
					ptrVelW.lerp( tmp, 0.5 );
					const toCells = ( sim.radius / ORB_R ) / stepsPerSec;
					U.ptrVel.value.copy( ptrVelW ).multiplyScalar( toCells * 1.4 ).clampLength( 0, 0.6 );
					U.ptrOrigin.value.copy( gRay.o );
					U.ptrDir.value.copy( gRay.d );
					U.ptrRange.value.set( gRay.t0, gRay.t1 );
					U.ptrStrength.value = gRay.hit ? Math.min( 1, ptrVelW.length() * 4 ) * 0.9 : 0;
					if ( ptrVelW.length() > 0.05 ) {

						lastInteract = t;
						pulse = Math.min( 2.0, pulse + dt * 2 );

					}

				} else {

					U.ptrStrength.value = 0;
					ptrVelW.multiplyScalar( 0.8 );

				}

				// ── gravity: user tilt + slow tidal sway (keeps the sea alive when idle) ──
				const tiltZ = THREE.MathUtils.degToRad( params.tilt ) + params.tide * 0.16 * Math.sin( t * 0.55 );
				const tiltX = params.tide * 0.12 * Math.sin( t * 0.37 + 1.3 );
				tmp.set( Math.sin( tiltZ ), - Math.cos( tiltZ ) * Math.cos( tiltX ), Math.sin( tiltX ) ).normalize().multiplyScalar( gMag );
				U.gravity.value.copy( tmp );

				// ── idle rain ──
				if ( params.autoDrops && t > nextDrop && t - lastInteract > 5 ) {

					dropRandom();
					nextDrop = t + 3.5 + Math.random() * 3;

				}

				// ── simulate ──
				accum += dt * stepsPerSec;
				let n = Math.floor( accum );
				accum -= n;
				n = Math.min( n, substeps * 2 );
				if ( ctx.shot ) n = substeps;
				if ( ! ctx.params.has( 'onosim' ) ) sim.step( n );
				mark( 'sim' );

				// bioluminescent ambient level (floor pool) follows activity
				pulse *= Math.exp( - dt * 0.8 );
				glowLevel = THREE.MathUtils.lerp( glowLevel, 0.45 + 0.5 * pulse, 1 - Math.exp( - dt * 3 ) );
				U.pool.value = params.glow * glowLevel;

				// ── fluid passes into our own targets (composite happens in the post chain) ──
				fluid.render();
				mark( 'fluid passes' );

				statT += dt;
				if ( statsDirty || statT > 0.5 ) {

					statT = 0; statsDirty = false;
					ctx.setStats( [
						[ 'particles', sim.count.toLocaleString() ],
						[ 'grid', `${sim.grid}³` ],
						[ 'substeps', `${substeps} × 5 kernels` ],
					] );

				}

			},

			resize() {

				syncSize();

			},

			onKey,

			dispose() {

				sim?.dispose();
				sim = null;
				fluid.dispose();
				envRT.dispose();
				this._bloom?.dispose?.();
				controls.enabled = true;

			},
		};

	},
};
