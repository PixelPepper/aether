// Chapter 01 — Emergence.
// 3D particle life: N particles of up to 8 species, each pair of species bound by an asymmetric attraction rule.
// Neighbour search is O(N) on the GPU: a uniform grid is rebuilt every frame with atomic counting, a multi-level
// workgroup prefix sum and an atomic-free scatter (see genesis/sim.js and genesis/scan.js).
import * as THREE from 'three/webgpu';
import { ParticleLife } from './genesis/sim.js';
import { createParticleMaterial, createDust, createBackground, createLure } from './genesis/render.js';
import { PRESETS, PALETTES, paletteColors, presetMatrix, randomMatrix, MAX_SPECIES } from './genesis/rules.js';

const COUNTS = [ 16384, 65536, 262144 ];
const MEAN_NEIGHBOURS = 30; // average neighbours inside rMax at uniform density (sets the size of the world)

export default {
	id: 'genesis',
	title: 'Emergence',
	subtitle: 'Matter learning to be alive: a quarter-million particles obeying one asymmetric table of likes and dislikes — and nothing else.',

	async create( ctx ) {

		const { renderer, camera, controls, tier, params } = ctx;
		const N = Math.max( 1024, parseInt( params.get( 'n' ) || '' ) || COUNTS[ Math.min( Math.max( tier, 0 ), 2 ) ] );
		const rMax0 = 1.0;
		const bound = rMax0 * Math.cbrt( N / MEAN_NEIGHBOURS );

		const sim = new ParticleLife( renderer, { count: N, bound, rMax: rMax0, rMaxMin: rMax0 * 0.6 } );
		const su = sim.u;

		// ── state / presets ──
		const state = {
			preset: PRESETS[ 0 ].name,
			species: PRESETS[ 0 ].species,
			radius: 1.0,
			friction: 0.04,
			force: 1.0,
			timeScale: 1.0,
			palette: PRESETS[ 0 ].palette,
			seed: PRESETS[ 0 ].seed,
		};

		const pal = paletteColors( state.palette );
		const { material, uniforms: ru } = createParticleMaterial( sim, pal );
		ru.size.value = rMax0 * 0.17;
		ru.focus.value = bound * 2.7;
		ru.dof.value = 0.9 / bound;

		const applyPalette = ( name ) => {

			const p = paletteColors( name );
			ru.core.array.forEach( ( c, i ) => c.setRGB( ...p.core[ i ] ) );
			ru.rim.array.forEach( ( c, i ) => c.setRGB( ...p.rim[ i ] ) );

		};

		const applyRules = ( preset, seed ) => {

			sim.setMatrix( presetMatrix( preset, seed ) );

		};

		const urlSeed = params.has( 'seed' ) ? parseInt( params.get( 'seed' ) ) : null;
		const urlPreset = params.has( 'preset' ) ? PRESETS[ parseInt( params.get( 'preset' ) ) % PRESETS.length ] : null;

		const applyPreset = ( p, { scatter = false } = {} ) => {

			state.preset = p.name;
			state.species = p.species;
			state.radius = p.rMax;
			state.friction = p.friction;
			state.force = p.force;
			state.palette = p.palette;
			state.seed = p.seed;
			su.species.value = p.species;
			su.rMax.value = rMax0 * p.rMax;
			sim.setGrid();
			sim.setFrictionHalfLife( p.friction );
			su.force.value = 10 * p.force;
			applyPalette( p.palette );
			applyRules( p, p.seed );
			if ( scatter ) sim.reset( p.seed + 11 );
			else sim.reassignSpecies();
			gui.controllersRecursive().forEach( ( c ) => c.updateDisplay() );

		};

		// ── scene ──
		const scene = new THREE.Scene();
		scene.backgroundNode = createBackground();

		const sprites = new THREE.Sprite( material );
		sprites.geometry = new THREE.PlaneGeometry( 1, 1 );
		sprites.count = N;
		sprites.frustumCulled = false;
		scene.add( sprites );

		const dust = createDust( renderer, bound, tier === 0 ? 1500 : tier === 1 ? 4000 : 9000 );
		scene.add( dust.object );

		const lure = createLure();
		lure.uniforms.size.value = bound * 0.22;
		scene.add( lure.object );

		// ── camera ──
		camera.fov = 38;
		camera.near = 0.05;
		camera.far = bound * 40;
		camera.updateProjectionMatrix();
		camera.position.set( bound * 1.05, bound * 0.62, bound * 2.45 );
		controls.target.set( 0, 0, 0 );
		controls.minDistance = bound * 0.6;
		controls.maxDistance = bound * 6;
		controls.autoRotate = true;
		controls.autoRotateSpeed = 0.28;
		controls.enablePan = false;
		controls.update();

		// Left = attract, right = repel/shockwave. Orbit with middle drag, or Shift/Alt + left drag.
		const savedButtons = { ...controls.mouseButtons };
		const savedTouches = { ...controls.touches };
		const savedPan = true;
		controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.ROTATE, RIGHT: null };
		controls.touches = { ONE: null, TWO: THREE.TOUCH.DOLLY_ROTATE };
		let orbitDrag = false;
		const onPointerDownCapture = ( e ) => {

			orbitDrag = e.button === 0 && ( e.shiftKey || e.altKey || e.ctrlKey || e.metaKey );
			// OrbitControls swaps ROTATE↔PAN when shift/ctrl/meta is held, so map accordingly.
			controls.mouseButtons.LEFT = ! orbitDrag ? null : ( e.altKey && ! e.shiftKey && ! e.ctrlKey && ! e.metaKey ) ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN;
			controls.enablePan = orbitDrag;

		};

		addEventListener( 'pointerdown', onPointerDownCapture, { capture: true } );

		// ── GUI ──
		const gui = ctx.gui;
		const actions = {
			reseed: () => {

				state.seed = ( Math.random() * 1e9 ) >>> 0;
				sim.setMatrix( randomMatrix( state.seed ) );
				state.preset = '(random)';
				gui.controllersRecursive().forEach( ( c ) => c.updateDisplay() );

			},
			soup: () => sim.reset( ( Math.random() * 1000 ) | 0 ),
		};
		gui.add( state, 'preset', PRESETS.map( ( p ) => p.name ) ).name( 'ecosystem' ).onChange( ( name ) => {

			const p = PRESETS.find( ( q ) => q.name === name );
			if ( p ) applyPreset( p, { scatter: true } );

		} );
		gui.add( actions, 'reseed' ).name( '⟳ reseed rules' );
		gui.add( actions, 'soup' ).name( '◌ primordial soup' );
		gui.add( state, 'species', 3, MAX_SPECIES, 1 ).name( 'species' ).onChange( ( v ) => {

			su.species.value = v; sim.reassignSpecies();

		} );
		gui.add( state, 'radius', 0.6, 1.6, 0.01 ).name( 'interaction radius' ).onChange( ( v ) => {

			su.rMax.value = rMax0 * v; sim.setGrid();

		} );
		gui.add( state, 'friction', 0.01, 0.25, 0.005 ).name( 'friction half-life' ).onChange( ( v ) => sim.setFrictionHalfLife( v ) );
		gui.add( state, 'force', 0.1, 3, 0.01 ).name( 'force scale' ).onChange( ( v ) => su.force.value = 10 * v );
		gui.add( state, 'timeScale', 0, 2, 0.01 ).name( 'time scale' );
		gui.add( state, 'palette', Object.keys( PALETTES ) ).name( 'palette' ).onChange( applyPalette );
		gui.add( ru.bright, 'value', 0.2, 3, 0.01 ).name( 'luminance' );

		// ── initial world ──
		applyPreset( urlPreset || PRESETS[ 0 ], { scatter: false } );
		if ( urlSeed !== null ) {

			state.seed = urlSeed; sim.setMatrix( randomMatrix( urlSeed ) ); state.preset = '(random)';

		}

		sim.reset( 1 );

		// Let the soup pre-organise a little before the first frame (cheap on real GPUs).
		const warm = params.has( 'warm' ) ? parseInt( params.get( 'warm' ) ) : [ ctx.shot ? 50 : 0, 90, 150 ][ Math.min( tier, 2 ) ];
		for ( let i = 0; i < warm; i ++ ) sim.step( 0.02 );
		await renderer.computeAsync( sim.forceKernel ); // flush; also guarantees pipelines are compiled

		// ── pointer state ──
		const ptr = new THREE.Vector3(), right = new THREE.Vector3(), up = new THREE.Vector3(), fwd = new THREE.Vector3();
		const ptrVel = new THREE.Vector3();
		let attract = 0, repel = 0, wasDown = false, statT = 0, lureAmt = 0;
		const ATTRACT = 140, REPEL = 260, CURRENT = 5;
		su.ptrRadius.value = bound * 0.32;

		return {
			scene,
			bloom: { strength: 0.85, radius: 0.55, threshold: 0.08 },

			update( dt, t, ctx ) {

				const p = ctx.pointer;
				const ray = p.ray;
				const cam = camera;

				// pointer → the point on the ray closest to the colony's centre
				const tC = Math.max( 0.0, - ray.origin.dot( ray.direction ) );
				ptr.copy( ray.direction ).multiplyScalar( tC ).add( ray.origin );
				cam.getWorldDirection( fwd );
				right.setFromMatrixColumn( cam.matrixWorld, 0 );
				up.setFromMatrixColumn( cam.matrixWorld, 1 );
				const halfH = Math.tan( THREE.MathUtils.degToRad( cam.fov * 0.5 ) ) * Math.max( tC, 1 );
				const halfW = halfH * cam.aspect;
				const target = right.multiplyScalar( p.velocity.x * halfW ).add( up.multiplyScalar( p.velocity.y * halfH ) );
				if ( target.length() > bound * 6 ) target.setLength( bound * 6 );
				ptrVel.lerp( target, 0.25 );

				const pressing = p.down && ! orbitDrag;
				const wantA = pressing && p.button === 0 ? ATTRACT : 0;
				const wantR = pressing && p.button === 2 ? REPEL : 0;
				attract += ( wantA - attract ) * Math.min( 1, dt * 6 );
				repel += ( wantR - repel ) * Math.min( 1, dt * 10 );

				if ( pressing && p.button === 2 && ! wasDown ) {

					su.shock.value.set( ptr.x, ptr.y, ptr.z, t );
					su.shockAmp.value = 420;

				}

				wasDown = pressing;

				su.time.value = t;
				su.ptr.value.copy( ptr );
				su.ptrVel.value.copy( ptrVel );
				su.rayO.value.copy( ray.origin );
				su.rayD.value.copy( ray.direction );
				su.viewDir.value.copy( fwd );
				su.attract.value = attract;
				su.repel.value = repel;
				su.current.value = ( p.inside && ! pressing ) ? CURRENT : 0;

				lureAmt += ( ( pressing ? 1 : 0 ) - lureAmt ) * Math.min( 1, dt * 8 );
				lure.uniforms.pos.value.copy( ptr );
				lure.uniforms.amount.value = lureAmt * 0.6;
				lure.uniforms.color.value.setRGB( ...( p.button === 2 ? [ 0.35, 0.75, 1.0 ] : [ 1.0, 0.62, 0.3 ] ) );
				lure.object.visible = lureAmt > 0.01;

				ru.focus.value = cam.position.length();

				const simDt = Math.min( Math.max( dt, 1 / 240 ), 1 / 30 ) * 1.2 * state.timeScale;
				if ( simDt > 0 ) sim.step( simDt );

				statT -= dt;
				if ( statT <= 0 ) {

					statT = 0.5;
					ctx.setStats( [
						[ 'particles', N.toLocaleString() ],
						[ 'species', String( su.species.value ) ],
						[ 'grid', `${sim.dim}³ = ${( sim.dim ** 3 ).toLocaleString()} cells` ],
						[ 'scan', `${sim.scan.levels}-level, ${sim.scan.passes.length} passes` ],
					] );

				}

			},

			onKey( e ) {

				if ( e.key === 'r' || e.key === 'R' ) actions.reseed();
				if ( e.key === 'b' || e.key === 'B' ) actions.soup();
				if ( e.key === 'p' || e.key === 'P' ) {

					const i = ( PRESETS.findIndex( ( q ) => q.name === state.preset ) + 1 ) % PRESETS.length;
					applyPreset( PRESETS[ i ], { scatter: true } );

				}

			},

			dispose() {

				removeEventListener( 'pointerdown', onPointerDownCapture, { capture: true } );
				controls.mouseButtons = savedButtons;
				controls.touches = savedTouches;
				controls.enablePan = savedPan;
				scene.remove( sprites, dust.object, lure.object );
				sprites.geometry.dispose();
				material.dispose();
				dust.dispose();
				lure.dispose();
				sim.dispose();
				scene.backgroundNode = null;

			},
		};

	},
};
