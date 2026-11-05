// Chapter 09 — Morphogenesis.
// Turing's reaction–diffusion in three dimensions: a 3D Gray–Scott system integrated on the GPU (shared-memory tiled
// 19-point Laplacian, ping-pong storage buffers), baked into a 3D texture and raymarched as an isosurface with
// subsurface transmission, field AO, thin-film iridescence and a studio floor reflection. See morpho/sim.js and
// morpho/render.js for the kernels and the shading.
import * as THREE from 'three/webgpu';
import { createSim } from './morpho/sim.js';
import { createRaymarch, MATERIALS } from './morpho/render.js';

// Known Gray–Scott (F, k) pairs and a direction in parameter space along which the volume's gradient varies them.
const PRESETS = {
	spectrum: { feed: 0.037, kill: 0.061, dFeed: 0.0, dKill: 0.0042, spread: 1.0 }, // spots ↔ worms ↔ maze ↔ holes
	coral: { feed: 0.0545, kill: 0.062, dFeed: 0.002, dKill: 0.0006, spread: 1.0 },
	mitosis: { feed: 0.0367, kill: 0.0649, dFeed: 0.0, dKill: 0.0008, spread: 1.0 },
	labyrinth: { feed: 0.029, kill: 0.057, dFeed: 0.001, dKill: 0.0005, spread: 1.0 },
	gyroid: { feed: 0.035, kill: 0.058, dFeed: 0.0, dKill: 0.0006, spread: 1.0 },
};

const TIERS = {
	0: { G: 48, iters: 4, steps: 110, refl: 40, stepVox: 0.8, du: 0.06, prerun: 320 },
	1: { G: 96, iters: 8, steps: 170, refl: 64, stepVox: 0.75, du: 0.085, prerun: 1400 },
	2: { G: 128, iters: 14, steps: 230, refl: 90, stepVox: 0.7, du: 0.1, prerun: 2000 },
};

export default {
	id: 'morpho',
	title: 'Morphogenesis',
	subtitle: 'Turing’s reaction–diffusion grown in three dimensions: two chemicals feeding, killing and diffusing on the GPU until coral, brain folds, sponges and dividing cells precipitate out of the chemistry.',

	async create( ctx ) {

		const { renderer, camera, controls, pointer, params } = ctx;
		const tierCfg = { ...( TIERS[ ctx.tier ] || TIERS[ 2 ] ) };
		if ( params.has( 'grid' ) ) tierCfg.G = Math.max( 32, Math.round( parseInt( params.get( 'grid' ) ) / 8 ) * 8 );
		const G = tierCfg.G;

		const DBG = params.has( 'mdebug' );
		const t00 = performance.now();
		const gpuIdle = () => renderer.backend.device?.queue.onSubmittedWorkDone();
		const log = async ( msg, wait = true ) => {

			if ( ! DBG ) return;
			if ( wait ) await gpuIdle();
			console.log( `[morpho] ${msg} @ ${( ( performance.now() - t00 ) / 1000 ).toFixed( 2 )}s` );

		};

		// ── simulation ──
		const sim = createSim( renderer, G );
		const su = sim.u;
		// Software adapters (SwiftShader) emulate workgroupBarrier with fibres, making the shared-memory tiled kernel
		// ~30× slower than direct reads; there (only) the untiled twin of the same update is used. ?tiled=1 forces tiling.
		const tiled = params.has( 'tiled' ) ? params.get( 'tiled' ) !== '0' : ! ctx.isSoftware;
		const stepAB = tiled ? sim.stepAB : sim.stepNaiveAB;
		const stepBA = tiled ? sim.stepBA : sim.stepNaiveBA;
		su.du.value = tierCfg.du;

		const state = {
			preset: PRESETS[ params.get( 'preset' ) ] ? params.get( 'preset' ) : 'spectrum',
			material: MATERIALS[ params.get( 'material' ) ] ? params.get( 'material' ) : 'nacre',
			speed: 1,
			drift: true,
			reseed: () => reseed( true ),
		};

		const applyPreset = ( name ) => {

			const p = PRESETS[ name ];
			su.feed.value = p.feed; su.kill.value = p.kill;
			su.dFeed.value = p.dFeed; su.dKill.value = p.dKill;
			su.spread.value = p.spread;
			guiCtl.forEach( ( c ) => c.updateDisplay() );

		};

		const guiCtl = [];
		applyPreset( state.preset );

		let salt = 1;
		const reseed = ( clear ) => {

			su.seedSalt.value = ( salt ++ ) * 1.618;
			su.seedClear.value = clear ? 1 : 0;
			su.seedDensity.value = clear ? 1.0 : 0.25;
			renderer.compute( sim.seedKernel );

		};

		// ── renderer side ──
		const S = 1.0;
		const rm = createRaymarch( sim, {
			halfSize: S, maxSteps: tierCfg.steps, reflSteps: tierCfg.refl, stepVox: tierCfg.stepVox,
			floorY: - 0.97, sphereR: 0.82,
		} );
		const ru = rm.u;

		const scene = new THREE.Scene();
		scene.add( rm.mesh );

		// material crossfade targets
		const matKeys = [ 'sssAmt', 'rough', 'irid', 'gold' ];
		const matTarget = {};
		const setMaterialTarget = ( name ) => {

			const m = MATERIALS[ name ];
			Object.assign( matTarget, {
				base: new THREE.Color( ...m.base ), sss: new THREE.Color( ...m.sss ), glow: new THREE.Color( ...m.glow ),
				sssAmt: m.sssAmt, rough: m.rough, irid: m.irid, gold: m.gold, filmMin: m.film[ 0 ], filmMax: m.film[ 1 ],
			} );

		};

		setMaterialTarget( state.material );
		const snapMaterial = () => {

			ru.base.value.copy( matTarget.base ); ru.sss.value.copy( matTarget.sss ); ru.glow.value.copy( matTarget.glow );
			for ( const k of [ ...matKeys, 'filmMin', 'filmMax' ] ) ru[ k ].value = matTarget[ k ];

		};

		snapMaterial();

		// ── init: seed and pre-grow so the sculpture is already structured on the first frame ──
		reseed( true );
		await log( 'seeded' );
		const warp = parseInt( params.get( 'warp' ) || '0' );
		const pre = Math.max( 0, ( params.has( 'prerun' ) ? parseInt( params.get( 'prerun' ) ) : tierCfg.prerun ) + warp );
		const CHUNK = 40;
		for ( let i = 0; i < pre; i += CHUNK * 2 ) {

			const list = [];
			for ( let j = 0; j < CHUNK && i + j * 2 < pre; j ++ ) list.push( stepAB, stepBA );
			renderer.compute( list );
			if ( i === 0 ) await log( `first ${list.length} iterations` );
			if ( ( i / ( CHUNK * 2 ) ) % 4 === 3 ) await new Promise( ( r ) => setTimeout( r, 0 ) );

		}

		await log( `pre-run ${pre} iterations` );
		// prime the growth-glow history so the pre-grown structure does not flash on the first frame
		const gain = su.glowGain.value;
		su.glowGain.value = 0;
		renderer.compute( [ sim.bake, sim.occKernel, sim.distKernel ] );
		await gpuIdle();
		su.glowGain.value = gain;
		await log( 'baked' );
		let frameNo = 0;

		// ── camera: slow museum turntable ──
		camera.position.set( 2.75, 0.85, 2.55 );
		controls.target.set( 0, - 0.1, 0 );
		controls.minDistance = 1.6;
		controls.maxDistance = 7;
		controls.maxPolarAngle = Math.PI * 0.53;
		controls.autoRotate = true;
		controls.autoRotateSpeed = 0.55;
		controls.update();
		const savedButtons = { ...controls.mouseButtons };
		const savedTouches = { ...controls.touches };

		// ── GUI ──
		const gui = ctx.gui;
		gui.add( state, 'preset', Object.keys( PRESETS ) ).name( 'morphology' ).onChange( ( v ) => {

			applyPreset( v );
			reseed( false ); // sprinkle fresh activator so the new chemistry has nuclei everywhere

		} );
		guiCtl.push( gui.add( su.feed, 'value', 0.01, 0.09, 0.0001 ).name( 'feed F' ) );
		guiCtl.push( gui.add( su.kill, 'value', 0.045, 0.07, 0.0001 ).name( 'kill k' ) );
		guiCtl.push( gui.add( su.ratio, 'value', 0.3, 0.7, 0.01 ).name( 'diffusion ratio' ) );
		guiCtl.push( gui.add( su.spread, 'value', 0, 2, 0.01 ).name( 'variation' ) );
		gui.add( state, 'speed', 0, 3, 0.05 ).name( 'sim speed' );
		gui.add( state, 'material', Object.keys( MATERIALS ) ).name( 'material' ).onChange( ( v ) => setMaterialTarget( v ) );
		gui.add( state, 'drift' ).name( 'parameter drift' );
		gui.add( state, 'reseed' ).name( 'reseed (R)' );

		// ── pointer: hover the sculpture → left drag seeds; elsewhere → orbit ──
		const sphere = new THREE.Sphere( new THREE.Vector3( 0, 0, 0 ), 0.9 );
		const tmpV = new THREE.Vector3();
		let dragSeed = false, wasDown = false, hoverS = 0, pressS = 0;
		const poke = params.has( 'poke' ); // headless test: seed through the screen centre during capture
		const pokeRay = new THREE.Ray();
		const raycaster = new THREE.Raycaster();

		let acc = 0, statsT = 0, lastIters = 0;
		let driftA = 0.0;
		const gradDir = new THREE.Vector3();

		const instance = {
			scene,
			bloom: { strength: 0.32, radius: 0.55, threshold: 0.55 },

			update( dt, t ) {

				// pointer / seeding
				let ray = pointer.ray;
				let inside = pointer.inside;
				let down = pointer.down && pointer.button === 0;
				if ( poke ) {

					raycaster.setFromCamera( new THREE.Vector2( 0.08, 0.12 ), camera );
					pokeRay.copy( raycaster.ray );
					ray = pokeRay; inside = true; down = true;

				}

				const hovering = inside && ray.intersectSphere( sphere, tmpV ) !== null;
				if ( down && ! wasDown ) dragSeed = hovering;
				if ( ! down ) dragSeed = false;
				wasDown = down;
				controls.enableRotate = ! ( hovering || dragSeed );
				controls.autoRotate = ! dragSeed;

				hoverS += ( ( hovering ? 1 : 0 ) - hoverS ) * Math.min( 1, dt * 8 );
				pressS += ( ( dragSeed ? 1 : 0 ) - pressS ) * Math.min( 1, dt * 10 );
				ru.hover.value = hoverS;
				ru.press.value = pressS;

				const list = [];
				if ( inside ) {

					// pointer ray in grid coordinates
					su.rayO.value.copy( ray.origin ).addScalar( S ).divideScalar( rm.vox );
					su.rayD.value.copy( ray.direction ).normalize();
					list.push( sim.pickKernel );

				}

				if ( dragSeed ) {

					su.seedSalt.value = ( salt ++ ) * 1.618;
					list.push( sim.brushKernel );

				}

				// parameter drift: the feed/kill gradient axis slowly precesses through the sculpture
				if ( state.drift ) driftA += dt * 0.05;
				gradDir.set( Math.sin( driftA ) * 0.55, Math.cos( driftA * 0.7 ), Math.cos( driftA ) * 0.35 ).normalize();
				su.gradDir.value.copy( gradDir );
				su.shapeT.value = t * 0.05;

				// iterations (always in A→B→A pairs so the result lives in buffer A)
				acc += ( tierCfg.iters / 2 ) * state.speed;
				const pairs = Math.floor( acc );
				acc -= pairs;
				for ( let i = 0; i < pairs; i ++ ) list.push( stepAB, stepBA );
				lastIters = pairs * 2;

				su.fade.value = Math.exp( - dt / 0.6 );
				list.push( sim.bake, sim.occKernel, sim.distKernel );
				renderer.compute( list );
				if ( DBG && frameNo < 4 ) log( `frame ${frameNo} compute (${list.length} dispatches)` );
				frameNo ++;

				// material crossfade
				const k = Math.min( 1, dt * 3 );
				ru.base.value.lerp( matTarget.base, k ); ru.sss.value.lerp( matTarget.sss, k ); ru.glow.value.lerp( matTarget.glow, k );
				for ( const key of [ ...matKeys, 'filmMin', 'filmMax' ] ) ru[ key ].value += ( matTarget[ key ] - ru[ key ].value ) * k;
				ru.time.value = t;

				statsT -= dt;
				if ( statsT <= 0 ) {

					statsT = 0.5;
					ctx.setStats( [
						[ 'grid', `${G}³ (${( sim.N / 1e6 ).toFixed( 2 )}M cells)` ],
						[ 'iterations/frame', String( lastIters ) ],
						[ 'F', su.feed.value.toFixed( 4 ) ],
						[ 'k', su.kill.value.toFixed( 4 ) ],
					] );

				}

			},

			onKey( e ) {

				if ( e.key === 'r' || e.key === 'R' ) reseed( true );

			},

			dispose() {

				controls.enableRotate = true;
				Object.assign( controls.mouseButtons, savedButtons );
				Object.assign( controls.touches, savedTouches );
				sim.dispose( renderer );
				rm.dispose();
				scene.remove( rm.mesh );

			},
		};

		return instance;

	},
};
