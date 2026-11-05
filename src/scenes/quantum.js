// Chapter 01 — Fluctuation.
// A real-time solver of the time-dependent Schrödinger equation: split-step Fourier on an N×N complex grid, with the
// 2D FFT written as workgroup-shared-memory radix-2 compute kernels (see quantum/solver.js). The wavefunction is drawn as
// a |ψ|² height field in domain colouring (hue = arg ψ), a Bohmian mist rides the probability current, and a detector
// screen turns the probability absorbed at the far wall into individual particle hits (Born rule), one dot at a time.
import * as THREE from 'three/webgpu';
import { SchrodingerSolver } from './quantum/solver.js';
import { SCENARIOS, buildPotential, defaultPacket, hoSigma, STADIUM, HO_OMEGA } from './quantum/scenarios.js';
import {
	WORLD, buildStudioEnv, backgroundNode, makeLook, createSurface, createFloor, createMonolithMaterial, createDetector, createMist,
	createPointerGizmo,
} from './quantum/render.js';

const TIERS = [
	{ N: 128, steps: 2, segs: 128, mist: 4096, hits: 8192, k: 140, kMax: 200 },
	{ N: 256, steps: 3, segs: 256, mist: 16384, hits: 16384, k: 170, kMax: 280 },
	{ N: 512, steps: 5, segs: 512, mist: 65536, hits: 32768, k: 190, kMax: 320 },
];

const TAU_FRAME = 2.0e-5; // simulated time per 60 Hz frame at time-scale 1 (ħ = m = 1, unit box)
const COLOUR_MODES = [ 'phase · domain colouring', 'probability |ψ|² only' ];
const POINTER_MODES = [ 'fire packet (drag = aim)', 'paint walls' ];
const MAX_SEGMENTS = 400;

export default {
	id: 'quantum',
	title: 'Fluctuation',
	subtitle: 'One particle, many paths: the Schrödinger equation solved live by GPU Fourier transforms — the colour is the phase of ψ, and every dot on the screen is one detection drawn from |ψ|².',

	async create( ctx ) {

		const { renderer, camera, controls, tier } = ctx;
		const cfg = TIERS[ Math.max( 0, Math.min( 2, tier ) ) ];
		const N = cfg.N;
		const disposables = [];
		let disposed = false;

		// ───────────────────────── solver ─────────────────────────
		const solver = new SchrodingerSolver( renderer, { N, hits: cfg.hits, mist: cfg.mist } );
		const rhoRef = 1 / ( 2 * Math.PI * 0.05 * 0.05 );
		solver.u.rhoRef.value = rhoRef;

		const params = {
			scenario: 'double slit',
			k: cfg.k,
			sigma: 0.05,
			slitSep: 0.15,
			slitWidth: 0.034,
			timeScale: 1,
			colour: COLOUR_MODES[ 0 ],
			mist: true,
			pointer: POINTER_MODES[ 0 ],
			kDef: cfg.k,
		};

		const setDt = () => solver.setTimeStep( TAU_FRAME * params.timeScale / cfg.steps );
		setDt();

		// ───────────────────────── scene ─────────────────────────
		const scene = new THREE.Scene();
		const envRT = buildStudioEnv( renderer );
		scene.environment = envRT.texture;
		scene.environmentIntensity = 0.9;
		scene.backgroundNode = backgroundNode();

		const look = makeLook();
		const surface = createSurface( solver, look, cfg.segs, rhoRef );
		scene.add( surface );
		const floor = createFloor();
		scene.add( floor );

		const key = new THREE.DirectionalLight( 0xdfeaff, 1.4 );
		key.position.set( 6, 7, - 5 );
		scene.add( key );

		const wallMat = createMonolithMaterial( solver, look, rhoRef );
		const barrierMat = createMonolithMaterial( solver, look, rhoRef, { opacity: 0.62, tint: 1.6 } );
		const monoliths = new THREE.Group();
		scene.add( monoliths );

		const paintGeo = new THREE.BoxGeometry( 1, 1, 1 );
		const painted = new THREE.InstancedMesh( paintGeo, wallMat, MAX_SEGMENTS );
		painted.count = 0;
		painted.frustumCulled = false;
		scene.add( painted );

		const detector = createDetector( solver, look );
		scene.add( detector );
		const mist = createMist( solver, look, rhoRef );
		scene.add( mist );
		const gizmo = createPointerGizmo();
		scene.add( gizmo );

		// ───────────────────────── scenarios ─────────────────────────
		let scn = 'slit';
		let launchedAt = 0;
		let clock = 0;
		let lastNorm = 1;
		let hitCount = 0;

		function clearMonoliths() {

			for ( const m of [ ...monoliths.children ] ) {

				m.geometry.dispose();
				monoliths.remove( m );

			}

			painted.count = 0;

		}

		function buildMonoliths( boxes ) {

			clearMonoliths();
			for ( const b of boxes ) {

				const w = ( b.x1 - b.x0 ) * WORLD, d = ( b.y1 - b.y0 ) * WORLD;
				if ( d <= 1e-4 ) continue;
				const geo = new THREE.BoxGeometry( w, b.h, d );
				const mesh = new THREE.Mesh( geo, b.kind === 'barrier' ? barrierMat : wallMat );
				mesh.position.set( ( ( b.x0 + b.x1 ) / 2 - 0.5 ) * WORLD, b.h / 2, ( ( b.y0 + b.y1 ) / 2 - 0.5 ) * WORLD );
				mesh.renderOrder = b.kind === 'barrier' ? 1 : 0;
				monoliths.add( mesh );

			}

			if ( scn === 'stadium' ) {

				const { cx, cy, a, r, wall } = STADIUM;
				const path = ( p, rr, hole ) => {

					// stadium outline in shape space (x → world x, y → −world z)
					const X = ( x ) => ( x - 0.5 ) * WORLD, Y = ( y ) => - ( y - 0.5 ) * WORLD;
					const seg = 48;
					const pts = [];
					for ( let i = 0; i <= seg; i ++ ) {

						const t = - Math.PI / 2 + Math.PI * i / seg;
						pts.push( new THREE.Vector2( X( cx + a + rr * Math.cos( t ) ), Y( cy + rr * Math.sin( t ) ) ) );

					}

					for ( let i = 0; i <= seg; i ++ ) {

						const t = Math.PI / 2 + Math.PI * i / seg;
						pts.push( new THREE.Vector2( X( cx - a + rr * Math.cos( t ) ), Y( cy + rr * Math.sin( t ) ) ) );

					}

					if ( hole ) pts.reverse();
					p.setFromPoints( pts );
					return p;

				};

				const shape = path( new THREE.Shape(), r + wall, false );
				shape.holes.push( path( new THREE.Path(), r, true ) );
				const geo = new THREE.ExtrudeGeometry( shape, { depth: 0.7, bevelEnabled: true, bevelThickness: 0.02, bevelSize: 0.02, bevelSegments: 2, curveSegments: 8 } );
				geo.rotateX( - Math.PI / 2 );
				monoliths.add( new THREE.Mesh( geo, wallMat ) );

			}

		}

		function rebuildPotential() {

			const { boxes } = buildPotential( scn, N, solver.potArray, params );
			solver.uploadPotential();
			buildMonoliths( boxes );

		}

		function launch( pk ) {

			solver.launch( pk.x, pk.y, pk.kx, pk.ky, pk.sigma );
			launchedAt = clock;
			lastNorm = 1;

		}

		function launchDefault() {

			launch( defaultPacket( scn, params ) );

		}

		function applyScenario( name ) {

			scn = SCENARIOS[ name ];
			params.sigma = scn === 'ho' ? hoSigma() : 0.05;
			sigmaCtl?.updateDisplay();
			rebuildPotential();
			solver.resetHits();
			hitCount = 0;
			launchDefault();
			look.screenFade.value = ( scn === 'slit' || scn === 'tunnel' ) ? 1 : 0;
			detector.visible = look.screenFade.value > 0;

		}

		// ───────────────────────── frame stepping ─────────────────────────
		let frameNo = 0;
		function frameList( steps, withMist ) {

			const l = solver.stepList( steps ).slice();
			l.push( solver.detect );
			l.push( solver.derive );
			if ( withMist ) l.push( solver.mistStep );
			return l;

		}

		// fast-forward without rendering (batched into a few compute passes)
		function simulate( frames ) {

			solver.u.frameTau.value = cfg.steps * solver.dt;
			const per = 24;
			const one = [ ...solver.stepList( cfg.steps ), solver.detect ];
			while ( frames > 0 ) {

				const n = Math.min( per, frames );
				const l = [];
				for ( let i = 0; i < n; i ++ ) l.push( ...one );
				renderer.compute( l );
				frames -= n;

			}

		}

		// ───────────────────────── GUI ─────────────────────────
		const gui = ctx.gui;
		gui.add( params, 'scenario', Object.keys( SCENARIOS ) ).name( 'scenario' ).onChange( applyScenario );
		gui.add( params, 'k', 60, cfg.kMax, 1 ).name( 'packet momentum k' ).onFinishChange( () => {

			if ( scn === 'tunnel' ) rebuildPotential();
			launchDefault();

		} );
		const sigmaCtl = gui.add( params, 'sigma', 0.015, 0.1, 0.001 ).name( 'packet width σ' ).onFinishChange( launchDefault );
		gui.add( params, 'slitSep', 0.06, 0.3, 0.005 ).name( 'slit separation' ).onFinishChange( () => {

			if ( scn === 'slit' ) {

				rebuildPotential();
				solver.resetHits();
				hitCount = 0;
				launchDefault();

			}

		} );
		gui.add( params, 'timeScale', 0.2, 2.5, 0.01 ).name( 'ħ/m · time scale' ).onChange( setDt );
		gui.add( params, 'colour', COLOUR_MODES ).name( 'colouring' ).onChange( ( v ) => look.mode.value = COLOUR_MODES.indexOf( v ) );
		gui.add( params, 'mist' ).name( 'Bohmian mist' ).onChange( ( v ) => {

			look.mistGain.value = v ? 1 : 0;
			solver.u.mistOn.value = v ? 1 : 0;

		} );
		gui.add( params, 'pointer', POINTER_MODES ).name( 'pointer (shift = walls)' );
		gui.add( {
			reset: () => {

				applyScenario( Object.keys( SCENARIOS ).find( ( k ) => SCENARIOS[ k ] === scn ) );

			},
		}, 'reset' ).name( 'reset experiment' );

		// ───────────────────────── initial state ─────────────────────────
		applyScenario( params.scenario );
		renderer.compute( solver.clearHits );
		// already in flight on frame 1; in capture mode let the experiment run a while so the still shows the build-up
		if ( ctx.shot ) {

			solver.u.time.value = - 100;
			const per = parseInt( ctx.params.get( 'qpre' ) || '330' );
			const runs = parseInt( ctx.params.get( 'qruns' ) || '2' );
			for ( let r = 0; r < runs; r ++ ) {

				simulate( per );
				launchDefault();

			}

			simulate( parseInt( ctx.params.get( 'qlast' ) || '150' ) );

		} else {

			simulate( 20 );

		}

		renderer.compute( solver.derive );

		// ───────────────────────── camera ─────────────────────────
		camera.fov = 36;
		camera.updateProjectionMatrix();
		camera.position.set( - 11.2, 4.6, 8.4 );
		controls.target.set( 0.9, 0.35, - 0.4 );
		controls.minDistance = 5;
		controls.maxDistance = 30;
		controls.maxPolarAngle = Math.PI * 0.47;
		controls.autoRotate = true;
		controls.autoRotateSpeed = 0;
		controls.update();

		// ───────────────────────── pointer ─────────────────────────
		const canvas = renderer.domElement;
		const raycaster = new THREE.Raycaster();
		const plane = new THREE.Plane( new THREE.Vector3( 0, 1, 0 ), 0 );
		const hitP = new THREE.Vector3();
		const ndc = new THREE.Vector2();
		const drag = { active: false, mode: null, start: new THREE.Vector2(), cur: new THREE.Vector2(), last: null };
		const paintQueue = [];

		const toDomain = ( clientX, clientY ) => {

			const r = canvas.getBoundingClientRect();
			ndc.set( ( ( clientX - r.left ) / r.width ) * 2 - 1, - ( ( clientY - r.top ) / r.height ) * 2 + 1 );
			raycaster.setFromCamera( ndc, camera );
			if ( ! raycaster.ray.intersectPlane( plane, hitP ) ) return null;
			return new THREE.Vector2( hitP.x / WORLD + 0.5, hitP.z / WORLD + 0.5 );

		};

		const inside = ( q ) => q && q.x > 0.02 && q.x < 0.98 && q.y > 0.02 && q.y < 0.98;

		const onDown = ( e ) => {

			if ( e.target !== canvas || ( e.button !== 0 && e.pointerType === 'mouse' ) ) return;
			const q = toDomain( e.clientX, e.clientY );
			if ( ! inside( q ) ) return;
			controls.enabled = false; // the plane belongs to the experiment; orbit by dragging the background
			drag.active = true;
			drag.mode = ( e.shiftKey || params.pointer === POINTER_MODES[ 1 ] ) ? 'paint' : 'fire';
			drag.start.copy( q );
			drag.cur.copy( q );
			drag.last = q.clone();
			if ( drag.mode === 'paint' ) paintQueue.push( [ q.clone(), q.clone() ] );

		};

		const onMove = ( e ) => {

			if ( ! drag.active ) return;
			const q = toDomain( e.clientX, e.clientY );
			if ( ! q ) return;
			drag.cur.copy( q );
			if ( drag.mode === 'paint' && q.distanceTo( drag.last ) > 0.012 ) {

				paintQueue.push( [ drag.last.clone(), q.clone() ] );
				drag.last.copy( q );

			}

		};

		const onUp = () => {

			if ( ! drag.active ) return;
			drag.active = false;
			controls.enabled = true;
			if ( drag.mode === 'fire' ) {

				const dir = drag.cur.clone().sub( drag.start );
				if ( dir.length() < 0.025 ) {

					if ( scn === 'slit' || scn === 'tunnel' ) dir.set( 1, 0 );
					else dir.set( 0.5 - drag.start.x, 0.5 - drag.start.y ).normalize().rotateAround( new THREE.Vector2(), 0.6 );

				}

				dir.normalize();
				launch( { x: drag.start.x, y: drag.start.y, kx: dir.x * params.k, ky: dir.y * params.k, sigma: params.sigma } );

			}

		};

		window.addEventListener( 'pointerdown', onDown, true );
		window.addEventListener( 'pointermove', onMove, true );
		window.addEventListener( 'pointerup', onUp, true );
		window.addEventListener( 'pointercancel', onUp, true );

		const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _p = new THREE.Vector3(), _up = new THREE.Vector3( 0, 1, 0 );

		function applyPaint() {

			while ( paintQueue.length ) {

				const [ a, b ] = paintQueue.shift();
				solver.u.pa.value.copy( a );
				solver.u.pb.value.copy( b );
				renderer.compute( solver.paint );
				if ( painted.count < MAX_SEGMENTS ) {

					const r = solver.u.pr.value;
					const len = a.distanceTo( b );
					_p.set( ( ( a.x + b.x ) / 2 - 0.5 ) * WORLD, 0.45, ( ( a.y + b.y ) / 2 - 0.5 ) * WORLD );
					_q.setFromAxisAngle( _up, - Math.atan2( b.y - a.y, b.x - a.x ) );
					_s.set( ( len + 2 * r ) * WORLD, 0.9, 2 * r * WORLD );
					_m.compose( _p, _q, _s );
					painted.setMatrixAt( painted.count ++, _m );
					painted.instanceMatrix.needsUpdate = true;

				}

			}

		}

		function updateGizmo() {

			const g = gizmo.userData;
			let at = null;
			if ( drag.active ) at = drag.start;
			else if ( ctx.pointer.inside && ! ctx.pointer.down ) {

				if ( ctx.pointer.ray.intersectPlane( plane, hitP ) ) {

					const q = new THREE.Vector2( hitP.x / WORLD + 0.5, hitP.z / WORLD + 0.5 );
					if ( inside( q ) ) at = q;

				}

			}

			gizmo.visible = !! at && ! ctx.shot;
			if ( ! at ) return;
			const paint = drag.active ? drag.mode === 'paint' : params.pointer === POINTER_MODES[ 1 ];
			const pos = drag.active && paint ? drag.cur : at;
			gizmo.position.set( ( pos.x - 0.5 ) * WORLD, 0.03, ( pos.y - 0.5 ) * WORLD );
			const rad = paint ? solver.u.pr.value * WORLD : 2 * params.sigma * WORLD;
			g.ring.scale.setScalar( rad );
			g.ring.material.color.setRGB( paint ? 2.2 : 0.6, paint ? 1.2 : 1.6, paint ? 0.4 : 2.4 );
			g.width.visible = false;
			const aiming = drag.active && ! paint;
			let dir = new THREE.Vector2( 1, 0 );
			if ( aiming ) {

				const d = drag.cur.clone().sub( drag.start );
				if ( d.length() > 0.02 ) dir = d.normalize();

			}

			const show = ! paint;
			g.shaft.visible = g.head.visible = show;
			const len = aiming ? Math.max( 0.8, drag.cur.distanceTo( drag.start ) * WORLD ) : 0.7 + 0.15 * Math.sin( clock * 4 );
			const ang = - Math.atan2( dir.y, dir.x );
			g.shaft.rotation.y = ang;
			g.shaft.position.set( Math.cos( - ang ) * rad, 0, Math.sin( - ang ) * rad ).multiplyScalar( 1 );
			g.shaft.position.set( dir.x * rad, 0, dir.y * rad );
			g.shaft.scale.set( len, 1, 1 );
			g.head.rotation.y = ang;
			g.head.position.set( dir.x * ( rad + len + 0.3 ), 0, dir.y * ( rad + len + 0.3 ) );

		}

		// ───────────────────────── stats ─────────────────────────
		let statT = 0, statPending = false;
		function pollStats() {

			if ( statPending ) return;
			statPending = true;
			Promise.all( [ solver.readNorm(), solver.readHits() ] ).then( ( [ n, h ] ) => {

				if ( disposed ) return;
				lastNorm = n;
				hitCount = h;
				ctx.setStats( [
					[ 'grid', `${N}² · FFT split-step` ],
					[ 'norm ∫|ψ|²', n.toFixed( 4 ) ],
					[ 'hits detected', h.toLocaleString() ],
				] );

			} ).catch( () => {} ).finally( () => statPending = false );

		}

		ctx.setStats( [ [ 'grid', `${N}² · FFT split-step` ], [ 'norm ∫|ψ|²', '1.0000' ], [ 'hits detected', '0' ] ] );

		// ───────────────────────── frame ─────────────────────────
		function update( dt, t ) {

			clock += dt;
			frameNo ++;
			applyPaint();

			const steps = ctx.shot ? cfg.steps : Math.max( 1, Math.min( cfg.steps * 2, Math.round( cfg.steps * dt * 60 ) ) );
			solver.u.frameTau.value = steps * solver.dt;
			solver.u.realDt.value = dt;
			solver.u.time.value = clock;
			solver.u.frame.value = frameNo;
			look.time.value = clock;
			renderer.compute( frameList( steps, params.mist ) );

			// keep the experiment running: a fresh, identical packet whenever the last one has been detected/absorbed
			const since = clock - launchedAt;
			if ( ( scn === 'slit' || scn === 'tunnel' ) && ( ( lastNorm < 0.03 && since > 2 ) || since > 16 ) ) launchDefault();
			if ( ( scn === 'ho' || scn === 'stadium' ) && lastNorm < 0.02 && since > 2 ) launchDefault();

			statT += dt;
			if ( statT > 0.5 ) {

				statT = 0;
				pollStats();

			}

			// slow cinematic sway around the target
			controls.autoRotateSpeed = 0.22 * Math.sin( t * 0.11 );
			updateGizmo();

		}

		return {
			scene,
			bloom: { strength: 0.55, radius: 0.55, threshold: 0.12 },
			update,
			onKey( e ) {

				if ( e.key === 'r' || e.key === 'R' ) launchDefault();

			},
			dispose() {

				disposed = true;
				window.removeEventListener( 'pointerdown', onDown, true );
				window.removeEventListener( 'pointermove', onMove, true );
				window.removeEventListener( 'pointerup', onUp, true );
				window.removeEventListener( 'pointercancel', onUp, true );
				controls.enabled = true;
				controls.autoRotate = false;
				controls.autoRotateSpeed = 2;
				controls.maxPolarAngle = Math.PI;
				clearMonoliths();
				surface.geometry.dispose(); surface.material.dispose();
				floor.geometry.dispose(); floor.material.dispose();
				wallMat.dispose(); barrierMat.dispose();
				paintGeo.dispose(); painted.dispose();
				detector.userData.dispose();
				mist.userData.dispose(); mist.geometry?.dispose?.();
				gizmo.userData.dispose();
				key.dispose();
				envRT.dispose();
				scene.environment = null;
				scene.backgroundNode = null;
				solver.dispose();
				for ( const d of disposables ) d();

			},
		};

	},
};

export { HO_OMEGA };
