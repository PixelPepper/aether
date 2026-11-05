// Chapter 13 — Silk
// GPU XPBD cloth with small substeps (Macklin et al. 2019): 12-colour Gauss–Seidel sweeps over stretch,
// shear and bending distance constraints, analytic SDF sculptures with Coulomb friction, sheet-vs-sheet and
// self collision through a GPU spatial hash (atomic counting sort + scan), long-range attachments, curl-mode
// turbulent wind with normal-dependent aerodynamic pressure, and a sheen / anisotropic GGX / thin-film /
// diffuse-transmission silk BRDF with woven micro-relief.
import * as THREE from 'three/webgpu';
import { createClothSim } from './silk/sim.js';
import { createSilkMaterial } from './silk/material.js';
import { buildStage, makeLayout } from './silk/stage.js';

export default {
	id: 'silk',
	title: 'Silk',
	subtitle: 'Three sheets of shot silk, hundreds of thousands of particles under XPBD small-step dynamics, billowing in curl-noise wind over sculpture — grab them.',

	async create( ctx ) {

		const { renderer, camera, controls, tier, params } = ctx;
		const DBG = params.has( 'silkDebug' );
		const t0 = performance.now();
		const dbg = ( m ) => DBG && console.warn( `[silk ${( ( performance.now() - t0 ) / 1000 ).toFixed( 1 )}s] ${m}` );
		const scene = new THREE.Scene();
		scene.background = new THREE.Color( 0x020202 );

		// ── resolution per tier ────────────────────────────────────────────────
		const N = parseInt( params.get( 'silkN' ) || ( tier === 0 ? 48 : tier === 1 ? 128 : 256 ) );
		const S = 3;
		const SIZE = 3.5;
		const defaults = {
			substeps: parseInt( params.get( 'silkSub' ) || ( tier === 0 ? 6 : tier === 1 ? 10 : 16 ) ),
			selfEvery: tier === 0 ? 2 : 4,
		};

		const layout = makeLayout();
		const prevShadow = { enabled: renderer.shadowMap.enabled, type: renderer.shadowMap.type };
		renderer.shadowMap.enabled = true;
		renderer.shadowMap.type = THREE.PCFShadowMap;

		const stage = await buildStage( ctx, scene, layout, tier );
		dbg( 'stage built' );

		const sim = createClothSim( renderer, { N, S, size: SIZE, tier, colliders: layout.colliders, hangSheets: layout.hangSheets, dropSheets: layout.dropSheets } );
		const { U, P, NN } = sim;

		// ── render geometry: storage buffers are the vertex buffers ─────────────
		const geo = new THREE.BufferGeometry();
		geo.setAttribute( 'position', sim.attrs.posA );
		geo.setAttribute( 'normal', sim.attrs.nrmA );
		geo.setAttribute( 'tangent', sim.attrs.tanA );
		const uvs = new Float32Array( P * 2 ), sheet = new Float32Array( P );
		for ( let s = 0; s < S; s ++ ) for ( let j = 0; j < N; j ++ ) for ( let i = 0; i < N; i ++ ) {

			const k = s * NN + j * N + i;
			uvs[ k * 2 ] = i / ( N - 1 ); uvs[ k * 2 + 1 ] = 1 - j / ( N - 1 );
			sheet[ k ] = s;

		}

		geo.setAttribute( 'uv', new THREE.BufferAttribute( uvs, 2 ) );
		geo.setAttribute( 'sheet', new THREE.BufferAttribute( sheet, 1 ) );
		const idx = new Uint32Array( S * ( N - 1 ) * ( N - 1 ) * 6 );
		let q = 0;
		for ( let s = 0; s < S; s ++ ) for ( let j = 0; j < N - 1; j ++ ) for ( let i = 0; i < N - 1; i ++ ) {

			const a = s * NN + j * N + i, b = a + 1, c = a + N, d = c + 1;
			// counter-clockwise around n = cross(dP/di, dP/dj) (the normal the compute pass writes);
			// alternate the quad diagonal so the triangulation has no directional bias
			if ( ( i + j ) & 1 ) {

				idx[ q ++ ] = a; idx[ q ++ ] = b; idx[ q ++ ] = c;
				idx[ q ++ ] = b; idx[ q ++ ] = d; idx[ q ++ ] = c;

			} else {

				idx[ q ++ ] = a; idx[ q ++ ] = b; idx[ q ++ ] = d;
				idx[ q ++ ] = a; idx[ q ++ ] = d; idx[ q ++ ] = c;

			}

		}

		geo.setIndex( new THREE.BufferAttribute( idx, 1 ) );

		const silk = createSilkMaterial( { threads: 640 } );
		const mesh = new THREE.Mesh( geo, silk.material );
		mesh.frustumCulled = false;
		mesh.castShadow = true;
		mesh.receiveShadow = true;
		scene.add( mesh );

		// ── controls / uniforms from GUI ────────────────────────────────────────
		const ui = {
			wind: 3.6,
			turbulence: 0.85,
			stretch: 0.85,
			bend: 0.45,
			substeps: defaults.substeps,
			colour: params.get( 'silkColor' ) || 'pearl',
			selfCollision: true,
			reset: () => reset(),
			drop: () => dropOnto(),
		};
		silk.setPalette( ui.colour );
		sim.schedule( ui.substeps, defaults.selfEvery, ui.selfCollision );

		const gui = ctx.gui;
		gui.add( ui, 'wind', 0, 9, 0.1 ).name( 'wind strength' );
		gui.add( ui, 'turbulence', 0, 1.6, 0.01 ).name( 'turbulence' );
		gui.add( ui, 'stretch', 0, 1, 0.01 ).name( 'stretch stiffness' );
		gui.add( ui, 'bend', 0, 1, 0.01 ).name( 'bend stiffness' );
		gui.add( ui, 'substeps', 4, 24, 1 ).name( 'substeps' ).onFinishChange( ( v ) => sim.schedule( v, defaults.selfEvery, ui.selfCollision ) );
		gui.add( ui, 'colour', [ 'pearl', 'gold', 'oil-slick' ] ).name( 'colour' ).onChange( ( v ) => silk.setPalette( v ) );
		gui.add( ui, 'selfCollision' ).name( 'self collision' ).onChange( ( v ) => sim.schedule( ui.substeps, defaults.selfEvery, v ) );
		gui.add( ui, 'reset' ).name( 'reset' );
		gui.add( ui, 'drop' ).name( 'drop onto sculpture' );

		// ── camera ──────────────────────────────────────────────────────────────
		camera.fov = 36;
		camera.near = 0.05; camera.far = 200;
		camera.updateProjectionMatrix();
		camera.position.set( 7.4, 1.15, 8.4 );
		controls.target.set( 0.1, 2.35, - 0.4 );
		controls.minDistance = 3.5;
		controls.maxDistance = 18;
		controls.maxPolarAngle = 1.56;
		controls.minPolarAngle = 0.35;
		controls.autoRotate = true;
		controls.autoRotateSpeed = 0.3;
		controls.update();

		// ── state ───────────────────────────────────────────────────────────────
		let mode = 'hang';
		let simTime = 0;
		const v3 = new THREE.Vector3();

		function placeAnchors( t ) {

			for ( let k = 0; k < sim.anchorsJS.length; k ++ ) {

				const a = sim.anchorsJS[ k ];
				const ph = a.phase, A = a.amp;
				sim.anchorPos.array[ k ].set(
					a.base.x + A * Math.sin( 0.31 * t + ph[ 0 ] ),
					a.base.y + A * 0.35 * Math.sin( 0.43 * t + ph[ 1 ] ),
					a.base.z + A * Math.sin( 0.23 * t + ph[ 2 ] ), 0 );

			}

		}

		function setFrameUniforms( dtFrame ) {

			const h = dtFrame / ui.substeps;
			U.dt.value = h;
			U.invDt.value = 1 / h;
			const h2 = h * h;
			const aStretch = Math.pow( 10, - 4 - 5 * ui.stretch );
			U.aStretch.value = aStretch / h2;
			U.aShear.value = aStretch * 8 / h2;
			U.aBend.value = Math.pow( 10, - 2.5 - 4 * ui.bend ) / h2;
			simTime += dtFrame;
			U.time.value = simTime;
			// slow gust envelope on top of the travelling gust fronts in the shader
			const g = 1 + 0.22 * Math.sin( 0.37 * simTime ) * Math.sin( 0.13 * simTime + 1.3 ) + 0.12 * Math.sin( 1.1 * simTime + Math.sin( 0.5 * simTime ) );
			U.windSpeed.value = ui.wind * g * ( mode === 'drop' ? 0.3 : 1 );
			U.turbulence.value = ui.turbulence;
			const wa = 0.95 + 0.25 * Math.sin( simTime * 0.07 );
			U.windDir.value.set( Math.sin( wa ), 0.04, Math.cos( wa ) ).normalize();
			if ( mode === 'hang' ) placeAnchors( simTime );

		}

		function reset() {

			endGrab();
			mode = 'hang';
			U.pinsOn.value = 1;
			placeAnchors( simTime );
			renderer.compute( [ sim.kResetHang, sim.kNormals ] );

		}

		function dropOnto() {

			endGrab();
			mode = 'drop';
			U.pinsOn.value = 0;
			renderer.compute( [ sim.kResetDrop, sim.kNormals ] );

		}

		// ── pointer: GPU pick on hover, grab + drag with a long-range tether ────────
		const canvas = renderer.domElement;
		const hover = { hit: false, idx: - 1, pos: new THREE.Vector3(), time: - 1e9 };
		let pickPending = false;
		let disposed = false;
		const grab = { active: false, depth: 0, target: new THREE.Vector3(), sheet: 0, ctrlWas: true, pending: false };

		async function runPick( ray ) {

			if ( pickPending || disposed ) return;
			pickPending = true;
			U.rayO.value.copy( ray.origin );
			U.rayD.value.copy( ray.direction ).normalize();
			try {

				renderer.compute( sim.pickList );
				const buf = await renderer.getArrayBufferAsync( sim.attrs.pickA );
				if ( disposed ) return;
				const u = new Uint32Array( buf ), f = new Float32Array( buf );
				hover.hit = u[ 1 ] < P;
				hover.idx = u[ 1 ];
				if ( hover.hit ) hover.pos.set( f[ 2 ], f[ 3 ], f[ 4 ] );
				hover.time = performance.now();
				if ( grab.pending ) {

					grab.pending = false;
					if ( hover.hit && ctx.pointer.down ) beginGrab();

				}

				if ( ! grab.active ) canvas.style.cursor = hover.hit ? 'grab' : '';

			} catch ( e ) {

				// readback can fail during teardown; ignore
			} finally {

				pickPending = false;

			}

		}

		function beginGrab() {

			const i = hover.idx;
			const s = Math.floor( i / NN ), r = i % NN;
			grab.active = true;
			grab.sheet = s;
			const ray = ctx.pointer.ray;
			grab.depth = Math.max( 0.5, v3.copy( hover.pos ).sub( ray.origin ).dot( ray.direction ) );
			grab.target.copy( hover.pos );
			U.grabIdx.value = i;
			U.grabSheet.value = s;
			U.grabI.value = r % N;
			U.grabJ.value = Math.floor( r / N );
			U.grabTarget.value.copy( hover.pos );
			U.grabOn.value = 1;
			grab.ctrlWas = controls.enabled;
			controls.enabled = false;
			canvas.style.cursor = 'grabbing';

		}

		function endGrab() {

			grab.pending = false;
			if ( ! grab.active ) return;
			grab.active = false;
			U.grabOn.value = 0;
			controls.enabled = grab.ctrlWas;
			canvas.style.cursor = '';

		}

		const onDown = ( e ) => {

			if ( e.button !== 0 ) return;
			const fresh = hover.hit && performance.now() - hover.time < 500;
			if ( fresh && e.pointerType === 'mouse' ) {

				beginGrab();

			} else if ( e.pointerType !== 'mouse' ) {

				grab.pending = true; // touch: no hover — pick now, grab when the result lands

			}

		};

		const onUp = () => endGrab();
		canvas.addEventListener( 'pointerdown', onDown, { capture: true } );
		addEventListener( 'pointerup', onUp );
		addEventListener( 'pointercancel', onUp );

		// clamp the drag target so a hanging sheet is never pulled beyond its own geodesic length from a pin
		function clampGrabTarget( t ) {

			if ( mode !== 'hang' ) return;
			const gi = U.grabI.value, gj = U.grabJ.value;
			for ( let pass = 0; pass < 2; pass ++ ) {

				for ( let a = 0; a < sim.A; a ++ ) {

					const k = grab.sheet * sim.A + a;
					const an = sim.anchorsJS[ k ];
					const L = Math.hypot( gi - an.ia, gj - an.ja ) * sim.spacing * 0.995;
					const ap = sim.anchorPos.array[ k ];
					v3.set( t.x - ap.x, t.y - ap.y, t.z - ap.z );
					const d = v3.length();
					if ( d > L ) t.set( ap.x, ap.y, ap.z ).addScaledVector( v3, L / d );

				}

			}

		}

		// ── headless self-test of the grab path: ?silkGrab=1 ──────────────────────
		const scripted = ctx.shot && params.has( 'silkGrab' );
		let frameNo = 0;
		const fakeRay = new THREE.Ray();
		const ndc = new THREE.Vector2();
		const scriptedRay = ( x, y ) => {

			ndc.set( x, y );
			const rc = new THREE.Raycaster();
			rc.setFromCamera( ndc, camera );
			fakeRay.copy( rc.ray );
			return fakeRay;

		};

		// ── initial pose + warm-up so frame 1 is already mid-billow ─────────────────
		placeAnchors( 0 );
		setFrameUniforms( 0 ); simTime = 0;
		await renderer.computeAsync( [ sim.kResetHang, sim.kNormals ] );
		dbg( 'reset done' );
		const warm = parseInt( params.get( 'silkWarm' ) || ( tier === 0 ? 12 : tier === 1 ? 60 : 90 ) );
		for ( let f = 0; f < warm; f ++ ) {

			setFrameUniforms( 1 / 60 );
			renderer.compute( sim.frameList );
			if ( DBG && f === 0 ) {

				await renderer.getArrayBufferAsync( sim.attrs.posA, null, 0, 16 );
				dbg( 'first sim frame (pipelines compiled)' );

			}

		}

		if ( DBG ) {

			await renderer.getArrayBufferAsync( sim.attrs.posA, null, 0, 16 );
			dbg( `warm-up ${warm} frames done` );

		}

		let statT = 0;
		const fmt = ( n ) => n >= 1e6 ? ( n / 1e6 ).toFixed( 2 ) + 'M' : n >= 1e3 ? ( n / 1e3 ).toFixed( 1 ) + 'k' : String( n );
		const stats = () => ctx.setStats( [
			[ 'particles', fmt( P ) + ` (${S}×${N}²)` ],
			[ 'constraints', fmt( sim.constraintCount ) + ' ×' + ui.substeps ],
			[ 'substeps', String( ui.substeps ) ],
			[ 'self-collision', ui.selfCollision ? `hash ${fmt( sim.T )} cells /${defaults.selfEvery}` : 'off' ],
		] );
		stats();

		return {
			scene,
			bloom: { strength: 0.32, radius: 0.55, threshold: 0.7 },

			update( dt, t, c ) {

				frameNo ++;
				if ( frameNo <= 3 ) dbg( 'frame ' + frameNo );
				const dtFrame = Math.min( Math.max( dt, 1 / 240 ), 1 / 30 );
				setFrameUniforms( dtFrame );

				// camera: slow ping-pong orbit across the designed 3/4 arc
				if ( controls.enabled && ! c.pointer.down ) {

					const az = controls.getAzimuthalAngle();
					if ( az > 1.3 ) controls.autoRotateSpeed = - Math.abs( controls.autoRotateSpeed );
					if ( az < 0.05 ) controls.autoRotateSpeed = Math.abs( controls.autoRotateSpeed );

				}

				// grab target follows the pointer ray at the grabbed depth
				if ( grab.active ) {

					const ray = scripted ? fakeRay : c.pointer.ray;
					const desired = v3.copy( ray.direction ).multiplyScalar( grab.depth ).add( ray.origin ).clone();
					desired.y = Math.max( desired.y, 0.05 );
					clampGrabTarget( desired );
					grab.target.lerp( desired, 1 - Math.exp( - dtFrame * 30 ) );
					U.grabTarget.value.copy( grab.target );

				}

				renderer.compute( sim.frameList );

				if ( scripted ) {

					// pick at frame 12, grab at 16, drag up-left until the end
					if ( frameNo === 12 ) runPick( scriptedRay( 0.02, 0.0 ) );
					if ( frameNo >= 16 && ! grab.active && hover.hit && frameNo < 18 ) beginGrab();
					if ( grab.active ) scriptedRay( 0.02 - ( frameNo - 16 ) * 0.006, 0.0 + ( frameNo - 16 ) * 0.012 );

				} else if ( ! grab.active && ( c.pointer.inside || grab.pending ) && ( frameNo % 2 === 0 || grab.pending ) ) {

					runPick( c.pointer.ray );

				}

				statT += dt;
				if ( statT > 0.5 ) {

					statT = 0; stats();

				}

			},

			onKey( e ) {

				if ( e.key === 'r' || e.key === 'R' ) reset();
				if ( e.key === 'd' || e.key === 'D' ) dropOnto();

			},

			dispose() {

				disposed = true;
				endGrab();
				canvas.removeEventListener( 'pointerdown', onDown, { capture: true } );
				removeEventListener( 'pointerup', onUp );
				removeEventListener( 'pointercancel', onUp );
				canvas.style.cursor = '';
				controls.enabled = true;
				scene.remove( mesh );
				geo.dispose();
				silk.material.dispose();
				sim.dispose();
				stage.dispose();
				renderer.shadowMap.enabled = prevShadow.enabled;
				renderer.shadowMap.type = prevShadow.type;

			},
		};

	},
};
