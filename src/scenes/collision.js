// Chapter 03 — Collision: a live galaxy merger.
//
// Hybrid N-body: a few thousand massive bodies (dark halos, bulges, SMBHs) integrated with exact direct
// summation (shared-memory tiles, Plummer softening, kick-drift-kick leapfrog); up to ~1M massless star/gas
// tracers integrated in their field (tiled again). Gas is CIC-splatted into a 3D grid every frame with
// integer atomics; that grid drives sticky-gas dissipation, density/shock-triggered starbursts (HII knots)
// and per-star dust extinction ray-marched toward the camera. Rendering: flux-conserving additive sprites
// into HDR, asinh (Lupton) stretch, bloom.
import * as THREE from 'three/webgpu';
import { Fn, vec3, vec4, float, dot, max, asinh, mix } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { createSim } from './collision/kernels.js';
import { createRenderers } from './collision/render.js';
import { planLayout, buildScene, sampleSatellite, mulberry32, MYR_PER_UNIT, KPC_PER_UNIT, T_GAS, T_SAT } from './collision/ics.js';

const TIERS = [
	{ nm: 512, nt: 16384, wg: 64, G: 48, epsM: 0.22, epsT: 0.24, starSize: 0.085, minPx: 1.4, knots: 28, extSteps: 8 },
	{ nm: 2048, nt: 131072, wg: 128, G: 64, epsM: 0.17, epsT: 0.18, starSize: 0.045, minPx: 1.3, knots: 90, extSteps: 10 },
	{ nm: 4096, nt: 524288, wg: 256, G: 96, epsM: 0.14, epsT: 0.15, starSize: 0.03, minPx: 1.2, knots: 220, extSteps: 12 },
];

export default {
	id: 'collision',
	title: 'Collision',
	subtitle: 'Two spiral galaxies in a live N-body merger — dark halos tiled through GPU shared memory, tidal tails flung from their disks, shocked gas igniting into starbursts.',

	async create( ctx ) {

		const { renderer, camera, controls } = ctx;
		const prm = ctx.params;
		const tier = Math.max( 0, Math.min( 2, ctx.tier | 0 ) );
		const T = { ...TIERS[ tier ] };
		if ( prm.has( 'stars' ) ) {

			const s = prm.get( 'stars' ).toLowerCase();
			const n = s.endsWith( 'm' ) ? parseFloat( s ) * 1048576 : s.endsWith( 'k' ) ? parseFloat( s ) * 1024 : parseFloat( s );
			T.nt = Math.max( 4096, Math.round( n / 256 ) * 256 );

		}

		if ( prm.has( 'bodies' ) ) T.nm = Math.max( 512, Math.round( parseFloat( prm.get( 'bodies' ) ) / T.wg ) * T.wg );

		const layout = planLayout( tier, T.nm, T.nt );
		const box = 30;
		const gridMin = new THREE.Vector3( - box / 2, - box / 2, - box / 2 );
		const gasPerGal = layout.gal[ 0 ].gas;
		const cell = box / T.G;
		// reference CIC count at the centre of a fresh gas disk: N c² / (2π Rg²)
		const countRef = gasPerGal * cell * cell / ( 2 * Math.PI * 1.7 * 1.7 ) * 0.75;
		const sfBase = T.knots / ( gasPerGal * 0.25 * 2.6 );

		const cfg = {
			...T, nGas: layout.nGas, gridMin, box, densNorm: 1 / countRef, sfBase,
			clusterLum: 0.0016,
			gasLum: 0.07 / gasPerGal,
		};

		const sim = createSim( cfg );
		const B = sim.buffers, U = sim.U;
		const R = createRenderers( sim, {
			nt: T.nt, nm: T.nm, nGas: layout.nGas, exposure: 9.0, starSize: T.starSize, minPx: T.minPx,
			glowSize: 0.16, glowGain: 0.0013, dmSize: 0.9, dmExposure: 0.9,
		} );

		const scene = new THREE.Scene();
		scene.background = new THREE.Color( 0x000000 );
		R.objects.forEach( ( o ) => scene.add( o ) );

		// ───────────────────────── parameters ─────────────────────────
		const P = {
			timeScale: prm.has( 'ts' ) ? parseFloat( prm.get( 'ts' ) ) : 1.0,
			substeps: tier === 0 ? 2 : 3,
			massRatio: 1.0,
			inclination: 62,
			pericentre: 2.4,
			showDM: prm.get( 'dm' ) === '1',
			projectile: 'dwarf galaxy',
			starFormation: 1.0,
			dust: 1.0,
			paused: false,
		};
		const UNITS_PER_SEC = 1.0;

		let simTime = 0, frameNo = 0, restarting = false;
		const nuclei = { a: new THREE.Vector3( 4, 0, - 5 ), b: new THREE.Vector3( - 4, 0, 5 ), mid: new THREE.Vector3(), sep: 13 };
		const smoothMid = new THREE.Vector3();
		let lastS = null;

		// ───────────────────────── initial conditions ─────────────────────────
		function upload( node, arr ) {

			const attr = node.value;
			attr.array.set( arr );
			attr.clearUpdateRanges();
			attr.needsUpdate = true;

		}

		async function restart() {

			restarting = true;
			const S = buildScene( {
				tier, layout, massRatio: P.massRatio, inclA: 22, azA: - 35, inclB: P.inclination, azB: 65,
				peri: P.pericentre, sep: 13, ecc: 0.9, seed: 7, eps: T.epsT, model: { haloA: 1.6 },
			} );
			lastS = S;
			upload( B.mPos, S.mPos ); upload( B.mVel, S.mVel );
			upload( B.tPos, S.tPos ); upload( B.tVel, S.tVel ); upload( B.tInfo, S.tInfo );
			upload( B.mAcc, new Float32Array( T.nm * 4 ) ); upload( B.tAcc, new Float32Array( T.nt * 4 ) );
			simTime = 0;
			U.time.value = 0;
			nuclei.a.fromArray( S.mPos, 0 ); nuclei.b.fromArray( S.mPos, 4 );
			nuclei.mid.addVectors( nuclei.a, nuclei.b ).multiplyScalar( 0.5 );
			smoothMid.copy( nuclei.mid );
			satCounter = 0; pertCounter = 0;
			// initial forces (zero-length kick) + grid + shading
			U.hM.value = 0; U.hT.value = 0;
			renderer.compute( [ sim.zeroStats, sim.mForceKick, sim.tForceKick, sim.gClear, sim.gSplat, sim.gResolve ] );
			restarting = false;

		}

		// ───────────────────────── stepping ─────────────────────────
		function step( dtSim ) {

			const n = Math.max( 1, Math.round( P.substeps ) );
			U.hM.value = dtSim / n;
			U.hT.value = dtSim;
			simTime += dtSim;
			U.time.value = simTime;
			U.frame.value = ( U.frame.value + 1 ) >>> 0;
			const list = [];
			for ( let s = 0; s < n; s ++ ) list.push( sim.mKickDrift, sim.mForceKick );
			list.push( sim.tKickDrift, sim.tForceKick, sim.gClear, sim.gSplat, sim.gResolve, sim.gasUpdate );
			renderer.compute( list );

		}

		// ───────────────────────── projectiles ─────────────────────────
		let satCounter = 0, pertCounter = 0;
		const satRng = mulberry32( 1234 );

		function rangeUpload( node, start, data ) {

			node.value.array.set( data, start * 4 );
			node.value.addUpdateRange( start * 4, data.length );
			node.value.needsUpdate = true;

		}

		function spawn( kind, origin, velocity ) {

			if ( kind === 'black hole' ) {

				const k = layout.pertStart + ( pertCounter ++ % layout.nPert );
				rangeUpload( B.mPos, k, [ origin.x, origin.y, origin.z, 0.16 ] );
				rangeUpload( B.mVel, k, [ velocity.x, velocity.y, velocity.z, 5 ] );
				rangeUpload( B.mAcc, k, [ 0, 0, 0, 0 ] );
				return;

			}

			const k = satCounter ++ % layout.nSat;
			const nStar = layout.satT - layout.satGas;
			const sat = sampleSatellite( layout.satM, layout.satT, layout.satGas, 0.09, 0.32, satRng );
			const add = ( arr, n, o, v ) => {

				for ( let i = 0; i < n; i ++ ) {

					arr.p[ i * 4 ] += o.x; arr.p[ i * 4 + 1 ] += o.y; arr.p[ i * 4 + 2 ] += o.z;
					arr.v[ i * 4 ] += v.x; arr.v[ i * 4 + 1 ] += v.y; arr.v[ i * 4 + 2 ] += v.z;

				}

			};

			add( { p: sat.mp, v: sat.mv }, layout.satM, origin, velocity );
			add( { p: sat.tp, v: sat.tv }, layout.satT, origin, velocity );
			// luminosity per star so a dwarf is ~6% of a big spiral
			for ( let i = layout.satGas; i < layout.satT; i ++ ) sat.ti[ i * 4 + 2 ] = 0.07 / nStar;
			const m0 = layout.satMStart + k * layout.satM;
			rangeUpload( B.mPos, m0, sat.mp ); rangeUpload( B.mVel, m0, sat.mv );
			rangeUpload( B.mAcc, m0, new Float32Array( layout.satM * 4 ) );
			const g0 = layout.satGasStart + k * layout.satGas;
			const s0 = layout.satStarStart + k * nStar;
			const nG4 = layout.satGas * 4;
			rangeUpload( B.tPos, g0, sat.tp.subarray( 0, nG4 ) ); rangeUpload( B.tVel, g0, sat.tv.subarray( 0, nG4 ) ); rangeUpload( B.tInfo, g0, sat.ti.subarray( 0, nG4 ) );
			rangeUpload( B.tPos, s0, sat.tp.subarray( nG4 ) ); rangeUpload( B.tVel, s0, sat.tv.subarray( nG4 ) ); rangeUpload( B.tInfo, s0, sat.ti.subarray( nG4 ) );
			rangeUpload( B.tAcc, g0, new Float32Array( nG4 ) );
			rangeUpload( B.tAcc, s0, new Float32Array( nStar * 4 ) );

		}

		const _ray = new THREE.Raycaster();
		const _v = new THREE.Vector3(), _f = new THREE.Vector3(), _o = new THREE.Vector3(), _vel = new THREE.Vector3();

		// throw along the pointer ray: aim at the depth of the orbit target, start ~11 units in front of it
		function throwAt( ndcX, ndcY, kind, lateral = null ) {

			_ray.setFromCamera( { x: ndcX, y: ndcY }, camera );
			const dir = _ray.ray.direction;
			camera.getWorldDirection( _f );
			const tAim = Math.max( 4, _v.subVectors( controls.target, _ray.ray.origin ).dot( _f ) / Math.max( dir.dot( _f ), 0.2 ) );
			const back = Math.min( 11, tAim - 2.5 );
			_o.copy( _ray.ray.origin ).addScaledVector( dir, tAim - back );
			const speed = kind === 'black hole' ? 0.85 : 0.7;
			_vel.copy( dir ).multiplyScalar( speed );
			if ( lateral ) _vel.add( lateral );
			spawn( kind, _o.clone(), _vel.clone() );

		}

		// pointer: click = drop a projectile along the ray; shift-drag = fling it with the drag velocity
		const el = renderer.domElement;
		const down = { x: 0, y: 0, t: 0, active: false, shift: false };
		const ndcOf = ( e ) => {

			const r = el.getBoundingClientRect();
			return [ ( ( e.clientX - r.left ) / r.width ) * 2 - 1, - ( ( e.clientY - r.top ) / r.height ) * 2 + 1 ];

		};

		const onDown = ( e ) => {

			if ( e.button !== 0 ) return;
			down.x = e.clientX; down.y = e.clientY; down.t = performance.now(); down.active = true; down.shift = e.shiftKey;
			if ( e.shiftKey ) controls.enabled = false;

		};

		const onUp = ( e ) => {

			if ( ! down.active ) return;
			down.active = false;
			const dx = e.clientX - down.x, dy = e.clientY - down.y, moved = Math.hypot( dx, dy );
			const dt = ( performance.now() - down.t ) / 1000;
			if ( down.shift ) {

				controls.enabled = true;
				const [ nx, ny ] = ndcOf( { clientX: down.x, clientY: down.y } );
				// lateral velocity from the drag, in world units at the target depth
				const dist = camera.position.distanceTo( controls.target );
				const worldPerPx = 2 * Math.tan( THREE.MathUtils.degToRad( camera.fov / 2 ) ) * dist / el.clientHeight;
				const right = _v.setFromMatrixColumn( camera.matrixWorld, 0 ).clone();
				const up = new THREE.Vector3().setFromMatrixColumn( camera.matrixWorld, 1 );
				const s = Math.min( 1.2, moved * worldPerPx / Math.max( dt, 0.15 ) * 0.08 ) / Math.max( moved, 1e-3 );
				const lat = right.multiplyScalar( dx * s ).addScaledVector( up, - dy * s );
				throwAt( nx, ny, P.projectile, lat );

			} else if ( moved < 6 && dt < 0.4 ) {

				const [ nx, ny ] = ndcOf( e );
				throwAt( nx, ny, P.projectile );

			}

		};

		el.addEventListener( 'pointerdown', onDown );
		addEventListener( 'pointerup', onUp );

		// ───────────────────────── camera ─────────────────────────
		camera.fov = 45; camera.near = 0.05; camera.far = 3000; camera.updateProjectionMatrix();
		controls.target.set( 0, 0, 0 );
		{

			// look perpendicular to the initial separation, from ~38° above the orbital plane
			const sepDir = new THREE.Vector3( - 0.63, 0, 0.78 ).normalize();
			const side = new THREE.Vector3( sepDir.z, 0, - sepDir.x );
			const el0 = THREE.MathUtils.degToRad( 38 ), d0 = 27;
			camera.position.copy( side.multiplyScalar( Math.cos( el0 ) * d0 ) ).add( new THREE.Vector3( 0, Math.sin( el0 ) * d0, 0 ) );

		}

		controls.minDistance = 3;
		controls.maxDistance = 120;
		controls.autoRotate = true;
		controls.autoRotateSpeed = 0.28;
		controls.update();

		// ───────────────────────── GUI ─────────────────────────
		const gui = ctx.gui;
		const restartBtn = { restart: () => restart(), throwDwarf: () => throwAt( ( Math.random() - 0.5 ) * 0.6, ( Math.random() - 0.5 ) * 0.4, P.projectile ) };
		gui.add( restartBtn, 'restart' ).name( '↺ restart encounter' );
		gui.add( P, 'timeScale', 0, 4, 0.05 ).name( 'time scale' );
		gui.add( P, 'massRatio', 0.2, 1, 0.05 ).name( 'mass ratio (B/A)' ).onFinishChange( restart );
		gui.add( P, 'inclination', 0, 180, 1 ).name( 'inclination B (°)' ).onFinishChange( restart );
		gui.add( P, 'pericentre', 1.0, 6.0, 0.1 ).name( 'impact parameter' ).onFinishChange( restart );
		gui.add( P, 'substeps', 1, 8, 1 ).name( 'N-body substeps' );
		gui.add( P, 'showDM' ).name( 'show dark matter' ).onChange( ( v ) => R.dmSprite.visible = v );
		gui.add( P, 'projectile', [ 'dwarf galaxy', 'black hole' ] ).name( 'click throws' );
		gui.add( P, 'starFormation', 0, 3, 0.05 ).name( 'star formation' ).onChange( ( v ) => U.sfEff.value = v );
		gui.add( P, 'dust', 0, 3, 0.05 ).name( 'dust' ).onChange( ( v ) => U.kappa.value = 0.35 * v );
		gui.add( restartBtn, 'throwDwarf' ).name( 'throw projectile' );
		R.dmSprite.visible = P.showDM;

		// ───────────────────────── readback (nuclei + burst counter) ─────────────────────────
		let rbBusy = false, burstsTotal = 0, lastBurstT = 0, lastBursts = 0, sfr = 0, rbFrames = 0;
		async function readback( sync = false ) {

			if ( rbBusy ) return;
			rbBusy = true;
			try {

				const buf = await renderer.getArrayBufferAsync( B.mPos.value, null, 0, 32 );
				const a = new Float32Array( buf );
				if ( Number.isFinite( a[ 0 ] ) ) {

					nuclei.a.set( a[ 0 ], a[ 1 ], a[ 2 ] ); nuclei.b.set( a[ 4 ], a[ 5 ], a[ 6 ] );
					nuclei.mid.addVectors( nuclei.a, nuclei.b ).multiplyScalar( 0.5 );
					nuclei.sep = nuclei.a.distanceTo( nuclei.b );
					if ( sync ) smoothMid.copy( nuclei.mid );

				}

				const sb = new Int32Array( await renderer.getArrayBufferAsync( B.statI.value ) );
				burstsTotal = sb[ 0 ];
				const dT = simTime - lastBurstT;
				if ( dT > 0.5 ) {

					const rate = ( burstsTotal - lastBursts ) / ( dT * MYR_PER_UNIT ); // bursts per Myr
					// each burst ≈ one 10^5.5 M☉ cluster at tier 0 granularity (scaled by gas resolution)
					const mPer = 3.2e5 * ( 2000 / gasPerGal ) * 6;
					const v = Math.max( 0, rate * mPer / 1e6 );
					sfr = sfr === 0 ? v : sfr * 0.6 + v * 0.4;
					lastBurstT = simTime; lastBursts = burstsTotal;

				}

			} catch ( e ) { /* readback can fail during teardown */ }

			rbBusy = false;

		}

		// ───────────────────────── boot ─────────────────────────
		await restart();
		await renderer.computeAsync( sim.shade );

		// debug: pre-advance the simulation (sim time units) before the first frame
		const warp = prm.has( 'warp' ) ? parseFloat( prm.get( 'warp' ) ) : 0;
		const autoThrow = prm.has( 'throw' ) ? parseFloat( prm.get( 'throw' ) ) : - 1;
		const autoBH = prm.has( 'bh' ) ? parseFloat( prm.get( 'bh' ) ) : - 1;
		let thrown = false, thrownBH = false;
		if ( warp > 0 ) {

			const h = 0.05;
			let k = 0;
			while ( simTime < warp - 1e-6 ) {

				if ( autoThrow >= 0 && ! thrown && simTime >= autoThrow ) {

					thrown = true; throwAt( 0.1, 0.05, 'dwarf galaxy' );

				}

				if ( autoBH >= 0 && ! thrownBH && simTime >= autoBH ) {

					thrownBH = true; throwAt( - 0.1, 0.0, 'black hole' );

				}

				step( Math.min( h, warp - simTime ) );
				if ( ++ k % 25 === 0 ) await renderer.computeAsync( sim.zeroStats );

			}

			await readback( true );

		}

		let statT = 0;
		const updateStats = () => {

			ctx.setStats( [
				[ 'stars + gas', T.nt.toLocaleString() ],
				[ 'massive bodies', T.nm.toLocaleString() ],
				[ 'sim time', `${( simTime * MYR_PER_UNIT ).toFixed( 0 )} Myr` ],
				[ 'nuclei sep.', `${( nuclei.sep * KPC_PER_UNIT ).toFixed( 1 )} kpc` ],
				[ 'star formation', `${sfr.toFixed( 1 )} M☉/yr` ],
			] );

		};

		updateStats();

		const updatePx = () => {

			const hPx = Math.max( 1, ctx.size.y * renderer.getPixelRatio() );
			R.RU.pxWorld.value = 2 * Math.tan( THREE.MathUtils.degToRad( camera.fov / 2 ) ) / hPx;

		};

		let bloomNode = null;

		return {
			scene,
			camera,

			post( _ctx, scenePass ) {

				const c = scenePass.getTextureNode( 'output' );
				// Lupton-style colour-preserving asinh stretch: compresses the bulges without bleaching their hue
				const stretched = Fn( () => {

					const rgb = c.rgb.toVar();
					const I = max( dot( rgb, vec3( 0.3, 0.45, 0.25 ) ), 1e-5 ).toVar();
					const beta = float( 2.2 );
					const s = asinh( I.mul( beta ) ).div( I.mul( beta ) );
					const out = rgb.mul( s ).mul( 1.25 ).toVar();
					// mild saturation boost for the faint arms
					const l = dot( out, vec3( 0.3, 0.45, 0.25 ) );
					return vec4( mix( vec3( l ), out, 1.18 ).max( 0 ), 1 );

				} )();
				bloomNode = bloom( stretched, 0.55, 0.55, 0.32 );
				return stretched.add( bloomNode );

			},

			update( dt, t ) {

				if ( restarting ) return;
				frameNo ++;
				updatePx();
				if ( ! P.paused && P.timeScale > 0 ) {

					const dtSim = P.timeScale * UNITS_PER_SEC * Math.min( dt, 1 / 30 );
					step( dtSim );
					if ( autoThrow >= 0 && ! thrown && simTime >= autoThrow ) {

						thrown = true; throwAt( 0.1, 0.05, 'dwarf galaxy' );

					}

					if ( autoBH >= 0 && ! thrownBH && simTime >= autoBH ) {

						thrownBH = true; throwAt( - 0.1, 0.0, 'black hole' );

					}

				}

				// gently follow the merging pair; keep the dust grid centred on the action
				if ( ++ rbFrames % ( ctx.shot ? 10 : 15 ) === 0 ) readback();
				smoothMid.lerp( nuclei.mid, 1 - Math.exp( - dt * 0.6 ) );
				const tgt = _v.copy( smoothMid ).multiplyScalar( 0.85 );
				controls.target.lerp( tgt, 1 - Math.exp( - dt * 0.8 ) );
				U.gridMin.value.copy( smoothMid ).subScalar( box / 2 );

				U.camPos.value.copy( camera.position );
				renderer.compute( sim.shade );

				statT += dt;
				if ( statT > 0.5 ) {

					statT = 0; updateStats();

				}

			},

			resize() {

				updatePx();

			},

			onKey( e ) {

				if ( e.key === 'r' || e.key === 'R' ) restart();
				if ( e.key === ' ' ) {

					P.paused = ! P.paused; e.preventDefault?.();

				}

				if ( e.key === 'b' || e.key === 'B' ) throwAt( ctx.pointer.ndc.x, ctx.pointer.ndc.y, 'black hole' );
				if ( e.key === 'd' || e.key === 'D' ) throwAt( ctx.pointer.ndc.x, ctx.pointer.ndc.y, 'dwarf galaxy' );

			},

			dispose() {

				el.removeEventListener( 'pointerdown', onDown );
				removeEventListener( 'pointerup', onUp );
				controls.enabled = true;
				controls.autoRotate = false;
				for ( const o of R.objects ) scene.remove( o );
				for ( const m of R.materials ) m.dispose();
				for ( const k of sim.kernels ) k.dispose?.();
				bloomNode?.dispose?.();
				// free GPU storage buffers (no public BufferAttribute.dispose() yet)
				const attrs = [ ...Object.values( B ), ...R.skyBuffers ].map( ( n ) => n.value );
				for ( const a of attrs ) {

					try {

						renderer._attributes?.delete( a );

					} catch ( e ) { /* already gone */ }

				}

				lastS = null;

			},
		};

	},
};

export { T_GAS, T_SAT };
