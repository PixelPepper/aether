// Chapter 05 — Ignition.
// A buoyant-plasma Eulerian fluid (MacCormack advection, vorticity confinement, Jacobi pressure projection) in a
// 3D storage-texture grid sitting on the limb of a young star, guided by an analytic bipolar active-region field,
// raymarched as a blackbody emitter/absorber and composited over an analytic limb-darkened photosphere.
import * as THREE from 'three/webgpu';
import { vec4 } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { createFluid } from './stellar/fluid.js';
import { createRender } from './stellar/render.js';
import { createBlackbodyLUT, makeBlackbody } from './stellar/blackbody.js';

const TIERS = {
	0: { NX: 40, NY: 30, NZ: 30, jacobi: 16, quality: 'low', prewarm: 36 },
	1: { NX: 96, NY: 72, NZ: 72, jacobi: 24, quality: 'medium', prewarm: 60 },
	2: { NX: 128, NY: 96, NZ: 96, jacobi: 32, quality: 'high', prewarm: 90 },
};

const QUALITY = {
	low: { steps: 40, scale: 0.5 },
	medium: { steps: 72, scale: 0.5 },
	high: { steps: 112, scale: 0.6 },
	ultra: { steps: 168, scale: 0.8 },
};

const STAR_R = 60;
const BOX_SIZE = new THREE.Vector3( 8, 6, 6 );
const BOX_MIN = new THREE.Vector3( - 4, - 0.3, - 3 );
const FLARE_PERIOD = 19;

export default {
	id: 'stellar',
	title: 'Ignition',
	subtitle: 'Buoyant plasma on a 3D Eulerian grid rises from the granulating photosphere of a young star, threads its magnetic loops and erupts — every voxel glowing as a Planck blackbody.',

	async create( ctx ) {

		const { renderer, camera, controls, pointer, tier } = ctx;
		const T = TIERS[ Math.max( 0, Math.min( 2, tier ) ) ] || TIERS[ 2 ];
		const scene = new THREE.Scene();

		// ── camera: we drive it (dragging stirs the star instead of orbiting) ──
		controls.enabled = false;
		camera.fov = 40;
		camera.near = 0.05;
		camera.far = 400;
		camera.updateProjectionMatrix();

		// ── simulation ──
		const fluid = createFluid( renderer, {
			NX: T.NX, NY: T.NY, NZ: T.NZ, jacobi: T.jacobi, origin: BOX_MIN, size: BOX_SIZE, starR: STAR_R,
		} );

		const lut = createBlackbodyLUT();
		const blackbody = makeBlackbody( lut );
		const qualityState = { quality: T.quality };
		const render = createRender( ctx, fluid, {
			origin: BOX_MIN, size: BOX_SIZE, starR: STAR_R, blackbody, steps: QUALITY[ T.quality ].steps,
		} );
		scene.backgroundNode = render.skyNode;

		// ── state ──
		let simTime = 0;
		let flareClock = FLARE_PERIOD * 0.52; // first eruption a few seconds in
		let flash = 0;
		let ptrAmt = 0;
		let statsTimer = 0;
		let peakK = 0;
		const ptrPrev = new THREE.Vector3();
		let ptrHas = false;

		const FU = fluid.U, RU = render.U;
		const params = {
			speed: 1.0,
			erupt: () => {

				flareClock = FLARE_PERIOD * 0.64;

			},
		};

		const flareEnvelope = ( clk ) => {

			const ph = ( clk % FLARE_PERIOD ) / FLARE_PERIOD;
			const ss = ( a, b, x ) => {

				const t = Math.min( 1, Math.max( 0, ( x - a ) / ( b - a ) ) );
				return t * t * ( 3 - 2 * t );

			};

			return ss( 0.62, 0.7, ph ) * ss( 0.97, 0.8, ph );

		};

		const kelvinOf = ( tn ) => RU.T0.value + RU.Tk.value * tn;

		// ── camera drift: low along the limb, looking up at the arch ──
		const target = new THREE.Vector3();
		const placeCamera = ( t ) => {

			const az = 0.28 * Math.sin( t * 0.043 ) + 0.1 * Math.sin( t * 0.017 + 1.3 );
			const dist = 10.6 + 0.9 * Math.sin( t * 0.031 + 0.4 );
			const hgt = 0.62 + 0.22 * Math.sin( t * 0.057 + 1.0 );
			camera.position.set( Math.sin( az ) * dist, hgt, Math.cos( az ) * dist );
			target.set( Math.sin( az ) * 0.6, 2.25 + 0.2 * Math.sin( t * 0.05 ), 0 );
			camera.up.set( Math.sin( t * 0.021 ) * 0.035, 1, 0 ).normalize();
			camera.lookAt( target );
			camera.updateMatrixWorld();

		};

		const setFlareUniforms = ( clk ) => {

			const e = flareEnvelope( clk );
			FU.erupt.value = e;
			FU.heatL.value = 1.0 + 0.25 * Math.sin( clk * 0.37 );
			FU.heatR.value = 0.45 + 0.15 * Math.sin( clk * 0.23 + 2 );
			return e;

		};

		const dbg = ctx.params.has( 'sdbg' ) ? ( ...a ) => console.warn( '[stellar]', ( performance.now() / 1000 ).toFixed( 2 ), ...a ) : () => {};
		dbg( 'create start' );

		// ── pre-warm: a few dozen steps so the very first frame is already a living loop ──
		fluid.init();
		if ( ctx.params.has( 'sdbg' ) ) await fluid.sync();
		dbg( 'init done' );
		const warmDt = 1 / 40;
		const prewarm = ctx.params.has( 'spw' ) ? parseInt( ctx.params.get( 'spw' ) ) : T.prewarm;
		for ( let i = 0; i < prewarm; i ++ ) {

			flareClock += warmDt;
			simTime += warmDt;
			FU.time.value = simTime;
			setFlareUniforms( flareClock );
			fluid.step( warmDt );
			if ( i === 0 ) {

				if ( ctx.params.has( 'sdbg' ) ) await fluid.sync();
				dbg( 'first step done' );

			}

		}

		if ( ctx.params.has( 'sdbg' ) ) await fluid.sync();
		dbg( 'prewarm done', prewarm );

		// ── GUI ──
		const gui = ctx.gui;
		gui.add( FU.buoy, 'value', 0, 4, 0.01 ).name( 'buoyancy' );
		gui.add( FU.vort, 'value', 0, 1.5, 0.01 ).name( 'vorticity' );
		gui.add( FU.cool, 'value', 0, 1.5, 0.01 ).name( 'cooling rate' );
		gui.add( FU.guide, 'value', 0, 20, 0.1 ).name( 'magnetic guide' );
		gui.add( RU.exposure, 'value', 0.1, 4, 0.01 ).name( 'emission exposure' );
		gui.add( qualityState, 'quality', Object.keys( QUALITY ) ).name( 'quality (steps/res)' ).onChange( ( q ) => {

			RU.steps.value = QUALITY[ q ].steps;
			resize( ctx.size.x, ctx.size.y, renderer.getPixelRatio() );

		} );
		gui.add( params, 'erupt' ).name( 'trigger eruption' );

		// ── sizing ──
		function resize( w, h, dpr ) {

			const s = QUALITY[ qualityState.quality ].scale;
			render.setSize( w * dpr * s, h * dpr * s );

		}

		resize( ctx.size.x, ctx.size.y, renderer.getPixelRatio() );

		// ── pointer → world: intersect the pointer ray with the vertical plane through the arch facing the camera ──
		const plane = new THREE.Plane();
		const boxCenter = new THREE.Vector3().copy( BOX_MIN ).addScaledVector( BOX_SIZE, 0.5 );
		const hit = new THREE.Vector3();
		const nrm = new THREE.Vector3();
		const right = new THREE.Vector3(), up = new THREE.Vector3();
		const inMin = BOX_MIN.clone().addScalar( 0.35 );
		const inMax = BOX_MIN.clone().add( BOX_SIZE ).subScalar( 0.35 );
		inMin.y = 0.15;

		const updatePointer = ( dt ) => {

			let active = false;
			if ( pointer.down && pointer.button === 0 ) {

				nrm.subVectors( camera.position, boxCenter ).setY( 0 ).normalize();
				plane.setFromNormalAndCoplanarPoint( nrm, boxCenter );
				if ( pointer.ray.intersectPlane( plane, hit ) ) {

					const inside = hit.x > BOX_MIN.x - 1 && hit.x < BOX_MIN.x + BOX_SIZE.x + 1 && hit.y > BOX_MIN.y - 1 && hit.y < BOX_MIN.y + BOX_SIZE.y + 0.5;
					if ( inside ) {

						hit.clamp( inMin, inMax );
						active = true;
						// momentum from the pointer's screen-space velocity, scaled to world units at the hit depth
						const d = camera.position.distanceTo( hit );
						const hh = d * Math.tan( THREE.MathUtils.degToRad( camera.fov * 0.5 ) );
						right.setFromMatrixColumn( camera.matrixWorld, 0 );
						up.setFromMatrixColumn( camera.matrixWorld, 1 );
						const vel = FU.ptrVel.value;
						vel.set( 0, 0, 0 )
							.addScaledVector( right, pointer.velocity.x * hh * camera.aspect )
							.addScaledVector( up, pointer.velocity.y * hh );
						if ( vel.length() > 9 ) vel.setLength( 9 );
						vel.y += 1.2; // freshly heated plasma wants to rise
						FU.ptrPos.value.copy( hit );
						if ( ! ptrHas ) flash = Math.min( 1.5, flash + 0.6 );
						ptrHas = true;
						ptrPrev.copy( hit );

					}

				}

			}

			if ( ! active ) ptrHas = false;
			ptrAmt += ( ( active ? 1 : 0 ) - ptrAmt ) * Math.min( 1, dt * 10 );
			FU.ptrAmt.value = ptrAmt;

		};

		placeCamera( 0 );
		render.updateCamera( camera );

		// ── post: composite volume over sky/limb, bloom ──
		let bloomNode = null;
		let frameNo = 0;

		return {
			scene,
			post( _ctx, scenePass ) {

				const sc = scenePass.getTextureNode( 'output' );
				const comp = vec4( render.composite( sc ), 1 );
				bloomNode = bloom( comp, 0.5, 0.55, 0.55 );
				return vec4( comp.rgb.add( bloomNode.rgb ), 1 );

			},
			update( dt, t ) {

				const sdt = Math.min( dt, 1 / 30 ) * params.speed;
				simTime += sdt;
				flareClock += sdt;
				FU.time.value = simTime;
				RU.simTime.value = simTime;
				const e = setFlareUniforms( flareClock );

				updatePointer( dt );
				flash *= Math.exp( - dt * 2.5 );
				RU.flash.value = flash * 0.4 + e * 0.15;

				fluid.step( sdt );

				placeCamera( t );
				render.updateCamera( camera );
				render.renderVolume();
				if ( frameNo ++ < 4 ) dbg( 'frame', frameNo );

				statsTimer -= dt;
				if ( statsTimer <= 0 ) {

					statsTimer = 0.5;
					fluid.samplePeak( ( tn ) => {

						peakK = Math.min( 21000, kelvinOf( tn ) );

					} );
					ctx.setStats( [
						[ 'grid', `${T.NX}×${T.NY}×${T.NZ}` ],
						[ 'pressure', `${fluid.jacobi} Jacobi it.` ],
						[ 'peak T', peakK ? `${Math.round( peakK / 10 ) * 10} K`.replace( /\B(?=(\d{3})+(?!\d))/g, ',' ) : '—' ],
						[ 'raymarch', `${RU.steps.value} st · ${render.size.join( '×' )}` ],
					] );

				}

			},
			resize,
			onKey( e ) {

				if ( e.key === 'f' || e.key === 'F' ) params.erupt();

			},
			dispose() {

				fluid.dispose();
				render.dispose();
				lut.dispose();
				bloomNode?.dispose?.();
				scene.backgroundNode = null;
				controls.enabled = true;

			},
		};

	},
};
