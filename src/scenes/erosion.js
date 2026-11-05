// Chapter 07 — Deep Time.
// A continent with a rising mountain spine is carved by rain into river networks, fans and deltas, live on the GPU.
// Simulation: ./erosion/sim.js (virtual-pipe shallow water + sediment transport + thermal + uplift).
// Rendering: ./erosion/render.js (displaced heightfield, material blending, water surface, sky, fog, storm cell).

import * as THREE from 'three/webgpu';
import { ErosionSim } from './erosion/sim.js';
import { makeRenderUniforms, makeTerrain, makeWater, makeOcean, makeSky, makeStorm } from './erosion/render.js';

const SIZE = 10; // world extent of the simulated square

function formatYears( y ) {

	if ( y >= 1e6 ) return ( y / 1e6 ).toFixed( 2 ) + ' Myr';
	if ( y >= 1e4 ) return Math.round( y / 1e3 ).toLocaleString( 'en-US' ) + ' kyr';
	return Math.round( y ).toLocaleString( 'en-US' ) + ' yr';

}

export default {
	id: 'erosion',
	title: 'Deep Time',
	subtitle: 'Rain carves a rising mountain range into rivers, fans and deltas: shallow-water pipe flow, sediment transport, talus collapse and tectonic uplift, a million years every few seconds.',

	async create( ctx ) {

		const { renderer, camera, controls, tier, pointer } = ctx;
		const N = tier >= 2 ? 1024 : tier === 1 ? 512 : 256;
		const sim = new ErosionSim( renderer, { N, size: SIZE, tier } );
		const U = makeRenderUniforms( sim );
		// a simulation tick spans more time on a coarser grid (cell size sets the flow time step)
		const yearsPerTick = 420 * ( 1024 / N );

		// ── scene ─────────────────────────────────────────────
		const scene = new THREE.Scene();
		scene.backgroundNode = makeSky( U );

		const geometry = new THREE.PlaneGeometry( SIZE, SIZE, N - 1, N - 1 );
		geometry.rotateX( - Math.PI / 2 );
		const terrain = makeTerrain( sim, U, geometry );
		const water = makeWater( sim, U, geometry );
		const ocean = makeOcean( sim, U );
		const storm = makeStorm( U, tier );
		scene.add( terrain, water, ocean, storm );

		// ── initial landscape: generate, then pre-run erosion so rivers already exist on frame 1 ──
		let seed = parseFloat( ctx.params.get( 'seed' ) || '1' );
		const prerunTicks = tier >= 2 ? 3200 : tier === 1 ? 1600 : 360;
		const prerunBatch = tier >= 2 ? 160 : tier === 1 ? 80 : 60;
		const dbg = ctx.params.has( 'dbg' );
		const T0 = performance.now();
		const log = ( m ) => dbg && console.warn( `[erosion] ${m} @ ${( ( performance.now() - T0 ) / 1000 ).toFixed( 2 )}s` );
		await sim.init( seed );
		log( 'init' );
		await sim.prerun( parseInt( ctx.params.get( 'prerun' ) ?? prerunTicks ), prerunBatch );
		log( 'prerun' );
		let low = await sim.readLow();
		log( 'readback' );

		// ── camera: low aerial, sun grazing from the left, slow orbit ──
		const saved = {
			enablePan: controls.enablePan, minDistance: controls.minDistance, maxDistance: controls.maxDistance,
			minPolarAngle: controls.minPolarAngle, maxPolarAngle: controls.maxPolarAngle, autoRotateSpeed: controls.autoRotateSpeed,
		};
		camera.fov = 38; camera.near = 0.03; camera.far = 1500; camera.updateProjectionMatrix();
		camera.position.set( 5.6, 2.15, 7.4 );
		controls.target.set( 0.2, 0.25, 0.1 );
		controls.minDistance = 2.5;
		controls.maxDistance = 22;
		controls.minPolarAngle = 0.25;
		controls.maxPolarAngle = 1.42;
		controls.enablePan = false;
		controls.autoRotate = true;
		controls.autoRotateSpeed = 0.28;
		controls.update();

		// ── GUI ─────────────────────────────────────────────
		const su = sim.u;
		const params = {
			iters: tier >= 2 ? 10 : tier === 1 ? 6 : 3,
			talus: 48,
			regenerate: () => regenerate(),
		};
		const gui = ctx.gui;
		gui.add( su.rain, 'value', 0, 0.004, 0.0001 ).name( 'rain rate' );
		gui.add( su.Kc, 'value', 0.1, 3, 0.01 ).name( 'sediment capacity' );
		gui.add( su.Ks, 'value', 0.002, 0.15, 0.001 ).name( 'erosion rate' );
		gui.add( su.Kd, 'value', 0.005, 0.3, 0.001 ).name( 'deposition rate' );
		gui.add( su.uplift, 'value', 0, 0.015, 0.0001 ).name( 'uplift rate' );
		gui.add( params, 'talus', 25, 70, 1 ).name( 'talus angle °' ).onChange( ( v ) => su.talus.value = Math.tan( v * Math.PI / 180 ) );
		gui.add( params, 'iters', 1, tier >= 2 ? 40 : 16, 1 ).name( 'iterations / frame' );
		gui.add( params, 'regenerate' ).name( 'regenerate terrain' );

		// ── pointer: drag on land = storm, right-drag / shift-drag = tectonic push, drag on sky = orbit ──
		const canvas = renderer.domElement;
		const ray = new THREE.Raycaster();
		const ndc = new THREE.Vector2();
		let mode = null;

		const heightAt = ( x, z ) => {

			const L = sim.lowN;
			const fx = THREE.MathUtils.clamp( ( x / SIZE + 0.5 ) * L - 0.5, 0, L - 1 );
			const fz = THREE.MathUtils.clamp( ( z / SIZE + 0.5 ) * L - 0.5, 0, L - 1 );
			const x0 = Math.floor( fx ), z0 = Math.floor( fz );
			const x1 = Math.min( x0 + 1, L - 1 ), z1 = Math.min( z0 + 1, L - 1 );
			const tx = fx - x0, tz = fz - z0;
			const a = low[ z0 * L + x0 ] * ( 1 - tx ) + low[ z0 * L + x1 ] * tx;
			const b = low[ z1 * L + x0 ] * ( 1 - tx ) + low[ z1 * L + x1 ] * tx;
			return a * ( 1 - tz ) + b * tz;

		};

		let maxH = 1.5;
		const refreshMax = () => {

			maxH = 0;
			for ( let i = 0; i < low.length; i ++ ) maxH = Math.max( maxH, low[ i ] );

		};

		refreshMax();

		// march the view ray over the CPU copy of the heightfield
		const hitP = new THREE.Vector3();
		const pick = ( r ) => {

			const o = r.origin, d = r.direction;
			const half = SIZE / 2;
			let t0 = 0, t1 = 60;
			for ( const [ oc, dc ] of [ [ o.x, d.x ], [ o.z, d.z ] ] ) {

				if ( Math.abs( dc ) < 1e-6 ) {

					if ( oc < - half || oc > half ) return null;
					continue;

				}

				let a = ( - half - oc ) / dc, b = ( half - oc ) / dc;
				if ( a > b ) [ a, b ] = [ b, a ];
				t0 = Math.max( t0, a ); t1 = Math.min( t1, b );

			}

			if ( t0 >= t1 ) return null;
			const step = SIZE / sim.lowN * 0.5;
			let prev = t0;
			for ( let t = t0; t <= t1; t += step ) {

				const y = o.y + d.y * t;
				if ( y < heightAt( o.x + d.x * t, o.z + d.z * t ) ) {

					let lo = prev, hi = t;
					for ( let k = 0; k < 10; k ++ ) {

						const m = ( lo + hi ) * 0.5;
						if ( o.y + d.y * m < heightAt( o.x + d.x * m, o.z + d.z * m ) ) hi = m; else lo = m;

					}

					return hitP.set( o.x + d.x * hi, o.y + d.y * hi, o.z + d.z * hi );

				}

				prev = t;
				if ( y > maxH + 0.5 && d.y > 0 ) break;

			}

			return null;

		};

		const onDown = ( e ) => {

			ndc.set( ( e.clientX / innerWidth ) * 2 - 1, - ( e.clientY / innerHeight ) * 2 + 1 );
			ray.setFromCamera( ndc, camera );
			if ( ! pick( ray.ray ) ) return; // sky: let OrbitControls orbit
			if ( e.button === 2 || ( e.button === 0 && ( e.shiftKey || e.altKey || e.ctrlKey || e.metaKey ) ) ) mode = 'raise';
			else if ( e.button === 0 ) mode = 'rain';
			if ( mode ) controls.enabled = false;

		};

		const onUp = () => {

			if ( mode ) controls.enabled = true;
			mode = null;

		};

		const onMenu = ( e ) => e.preventDefault();
		canvas.addEventListener( 'pointerdown', onDown, { capture: true } );
		addEventListener( 'pointerup', onUp );
		canvas.addEventListener( 'contextmenu', onMenu );

		// ── storm state ─────────────────────────────────────
		const stormPos = new THREE.Vector2( - 0.6, - 0.2 );
		const stormTarget = new THREE.Vector2();
		let stormInt = 0.35, rainVis = 0.4;
		const pushP = new THREE.Vector2();

		// ── regenerate ──────────────────────────────────────
		let boost = 0, busy = false;
		async function regenerate() {

			if ( busy ) return;
			busy = true;
			seed += 1;
			await sim.init( seed );
			boost = prerunTicks;
			low = await sim.readLow();
			refreshMax();
			busy = false;

		}

		// ── stats ───────────────────────────────────────────
		let statT = 1, readT = 0, reading = false, frameN = 0;
		const pushStats = () => ctx.setStats( [
			[ 'grid', `${N}² · ${( N * N / 1e6 ).toFixed( 2 )}M cells` ],
			[ 'iterations / frame', String( params.iters + ( boost > 0 ? Math.min( boost, params.iters * 4 ) : 0 ) ) ],
			[ 'simulated', formatYears( sim.ticks * yearsPerTick ) ],
		] );

		return {
			scene,
			bloom: { strength: 0.32, radius: 0.55, threshold: 0.85 },

			update( dt, t ) {

				U.time.value = t;
				if ( dbg ) {

					frameN ++;
					log( `frame ${frameN}` );

				}

				// storm: follows the pointer while raining, otherwise wanders over the range
				let hit = null;
				if ( mode && pointer.inside ) hit = pick( pointer.ray );
				if ( mode === 'rain' && hit ) {

					stormTarget.set( hit.x, hit.z );
					stormInt += ( 1 - stormInt ) * Math.min( 1, dt * 3 );
					rainVis += ( 1 - rainVis ) * Math.min( 1, dt * 3 );

				} else {

					stormTarget.set( 2.1 * Math.sin( t * 0.045 + 0.3 ), 1.5 * Math.sin( t * 0.071 + 2.0 ) );
					stormInt += ( 0.32 - stormInt ) * Math.min( 1, dt * 0.8 );
					rainVis += ( 0.45 - rainVis ) * Math.min( 1, dt * 0.8 );

				}

				const k = mode === 'rain' ? 1 - Math.exp( - dt * 6 ) : 1 - Math.exp( - dt * 0.6 );
				stormPos.lerp( stormTarget, k );
				const cloudY = Math.max( maxH, heightAt( stormPos.x, stormPos.y ) ) + 0.55;
				U.storm.value.set( stormPos.x, cloudY, stormPos.y );
				U.stormVis.value = Math.min( 1, stormInt * 1.6 );
				U.stormRain.value = rainVis;
				su.storm.value.set( ( stormPos.x / SIZE + 0.5 ) * N, ( stormPos.y / SIZE + 0.5 ) * N, 0.018 * stormInt );

				let iters = params.iters;
				if ( boost > 0 ) {

					const extra = Math.min( boost, iters * 4 );
					boost -= extra;
					iters += extra;

				}

				// tectonic push under the pointer (world rate ≈ 0.35 units / s at the centre)
				if ( mode === 'raise' && hit ) {

					pushP.set( ( hit.x / SIZE + 0.5 ) * N, ( hit.z / SIZE + 0.5 ) * N );
					su.push.value.set( pushP.x, pushP.y, 0.35 * dt / sim.cellW / iters );

				} else su.push.value.z = 0;

				if ( ! busy ) sim.step( iters, true );

				// refresh the CPU height copy now and then (pointer picking, cloud altitude)
				readT += dt;
				if ( ! ctx.shot && ! reading && ! busy && readT > 0.35 ) {

					readT = 0; reading = true;
					sim.readLow().then( ( a ) => {

						low = a; refreshMax();

					} ).catch( () => {} ).finally( () => reading = false );

				}

				statT += dt;
				if ( statT > 0.5 ) {

					statT = 0; pushStats();

				}

			},

			onKey( e ) {

				if ( e.key === 'r' || e.key === 'R' ) regenerate();

			},

			dispose() {

				canvas.removeEventListener( 'pointerdown', onDown, { capture: true } );
				removeEventListener( 'pointerup', onUp );
				canvas.removeEventListener( 'contextmenu', onMenu );
				Object.assign( controls, saved );
				controls.enabled = true;
				geometry.dispose();
				ocean.geometry.dispose();
				terrain.material.dispose();
				water.material.dispose();
				ocean.material.dispose();
				storm.userData.dispose();
				scene.backgroundNode = null;
				sim.dispose();

			},
		};

	},
};
