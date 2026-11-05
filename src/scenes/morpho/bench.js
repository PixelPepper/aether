// Dev-only micro benchmark for the morpho kernels (not part of the build). Open /src/scenes/morpho/bench.html?G=48
import * as THREE from 'three/webgpu';
import { createSim } from './sim.js';

const q = new URLSearchParams( location.search );
const G = parseInt( q.get( 'G' ) || '48' );
const out = ( s ) => {

	console.log( s ); document.getElementById( 'o' ).textContent += s + '\n';

};

const renderer = new THREE.WebGPURenderer( { canvas: document.getElementById( 'c' ) } );
await renderer.init();
const dev = renderer.backend.device;
const idle = () => dev.queue.onSubmittedWorkDone();
const sim = createSim( renderer, G );
const T = async ( name, fn ) => {

	const t0 = performance.now(); fn(); await idle(); out( `${name}: ${( performance.now() - t0 ).toFixed( 1 )} ms` );

};

await T( 'seed (compile+run)', () => renderer.compute( sim.seedKernel ) );
await T( 'seed', () => renderer.compute( sim.seedKernel ) );
await T( 'stepAB (compile+run)', () => renderer.compute( sim.stepAB ) );
await T( 'stepBA (compile+run)', () => renderer.compute( sim.stepBA ) );
await T( 'step x2', () => renderer.compute( [ sim.stepAB, sim.stepBA ] ) );
if ( ! q.has( 'quick' ) ) await T( 'step x10', () => renderer.compute( Array( 5 ).fill( 0 ).flatMap( () => [ sim.stepAB, sim.stepBA ] ) ) );
if ( sim.stepNaiveAB ) {

	await T( 'naive (compile+run)', () => renderer.compute( [ sim.stepNaiveAB, sim.stepNaiveBA ] ) );
	await T( 'naive x10', () => renderer.compute( Array( 5 ).fill( 0 ).flatMap( () => [ sim.stepNaiveAB, sim.stepNaiveBA ] ) ) );

}

await T( 'bake (compile+run)', () => renderer.compute( sim.bake ) );
await T( 'occ (compile+run)', () => renderer.compute( sim.occKernel ) );
await T( 'dist (compile+run)', () => renderer.compute( sim.distKernel ) );
await T( 'bake+occ+dist', () => renderer.compute( [ sim.bake, sim.occKernel, sim.distKernel ] ) );
await T( 'pick (compile+run)', () => renderer.compute( sim.pickKernel ) );
await T( 'brush (compile+run)', () => renderer.compute( sim.brushKernel ) );
out( "DONE" );
window.__ready = true;

// correctness: tiled vs untiled must agree
if ( q.has( 'check' ) ) {

	const run = async ( a, b ) => {

		sim.u.seedSalt.value = 3.3; sim.u.seedClear.value = 1; sim.u.seedDensity.value = 1;
		renderer.compute( sim.seedKernel );
		renderer.compute( Array( 4 ).fill( 0 ).flatMap( () => [ a, b ] ) );
		return new Float32Array( await renderer.getArrayBufferAsync( sim.bufA.value ) );

	};

	const r1 = await run( sim.stepAB, sim.stepBA );
	const r2 = await run( sim.stepNaiveAB, sim.stepNaiveBA );
	let md = 0, sum = 0;
	for ( let i = 0; i < r1.length; i ++ ) {

		md = Math.max( md, Math.abs( r1[ i ] - r2[ i ] ) ); sum += r1[ i ];

	}

	out( `tiled vs naive after 8 iters: max |diff| = ${md}, mean = ${( sum / r1.length ).toFixed( 5 )}` );

}

out( 'DONE' );
window.__ready = true;
