// Real-GPU benchmark. Run on a machine with a real GPU and Chrome (or Edge) installed:
//
//   npm install                     (once)
//   npm i --no-save playwright      (once; no browser download needed, it drives the installed Chrome)
//   node tools/bench.mjs [--tiers 2] [--w 1920] [--h 1080] [--warm 5] [--measure 8] [--only a,b] [--browser chrome|msedge]
//
// For each chapter and tier it opens the app in a visible browser window (headed, so the real GPU and
// swap chain are used), with vsync and the frame-rate limiter disabled, waits `warm` seconds, then measures
// for `measure` seconds: mean fps, 1% low fps, GPU ms (timestamp queries), console errors. It saves a
// screenshot per chapter, then cycles through every chapter twice to catch dispose/leak errors.
// Output: bench/results.json, bench/REPORT.md, bench/<id>-t<tier>.png
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import os from 'node:os';

const require = createRequire( import.meta.url );
const { chromium } = require( 'playwright' );

const args = process.argv.slice( 2 );
const opt = ( k, d ) => {

	const i = args.indexOf( '--' + k );
	return i >= 0 ? args[ i + 1 ] : d;

};

const W = parseInt( opt( 'w', '1920' ) ), H = parseInt( opt( 'h', '1080' ) );
const tiers = opt( 'tiers', '2' ).split( ',' ).map( Number );
const WARM = parseFloat( opt( 'warm', '5' ) ) * 1000;
const MEASURE = parseFloat( opt( 'measure', '8' ) ) * 1000;
const browserChannel = opt( 'browser', 'chrome' );
const port = parseInt( opt( 'port', '5199' ) );
const root = path.resolve( path.dirname( new URL( import.meta.url ).pathname.replace( /^\/([A-Za-z]:)/, '$1' ) ), '..' );
const outDir = path.join( root, 'bench' );
mkdirSync( outDir, { recursive: true } );

// chapter ids, in order, from the registry
const registry = readFileSync( path.join( root, 'src/scenes/index.js' ), 'utf8' );
let ids = [ ...registry.matchAll( /id:\s*'([^']+)'/g ) ].map( ( m ) => m[ 1 ] );
if ( opt( 'only' ) ) ids = opt( 'only' ).split( ',' );

function portOpen( p ) {

	return new Promise( ( res ) => {

		const s = net.connect( p, '127.0.0.1' );
		s.on( 'connect', () => {

			s.end(); res( true );

		} );
		s.on( 'error', () => res( false ) );

	} );

}

let server = null;
if ( ! ( await portOpen( port ) ) ) {

	server = spawn( process.platform === 'win32' ? 'npx.cmd' : 'npx', [ 'vite', '--port', String( port ), '--strictPort', '--host', '127.0.0.1' ], { cwd: root, stdio: 'ignore', shell: process.platform === 'win32' } );
	for ( let i = 0; i < 150 && ! ( await portOpen( port ) ); i ++ ) await new Promise( ( r ) => setTimeout( r, 200 ) );

}

const browser = await chromium.launch( {
	channel: browserChannel,
	headless: false,
	args: [ '--disable-gpu-vsync', '--disable-frame-rate-limit', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', `--window-size=${W},${H + 120}` ],
} );
const page = await browser.newPage( { viewport: { width: W, height: H }, deviceScaleFactor: 1 } );

let errors = [];
page.on( 'console', ( m ) => {

	if ( m.type() === 'error' ) errors.push( m.text().slice( 0, 400 ) );

} );
page.on( 'pageerror', ( e ) => errors.push( ( 'pageerror: ' + e.message ).slice( 0, 400 ) ) );

const sleep = ( ms ) => new Promise( ( r ) => setTimeout( r, ms ) );
const results = [];
let adapter = null;

for ( const tier of tiers ) {

	for ( const id of ids ) {

		errors = [];
		const url = `http://127.0.0.1:${port}/?c=${id}&tier=${tier}&q=1`;
		process.stdout.write( `${id} tier ${tier} … ` );
		const row = { id, tier, url };
		try {

			await page.goto( url, { waitUntil: 'load' } );
			await page.waitForFunction( () => window.__aether && window.__aether.current && ! window.__aether.switching, null, { timeout: 120000 } );
			await page.evaluate( () => document.getElementById( 'enter-silent' )?.click() );
			adapter = adapter || await page.evaluate( () => window.__aether.adapter );
			await sleep( WARM );
			await page.evaluate( () => window.__aether.perf.reset() );
			await sleep( MEASURE );
			const m = await page.evaluate( () => ( { t: window.__aether.perf.recent(), gpu: window.__aether.gpuMs, stats: document.getElementById( 'st-extra' ).innerText.replace( /\n/g, ' ' ) } ) );
			const t = m.t.filter( ( x ) => x > 0 ).sort( ( a, b ) => a - b );
			const mean = t.reduce( ( a, b ) => a + b, 0 ) / Math.max( 1, t.length );
			const p99 = t[ Math.floor( t.length * 0.99 ) ] || mean;
			Object.assign( row, {
				fps: +( 1000 / mean ).toFixed( 1 ),
				low1: +( 1000 / p99 ).toFixed( 1 ),
				frameMs: +mean.toFixed( 2 ),
				gpuMs: m.gpu !== null ? +m.gpu.toFixed( 2 ) : null,
				frames: t.length,
				stats: m.stats,
			} );
			await page.screenshot( { path: path.join( outDir, `${id}-t${tier}.png` ) } );

		} catch ( e ) {

			row.fail = String( e.message ).slice( 0, 300 );

		}

		row.errors = errors.slice( 0, 8 );
		results.push( row );
		console.log( row.fail ? `FAIL ${row.fail}` : `${row.fps} fps (1% low ${row.low1}), gpu ${row.gpuMs} ms${row.errors.length ? `, ${row.errors.length} errors` : ''}` );

	}

}

// chapter cycling: every chapter twice via the in-app switcher, checking for errors and heap growth
errors = [];
let cycle = {};
try {

	await page.goto( `http://127.0.0.1:${port}/?c=${ids[ 0 ]}&tier=${tiers[ 0 ]}&q=1` );
	await page.waitForFunction( () => window.__aether && window.__aether.current, null, { timeout: 120000 } );
	await page.evaluate( () => document.getElementById( 'enter-silent' )?.click() );
	const heap0 = await page.evaluate( () => performance.memory?.usedJSHeapSize || 0 );
	const all = await page.evaluate( () => window.__aether.chapters );
	for ( let round = 0; round < 2; round ++ ) {

		for ( let i = 0; i < all.length; i ++ ) {

			await page.evaluate( ( k ) => window.__aether.goTo( k ), ( i + 1 ) % all.length );
			await page.waitForFunction( ( k ) => window.__aether.index === k && ! window.__aether.switching, ( i + 1 ) % all.length, { timeout: 120000 } );
			await sleep( 1500 );

		}

	}

	const heap1 = await page.evaluate( () => performance.memory?.usedJSHeapSize || 0 );
	cycle = { switches: all.length * 2, heapStartMB: +( heap0 / 1e6 ).toFixed( 1 ), heapEndMB: +( heap1 / 1e6 ).toFixed( 1 ), errors: errors.slice( 0, 12 ) };

} catch ( e ) {

	cycle = { fail: String( e.message ).slice( 0, 300 ), errors: errors.slice( 0, 12 ) };

}

console.log( 'cycle', JSON.stringify( cycle ) );

const meta = { date: new Date().toISOString(), host: os.hostname(), cpu: os.cpus()[ 0 ]?.model, adapter, browser: browserChannel, viewport: `${W}x${H}`, warmS: WARM / 1000, measureS: MEASURE / 1000 };
writeFileSync( path.join( outDir, 'results.json' ), JSON.stringify( { meta, results, cycle }, null, 2 ) );

const md = [
	`# AETHER real-GPU benchmark`,
	``,
	`${meta.date} · ${meta.host} · ${meta.cpu} · GPU: ${adapter?.vendor} ${adapter?.architecture} ${adapter?.description || ''} · ${meta.browser} · ${meta.viewport}, pixel ratio 1, vsync off`,
	``,
	`| chapter | tier | fps | 1% low | GPU ms | errors | stats |`,
	`|---|---|---|---|---|---|---|`,
	...results.map( ( r ) => `| ${r.id} | ${r.tier} | ${r.fail ? 'FAIL' : r.fps} | ${r.low1 ?? ''} | ${r.gpuMs ?? ''} | ${r.fail ? r.fail.replace( /\|/g, '/' ) : r.errors.length ? r.errors.map( ( e ) => e.replace( /\|/g, '/' ) ).join( '<br>' ) : ''} | ${r.stats || ''} |` ),
	``,
	`Chapter cycling: ${JSON.stringify( cycle )}`,
].join( '\n' );
writeFileSync( path.join( outDir, 'REPORT.md' ), md + '\n' );
console.log( '\n' + md );

await browser.close();
server?.kill();
process.exit( 0 );
