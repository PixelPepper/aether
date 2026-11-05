// Headless capture harness. WebGPU runs on SwiftShader (CPU) here, so keep counts modest (tier=0 is auto-selected).
//
//   node tools/shot.mjs <chapterId> [--frames 90] [--w 1280] [--h 720] [--out shots/x.png] [--hud] [--q 1] [--port 5173] [--extra "k=v&k2=v2"]
//
// Starts (or reuses) a Vite dev server, opens ?c=<id>&shot=1, waits for window.__ready (set after N frames),
// prints console errors / page errors, writes a PNG. Exit code 1 on any page error or timeout.
import { createRequire } from 'node:module';
const { chromium } = createRequire( import.meta.url )( 'playwright' );
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import net from 'node:net';
import path from 'node:path';

const args = process.argv.slice( 2 );
const id = args[ 0 ];
if ( ! id ) {

	console.error( 'usage: node tools/shot.mjs <chapterId> [--frames N] [--w W] [--h H] [--out file] [--hud] [--extra qs]' );
	process.exit( 2 );

}

const opt = ( k, d ) => {

	const i = args.indexOf( '--' + k );
	return i >= 0 ? args[ i + 1 ] : d;

};

const frames = opt( 'frames', '90' );
const W = parseInt( opt( 'w', '1280' ) ), H = parseInt( opt( 'h', '720' ) );
const port = parseInt( opt( 'port', '5173' ) );
const out = opt( 'out', `shots/${id}.png` );
const timeout = parseInt( opt( 'timeout', '240' ) ) * 1000;
const extra = opt( 'extra', '' );
const q = opt( 'q', '1' );

const root = path.resolve( path.dirname( new URL( import.meta.url ).pathname ), '..' );

function portOpen( p ) {

	return new Promise( ( res ) => {

		const s = net.connect( p, '127.0.0.1' );
		s.on( 'connect', () => {

			s.end(); res( true );

		} );
		s.on( 'error', () => res( false ) );

	} );

}

// ── capture semaphore: many agents share 4 CPU cores running SwiftShader; allow SLOTS concurrent captures ──
import { mkdirSync as mk, rmSync, readFileSync as rf, writeFileSync as wf } from 'node:fs';
const SLOTS = parseInt( process.env.SHOT_SLOTS || '1000' ); // effectively no queue: every capture runs in parallel
let slot = null;
const waitStart = Date.now();
while ( slot === null ) {

	for ( let i = 0; i < SLOTS && slot === null; i ++ ) {

		const dir = `/tmp/aether-shot-slot-${i}`;
		try {

			mk( dir ); wf( dir + '/pid', String( process.pid ) ); slot = dir;

		} catch {

			// stale lock? (owner process gone)
			try {

				const pid = parseInt( rf( dir + '/pid', 'utf8' ) );
				try {

					process.kill( pid, 0 );

				} catch {

					rmSync( dir, { recursive: true, force: true } );

				}

			} catch {}

		}

	}

	if ( slot === null ) {

		if ( Date.now() - waitStart > 15 * 60 * 1000 ) {

			console.log( '[slot] waited 15 min for a capture slot, giving up' ); process.exit( 3 );

		}

		await new Promise( ( r ) => setTimeout( r, 1500 ) );

	}

}

if ( Date.now() - waitStart > 3000 ) console.log( `[slot] waited ${( ( Date.now() - waitStart ) / 1000 ).toFixed( 0 )}s for a capture slot` );
const releaseSlot = () => {

	try {

		rmSync( slot, { recursive: true, force: true } );

	} catch {}

};

process.on( 'exit', releaseSlot );
process.on( 'SIGINT', () => process.exit( 130 ) );
process.on( 'SIGTERM', () => process.exit( 143 ) );

let server = null;
if ( ! ( await portOpen( port ) ) ) {

	server = spawn( 'npx', [ 'vite', '--port', String( port ), '--strictPort' ], { cwd: root, stdio: 'ignore' } );
	for ( let i = 0; i < 100 && ! ( await portOpen( port ) ); i ++ ) await new Promise( ( r ) => setTimeout( r, 200 ) );

}

const browser = await chromium.launch( { channel: process.env.SHOT_CHANNEL || 'chromium', args: [ '--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan', '--ignore-gpu-blocklist' ] } );
const page = await browser.newPage( { viewport: { width: W, height: H }, deviceScaleFactor: 1 } );

let bad = false;
page.on( 'console', ( m ) => {

	const t = m.type();
	if ( t === 'error' || t === 'warning' ) {

		console.log( `[${t}] ${m.text()}` );
		if ( t === 'error' ) bad = true;

	}

} );
page.on( 'pageerror', ( e ) => {

	console.log( '[pageerror]', e.message, e.stack?.split( '\n' ).slice( 0, 4 ).join( ' | ' ) );
	bad = true;

} );

const url = `http://127.0.0.1:${port}/?c=${id}&shot=1&frames=${frames}&q=${q}${args.includes( '--hud' ) ? '&hud=1' : ''}${extra ? '&' + extra : ''}`;
console.log( 'open', url );
const t0 = Date.now();
await page.goto( url );
let ok = true;
try {

	await page.waitForFunction( () => window.__ready || window.__error, null, { timeout, polling: 500 } );
	const err = await page.evaluate( () => window.__error );
	if ( err ) {

		console.log( '[chapter error]', err ); ok = false;

	}

} catch ( e ) {

	console.log( '[timeout] waiting for __ready after', ( Date.now() - t0 ) / 1000, 's' );
	ok = false;

}

mkdirSync( path.dirname( path.resolve( root, out ) ), { recursive: true } );
await page.screenshot( { path: path.resolve( root, out ) } );
console.log( `wrote ${out} in ${( ( Date.now() - t0 ) / 1000 ).toFixed( 1 )}s`, ok && ! bad ? 'OK' : 'WITH ERRORS' );
await browser.close();
server?.kill();
process.exit( ok && ! bad ? 0 : 1 );
