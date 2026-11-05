import * as THREE from 'three/webgpu';
import { pass, renderOutput, uniform, vec2, vec3, vec4, float, uv, screenUV, length, smoothstep, mix, fract, sin, dot, time, Fn } from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { chapters } from './scenes/index.js';
import { Score } from './core/audio.js';

const $ = ( id ) => document.getElementById( id );
const params = new URLSearchParams( location.search );
const SHOT = params.has( 'shot' ); // deterministic capture mode for tools/shot.mjs
const SHOT_FRAMES = parseInt( params.get( 'frames' ) || '90' );
const FIXED_Q = params.has( 'q' ) ? parseFloat( params.get( 'q' ) ) : null;

// ───────────────────────── renderer ─────────────────────────
const canvas = $( 'gl' );

if ( ! navigator.gpu ) {

	$( 'boot-msg' ).innerHTML = 'Aether needs <b>WebGPU</b>.<br>Use a recent Chrome, Edge, Safari 26+ or Firefox 141+ on a GPU-equipped device.';
	throw new Error( 'WebGPU unavailable' );

}

const renderer = new THREE.WebGPURenderer( {
	canvas,
	antialias: false,
	powerPreference: 'high-performance',
	requiredLimits: undefined,
	trackTimestamp: ! SHOT,
} );

try {

	await renderer.init();

} catch ( e ) {

	$( 'boot-msg' ).textContent = 'GPU init failed: ' + e.message;
	throw e;

}

if ( ! renderer.backend.isWebGPUBackend ) {

	$( 'boot-msg' ).innerHTML = 'WebGPU adapter unavailable — this experience relies on compute shaders.';
	throw new Error( 'not WebGPU backend' );

}

renderer.toneMapping = THREE.AgXToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.setClearColor( 0x000000, 1 );

const maxDPR = Math.min( window.devicePixelRatio || 1, 2 );
let pixelRatio = FIXED_Q ?? Math.min( maxDPR, 1.5 );
renderer.setPixelRatio( pixelRatio );
renderer.setSize( innerWidth, innerHeight, false );

// GPU tier heuristic used by chapters to pick particle counts.
const adapterInfo = renderer.backend.adapter?.info || {};
const isSoftware = /swiftshader|llvmpipe|software/i.test( `${adapterInfo.vendor} ${adapterInfo.architecture} ${adapterInfo.description}` );
const isMobile = /Android|iPhone|iPad/i.test( navigator.userAgent );
const tier = params.has( 'tier' ) ? parseInt( params.get( 'tier' ) ) : ( isSoftware ? 0 : isMobile ? 1 : 2 );

$( 'st-backend' ).textContent = isSoftware ? 'WebGPU (sw)' : 'WebGPU';

// ───────────────────────── shared camera / controls ─────────────────────────
const camera = new THREE.PerspectiveCamera( 50, innerWidth / innerHeight, 0.01, 2000 );
const controls = new OrbitControls( camera, canvas );
controls.enableDamping = true;
controls.dampingFactor = 0.06;

// ───────────────────────── pointer ─────────────────────────
const raycaster = new THREE.Raycaster();
const pointer = {
	ndc: new THREE.Vector2(),
	prevNdc: new THREE.Vector2(),
	velocity: new THREE.Vector2(), // ndc units / second
	down: false,
	button: 0,
	ray: raycaster.ray,
	inside: false,
};

canvas.addEventListener( 'pointermove', ( e ) => {

	pointer.ndc.set( ( e.clientX / innerWidth ) * 2 - 1, - ( e.clientY / innerHeight ) * 2 + 1 );
	pointer.inside = true;

} );
canvas.addEventListener( 'pointerdown', ( e ) => {

	pointer.down = true; pointer.button = e.button;

} );
addEventListener( 'pointerup', () => pointer.down = false );
canvas.addEventListener( 'pointerleave', () => pointer.inside = false );

// ───────────────────────── GUI ─────────────────────────
const gui = new GUI( { title: 'controls' } );
gui.close();
let guiVisible = ! SHOT && innerWidth > 640;
gui.domElement.classList.toggle( 'closed-hard', ! guiVisible );

// ───────────────────────── sound ─────────────────────────
const score = new Score();
const soundBtn = $( 'sound' );
function setSound( on ) {

	if ( on ) score.start(); else score.stop();
	soundBtn.textContent = on ? 'sound on' : 'sound off';
	soundBtn.setAttribute( 'aria-pressed', String( on ) );

}

soundBtn.onclick = () => setSound( ! score.enabled );

// ───────────────────────── context passed to chapters ─────────────────────────
const ctx = {
	THREE, renderer, camera, controls, pointer, gui: null, tier, isSoftware, shot: SHOT,
	params,
	size: new THREE.Vector2( innerWidth, innerHeight ),
	setStats: ( lines ) => {

		$( 'st-extra' ).innerHTML = lines.map( ( [ k, v ] ) => `<div><span>${k}</span><b>${v}</b></div>` ).join( '' );

	},
};

// ───────────────────────── post-processing ─────────────────────────
// Default chain: scene pass → bloom → grain + vignette → (renderer tone mapping + sRGB).
// Chapters can override with instance.post(ctx, scenePassNode) returning a vec4 node.
function buildPipeline( inst ) {

	const scene = inst.scene;
	const cam = inst.camera || camera;
	const pipe = new THREE.RenderPipeline( renderer );
	const scenePass = pass( scene, cam );
	let node;

	if ( inst.post ) {

		node = inst.post( ctx, scenePass );

	} else {

		const b = inst.bloom || {};
		const color = scenePass.getTextureNode( 'output' );
		const bl = bloom( color, b.strength ?? 0.8, b.radius ?? 0.6, b.threshold ?? 0.0 );
		inst._bloomNode = bl;
		node = color.add( bl );

	}

	node = finishGrade( node );
	if ( SHOT ) {

		// headless capture: SwiftShader cannot present a WebGPU canvas, so we render into shotRT and read back.
		node = renderOutput( node, renderer.toneMapping, renderer.outputColorSpace );
		pipe.outputColorTransform = false;

	}

	pipe.outputNode = node;
	return pipe;

}

const gradeUniforms = { vignette: uniform( 0.9 ), grain: uniform( 0.035 ) };

const finishGrade = Fn( ( [ col ] ) => {

	const c = vec4( col ).toVar();
	const d = length( screenUV.sub( 0.5 ).mul( vec2( 1.0, 0.85 ) ) );
	const vig = smoothstep( 0.85, 0.2, d ).mul( gradeUniforms.vignette ).add( float( 1 ).sub( gradeUniforms.vignette ) );
	const n = fract( sin( dot( screenUV.add( fract( time.mul( 0.1 ) ) ), vec2( 12.9898, 78.233 ) ) ).mul( 43758.5453 ) ).sub( 0.5 );
	c.rgb.assign( c.rgb.mul( vig ).add( n.mul( gradeUniforms.grain ) ) );
	return c;

} );

// ───────────────────────── chapter lifecycle ─────────────────────────
let current = null; // { meta, inst, pipe, folder }
let currentIndex = - 1;
let switching = false;

const nav = $( 'chapters' );
chapters.forEach( ( c, i ) => {

	const b = document.createElement( 'button' );
	b.textContent = c.label;
	b.onclick = () => goTo( i );
	nav.appendChild( b );

} );

async function goTo( index ) {

	index = ( index + chapters.length ) % chapters.length;
	if ( switching || index === currentIndex ) return;
	switching = true;

	$( 'fade' ).classList.remove( 'clear' );
	$( 'chapter' ).classList.add( 'swap' );
	if ( current ) await sleep( SHOT ? 0 : 900 );

	if ( current ) {

		try {

			current.inst.dispose?.();

		} catch ( e ) {

			console.error( e );

		}

		current.pipe.dispose?.();
		current.folder.destroy();
		current = null;

	}

	ctx.setStats( [] );
	controls.reset();
	controls.enabled = true;
	controls.autoRotate = false;
	camera.fov = 50; camera.near = 0.01; camera.far = 2000; camera.updateProjectionMatrix();

	const meta = chapters[ index ];
	score.setChapter( meta.id );
	const mod = ( await meta.load() ).default;
	const folder = gui.addFolder( mod.title || meta.label );
	ctx.gui = folder;

	let inst;
	try {

		inst = await mod.create( ctx );

	} catch ( e ) {

		console.error( `chapter ${meta.id} failed`, e );
		$( 'boot-msg' ).textContent = `chapter ${meta.id} failed: ${e.message}`;
		window.__error = String( e && e.stack || e );
		switching = false;
		throw e;

	}

	const pipe = buildPipeline( inst );
	current = { meta, mod, inst, pipe, folder };
	currentIndex = index;

	$( 'ch-num' ).textContent = String( index + 1 ).padStart( 2, '0' ) + ' / ' + String( chapters.length ).padStart( 2, '0' );
	$( 'ch-title' ).textContent = mod.title || meta.label;
	$( 'ch-sub' ).textContent = mod.subtitle || '';
	[ ...nav.children ].forEach( ( b, i ) => b.classList.toggle( 'active', i === index ) );
	if ( ! SHOT ) history.replaceState( null, '', '?' + new URLSearchParams( { ...Object.fromEntries( params ), c: meta.id } ) );

	resize();
	await sleep( 50 );
	$( 'chapter' ).classList.remove( 'swap' );
	$( 'fade' ).classList.add( 'clear' );
	switching = false;

}

const sleep = ( ms ) => new Promise( ( r ) => setTimeout( r, ms ) );

// ───────────────────────── resize / adaptive resolution ─────────────────────────
function resize() {

	ctx.size.set( innerWidth, innerHeight );
	camera.aspect = innerWidth / innerHeight;
	camera.updateProjectionMatrix();
	if ( current?.inst.camera && current.inst.camera !== camera ) {

		current.inst.camera.aspect = camera.aspect;
		current.inst.camera.updateProjectionMatrix();

	}

	renderer.setPixelRatio( pixelRatio );
	renderer.setSize( innerWidth, innerHeight, false );
	shotRT?.setSize( Math.round( innerWidth * pixelRatio ), Math.round( innerHeight * pixelRatio ) );
	current?.inst.resize?.( innerWidth, innerHeight, pixelRatio );
	$( 'st-res' ).textContent = `${Math.round( innerWidth * pixelRatio )}×${Math.round( innerHeight * pixelRatio )}`;

}

addEventListener( 'resize', resize );

let frameTimes = [];
let gpuMs = null; // GPU time per frame (render + compute) from timestamp queries, when the adapter supports them
let gpuPending = false;

function sampleGpu() {

	if ( SHOT || gpuPending || ! renderer.hasFeature?.( 'timestamp-query' ) ) return;
	gpuPending = true;
	Promise.all( [ renderer.resolveTimestampsAsync( 'render' ), renderer.resolveTimestampsAsync( 'compute' ) ] ).then( () => {

		const ms = ( renderer.info.render.timestamp || 0 ) + ( renderer.info.compute.timestamp || 0 );
		if ( ms > 0 ) gpuMs = gpuMs === null ? ms : gpuMs * 0.8 + ms * 0.2;

	} ).catch( () => {} ).finally( () => gpuPending = false );

}

function adapt( dtMs ) {

	if ( FIXED_Q !== null || SHOT ) return;
	frameTimes.push( dtMs );
	if ( frameTimes.length < 45 ) return;
	frameTimes.sort( ( a, b ) => a - b );
	const median = frameTimes[ frameTimes.length >> 1 ];
	frameTimes = [];
	let next = pixelRatio;
	// Frame time is vsync-capped, so it can only tell us we are too slow. GPU time tells us when there is headroom.
	if ( median > 22 || ( gpuMs !== null && gpuMs > 15 ) ) next = Math.max( 0.5, pixelRatio * 0.85 );
	else if ( gpuMs !== null ? gpuMs < 9 && median < 18 : median < 14 ) next = Math.min( maxDPR, pixelRatio * 1.08 );
	if ( Math.abs( next - pixelRatio ) > 0.02 ) {

		pixelRatio = next;
		resize();

	}

}

// ───────────────────────── keys ─────────────────────────
addEventListener( 'keydown', ( e ) => {

	if ( e.target.tagName === 'INPUT' ) return;
	if ( e.key === 'ArrowRight' ) goTo( currentIndex + 1 );
	if ( e.key === 'ArrowLeft' ) goTo( currentIndex - 1 );
	if ( e.key === 'h' || e.key === 'H' ) $( 'hud' ).classList.toggle( 'hidden' );
	if ( e.key === 'g' || e.key === 'G' ) {

		guiVisible = ! guiVisible;
		gui.domElement.classList.toggle( 'closed-hard', ! guiVisible );

	}

	if ( e.key === 'm' || e.key === 'M' ) setSound( ! score.enabled );
	if ( e.key === 't' || e.key === 'T' ) setTour( ! tour.on );
	current?.inst.onKey?.( e );

} );

// ───────────────────────── tour (hands-free) ─────────────────────────
const tour = { on: false, timer: 0, every: parseFloat( params.get( 'tourEvery' ) || '45' ) };
function setTour( on ) {

	tour.on = on; tour.timer = 0;
	$( 'hint-tour' ).textContent = on ? 'tour on' : 'tour';
	if ( on ) controls.autoRotate = true;

}

if ( params.has( 'tour' ) ) setTour( true );

// ───────────────────────── loop ─────────────────────────
const clock = { t: 0, last: performance.now() };
let fpsAcc = 0, fpsN = 0, frame = 0;

async function tick() {

	const now = performance.now();
	let dt = Math.min( ( now - clock.last ) / 1000, 1 / 20 );
	clock.last = now;
	if ( SHOT ) dt = 1 / 60;

	if ( current && ! switching ) {

		pointer.velocity.subVectors( pointer.ndc, pointer.prevNdc ).divideScalar( Math.max( dt, 1e-4 ) );
		pointer.prevNdc.copy( pointer.ndc );
		raycaster.setFromCamera( pointer.ndc, current.inst.camera || camera );

		clock.t += dt;
		if ( controls.enabled ) controls.update( dt );
		current.inst.update?.( dt, clock.t, ctx );
		score.setPress( pointer.down ? 1 : Math.min( 1, pointer.velocity.length() * 0.15 ) );
		if ( SHOT ) {

			if ( shotBusy ) return;
			renderer.setRenderTarget( shotRT );
			current.pipe.render();
			renderer.setRenderTarget( null );

		} else {

			current.pipe.render();

		}

		frame ++;

	}

	fpsAcc += dt; fpsN ++;
	if ( fpsAcc > 0.5 ) {

		$( 'st-fps' ).textContent = ( fpsN / fpsAcc ).toFixed( 0 ) + ( gpuMs !== null ? ` · ${gpuMs.toFixed( 1 )}ms gpu` : '' );
		sampleGpu();
		fpsAcc = 0; fpsN = 0;

	}

	adapt( dt * 1000 );

	if ( tour.on && ! switching && ! SHOT ) {

		tour.timer += dt;
		if ( pointer.down ) tour.timer = 0; // someone is playing: don't yank the chapter away
		if ( tour.timer > tour.every ) {

			tour.timer = 0;
			goTo( currentIndex + 1 );

		}

	}

	if ( SHOT && current && frame >= SHOT_FRAMES && ! window.__ready && ! shotBusy ) {

		shotBusy = true;
		await blitShot();
		window.__ready = true;

	}

}

// ───────────────────────── headless capture ─────────────────────────
let shotBusy = false;
const shotRT = SHOT ? new THREE.RenderTarget( 1, 1, { type: THREE.UnsignedByteType, depthBuffer: false } ) : null;

async function blitShot() {

	const w = shotRT.width, h = shotRT.height;
	const px = await renderer.readRenderTargetPixelsAsync( shotRT, 0, 0, w, h );
	const c2 = document.createElement( 'canvas' );
	c2.width = w; c2.height = h;
	c2.style.cssText = 'position:fixed;inset:0;width:100vw;height:100vh;';
	const g = c2.getContext( '2d' );
	const img = g.createImageData( w, h );
	img.data.set( px.subarray( 0, w * h * 4 ) );
	g.putImageData( img, 0, 0 );
	canvas.after( c2 );
	canvas.style.display = 'none';

}

// ───────────────────────── boot ─────────────────────────
const startId = params.get( 'c' );
const startIndex = Math.max( 0, chapters.findIndex( ( c ) => c.id === startId ) );

$( 'boot-msg' ).textContent = 'compiling shaders…';
try {

	await goTo( startIndex );

} catch ( e ) {

	throw e;

}

renderer.setAnimationLoop( tick );

if ( SHOT ) {

	$( 'boot' ).remove();
	$( 'hud' ).classList.toggle( 'hidden', params.get( 'hud' ) !== '1' );

} else {

	// the first chapter is already running behind a translucent veil; entering is also the user gesture audio needs
	$( 'boot' ).classList.add( 'ready' );
	$( 'boot-msg' ).textContent = 'Five real-time simulations, computed on your GPU.';
	$( 'enter-row' ).hidden = false;
	$( 'enter-sound' ).focus();
	const enter = ( withSound ) => {

		if ( withSound ) setSound( true );
		$( 'boot' ).classList.add( 'gone' );
		setTimeout( () => $( 'boot' ).remove(), 1200 );

	};

	$( 'enter-sound' ).onclick = () => enter( true );
	$( 'enter-silent' ).onclick = () => enter( false );

}

window.__aether = { ctx, goTo, get current() {

	return current;

} };
