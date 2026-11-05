// Rendering for the Schrödinger chapter: phase-coloured |ψ|² height field, glass monoliths, Born-rule detector screen,
// Bohmian mist, dark studio environment.
import * as THREE from 'three/webgpu';
import {
	Fn, vec2, vec3, vec4, float, uniform, texture, uv, varying, positionLocal, positionWorld, normalWorld, normalView, cameraViewMatrix,
	positionViewDirection, length, atan, cos, pow, exp, max, min, mix, smoothstep, abs, fract, fwidth, normalize, dot, clamp, screenUV, sin,
} from 'three/tsl';

export const WORLD = 10; // domain [0,1]² ↦ world x,z ∈ [−5, 5]

// cyclic, perceptually smooth phase palette: cyan → indigo → rose → amber → (white-gold) → cyan
const PAL = [
	[ 0.04, 0.62, 1.0 ],
	[ 0.32, 0.12, 1.0 ],
	[ 1.0, 0.1, 0.42 ],
	[ 1.0, 0.56, 0.1 ],
];

export const phasePalette = Fn( ( [ theta ] ) => {

	const acc = vec3( 0 ).toVar();
	const wsum = float( 0 ).toVar();
	for ( let i = 0; i < 4; i ++ ) {

		const w = pow( cos( theta.sub( i * Math.PI / 2 ) ).mul( 0.5 ).add( 0.5 ), 2.2 );
		acc.addAssign( vec3( ...PAL[ i ] ).mul( w ) );
		wsum.addAssign( w );

	}

	return acc.div( wsum );

} );

// probability-only palette: ink → deep blue → cyan → ice white
export const heatPalette = Fn( ( [ t ] ) => {

	const a = vec3( 0.02, 0.06, 0.35 ), b = vec3( 0.05, 0.55, 1.0 ), c = vec3( 0.85, 0.95, 1.0 );
	return mix( mix( a, b, smoothstep( 0, 0.45, t ) ), c, smoothstep( 0.4, 1.0, t ) );

} );

/** Dark studio: a black room lit by a few soft-box strips, prefiltered into a PMREM. */
export function buildStudioEnv( renderer ) {

	const env = new THREE.Scene();
	const geos = [], mats = [];
	const box = ( w, h, col, k, pos ) => {

		const g = new THREE.PlaneGeometry( w, h );
		const m = new THREE.MeshBasicNodeMaterial( { color: new THREE.Color( col ).multiplyScalar( k ), side: THREE.DoubleSide } );
		const mesh = new THREE.Mesh( g, m );
		mesh.position.set( ...pos );
		mesh.lookAt( 0, 0, 0 );
		env.add( mesh );
		geos.push( g ); mats.push( m );

	};

	const shell = new THREE.SphereGeometry( 40, 32, 16 );
	const shellMat = new THREE.MeshBasicNodeMaterial( { side: THREE.BackSide } );
	shellMat.colorNode = Fn( () => {

		const y = normalize( positionLocal ).y;
		return mix( vec3( 0.004, 0.005, 0.008 ), vec3( 0.02, 0.025, 0.04 ), smoothstep( - 0.2, 0.8, y ) );

	} )();
	env.add( new THREE.Mesh( shell, shellMat ) );
	geos.push( shell ); mats.push( shellMat );

	box( 16, 2.2, 0xffffff, 5.0, [ 0, 14, 2 ] ); // overhead soft box
	box( 2.0, 14, 0x8fd0ff, 2.6, [ - 16, 5, - 3 ] ); // cool side strip
	box( 2.0, 12, 0xffc48a, 1.6, [ 15, 4, 6 ] ); // warm kicker
	box( 26, 0.9, 0xd8e8ff, 3.0, [ 0, 6, - 18 ] ); // back rim line
	box( 6, 6, 0x6a5cff, 0.5, [ 6, 9, 14 ] ); // faint violet fill

	const pmrem = new THREE.PMREMGenerator( renderer );
	const rt = pmrem.fromScene( env, 0.035, 0.1, 100 );
	pmrem.dispose();
	geos.forEach( ( g ) => g.dispose() );
	mats.forEach( ( m ) => m.dispose() );
	return rt;

}

export function backgroundNode() {

	return Fn( () => {

		const p = screenUV.sub( vec2( 0.5, 0.62 ) );
		const r = length( p.mul( vec2( 1.0, 1.6 ) ) );
		const base = mix( vec3( 0.012, 0.016, 0.03 ), vec3( 0.001, 0.0012, 0.002 ), smoothstep( 0.0, 0.9, r ) );
		return base;

	} )();

}

/** shared uniforms the materials read */
export function makeLook() {

	return {
		mode: uniform( 0 ), // 0 phase domain colouring, 1 probability only
		glow: uniform( 2.4 ),
		mistGain: uniform( 1.0 ),
		time: uniform( 0 ),
		screenFade: uniform( 1 ),
	};

}

// sample helpers ------------------------------------------------------------
const fieldColor = ( f, look, rhoRef ) => {

	// f = (Re ψ, Im ψ, h, V); amplitude relative to the reference packet peak
	const amp = length( f.xy ).div( Math.sqrt( rhoRef ) );
	const phaseCol = phasePalette( atan( f.y, f.x ) );
	const probCol = heatPalette( clamp( pow( amp, 0.8 ).mul( 1.1 ), 0, 1 ) );
	return { amp, col: mix( phaseCol, probCol, look.mode ) };

};

/** |ψ|² height field: displaced dense plane, normals from the filtered field, domain colouring into HDR. */
export function createSurface( solver, look, segs, rhoRef ) {

	const N = solver.N;
	const geo = new THREE.PlaneGeometry( WORLD, WORLD, segs, segs );
	geo.rotateX( - Math.PI / 2 );

	const mat = new THREE.MeshPhysicalNodeMaterial( {
		roughness: 0.2, metalness: 0.0, clearcoat: 1.0, clearcoatRoughness: 0.05, specularIntensity: 1.0, ior: 1.45,
	} );

	const d = vec2( uv().x, float( 1 ).sub( uv().y ) ); // domain coordinates
	const tex = solver.field;
	mat.positionNode = positionLocal.add( vec3( 0, texture( tex, d ).level( 0 ).z, 0 ) );

	const e = 1 / N;
	const nWorld = Fn( () => {

		const hl = texture( tex, d.sub( vec2( e, 0 ) ) ).z, hr = texture( tex, d.add( vec2( e, 0 ) ) ).z;
		const hd = texture( tex, d.sub( vec2( 0, e ) ) ).z, hu = texture( tex, d.add( vec2( 0, e ) ) ).z;
		return normalize( vec3( hl.sub( hr ), 2 * e * WORLD, hd.sub( hu ) ) );

	} )();
	const nView = cameraViewMatrix.mul( vec4( nWorld, 0 ) ).xyz.normalize();
	mat.normalNode = nView;
	mat.clearcoatNormalNode = nView;

	const f = texture( tex, d ).toVar( 'qf' );
	const { amp, col } = fieldColor( f, look, rhoRef );
	const ampV = amp.toVar( 'qamp' );
	const colV = col.toVar( 'qcol' );
	// lab floor: fine grid + equipotential contours, fading into the absorbing margin
	const edge = min( min( d.x, d.y ), min( float( 1 ).sub( d.x ), float( 1 ).sub( d.y ) ) );
	const fade = smoothstep( 0.0, 0.12, edge );
	const g = d.mul( 20 );
	const gl = abs( fract( g.sub( 0.5 ) ).sub( 0.5 ) ).div( fwidth( g ).mul( 1.2 ) );
	const grid = float( 1 ).sub( min( min( gl.x, gl.y ), 1 ) );
	const v = f.w;
	const cv = v.mul( 14 );
	const cl = abs( fract( cv.sub( 0.5 ) ).sub( 0.5 ) ).div( fwidth( cv ).add( 1e-4 ) );
	const contour = float( 1 ).sub( min( cl, 1 ) ).mul( smoothstep( 0.003, 0.03, v ) ).mul( smoothstep( 1.0, 0.6, v ) );
	const lines = vec3( 0.035, 0.05, 0.07 ).mul( grid ).add( vec3( 0.12, 0.06, 0.25 ).mul( contour ) ).mul( fade );
	const glow = pow( ampV, 1.15 ).mul( look.glow );

	mat.colorNode = vec3( 0.006, 0.007, 0.01 ).add( colV.mul( min( ampV, 1 ).mul( 0.22 ) ) );
	mat.emissiveNode = colV.mul( glow ).add( lines );

	const mesh = new THREE.Mesh( geo, mat );
	mesh.frustumCulled = false;
	return mesh;

}

export function createFloor() {

	const geo = new THREE.PlaneGeometry( 120, 120, 1, 1 );
	geo.rotateX( - Math.PI / 2 );
	const mat = new THREE.MeshPhysicalNodeMaterial( { roughness: 0.32, metalness: 0, clearcoat: 0.5, clearcoatRoughness: 0.12 } );
	const r = length( positionWorld.xz );
	const fade = smoothstep( 30, 7, r );
	mat.colorNode = vec3( 0.006, 0.007, 0.01 );
	mat.specularIntensityNode = fade;
	mat.clearcoatNode = fade.mul( 0.5 );
	const mesh = new THREE.Mesh( geo, mat );
	mesh.position.y = - 0.004;
	return mesh;

}

/** Black glass with the wave's light bleeding onto the faces that look at it. */
export function createMonolithMaterial( solver, look, rhoRef, { opacity = 1, tint = 1 } = {} ) {

	const mat = new THREE.MeshPhysicalNodeMaterial( {
		color: 0x020305, roughness: 0.05, metalness: 0.0, clearcoat: 1.0, clearcoatRoughness: 0.02, ior: 1.7, specularIntensity: 1,
		transparent: opacity < 1, opacity, depthWrite: opacity >= 1,
	} );

	mat.emissiveNode = Fn( () => {

		const base = positionWorld.xz.div( WORLD ).add( 0.5 );
		const n = normalWorld;
		const acc = vec3( 0 ).toVar();
		for ( let i = 1; i <= 4; i ++ ) {

			const q = base.add( n.xz.mul( 0.009 * i ) );
			const f = texture( solver.field, q );
			const { amp, col } = fieldColor( f, look, rhoRef );
			acc.addAssign( col.mul( pow( amp, 1.1 ) ).mul( 1.0 / i ) );

		}

		const hfall = exp( max( positionWorld.y, 0 ).mul( - 2.0 ) );
		const fres = pow( float( 1 ).sub( max( dot( normalView, positionViewDirection ), 0 ) ), 5 );
		return acc.mul( hfall ).mul( 0.55 * tint ).add( vec3( 0.03, 0.06, 0.1 ).mul( fres ) );

	} )();

	return mat;

}

/** Detector screen: a dark glass panel; every Born-rule detection is a spark that settles into a phosphor dot. */
export function createDetector( solver, look, { height = 2.4 } = {} ) {

	const group = new THREE.Group();
	const xs = WORLD / 2 + 0.35;

	const panelGeo = new THREE.BoxGeometry( 0.06, height, WORLD * 0.98 );
	const panelMat = new THREE.MeshPhysicalNodeMaterial( { color: 0x010203, roughness: 0.08, clearcoat: 1, clearcoatRoughness: 0.03, metalness: 0 } );
	panelMat.emissiveNode = Fn( () => {

		// faint calibration ticks along the face
		const zz = positionLocal.z.div( WORLD ).add( 0.5 ).mul( 40 );
		const tick = float( 1 ).sub( min( abs( fract( zz.sub( 0.5 ) ).sub( 0.5 ) ).div( fwidth( zz ).mul( 1.0 ) ), 1 ) );
		const yy = positionLocal.y.div( height ).add( 0.5 );
		const nearEdge = smoothstep( 0.035, 0.0, yy ).add( smoothstep( 0.965, 1.0, yy ) );
		return vec3( 0.03, 0.07, 0.11 ).mul( tick.mul( nearEdge ).add( nearEdge.mul( 0.25 ) ) ).mul( look.screenFade );

	} )();
	const panel = new THREE.Mesh( panelGeo, panelMat );
	panel.position.set( xs, height / 2, 0 );
	group.add( panel );

	// hit sprites
	const h = solver.hits.toAttribute();
	const mat = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
	const age = look.time.sub( h.z );
	const flash = exp( age.mul( - 5.0 ) ).mul( age.greaterThanEqual( 0 ).select( 1, 0 ) );
	mat.positionNode = vec3( xs - 0.045, float( 0.1 ).add( h.y.mul( height - 0.2 ) ), h.x.sub( 0.5 ).mul( WORLD ) );
	mat.scaleNode = float( 0.034 ).mul( float( 1 ).add( flash.mul( 3.5 ) ) ).mul( h.w ).mul( look.screenFade );
	const disc = Fn( () => {

		const r = length( uv().sub( 0.5 ) ).mul( 2 );
		return pow( max( float( 1 ).sub( r ), 0 ), 2.2 );

	} )();
	const phosphor = vec3( 0.45, 0.8, 1.0 );
	const hot = vec3( 1.0, 0.85, 0.6 );
	mat.colorNode = vec4( mix( phosphor.mul( 0.9 ), hot.mul( 22 ), flash ).mul( disc ), 1 );
	mat.opacityNode = disc;
	const sprites = new THREE.Sprite( mat );
	sprites.count = solver.MAX_HITS;
	sprites.frustumCulled = false;
	sprites.renderOrder = 2;
	group.add( sprites );

	group.userData.dispose = () => {

		panelGeo.dispose(); panelMat.dispose(); mat.dispose(); sprites.geometry?.dispose?.();

	};

	return group;

}

/** Bohmian mist: particles that ride the probability current a little above the surface. */
export function createMist( solver, look, rhoRef ) {

	const m = solver.mist.toAttribute();
	const mat = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
	const fV = texture( solver.field, m.xy ).level( 0 );
	const lift = m.w.mul( m.w ).mul( 0.55 ).add( 0.05 ).add( m.z.mul( 0.25 ) );
	mat.positionNode = vec3( m.x.sub( 0.5 ).mul( WORLD ), fV.z.add( lift ), m.y.sub( 0.5 ).mul( WORLD ) );
	const alive = smoothstep( 0.0, 0.12, m.z ).mul( smoothstep( 1.0, 0.65, m.z ) ).mul( m.z.lessThan( 1 ).select( 1, 0 ) );
	mat.scaleNode = float( 0.05 ).add( m.w.mul( 0.05 ) ).mul( alive ).mul( look.mistGain.greaterThan( 0 ).select( 1, 0 ) );
	const vCol = varying( Fn( () => {

		const { col } = fieldColor( fV, look, rhoRef );
		return mix( col, vec3( 1 ), 0.25 ).mul( alive ).mul( look.mistGain ).mul( float( 1 ).sub( m.w.mul( 0.6 ) ) );

	} )() );
	const disc = Fn( () => {

		const r = length( uv().sub( 0.5 ) ).mul( 2 );
		return pow( max( float( 1 ).sub( r ), 0 ), 3.0 );

	} )();
	mat.colorNode = vec4( vCol.mul( disc ).mul( 0.5 ), 1 );
	mat.opacityNode = disc;
	const sprites = new THREE.Sprite( mat );
	sprites.count = solver.MIST;
	sprites.frustumCulled = false;
	sprites.renderOrder = 3;
	sprites.userData.dispose = () => mat.dispose();
	return sprites;

}

/** Pointer affordances: hover reticle, launch arrow with packet-width ring, wall brush. */
export function createPointerGizmo() {

	const group = new THREE.Group();
	const mk = ( geo, col, k ) => {

		const m = new THREE.MeshBasicNodeMaterial( { color: new THREE.Color( col ).multiplyScalar( k ), transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending } );
		const mesh = new THREE.Mesh( geo, m );
		mesh.renderOrder = 10;
		return mesh;

	};

	const ring = mk( new THREE.RingGeometry( 0.9, 1.0, 64 ).rotateX( - Math.PI / 2 ), 0x7fd6ff, 1.4 );
	const width = mk( new THREE.RingGeometry( 0.96, 1.0, 64 ).rotateX( - Math.PI / 2 ), 0xffffff, 0.6 );
	const shaft = mk( new THREE.PlaneGeometry( 1, 0.05 ).rotateX( - Math.PI / 2 ).translate( 0.5, 0, 0 ), 0x7fd6ff, 2.0 );
	const headGeo = new THREE.BufferGeometry();
	headGeo.setAttribute( 'position', new THREE.Float32BufferAttribute( [ 0, 0, 0, - 0.35, 0, 0.17, - 0.35, 0, - 0.17 ], 3 ) );
	const head = mk( headGeo, 0x7fd6ff, 2.4 );
	head.material.side = THREE.DoubleSide;
	group.add( ring, width, shaft, head );
	group.userData = { ring, width, shaft, head };
	group.visible = false;
	group.userData.dispose = () => group.traverse( ( o ) => {

		if ( o.isMesh ) {

			o.geometry.dispose(); o.material.dispose();

		}

	} );
	return group;

}

export { sin };
