// Studio stage for Silk: procedural softbox environment (PMREM), dark cyclorama, sculptures, lights.
import * as THREE from 'three/webgpu';
import {
	Fn, float, vec3, color, positionLocal, positionWorld, normalize, smoothstep, mix, length, uniform,
	mx_noise_float, abs,
} from 'three/tsl';

// Collider description shared by the simulation (analytic SDFs) and the visible meshes.
export function makeLayout() {

	const knotQuat = new THREE.Quaternion().setFromEuler( new THREE.Euler( - 0.22, - 0.55, 0.12, 'YXZ' ) );
	return {
		colliders: {
			sphere: { pos: [ - 2.55, 0.92, 1.05 ], r: 0.92 },
			capsule: { a: [ 2.75, 0.44, - 0.15 ], b: [ 2.25, 2.55, - 0.75 ], r: 0.44 },
			knot: { pos: new THREE.Vector3( 0.05, 2.12, - 0.35 ), quat: knotQuat, R: 1.0, tube: 0.27, p: 2, q: 3 },
			plinth: { pos: [ 0.05, 0, - 0.35 ], r: 0.62, h: 0.42, round: 0.05 },
			cyc: { wallZ: - 6.5, cove: 2.6 },
		},
		// hang mode: two top-corner anchors per sheet (world), slight lean, sway amplitude
		hangSheets: [
			{ a: [ - 2.05, 5.15, - 1.75 ], b: [ 1.45, 5.35, - 2.45 ], lean: [ 0, 0, - 0.15 ], bulge: 0.45, sway: 0.28 },
			{ a: [ - 4.55, 4.25, 0.25 ], b: [ - 2.0, 4.6, - 1.35 ], lean: [ 0.1, 0, 0 ], bulge: 0.35, sway: 0.25 },
			{ a: [ 1.25, 4.85, - 0.65 ], b: [ 4.05, 4.45, 1.1 ], lean: [ 0, 0, - 0.1 ], bulge: 0.4, sway: 0.3 },
		],
		// drop mode: sheet centres above the sculptures
		dropSheets: [
			{ c: [ 0.05, 4.45, - 0.35 ], ry: 0.35, tilt: 0.08 },
			{ c: [ - 2.55, 2.75, 1.05 ], ry: - 0.3, tilt: - 0.06 },
			{ c: [ 2.45, 3.55, - 0.45 ], ry: 0.8, tilt: 0.1 },
		],
	};

}

function softbox( w, h, rgb, intensity ) {

	const m = new THREE.MeshBasicNodeMaterial( { side: THREE.DoubleSide } );
	// soft falloff toward the edges like a diffusion panel
	m.colorNode = Fn( () => {

		const p = positionLocal.xy.div( vec3( w * 0.5, h * 0.5, 1 ).xy );
		const e = smoothstep( 1.0, 0.55, abs( p.x ) ).mul( smoothstep( 1.0, 0.55, abs( p.y ) ) );
		return color( new THREE.Color( ...rgb ) ).mul( intensity ).mul( e );

	} )();
	return new THREE.Mesh( new THREE.PlaneGeometry( w, h ), m );

}

export async function buildStage( ctx, scene, layout, tier ) {

	const { renderer } = ctx;
	const disposables = [];
	const track = ( o ) => {

		disposables.push( o ); return o;

	};

	// ── environment: dark studio with softboxes, prefiltered ───────────────────
	const env = new THREE.Scene();
	const domeMat = track( new THREE.MeshBasicNodeMaterial( { side: THREE.BackSide } ) );
	domeMat.colorNode = Fn( () => {

		const d = normalize( positionLocal );
		const g = smoothstep( - 0.3, 0.9, d.y );
		return mix( color( 0x050404 ), color( 0x16181d ), g ).mul( 0.8 );

	} )();
	const dome = new THREE.Mesh( track( new THREE.SphereGeometry( 30, 32, 16 ) ), domeMat );
	env.add( dome );
	const boxes = [
		// [w,h,rgb,intensity,pos,lookAt]
		[ 7, 4.5, [ 1.0, 0.92, 0.82 ], 5.0, [ - 10, 9, 9 ] ], // warm key softbox
		[ 2.2, 12, [ 0.72, 0.84, 1.0 ], 9.0, [ - 3, 7, - 14 ] ], // cool rim strip behind
		[ 2.2, 12, [ 1.0, 0.78, 0.6 ], 4.0, [ 13, 6, - 6 ] ], // warm kicker strip right
		[ 10, 10, [ 0.9, 0.92, 1.0 ], 1.1, [ 0, 18, 0 ] ], // overhead
	];
	for ( const [ w, h, rgb, I, p ] of boxes ) {

		const b = softbox( w, h, rgb, I );
		track( b.geometry ); track( b.material );
		b.position.set( ...p );
		b.lookAt( 0, 2, 0 );
		env.add( b );

	}

	const pmrem = new THREE.PMREMGenerator( renderer );
	const envRT = pmrem.fromScene( env, 0.035, 0.1, 100 );
	pmrem.dispose();
	scene.environment = envRT.texture;
	scene.environmentIntensity = 0.55;
	disposables.push( envRT );

	// ── cyclorama: floor → cove → wall, dark gradient ──────────────────────────
	const { cyc } = layout.colliders;
	const prof = [];
	const zFront = 16, yTop = 16, segArc = 24;
	prof.push( [ zFront, 0 ] );
	prof.push( [ cyc.wallZ + cyc.cove, 0 ] );
	for ( let k = 1; k < segArc; k ++ ) {

		const a = ( k / segArc ) * Math.PI * 0.5;
		prof.push( [ cyc.wallZ + cyc.cove - Math.sin( a ) * cyc.cove, cyc.cove - Math.cos( a ) * cyc.cove ] );

	}

	prof.push( [ cyc.wallZ, cyc.cove ] );
	prof.push( [ cyc.wallZ, yTop ] );
	// subdivide long runs so lighting / shadows interpolate nicely
	const pts = [];
	for ( let k = 0; k < prof.length - 1; k ++ ) {

		const [ z0, y0 ] = prof[ k ], [ z1, y1 ] = prof[ k + 1 ];
		const n = Math.max( 1, Math.ceil( Math.hypot( z1 - z0, y1 - y0 ) / 0.5 ) );
		for ( let q = 0; q < n; q ++ ) pts.push( [ z0 + ( z1 - z0 ) * q / n, y0 + ( y1 - y0 ) * q / n ] );

	}

	pts.push( prof[ prof.length - 1 ] );
	const xs = 40, X0 = - 20, X1 = 20;
	const cpos = [], cidx = [];
	for ( let k = 0; k < pts.length; k ++ ) for ( let ix = 0; ix <= xs; ix ++ ) cpos.push( X0 + ( X1 - X0 ) * ix / xs, pts[ k ][ 1 ], pts[ k ][ 0 ] );
	for ( let k = 0; k < pts.length - 1; k ++ ) for ( let ix = 0; ix < xs; ix ++ ) {

		const a = k * ( xs + 1 ) + ix, b = a + 1, c = a + xs + 1, d = c + 1;
		cidx.push( a, b, c, b, d, c );

	}

	const cycGeo = track( new THREE.BufferGeometry() );
	cycGeo.setAttribute( 'position', new THREE.Float32BufferAttribute( cpos, 3 ) );
	cycGeo.setIndex( cidx );
	cycGeo.computeVertexNormals();
	const cycMat = track( new THREE.MeshStandardNodeMaterial( { roughness: 0.82, metalness: 0 } ) );
	cycMat.colorNode = Fn( () => {

		const p = positionWorld;
		const r = length( p.xz.sub( vec3( 0, 0, - 0.5 ).xz ) );
		const pool = smoothstep( 9.0, 1.5, r ).mul( 0.6 ).add( 0.4 );
		const grain = mx_noise_float( p.mul( 3.0 ) ).mul( 0.04 ).add( 1 );
		return mix( color( 0x0b0a0a ), color( 0x2a2724 ), pool ).mul( grain );

	} )();
	const cycMesh = new THREE.Mesh( cycGeo, cycMat );
	cycMesh.receiveShadow = true;
	scene.add( cycMesh );

	// ── sculptures ──────────────────────────────────────────────────────────────
	const { sphere, capsule, knot, plinth } = layout.colliders;
	const stoneMat = track( new THREE.MeshPhysicalNodeMaterial( { color: 0x0d0d10, roughness: 0.32, metalness: 0.0, clearcoat: 0.8, clearcoatRoughness: 0.12 } ) );
	const bronzeMat = track( new THREE.MeshPhysicalNodeMaterial( { color: 0x6b4a33, roughness: 0.28, metalness: 1.0 } ) );
	bronzeMat.colorNode = Fn( () => {

		const n = mx_noise_float( positionLocal.mul( 2.5 ) ).mul( 0.5 ).add( 0.5 );
		return mix( color( 0x2b1d14 ), color( 0x7a5a3e ), n.mul( 0.6 ).add( 0.25 ) );

	} )();
	const plasterMat = track( new THREE.MeshStandardNodeMaterial( { color: 0x1d1b1a, roughness: 0.6, metalness: 0 } ) );

	const seg = tier === 0 ? [ 200, 20 ] : [ 480, 48 ];
	const knotMesh = new THREE.Mesh( track( new THREE.TorusKnotGeometry( knot.R, knot.tube - 0.004, seg[ 0 ], seg[ 1 ], knot.p, knot.q ) ), bronzeMat );
	knotMesh.position.copy( knot.pos );
	knotMesh.quaternion.copy( knot.quat );

	const sphereMesh = new THREE.Mesh( track( new THREE.SphereGeometry( sphere.r - 0.004, tier === 0 ? 48 : 96, tier === 0 ? 24 : 64 ) ), stoneMat );
	sphereMesh.position.set( ...sphere.pos );

	const a = new THREE.Vector3( ...capsule.a ), b = new THREE.Vector3( ...capsule.b );
	const capMesh = new THREE.Mesh( track( new THREE.CapsuleGeometry( capsule.r - 0.004, a.distanceTo( b ), 16, tier === 0 ? 32 : 64 ) ), stoneMat );
	capMesh.position.copy( a ).add( b ).multiplyScalar( 0.5 );
	capMesh.quaternion.setFromUnitVectors( new THREE.Vector3( 0, 1, 0 ), b.clone().sub( a ).normalize() );

	const plinthMesh = new THREE.Mesh( track( new THREE.CylinderGeometry( plinth.r, plinth.r, plinth.h, 64, 1 ) ), plasterMat );
	plinthMesh.position.set( plinth.pos[ 0 ], plinth.h * 0.5, plinth.pos[ 2 ] );

	for ( const m of [ knotMesh, sphereMesh, capMesh, plinthMesh ] ) {

		m.castShadow = true; m.receiveShadow = true;
		scene.add( m );

	}

	// ── lights ───────────────────────────────────────────────────────────────
	const key = new THREE.SpotLight( 0xfff0e2, 750, 0, 0.62, 0.75, 2 );
	key.position.set( - 6.0, 9.0, 6.5 );
	key.target.position.set( 0, 1.6, - 0.4 );
	key.castShadow = true;
	const sm = tier === 0 ? 512 : tier === 1 ? 1024 : 2048;
	key.shadow.mapSize.set( sm, sm );
	key.shadow.camera.near = 3;
	key.shadow.camera.far = 30;
	key.shadow.bias = - 0.0004;
	key.shadow.normalBias = 0.025;
	key.shadow.radius = 3;
	scene.add( key, key.target );

	const rim = new THREE.SpotLight( 0xa9c6ff, 1500, 0, 0.7, 0.8, 2 );
	rim.position.set( - 2.0, 7.0, - 9.0 );
	rim.target.position.set( 0.6, 2.4, 0.5 );
	scene.add( rim, rim.target );

	const kicker = new THREE.SpotLight( 0xffb27a, 420, 0, 0.6, 0.9, 2 );
	kicker.position.set( 9.0, 2.5, - 3.5 );
	kicker.target.position.set( 0, 2.0, 0 );
	scene.add( kicker, kicker.target );

	const hemi = new THREE.HemisphereLight( 0x6d7890, 0x0c0a09, 0.25 );
	scene.add( hemi );

	const lights = { key, rim, kicker, hemi };
	const intensity = uniform( 1 );

	return {
		lights, intensity, meshes: { knotMesh, sphereMesh, capMesh, plinthMesh, cycMesh },
		dispose() {

			for ( const o of disposables ) o.dispose?.();
			key.shadow.dispose?.();
			scene.environment = null;

		},
	};

}
