// Genesis — rendering: velocity-stretched HDR billboards, marine-snow dust, abyssal backdrop, pointer lure.

import * as THREE from 'three/webgpu';
import {
	Fn, uniform, uniformArray, instancedArray, instanceIndex, varying, uv, screenUV, time,
	float, vec2, vec3, vec4, dot, max, min, mix, clamp, smoothstep, fract, length, abs, exp, atan, hash, sin, cos, mod,
	modelViewMatrix,
} from 'three/tsl';
import { MAX_SPECIES } from './rules.js';

export function createParticleMaterial( sim, palette ) {

	const r = {
		size: uniform( 0.16 ),
		bright: uniform( 1.0 ),
		stretch: uniform( 0.05 ),
		dof: uniform( 0.035 ),
		focus: uniform( 20 ),
		densRef: uniform( 8 ),
		speedRef: uniform( 3 ),
		core: uniformArray( palette.core.map( ( c ) => new THREE.Color( ...c ) ), 'color' ),
		rim: uniformArray( palette.rim.map( ( c ) => new THREE.Color( ...c ) ), 'color' ),
	};

	const P = sim.pos.toAttribute();
	const V = sim.vel.toAttribute();

	const mv = modelViewMatrix.mul( vec4( P.xyz, 1 ) );
	const viewZ = mv.z.negate();
	const vView = modelViewMatrix.mul( vec4( V.xyz, 0 ) ).xy;
	const speed = length( V.xyz );
	const densN = V.w.div( r.densRef );
	const sp = P.w.toInt();
	const seed = fract( P.w );

	// cheap depth of field: out-of-focus sprites grow and dim (energy preserving)
	const coc = clamp( abs( viewZ.sub( r.focus ) ).mul( r.dof ), 0, 2.5 );
	const dense = smoothstep( 0.1, 1.6, densN );
	const sizeW = r.size.mul( seed.mul( 0.6 ).add( 0.7 ) ).mul( mix( 1.35, 0.75, dense ) ).mul( coc.add( 1 ) );
	const sv = length( vView );
	const stretch = clamp( sv.mul( r.stretch ), 0, 5 ).add( 1 );

	const mat = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
	mat.positionNode = P.xyz;
	mat.scaleNode = vec2( sizeW.mul( stretch ), sizeW );
	mat.rotationNode = atan( vView.y, vView.x );

	const vColor = varying( Fn( () => {

		const core = r.core.element( sp );
		const rim = r.rim.element( sp );
		const fast = smoothstep( 0, r.speedRef, speed );
		// interior of a cell → deep core tint; sparse membranes / fast swimmers → bright rim tint
		const base = mix( rim, core, dense.mul( 0.85 ) );
		const glow = base.mul( fast.mul( 1.6 ).add( 0.45 ) ).add( rim.mul( fast.mul( fast ).mul( 0.6 ) ) );
		const energy = float( 1 ).div( coc.add( 1 ).mul( coc.add( 1 ) ).mul( stretch.sqrt() ) );
		const crowd = float( 1 ).div( densN.mul( 0.35 ).add( 1 ) );
		return glow.mul( energy ).mul( crowd ).mul( r.bright );

	} )(), 'vGenesisColor' );

	mat.colorNode = Fn( () => {

		const d = uv().sub( 0.5 ).mul( 2 );
		const q = dot( d, d );
		const soft = max( float( 1 ).sub( q ), 0 );
		const shape = soft.mul( soft ).mul( 0.55 ).add( exp( q.mul( - 14 ) ).mul( 0.9 ) );
		return vec4( vColor.mul( shape ), 1 );

	} )();

	return { material: mat, uniforms: r };

}

// Marine snow: faint flakes sinking through a large volume around the colony, for parallax and depth.
export function createDust( renderer, bound, count ) {

	const buf = instancedArray( count, 'vec4' );
	const init = Fn( () => {

		const i = float( instanceIndex ).mul( 5 ).add( 911 );
		const p = vec3( hash( i ), hash( i.add( 1 ) ), hash( i.add( 2 ) ) ).sub( 0.5 ).mul( vec3( 2, 2, 2 ) ).mul( bound * 3.2 );
		buf.element( instanceIndex ).assign( vec4( p, hash( i.add( 3 ) ) ) );

	} )().compute( count );
	renderer.compute( init );

	const D = buf.toAttribute();
	const range = bound * 6.4;
	const fall = time.mul( 0.012 * bound ).mul( D.w.mul( 0.7 ).add( 0.3 ) );
	const y = mod( D.y.sub( fall ).add( range * 0.5 ), range ).sub( range * 0.5 );
	const sway = vec3( sin( time.mul( 0.21 ).add( D.w.mul( 40 ) ) ), 0, cos( time.mul( 0.17 ).add( D.w.mul( 31 ) ) ) ).mul( bound * 0.02 );

	const mat = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
	mat.positionNode = vec3( D.x, y, D.z ).add( sway );
	mat.scaleNode = vec2( D.w.mul( 0.7 ).add( 0.3 ).mul( bound * 0.012 ) );
	const tint = varying( mix( vec3( 0.25, 0.55, 0.7 ), vec3( 0.45, 0.4, 0.8 ), D.w ).mul( 0.05 ) );
	mat.colorNode = Fn( () => {

		const d = uv().sub( 0.5 ).mul( 2 );
		const q = dot( d, d );
		return vec4( tint.mul( exp( q.mul( - 6 ) ) ).mul( max( float( 1 ).sub( q ), 0 ) ), 1 );

	} )();

	const sprite = new THREE.Sprite( mat );
	sprite.geometry = new THREE.PlaneGeometry( 1, 1 ); // own quad: Sprite's default geometry is shared module-wide
	sprite.count = count;
	sprite.frustumCulled = false;
	sprite.renderOrder = - 1;

	return {
		object: sprite,
		dispose() {

			init.dispose();
			mat.dispose();
			sprite.geometry.dispose();
			try {

				renderer._attributes?.delete( buf.value );

			} catch ( e ) { /* noop */ }

		},
	};

}

// Abyssal backdrop: dark vertical gradient, a faint cold glow behind the colony, a hint of caustic shimmer.
export function createBackground() {

	return Fn( () => {

		const p = screenUV;
		const top = vec3( 0.0016, 0.0030, 0.0075 );
		const bottom = vec3( 0.0030, 0.0110, 0.0170 );
		const g = mix( bottom, top, smoothstep( 0, 1, p.y ) ).toVar();
		const c = p.sub( vec2( 0.5, 0.52 ) ).mul( vec2( 1.6, 1 ) );
		const glow = exp( dot( c, c ).mul( - 5 ) );
		g.addAssign( vec3( 0.006, 0.016, 0.026 ).mul( glow ) );
		// slow light shafts from above
		const shafts = sin( p.x.mul( 9 ).add( sin( p.x.mul( 3.1 ).add( time.mul( 0.05 ) ) ).mul( 1.5 ) ) ).mul( 0.5 ).add( 0.5 );
		g.addAssign( vec3( 0.002, 0.006, 0.009 ).mul( shafts.pow( 3 ) ).mul( smoothstep( 0.2, 1.0, p.y ) ) );
		return g;

	} )();

}

// Pointer lure: a soft glow marking the attractor / repulsor.
export function createLure() {

	const u = { pos: uniform( new THREE.Vector3() ), amount: uniform( 0 ), color: uniform( new THREE.Color( 1, 0.7, 0.35 ) ), size: uniform( 1 ) };
	const mat = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending } );
	mat.positionNode = u.pos;
	mat.scaleNode = vec2( u.size );
	mat.colorNode = Fn( () => {

		const d = uv().sub( 0.5 ).mul( 2 );
		const q = dot( d, d );
		const ring = exp( abs( q.sqrt().sub( 0.55 ) ).mul( - 18 ) ).mul( 0.6 );
		const core = exp( q.mul( - 30 ) ).mul( 1.5 );
		const halo = exp( q.mul( - 4 ) ).mul( 0.12 );
		return vec4( u.color.mul( ring.add( core ).add( halo ) ).mul( u.amount ).mul( max( float( 1 ).sub( q ), 0 ) ), 1 );

	} )();
	const sprite = new THREE.Sprite( mat );
	sprite.geometry = new THREE.PlaneGeometry( 1, 1 );
	sprite.frustumCulled = false;
	sprite.renderOrder = 10;
	return { object: sprite, uniforms: u, dispose() {

		mat.dispose(); sprite.geometry.dispose();

	} };

}

export { MAX_SPECIES };
