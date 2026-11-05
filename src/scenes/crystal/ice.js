// Ice shading for the grown crystal + the studio it is photographed in.
//
// The flake is a thin slab whose two faces carry the relief baked from the crystal-mass field (ribs and ridges carry
// more mass). Per fragment we trace the view ray through the slab analytically:
//   top face (normal N) → refraction with a per-channel IOR (dispersion) → bottom face (normal mirrored through the
//   mid-plane) → refraction out, or total internal reflection back through the top.
// The transmitted ray looks up a dark-field ("Rheinberg") illumination: straight-through light is the velvet backdrop
// (so flat ice is nearly invisible, as in macro photographs), while rays deflected by ribs, ridges and edges pick up a
// coloured annular light — the technique Libbrecht uses to photograph snow crystals. Reflections, a faked internal
// reflection off the bottom face, Beer–Lambert absorption (faint cyan), hashed microfacet glints, a rim term, and a
// glow along the freshly grown front complete it.

import * as THREE from 'three/webgpu';
import {
	Fn, If, Discard, uniform, texture, uv, vec2, vec3, vec4, float, positionLocal, positionViewDirection,
	transformNormalToView, normalize, dot, max, min, mix, smoothstep, pow, exp, refract, reflect, select, clamp, abs,
	floor, hash, atan, acos, sin, cos, length, screenUV, mx_noise_float, uint,
} from 'three/tsl';

// view-space light rig (the "studio" moves with the camera)
const L_KEY = new THREE.Vector3( - 0.5, 0.62, 0.6 ).normalize();
const L_FILL = new THREE.Vector3( 0.66, - 0.22, 0.72 ).normalize();
const L_SPARK = new THREE.Vector3( 0.15, 0.35, 0.92 ).normalize();

export function createStudio() {

	const S = {
		// optics
		ior: uniform( 1.31 ), disp: uniform( 0.022 ), // ice n≈1.31; dispersion exaggerated ~4× so it reads on screen
		thick: uniform( 0.024 ), // half-thickness of a rib (β-normalised mass 1) in flake-local units (radius 1)
		bump: uniform( 1.0 ), // relief → normal strength
		nrmR: uniform( 2.2 ), // normal sampling radius in texels
		texel: uniform( 1 / 1024 ),
		absorb: uniform( 1.4 ),
		internal: uniform( 2.5 ),
		ringI: uniform( 1.8 ), ring0: uniform( 0.07 ), ring1: uniform( 0.3 ), ring2: uniform( 0.5 ), ring3: uniform( 1.0 ),
		glint: uniform( 24.0 ), glintRes: uniform( 512 ), glintTilt: uniform( 1.1 ), glintDensity: uniform( 0.6 ),
		rim: uniform( 0.3 ),
		front: uniform( 0.5 ), // quasi-liquid front glow
		flash: uniform( 0 ), flashCol: uniform( new THREE.Color( 1.0, 0.75, 0.45 ) ),
		tanHalf: uniform( Math.tan( THREE.MathUtils.degToRad( 25 ) ) ), aspect: uniform( 16 / 9 ),
		lKey: uniform( L_KEY ), lFill: uniform( L_FILL ), lSpark: uniform( L_SPARK ),
	};

	// ── dark velvet backdrop as a function of view-space direction ──
	const velvet = Fn( ( [ dirIn ] ) => {

		const d = normalize( dirIn.add( vec3( 0, 0, 1e-5 ) ) ); // guard: zero vector from a failed refraction
		const g = smoothstep( - 0.45, 0.55, d.y.mul( 0.85 ).sub( d.x.mul( 0.4 ) ) );
		const base = mix( vec3( 0.006, 0.007, 0.017 ), vec3( 0.016, 0.020, 0.048 ), g );
		const dy = d.y.sub( 0.04 );
		const pool = exp( d.x.mul( d.x ).add( dy.mul( dy ) ).mul( - 7.0 ) ).mul( vec3( 0.010, 0.015, 0.034 ) );
		const n = mx_noise_float( d.mul( vec3( 2.2, 2.2, 1.0 ) ) ).mul( 0.5 ).add( 0.5 );
		const n2 = mx_noise_float( d.mul( 7.0 ).add( 3.1 ) ).mul( 0.5 ).add( 0.5 );
		return base.add( pool ).mul( n.mul( 0.5 ).add( n2.mul( 0.15 ) ).add( 0.62 ) );

	} );

	// ── dark-field annular illumination seen by transmitted rays (relative to the undeviated ray) ──
	const envT = Fn( ( [ dir, straight ] ) => {

		const c = clamp( dot( dir, straight ), - 1, 1 );
		const ang = acos( c );
		const perp = dir.sub( straight.mul( c ) );
		const phi = atan( perp.y, perp.x.add( 1e-6 ) );
		const ring = smoothstep( S.ring0, S.ring1, ang ).mul( float( 1 ).sub( smoothstep( S.ring2, S.ring3, ang ).mul( 0.55 ) ) );
		const warm = vec3( 1.0, 0.55, 0.16 ), cool = vec3( 0.12, 0.6, 1.0 ), vio = vec3( 0.62, 0.3, 1.0 );
		const col = mix( cool, warm, smoothstep( - 0.75, 0.75, sin( phi ) ) ).toVar();
		col.assign( mix( col, vio, pow( abs( cos( phi ) ), 4.0 ).mul( 0.55 ) ) );
		return velvet( dir ).add( col.mul( ring ).mul( S.ringI ) );

	} );

	// ── reflection environment: warm key softbox, cool fill, faint top gradient ──
	const envR = Fn( ( [ dir ] ) => {

		const k = max( dot( dir, S.lKey ), 0 );
		const f = max( dot( dir, S.lFill ), 0 );
		const key = pow( k, 18.0 ).mul( 2.2 ).add( pow( k, 300.0 ).mul( 9.0 ) );
		const fill = pow( f, 14.0 ).mul( 0.9 );
		const top = smoothstep( - 0.1, 1.0, dir.y ).mul( 0.02 );
		return velvet( dir ).add( vec3( 1.0, 0.9, 0.78 ).mul( key ) ).add( vec3( 0.45, 0.72, 1.0 ).mul( fill.add( top ) ) );

	} );

	// background node: view direction of each screen pixel
	const background = Fn( () => {

		const ndc = vec2( screenUV.x.mul( 2 ).sub( 1 ), screenUV.y.mul( 2 ).sub( 1 ).negate() );
		const dir = normalize( vec3( ndc.x.mul( S.tanHalf ).mul( S.aspect ), ndc.y.mul( S.tanHalf ), - 1 ) );
		return vec4( velvet( dir ), 1 );

	} )();

	return { S, velvet, envT, envR, background };

}

// relief height from a bake sample: mass relief, tapered to zero at the crystal edge so the faces meet
const heightOf = ( s ) => s.y.mul( smoothstep( 0.5, 0.85, s.x ) );

/**
 * Ice material for one face of one flake. `P` holds per-flake uniforms (fade, now, seed).
 */
export function createIceMaterial( studio, tex, P, debug = '' ) {

	const { S, envT, envR } = studio;
	const mat = new THREE.MeshBasicNodeMaterial();
	mat.side = THREE.FrontSide;

	if ( debug !== 'tex' ) mat.positionNode = Fn( () => {

		const s = texture( tex, uv() ).level( 0 );
		return vec3( positionLocal.x, positionLocal.y, heightOf( s ).mul( S.thick ) );

	} )();

	mat.colorNode = Fn( () => {

		const uv0 = uv();
		const s = texture( tex, uv0 );
		If( s.x.lessThan( 0.5 ), () => {

			Discard();

		} );

		// sublimation dissolve while a flake is retired
		const gc = floor( uv0.mul( S.glintRes ) );
		const gid = gc.x.add( gc.y.mul( 4099.0 ) ).add( P.seed );
		If( hash( gid ).lessThan( P.fade ), () => {

			Discard();

		} );

		if ( debug === 'tex' ) return vec4( s.x.mul( 0.2 ), s.y.mul( 0.5 ), s.w.add( s.z.mul( 0.02 ) ), 1 );

		const h = heightOf( s );
		const e = S.texel.mul( S.nrmR );
		const hx = heightOf( texture( tex, uv0.add( vec2( e, 0 ) ) ) ).sub( heightOf( texture( tex, uv0.sub( vec2( e, 0 ) ) ) ) );
		const hy = heightOf( texture( tex, uv0.add( vec2( 0, e ) ) ) ).sub( heightOf( texture( tex, uv0.sub( vec2( 0, e ) ) ) ) );
		const k = S.thick.mul( S.bump ).div( e.mul( 4.0 ) ); // uv → local units: ×2, central difference: /2e
		const nL = normalize( vec3( hx.mul( k ).negate(), hy.mul( k ).negate(), 1 ) ).toVar();

		const N = normalize( transformNormalToView( nL ) ).toVar();
		const Pn = normalize( transformNormalToView( vec3( 0, 0, 1 ) ) ).toVar();
		const V = positionViewDirection.toVar();
		const I = V.negate();
		const NdV = clamp( dot( N, V ), 0.0, 1.0 );

		const F0 = 0.03;
		const F = pow( float( 1 ).sub( NdV ), 5.0 ).mul( 1 - F0 ).add( F0 );
		const Nb = N.sub( Pn.mul( dot( N, Pn ).mul( 2.0 ) ) ).toVar(); // exit face: N mirrored through the mid-plane

		const through = ( eta ) => {

			const t1 = refract( I, N, float( 1 ).div( eta ) );
			const t2 = refract( t1, Nb.negate(), eta );
			const r1 = reflect( t1, Nb );
			const t3 = refract( r1, N.negate(), eta );
			const tir = dot( t2, t2 ).lessThan( 0.01 );
			const o = select( tir, t3, t2 );
			// double total internal reflection (steep facets): fall back to the mirror direction
			return { t1, out: select( dot( o, o ).lessThan( 0.01 ), reflect( I, N ), o ), r1, t3 };

		};

		const oR = through( S.ior.sub( S.disp ) );
		const oG = through( S.ior );
		const oB = through( S.ior.add( S.disp ) );
		const trans = vec3( envT( oR.out, I ).x, envT( oG.out, I ).y, envT( oB.out, I ).z );

		// Beer–Lambert: red absorbed first → faint blue-cyan in thick ribs
		const cosT = max( abs( dot( oG.t1, Pn ) ), 0.3 );
		const path = h.mul( 2.0 ).div( cosT );
		const absorb = exp( vec3( 0.55, 0.17, 0.06 ).mul( path ).mul( S.absorb ).negate() );

		const refl = envR( reflect( I, N ) ).mul( F );

		// internal reflection off the bottom face, back out through the top
		const ci = abs( dot( oG.t1, Nb ) );
		const Fi = pow( float( 1 ).sub( ci ), 5.0 ).mul( 1 - F0 ).add( F0 );
		const intern = envR( oG.t3 ).mul( Fi ).mul( S.internal ).mul( absorb ).mul( absorb );

		// microfacet glints: each lattice-scale facet gets a random tilt; facets aligned with a light's half-vector flare
		const r1 = hash( gid.add( 17.0 ) ), r2 = hash( gid.add( 31.0 ) ), r3 = hash( gid.add( 47.0 ) );
		const tilt = vec2( r1.sub( 0.5 ), r2.sub( 0.5 ) ).mul( S.glintTilt );
		const nG = normalize( transformNormalToView( normalize( nL.add( vec3( tilt, 0 ) ) ) ) );
		const g1 = pow( max( dot( nG, normalize( S.lKey.add( V ) ) ), 0 ), 500.0 );
		const g2 = pow( max( dot( nG, normalize( S.lSpark.add( V ) ) ), 0 ), 700.0 );
		const g3 = pow( max( dot( nG, normalize( S.lFill.add( V ) ) ), 0 ), 500.0 );
		const slope = clamp( length( nL.xy ).mul( 3.0 ), 0, 1 );
		const gmask = select( r3.lessThan( S.glintDensity.mul( slope.mul( 0.7 ).add( 0.3 ) ) ), float( 1 ), float( 0 ) );
		const glint = vec3( 1.0, 0.93, 0.85 ).mul( g1 ).add( vec3( 0.85, 0.95, 1.0 ).mul( g2 ) ).add( vec3( 0.6, 0.8, 1.0 ).mul( g3.mul( 0.6 ) ) )
			.mul( gmask ).mul( S.glint );

		const rim = vec3( 0.45, 0.75, 1.0 ).mul( pow( float( 1 ).sub( NdV ), 3.0 ) ).mul( S.rim );

		// faint volume scattering in thick ice
		const scatter = vec3( 0.010, 0.030, 0.055 ).mul( h );

		// glow of the freshly frozen front: young ice (born < ~1 s ago) and the quasi-liquid layer just outside it
		const age = P.now.sub( s.z );
		const fresh = exp( age.mul( - 1.6 ) ).mul( smoothstep( 0.5, 0.7, s.x ) );
		const glowCol = mix( vec3( 0.55, 0.85, 1.0 ), S.flashCol, S.flash );
		const front = glowCol.mul( fresh.mul( 0.6 ).add( s.w.mul( s.w ).mul( 0.3 ) ) ).mul( S.front ).mul( S.flash.mul( 2.5 ).add( 1.0 ) );

		const col = trans.mul( absorb ).mul( float( 1 ).sub( F ) ).add( refl ).add( intern ).add( glint ).add( rim ).add( scatter ).add( front );
		return vec4( col.mul( float( 1 ).sub( P.fade.mul( 0.6 ) ) ), 1 );

	} )();

	return mat;

}

/**
 * Additive halo around a flake: the quasi-liquid boundary layer of the growth front, plus a shock ring when the
 * cloud layer changes.
 */
export function createHaloMaterial( studio, tex, P ) {

	const { S } = studio;
	const mat = new THREE.MeshBasicNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );

	mat.colorNode = Fn( () => {

		const uv0 = uv();
		const t = S.texel.mul( 2.5 );
		const q = texture( tex, uv0 ).w.mul( 0.4 )
			.add( texture( tex, uv0.add( vec2( t, 0 ) ) ).w.mul( 0.15 ) ).add( texture( tex, uv0.sub( vec2( t, 0 ) ) ).w.mul( 0.15 ) )
			.add( texture( tex, uv0.add( vec2( 0, t ) ) ).w.mul( 0.15 ) ).add( texture( tex, uv0.sub( vec2( 0, t ) ) ).w.mul( 0.15 ) );
		const glowCol = mix( vec3( 0.35, 0.7, 1.0 ), S.flashCol, S.flash );
		const front = glowCol.mul( q.mul( q ).mul( 0.6 ) ).mul( S.front ).mul( S.flash.mul( 3.0 ).add( 1.0 ) );

		// expanding hexagonal shock front from a cloud-layer change
		const p = uv0.mul( 2 ).sub( 1 );
		const ap = abs( p );
		const hexd = max( ap.x.mul( 0.8660254 ).add( ap.y.mul( 0.5 ) ), ap.y );
		const rr = P.ring;
		const ringW = 0.035;
		const dr = hexd.sub( rr ).div( ringW );
		const ring = exp( dr.mul( dr ).negate() ).mul( S.flash ).mul( smoothstep( 0.0, 0.15, rr ) );
		const col = front.add( S.flashCol.mul( ring ).mul( 0.9 ) );
		return vec4( col.mul( float( 1 ).sub( P.fade ) ), 1 );

	} )();

	return mat;

}
