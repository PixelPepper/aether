// Procedural environment for the Primordial Sea: a dark nebula sky baked once into an HDR equirect map,
// analytic pin-sharp stars for primary rays, and a dark obsidian floor with mirror-ish reflections,
// a contact shadow and a bioluminescent light pool under the orb.
import * as THREE from 'three/webgpu';
import {
	Fn, uv, vec2, vec3, vec4, float, texture, equirectUV, mx_fractal_noise_float, mx_noise_float, hash,
	smoothstep, mix, exp, pow, max, min, abs, dot, normalize, length, floor, fract, sin, cos, clamp, If, select, sqrt,
} from 'three/tsl';

export const ORB_C = new THREE.Vector3( 0, 1.12, 0 );
export const ORB_R = 1.0;
export const FLOOR_Y = 0.0;
export const MOON_DIR = new THREE.Vector3( - 0.62, 0.36, - 0.70 ).normalize();
export const GLOW_COLOR = new THREE.Color( 0.08, 1.0, 0.62 );

const moonV = vec3( MOON_DIR.x, MOON_DIR.y, MOON_DIR.z );

// ─────────── sky radiance (used only for the one-time bake) ───────────
const skyRadiance = Fn( ( [ d ] ) => {

	const y = d.y;
	const up = smoothstep( - 0.02, 0.25, y );

	// base gradient + horizon haze
	const col = mix( vec3( 0.0035, 0.0045, 0.010 ), vec3( 0.0012, 0.0016, 0.0045 ), smoothstep( 0.0, 0.9, y ) ).toVar();
	col.addAssign( vec3( 0.010, 0.030, 0.045 ).mul( exp( abs( y ).mul( - 10.0 ) ) ) );
	col.addAssign( vec3( 0.020, 0.010, 0.040 ).mul( exp( abs( y ).mul( - 3.0 ) ) ).mul( 0.5 ) );

	// nebula: domain-warped fBm, concentrated along a tilted galactic band
	const band = exp( pow( dot( d, normalize( vec3( 0.35, 0.82, 0.45 ) ) ), 2.0 ).mul( - 7.0 ) );
	const warp = vec3(
		mx_fractal_noise_float( d.mul( 1.7 ), 3, 2.0, 0.5 ),
		mx_fractal_noise_float( d.mul( 1.7 ).add( 5.2 ), 3, 2.0, 0.5 ),
		mx_fractal_noise_float( d.mul( 1.7 ).add( 11.7 ), 3, 2.0, 0.5 ),
	);
	const q = d.mul( 2.6 ).add( warp.mul( 0.9 ) );
	const n = mx_fractal_noise_float( q, 6, 2.05, 0.52 );
	const n2 = mx_fractal_noise_float( q.mul( 0.7 ).add( 3.3 ), 4, 2.0, 0.5 );
	const dust = smoothstep( 0.05, 0.55, mx_fractal_noise_float( d.mul( 7.0 ).add( warp ), 4, 2.0, 0.5 ) );
	const dens = pow( smoothstep( - 0.25, 0.85, n ), 2.2 ).mul( band.mul( 1.2 ).add( 0.18 ) ).mul( float( 1 ).sub( dust.mul( 0.75 ) ) );
	const violet = vec3( 0.16, 0.045, 0.34 ), magenta = vec3( 0.42, 0.05, 0.24 ), teal = vec3( 0.02, 0.26, 0.32 );
	const hue = mix( violet, magenta, smoothstep( - 0.2, 0.5, n2 ) );
	const nebCol = mix( hue, teal, smoothstep( 0.15, 0.7, n2.negate() ) );
	col.addAssign( nebCol.mul( dens ).mul( 0.42 ).mul( up ) );
	// hot cores
	col.addAssign( vec3( 0.9, 0.55, 0.85 ).mul( pow( smoothstep( 0.45, 1.0, n ), 3.0 ) ).mul( band ).mul( 0.25 ).mul( up ) );

	// soft background stars (sharp ones are analytic)
	const sp = d.mul( 160 ).add( 500 );
	const cell = floor( sp );
	const h = hash( cell.x.add( cell.y.mul( 1031 ) ).add( cell.z.mul( 7919 ) ) );
	const f = fract( sp ).sub( 0.5 );
	const s = smoothstep( 0.35, 0.0, length( f ) ).mul( smoothstep( 0.93, 1.0, h ) );
	col.addAssign( vec3( 0.6, 0.7, 1.0 ).mul( s ).mul( 0.25 ).mul( up ) );

	// moon / key light: small hot disk + halos (gives the fluid & glass a consistent highlight)
	const md = max( dot( d, moonV ), 0.0 );
	col.addAssign( vec3( 0.85, 0.92, 1.0 ).mul( smoothstep( 0.99935, 0.99965, md ) ).mul( 40.0 ) );
	col.addAssign( vec3( 0.55, 0.75, 1.0 ).mul( pow( md, 900.0 ) ).mul( 2.0 ) );
	col.addAssign( vec3( 0.25, 0.4, 0.7 ).mul( pow( md, 40.0 ) ).mul( 0.05 ) );

	// below horizon the bake stores the (unlit) ground for reflections
	const ground = vec3( 0.0025, 0.0035, 0.005 ).add( vec3( 0.010, 0.030, 0.045 ).mul( exp( abs( y ).mul( - 14.0 ) ) ) );
	return select( y.lessThan( 0.0 ), ground, col );

} );

export function bakeEnvironment( renderer, width = 1024 ) {

	const rt = new THREE.RenderTarget( width, width / 2, { type: THREE.HalfFloatType, depthBuffer: false } );
	rt.texture.minFilter = THREE.LinearFilter;
	rt.texture.magFilter = THREE.LinearFilter;
	rt.texture.wrapS = THREE.RepeatWrapping;
	const mat = new THREE.NodeMaterial();
	mat.fragmentNode = Fn( () => {

		const t = uv();
		const phi = t.x.sub( 0.5 ).mul( Math.PI * 2 );
		const th = t.y.sub( 0.5 ).mul( Math.PI );
		const d = vec3( cos( phi ).mul( cos( th ) ), sin( th ), sin( phi ).mul( cos( th ) ) );
		return vec4( skyRadiance( d ), 1 );

	} )();
	const quad = new THREE.QuadMesh( mat );
	const prev = renderer.getRenderTarget();
	renderer.setRenderTarget( rt );
	quad.render( renderer );
	renderer.setRenderTarget( prev );
	mat.dispose();
	return rt;

}

// ─────────── runtime helpers used by the composite ───────────

export function makeEnv( envTex ) {

	const env = Fn( ( [ dir ] ) => texture( envTex, equirectUV( dir ) ).level( 0 ).rgb );

	// pin-sharp stars for primary rays; pixAngle = radians per pixel (keeps them ~1–2 px)
	const stars = Fn( ( [ d, pixAngle ] ) => {

		const out = vec3( 0 ).toVar();
		for ( const [ scale, thr, amp ] of [ [ 260, 0.985, 1.6 ], [ 120, 0.992, 3.5 ] ] ) {

			const sp = d.mul( scale ).add( 1000 );
			const cell = floor( sp );
			const h = hash( cell.x.add( cell.y.mul( 1543 ) ).add( cell.z.mul( 6151 ) ) );
			const h2 = hash( cell.x.mul( 3.1 ).add( cell.y.mul( 917 ) ).add( cell.z.mul( 2741 ) ).add( 77 ) );
			const jit = vec3( h2, fract( h2.mul( 13.7 ) ), fract( h2.mul( 71.3 ) ) ).sub( 0.5 ).mul( 0.6 );
			const dist = length( fract( sp ).sub( 0.5 ).sub( jit ) );
			const rad = max( pixAngle.mul( scale ).mul( 0.9 ), 0.06 );
			const s = smoothstep( rad, 0.0, dist ).mul( smoothstep( thr, 1.0, h ) );
			const tint = mix( vec3( 1.0, 0.75, 0.55 ), vec3( 0.6, 0.8, 1.0 ), h2 );
			out.addAssign( tint.mul( s ).mul( amp ) );

		}

		return out.mul( smoothstep( - 0.02, 0.2, d.y ) );

	} );

	return { env, stars };

}

// analytic ray–sphere; returns vec2(t0, t1) or (-1,-1)
export const raySphere = Fn( ( [ ro, rd, c, r ] ) => {

	const oc = ro.sub( c );
	const b = dot( oc, rd );
	const cc = dot( oc, oc ).sub( r.mul( r ) );
	const h = b.mul( b ).sub( cc );
	const sq = sqrt( max( h, 0.0 ) );
	return select( h.greaterThan( 0.0 ), vec2( b.negate().sub( sq ), b.negate().add( sq ) ), vec2( - 1.0 ) );

} );

/**
 * Background radiance along a ray (sky or floor). `withStars` adds analytic stars (primary rays only).
 * `U` = { glow (bioluminescence level), pixAngle, time }
 */
export function makeBackground( E, U, withStars ) {

	const orbC = vec3( ORB_C.x, ORB_C.y, ORB_C.z );
	const glowC = vec3( GLOW_COLOR.r, GLOW_COLOR.g, GLOW_COLOR.b );

	return Fn( ( [ ro, rd ] ) => {

		const sky = E.env( rd ).toVar();
		if ( withStars ) sky.addAssign( E.stars( rd, U.pixAngle ) );

		const out = sky.toVar();

		If( rd.y.lessThan( - 1e-4 ), () => {

			const t = float( FLOOR_Y ).sub( ro.y ).div( rd.y );
			If( t.greaterThan( 0.0 ), () => {

				const P = ro.add( rd.mul( t ) );
				const rel = P.xz.sub( orbC.xz );
				const r2 = dot( rel, rel );
				const cosT = rd.y.negate();
				const F = float( 0.03 ).add( float( 0.97 ).mul( pow( float( 1 ).sub( cosT ), 5.0 ) ) );

				// mirror reflection of the sky, slightly rippled
				const rip = sin( P.x.mul( 7.0 ).add( sin( P.z.mul( 3.1 ) ).mul( 2.0 ) ).add( U.time.mul( 0.3 ) ) ).mul( sin( P.z.mul( 6.3 ).sub( U.time.mul( 0.23 ) ) ) ).mul( 0.018 );
				const rr = normalize( vec3( rd.x.add( rip ), rd.y.negate(), rd.z.sub( rip ) ) );
				const refl = E.env( rr ).toVar();

				// orb seen in the floor mirror: dark glassy body + a hint of its bioluminescence
				const hit = raySphere( P, rr, orbC, ORB_R );
				If( hit.x.greaterThan( 0.0 ), () => {

					const Q = P.add( rr.mul( hit.x ) );
					const n = normalize( Q.sub( orbC ) );
					const rim = pow( float( 1 ).sub( abs( dot( n, rr ) ) ), 3.0 );
					const lower = smoothstep( 0.3, - 0.6, n.y );
					refl.assign( mix( refl.mul( 0.4 ), vec3( 0.002, 0.018, 0.024 ).add( glowC.mul( U.pool ).mul( 0.35 ).mul( lower ) ), 0.85 ).add( vec3( 0.25, 0.4, 0.6 ).mul( rim ).mul( 0.05 ) ) );

				} );

				// diffuse: dark obsidian + bioluminescent pool + contact shadow
				const pool = glowC.mul( U.pool ).mul( exp( r2.mul( - 1.3 ) ).mul( 0.20 ).add( exp( r2.mul( - 6.0 ) ).mul( 0.25 ) ) );
				const caust = sin( P.x.mul( 9.0 ).add( sin( P.z.mul( 7.0 ).add( U.time.mul( 0.9 ) ) ).mul( 1.6 ) ) ).mul( sin( P.z.mul( 8.0 ).add( sin( P.x.mul( 6.0 ).sub( U.time.mul( 0.7 ) ) ).mul( 1.6 ) ) ) );
				const caustics = glowC.mul( U.pool ).mul( pow( float( 1 ).sub( abs( caust ) ), 8.0 ) ).mul( exp( r2.mul( - 2.5 ) ) ).mul( 0.35 );
				const ao = float( 1 ).sub( exp( r2.mul( - 5.0 ) ).mul( 0.8 ) );
				const base = vec3( 0.0018, 0.0022, 0.0030 );
				const rings = smoothstep( 0.92, 1.0, sin( length( rel ).mul( 18.0 ).sub( U.time.mul( 0.6 ) ) ) ).mul( exp( r2.mul( - 0.35 ) ) ).mul( 0.0035 );
				const floorCol = base.add( pool ).add( caustics ).add( vec3( 0.2, 0.5, 0.6 ).mul( rings ) ).mul( ao ).add( refl.mul( F ).mul( mix( 0.6, 1.0, ao ) ) );

				// distance haze into the horizon colour
				const fog = float( 1 ).sub( exp( t.mul( - 0.045 ) ) );
				out.assign( mix( floorCol, E.env( normalize( vec3( rd.x, 0.002, rd.z ) ) ), fog ) );

			} );

		} );

		return out;

	} );

}
