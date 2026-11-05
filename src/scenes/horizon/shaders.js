// Hand-written WGSL for the hot inner loops, exposed to TSL through wgsl()/wgslFn().
// TSL drives the kernels (loops, buffers, textures, compositing); these functions are pure math.

import { wgsl, wgslFn } from 'three/tsl';

// ───────────────────────── Kerr–Schild geodesics ─────────────────────────
const kerrCommon = wgsl( /* wgsl */`
fn kerr_r_( x: vec3f, a: f32 ) -> f32 {
	let a2 = a * a;
	let k = dot( x, x ) - a2;
	return sqrt( 0.5 * ( k + sqrt( k * k + 4.0 * a2 * x.z * x.z ) ) );
}

struct KDeriv { dx: vec3f, dp: vec3f };

// Hamiltonian H = ½ g^{μν} p_μ p_ν with g^{μν} = η^{μν} − f l^μ l^ν (Kerr–Schild), p_t = −E conserved.
//   dx/dλ = p − f L l,   dp/dλ = ½ L² ∇f + f L ∇L,   L = l^μ p_μ = E + l·p
fn kerr_rhs( x: vec3f, p: vec3f, a: f32, E: f32 ) -> KDeriv {
	let a2 = a * a;
	let z2 = x.z * x.z;
	let k = dot( x, x ) - a2;
	let D = sqrt( k * k + 4.0 * a2 * z2 );
	let r2 = 0.5 * ( k + D );
	let r = sqrt( r2 );
	let S = r2 + a2;
	let iS = 1.0 / S;
	let l = vec3f( ( r * x.x + a * x.y ) * iS, ( r * x.y - a * x.x ) * iS, x.z / r );
	let W = r2 * r2 + a2 * z2;
	let f = 2.0 * r2 * r / W;
	let L = E + dot( l, p );
	let gr = vec3f( x.x * r, x.y * r, x.z * S / r ) / D;
	let gf = ( f / ( r * W ) ) * ( ( 3.0 * a2 * z2 - r2 * r2 ) * gr - vec3f( 0.0, 0.0, 2.0 * a2 * r * x.z ) );
	let A = ( x.x * p.x + x.y * p.y - 2.0 * r * ( l.x * p.x + l.y * p.y ) ) * iS - p.z * x.z / r2;
	let gL = A * gr + vec3f( ( r * p.x - a * p.y ) * iS, ( r * p.y + a * p.x ) * iS, p.z / r );
	var o: KDeriv;
	o.dx = p - f * L * l;
	o.dp = 0.5 * L * L * gf + f * L * gL;
	return o;
}
` );

// One RK4 step with a geometric step-size law: coordinate step ∝ (r − 0.72 r₊), i.e. ~k·r far away and
// much finer near the horizon / photon shell.  prm = (a_trace, E, k, r₊).  Returns columns (x', p', (r', h, r)).
export const geoStep = wgslFn( /* wgsl */`
fn geo_step( x: vec3f, p: vec3f, prm: vec4f ) -> mat3x3f {
	let a = prm.x;
	let E = prm.y;
	let r0 = kerr_r_( x, a );
	let k1 = kerr_rhs( x, p, a, E );
	let hc = prm.z * max( r0 - 0.72 * prm.w, 0.04 );
	let h = hc / max( length( k1.dx ), 1e-4 );
	let k2 = kerr_rhs( x + 0.5 * h * k1.dx, p + 0.5 * h * k1.dp, a, E );
	let k3 = kerr_rhs( x + 0.5 * h * k2.dx, p + 0.5 * h * k2.dp, a, E );
	let k4 = kerr_rhs( x + h * k3.dx, p + h * k3.dp, a, E );
	let xn = x + ( h / 6.0 ) * ( k1.dx + 2.0 * ( k2.dx + k3.dx ) + k4.dx );
	let pn = p + ( h / 6.0 ) * ( k1.dp + 2.0 * ( k2.dp + k3.dp ) + k4.dp );
	return mat3x3f( xn, pn, vec3f( kerr_r_( xn, a ), h, r0 ) );
}
`, [ kerrCommon ] );

export const kerrR = wgslFn( /* wgsl */`
fn kerr_r( x: vec3f, a: f32 ) -> f32 {
	return kerr_r_( x, a );
}
`, [ kerrCommon ] );

// Weak-field completion of the bending from the termination radius to infinity:
//   δ = (2M/b)(1 − s/r) toward the hole (first order in M/b; terminated at r ≳ 60 M so the residual is tiny).
export const escapeDir = wgslFn( /* wgsl */`
fn escape_dir( x: vec3f, v: vec3f ) -> vec3f {
	let d = normalize( v );
	let r = length( x );
	let s0 = dot( x, d );
	let bv = x - s0 * d;
	let b2 = max( dot( bv, bv ), 1e-6 );
	let delta = ( 2.0 / b2 ) * ( 1.0 - s0 / r );
	return normalize( d - delta * bv );
}
` );

// Redshift g = ν_obs/ν_emit of a photon (E, L_z of the traced, time-reversed ray) emitted by gas on a prograde
// circular equatorial orbit of radius r around spin a (physical), seen by our static observer.
// mode 2: full (Doppler + gravitational + transverse), 1: no Doppler (gravitational + time dilation), 0: none.
export const diskG = wgslFn( /* wgsl */`
fn disk_g( r: f32, E: f32, Lz: f32, a: f32, mode: f32 ) -> f32 {
	let sr = sqrt( r );
	let r15 = r * sr;
	let ut = ( r15 + a ) / ( sqrt( r15 ) * sqrt( max( r15 - 3.0 * sr + 2.0 * a, 1e-4 ) ) );
	let gFull = 1.0 / max( ut * ( E + Lz / ( r15 + a ) ), 1e-4 );
	let gGrav = 1.0 / ( ut * E );
	return select( select( 1.0, gGrav, mode > 0.5 ), gFull, mode > 1.5 );
}
` );

// ───────────────────────── noise ─────────────────────────
const noiseCommon = wgsl( /* wgsl */`
fn hz_pcg3( v0: vec3u ) -> vec3u {
	var v = v0 * 1664525u + 1013904223u;
	v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
	v = v ^ ( v >> vec3u( 16u ) );
	v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
	return v;
}
fn hz_h1( p: vec3i ) -> f32 {
	return f32( hz_pcg3( bitcast<vec3u>( p ) ).x >> 8u ) * ( 1.0 / 16777216.0 );
}
fn hz_h3( p: vec3u ) -> vec3f {
	return vec3f( hz_pcg3( p ) >> vec3u( 8u ) ) * ( 1.0 / 16777216.0 );
}
fn hz_vnoise( p: vec3f ) -> f32 {
	let i = vec3i( floor( p ) );
	let f = fract( p );
	let u = f * f * f * ( f * ( f * 6.0 - 15.0 ) + 10.0 );
	let n000 = hz_h1( i );
	let n100 = hz_h1( i + vec3i( 1, 0, 0 ) );
	let n010 = hz_h1( i + vec3i( 0, 1, 0 ) );
	let n110 = hz_h1( i + vec3i( 1, 1, 0 ) );
	let n001 = hz_h1( i + vec3i( 0, 0, 1 ) );
	let n101 = hz_h1( i + vec3i( 1, 0, 1 ) );
	let n011 = hz_h1( i + vec3i( 0, 1, 1 ) );
	let n111 = hz_h1( i + vec3i( 1, 1, 1 ) );
	return mix( mix( mix( n000, n100, u.x ), mix( n010, n110, u.x ), u.y ),
	            mix( mix( n001, n101, u.x ), mix( n011, n111, u.x ), u.y ), u.z );
}
fn hz_fbm( p0: vec3f, oct: i32 ) -> f32 {
	var s = 0.0;
	var amp = 0.5;
	var p = p0;
	var norm = 0.0;
	for ( var i = 0; i < oct; i++ ) {
		s += amp * hz_vnoise( p );
		norm += amp;
		p = p * 2.07 + vec3f( 5.31, 1.73, 7.97 );
		amp *= 0.5;
	}
	return s / norm;
}
` );

// Turbulent surface density of the disk in the co-rotating frame.  Keplerian differential rotation
// Ω(r) = 1/(r^{3/2} + a) advects the pattern; two staggered advection phases are cross-faded (flow-noise trick)
// so the shear stays bounded while the texture visibly winds into trailing spirals.
export const diskTurb = wgslFn( /* wgsl */`
fn disk_turb( r: f32, phi: f32, t: f32, a: f32, period: f32 ) -> f32 {
	let om = 1.0 / ( r * sqrt( r ) + a );
	let c = t / period;
	let f1 = fract( c );
	let f2 = fract( c + 0.5 );
	let s1 = floor( c );
	let s2 = floor( c + 0.5 );
	let w1 = 1.0 - abs( 2.0 * f1 - 1.0 );
	let lr = log( r );
	let ph1 = phi - om * f1 * period;
	let ph2 = phi - om * f2 * period;
	let q1 = vec3f( cos( ph1 ) * 1.7, sin( ph1 ) * 1.7, lr * 7.5 + s1 * 3.17 );
	let q2 = vec3f( cos( ph2 ) * 1.7, sin( ph2 ) * 1.7, lr * 7.5 + s2 * 3.17 + 1.37 );
	let n1 = hz_fbm( q1 * vec3f( 1.0, 1.0, 1.0 ), 5 );
	let n2 = hz_fbm( q2, 5 );
	let w2 = 1.0 - w1;
	let n = 0.5 + ( w1 * n1 + w2 * n2 - 0.5 ) / sqrt( w1 * w1 + w2 * w2 );
	return clamp( n, 0.0, 1.0 );
}
`, [ noiseCommon ] );

// ───────────────────────── sky: Milky-Way band + two star layers ─────────────────────────
// Stars are discs of fixed solid angle on the celestial sphere, rendered with a Gaussian of width
// max(σ★, pixel footprint on the sky).  Because the footprint comes from the neighbouring rays' *lensed* escape
// directions, the image of each star carries exactly the lensing magnification (surface brightness is conserved),
// and stars smear into arcs / Einstein rings near the critical curve.  When the footprint exceeds a grid cell the
// layer smoothly falls back to its mean radiance (a lensing-aware mip level).
const skyCommon = wgsl( /* wgsl */`
fn hz_tint( T: f32 ) -> vec3f {
	let l = vec3f( 0.612, 0.549, 0.464 );
	let b = 1.0 / ( l * l * l * l * l * ( exp( 14388.0 / ( l * T ) ) - 1.0 ) );
	return b / ( 0.2126 * b.x + 0.7152 * b.y + 0.0722 * b.z );
}
fn hz_face_dir( face: u32, uv: vec2f ) -> vec3f {
	switch face {
		case 0u: { return vec3f( 1.0, uv.x, uv.y ); }
		case 1u: { return vec3f( -1.0, uv.x, uv.y ); }
		case 2u: { return vec3f( uv.x, 1.0, uv.y ); }
		case 3u: { return vec3f( uv.x, -1.0, uv.y ); }
		case 4u: { return vec3f( uv.x, uv.y, 1.0 ); }
		default: { return vec3f( uv.x, uv.y, -1.0 ); }
	}
}
fn hz_stars( d: vec3f, fp: f32, N: f32, seed: u32, dens: f32, flux: f32 ) -> vec3f {
	let ad = abs( d );
	var face = 0u;
	var uv = vec2f( 0.0 );
	if ( ad.x >= ad.y && ad.x >= ad.z ) { face = select( 1u, 0u, d.x > 0.0 ); uv = d.yz / ad.x; }
	else if ( ad.y >= ad.z ) { face = select( 3u, 2u, d.y > 0.0 ); uv = d.xz / ad.y; }
	else { face = select( 5u, 4u, d.z > 0.0 ); uv = d.xy / ad.z; }
	let g = ( uv * 0.5 + 0.5 ) * N;
	let c = floor( g );
	let cellAng = 2.0 / N;
	let sig = max( fp * 0.55, 2.5e-4 );
	let wm = smoothstep( 0.22, 0.6, sig / cellAng );
	// mean radiance of the layer: dens · <flux> / Ω_cell   (<u^-1.5> over [0.01,1] /1000 ≈ 0.018)
	let mean = hz_tint( 6500.0 ) * dens * flux * 0.018 / ( cellAng * cellAng * 0.75 );
	var col = vec3f( 0.0 );
	if ( wm < 0.999 ) {
		for ( var j = -1; j <= 1; j++ ) {
			for ( var i = -1; i <= 1; i++ ) {
				let cc = c + vec2f( f32( i ), f32( j ) );
				if ( any( cc < vec2f( 0.0 ) ) || any( cc >= vec2f( N ) ) ) { continue; }
				let cu = vec2u( cc );
				let h = hz_h3( vec3u( cu.x, cu.y, face * 7919u + seed ) );
				if ( h.z > dens ) { continue; }
				let suv = ( cc + 0.15 + 0.7 * h.xy ) / N * 2.0 - 1.0;
				let sd = normalize( hz_face_dir( face, suv ) );
				let dv = sd - d;
				let ang2 = dot( dv, dv );
				let h2 = hz_h3( vec3u( cu.x + 911u, cu.y + 37u, face + seed * 131u ) );
				let fl = flux * pow( max( h2.x, 0.01 ), -1.5 ) * 0.001;
				let T = 2600.0 + 26000.0 * h2.y * h2.y * h2.y;
				col += hz_tint( T ) * fl * exp( -0.5 * ang2 / ( sig * sig ) ) / ( 6.2831853 * sig * sig );
			}
		}
	}
	return mix( col, mean, wm );
}
fn hz_galaxy( d: vec3f ) -> vec3f {
	let ng = normalize( vec3f( 0.22, 0.93, -0.30 ) );
	let gc = normalize( cross( ng, vec3f( 0.0, 0.0, 1.0 ) ) );
	let sb = dot( d, ng );
	let core = max( dot( d, gc ), 0.0 );
	let n1 = hz_fbm( d * 3.1 + vec3f( 2.0 ), 5 );
	let n2 = hz_fbm( d * 7.3 + vec3f( 11.0 ), 4 );
	let n3 = hz_fbm( d * 17.0 + vec3f( 5.0 ), 3 );
	let width = 0.10 + 0.05 * n1 + 0.10 * pow( core, 6.0 );
	let band = exp( -0.5 * sb * sb / ( width * width ) );
	let bulge = pow( core, 10.0 ) * exp( -0.5 * sb * sb / 0.03 );
	let dust = smoothstep( 0.42, 0.68, n2 ) * exp( -0.5 * sb * sb / ( 0.035 * 0.035 ) ) * ( 0.6 + 0.4 * n3 );
	let clumps = pow( n3, 3.0 ) * 2.0;
	var col = band * ( 0.35 + 0.65 * n1 ) * ( 0.6 + clumps ) * mix( vec3f( 0.55, 0.65, 1.0 ), vec3f( 1.0, 0.82, 0.62 ), 0.3 + 0.7 * pow( core, 3.0 ) );
	col += bulge * vec3f( 1.0, 0.78, 0.5 ) * 1.6;
	col *= 1.0 - 0.85 * dust;
	let hii = smoothstep( 0.72, 0.86, n2 * n3 * 1.6 ) * band;
	col += hii * vec3f( 0.9, 0.25, 0.35 ) * 0.5;
	let haze = 0.05 * ( 0.5 + 0.5 * n1 ) * vec3f( 0.45, 0.5, 0.75 );
	return col + haze;
}
` , [ noiseCommon ] );

export const skyColor = wgslFn( /* wgsl */`
fn sky_color( d: vec3f, fp: f32, prm: vec4f ) -> vec3f {
	// prm = (galaxy brightness, star brightness, unused, unused)
	let gal = hz_galaxy( d ) * prm.x;
	let s1 = hz_stars( d, fp, 210.0, 17u, 0.55, 1.0e-6 * prm.y );
	let s2 = hz_stars( d, fp, 42.0, 71u, 0.35, 1.6e-5 * prm.y );
	return gal + s1 + s2;
}
`, [ skyCommon ] );
