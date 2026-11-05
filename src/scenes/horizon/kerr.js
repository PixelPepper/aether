// CPU-side Kerr physics for the Event Horizon chapter (double precision, units G = c = M = 1).
//
// Everything here mirrors the GPU integrator in ./shaders.js so the CPU can (a) build the observer's
// orthonormal tetrad that turns pixels into initial photon momenta, (b) trace the single geodesic under the
// mouse cursor for picking, and (c) precompute the Novikov–Thorne / Page–Thorne radial temperature profile.
//
// Coordinates: ingoing Kerr–Schild Cartesian (t, x, y, z), metric g = η + f l⊗l with
//   r⁴ − (ρ² − a²) r² − a² z² = 0,   f = 2 r³ / (r⁴ + a² z²),   l = (1, (r x + a y)/(r²+a²), (r y − a x)/(r²+a²), z/r).
// The spin axis is +z.  Photons are traced from the camera outward, which is the time reverse of the physical
// photon; time reversal of Kerr(a) is Kerr(−a), so the integrator runs with a_trace = −a (see horizon.js).

export function kerrR( x, y, z, a ) {

	const a2 = a * a;
	const k = x * x + y * y + z * z - a2;
	return Math.sqrt( 0.5 * ( k + Math.sqrt( k * k + 4 * a2 * z * z ) ) );

}

export const horizonRadius = ( a ) => 1 + Math.sqrt( Math.max( 0, 1 - a * a ) );

// Prograde ISCO for signed spin (a < 0 ⇒ retrograde disk). Bardeen, Press & Teukolsky 1972.
export function iscoRadius( a ) {

	const z1 = 1 + Math.cbrt( 1 - a * a ) * ( Math.cbrt( 1 + a ) + Math.cbrt( 1 - a ) );
	const z2 = Math.sqrt( 3 * a * a + z1 * z1 );
	return 3 + z2 - Math.sign( a ) * Math.sqrt( ( 3 - z1 ) * ( 3 + z1 + 2 * z2 ) );

}

// Equatorial circular photon orbit (prograde for a>0).
export function photonOrbit( a ) {

	return 2 * ( 1 + Math.cos( ( 2 / 3 ) * Math.acos( - a ) ) );

}

// Page & Thorne (1974) flux of a Novikov–Thorne disk, F(r) ∝ 1/(x⁴(x³−3x+2a)) · [x − x0 − 3a/2 ln(x/x0) − Σ c_i ln((x−x_i)/(x0−x_i))]
// with x = √r and x_i the roots of x³ − 3x + 2a = 0.  Returns a function of r (arbitrary normalisation).
export function pageThorneFlux( aIn ) {

	const a = Math.abs( aIn ) < 1e-5 ? 1e-5 : aIn;
	const x0 = Math.sqrt( iscoRadius( a ) );
	const th = Math.acos( a ) / 3;
	const x1 = 2 * Math.cos( th - Math.PI / 3 );
	const x2 = 2 * Math.cos( th + Math.PI / 3 );
	const x3 = - 2 * Math.cos( th );
	const c1 = 3 * ( x1 - a ) ** 2 / ( x1 * ( x1 - x2 ) * ( x1 - x3 ) );
	const c2 = 3 * ( x2 - a ) ** 2 / ( x2 * ( x2 - x1 ) * ( x2 - x3 ) );
	const c3 = 3 * ( x3 - a ) ** 2 / ( x3 * ( x3 - x1 ) * ( x3 - x2 ) );

	return ( r ) => {

		const x = Math.sqrt( r );
		if ( x <= x0 ) return 0;
		const br = x - x0 - 1.5 * a * Math.log( x / x0 )
			- c1 * Math.log( ( x - x1 ) / ( x0 - x1 ) )
			- c2 * Math.log( ( x - x2 ) / ( x0 - x2 ) )
			- c3 * Math.log( ( x - x3 ) / ( x0 - x3 ) );
		return Math.max( 0, br / ( x ** 4 * ( x ** 3 - 3 * x + 2 * a ) ) );

	};

}

// Kerr–Schild covariant metric at a point (index order t, x, y, z) plus the scalars we need.
export function ksMetric( x, y, z, a ) {

	const r = kerrR( x, y, z, a );
	const S = r * r + a * a;
	const l = [ 1, ( r * x + a * y ) / S, ( r * y - a * x ) / S, z / r ];
	const f = 2 * r * r * r / ( r ** 4 + a * a * z * z );
	const g = [];
	for ( let i = 0; i < 4; i ++ ) {

		g.push( [ 0, 0, 0, 0 ] );
		for ( let j = 0; j < 4; j ++ ) g[ i ][ j ] = ( i === j ? ( i === 0 ? - 1 : 1 ) : 0 ) + f * l[ i ] * l[ j ];

	}

	return { g, f, r };

}

const dot4 = ( g, u, v ) => {

	let s = 0;
	for ( let i = 0; i < 4; i ++ ) for ( let j = 0; j < 4; j ++ ) s += u[ i ] * g[ i ][ j ] * v[ j ];
	return s;

};

const lower = ( g, v ) => [ 0, 1, 2, 3 ].map( ( i ) => g[ i ][ 0 ] * v[ 0 ] + g[ i ][ 1 ] * v[ 1 ] + g[ i ][ 2 ] * v[ 2 ] + g[ i ][ 3 ] * v[ 3 ] );

// Orthonormal tetrad of a static observer (u ∝ ∂t) at camera position `pos`, oriented by the camera's
// Euclidean (coordinate) forward / up / right vectors.  Returns covariant components of u, e_right, e_up, e_fwd
// (each as [t, x, y, z]) so that a pixel direction n gives p_μ = u_μ + n^i e_iμ (null, unit energy for the observer).
export function staticTetrad( pos, fwd, up, right, a ) {

	const { g, f } = ksMetric( pos[ 0 ], pos[ 1 ], pos[ 2 ], a );
	const ff = Math.min( f, 0.97 ); // static observers exist only outside the ergosphere (f < 1); camera is kept far outside
	const u = [ 1 / Math.sqrt( 1 - ff ), 0, 0, 0 ];
	const basis = [ { v: u, n: - 1 } ];
	const ortho = ( vec ) => {

		let w = [ 0, vec[ 0 ], vec[ 1 ], vec[ 2 ] ];
		for ( const { v, n } of basis ) {

			const s = dot4( g, w, v ) / n;
			w = w.map( ( c, i ) => c - s * v[ i ] );

		}

		const nn = Math.sqrt( Math.max( 1e-12, dot4( g, w, w ) ) );
		w = w.map( ( c ) => c / nn );
		basis.push( { v: w, n: 1 } );
		return w;

	};

	const eF = ortho( fwd );
	const eU = ortho( up );
	const eR = ortho( right );
	return { u: lower( g, u ), eR: lower( g, eR ), eU: lower( g, eU ), eF: lower( g, eF ), f };

}

// Hamiltonian right-hand side, identical to kerr_rhs() in the WGSL.
export function rhs( x, p, a, E, out ) {

	const a2 = a * a, z2 = x[ 2 ] * x[ 2 ];
	const k = x[ 0 ] * x[ 0 ] + x[ 1 ] * x[ 1 ] + z2 - a2;
	const D = Math.sqrt( k * k + 4 * a2 * z2 );
	const r2 = 0.5 * ( k + D ), r = Math.sqrt( r2 );
	const S = r2 + a2, iS = 1 / S;
	const lx = ( r * x[ 0 ] + a * x[ 1 ] ) * iS, ly = ( r * x[ 1 ] - a * x[ 0 ] ) * iS, lz = x[ 2 ] / r;
	const W = r2 * r2 + a2 * z2;
	const f = 2 * r2 * r / W;
	const L = E + lx * p[ 0 ] + ly * p[ 1 ] + lz * p[ 2 ];
	const grx = x[ 0 ] * r / D, gry = x[ 1 ] * r / D, grz = x[ 2 ] * S / ( r * D );
	const cf = f / ( r * W ), q = 3 * a2 * z2 - r2 * r2;
	const gfx = cf * q * grx, gfy = cf * q * gry, gfz = cf * ( q * grz - 2 * a2 * r * x[ 2 ] );
	const A = ( x[ 0 ] * p[ 0 ] + x[ 1 ] * p[ 1 ] - 2 * r * ( lx * p[ 0 ] + ly * p[ 1 ] ) ) * iS - p[ 2 ] * x[ 2 ] / r2;
	const gLx = A * grx + ( r * p[ 0 ] - a * p[ 1 ] ) * iS;
	const gLy = A * gry + ( r * p[ 1 ] + a * p[ 0 ] ) * iS;
	const gLz = A * grz + p[ 2 ] / r;
	out[ 0 ] = p[ 0 ] - f * L * lx; out[ 1 ] = p[ 1 ] - f * L * ly; out[ 2 ] = p[ 2 ] - f * L * lz;
	out[ 3 ] = 0.5 * L * L * gfx + f * L * gLx;
	out[ 4 ] = 0.5 * L * L * gfy + f * L * gLy;
	out[ 5 ] = 0.5 * L * L * gfz + f * L * gLz;
	return r;

}

// Trace one photon (time-reversed, spin aTrace) and report the first crossing of the equatorial plane
// inside [rMin, rMax]; returns { r, phi } or { r, phi, outside: true } for the first crossing at any radius, or null.
export function pickDisk( pos, p0, E, aTrace, rPlus, rMin, rMax ) {

	const x = pos.slice(), p = p0.slice();
	const k1 = new Float64Array( 6 ), k2 = new Float64Array( 6 ), k3 = new Float64Array( 6 ), k4 = new Float64Array( 6 );
	const xt = [ 0, 0, 0 ], pt = [ 0, 0, 0 ];
	let firstAny = null;
	for ( let i = 0; i < 4000; i ++ ) {

		const r0 = rhs( x, p, aTrace, E, k1 );
		const v = Math.hypot( k1[ 0 ], k1[ 1 ], k1[ 2 ] );
		const h = 0.04 * Math.max( r0 - 0.72 * rPlus, 0.04 ) / Math.max( v, 1e-6 );
		const stage = ( src, s ) => {

			for ( let j = 0; j < 3; j ++ ) {

				xt[ j ] = x[ j ] + s * src[ j ]; pt[ j ] = p[ j ] + s * src[ j + 3 ];

			}

		};

		stage( k1, 0.5 * h ); rhs( xt, pt, aTrace, E, k2 );
		stage( k2, 0.5 * h ); rhs( xt, pt, aTrace, E, k3 );
		stage( k3, h ); rhs( xt, pt, aTrace, E, k4 );
		const zOld = x[ 2 ];
		const xOld = x.slice();
		for ( let j = 0; j < 3; j ++ ) {

			x[ j ] += h / 6 * ( k1[ j ] + 2 * k2[ j ] + 2 * k3[ j ] + k4[ j ] );
			p[ j ] += h / 6 * ( k1[ j + 3 ] + 2 * k2[ j + 3 ] + 2 * k3[ j + 3 ] + k4[ j + 3 ] );

		}

		if ( zOld * x[ 2 ] < 0 ) {

			const t = zOld / ( zOld - x[ 2 ] );
			const hx = xOld[ 0 ] + t * ( x[ 0 ] - xOld[ 0 ] ), hy = xOld[ 1 ] + t * ( x[ 1 ] - xOld[ 1 ] );
			const rh = kerrR( hx, hy, 0, aTrace );
			const phi = Math.atan2( hy, hx );
			if ( rh >= rMin && rh <= rMax ) return { r: rh, phi };
			if ( ! firstAny ) firstAny = { r: rh, phi, outside: true };

		}

		const rn = kerrR( x[ 0 ], x[ 1 ], x[ 2 ], aTrace );
		if ( rn < rPlus * 1.01 ) break;
		if ( rn > 200 ) break;

	}

	return firstAny;

}
