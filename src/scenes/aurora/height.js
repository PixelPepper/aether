// Aurora — procedural fjord heightfield (pure JS, km). A glacial trough winding north from the observer,
// ridged-multifractal walls, a low range closing the far end, a rocky promontory to stand on.

function makePerm( seed ) {

	const p = new Uint8Array( 512 );
	const a = Array.from( { length: 256 }, ( _, i ) => i );
	let s = seed >>> 0;
	for ( let i = 255; i > 0; i -- ) {

		s = ( s * 1664525 + 1013904223 ) >>> 0;
		const j = s % ( i + 1 );
		[ a[ i ], a[ j ] ] = [ a[ j ], a[ i ] ];

	}

	for ( let i = 0; i < 512; i ++ ) p[ i ] = a[ i & 255 ];
	return p;

}

const P = makePerm( 1337 );
const GX = new Float32Array( 256 ), GY = new Float32Array( 256 );
for ( let i = 0; i < 256; i ++ ) {

	const a = ( P[ i ] / 256 ) * Math.PI * 2 + i * 0.618;
	GX[ i ] = Math.cos( a ); GY[ i ] = Math.sin( a );

}

// 2D gradient noise in [-1, 1]
export function noise2( x, y ) {

	const xi = Math.floor( x ), yi = Math.floor( y );
	const xf = x - xi, yf = y - yi;
	const u = xf * xf * xf * ( xf * ( xf * 6 - 15 ) + 10 );
	const v = yf * yf * yf * ( yf * ( yf * 6 - 15 ) + 10 );
	const X = xi & 255, Y = yi & 255;
	const g = ( h, dx, dy ) => GX[ h ] * dx + GY[ h ] * dy;
	const a = g( P[ P[ X ] + Y ], xf, yf );
	const b = g( P[ P[ X + 1 ] + Y ], xf - 1, yf );
	const c = g( P[ P[ X ] + Y + 1 ], xf, yf - 1 );
	const d = g( P[ P[ X + 1 ] + Y + 1 ], xf - 1, yf - 1 );
	return 1.42 * ( a + u * ( b - a ) + v * ( c - a ) + u * v * ( a - b - c + d ) );

}

export function fbm( x, y, oct = 5 ) {

	let s = 0, a = 0.5, f = 1;
	for ( let i = 0; i < oct; i ++ ) {

		s += a * noise2( x * f, y * f );
		f *= 2.03; a *= 0.5;

	}

	return s;

}

// ridged multifractal (sharp arêtes), returns ~[0, 1]
export function ridged( x, y, oct = 7 ) {

	let s = 0, a = 0.55, f = 1, w = 1;
	for ( let i = 0; i < oct; i ++ ) {

		let n = 1 - Math.abs( noise2( x * f + i * 17.3, y * f - i * 9.1 ) );
		n *= n;
		s += n * a * w;
		w = Math.min( 1, n * 1.6 );
		f *= 2.07; a *= 0.48;

	}

	return s;

}

const smooth = ( a, b, x ) => {

	const t = Math.min( 1, Math.max( 0, ( x - a ) / ( b - a ) ) );
	return t * t * ( 3 - 2 * t );

};

export const EYE = 0.0075; // eye height above the water (km)

// fjord centre line and half width (z is south-positive; the fjord runs toward −z)
export function channel( z ) {

	const n = Math.max( 0, - z );
	const xc = 0.55 * Math.sin( n * 0.13 + 0.4 ) * Math.min( 1, n / 4 ) + 0.9 * Math.sin( n * 0.045 - 0.6 ) * Math.min( 1, n / 12 ) - 0.15;
	const hw = 0.55 + n * 0.075 + 0.25 * Math.sin( n * 0.21 );
	return [ xc, hw ];

}

export function height( x, z ) {

	const [ xc, hw ] = channel( z );
	const d = Math.abs( x - xc ) / hw; // 0 at the fjord axis, 1 at the shore
	const dist = Math.hypot( x, z );
	// walls: ridged massifs whose height grows away from the water and with distance from the observer
	const r = ridged( x * 0.32 + 3.1, z * 0.32 - 7.3 );
	const big = 0.5 + 0.5 * fbm( x * 0.06 + 11, z * 0.06 - 4, 3 );
	const wall = smooth( 0.82, 2.6, d );
	const reach = Math.min( 1, 0.12 + dist * 0.11 ); // keep the nearest slopes modest
	let h = wall * ( 0.25 + 1.15 * r ) * ( 0.55 + 0.75 * big ) * reach * 1.05;
	// far range closing the fjord
	const far = smooth( - 34, - 50, z );
	h = Math.max( h, far * ( 0.35 + 1.2 * ridged( x * 0.11 + 40, z * 0.11 + 2 ) ) * ( 0.6 + 0.4 * big ) );
	// shoreline: underwater below the shore, small skerries near the mouth
	const shore = smooth( 0.55, 1.0, d );
	h = h * shore + ( shore - 1 ) * 0.06 - 0.004;
	h += 0.004 * fbm( x * 9, z * 9, 3 ) * shore;
	// promontory under the observer: rocky knoll, water starting ~50 m to the north
	const qx = x / 0.32, qz = ( z + 0.004 ) / ( z < 0 ? 0.055 : 0.5 );
	const knoll = 1 - smooth( 0.55, 1.05, Math.hypot( qx, qz ) );
	const knollH = knoll * ( 0.0072 + 0.004 * fbm( x * 60, z * 60, 3 ) ) - 0.0015 * ( 1 - knoll );
	h = Math.max( h, knollH );
	// a few rocks breaking the surface just off the shore
	for ( const [ rx, rz, rr, rh ] of ROCKS ) {

		const dd = Math.hypot( x - rx, ( z - rz ) * 1.3 ) / rr;
		if ( dd < 1 ) h = Math.max( h, rh * ( 1 - dd * dd ) * ( 0.8 + 0.4 * noise2( x * 400, z * 400 ) ) - 0.0004 );

	}

	return h;

}

const ROCKS = [
	[ - 0.062, - 0.085, 0.006, 0.0018 ], [ - 0.048, - 0.11, 0.004, 0.0012 ], [ 0.09, - 0.14, 0.008, 0.0022 ],
	[ 0.04, - 0.24, 0.005, 0.0011 ], [ - 0.16, - 0.36, 0.012, 0.0035 ],
];
