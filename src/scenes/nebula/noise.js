// Procedural noise for the Stellar Nursery chapter.
// All hot functions use setLayout() so they compile to real WGSL functions instead of being inlined
// at every call site (keeps the fBm-heavy compute kernels small enough to compile fast on SwiftShader).
import { Fn, vec3, float, floor, fract, dot, mix, mod, abs, mat3 } from 'three/tsl';

// Dave Hoskins' sine-free hash (vec3 -> vec3 in [-1, 1]).
export const hash33 = /*@__PURE__*/ Fn( ( [ p ] ) => {

	const q = fract( p.mul( vec3( 0.1031, 0.1030, 0.0973 ) ) ).toVar();
	q.addAssign( dot( q, q.yxz.add( 33.33 ) ) );
	return fract( q.xxy.add( q.yxx ).mul( q.zyx ) ).mul( 2.0 ).sub( 1.0 );

} ).setLayout( { name: 'nb_hash33', type: 'vec3', inputs: [ { name: 'p', type: 'vec3' } ] } );

const O = [
	[ 0, 0, 0 ], [ 1, 0, 0 ], [ 0, 1, 0 ], [ 1, 1, 0 ],
	[ 0, 0, 1 ], [ 1, 0, 1 ], [ 0, 1, 1 ], [ 1, 1, 1 ],
];

function gradNoiseBody( p, period ) {

	const i = floor( p ).toVar();
	const f = fract( p ).toVar();
	const u = f.mul( f ).mul( f ).mul( f.mul( f.mul( 6.0 ).sub( 15.0 ) ).add( 10.0 ) ).toVar();
	const n = O.map( ( o ) => {

		const ov = vec3( o[ 0 ], o[ 1 ], o[ 2 ] );
		const c = i.add( ov );
		const g = hash33( period ? mod( c, period ) : c );
		return dot( g, f.sub( ov ) );

	} );
	const x00 = mix( n[ 0 ], n[ 1 ], u.x ), x10 = mix( n[ 2 ], n[ 3 ], u.x );
	const x01 = mix( n[ 4 ], n[ 5 ], u.x ), x11 = mix( n[ 6 ], n[ 7 ], u.x );
	return mix( mix( x00, x10, u.y ), mix( x01, x11, u.y ), u.z ).mul( 1.6 );

}

// Quintic gradient noise, roughly in [-1, 1].
export const gnoise = /*@__PURE__*/ Fn( ( [ p ] ) => gradNoiseBody( p, null ) )
	.setLayout( { name: 'nb_gnoise', type: 'float', inputs: [ { name: 'p', type: 'vec3' } ] } );

// Periodic (tileable) gradient noise with integer period `per` (per axis).
export const gnoiseP = /*@__PURE__*/ Fn( ( [ p, per ] ) => gradNoiseBody( p, per ) )
	.setLayout( { name: 'nb_gnoiseP', type: 'float', inputs: [ { name: 'p', type: 'vec3' }, { name: 'per', type: 'vec3' } ] } );

// rotation + scale between octaves (iq's m3, scaled ~2.02) to break up lattice alignment
const M3 = mat3( 0.00, 1.616, 1.212, - 1.616, 0.7272, - 0.9696, - 1.212, - 0.9696, 1.2928 );

// fBm (unrolled in JS, each octave is one gnoise() call). Returns ~[-1, 1].
export function fbm( p, octaves = 4, gain = 0.5 ) {

	let sum = float( 0 ), amp = 0.5, norm = 0;
	let q = vec3( p );
	for ( let k = 0; k < octaves; k ++ ) {

		sum = sum.add( gnoise( q ).mul( amp ) );
		norm += amp;
		amp *= gain;
		q = M3.mul( q ).add( vec3( 1.7, 9.2, 3.3 ) );

	}

	return sum.div( norm );

}

// ridged multifractal-ish: sharp creases along noise zero-crossings -> filaments / sheets. Returns [0, 1].
export function ridged( p, octaves = 4, gain = 0.5 ) {

	let sum = float( 0 ), amp = 0.5, norm = 0;
	let q = vec3( p );
	for ( let k = 0; k < octaves; k ++ ) {

		const r = float( 1 ).sub( abs( gnoise( q ) ) );
		sum = sum.add( r.mul( r ).mul( amp ) );
		norm += amp;
		amp *= gain;
		q = M3.mul( q ).add( vec3( 4.1, 2.7, 8.6 ) );

	}

	return sum.div( norm );

}

// Tileable fBm on a lattice of period `per0` (doubling per octave). Returns ~[-1, 1].
export function fbmP( p, per0, octaves = 4, gain = 0.5 ) {

	let sum = float( 0 ), amp = 0.5, norm = 0, scale = 1;
	for ( let k = 0; k < octaves; k ++ ) {

		const per = per0 * scale;
		sum = sum.add( gnoiseP( vec3( p ).mul( scale ).add( vec3( 13.0 * k, 7.0 * k, 3.0 * k ) ), vec3( per, per, per ) ).mul( amp ) );
		norm += amp;
		amp *= gain;
		scale *= 2;

	}

	return sum.div( norm );

}

export function ridgedP( p, per0, octaves = 4, gain = 0.55 ) {

	let sum = float( 0 ), amp = 0.5, norm = 0, scale = 1;
	for ( let k = 0; k < octaves; k ++ ) {

		const per = per0 * scale;
		const r = float( 1 ).sub( abs( gnoiseP( vec3( p ).mul( scale ).add( vec3( 5.0 * k, 11.0 * k, 17.0 * k ) ), vec3( per, per, per ) ) ) );
		sum = sum.add( r.mul( r ).mul( amp ) );
		norm += amp;
		amp *= gain;
		scale *= 2;

	}

	return sum.div( norm );

}
