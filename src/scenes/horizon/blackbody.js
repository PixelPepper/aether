// Blackbody colour lookup: Planck spectrum integrated against the CIE 1931 2° colour-matching functions
// (multi-lobe Gaussian fit of Wyman, Sloan & Shirley 2013), converted to linear Rec.709 / sRGB.
//
// The LUT is a 256×2 half-float DataTexture:
//   row 0: rgb = chromaticity (linear RGB / luminance Y), a = log10( Y(T) / Y(6500 K) ) — T log-spaced in [T_MIN, T_MAX]
//   row 1: r   = Novikov–Thorne disk temperature profile T(r)/T_peak, sampled at r = r_isco + u²·(R_MAX − r_isco)
// Storing luminance logarithmically keeps 60 orders of magnitude of Wien-tail dimming filterable in half floats.

import * as THREE from 'three/webgpu';
import { iscoRadius, pageThorneFlux } from './kerr.js';

export const LUT_N = 256;
export const T_MIN = 250;
export const T_MAX = 150000;
export const R_MAX = 60;

const lobe = ( l, mu, s1, s2 ) => {

	const t = ( l - mu ) / ( l < mu ? s1 : s2 );
	return Math.exp( - 0.5 * t * t );

};

const cie = ( l ) => [
	1.056 * lobe( l, 599.8, 37.9, 31.0 ) + 0.362 * lobe( l, 442.0, 16.0, 26.7 ) - 0.065 * lobe( l, 501.1, 20.4, 26.2 ),
	0.821 * lobe( l, 568.8, 46.9, 40.5 ) + 0.286 * lobe( l, 530.9, 16.3, 31.1 ),
	1.217 * lobe( l, 437.0, 11.8, 36.0 ) + 0.681 * lobe( l, 459.0, 26.0, 13.8 ),
];

// Planck B_λ up to a constant: λ⁻⁵ / (exp(hc/λkT) − 1), λ in nm.
const C2 = 1.4387769e7; // hc/k in nm·K
const planck = ( l, T ) => 1 / ( Math.pow( l, 5 ) * Math.expm1( C2 / ( l * T ) ) ) * 1e15;

export function blackbodyXYZ( T ) {

	let X = 0, Y = 0, Z = 0;
	for ( let l = 360; l <= 830; l += 2.5 ) {

		const b = planck( l, T );
		const [ x, y, z ] = cie( l );
		X += b * x; Y += b * y; Z += b * z;

	}

	return [ X, Y, Z ];

}

export function blackbodyRGB( T ) {

	const [ X, Y, Z ] = blackbodyXYZ( T );
	let r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
	let g = - 0.9689 * X + 1.8758 * Y + 0.0415 * Z;
	let b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z;
	r = Math.max( r, 0 ); g = Math.max( g, 0 ); b = Math.max( b, 0 );
	return { rgb: [ r, g, b ], Y };

}

const Y_REF = blackbodyXYZ( 6500 )[ 1 ];

// log10 of luminance relative to 6500 K
export const relLogLum = ( T ) => Math.log10( Math.max( blackbodyXYZ( T )[ 1 ], 1e-300 ) / Y_REF );

export function createLUT() {

	const data = new Uint16Array( LUT_N * 2 * 4 );
	const h = THREE.DataUtils.toHalfFloat;
	const lmin = Math.log( T_MIN ), lmax = Math.log( T_MAX );
	for ( let i = 0; i < LUT_N; i ++ ) {

		const T = Math.exp( lmin + ( lmax - lmin ) * i / ( LUT_N - 1 ) );
		const { rgb, Y } = blackbodyRGB( T );
		const yy = Math.max( Y, 1e-300 );
		// chromaticity normalised to luminance: rgb / Y  (Y = 0.2126 R + 0.7152 G + 0.0722 B for in-gamut colours)
		const lum = Math.max( 0.2126 * rgb[ 0 ] + 0.7152 * rgb[ 1 ] + 0.0722 * rgb[ 2 ], 1e-300 );
		const o = i * 4;
		data[ o ] = h( Math.min( rgb[ 0 ] / lum, 60 ) );
		data[ o + 1 ] = h( Math.min( rgb[ 1 ] / lum, 60 ) );
		data[ o + 2 ] = h( Math.min( rgb[ 2 ] / lum, 60 ) );
		data[ o + 3 ] = h( Math.max( Math.log10( yy / Y_REF ), - 60000 ) );

	}

	const tex = new THREE.DataTexture( data, LUT_N, 2, THREE.RGBAFormat, THREE.HalfFloatType );
	tex.minFilter = THREE.LinearFilter;
	tex.magFilter = THREE.LinearFilter;
	tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
	tex.generateMipmaps = false;
	tex.colorSpace = THREE.NoColorSpace;
	tex.needsUpdate = true;
	return tex;

}

// Fill row 1 with T(r)/T_peak for spin a; returns { rIsco, rPeak }.
export function writeDiskProfile( tex, a ) {

	const F = pageThorneFlux( a );
	const rIsco = iscoRadius( a );
	let fmax = 0, rPeak = rIsco;
	for ( let i = 1; i <= 4000; i ++ ) {

		const r = rIsco + ( R_MAX - rIsco ) * ( i / 4000 ) ** 2;
		const v = F( r );
		if ( v > fmax ) {

			fmax = v; rPeak = r;

		}

	}

	const data = tex.image.data;
	const h = THREE.DataUtils.toHalfFloat;
	for ( let i = 0; i < LUT_N; i ++ ) {

		const u = i / ( LUT_N - 1 );
		const r = rIsco + ( R_MAX - rIsco ) * u * u;
		const t = Math.pow( F( r ) / fmax, 0.25 );
		const o = ( LUT_N + i ) * 4;
		data[ o ] = h( t ); data[ o + 1 ] = h( t ); data[ o + 2 ] = h( t ); data[ o + 3 ] = h( 1 );

	}

	tex.needsUpdate = true;
	return { rIsco, rPeak };

}
