// Planck blackbody → linear sRGB lookup, integrated against CIE 1931 2° colour matching functions.
// Stored as (chromaticity rgb with luminance 1, log2 of luminance relative to a 6000 K emitter).
// Splitting chroma from (log) luminance keeps the filtered lookup smooth across the huge dynamic range
// of the Wien tail, and lets the shader apply a contrast exponent in log space.
import * as THREE from 'three/webgpu';
import { texture, vec2, clamp, exp2, max } from 'three/tsl';

export const BB_MIN = 1000;
export const BB_MAX = 21000;
const N = 256;

// Wyman, Sloan & Shirley (2013) multi-lobe fit of the CIE 1931 observer
const g = ( l, m, s1, s2 ) => {

	const t = ( l - m ) / ( l < m ? s1 : s2 );
	return Math.exp( - 0.5 * t * t );

};

const xBar = ( l ) => 1.056 * g( l, 599.8, 37.9, 31.0 ) + 0.362 * g( l, 442.0, 16.0, 26.7 ) - 0.065 * g( l, 501.1, 20.4, 26.2 );
const yBar = ( l ) => 0.821 * g( l, 568.8, 46.9, 40.5 ) + 0.286 * g( l, 530.9, 16.3, 31.1 );
const zBar = ( l ) => 1.217 * g( l, 437.0, 11.8, 36.0 ) + 0.681 * g( l, 459.0, 26.0, 13.8 );

function planckXYZ( T ) {

	let X = 0, Y = 0, Z = 0;
	for ( let l = 380; l <= 780; l += 2 ) {

		const lm = l * 1e-9;
		// second radiation constant c2 = hc/k = 0.014387769 m·K ; constant prefactor 2hc² dropped (normalised later)
		const B = 1 / ( Math.pow( lm, 5 ) * ( Math.exp( 0.014387769 / ( lm * T ) ) - 1 ) );
		X += B * xBar( l ); Y += B * yBar( l ); Z += B * zBar( l );

	}

	return [ X, Y, Z ];

}

export function createBlackbodyLUT() {

	const Yref = planckXYZ( 6000 )[ 1 ];
	const data = new Uint16Array( N * 4 );
	const toHalf = THREE.DataUtils.toHalfFloat;

	for ( let i = 0; i < N; i ++ ) {

		const T = BB_MIN + ( BB_MAX - BB_MIN ) * i / ( N - 1 );
		const [ X, Y, Z ] = planckXYZ( T );
		// XYZ → linear Rec.709 / sRGB primaries
		let r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
		let gg = - 0.9689 * X + 1.8758 * Y + 0.0415 * Z;
		let b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z;
		r = Math.max( 0, r / Y ); gg = Math.max( 0, gg / Y ); b = Math.max( 0, b / Y );
		// renormalise luminance after gamut clipping
		const lum = 0.2126 * r + 0.7152 * gg + 0.0722 * b;
		r /= lum; gg /= lum; b /= lum;
		const l2 = Math.max( - 24, Math.log2( Y / Yref ) );
		data[ i * 4 + 0 ] = toHalf( r );
		data[ i * 4 + 1 ] = toHalf( gg );
		data[ i * 4 + 2 ] = toHalf( b );
		data[ i * 4 + 3 ] = toHalf( l2 );

	}

	const tex = new THREE.DataTexture( data, N, 1, THREE.RGBAFormat, THREE.HalfFloatType );
	tex.minFilter = THREE.LinearFilter;
	tex.magFilter = THREE.LinearFilter;
	tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
	tex.generateMipmaps = false;
	tex.needsUpdate = true;
	return tex;

}

// Returns a TSL function kelvin → linear radiance (relative: 6000 K ≈ 1), with log-space contrast `gamma`.
export function makeBlackbody( lutTex ) {

	const lut = texture( lutTex );
	return ( kelvin, gamma ) => {

		const u = clamp( kelvin.sub( BB_MIN ).div( BB_MAX - BB_MIN ), 0, 1 ).mul( ( N - 1 ) / N ).add( 0.5 / N );
		const s = lut.sample( vec2( u, 0.5 ) ).level( 0 );
		return max( s.rgb, 0 ).mul( exp2( s.a.mul( gamma ) ) );

	};

}
