// Spectral data shared by the JS side (normalisations, presets) and baked into the WGSL.

// CIE 1931 2° colour matching functions — multi-lobe analytic fit of Wyman, Sloan & Shirley (JCGT 2013).
const g = ( x, mu, s1, s2 ) => {

	const t = ( x - mu ) / ( x < mu ? s1 : s2 );
	return Math.exp( - 0.5 * t * t );

};

export function cmf( l ) {

	const x = 1.056 * g( l, 599.8, 37.9, 31.0 ) + 0.362 * g( l, 442.0, 16.0, 26.7 ) - 0.065 * g( l, 501.1, 20.4, 26.2 );
	const y = 0.821 * g( l, 568.8, 46.9, 40.5 ) + 0.286 * g( l, 530.9, 16.3, 31.1 );
	const z = 1.217 * g( l, 437.0, 11.8, 36.0 ) + 0.681 * g( l, 459.0, 26.0, 13.8 );
	return [ Math.max( x, 0 ), y, z ];

}

// Planck's law, unnormalised (λ in nm). Only ratios matter: every emitter is normalised to luminance Y = 1.
export function planck( lnm, T ) {

	const l = lnm * 1e-3; // µm
	return 1 / ( l ** 5 * ( Math.exp( 14387.77 / ( l * T ) ) - 1 ) );

}

export function blackbodyNorm( T ) {

	let Y = 0;
	for ( let l = 360; l <= 830; l += 1 ) Y += planck( l, T ) * cmf( l )[ 1 ];
	return 1 / Y;

}

export function blackbodyXYZ( T ) {

	const k = blackbodyNorm( T );
	const s = [ 0, 0, 0 ];
	for ( let l = 360; l <= 830; l += 1 ) {

		const c = cmf( l ), p = planck( l, T ) * k;
		s[ 0 ] += p * c[ 0 ]; s[ 1 ] += p * c[ 1 ]; s[ 2 ] += p * c[ 2 ];

	}

	return s;

}

// Glass dispersion: Sellmeier coefficients (λ in µm): n² = 1 + Σ Bᵢ λ² / (λ² − Cᵢ)
export const GLASSES = {
	'BK7 crown': { B: [ 1.03961212, 0.231792344, 1.01046945 ], C: [ 0.00600069867, 0.0200179144, 103.560653 ] },
	'SF11 dense flint': { B: [ 1.73759695, 0.313747346, 1.89878101 ], C: [ 0.013188707, 0.0623068142, 155.23629 ] },
	'N-SF66 extreme flint': { B: [ 2.0245976, 0.470187196, 2.59970433 ], C: [ 0.0147053225, 0.0692998276, 161.817601 ] },
	'diamond': { B: [ 0.3306, 4.3356, 0 ], C: [ 0.1750 * 0.1750, 0.1060 * 0.1060, 1 ] },
};

export function sellmeier( glass, lnm ) {

	const l2 = ( lnm * 1e-3 ) ** 2;
	let n2 = 1;
	for ( let i = 0; i < 3; i ++ ) n2 += glass.B[ i ] * l2 / ( l2 - glass.C[ i ] );
	return Math.sqrt( n2 );

}

// Emission line spectra (nm, relative radiant power).
export const LINES = {
	hydrogen: [ [ 656.28, 1.0 ], [ 486.13, 0.36 ], [ 434.05, 0.18 ], [ 410.17, 0.10 ] ],
	mercury: [ [ 404.66, 0.32 ], [ 435.83, 1.0 ], [ 546.07, 1.0 ], [ 576.96, 0.24 ], [ 579.07, 0.26 ] ],
};

// Normalise a line list so that its luminance is 1 (Σ wᵢ ȳ(λᵢ) = 1).
export function normLines( lines ) {

	let Y = 0;
	for ( const [ l, w ] of lines ) Y += w * cmf( l )[ 1 ];
	return lines.map( ( [ l, w ] ) => [ l, w / Y ] );

}

export function linesXYZ( lines ) {

	const s = [ 0, 0, 0 ];
	for ( const [ l, w ] of normLines( lines ) ) {

		const c = cmf( l );
		s[ 0 ] += w * c[ 0 ]; s[ 1 ] += w * c[ 1 ]; s[ 2 ] += w * c[ 2 ];

	}

	return s;

}

export const LIGHTS = [
	{ name: 'daylight 5800 K', mode: 0, T: 5800 },
	{ name: 'tungsten 2900 K', mode: 0, T: 2900 },
	{ name: 'hydrogen Balmer lines', mode: 1, T: 6500 },
	{ name: 'mercury vapour lines', mode: 2, T: 6500 },
	{ name: 'blue sky 12000 K', mode: 0, T: 12000 },
];
