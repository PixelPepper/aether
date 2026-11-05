// Aurora — shared physical model: grid mapping, atmosphere, emission spectra.
//
// World frame (km): x east, y up (altitude), z south; the observer stands at the origin looking north (−z).
// The emission volume is observer-centric (azimuth φ, altitude h, ground range ρ) so that its angular
// resolution is spent where a ground observer can see it: every raymarch ray has constant φ and walks
// monotonically outward in ρ and upward in h.
import { uniform, vec3, float, pow, clamp, sqrt, asinh, smoothstep, exp, max, sinh, cosh } from 'three/tsl';
import * as THREE from 'three/webgpu';

export const R_E = 6371; // km

export const GRID = {
	h0: 80, h1: 450, // altitude band (km)
	rhoMax: 1100, // ground range (km)
	K: 3.5, // azimuth warp: denser cells toward magnetic north
	p: 0.6, // altitude warp: denser cells at the sharp lower border
};

const ASK = Math.asinh( GRID.K );

// ── world → normalised grid coordinates (all in [0,1]) ──
export const gU = ( phi ) => asinh( phi.mul( GRID.K / Math.PI ) ).mul( 0.5 / ASK ).add( 0.5 );
export const gV = ( h ) => pow( clamp( h.sub( GRID.h0 ).mul( 1 / ( GRID.h1 - GRID.h0 ) ), 0.0, 1.0 ), GRID.p );
export const gW = ( rho ) => sqrt( clamp( rho.mul( 1 / GRID.rhoMax ), 0.0, 1.0 ) );

// ── grid → world (voxel centres) and cell sizes ──
export const invU = ( u ) => sinh( u.mul( 2 ).sub( 1 ).mul( ASK ) ).mul( Math.PI / GRID.K ); // φ
export const dPhidU = ( u ) => cosh( u.mul( 2 ).sub( 1 ).mul( ASK ) ).mul( 2 * ASK * Math.PI / GRID.K );
export const invV = ( v ) => pow( v, 1 / GRID.p ).mul( GRID.h1 - GRID.h0 ).add( GRID.h0 ); // h
export const dHdV = ( v ) => pow( max( v, 1e-4 ), 1 / GRID.p - 1 ).mul( ( GRID.h1 - GRID.h0 ) / GRID.p );
export const invW = ( w ) => w.mul( w ).mul( GRID.rhoMax ); // ρ
export const dRhodW = ( w ) => w.mul( 2 * GRID.rhoMax );

// ── atmosphere (normalised so the column mass above 100 km is 1) ──
// Two-exponential fit to MSIS-like profiles: scale height ~6.5 km in the lower thermosphere, ~45 km above ~200 km.
export const airRho = ( h ) => exp( h.sub( 100 ).mul( - 1 / 6.5 ) ).mul( 0.150 ).add( exp( h.sub( 100 ).mul( - 1 / 45 ) ).mul( 5.33e-4 ) );
export const colMass = ( h ) => exp( h.sub( 100 ).mul( - 1 / 6.5 ) ).mul( 0.976 ).add( exp( h.sub( 100 ).mul( - 1 / 45 ) ).mul( 0.024 ) );
// electron range–energy relation R ∝ E^1.67 (column mass units above; E in keV): 10 keV stops at ~103 km,
// 1 keV at ~135 km, 200 eV at ~250 km.
export const rangeOf = ( E ) => pow( max( E, 0.0 ), 1.67 ).mul( 0.01265 );
export const energyOf = ( Q ) => pow( max( Q, 0.0 ).mul( 1 / 0.01265 ), 1 / 1.67 );

// ── emission lines ──
// Linear-sRGB colours of the auroral lines as a camera records them ("vivid") and as the dark-adapted eye sees
// them (mesopic: rods dominate, red 630 nm nearly invisible, everything desaturated toward a pale cyan-green).
export const LINES = {
	green: { vivid: [ 0.10, 1.0, 0.24 ], eye: [ 0.42, 0.62, 0.52 ] }, // O(¹S) 557.7 nm
	red: { vivid: [ 1.0, 0.035, 0.028 ], eye: [ 0.10, 0.07, 0.08 ] }, // O(¹D) 630.0 nm
	blue: { vivid: [ 0.30, 0.07, 1.0 ], eye: [ 0.32, 0.36, 0.55 ] }, // N₂⁺ 1NG 427.8 nm
	magenta: { vivid: [ 1.0, 0.10, 0.60 ], eye: [ 0.40, 0.30, 0.42 ] }, // N₂ 1PG + N₂⁺ lower border
};

export function createEmission() {

	const u = {
		cGreen: uniform( new THREE.Color() ),
		cRed: uniform( new THREE.Color() ),
		cBlue: uniform( new THREE.Color() ),
		cMag: uniform( new THREE.Color() ),
		redYield: uniform( 5.0 ),
	};

	const setRealism = ( r ) => {

		const lerp = ( c, L ) => c.value.setRGB( ...L.vivid.map( ( v, i ) => v + ( L.eye[ i ] - v ) * r ) );
		lerp( u.cGreen, LINES.green ); lerp( u.cRed, LINES.red ); lerp( u.cBlue, LINES.blue ); lerp( u.cMag, LINES.magenta );

	};

	setRealism( 0.15 );

	// Volume emissivity → radiance per km of path. `fast` holds the prompt/short-lived excitation (N₂, N₂⁺ and O(¹S),
	// τ≈0.7 s), `slow` the metastable O(¹D) reservoir (τ = 110 s in nature, compressed here).
	// Branching per unit deposited energy depends only on altitude, so it is applied here and not stored:
	//   557.7 green   quenched by O₂ below ~95 km, fades above ~200 km where O(¹D) takes over
	//   630.0 red     collisionally quenched by N₂ below ~200 km; long-lived and diffuse above
	//   427.8 blue    ∝ N₂ ionisation, follows the N₂ fraction (falls off with height as O dominates)
	//   magenta       N₂ 1PG + N₂⁺ at the bottom edge (<100 km) when the precipitation is hard
	const emit = ( fast, slow, h ) => {

		const yG = smoothstep( 89, 104, h ).mul( float( 1 ).sub( smoothstep( 185, 300, h ).mul( 0.8 ) ) );
		const yB = float( 0.16 ).mul( float( 1 ).sub( smoothstep( 130, 260, h ).mul( 0.6 ) ) );
		const yM = float( 1 ).sub( smoothstep( 92, 106, h ) ).mul( 0.9 );
		const yR = smoothstep( 165, 260, h ).mul( u.redYield ).mul( float( 1 ).sub( smoothstep( 360, 450, h ).mul( 0.6 ) ) );
		return vec3( u.cGreen ).mul( yG ).add( vec3( u.cBlue ).mul( yB ) ).add( vec3( u.cMag ).mul( yM ) ).mul( fast )
			.add( vec3( u.cRed ).mul( yR.mul( slow ) ) );

	};

	return { u, emit, setRealism };

}

// atmospheric extinction (Rayleigh + aerosol) for the aurora lines and starlight; e = sin(elevation)
export const TAU = [ 0.09, 0.13, 0.30 ];
// Kasten–Young air mass (elevation in degrees ≈ 57.3·sinE is fine where the correction matters)
export const airmass = ( sinE ) => float( 1 ).div( max( sinE, 0.0 ).add( pow( max( sinE, 0.0 ).mul( 57.3 ).add( 6.07995 ), - 1.6364 ).mul( 0.50572 ) ) );
export const extinction = ( sinE ) => exp( vec3( ...TAU ).mul( airmass( sinE ).negate() ) );
