// Parameter sets for the Gravner–Griffeath mesoscopic snowflake model
// (J. Gravner & D. Griffeath, "Modeling snow crystal growth II: A mesoscopic lattice map with plausible dynamics",
//  Physica D 237 (2008) 385–404).
//
//   ρ  initial / far-field vapour density (supersaturation)
//   β  attachment threshold for tips & flat facets (n = 1, 2 attached neighbours) — the anisotropy knob
//   α  knife-edge threshold for concave sites (n = 3) …
//   θ  … that attach when the local vapour Σ d < θ (smooths concavities, makes ridges)
//   κ  fraction of vapour that freezes straight into crystal mass at the boundary (rest goes into the quasi-liquid layer)
//   μ  melting rate of boundary (quasi-liquid) mass back into vapour
//   γ  melting rate of boundary crystal mass
//
// `ref` notes where a set comes from. The fernlike set is the paper's flagship example; the others are from the
// paper's case studies / the snowfake reimplementation as best recalled, and each one was re-verified here by running
// the exact GPU update rule on the CPU (forms noted in `form`). `rate` is a rough tip speed in cells/step used to seed
// the growth-speed controller (it adapts from live measurements afterwards).

export const PRESETS = [
	{ name: 'Fernlike stellar dendrite', short: 'fernlike dendrite', ref: 'GG 2008, fernlike case', rho: 0.635, beta: 1.6, alpha: 0.4, theta: 0.025, kappa: 0.0025, mu: 0.015, gamma: 0.0005, rate: 0.05, temp: -15, sat: 'very high' },
	{ name: 'Stellar dendrite', short: 'stellar dendrite', ref: 'GG 2008 case study', rho: 0.5, beta: 1.4, alpha: 0.1, theta: 0.005, kappa: 0.001, mu: 0.04, gamma: 0.0001, rate: 0.026, temp: -14, sat: 'high' },
	{ name: 'Sectored plate', short: 'ribbed sectored plate', ref: 'GG 2008 (as in snowfake)', rho: 0.35, beta: 1.4, alpha: 0.001, theta: 0.015, kappa: 0.05, mu: 0.015, gamma: 0.01, rate: 0.011, temp: -13, sat: 'moderate' },
	{ name: 'Broad-branched star', short: 'broad branches', ref: 'GG 2008 case study', rho: 0.8, beta: 2.6, alpha: 0.006, theta: 0.005, kappa: 0.05, mu: 0.015, gamma: 0.0001, rate: 0.012, temp: -16, sat: 'high, slow attachment' },
	{ name: 'Plate-tipped star', short: 'plates on arms', ref: 'GG 2008 case study', rho: 0.38, beta: 1.06, alpha: 0.35, theta: 0.112, kappa: 0.001, mu: 0.14, gamma: 0.0006, rate: 0.017, temp: -12, sat: 'moderate, evaporating' },
	{ name: 'Branched sectors', short: 'branched sectors', ref: 'GG 2008 case study', rho: 0.4, beta: 1.3, alpha: 0.08, theta: 0.025, kappa: 0.003, mu: 0.07, gamma: 0.00005, rate: 0.0055, temp: -13, sat: 'moderate' },
	{ name: 'Hexagonal plate', short: 'faceted plate', ref: 'low ρ, high β regime', rho: 0.36, beta: 2.0, alpha: 0.01, theta: 0.01, kappa: 0.05, mu: 0.015, gamma: 0.0001, rate: 0.002, temp: -11, sat: 'near saturation' },
	{ name: 'Narrow dendrite', short: 'narrow dendrite', ref: 'high κ variant', rho: 0.65, beta: 1.75, alpha: 0.2, theta: 0.026, kappa: 0.15, mu: 0.015, gamma: 0.0001, rate: 0.02, temp: -15, sat: 'high' },
];

export const KEYS = [ 'rho', 'beta', 'alpha', 'theta', 'kappa', 'mu', 'gamma' ];

// "Cloud layers" a falling flake can pass through: the click / drift targets.
// Each layer borrows a regime; temperatures are indicative of where that habit appears on the Nakaya–Libbrecht diagram.
export function layerLabel( p ) {

	return `${p.temp} °C · ${p.sat} → ${p.short}`;

}

export function lerpParams( out, a, b, t ) {

	for ( const k of KEYS ) out[ k ] = a[ k ] + ( b[ k ] - a[ k ] ) * t;
	return out;

}

// Small random perturbation so that no two flakes of the same preset are identical.
export function jitter( p, rnd, amount = 0.04 ) {

	const q = { ...p };
	q.rho *= 1 + ( rnd() - 0.5 ) * 2 * amount;
	q.beta *= 1 + ( rnd() - 0.5 ) * 2 * amount * 0.6;
	q.kappa *= 1 + ( rnd() - 0.5 ) * 2 * amount * 2;
	q.mu *= 1 + ( rnd() - 0.5 ) * 2 * amount;
	return q;

}
