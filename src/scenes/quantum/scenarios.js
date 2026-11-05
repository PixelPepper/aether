// Scenario definitions: potentials V(x,y) + complex absorbing potential W(x,y) on the unit square, the default packet,
// and an analytic description of the barriers so the renderer can build matching monoliths.

export const SCENARIOS = {
	'double slit': 'slit',
	'tunnelling': 'tunnel',
	'harmonic oscillator': 'ho',
	'quantum billiard (stadium)': 'stadium',
};

export const V_WALL = 1.2e5; // ≫ k²/2 for every k the GUI allows
export const CAP_WIDTH = 0.085;
export const CAP_STRENGTH = 3.2e4;
export const HO_OMEGA = 700;

// stadium geometry (domain units)
export const STADIUM = { cx: 0.5, cy: 0.5, a: 0.17, r: 0.23, wall: 0.03 };

function stadiumDist( x, y ) {

	// signed distance to a Bunimovich stadium (negative inside)
	const { cx, cy, a, r } = STADIUM;
	const px = Math.max( Math.abs( x - cx ) - a, 0 );
	const py = y - cy;
	return Math.hypot( px, py ) - r;

}

/**
 * Fill `pot` (Float32Array N*N*2 as (V, W)) for the scenario. Returns the list of solid barriers
 * [{ x0, x1, y0, y1, h, kind }] in domain coordinates for rendering.
 */
export function buildPotential( id, N, pot, p ) {

	const dx = 1 / N;
	const boxes = [];
	const cov1 = ( c, half, x ) => Math.min( 1, Math.max( 0, ( half - Math.abs( x - c ) ) / dx + 0.5 ) );

	// slit wall
	const xw = 0.36, tw = 0.022, sw = p.slitWidth, sd = p.slitSep;
	const slits = [ 0.5 - sd / 2, 0.5 + sd / 2 ];
	// tunnelling barrier: height a little above the packet energy, thickness ~ one decay length
	const E = 0.5 * p.k * p.k;
	const Vb = 1.25 * E;
	const kappa = Math.sqrt( 2 * ( Vb - E ) );
	const tb = Math.max( 1.5 * dx, 0.95 / kappa );
	const xb = 0.52;

	for ( let j = 0; j < N; j ++ ) {

		const y = j * dx;
		for ( let i = 0; i < N; i ++ ) {

			const x = i * dx;
			let V = 0;

			if ( id === 'slit' ) {

				const cx = cov1( xw, tw / 2, x );
				if ( cx > 0 ) {

					let open = 0;
					for ( const s of slits ) open = Math.max( open, cov1( s, sw / 2, y ) );
					V = V_WALL * cx * ( 1 - open );

				}

			} else if ( id === 'tunnel' ) {

				V = Vb * cov1( xb, tb / 2, x );

			} else if ( id === 'ho' ) {

				const rx = x - 0.5, ry = y - 0.5;
				V = 0.5 * HO_OMEGA * HO_OMEGA * ( rx * rx + ry * ry );

			} else if ( id === 'stadium' ) {

				const d = stadiumDist( x, y );
				V = V_WALL * Math.min( 1, Math.max( 0, d / dx + 0.5 ) );

			}

			// complex absorbing potential on every edge (quadratic ramp); in the stadium the wall keeps the wave away from it
			const s = ( t ) => {

				const a = Math.max( 0, CAP_WIDTH - t ) / CAP_WIDTH, b = Math.max( 0, t - ( 1 - CAP_WIDTH ) ) / CAP_WIDTH;
				return a * a + b * b;

			};

			const W = CAP_STRENGTH * ( s( x ) + s( y ) );
			const o = ( j * N + i ) * 2;
			pot[ o ] = V;
			pot[ o + 1 ] = W;

		}

	}

	if ( id === 'slit' ) {

		const x0 = xw - tw / 2, x1 = xw + tw / 2;
		const edges = [ 0, slits[ 0 ] - sw / 2, slits[ 0 ] + sw / 2, slits[ 1 ] - sw / 2, slits[ 1 ] + sw / 2, 1 ];
		for ( let e = 0; e < edges.length; e += 2 ) boxes.push( { x0, x1, y0: edges[ e ], y1: edges[ e + 1 ], h: 1.25, kind: 'wall' } );

	} else if ( id === 'tunnel' ) {

		boxes.push( { x0: xb - tb / 2, x1: xb + tb / 2, y0: 0, y1: 1, h: 0.55 * Vb / ( 0.5 * 150 * 150 ), kind: 'barrier' } );

	}

	return { boxes, Vb };

}

/** default packet for a scenario: { x, y, kx, ky, sigma } */
export function defaultPacket( id, p ) {

	if ( id === 'slit' ) return { x: 0.15, y: 0.5, kx: p.k, ky: 0, sigma: p.sigma };
	if ( id === 'tunnel' ) {

		const a = 0.32;
		return { x: 0.2, y: 0.36, kx: p.k * Math.cos( a ), ky: p.k * Math.sin( a ), sigma: p.sigma };

	}

	if ( id === 'ho' ) {

		// coherent state: σ = sqrt(ħ/2mω); start displaced, launched tangentially → it orbits without spreading
		const r0 = 0.2;
		return { x: 0.5 + r0, y: 0.5, kx: 0, ky: HO_OMEGA * r0 * p.k / p.kDef, sigma: p.sigma };

	}

	// stadium
	return { x: 0.42, y: 0.43, kx: p.k * Math.cos( 0.55 ), ky: p.k * Math.sin( 0.55 ), sigma: p.sigma * 0.8 };

}

export const hoSigma = () => Math.sqrt( 1 / ( 2 * HO_OMEGA ) );

export { stadiumDist };
