// Flat ΛCDM background + linear theory on the CPU.
// Code units (Klypin & Holtzman 1997 PM convention): H0 = 1, lengths in mesh cells, time variable = scale factor a.
//   ȧ = da/dτ = sqrt( Ωm / a + ΩΛ a² )
//   dx/da = p / ( a² ȧ )          (p = a² dx/dτ, canonical comoving momentum)
//   dp/da = −∇φ / ȧ
//   ∇²φ   = (3/2) Ωm δ / a

export class Cosmology {

	constructor( { Om = 0.3, OL = 0.7, h = 0.7 } = {} ) {

		this.Om = Om; this.OL = OL; this.h = h;
		this.gamma = Om * h; // BBKS shape parameter Γ = Ωm h
		this._buildGrowth();

	}

	adot( a ) {

		return Math.sqrt( this.Om / a + this.OL * a * a );

	}

	// Linear growth factor D(a) ∝ (5Ωm/2) E(a) ∫₀ᵃ da' / ȧ(a')³, normalised to D(1) = 1, tabulated in ln a.
	_buildGrowth() {

		const n = 2048, lnA0 = Math.log( 1e-5 ), lnA1 = Math.log( 2.0 );
		this._lnA0 = lnA0; this._dln = ( lnA1 - lnA0 ) / ( n - 1 );
		const I = new Float64Array( n ), D = new Float64Array( n );
		let acc = 0, prev = 0;
		for ( let i = 0; i < n; i ++ ) {

			const a = Math.exp( lnA0 + i * this._dln );
			const f = a / Math.pow( this.adot( a ), 3 ); // integrand in d ln a
			if ( i === 0 ) acc = f / 2.5; // ∫₀^a0 ≈ analytic EdS tail: a^2.5/Ωm^1.5 /2.5  (f = a^2.5/Ωm^1.5)
			else acc += 0.5 * ( f + prev ) * this._dln;
			prev = f;
			I[ i ] = acc;
			const E = this.adot( a ) / a;
			D[ i ] = 2.5 * this.Om * E * acc;

		}

		const d1 = this._interp( D, 0 );
		this._D = D; this._I = I;
		this._Dnorm = 1 / d1;

	}

	_interp( arr, lnA ) {

		const x = ( lnA - this._lnA0 ) / this._dln;
		const i = Math.max( 0, Math.min( arr.length - 2, Math.floor( x ) ) );
		const t = x - i;
		return arr[ i ] * ( 1 - t ) + arr[ i + 1 ] * t;

	}

	D( a ) {

		return this._interp( this._D, Math.log( a ) ) * this._Dnorm;

	}

	// dD/da (normalised), analytic from the integral form
	dDda( a ) {

		const I = this._interp( this._I, Math.log( a ) );
		const ad = this.adot( a );
		const E = ad / a;
		const dE = - 1.5 * this.Om / ( a * a * a * a ) / E;
		return 2.5 * this.Om * ( dE * I + E / ( ad * ad * ad ) ) * this._Dnorm;

	}

	// growth rate f = dlnD/dlna
	f( a ) {

		return this.dDda( a ) * a / this.D( a );

	}

	// Zel'dovich: x = q + D(a) s,  p = a² ȧ D'(a) s
	zaMomentumFactor( a ) {

		return a * a * this.adot( a ) * this.dDda( a );

	}

	_simpson( fn, a0, a1, n = 16 ) {

		const h = ( a1 - a0 ) / n;
		let s = fn( a0 ) + fn( a1 );
		for ( let i = 1; i < n; i ++ ) s += fn( a0 + i * h ) * ( i & 1 ? 4 : 2 );
		return s * h / 3;

	}

	// ∫ da / ȧ   (kick)
	kick( a0, a1 ) {

		if ( a1 === a0 ) return 0;
		return this._simpson( ( a ) => 1 / this.adot( a ), a0, a1 );

	}

	// ∫ da / ( a² ȧ )  (drift)
	drift( a0, a1 ) {

		if ( a1 === a0 ) return 0;
		return this._simpson( ( a ) => 1 / ( a * a * this.adot( a ) ), a0, a1 );

	}

	// BBKS (Bardeen, Bond, Kaiser & Szalay 1986) transfer function, k in h/Mpc
	static bbks( k, gamma ) {

		const q = k / gamma;
		if ( q < 1e-8 ) return 1;
		const l = Math.log( 1 + 2.34 * q ) / ( 2.34 * q );
		const p = 1 + 3.89 * q + Math.pow( 16.1 * q, 2 ) + Math.pow( 5.46 * q, 3 ) + Math.pow( 6.71 * q, 4 );
		return l * Math.pow( p, - 0.25 );

	}

	// Amplitude A such that P(k) = A kⁿ T(k)² has the requested σ8 (top hat, R = 8 Mpc/h) at z = 0.
	normalise( sigma8, ns ) {

		const R = 8;
		let s = 0;
		const n = 4000, l0 = Math.log( 1e-4 ), l1 = Math.log( 50 ), dl = ( l1 - l0 ) / n;
		for ( let i = 0; i <= n; i ++ ) {

			const k = Math.exp( l0 + i * dl );
			const x = k * R;
			const W = x < 1e-3 ? 1 : 3 * ( Math.sin( x ) - x * Math.cos( x ) ) / ( x * x * x );
			const T = Cosmology.bbks( k, this.gamma );
			const v = k * k * k * Math.pow( k, ns ) * T * T * W * W; // k³ P W² (d ln k)
			s += ( i === 0 || i === n ? 0.5 : 1 ) * v;

		}

		s *= dl / ( 2 * Math.PI * Math.PI );
		return sigma8 * sigma8 / s;

	}

}
