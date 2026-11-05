// Split-step Fourier solver for the 2D time-dependent Schrödinger equation, entirely in compute shaders.
//
//   i ∂ψ/∂t = −½∇²ψ + (V − iW)ψ            (ħ = m = 1, unit square domain, N×N cells)
//
// One Strang step:  ψ ← e^{−i(V−iW)dt/2} · F⁻¹[ e^{−i k² dt/2} · F[ e^{−i(V−iW)dt/2} ψ ] ]
//
// The 2D FFT is separable. Every 1D transform of length N runs inside ONE workgroup (N/2 invocations, one radix-2
// butterfly each per stage) on a row or column held in workgroup shared memory, with a workgroupBarrier between the
// log2(N) stages. Forward transforms are decimation-in-frequency (natural order in → bit-reversed out) and inverse
// transforms are decimation-in-time (bit-reversed in → natural out), so no bit-reversal permutation is ever executed:
// k-space lives in bit-reversed order and the kinetic phase table is simply stored permuted.
//
// Kernels are fused so a full step costs two dispatches:
//   row  pass : [row IFFT] → potential phase/absorption → [row FFT]
//   col  pass : col FFT → kinetic phase (·1/N² normalisation folded in) → col IFFT
// A frame of S steps is  rowFirst, col, (rowMid, col)×(S−1), rowLast  =  2S+1 dispatches in one compute pass.
// Each workgroup owns exactly one row/column, so all transforms are done in place on a single storage buffer.

import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, float, uint, vec2, vec4, uvec2, cos, sin, exp, dot, pow, min, max, clamp, sqrt, atan, length, smoothstep, select,
	array, instanceIndex, localId, workgroupId, workgroupArray, workgroupBarrier, instancedArray, storage, uniform, textureStore, hash, atomicAdd,
} from 'three/tsl';

const cmul = ( a, b ) => vec2( a.x.mul( b.x ).sub( a.y.mul( b.y ) ), a.x.mul( b.y ).add( a.y.mul( b.x ) ) );

function bitrev( i, bits ) {

	let r = 0;
	for ( let b = 0; b < bits; b ++ ) {

		r = ( r << 1 ) | ( i & 1 );
		i >>= 1;

	}

	return r;

}

export class SchrodingerSolver {

	constructor( renderer, { N, hits = 8192, mist = 4096, fft = 'shared' } ) {

		this.renderer = renderer;
		this.fft = fft; // 'shared': workgroup-memory kernels (GPUs) · 'global': barrier-free Stockham ping-pong (SwiftShader)
		this.N = N;
		this.H = N >> 1;
		this.L = Math.log2( N );
		this.dx = 1 / N;
		this.MAX_HITS = hits;
		this.MIST = mist;

		const NN = N * N;
		this.psi = instancedArray( NN, 'vec2' ).setName( 'psi' );
		if ( fft === 'global' ) this.psiB = instancedArray( NN, 'vec2' ).setName( 'psiB' );
		this.potArray = new Float32Array( NN * 2 ); // (V, W) per cell
		this.pot = instancedArray( this.potArray, 'vec2' ).setName( 'pot' );
		this.kArray = new Float32Array( N * 2 );
		this.ktab = instancedArray( this.kArray, 'vec2' ).setName( 'ktab' );
		this.rowSum = instancedArray( N, 'float' ).setName( 'rowSum' );

		// detector / Born-rule sampling
		this.counterAttr = new THREE.StorageBufferAttribute( new Uint32Array( 4 ), 1 );
		this.counter = storage( this.counterAttr, 'uint', 4 ).toAtomic();
		this.rng = instancedArray( new Uint32Array( N ).map( ( _, i ) => ( i * 2654435761 ) >>> 0 ), 'uint' ).setName( 'rng' );
		const hitInit = new Float32Array( hits * 4 );
		for ( let i = 0; i < hits; i ++ ) hitInit[ i * 4 + 2 ] = - 1e9;
		this.hits = instancedArray( hitInit, 'vec4' ).setName( 'hits' );

		// Bohmian mist particles: (u, v, age, height-offset)
		const mistInit = new Float32Array( mist * 4 );
		for ( let i = 0; i < mist; i ++ ) {

			mistInit[ i * 4 ] = Math.random(); mistInit[ i * 4 + 1 ] = Math.random();
			mistInit[ i * 4 + 2 ] = 1 + Math.random(); mistInit[ i * 4 + 3 ] = Math.random();

		}

		this.mist = instancedArray( mistInit, 'vec4' ).setName( 'mist' );

		// display field: (Re ψ, Im ψ, height, V/Vref), hardware-filtered in the materials
		this.field = new THREE.StorageTexture( N, N );
		this.field.type = THREE.HalfFloatType;
		this.field.format = THREE.RGBAFormat;
		this.field.minFilter = THREE.LinearFilter;
		this.field.magFilter = THREE.LinearFilter;
		this.field.generateMipmaps = false;
		this.field.wrapS = this.field.wrapT = THREE.ClampToEdgeWrapping;

		this.u = {
			half: uniform( 0 ), // dt/2
			full: uniform( 0 ), // dt
			frameTau: uniform( 0 ), // sim time advanced this frame (detector + mist)
			realDt: uniform( 1 / 60 ),
			time: uniform( 0 ), // display clock for hit flashes
			rate: uniform( 900 ), // detections per unit of absorbed probability
			detX: uniform( 0.9 ), // detector strip starts here (domain x)
			heightScale: uniform( 1.25 ),
			rhoRef: uniform( 1 / ( 2 * Math.PI * 0.05 * 0.05 ) ),
			vRef: uniform( 1e5 ),
			// packet
			c: uniform( new THREE.Vector2( 0.15, 0.5 ) ),
			k: uniform( new THREE.Vector2( 150, 0 ) ),
			sigma: uniform( 0.05 ),
			// wall painting (capsule a→b)
			pa: uniform( new THREE.Vector2() ),
			pb: uniform( new THREE.Vector2() ),
			pr: uniform( 0.012 ),
			pv: uniform( 1e5 ),
			mistOn: uniform( 1 ),
			frame: uniform( 0, 'uint' ),
		};

		this.dt = 1e-5;
		this._lists = new Map();
		this._build();

	}

	// ───────────────────────── kernels ─────────────────────────

	_stage( sh, t, s, sign, dif ) {

		const half = 1 << s;
		const k = t.bitAnd( half - 1 );
		const j0 = t.mul( 2 ).sub( k ).toVar();
		const j1 = j0.add( half ).toVar();
		const a = sh.element( j0 ).toVar();
		const b = sh.element( j1 ).toVar();

		if ( s === 0 ) {

			sh.element( j0 ).assign( a.add( b ) );
			sh.element( j1 ).assign( a.sub( b ) );

		} else {

			const ang = k.toFloat().mul( sign * Math.PI / half );
			const w = vec2( cos( ang ), sin( ang ) ).toVar();
			if ( dif ) {

				sh.element( j0 ).assign( a.add( b ) );
				sh.element( j1 ).assign( cmul( a.sub( b ), w ) );

			} else {

				const bw = cmul( b, w ).toVar();
				sh.element( j0 ).assign( a.add( bw ) );
				sh.element( j1 ).assign( a.sub( bw ) );

			}

		}

		workgroupBarrier();

	}

	_dif( sh, t, sign ) {

		for ( let s = this.L - 1; s >= 0; s -- ) this._stage( sh, t, s, sign, true );

	}

	_dit( sh, t, sign ) {

		for ( let s = 0; s < this.L; s ++ ) this._stage( sh, t, s, sign, false );

	}

	_potFactor( g, tau ) {

		const vw = this.pot.element( g ).toVar();
		const ph = vw.x.mul( tau );
		return vec2( cos( ph ), sin( ph ).negate() ).mul( exp( vw.y.mul( tau ).negate() ) );

	}

	_rowKernel( mode ) {

		const { N, H, psi } = this;
		const sh = workgroupArray( 'vec2', N );
		const u = this.u;

		return Fn( () => {

			const t = localId.x;
			const base = workgroupId.x.mul( N );
			const i1 = t.add( H );
			const g0 = base.add( t ).toVar(), g1 = base.add( i1 ).toVar();

			if ( mode === 'first' ) {

				sh.element( t ).assign( cmul( psi.element( g0 ), this._potFactor( g0, u.half ) ) );
				sh.element( i1 ).assign( cmul( psi.element( g1 ), this._potFactor( g1, u.half ) ) );

			} else {

				sh.element( t ).assign( psi.element( g0 ) );
				sh.element( i1 ).assign( psi.element( g1 ) );

			}

			workgroupBarrier();

			if ( mode !== 'first' ) {

				this._dit( sh, t, + 1 );
				const tau = mode === 'mid' ? u.full : u.half;
				sh.element( t ).assign( cmul( sh.element( t ), this._potFactor( g0, tau ) ) );
				sh.element( i1 ).assign( cmul( sh.element( i1 ), this._potFactor( g1, tau ) ) );
				workgroupBarrier();

			}

			if ( mode !== 'last' ) this._dif( sh, t, - 1 );

			psi.element( g0 ).assign( sh.element( t ) );
			psi.element( g1 ).assign( sh.element( i1 ) );

		} )().compute( N * H, [ H ] ).setName( 'tdse-row-' + mode );

	}

	_colKernel() {

		const { N, H, psi, ktab } = this;
		const sh = workgroupArray( 'vec2', N );

		return Fn( () => {

			const t = localId.x;
			const col = workgroupId.x;
			const i1 = t.add( H );
			const g0 = t.mul( N ).add( col ).toVar(), g1 = i1.mul( N ).add( col ).toVar();

			sh.element( t ).assign( psi.element( g0 ) );
			sh.element( i1 ).assign( psi.element( g1 ) );
			workgroupBarrier();

			this._dif( sh, t, - 1 );

			// kinetic propagator, separable: e^{-i(kx²+ky²)dt/2}/N² = f[kx]·f[ky]   (tables stored in bit-reversed order)
			const kc = ktab.element( col ).toVar();
			sh.element( t ).assign( cmul( sh.element( t ), cmul( kc, ktab.element( t ) ) ) );
			sh.element( i1 ).assign( cmul( sh.element( i1 ), cmul( kc, ktab.element( i1 ) ) ) );
			workgroupBarrier();

			this._dit( sh, t, + 1 );

			psi.element( g0 ).assign( sh.element( t ) );
			psi.element( g1 ).assign( sh.element( i1 ) );

		} )().compute( N * H, [ H ] ).setName( 'tdse-col' );

	}

	_build() {

		const { N, psi, pot, u } = this;
		const NN = N * N;
		const dx = this.dx;

		if ( this.fft === 'shared' ) {

			this.rowFirst = this._rowKernel( 'first' );
			this.rowMid = this._rowKernel( 'mid' );
			this.rowLast = this._rowKernel( 'last' );
			this.col = this._colKernel();

		} else if ( this.fft === 'line' ) {

			this._buildLineFFT();

		} else {

			this._buildGlobalFFT();

		}

		// Gaussian wavepacket  ψ = (2πσ²)^{-1/2} exp(−|r−c|²/4σ²) exp(i k·(r−c))
		this.initPacket = Fn( () => {

			const x = instanceIndex.mod( N ), y = instanceIndex.div( N );
			const p = vec2( x.toFloat(), y.toFloat() ).mul( dx );
			const d = p.sub( u.c ).toVar();
			const s2 = u.sigma.mul( u.sigma );
			const amp = exp( dot( d, d ).div( s2.mul( - 4 ) ) ).div( sqrt( s2.mul( 2 * Math.PI ) ) );
			const ph = dot( u.k, d );
			// do not start inside a wall
			const inWall = smoothstep( 0.0, 0.5, pot.element( instanceIndex ).x.div( u.vRef ) );
			psi.element( instanceIndex ).assign( vec2( cos( ph ), sin( ph ) ).mul( amp ).mul( float( 1 ).sub( inWall ) ) );

		} )().compute( NN, [ 64 ] ).setName( 'tdse-packet' );

		this.zero = Fn( () => {

			psi.element( instanceIndex ).assign( vec2( 0 ) );

		} )().compute( NN, [ 64 ] );

		// norm: per-row Σ|ψ|²
		this.normPass = Fn( () => {

			const acc = float( 0 ).toVar();
			const base = instanceIndex.mul( N );
			Loop( { start: uint( 0 ), end: uint( N ), type: 'uint', condition: '<' }, ( { i } ) => {

				const p = psi.element( base.add( i ) );
				acc.addAssign( dot( p, p ) );

			} );
			this.rowSum.element( instanceIndex ).assign( acc.mul( dx * dx ) );

		} )().compute( N, [ 64 ] ).setName( 'tdse-norm' );

		// display field
		this.derive = Fn( () => {

			const x = instanceIndex.mod( N ), y = instanceIndex.div( N );
			const p = psi.element( instanceIndex ).toVar();
			const rho = dot( p, p );
			const h = pow( rho.div( u.rhoRef ).add( 1e-9 ), 0.55 ).mul( u.heightScale );
			const v = pot.element( instanceIndex ).x.div( u.vRef );
			textureStore( this.field, uvec2( x, y ), vec4( p.x, p.y, h, v ) ).toWriteOnly();

		} )().compute( NN, [ 64 ] ).setName( 'tdse-field' );

		// wall painting: capsule from pa to pb
		this.paint = Fn( () => {

			const x = instanceIndex.mod( N ), y = instanceIndex.div( N );
			const p = vec2( x.toFloat(), y.toFloat() ).mul( dx );
			const ab = u.pb.sub( u.pa );
			const hh = clamp( dot( p.sub( u.pa ), ab ).div( max( dot( ab, ab ), 1e-10 ) ), 0, 1 );
			const d = length( p.sub( u.pa ).sub( ab.mul( hh ) ) );
			const cov = smoothstep( u.pr.add( dx ), u.pr.sub( dx * 0.5 ), d );
			const e = pot.element( instanceIndex );
			e.x.assign( max( e.x, cov.mul( u.pv ) ) );

		} )().compute( NN, [ 64 ] ).setName( 'tdse-paint' );

		this._buildDetector();
		this._buildMist();

	}

	// Barrier-free fallback: radix-2 Stockham autosort FFT, one dispatch per stage, ping-ponging between two storage
	// buffers (natural order in and out). Software rasterisers serialise workgroup barriers badly, so tier 0 uses this.
	// A Strang step = row FFT (V/2 fused into stage 0) → col FFT (kinetic phase fused into the last stage)
	//               → col IFFT → row IFFT (V/2 fused into the last stage): 4·log2N dispatches, ending back in `psi`.
	_buildGlobalFFT() {

		const { N, H, L, u } = this;
		const variants = new Map();
		const kernel = ( src, dst ) => {

			const p = { s: uniform( 0, 'uint' ), axis: uniform( 0, 'uint' ), sign: uniform( 1 ), pre: uniform( 0, 'uint' ), post: uniform( 0, 'uint' ) };
			const node = Fn( () => {

				const t = instanceIndex;
				const isRow = p.axis.equal( 0 );
				const line = isRow.select( t.div( H ), t.mod( N ) ).toVar();
				const j = isRow.select( t.mod( H ), t.div( N ) ).toVar();
				const at = ( q ) => isRow.select( line.mul( N ).add( q ), q.mul( N ).add( line ) );
				const ns = uint( 1 ).shiftLeft( p.s ).toVar();
				const k = j.bitAnd( ns.sub( 1 ) ).toVar();
				const a0 = at( j ).toVar(), a1 = at( j.add( H ) ).toVar();
				const in0 = src.element( a0 ).toVar(), in1 = src.element( a1 ).toVar();
				If( p.pre.equal( 1 ), () => {

					in0.assign( cmul( in0, this._potFactor( a0, u.half ) ) );
					in1.assign( cmul( in1, this._potFactor( a1, u.half ) ) );

				} );
				const ang = k.toFloat().mul( p.sign ).mul( Math.PI ).div( ns.toFloat() );
				in1.assign( cmul( in1, vec2( cos( ang ), sin( ang ) ) ) );
				const d0 = j.mul( 2 ).sub( k ).toVar(), d1 = d0.add( ns ).toVar();
				const o0 = in0.add( in1 ).toVar(), o1 = in0.sub( in1 ).toVar();
				const b0 = at( d0 ).toVar(), b1 = at( d1 ).toVar();
				If( p.post.equal( 1 ), () => {

					// kinetic propagator in natural (kx, ky) order, 1/N² folded into the table
					const kl = this.ktab.element( line ).toVar();
					o0.assign( cmul( o0, cmul( kl, this.ktab.element( d0 ) ) ) );
					o1.assign( cmul( o1, cmul( kl, this.ktab.element( d1 ) ) ) );

				} ).ElseIf( p.post.equal( 2 ), () => {

					o0.assign( cmul( o0, this._potFactor( b0, u.half ) ) );
					o1.assign( cmul( o1, this._potFactor( b1, u.half ) ) );

				} );
				dst.element( b0 ).assign( o0 );
				dst.element( b1 ).assign( o1 );

			} )().compute( N * H, [ 64 ] ).setName( 'tdse-stockham' );
			return { node, p };

		};

		const A = this.psi, B = this.psiB;
		const stages = [];
		let flip = false;
		const pass = ( axis, sign, preFirst, postLast ) => {

			for ( let s = 0; s < L; s ++ ) {

				const { node, p } = flip ? kernel( B, A ) : kernel( A, B );
				p.s.value = s; p.axis.value = axis; p.sign.value = sign;
				p.pre.value = ( s === 0 ) ? preFirst : 0;
				p.post.value = ( s === L - 1 ) ? postLast : 0;
				stages.push( node );
				flip = ! flip;

			}

		};

		pass( 0, - 1, 1, 0 ); // row FFT, V/2 on load
		pass( 1, - 1, 0, 1 ); // col FFT, kinetic on store
		pass( 1, + 1, 0, 0 ); // col IFFT
		pass( 0, + 1, 0, 2 ); // row IFFT, V/2 on store
		this.stockhamStep = stages;
		variants.clear();

	}

	// Software-rasteriser path: one invocation transforms a whole row/column held in a private array (no barriers, no
	// shared memory, twiddles from a table). Same fusion and bit-reversed k-space as the shared-memory kernels.
	_buildLineFFT() {

		const { N, H, L, psi, u } = this;
		const tw = new Float32Array( H * 2 );
		for ( let m = 0; m < H; m ++ ) {

			tw[ m * 2 ] = Math.cos( - 2 * Math.PI * m / N );
			tw[ m * 2 + 1 ] = Math.sin( - 2 * Math.PI * m / N );

		}

		this.twiddle = instancedArray( tw, 'vec2' ).setName( 'twiddle' );

		const stages = ( a, sign, dif ) => {

			const order = [];
			for ( let s = 0; s < L; s ++ ) order.push( s );
			if ( dif ) order.reverse();
			for ( const s of order ) {

				const half = 1 << s;
				Loop( { start: uint( 0 ), end: uint( H ), type: 'uint', condition: '<' }, ( { i: t } ) => {

					const k = t.bitAnd( half - 1 );
					const j0 = t.mul( 2 ).sub( k ).toVar();
					const j1 = j0.add( half ).toVar();
					const x0 = a.element( j0 ).toVar(), x1 = a.element( j1 ).toVar();
					const w0 = this.twiddle.element( k.shiftLeft( L - 1 - s ) );
					const w = sign < 0 ? w0 : vec2( w0.x, w0.y.negate() );
					if ( dif ) {

						a.element( j0 ).assign( x0.add( x1 ) );
						a.element( j1 ).assign( cmul( x0.sub( x1 ), w ) );

					} else {

						const bw = cmul( x1, w ).toVar();
						a.element( j0 ).assign( x0.add( bw ) );
						a.element( j1 ).assign( x0.sub( bw ) );

					}

				} );

			}

		};

		const kernel = ( axis, mode ) => Fn( () => {

			const line = instanceIndex;
			const a = array( 'vec2', N ).toVar( 'line' );
			const at = ( p ) => axis === 'row' ? line.mul( N ).add( p ) : p.mul( N ).add( line );
			const each = ( fn ) => Loop( { start: uint( 0 ), end: uint( N ), type: 'uint', condition: '<' }, ( { i } ) => fn( i ) );

			each( ( i ) => {

				const g = at( i ).toVar();
				if ( mode === 'first' ) a.element( i ).assign( cmul( psi.element( g ), this._potFactor( g, u.half ) ) );
				else a.element( i ).assign( psi.element( g ) );

			} );

			if ( axis === 'row' ) {

				if ( mode !== 'first' ) {

					stages( a, + 1, false );
					const tau = mode === 'mid' ? u.full : u.half;
					each( ( i ) => a.element( i ).assign( cmul( a.element( i ), this._potFactor( at( i ), tau ) ) ) );

				}

				if ( mode !== 'last' ) stages( a, - 1, true );

			} else {

				stages( a, - 1, true );
				const kl = this.ktab.element( line ).toVar();
				each( ( i ) => a.element( i ).assign( cmul( a.element( i ), cmul( kl, this.ktab.element( i ) ) ) ) );
				stages( a, + 1, false );

			}

			each( ( i ) => psi.element( at( i ) ).assign( a.element( i ) ) );

		} )().compute( N, [ 4 ] ).setName( `tdse-line-${axis}-${mode}` );

		this.rowFirst = kernel( 'row', 'first' );
		this.rowMid = kernel( 'row', 'mid' );
		this.rowLast = kernel( 'row', 'last' );
		this.col = kernel( 'col', 'col' );

	}

	// Born rule at the far wall: the absorbing layer IS the detector. Probability removed from cell (x,y) during this frame,
	// (1 − e^{−2WΔt})|ψ|²dx², is summed per row and turned into discrete detection events (Bernoulli trials per row).
	_buildDetector() {

		const { N, psi, pot, u } = this;
		const dx = this.dx;
		const MAXH = this.MAX_HITS;
		const xs = Math.floor( N * 0.86 );

		this.detect = Fn( () => {

			const y = instanceIndex;
			const acc = float( 0 ).toVar();
			Loop( { start: uint( xs ), end: uint( N ), type: 'uint', condition: '<' }, ( { i } ) => {

				const g = y.mul( N ).add( i );
				const p = psi.element( g );
				const w = pot.element( g ).y;
				acc.addAssign( dot( p, p ).mul( float( 1 ).sub( exp( w.mul( u.frameTau ).mul( - 2 ) ) ) ) );

			} );
			const m = acc.mul( dx * dx ).mul( u.rate ).toVar();
			const st = this.rng.element( y ).toVar();

			for ( let j = 0; j < 2; j ++ ) {

				st.assign( st.mul( 747796405 ).add( 2891336453 ) );
				const r = hash( st );
				If( r.lessThan( m.sub( j ) ), () => {

					const slot = atomicAdd( this.counter.element( 0 ), uint( 1 ) ).toVar();
					const r2 = hash( st.add( 17 ) ), r3 = hash( st.add( 91 ) );
					this.hits.element( slot.bitAnd( MAXH - 1 ) ).assign( vec4( y.toFloat().add( r2 ).mul( dx ), r3, u.time, 1 ) );

				} );

			}

			this.rng.element( y ).assign( st );

		} )().compute( N, [ 64 ] ).setName( 'tdse-detect' );

		this.clearHits = Fn( () => {

			this.hits.element( instanceIndex ).assign( vec4( 0, 0, - 1e9, 0 ) );

		} )().compute( MAXH, [ 64 ] );

	}

	// de Broglie–Bohm guidance: v = (ħ/m) ∇arg ψ. Particles sampled from |ψ|² and advected by v stay |ψ|²-distributed
	// (equivariance), so the mist draws the probability current as luminous streamlines.
	_buildMist() {

		const { N, psi, u } = this;
		const dx = this.dx;

		const psiAt = ( ix, iy ) => psi.element( iy.mul( N ).add( ix ) );
		const rhoAt = ( q ) => {

			const ix = clamp( q.x.mul( N ), 0, N - 1 ).toUint();
			const iy = clamp( q.y.mul( N ), 0, N - 1 ).toUint();
			const p = psiAt( ix, iy );
			return dot( p, p );

		};

		this.mistStep = Fn( () => {

			const st = this.mist.element( instanceIndex ).toVar();
			const seed = instanceIndex.mul( 7919 ).add( u.frame.mul( 104729 ) ).toVar();

			// advect (2 midpoint-free Euler substeps)
			for ( let s = 0; s < 2; s ++ ) {

				const ix = clamp( st.x.mul( N ), 1, N - 2 ).toUint();
				const iy = clamp( st.y.mul( N ), 1, N - 2 ).toUint();
				const xp = psiAt( ix.add( 1 ), iy ), xm = psiAt( ix.sub( 1 ), iy );
				const yp = psiAt( ix, iy.add( 1 ) ), ym = psiAt( ix, iy.sub( 1 ) );
				const cx = cmul( xp, vec2( xm.x, xm.y.negate() ) );
				const cy = cmul( yp, vec2( ym.x, ym.y.negate() ) );
				const v = vec2( atan( cx.y, cx.x.add( 1e-30 ) ), atan( cy.y, cy.x.add( 1e-30 ) ) ).div( 2 * dx ).toVar();
				const sp = length( v );
				v.mulAssign( min( 1, float( 600 ).div( sp.add( 1e-6 ) ) ) );
				st.xy.addAssign( v.mul( u.frameTau.mul( 0.5 ) ) );

			}

			const life = hash( instanceIndex.add( 12345 ) ).mul( 2.5 ).add( 1.5 );
			st.z.addAssign( u.realDt.div( life ) );

			const rho = rhoAt( st.xy );
			const dead = st.z.greaterThanEqual( 1 ).or( rho.lessThan( u.rhoRef.mul( 2e-4 ) ) ).or( st.x.lessThan( 0.01 ) ).or( st.x.greaterThan( 0.99 ) )
				.or( st.y.lessThan( 0.01 ) ).or( st.y.greaterThan( 0.99 ) );

			If( dead, () => {

				st.z.assign( 2 ); // stays invisible unless a candidate is accepted
				const found = float( 0 ).toVar();
				for ( let j = 0; j < 4; j ++ ) {

					If( found.lessThan( 0.5 ), () => {

						const q = vec2( hash( seed.add( j * 3 + 1 ) ), hash( seed.add( j * 3 + 2 ) ) ).mul( 0.96 ).add( 0.02 ).toVar();
						const acc = rhoAt( q ).div( u.rhoRef.mul( 0.08 ) );
						If( hash( seed.add( j * 3 + 3 ) ).lessThan( acc ).and( u.mistOn.greaterThan( 0.5 ) ), () => {

							st.xy.assign( q );
							st.z.assign( hash( seed.add( 77 ) ).mul( 0.1 ) );
							st.w.assign( hash( seed.add( 78 ) ) );
							found.assign( 1 );

						} );

					} );

				}

			} );

			this.mist.element( instanceIndex ).assign( st );

		} )().compute( this.MIST, [ 64 ] ).setName( 'tdse-mist' );

	}

	// ───────────────────────── host side ─────────────────────────

	setTimeStep( dt ) {

		const { N, L } = this;
		this.dt = dt;
		this.u.half.value = dt / 2;
		this.u.full.value = dt;
		for ( let j = 0; j < N; j ++ ) {

			const n = this.fft === 'global' ? j : bitrev( j, L ); // shared-memory path keeps k-space bit-reversed
			const kk = 2 * Math.PI * ( n < N / 2 ? n : n - N );
			const ph = - 0.5 * kk * kk * dt;
			this.kArray[ j * 2 ] = Math.cos( ph ) / N;
			this.kArray[ j * 2 + 1 ] = Math.sin( ph ) / N;

		}

		this.ktab.value.needsUpdate = true;

	}

	uploadPotential() {

		this.pot.value.needsUpdate = true;

	}

	stepList( steps ) {

		let l = this._lists.get( steps );
		if ( ! l && this.fft === 'global' ) {

			l = [];
			for ( let s = 0; s < steps; s ++ ) l.push( ...this.stockhamStep );
			this._lists.set( steps, l );

		} else if ( ! l ) {

			l = [ this.rowFirst ];
			for ( let s = 0; s < steps; s ++ ) {

				l.push( this.col );
				l.push( s < steps - 1 ? this.rowMid : this.rowLast );

			}

			this._lists.set( steps, l );

		}

		return l;

	}

	launch( cx, cy, kx, ky, sigma ) {

		this.u.c.value.set( cx, cy );
		this.u.k.value.set( kx, ky );
		this.u.sigma.value = sigma;
		this.renderer.compute( this.initPacket );

	}

	async readNorm() {

		this.renderer.compute( this.normPass );
		const buf = await this.renderer.getArrayBufferAsync( this.rowSum.value );
		const a = new Float32Array( buf );
		let s = 0;
		for ( let i = 0; i < a.length; i ++ ) s += a[ i ];
		return s;

	}

	async readHits() {

		const buf = await this.renderer.getArrayBufferAsync( this.counterAttr );
		return new Uint32Array( buf )[ 0 ];

	}

	resetHits() {

		this.counterAttr.array.fill( 0 );
		this.counterAttr.needsUpdate = true;
		this.renderer.compute( this.clearHits );

	}

	dispose() {

		const nodes = [ ...( this.stockhamStep || [] ), this.rowFirst, this.rowMid, this.rowLast, this.col, this.initPacket, this.zero, this.normPass, this.derive,
			this.paint, this.detect, this.clearHits, this.mistStep ];
		for ( const n of nodes ) n?.dispose?.();
		for ( const b of [ this.twiddle, this.psi, this.psiB, this.pot, this.ktab, this.rowSum, this.rng, this.hits, this.mist ] ) b?.value?.dispose?.();
		this.counterAttr.dispose?.();
		this.field.dispose();
		this._lists.clear();

	}

}
