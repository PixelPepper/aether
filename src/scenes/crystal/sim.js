// Gravner–Griffeath mesoscopic snow-crystal model on a hexagonal (triangular) lattice, entirely in compute shaders.
//
// Lattice: axial coordinates (q, r) stored row-major in an N×N array; the 6 neighbours of (q, r) are
// (q±1, r), (q, r±1), (q+1, r−1), (q−1, r+1). Physical position: x = q + r/2, y = r·√3/2.
// The live domain is the hexagon of hex-radius RD around the centre (a D6-symmetric box). Cells outside it are never
// written and keep vapour density ρ: a Dirichlet far-field reservoir.
//
// Per-site state vec4: (a, b, c, d)
//   a  0 = vapour / boundary site; > 0 = attached to the crystal (the value is 1 + growth-time at attachment)
//   b  boundary (quasi-liquid) mass
//   c  crystal mass (after attachment: the final ice mass of the cell = relief height of the rendered flake)
//   d  diffusive (vapour) mass
//
// One model step = two dispatches with ping-pong buffers:
//   K1 (S0→S1): diffusion with reflecting crystal boundary (the 7-point average where attached neighbours are replaced
//               by the centre value), optional ±σ noise, then *freezing* at boundary sites:
//               b += (1−κ)d, c += κd, d = 0.
//   K2 (S1→S0): *attachment* at boundary sites with n attached neighbours:
//               n ≤ 2: b ≥ β;   n = 3: b ≥ 1  or  (Σ_{N_x} d < θ and b ≥ α);   n ≥ 4: always.
//               On attachment c ← b + c, b ← 0. Otherwise *melting*: d += μb + γc, b ← (1−μ)b, c ← (1−γ)c.
//   (Melting in the paper acts on the new boundary ∂A_{t+1}; sites that just became boundary sites carry b = c = 0,
//    so applying it to the non-attaching old boundary sites in the same pass is exactly equivalent.)
// Attachment events are counted with atomics (radius, cells, mass) and read back asynchronously.
//
// A bake pass resamples the hex lattice into a Cartesian rgba16f storage texture for rendering:
//   r = attached indicator (barycentric), g = relief height, b = birth time, a = quasi-liquid layer (b/β) at the front.

import * as THREE from 'three/webgpu';
import {
	Fn, If, Return, uniform, instancedArray, storage, globalId, instanceIndex, int, uint, float, vec4, uvec2,
	max, abs, floor, select, clamp, textureStore, atomicAdd, atomicMax, atomicStore, hash,
} from 'three/tsl';

const SQ3_2 = Math.sqrt( 3 ) / 2;

export class SnowLattice {

	constructor( renderer, { N, bakeSize } ) {

		this.renderer = renderer;
		this.N = N;
		this.C = N >> 1;
		this.RD = this.C - 2; // live hex radius
		this.B = bakeSize;
		this.attrs = [];
		this.computes = [];

		this.S0 = instancedArray( N * N, 'vec4' );
		this.S1 = instancedArray( N * N, 'vec4' );
		this.attrs.push( this.S0.value, this.S1.value );

		this.statsAttr = new THREE.StorageBufferAttribute( new Uint32Array( 4 ), 1 );
		this.attrs.push( this.statsAttr );
		this.stats = storage( this.statsAttr, 'uint', 4 ).toAtomic();

		this.u = {
			rho: uniform( 0.6 ), beta: uniform( 1.6 ), alpha: uniform( 0.4 ), theta: uniform( 0.025 ),
			kappa: uniform( 0.0025 ), mu: uniform( 0.015 ), gamma: uniform( 0.0005 ), sigma: uniform( 0 ),
			time: uniform( 1 ), // 1 + growth time (s); stored in `a` on attachment
			offX: uniform( 0, 'int' ), offY: uniform( 0, 'int' ),
			ratio: uniform( 1 ), seed: uniform( 0, 'uint' ),
		};

		this.textures = [ this.makeTexture(), this.makeTexture() ];
		this.current = 0; // which texture the bake writes

		this.build();

		this.steps = 0;
		this.radius = 0;
		this.cells = 1;
		this.mass = 1;
		this._pending = false;
		this._stepLists = new Map();
		this._readSerial = 0;

	}

	makeTexture() {

		const t = new THREE.StorageTexture( this.B, this.B );
		t.type = THREE.HalfFloatType;
		t.format = THREE.RGBAFormat;
		t.generateMipmaps = false;
		t.minFilter = THREE.LinearFilter;
		t.magFilter = THREE.LinearFilter;
		t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
		return t;

	}

	build() {

		const { N, C, RD, u, stats } = this;
		const NB = [ 1, - 1, N, - N, 1 - N, N - 1 ];

		const site = () => {

			const q = int( globalId.x ).add( u.offX ).toVar( 'q' );
			const r = int( globalId.y ).add( u.offY ).toVar( 'r' );
			const x = q.sub( C ), y = r.sub( C );
			const hd = max( max( abs( x ), abs( y ) ), abs( x.add( y ) ) ).toVar( 'hd' );
			If( hd.greaterThan( int( RD ) ), () => {

				Return();

			} );
			return { i: r.mul( N ).add( q ).toVar( 'i' ), hd };

		};

		// K1: diffusion (reflecting boundary) + noise + freezing
		const makeK1 = ( src, dst ) => Fn( () => {

			const { i } = site();
			const s = src.element( i ).toVar( 's' );
			If( s.x.greaterThan( 0.0 ), () => {

				dst.element( i ).assign( s );
				Return();

			} );
			const sum = s.w.toVar( 'sum' );
			const nA = float( 0 ).toVar( 'nA' );
			for ( const o of NB ) {

				const yv = src.element( i.add( int( o ) ) );
				const att = yv.x.greaterThan( 0.0 );
				sum.addAssign( select( att, s.w, yv.w ) );
				nA.addAssign( select( att, float( 1 ), float( 0 ) ) );

			}

			const d = sum.mul( 1 / 7 ).toVar( 'd' );
			If( u.sigma.greaterThan( 0.0 ), () => {

				const h = hash( uint( i ).add( u.seed ).add( uint( s.w.mul( 8388608.0 ) ) ) );
				d.mulAssign( select( h.greaterThan( 0.5 ), u.sigma, u.sigma.negate() ).add( 1.0 ) );

			} );
			const b = s.y.toVar( 'b' ), c = s.z.toVar( 'c' );
			If( nA.greaterThan( 0.5 ), () => {

				b.addAssign( float( 1 ).sub( u.kappa ).mul( d ) );
				c.addAssign( u.kappa.mul( d ) );
				d.assign( 0 );

			} );
			dst.element( i ).assign( vec4( s.x, b, c, d ) );

		} )().computeKernel( [ 8, 8, 1 ] );

		// K2: attachment + melting
		const makeK2 = ( src, dst ) => Fn( () => {

			const { i, hd } = site();
			const s = src.element( i ).toVar( 's' );
			If( s.x.greaterThan( 0.0 ), () => {

				dst.element( i ).assign( s );
				Return();

			} );
			const sum = s.w.toVar( 'sum' );
			const nA = float( 0 ).toVar( 'nA' );
			for ( const o of NB ) {

				const yv = src.element( i.add( int( o ) ) );
				nA.addAssign( select( yv.x.greaterThan( 0.0 ), float( 1 ), float( 0 ) ) );
				sum.addAssign( yv.w );

			}

			If( nA.lessThan( 0.5 ), () => {

				dst.element( i ).assign( s );
				Return();

			} );

			const b = s.y, c = s.z, d = s.w;
			const tip = nA.lessThan( 2.5 ).and( b.greaterThanEqual( u.beta ) );
			const kink = nA.greaterThan( 2.5 ).and( nA.lessThan( 3.5 ) ).and(
				b.greaterThanEqual( 1.0 ).or( sum.lessThan( u.theta ).and( b.greaterThanEqual( u.alpha ) ) ) );
			const deep = nA.greaterThan( 3.5 );

			If( tip.or( kink ).or( deep ), () => {

				const cc = b.add( c );
				dst.element( i ).assign( vec4( u.time, 0, cc, 0 ) );
				atomicMax( stats.element( 0 ), uint( hd ) );
				atomicAdd( stats.element( 1 ), uint( 1 ) );
				atomicAdd( stats.element( 2 ), uint( cc.mul( 64.0 ) ) );

			} ).Else( () => {

				dst.element( i ).assign( vec4( 0, b.mul( float( 1 ).sub( u.mu ) ), c.mul( float( 1 ).sub( u.gamma ) ),
					d.add( u.mu.mul( b ) ).add( u.gamma.mul( c ) ) ) );

			} );

		} )().computeKernel( [ 8, 8, 1 ] );

		this.k1 = makeK1( this.S0, this.S1 );
		this.k2 = makeK2( this.S1, this.S0 );

		// reset: vapour ρ everywhere, a single frozen seed cell at the centre
		const centre = C * N + C;
		this.kReset = Fn( () => {

			const i = instanceIndex;
			const v = select( i.equal( uint( centre ) ), vec4( 1, 0, 1, 0 ), vec4( 0, 0, 0, u.rho ) );
			this.S0.element( i ).assign( v );
			this.S1.element( i ).assign( v );
			If( i.lessThan( uint( 4 ) ), () => {

				atomicStore( stats.element( i ), uint( 0 ) );

			} );

		} )().compute( N * N, [ 64 ] );

		// cloud-layer jump: rescale vapour everywhere (both buffers, incl. the far-field reservoir)
		this.kRescale = Fn( () => {

			const i = instanceIndex;
			const a = this.S0.element( i );
			If( a.x.lessThanEqual( 0.0 ), () => {

				a.w.mulAssign( u.ratio );

			} );
			const b = this.S1.element( i );
			If( b.x.lessThanEqual( 0.0 ), () => {

				b.w.mulAssign( u.ratio );

			} );

		} )().compute( N * N, [ 64 ] );

		// bake: hex lattice → Cartesian texture (barycentric interpolation inside the lattice triangle)
		const B = this.B;
		const makeBake = ( tex ) => Fn( () => {

			const px = globalId.x, py = globalId.y;
			If( px.greaterThanEqual( uint( B ) ).or( py.greaterThanEqual( uint( B ) ) ), () => {

				Return();

			} );
			// texel centre → physical lattice units (domain hex radius RD maps to ±1 in uv space)
			const X = float( px ).add( 0.5 ).div( B ).mul( 2 ).sub( 1 ).mul( RD );
			const Y = float( py ).add( 0.5 ).div( B ).mul( 2 ).sub( 1 ).mul( RD );
			const rr = Y.div( SQ3_2 );
			const qq = X.sub( rr.mul( 0.5 ) );
			const q0 = floor( qq ), r0 = floor( rr );
			const fq = qq.sub( q0 ), fr = rr.sub( r0 );
			const upper = fq.add( fr ).greaterThan( 1.0 );
			// triangle vertices (relative to q0, r0) and weights
			const aq = select( upper, float( 1 ), float( 0 ) ), ar = aq; // (0,0) or (1,1)
			const wa = select( upper, fq.add( fr ).sub( 1 ), float( 1 ).sub( fq ).sub( fr ) );
			const wb = select( upper, float( 1 ).sub( fr ), fq ); // (1,0)
			const wc = select( upper, float( 1 ).sub( fq ), fr ); // (0,1)

			const acc = vec4( 0 ).toVar( 'acc' );
			const tap = ( dq, dr, w ) => {

				const q = clamp( int( q0.add( dq ) ).add( C ), 0, N - 1 );
				const r = clamp( int( r0.add( dr ) ).add( C ), 0, N - 1 );
				const s = this.S0.element( r.mul( N ).add( q ) );
				const att = s.x.greaterThan( 0.0 );
				const ind = select( att, float( 1 ), float( 0 ) );
				// relief: crystal mass normalised by the tip threshold β (ribs carry more mass than the plates between them)
				const h = select( att, clamp( s.z.div( u.beta ), 0.12, 2.0 ), float( 0 ) );
				const birth = select( att, s.x.sub( 1 ), float( 0 ) );
				const qll = select( att, float( 0 ), clamp( s.y.div( u.beta ), 0, 1.5 ) );
				acc.addAssign( vec4( ind, h, birth, qll ).mul( w ) );

			};

			tap( aq, ar, wa );
			tap( float( 1 ), float( 0 ), wb );
			tap( float( 0 ), float( 1 ), wc );
			// birth is averaged over attached taps only
			const ind = acc.x;
			const out = vec4( ind, acc.y, acc.z.div( max( ind, 1e-3 ) ), acc.w );
			textureStore( tex, uvec2( px, py ), out ).toWriteOnly();

		} )().computeKernel( [ 8, 8, 1 ] );

		this.kBake = [ makeBake( this.textures[ 0 ] ), makeBake( this.textures[ 1 ] ) ];
		this.bakeDispatch = [ Math.ceil( B / 8 ), Math.ceil( B / 8 ), 1 ];

		this.computes.push( this.k1, this.k2, this.kReset, this.kRescale, ...this.kBake );

	}

	setParams( p ) {

		const u = this.u;
		u.beta.value = p.beta;
		u.alpha.value = p.alpha;
		u.theta.value = p.theta;
		u.kappa.value = p.kappa;
		u.mu.value = p.mu;
		u.gamma.value = p.gamma;

	}

	/** Change far-field vapour density: rescale the whole vapour field (a cloud layer of different supersaturation). */
	setRho( rho ) {

		const u = this.u;
		const old = u.rho.value;
		u.rho.value = rho;
		if ( Math.abs( rho / old - 1 ) < 1e-5 ) return;
		u.ratio.value = rho / old;
		this.renderer.compute( this.kRescale );

	}

	reset( p, growTime = 0 ) {

		this.u.rho.value = p.rho;
		this.setParams( p );
		this.u.time.value = 1 + growTime;
		this.renderer.compute( this.kReset );
		this.steps = 0;
		this.radius = 0;
		this.cells = 1;
		this.mass = 1;
		this._readSerial ++; // discard in-flight readbacks of the previous flake

	}

	/** Run `n` model steps over the active region (crystal radius + diffusion margin). */
	step( n ) {

		if ( n <= 0 ) return;
		const { C, RD } = this;
		const margin = Math.max( 24, 2.5 * Math.sqrt( this.steps + n ) + 12 );
		const R = Math.min( RD, Math.ceil( this.radius + margin ) );
		const W = 2 * R + 1;
		this.u.offX.value = C - R;
		this.u.offY.value = C - R;
		this.u.seed.value = ( Math.random() * 0xffffff ) >>> 0;
		let list = this._stepLists.get( n );
		if ( ! list ) {

			list = [];
			for ( let k = 0; k < n; k ++ ) list.push( this.k1, this.k2 );
			this._stepLists.set( n, list );

		}

		const g = Math.ceil( W / 8 );
		this.renderer.compute( list, [ g, g, 1 ] );
		this.steps += n;

	}

	bake() {

		this.renderer.compute( this.kBake[ this.current ], this.bakeDispatch );

	}

	get texture() {

		return this.textures[ this.current ];

	}

	/** Async stats readback (radius in cells, attached cells, crystal mass). Non-blocking; at most one in flight. */
	poll() {

		if ( this._pending ) return this._promise;
		this._pending = true;
		const serial = this._readSerial;
		const stepsAt = this.steps;
		this._promise = this.renderer.getArrayBufferAsync( this.statsAttr ).then( ( buf ) => {

			if ( serial !== this._readSerial ) return null;
			const a = new Uint32Array( buf );
			this.radius = Math.max( this.radius, a[ 0 ] );
			this.cells = Math.max( 1, a[ 1 ] + 1 );
			this.mass = 1 + a[ 2 ] / 64;
			return { radius: this.radius, steps: stepsAt };

		} ).catch( () => null ).finally( () => {

			this._pending = false;

		} );
		return this._promise;

	}

	dispose() {

		for ( const c of this.computes ) c.dispose();
		for ( const t of this.textures ) t.dispose();
		const attrs = this.renderer._attributes;
		for ( const a of this.attrs ) {

			try {

				attrs?.delete( a );

			} catch ( e ) { /* never uploaded */ }

		}

		this.attrs.length = 0;
		this._stepLists.clear();

	}

}
