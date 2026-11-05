// GPU Particle-Mesh N-body solver in comoving coordinates (WebGPU compute, TSL).
//
//  per step:  CIC deposit (fixed-point atomicAdd into an i32 grid)
//          →  i32 → complex δ (+ clear)
//          →  3D FFT: x, y, [z · Green · z⁻¹] (fused in shared memory), y⁻¹, x⁻¹   (radix-2 Stockham, one line per workgroup)
//          →  −∇φ by central differences (+ smoothed density → 3D texture for the volume pass)
//          →  CIC gather + kick + drift (leapfrog, KDK with merged half-kicks)
//
//  initial conditions are built on the GPU with the same FFT: white noise → δ(k)·√P(k) (+ primordial seed peaks)
//  → Zel'dovich displacement s(k) = i k δ(k) / k² → two inverse 3D FFTs (s_x + i s_y packed, s_z) → lattice + D(a)·s.

import * as THREE from 'three/webgpu';
import {
	Fn, instancedArray, instanceIndex, uint, int, float, vec2, vec3, vec4, uvec3, uniform, uniformArray,
	workgroupArray, workgroupBarrier, atomicAdd, atomicLoad, atomicStore, floor, sin, cos, sqrt, log, pow, exp,
	max, select, textureStore, Loop,
} from 'three/tsl';

export const MAX_SEEDS = 8;

export class ParticleMesh {

	constructor( renderer, { N, NL, L } ) {

		this.renderer = renderer;
		this.N = N; // mesh cells per side (power of two)
		this.NL = NL; // particles per side (power of two)
		this.L = L; // box size, Mpc/h
		this.N3 = N * N * N;
		this.NP = NL * NL * NL;
		this.LOG = Math.log2( N );
		this.mass = this.N3 / this.NP; // particle mass in units of mean cell mass
		this.FIX = 65536; // fixed-point scale for the atomic mass grid

		const N3 = this.N3, NP = this.NP;
		this.grid = instancedArray( N3, 'int' ).toAtomic().setName( 'cwGrid' );
		this.C = instancedArray( N3, 'vec2' ).setName( 'cwC' );
		this.Dk = instancedArray( N3, 'vec2' ).setName( 'cwD' );
		this.rho = instancedArray( N3, 'float' ).setName( 'cwRho' );
		this.F = instancedArray( N3, 'vec4' ).setName( 'cwF' );
		this.pos = instancedArray( NP, 'vec4' ).setName( 'cwPos' ); // xyz grid units, w = local density
		this.vel = instancedArray( NP, 'vec4' ).setName( 'cwVel' ); // xyz momentum, w = random

		const vt = new THREE.Storage3DTexture( N, N, N );
		vt.type = THREE.HalfFloatType;
		vt.format = THREE.RGBAFormat;
		vt.generateMipmaps = false;
		vt.wrapS = vt.wrapT = vt.wrapR = THREE.RepeatWrapping;
		vt.minFilter = vt.magFilter = THREE.LinearFilter;
		this.volTex = vt;

		this.u = {
			kick: uniform( 0 ),
			drift: uniform( 0 ),
			green: uniform( 0 ),
			seed: uniform( 1, 'uint' ),
			amp: uniform( 1 ), // P(k) amplitude A
			ns: uniform( 0.96 ),
			gamma: uniform( 0.21 ),
			D0: uniform( 0 ),
			P0: uniform( 0 ),
			seedR: uniform( 2.5 ),
			seeds: uniformArray( Array.from( { length: MAX_SEEDS }, () => new THREE.Vector4() ), 'vec4' ),
		};

		this._build();

	}

	// ───────────────────────── helpers ─────────────────────────

	cellIndex( ix, iy, iz ) {

		const N = this.N;
		return ix.add( iy.mul( N ) ).add( iz.mul( N * N ) );

	}

	// periodic CIC stencil around p (grid units, cell centres at i+0.5)
	cic( p ) {

		const g = p.sub( 0.5 );
		const f = floor( g );
		const d = g.sub( f ).toVar();
		const m = int( this.N - 1 );
		const x0 = int( f.x ).bitAnd( m ).toVar(), y0 = int( f.y ).bitAnd( m ).toVar(), z0 = int( f.z ).bitAnd( m ).toVar();
		const x1 = x0.add( 1 ).bitAnd( m ).toVar(), y1 = y0.add( 1 ).bitAnd( m ).toVar(), z1 = z0.add( 1 ).bitAnd( m ).toVar();
		const corners = [];
		for ( let c = 0; c < 8; c ++ ) {

			const cx = c & 1, cy = ( c >> 1 ) & 1, cz = ( c >> 2 ) & 1;
			const w = ( cx ? d.x : d.x.oneMinus() ).mul( cy ? d.y : d.y.oneMinus() ).mul( cz ? d.z : d.z.oneMinus() );
			const idx = this.cellIndex( cx ? x1 : x0, cy ? y1 : y0, cz ? z1 : z0 );
			corners.push( { w, idx } );

		}

		return corners;

	}

	// PCG hash → [0,1)
	static pcg( v ) {

		const state = v.mul( 747796405 ).add( 2891336453 );
		const word = state.shiftRight( state.shiftRight( 28 ).add( 4 ) ).bitXor( state ).mul( 277803737 );
		return word.shiftRight( 22 ).bitXor( word );

	}

	// radix-2 Stockham FFT along one axis; one workgroup (N/2 threads) per line, ping-pong in workgroup memory.
	// mode: 'fwd' | 'inv' | 'poisson' (forward → multiply by Green's function → inverse, all in shared memory)
	makeFFT( buf, axis, mode ) {

		const N = this.N, H = N >> 1, LOG = this.LOG;
		// one workgroup array holds both ping-pong halves; stages run in a real loop so the shader stays small
		// (SwiftShader turns every barrier site into a coroutine split — unrolled code compiles very slowly there)
		const W = workgroupArray( 'vec2', 2 * N );
		const green = this.u.green;

		const fn = Fn( () => {

			const t = instanceIndex.bitAnd( uint( H - 1 ) ).toVar( 't' );
			const lineId = instanceIndex.shiftRight( uint( LOG - 1 ) ).toVar( 'lineId' );
			let base, stride;
			if ( axis === 0 ) {

				base = lineId.mul( uint( N ) ); stride = 1;

			} else if ( axis === 1 ) {

				base = lineId.bitAnd( uint( N - 1 ) ).add( lineId.shiftRight( uint( LOG ) ).mul( uint( N * N ) ) ); stride = N;

			} else {

				base = lineId; stride = N * N;

			}

			const i0 = base.add( t.mul( uint( stride ) ) ).toVar( 'i0' );
			const i1 = i0.add( uint( H * stride ) ).toVar( 'i1' );
			const tH = t.add( uint( H ) ).toVar( 'tH' );
			const srcOff = uint( 0 ).toVar( 'srcOff' );
			const dstOff = uint( N ).toVar( 'dstOff' );

			W.element( t ).assign( buf.element( i0 ) );
			W.element( tH ).assign( buf.element( i1 ) );
			workgroupBarrier();

			const stages = ( sign ) => {

				const Ns = uint( 1 ).toVar();
				Loop( { start: uint( 0 ), end: uint( LOG ), type: 'uint', condition: '<' }, () => {

					const k = t.bitAnd( Ns.sub( 1 ) ).toVar();
					const v0 = W.element( srcOff.add( t ) ).toVar();
					const v1 = W.element( srcOff.add( tH ) ).toVar();
					const ang = float( k ).mul( sign * Math.PI ).div( float( Ns ) ).toVar();
					const c = cos( ang ).toVar(), sn = sin( ang ).toVar();
					const w1 = vec2( v1.x.mul( c ).sub( v1.y.mul( sn ) ), v1.x.mul( sn ).add( v1.y.mul( c ) ) ).toVar();
					const o = dstOff.add( t.sub( k ).shiftLeft( uint( 1 ) ).add( k ) ).toVar();
					W.element( o ).assign( v0.add( w1 ) );
					W.element( o.add( Ns ) ).assign( v0.sub( w1 ) );
					workgroupBarrier();
					const tmp = srcOff.toVar();
					srcOff.assign( dstOff );
					dstOff.assign( tmp );
					Ns.shiftLeftAssign( uint( 1 ) );

				} );

			};

			stages( mode === 'inv' ? 1 : - 1 );

			if ( mode === 'poisson' ) {

				// lineId = x + y N ; element index along z = kz
				const fx = float( lineId.bitAnd( uint( N - 1 ) ) ).mul( Math.PI / N );
				const fy = float( lineId.shiftRight( uint( LOG ) ) ).mul( Math.PI / N );
				const sxy = sin( fx ).pow2().add( sin( fy ).pow2() ).toVar();
				for ( const m of [ t, tH ] ) {

					const sz = sin( float( m ).mul( Math.PI / N ) ).pow2();
					const k2 = sxy.add( sz ).mul( 4 );
					const G = select( k2.greaterThan( 1e-9 ), green.div( max( k2, 1e-9 ) ), float( 0 ) );
					const e = W.element( srcOff.add( m ) );
					e.assign( e.mul( G ) );

				}

				workgroupBarrier();
				stages( 1 );

			}

			buf.element( i0 ).assign( W.element( srcOff.add( t ) ) );
			buf.element( i1 ).assign( W.element( srcOff.add( tH ) ) );

		} );

		return fn().compute( N * N * H, [ H ] ).setName( `cwFFT_${axis}_${mode}` );

	}

	// ───────────────────────── kernels ─────────────────────────

	_build() {

		const { N, NL, N3, NP, LOG, FIX } = this;
		const { grid, C, Dk, rho, F, pos, vel, u } = this;
		const mFix = this.mass * FIX;
		const mask = uint( N - 1 );

		// ── CIC deposit ──
		this.kDeposit = Fn( () => {

			const p = pos.element( instanceIndex ).xyz;
			for ( const { w, idx } of this.cic( p ) ) {

				atomicAdd( grid.element( idx ), int( w.mul( mFix ).add( 0.5 ) ) );

			}

		} )().compute( NP ).setName( 'cwDeposit' );

		// ── fixed point → complex density contrast, clear the atomic grid ──
		this.kToComplex = Fn( () => {

			const v = atomicLoad( grid.element( instanceIndex ) ).toVar();
			atomicStore( grid.element( instanceIndex ), int( 0 ) );
			const r = float( v ).mul( 1 / FIX ).toVar();
			rho.element( instanceIndex ).assign( r );
			C.element( instanceIndex ).assign( vec2( r.sub( 1 ), 0 ) );

		} )().compute( N3 ).setName( 'cwToComplex' );

		this.fft = {
			fx: this.makeFFT( C, 0, 'fwd' ),
			fy: this.makeFFT( C, 1, 'fwd' ),
			fz: this.makeFFT( C, 2, 'fwd' ),
			pz: this.makeFFT( C, 2, 'poisson' ),
			iz: this.makeFFT( C, 2, 'inv' ),
			iy: this.makeFFT( C, 1, 'inv' ),
			ix: this.makeFFT( C, 0, 'inv' ),
			dz: this.makeFFT( Dk, 2, 'inv' ),
			dy: this.makeFFT( Dk, 1, 'inv' ),
			dx: this.makeFFT( Dk, 0, 'inv' ),
		};

		// ── −∇φ (2-point central differences) + 7-point smoothed density ──
		this.kGradient = Fn( () => {

			const i = instanceIndex;
			const x = i.bitAnd( mask ), y = i.shiftRight( uint( LOG ) ).bitAnd( mask ), z = i.shiftRight( uint( 2 * LOG ) );
			const xp = x.add( 1 ).bitAnd( mask ), xm = x.add( N - 1 ).bitAnd( mask );
			const yp = y.add( 1 ).bitAnd( mask ), ym = y.add( N - 1 ).bitAnd( mask );
			const zp = z.add( 1 ).bitAnd( mask ), zm = z.add( N - 1 ).bitAnd( mask );
			const ci = ( a, b, c ) => this.cellIndex( a, b, c );
			const phi = ( a, b, c ) => C.element( ci( a, b, c ) ).x;
			const gx = phi( xm, y, z ).sub( phi( xp, y, z ) ).mul( 0.5 );
			const gy = phi( x, ym, z ).sub( phi( x, yp, z ) ).mul( 0.5 );
			const gz = phi( x, y, zm ).sub( phi( x, y, zp ) ).mul( 0.5 );
			const R = ( a, b, c ) => rho.element( ci( a, b, c ) );
			const rs = rho.element( i ).mul( 2 )
				.add( R( xp, y, z ) ).add( R( xm, y, z ) )
				.add( R( x, yp, z ) ).add( R( x, ym, z ) )
				.add( R( x, y, zp ) ).add( R( x, y, zm ) ).mul( 1 / 8 ).toVar();
			F.element( i ).assign( vec4( gx, gy, gz, rs ) );
			textureStore( this.volTex, uvec3( x, y, z ), vec4( rs, 0, 0, 1 ) ).toWriteOnly();

		} )().compute( N3 ).setName( 'cwGradient' );

		// ── CIC gather + kick + drift ──
		this.kKickDrift = Fn( () => {

			const P = pos.element( instanceIndex );
			const V = vel.element( instanceIndex );
			const x = P.xyz.toVar();
			const g = vec4( 0 ).toVar();
			for ( const { w, idx } of this.cic( x ) ) g.addAssign( F.element( idx ).mul( w ) );
			const p = V.xyz.add( g.xyz.mul( u.kick ) ).toVar();
			const xn = x.add( p.mul( u.drift ) ).toVar();
			xn.assign( xn.sub( floor( xn.mul( 1 / N ) ).mul( N ) ) );
			P.assign( vec4( xn, g.w ) );
			V.assign( vec4( p, V.w ) );

		} )().compute( NP ).setName( 'cwKickDrift' );

		// ── IC: Gaussian white noise (Box–Muller on PCG hashes) ──
		this.kNoise = Fn( () => {

			const h1 = ParticleMesh.pcg( instanceIndex.mul( 2 ).bitXor( ParticleMesh.pcg( u.seed ) ) ).toVar();
			const h2 = ParticleMesh.pcg( h1.add( 0x9e3779b9 ) ).toVar();
			const u1 = float( h1.shiftRight( 8 ) ).add( 0.5 ).mul( 1 / 16777216 );
			const u2 = float( h2.shiftRight( 8 ) ).mul( 1 / 16777216 );
			const g = sqrt( log( u1 ).mul( - 2 ) ).mul( cos( u2.mul( 2 * Math.PI ) ) );
			C.element( instanceIndex ).assign( vec2( g, 0 ) );

		} )().compute( N3 ).setName( 'cwNoise' );

		// ── IC: shape the spectrum, add primordial seed peaks, build Zel'dovich displacement spectra ──
		const Vcell = Math.pow( this.L / N, 3 );
		this.kShape = Fn( () => {

			const i = instanceIndex;
			const ix = i.bitAnd( mask ), iy = i.shiftRight( uint( LOG ) ).bitAnd( mask ), iz = i.shiftRight( uint( 2 * LOG ) );
			const fk = ( v ) => float( int( v ).sub( select( v.greaterThanEqual( uint( N / 2 ) ), int( N ), int( 0 ) ) ) );
			const kf = vec3( fk( ix ), fk( iy ), fk( iz ) ).mul( 2 * Math.PI / N ).toVar(); // rad / cell
			const k2 = kf.dot( kf ).toVar();
			const kphys = sqrt( k2 ).mul( N / this.L ).toVar(); // h/Mpc
			// BBKS
			const q = kphys.div( u.gamma ).max( 1e-6 ).toVar();
			const T = log( q.mul( 2.34 ).add( 1 ) ).div( q.mul( 2.34 ) ).mul(
				pow( q.mul( 3.89 ).add( 1 ).add( q.mul( 16.1 ).pow2() ).add( pow( q.mul( 5.46 ), 3 ) ).add( pow( q.mul( 6.71 ), 4 ) ), - 0.25 ) );
			const P = u.amp.mul( pow( kphys.max( 1e-6 ), u.ns ) ).mul( T.mul( T ) );
			const amp = sqrt( P.div( Vcell ) );
			const d = C.element( i ).mul( amp ).toVar();
			// primordial seed peaks: Gaussian δ(x) = A exp(−|x−q0|²/2R²)  →  A (2πR²)^{3/2} e^{−k²R²/2} e^{−ik·q0}
			const R2 = u.seedR.mul( u.seedR );
			const env = exp( k2.mul( R2 ).mul( - 0.5 ) ).mul( pow( R2.mul( 2 * Math.PI ), 1.5 ) ).toVar();
			for ( let s = 0; s < MAX_SEEDS; s ++ ) {

				const sd = u.seeds.element( s );
				const ph = kf.dot( sd.xyz.sub( 0.5 ) ).negate();
				d.addAssign( vec2( cos( ph ), sin( ph ) ).mul( sd.w.mul( env ) ) );

			}

			// s(k) = i k δ / k² ; zero the derivative at Nyquist ; 1/N³ for the unnormalised inverse FFT
			const nyq = ( v ) => select( v.equal( uint( N / 2 ) ), float( 0 ), float( 1 ) );
			const kd = kf.mul( vec3( nyq( ix ), nyq( iy ), nyq( iz ) ) );
			const inv = select( k2.greaterThan( 1e-12 ), float( 1 / N3 ).div( k2.max( 1e-12 ) ), float( 0 ) ).toVar();
			const id = vec2( d.y.negate(), d.x ).mul( inv ).toVar(); // i δ / k²
			const sx = id.mul( kd.x ), sy = id.mul( kd.y ), sz = id.mul( kd.z );
			C.element( i ).assign( vec2( sx.x.sub( sy.y ), sx.y.add( sy.x ) ) ); // s_x + i s_y
			Dk.element( i ).assign( sz );

		} )().compute( N3 ).setName( 'cwShape' );

		// ── IC: lattice + Zel'dovich displacement ──
		this.kParticles = Fn( () => {

			const i = instanceIndex;
			const lm = uint( NL - 1 ), lg = Math.log2( NL );
			const a = i.bitAnd( lm ), b = i.shiftRight( uint( lg ) ).bitAnd( lm ), c = i.shiftRight( uint( 2 * lg ) );
			const q = vec3( float( a ), float( b ), float( c ) ).add( 0.5 ).mul( N / NL ).toVar();
			const s = vec3( 0 ).toVar();
			for ( const { w, idx } of this.cic( q ) ) {

				const cxy = C.element( idx ), cz = Dk.element( idx );
				s.addAssign( vec3( cxy.x, cxy.y, cz.x ).mul( w ) );

			}

			const x = q.add( s.mul( u.D0 ) ).toVar();
			x.assign( x.sub( floor( x.mul( 1 / N ) ).mul( N ) ) );
			pos.element( i ).assign( vec4( x, 1 ) );
			const r = float( ParticleMesh.pcg( i.add( 12345 ) ).shiftRight( 8 ) ).mul( 1 / 16777216 );
			vel.element( i ).assign( vec4( s.mul( u.P0 ), r ) );

		} )().compute( NP ).setName( 'cwParticles' );

		this.stepList = [ this.kDeposit, this.kToComplex, this.fft.fx, this.fft.fy, this.fft.pz, this.fft.iy, this.fft.ix, this.kGradient, this.kKickDrift ];
		this.icList = [ this.kNoise, this.fft.fx, this.fft.fy, this.fft.fz, this.kShape, this.fft.iz, this.fft.iy, this.fft.ix, this.fft.dz, this.fft.dy, this.fft.dx, this.kParticles ];
		// density-only pass (deposit → grid) to refresh colours / the volume right after the ICs
		this.densityList = [ this.kDeposit, this.kToComplex, this.kGradient ];

	}

	// ───────────────────────── host side ─────────────────────────

	setSeeds( seeds, amp ) {

		const arr = this.u.seeds.array;
		for ( let i = 0; i < MAX_SEEDS; i ++ ) {

			const s = seeds[ i ];
			if ( s ) arr[ i ].set( s.x, s.y, s.z, amp * ( s.w ?? 1 ) );
			else arr[ i ].set( 0, 0, 0, 0 );

		}

	}

	dispose() {

		const nodes = [ this.kDeposit, this.kToComplex, this.kGradient, this.kKickDrift, this.kNoise, this.kShape, this.kParticles, ...Object.values( this.fft ) ];
		for ( const n of nodes ) n.dispose?.();
		for ( const b of [ this.grid, this.C, this.Dk, this.rho, this.F, this.pos, this.vel ] ) {

			b.value?.dispose?.();

		}

		this.volTex.dispose();

	}

}
