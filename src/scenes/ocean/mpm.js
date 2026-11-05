// MLS-MPM (Hu et al. 2018) weakly-compressible fluid on the GPU, in TSL compute shaders.
//
// Units: everything lives in *grid space* (node i sits at coordinate i, dx = 1) and *per-substep time* (dt = 1).
// Grid nodes carry mass + momentum as fixed-point i32 (WebGPU has no float atomics) so P2G can scatter with atomicAdd.
//
// Per substep:
//   clear   – zero the atomic grid
//   p2g1    – scatter APIC mass/momentum  m·w·(v + C·d)
//   p2g2    – gather density from grid mass, Tait EOS pressure + viscous stress, scatter  −V·4·σ·d·w
//   grid    – momentum → velocity, gravity, pointer stirring, spherical free-slip wall → float velocity grid
//   g2p     – gather velocity + affine C (APIC), advect, clamp into sphere, bioluminescence stimulus
import * as THREE from 'three/webgpu';
import {
	Fn, storage, instancedArray, instanceIndex, float, int, vec3, vec4, floor, If,
	atomicAdd, atomicLoad, atomicStore, max, min, pow, length, dot, clamp, exp, smoothstep, sqrt, mix, select,
} from 'three/tsl';

const FP = 1e6; // fixed-point scale for grid atomics
const INV_FP = 1 / FP;

// ─────────────────────────── CPU-side initial conditions ───────────────────────────

function mulberry( seed ) {

	return () => {

		seed |= 0; seed = seed + 0x6D2B79F5 | 0;
		let t = Math.imul( seed ^ seed >>> 15, 1 | seed );
		t = t + Math.imul( t ^ t >>> 7, 61 | t ) ^ t;
		return ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296;

	};

}

// Fill a region with a jittered lattice whose spacing is chosen (bisection) so that exactly ≥count points fit.
function latticeFill( count, inside, bmin, bmax, rand ) {

	const countAt = ( s ) => {

		let n = 0;
		for ( let z = bmin[ 2 ]; z <= bmax[ 2 ]; z += s )
			for ( let y = bmin[ 1 ]; y <= bmax[ 1 ]; y += s )
				for ( let x = bmin[ 0 ]; x <= bmax[ 0 ]; x += s ) if ( inside( x, y, z ) ) n ++;
		return n;

	};

	let lo = 0.15, hi = 3.0; // lo → many points, hi → few
	for ( let it = 0; it < 22; it ++ ) {

		const mid = 0.5 * ( lo + hi );
		if ( countAt( mid ) >= count ) lo = mid; else hi = mid;

	}

	const s = lo;
	const pts = [];
	const j = 0.18 * s;
	for ( let z = bmin[ 2 ]; z <= bmax[ 2 ]; z += s )
		for ( let y = bmin[ 1 ]; y <= bmax[ 1 ]; y += s )
			for ( let x = bmin[ 0 ]; x <= bmax[ 0 ]; x += s )
				if ( inside( x, y, z ) ) pts.push( x + ( rand() - 0.5 ) * j, y + ( rand() - 0.5 ) * j, z + ( rand() - 0.5 ) * j );

	// shuffle point order so any contiguous index range is spatially random (used for blob recycling)
	const n = pts.length / 3;
	const order = new Uint32Array( n );
	for ( let i = 0; i < n; i ++ ) order[ i ] = i;
	for ( let i = n - 1; i > 0; i -- ) {

		const k = Math.floor( rand() * ( i + 1 ) );
		const t = order[ i ]; order[ i ] = order[ k ]; order[ k ] = t;

	}

	return { s, pts, order, n };

}

// ─────────────────────────── simulation ───────────────────────────

export class MPMFluid {

	/**
	 * @param {object} o
	 * @param {number} o.count particles
	 * @param {number} o.grid  grid resolution per axis
	 * @param {object} o.U     shared uniforms (see ocean.js) – survive rebuilds
	 */
	constructor( renderer, { count, grid, U } ) {

		this.renderer = renderer;
		this.count = count;
		this.grid = grid;
		this.U = U;

		const G = grid;
		const cells = G * G * G;
		this.cells = cells;
		this.center = G / 2;
		this.radius = G / 2 - 3; // spherical container radius (grid units)
		const C = this.center, R = this.radius, Rp = R - 0.5;
		this.particleBound = Rp;

		// ── initial dam-break column inside the sphere ──
		const rand = mulberry( 1337 + count );
		const Rf = Rp - 0.6;
		const inside = ( x, y, z ) => {

			const dx = x - C, dy = y - C, dz = z - C;
			return dx * dx + dy * dy + dz * dz < Rf * Rf && dx < - 0.06 * R && dy < 0.5 * R;

		};

		const fill = latticeFill( count, inside, [ C - Rf, C - Rf, C - Rf ], [ C + Rf, C + Rf, C + Rf ], rand );
		this.spacing = fill.s;
		this.restDensity = 1 / ( fill.s * fill.s * fill.s ); // particles (mass 1) per cell at rest

		const init = new Float32Array( count * 4 );
		for ( let i = 0; i < count; i ++ ) {

			const k = fill.order[ i % fill.n ];
			init[ i * 4 ] = fill.pts[ k * 3 ];
			init[ i * 4 + 1 ] = fill.pts[ k * 3 + 1 ];
			init[ i * 4 + 2 ] = fill.pts[ k * 3 + 2 ];
			init[ i * 4 + 3 ] = 1;

		}

		// ── blob template: a lattice ball at rest density with blobCount particles ──
		this.blobCount = Math.max( 64, Math.round( count * 0.07 ) );
		const blobR = Math.cbrt( 3 * this.blobCount / ( 4 * Math.PI * this.restDensity ) ) * 1.04;
		this.blobRadius = blobR;
		const bl = latticeFill( this.blobCount, ( x, y, z ) => x * x + y * y + z * z < blobR * blobR, [ - blobR, - blobR, - blobR ], [ blobR, blobR, blobR ], rand );
		const blobArr = new Float32Array( this.blobCount * 4 );
		for ( let i = 0; i < this.blobCount; i ++ ) {

			const k = bl.order[ i % bl.n ];
			blobArr.set( [ bl.pts[ k * 3 ], bl.pts[ k * 3 + 1 ], bl.pts[ k * 3 + 2 ], 0 ], i * 4 );

		}

		// ── buffers ──
		this.attrs = [];
		const mk = ( arr, n, type ) => {

			const a = new THREE.StorageInstancedBufferAttribute( arr, 4 );
			this.attrs.push( a );
			return storage( a, type, n );

		};

		this.initPos = mk( init, count, 'vec4' ).toReadOnly();
		this.blobTpl = mk( blobArr, this.blobCount, 'vec4' ).toReadOnly();
		this.pos = mk( new Float32Array( init ), count, 'vec4' ); // xyz grid pos, w = density / rest
		this.vel = mk( new Float32Array( count * 4 ), count, 'vec4' ); // xyz vel (cells/step), w = glow
		this.c0 = mk( new Float32Array( count * 4 ), count, 'vec4' ); // affine C rows
		this.c1 = mk( new Float32Array( count * 4 ), count, 'vec4' );
		this.c2 = mk( new Float32Array( count * 4 ), count, 'vec4' );

		const gridAttr = new THREE.StorageBufferAttribute( new Int32Array( cells * 4 ), 1 );
		this.attrs.push( gridAttr );
		this.gridI = storage( gridAttr, 'int', cells * 4 ).toAtomic();
		this.gridV = instancedArray( cells, 'vec4' );
		this.attrs.push( this.gridV.value );

		this.uRest = this.U.rest; // set from here
		this.U.rest.value = this.restDensity;

		this._build();

	}

	_build() {

		const { count, cells, U } = this;
		const G = this.grid, GG = G * G;
		const C = this.center, R = this.radius, Rp = this.particleBound;
		const { pos, vel, c0, c1, c2, gridI, gridV } = this;
		const cvec = vec3( C, C, C );

		// quadratic B-spline stencil around p (grid space)
		const stencil = ( p ) => {

			const base = floor( p.sub( 0.5 ) ).toVar();
			const fx = p.sub( base ).toVar();
			const a = vec3( 1.5 ).sub( fx ), b = fx.sub( 1.0 ), c = fx.sub( 0.5 );
			const w = [ a.mul( a ).mul( 0.5 ).toVar(), vec3( 0.75 ).sub( b.mul( b ) ).toVar(), c.mul( c ).mul( 0.5 ).toVar() ];
			const bi = int( base.x ).add( int( base.y ).mul( int( G ) ) ).add( int( base.z ).mul( int( GG ) ) ).toVar();
			return { fx, w, bi };

		};

		const forEachNode = ( st, fn ) => {

			for ( let gz = 0; gz < 3; gz ++ ) for ( let gy = 0; gy < 3; gy ++ ) for ( let gx = 0; gx < 3; gx ++ ) {

				const weight = st.w[ gx ].x.mul( st.w[ gy ].y ).mul( st.w[ gz ].z );
				const d = vec3( gx, gy, gz ).sub( st.fx ); // node − particle
				const idx = st.bi.add( int( gx + gy * G + gz * GG ) );
				fn( weight, d, idx );

			}

		};

		const addFP = ( slot, v ) => atomicAdd( gridI.element( slot ), int( v.mul( FP ) ) );

		// ── clear ──
		this.kClear = Fn( () => {

			const i4 = int( instanceIndex ).mul( int( 4 ) );
			atomicStore( gridI.element( i4 ), int( 0 ) );
			atomicStore( gridI.element( i4.add( int( 1 ) ) ), int( 0 ) );
			atomicStore( gridI.element( i4.add( int( 2 ) ) ), int( 0 ) );
			atomicStore( gridI.element( i4.add( int( 3 ) ) ), int( 0 ) );

		} )().compute( cells, [ 64 ] ).setName( 'mpm clear' );

		// ── P2G 1: APIC mass + momentum ──
		this.kP2G1 = Fn( () => {

			const p = pos.element( instanceIndex ).xyz.toVar();
			const v = vel.element( instanceIndex ).xyz.toVar();
			const r0 = c0.element( instanceIndex ).xyz.toVar();
			const r1 = c1.element( instanceIndex ).xyz.toVar();
			const r2 = c2.element( instanceIndex ).xyz.toVar();
			const st = stencil( p );

			forEachNode( st, ( w, d, idx ) => {

				const q = vec3( dot( r0, d ), dot( r1, d ), dot( r2, d ) ).add( v ).mul( w );
				const i4 = idx.mul( int( 4 ) );
				addFP( i4, w );
				addFP( i4.add( int( 1 ) ), q.x );
				addFP( i4.add( int( 2 ) ), q.y );
				addFP( i4.add( int( 3 ) ), q.z );

			} );

		} )().compute( count, [ 64 ] ).setName( 'mpm p2g mass' );

		// ── P2G 2: density → pressure + viscosity → momentum update ──
		this.kP2G2 = Fn( () => {

			const p = pos.element( instanceIndex ).xyz.toVar();
			const r0 = c0.element( instanceIndex ).xyz.toVar();
			const r1 = c1.element( instanceIndex ).xyz.toVar();
			const r2 = c2.element( instanceIndex ).xyz.toVar();
			const st = stencil( p );

			const density = float( 0 ).toVar();
			forEachNode( st, ( w, d, idx ) => {

				density.addAssign( float( atomicLoad( gridI.element( idx.mul( int( 4 ) ) ) ) ).mul( INV_FP ).mul( w ) );

			} );
			density.assign( max( density, 1e-3 ) );
			pos.element( instanceIndex ).w.assign( density.div( U.rest ) );

			const volume = float( 1 ).div( density );
			// Tait-style EOS, no tension; clamp to keep fixed-point range safe
			const pressure = clamp( U.stiffK.mul( pow( density.div( U.rest ), 5.0 ).sub( 1.0 ) ), 0.0, U.pressureMax );
			// stress σ = −p I + μ (C + Cᵀ); term = −V · 4 · σ (dt = 1, dx = 1)
			const k = volume.mul( - 4.0 );
			const mu = U.mu;
			const t0 = vec3( r0.x.add( r0.x ).mul( mu ).sub( pressure ), r0.y.add( r1.x ).mul( mu ), r0.z.add( r2.x ).mul( mu ) ).mul( k ).toVar();
			const t1 = vec3( r1.x.add( r0.y ).mul( mu ), r1.y.add( r1.y ).mul( mu ).sub( pressure ), r1.z.add( r2.y ).mul( mu ) ).mul( k ).toVar();
			const t2 = vec3( r2.x.add( r0.z ).mul( mu ), r2.y.add( r1.z ).mul( mu ), r2.z.add( r2.z ).mul( mu ).sub( pressure ) ).mul( k ).toVar();

			forEachNode( st, ( w, d, idx ) => {

				const i4 = idx.mul( int( 4 ) );
				addFP( i4.add( int( 1 ) ), dot( t0, d ).mul( w ) );
				addFP( i4.add( int( 2 ) ), dot( t1, d ).mul( w ) );
				addFP( i4.add( int( 3 ) ), dot( t2, d ).mul( w ) );

			} );

		} )().compute( count, [ 64 ] ).setName( 'mpm p2g stress' );

		// ── grid update ──
		this.kGrid = Fn( () => {

			const i = int( instanceIndex );
			const i4 = i.mul( int( 4 ) );
			const m = float( atomicLoad( gridI.element( i4 ) ) ).mul( INV_FP ).toVar();
			const out = vec4( 0 ).toVar();

			If( m.greaterThan( 1e-5 ), () => {

				const mom = vec3(
					float( atomicLoad( gridI.element( i4.add( int( 1 ) ) ) ) ),
					float( atomicLoad( gridI.element( i4.add( int( 2 ) ) ) ) ),
					float( atomicLoad( gridI.element( i4.add( int( 3 ) ) ) ) ),
				);
				const v = mom.mul( INV_FP ).div( m ).toVar();
				v.addAssign( U.gravity );

				const x = vec3( float( i.mod( int( G ) ) ), float( i.div( int( G ) ).mod( int( G ) ) ), float( i.div( int( GG ) ) ) ).toVar();

				// pointer stirring: velocity-match nodes near the pointer ray (segment inside the sphere)
				If( U.ptrStrength.greaterThan( 0.0 ), () => {

					const rel = x.sub( U.ptrOrigin );
					const t = clamp( dot( rel, U.ptrDir ), U.ptrRange.x, U.ptrRange.y );
					const dd = rel.sub( U.ptrDir.mul( t ) );
					const f = exp( dot( dd, dd ).div( U.ptrRadius.mul( U.ptrRadius ) ).negate() ).mul( U.ptrStrength );
					v.assign( mix( v, U.ptrVel, clamp( f, 0, 0.85 ) ) );

				} );

				// spherical container: free-slip, slight friction
				const rel = x.sub( cvec );
				const dist = length( rel );
				If( dist.greaterThan( R - 0.5 ), () => {

					const n = rel.div( dist );
					const vn = dot( v, n );
					If( vn.greaterThan( 0 ), () => {

						v.subAssign( n.mul( vn ) );
						v.mulAssign( 0.985 );

					} );

				} );

				out.assign( vec4( v, m ) );

			} );

			gridV.element( instanceIndex ).assign( out );

		} )().compute( cells, [ 64 ] ).setName( 'mpm grid' );

		// ── G2P: APIC gather + advection ──
		this.kG2P = Fn( () => {

			const pe = pos.element( instanceIndex );
			const ve = vel.element( instanceIndex );
			const p = pe.xyz.toVar();
			const st = stencil( p );
			const v = vec3( 0 ).toVar();
			const b0 = vec3( 0 ).toVar(), b1 = vec3( 0 ).toVar(), b2 = vec3( 0 ).toVar();

			forEachNode( st, ( w, d, idx ) => {

				const gv = gridV.element( idx ).xyz.mul( w ).toVar();
				v.addAssign( gv );
				b0.addAssign( d.mul( gv.x ) );
				b1.addAssign( d.mul( gv.y ) );
				b2.addAssign( d.mul( gv.z ) );

			} );

			// APIC: C = B · D⁻¹, D⁻¹ = 4 for quadratic B-splines with dx = 1
			const C0 = b0.mul( 4 ).toVar(), C1 = b1.mul( 4 ).toVar(), C2 = b2.mul( 4 ).toVar();

			// CFL guard
			const sp = length( v );
			v.mulAssign( min( 1.0, float( U.maxSpeed ).div( max( sp, 1e-6 ) ) ) );

			p.addAssign( v );

			// sphere clamp
			const rel = p.sub( cvec );
			const dist = length( rel ).toVar();
			If( dist.greaterThan( Rp ), () => {

				const n = rel.div( dist );
				p.assign( cvec.add( n.mul( Rp ) ) );
				const vn = dot( v, n );
				If( vn.greaterThan( 0 ), () => {

					v.subAssign( n.mul( vn ) );

				} );

			} );
			p.assign( clamp( p, vec3( 1.6 ), vec3( G - 2.6 ) ) );

			// bioluminescence: dinoflagellate-like flash on shear stress, slow decay
			const sxx = C0.x, syy = C1.y, szz = C2.z;
			const sxy = C0.y.add( C1.x ), sxz = C0.z.add( C2.x ), syz = C1.z.add( C2.y );
			const shear = sqrt( sxx.mul( sxx ).add( syy.mul( syy ) ).add( szz.mul( szz ) ).mul( 4 ).add( sxy.mul( sxy ).add( sxz.mul( sxz ) ).add( syz.mul( syz ) ).mul( 2 ) ) );
			const speed = length( v );
			const rho = pe.w;
			const spray = smoothstep( 0.75, 0.25, rho ).mul( smoothstep( 0.05, 0.25, speed.div( U.maxSpeed ) ) );
			const stim = smoothstep( U.glowLo, U.glowHi, shear.mul( 0.7 ).add( speed.mul( 0.6 ) ) ).add( spray );
			const g = max( ve.w.mul( U.glowDecay ), clamp( stim, 0, 1.5 ) );

			pe.assign( vec4( p, rho ) );
			ve.assign( vec4( v, g ) );
			c0.element( instanceIndex ).assign( vec4( C0, 0 ) );
			c1.element( instanceIndex ).assign( vec4( C1, 0 ) );
			c2.element( instanceIndex ).assign( vec4( C2, 0 ) );

		} )().compute( count, [ 64 ] ).setName( 'mpm g2p' );

		// ── reset to dam break (optionally mirrored) ──
		this.kReset = Fn( () => {

			const ip = this.initPos.element( instanceIndex ).xyz;
			const mx = select( U.mirror.greaterThan( 0.5 ), float( 2 * C ).sub( ip.x ), ip.x );
			const mz = select( U.mirror.greaterThan( 0.5 ), float( 2 * C ).sub( ip.z ), ip.z );
			pos.element( instanceIndex ).assign( vec4( mx, ip.y, mz, 1 ) );
			vel.element( instanceIndex ).assign( vec4( 0 ) );
			c0.element( instanceIndex ).assign( vec4( 0 ) );
			c1.element( instanceIndex ).assign( vec4( 0 ) );
			c2.element( instanceIndex ).assign( vec4( 0 ) );

		} )().compute( count, [ 64 ] ).setName( 'mpm reset' );

		// ── recycle a contiguous (spatially random) slice of particles into a falling blob ──
		this.kBlob = Fn( () => {

			const j = int( instanceIndex ).add( U.blobStart ).mod( int( count ) );
			const o = this.blobTpl.element( instanceIndex ).xyz;
			pos.element( j ).assign( vec4( U.blobCenter.add( o ), 1 ) );
			vel.element( j ).assign( vec4( U.blobVel, 0.9 ) );
			c0.element( j ).assign( vec4( 0 ) );
			c1.element( j ).assign( vec4( 0 ) );
			c2.element( j ).assign( vec4( 0 ) );

		} )().compute( this.blobCount, [ 64 ] ).setName( 'mpm blob' );

		this.stepList = [ this.kClear, this.kP2G1, this.kP2G2, this.kGrid, this.kG2P ];
		this._blobCursor = 0;

	}

	step( n ) {

		for ( let i = 0; i < n; i ++ ) this.renderer.compute( this.stepList );

	}

	reset( mirror = false ) {

		this.U.mirror.value = mirror ? 1 : 0;
		this.renderer.compute( this.kReset );

	}

	/** center in grid units */
	dropBlob( center, velocity ) {

		const U = this.U;
		const lim = this.particleBound - this.blobRadius - 0.8;
		const c = center.clone().subScalar( this.center );
		if ( c.length() > lim ) c.setLength( lim );
		U.blobCenter.value.copy( c ).addScalar( this.center );
		U.blobVel.value.copy( velocity );
		U.blobStart.value = this._blobCursor;
		this._blobCursor = ( this._blobCursor + this.blobCount ) % this.count;
		this.renderer.compute( this.kBlob );

	}

	dispose() {

		for ( const k of [ this.kClear, this.kP2G1, this.kP2G2, this.kGrid, this.kG2P, this.kReset, this.kBlob ] ) k?.dispose?.();
		const attrs = this.renderer._attributes;
		for ( const a of this.attrs ) {

			try {

				attrs?.delete( a );

			} catch ( e ) { /* not yet uploaded */ }

		}

		this.attrs.length = 0;

	}

}
