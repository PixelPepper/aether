// 3D Eulerian plasma solver on rgba16float Storage3DTextures (velocity+temperature, density) with
// fp32 storage buffers for the pressure Poisson problem.
//
// Per step:
//   1. advect      semi-Lagrangian back-trace of (u,T) and ρ             VT_A,DN_A → VT_T,DN_T
//   2. maccormack  forward re-trace, error compensation, min/max limiter  → VT_B,DN_B
//   3. curl        ω = ∇×u                                                → CURL
//   4. forces      vorticity confinement, buoyancy (T) and weight (ρ), magnetic guiding (anisotropic drag across an
//                  analytic bipolar active-region field), granulation forcing in the photospheric source layer,
//                  footpoint heating, pointer splat, density-dependent radiative cooling, sponge   → VT_T,DN_A
//   5. divergence  ∇·u                                                    → div
//   6. jacobi ×N   ∇²p = ∇·u   (warm-started, Neumann floor, open top & sides)
//   7. project     u -= ∇p                                                → VT_A
import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, int, uint, float, vec2, vec3, vec4, ivec3, uniform, instancedArray, instanceIndex,
	texture3D, texture3DLoad, textureStore, select, max, min, clamp, exp, length, normalize, cross, dot,
	smoothstep, mix, floor, sin, inverseSqrt, atomicMax, atomicStore,
} from 'three/tsl';
import { worley, granuleShade } from './noise.js';

export const GRAN_SCALE = 0.62; // world units per granule (shared with the photosphere shader)
export const GRAN_SPEED = 0.22;

export function createFluid( renderer, cfg ) {

	const { NX, NY, NZ, jacobi, origin, size, starR } = cfg;
	const h = size.x / NX;
	const COUNT = NX * NY * NZ;
	const dims = vec3( NX, NY, NZ );
	const maxC = ivec3( NX - 1, NY - 1, NZ - 1 );

	const textures = [];
	const mk = ( name ) => {

		const t = new THREE.Storage3DTexture( NX, NY, NZ );
		t.name = name;
		t.type = THREE.HalfFloatType;
		t.format = THREE.RGBAFormat;
		t.minFilter = THREE.LinearFilter;
		t.magFilter = THREE.LinearFilter;
		t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
		t.generateMipmaps = false;
		textures.push( t );
		return t;

	};

	const VT_A = mk( 'VT_A' ), VT_B = mk( 'VT_B' ), VT_T = mk( 'VT_T' );
	const DN_A = mk( 'DN_A' ), DN_B = mk( 'DN_B' ), DN_T = mk( 'DN_T' );
	const CURL = mk( 'CURL' );

	const P0 = instancedArray( COUNT, 'float' );
	const P1 = instancedArray( COUNT, 'float' );
	const DIV = instancedArray( COUNT, 'float' );
	const PEAK = instancedArray( 1, 'uint' ).toAtomic();

	const U = {
		dt: uniform( 1 / 60 ),
		time: uniform( 0 ),
		buoy: uniform( 1.6 ),
		weight: uniform( 1.1 ),
		vort: uniform( 0.45 ),
		cool: uniform( 0.45 ),
		guide: uniform( 7.0 ),
		erupt: uniform( 0 ),
		heatL: uniform( 1.0 ),
		heatR: uniform( 0.45 ),
		q1: uniform( new THREE.Vector3( - 2.0, - 0.75, - 0.3 ) ),
		q2: uniform( new THREE.Vector3( 2.0, - 0.75, 0.3 ) ),
		ptrPos: uniform( new THREE.Vector3( 0, 2, 0 ) ),
		ptrVel: uniform( new THREE.Vector3() ),
		ptrAmt: uniform( 0 ),
		ptrRad: uniform( 0.45 ),
	};

	const originN = vec3( origin.x, origin.y, origin.z );

	// ── helpers (inlined into each kernel) ──
	const cellOf = () => {

		const i = int( instanceIndex );
		return ivec3( i.mod( int( NX ) ), i.div( int( NX ) ).mod( int( NY ) ), i.div( int( NX * NY ) ) );

	};

	const worldOf = ( c ) => vec3( c ).add( 0.5 ).mul( h ).add( originN );
	const ld = ( tex, c ) => texture3DLoad( tex, c );
	const ldo = ( tex, c, x, y, z ) => texture3DLoad( tex, clamp( c.add( ivec3( x, y, z ) ), ivec3( 0 ), maxC ) );
	const smp = ( tex, uvw ) => texture3D( tex, uvw );
	const surfaceY = ( p ) => dot( p.xz, p.xz ).div( - 2 * starR ); // paraxial sphere cap

	// bipolar active-region field: two sub-photospheric magnetic charges (a sunspot pair), plus a weak
	// overlying arcade component. Field lines leave the leading spot and re-enter the follower: loops.
	const bField = ( p ) => {

		const d1 = p.sub( U.q1 ), d2 = p.sub( U.q2 );
		const r1 = dot( d1, d1 ).add( 0.04 ), r2 = dot( d2, d2 ).add( 0.04 );
		return d1.mul( inverseSqrt( r1 ).div( r1 ) ).sub( d2.mul( inverseSqrt( r2 ).div( r2 ) ) );

	};

	// ── 0. init: photospheric layer + a pre-formed loop of plasma along the bipole arcade ──
	const initK = Fn( () => {

		const c = cellOf().toVar();
		const p = worldOf( c ).toVar();
		const yS = p.y.sub( surfaceY( p ) ).toVar();
		const layer = smoothstep( 0.5, 0.0, yS );
		// circle through both footpoints (2D line-charge arcade approximation), apex ≈ 3.1
		const fx = U.q1.x.abs().add( U.q2.x.abs() ).mul( 0.5 );
		const yc = float( 0.55 );
		const rr = fx.mul( fx ).add( yc.mul( yc ) ).sqrt().add( 0.15 );
		const dxy = length( vec2( p.x, p.y.sub( yc ) ) ).sub( rr );
		const zc = p.x.div( fx ).mul( 0.3 );
		const tube = smoothstep( 0.42, 0.0, length( vec2( dxy, p.z.sub( zc ) ) ) ).mul( smoothstep( - 0.1, 0.4, yS ) );
		const thread = sin( p.x.mul( 5.3 ).add( p.y.mul( 3.1 ) ).add( p.z.mul( 7.0 ) ) ).mul( 0.5 ).add( 0.5 );
		const rho = layer.mul( 0.9 ).add( tube.mul( mix( 0.6, 1.5, thread ) ) );
		const T = layer.mul( 1.0 ).add( tube.mul( mix( 2.0, 0.5, thread ) ) );
		// siphon flow along the loop, from the left (hot) footpoint to the right
		const tang = normalize( vec3( p.y.sub( yc ), p.x.negate(), 0.001 ) );
		const v = tang.mul( tube ).mul( 1.2 );
		textureStore( VT_A, c, vec4( v, T ) );
		textureStore( DN_A, c, vec4( rho, 0, 0, 1 ) );
		P0.element( instanceIndex ).assign( 0 );

	} )().compute( COUNT );

	// ── 1. semi-Lagrangian advection ──
	const advectK = Fn( () => {

		const c = cellOf().toVar();
		const u = ld( VT_A, c ).xyz;
		const uvw = vec3( c ).add( 0.5 ).sub( u.mul( U.dt ).div( h ) ).div( dims ).toVar();
		textureStore( VT_T, c, smp( VT_A, uvw ) );
		textureStore( DN_T, c, smp( DN_A, uvw ) );

	} )().compute( COUNT );

	// ── 2. MacCormack correction with Selle et al. min/max limiter ──
	const macK = Fn( () => {

		const c = cellOf().toVar();
		const vtN = ld( VT_A, c ).toVar();
		const dnN = ld( DN_A, c ).toVar();
		const off = vtN.xyz.mul( U.dt ).div( h ).toVar();
		const fw = vec3( c ).add( 0.5 ).add( off ).div( dims );
		const vtC = ld( VT_T, c ).add( vtN.sub( smp( VT_T, fw ) ).mul( 0.5 ) ).toVar();
		const dnC = ld( DN_T, c ).add( dnN.sub( smp( DN_T, fw ) ).mul( 0.5 ) ).toVar();

		const b0 = ivec3( floor( vec3( c ).sub( off ) ) ).toVar();
		const vMin = vec4( 1e4 ).toVar(), vMax = vec4( - 1e4 ).toVar();
		const dMin = vec4( 1e4 ).toVar(), dMax = vec4( - 1e4 ).toVar();
		for ( let k = 0; k < 8; k ++ ) {

			const cc = clamp( b0.add( ivec3( k & 1, ( k >> 1 ) & 1, ( k >> 2 ) & 1 ) ), ivec3( 0 ), maxC );
			const a = ld( VT_A, cc ), b = ld( DN_A, cc );
			vMin.assign( min( vMin, a ) ); vMax.assign( max( vMax, a ) );
			dMin.assign( min( dMin, b ) ); dMax.assign( max( dMax, b ) );

		}

		textureStore( VT_B, c, clamp( vtC, vMin, vMax ) );
		textureStore( DN_B, c, clamp( dnC, dMin, dMax ) );

	} )().compute( COUNT );

	// ── 3. vorticity ──
	const curlK = Fn( () => {

		const c = cellOf().toVar();
		const L = ldo( VT_B, c, - 1, 0, 0 ).xyz, R = ldo( VT_B, c, 1, 0, 0 ).xyz;
		const D = ldo( VT_B, c, 0, - 1, 0 ).xyz, Uu = ldo( VT_B, c, 0, 1, 0 ).xyz;
		const K = ldo( VT_B, c, 0, 0, - 1 ).xyz, F = ldo( VT_B, c, 0, 0, 1 ).xyz;
		const w = vec3(
			Uu.z.sub( D.z ).sub( F.y.sub( K.y ) ),
			F.x.sub( K.x ).sub( R.z.sub( L.z ) ),
			R.y.sub( L.y ).sub( Uu.x.sub( D.x ) )
		).mul( 0.5 / h ).toVar();
		textureStore( CURL, c, vec4( w, length( w ) ) );

	} )().compute( COUNT );

	// ── 4. forces & sources ──
	const forcesK = Fn( () => {

		const c = cellOf().toVar();
		const p = worldOf( c ).toVar();
		const vt = ld( VT_B, c ).toVar();
		const v = vt.xyz.toVar();
		const T = vt.w.toVar();
		const rho = ld( DN_B, c ).x.toVar();
		const dt = U.dt;

		// vorticity confinement  f = ε h (N × ω),  N = ∇|ω| / |∇|ω||
		const wC = ld( CURL, c ).xyz;
		const eta = vec3(
			ldo( CURL, c, 1, 0, 0 ).w.sub( ldo( CURL, c, - 1, 0, 0 ).w ),
			ldo( CURL, c, 0, 1, 0 ).w.sub( ldo( CURL, c, 0, - 1, 0 ).w ),
			ldo( CURL, c, 0, 0, 1 ).w.sub( ldo( CURL, c, 0, 0, - 1 ).w )
		);
		const Nv = eta.div( length( eta ).add( 1e-5 ) );
		const fConf = cross( Nv, wC ).mul( U.vort.mul( h * 2.0 ) );

		// thermal buoyancy vs. gravitational weight of the dense (cool) plasma
		const fB = vec3( 0, T.mul( U.buoy ).sub( rho.mul( U.weight ) ), 0 );
		v.addAssign( fConf.add( fB ).mul( dt ) );

		// magnetic guiding: frozen-in plasma slides freely along B, motion across B is strongly damped
		const yS = p.y.sub( surfaceY( p ) ).toVar();
		const B = bField( p ).toVar();
		const bm = length( B ).toVar();
		const bh = B.div( bm.add( 1e-6 ) ).toVar();
		const vpar = bh.mul( dot( v, bh ) );
		const open = U.erupt.mul( smoothstep( 1.4, 3.4, yS ) ); // loss of equilibrium above the arcade during eruptions
		const kk = U.guide.mul( bm.div( bm.add( 0.02 ) ) ).mul( float( 1 ).sub( open ) );
		v.assign( vpar.add( v.sub( vpar ).mul( exp( kk.mul( dt ).negate() ) ) ) );

		// photospheric source layer: granular convection forcing (hot upflows in cells, cold downdrafts in lanes)
		const sl = smoothstep( 0.42, 0.0, yS ).toVar();
		If( sl.greaterThan( 0.001 ), () => {

			const w = worley( p.xz.div( GRAN_SCALE ), U.time.mul( GRAN_SPEED ) );
			const gC = granuleShade( w );
			const outward = w.zw.negate().div( length( w.zw ).add( 1e-3 ) ).mul( smoothstep( 0.05, 0.55, w.x ) ).mul( 0.9 );
			const vTgt = vec3( outward.x, gC.sub( 0.42 ).mul( 1.8 ), outward.y );
			const relax = min( sl.mul( dt ).mul( 5.0 ), 1 );
			v.assign( mix( v, vTgt, relax ) );
			T.assign( mix( T, gC.mul( 0.45 ).add( 0.72 ), relax ) );
			rho.assign( mix( rho, float( 0.85 ), relax ) );

		} );

		// footpoint heating above both spots: chromospheric evaporation drives plasma up into the loops
		const bUp = bh.mul( select( bh.y.lessThan( 0 ), float( - 1 ), float( 1 ) ) );
		const fp1 = vec3( U.q1.x, 0.32, U.q1.z ), fp2 = vec3( U.q2.x, 0.32, U.q2.z );
		const d1 = p.sub( fp1 ), d2 = p.sub( fp2 );
		const pulse = sin( U.time.mul( 1.3 ) ).mul( 0.25 ).add( sin( U.time.mul( 3.7 ).add( 1.0 ) ).mul( 0.15 ) ).add( 1 );
		const g1 = exp( dot( d1, d1 ).mul( - 1 / ( 0.34 * 0.34 ) ) ).mul( U.heatL ).mul( pulse );
		const g2 = exp( dot( d2, d2 ).mul( - 1 / ( 0.30 * 0.30 ) ) ).mul( U.heatR ).mul( float( 2 ).sub( pulse ) );
		const gH = g1.add( g2 ).mul( U.erupt.mul( 2.5 ).add( 1 ) ).toVar();
		T.addAssign( gH.mul( dt ).mul( 5.0 ) );
		rho.addAssign( gH.mul( dt ).mul( 1.6 ) );
		v.addAssign( bUp.mul( gH ).mul( dt ).mul( 9.0 ) );

		// eruption kick: the destabilised loop top is pushed outward
		const dTop = length( vec2( p.x.mul( 0.6 ), yS.sub( 2.9 ) ) );
		v.y.addAssign( U.erupt.mul( smoothstep( 1.6, 0.0, dTop ) ).mul( dt ).mul( 6.0 ) );

		// pointer: heat + momentum splat (stirring the star)
		const dp = p.sub( U.ptrPos );
		const gp = exp( dot( dp, dp ).div( U.ptrRad.mul( U.ptrRad ) ).negate() ).mul( U.ptrAmt ).toVar();
		T.addAssign( gp.mul( dt ).mul( 14.0 ) );
		rho.addAssign( gp.mul( dt ).mul( 2.5 ) );
		v.assign( mix( v, U.ptrVel, min( gp.mul( dt ).mul( 10.0 ), 0.8 ) ) );

		// optically-thin radiative losses ∝ n²Λ(T): dense plasma cools faster → thermal condensation into
		// cool, heavy filament threads that rain back down the loop legs
		T.mulAssign( exp( U.cool.mul( rho.mul( 0.7 ).add( 0.3 ) ).mul( dt ).negate() ) );
		rho.mulAssign( exp( dt.mul( - 0.06 ) ) );

		// sponge layer at the open boundaries
		const uvw = vec3( c ).add( 0.5 ).div( dims );
		const sp = smoothstep( 0.0, 0.07, uvw.x ).mul( smoothstep( 1.0, 0.93, uvw.x ) )
			.mul( smoothstep( 0.0, 0.07, uvw.z ) ).mul( smoothstep( 1.0, 0.93, uvw.z ) )
			.mul( smoothstep( 1.0, 0.9, uvw.y ) );
		const damp = mix( exp( dt.mul( - 4.0 ) ), float( 1 ), sp );
		v.mulAssign( damp );
		T.assign( clamp( T.mul( damp ), 0, 6 ) );
		rho.assign( clamp( rho.mul( damp ), 0, 4 ) );

		// no inflow through the floor
		If( c.y.equal( int( 0 ) ), () => {

			v.y.assign( max( v.y, 0 ) );

		} );

		textureStore( VT_T, c, vec4( v, T ) );
		textureStore( DN_A, c, vec4( rho, 0, 0, 1 ) );

	} )().compute( COUNT );

	// ── 5. divergence ──
	const divK = Fn( () => {

		const c = cellOf().toVar();
		const C = ld( VT_T, c ).xyz;
		const R = ldo( VT_T, c, 1, 0, 0 ).x, L = ldo( VT_T, c, - 1, 0, 0 ).x;
		const Uu = ldo( VT_T, c, 0, 1, 0 ).y;
		const D = select( c.y.greaterThan( int( 0 ) ), ldo( VT_T, c, 0, - 1, 0 ).y, C.y.negate() );
		const F = ldo( VT_T, c, 0, 0, 1 ).z, K = ldo( VT_T, c, 0, 0, - 1 ).z;
		DIV.element( instanceIndex ).assign( R.sub( L ).add( Uu.sub( D ) ).add( F.sub( K ) ).mul( 0.5 / h ) );

	} )().compute( COUNT );

	// ── 6. Jacobi ──
	const SX = 1, SY = NX, SZ = NX * NY;
	const pNeighbours = ( src ) => {

		const i = int( instanceIndex ).toVar();
		const c = cellOf().toVar();
		const pC = src.element( i ).toVar();
		const at = ( off ) => src.element( clamp( i.add( int( off ) ), int( 0 ), int( COUNT - 1 ) ) );
		return {
			i, c, pC,
			L: select( c.x.greaterThan( int( 0 ) ), at( - SX ), float( 0 ) ),
			R: select( c.x.lessThan( int( NX - 1 ) ), at( SX ), float( 0 ) ),
			D: select( c.y.greaterThan( int( 0 ) ), at( - SY ), pC ), // solid floor: Neumann
			U: select( c.y.lessThan( int( NY - 1 ) ), at( SY ), float( 0 ) ), // open top: Dirichlet
			K: select( c.z.greaterThan( int( 0 ) ), at( - SZ ), float( 0 ) ),
			F: select( c.z.lessThan( int( NZ - 1 ) ), at( SZ ), float( 0 ) ),
		};

	};

	const jacobiK = ( src, dst ) => Fn( () => {

		const n = pNeighbours( src );
		const s = n.L.add( n.R ).add( n.D ).add( n.U ).add( n.K ).add( n.F );
		dst.element( n.i ).assign( s.sub( DIV.element( n.i ).mul( h * h ) ).div( 6 ) );

	} )().compute( COUNT );

	const jac01 = jacobiK( P0, P1 );
	const jac10 = jacobiK( P1, P0 );

	// ── 7. projection ──
	const projK = Fn( () => {

		const n = pNeighbours( P0 );
		const grad = vec3( n.R.sub( n.L ), n.U.sub( n.D ), n.F.sub( n.K ) ).mul( 0.5 / h );
		const vt = ld( VT_T, n.c ).toVar();
		textureStore( VT_A, n.c, vec4( vt.xyz.sub( grad ), vt.w ) );

	} )().compute( COUNT );

	// ── stats: peak temperature (column max → one atomic per column) ──
	const peakReset = Fn( () => {

		atomicStore( PEAK.element( 0 ), uint( 0 ) );

	} )().compute( 1 );

	const peakK = Fn( () => {

		const i = int( instanceIndex );
		const x = i.mod( int( NX ) ), z = i.div( int( NX ) );
		const m = float( 0 ).toVar();
		Loop( { start: int( 0 ), end: int( NY ), type: 'int' }, ( { i: y } ) => {

			m.assign( max( m, texture3DLoad( VT_A, ivec3( x, y, z ) ).w ) );

		} );
		atomicMax( PEAK.element( 0 ), uint( m.mul( 1000 ) ) );

	} )().compute( NX * NZ );

	// pass list for one step
	const stepList = [ advectK, macK, curlK, forcesK, divK ];
	for ( let k = 0; k < jacobi; k += 2 ) stepList.push( jac01, jac10 );
	stepList.push( projK );

	const allNodes = [ initK, advectK, macK, curlK, forcesK, divK, jac01, jac10, projK, peakReset, peakK ];

	let peakPending = false;

	return {
		U, h, COUNT, textures, VT: VT_A, DN: DN_A, jacobi,
		nodes: { peakReset },
		sync() {

			renderer.compute( peakReset );
			return renderer.getArrayBufferAsync( PEAK.value );

		},
		init() {

			renderer.compute( initK );

		},
		step( dt ) {

			U.dt.value = dt;
			renderer.compute( stepList );

		},
		// async readback of the peak normalised temperature (call ≤ 2×/s)
		samplePeak( cb ) {

			if ( peakPending ) return;
			peakPending = true;
			renderer.compute( [ peakReset, peakK ] );
			renderer.getArrayBufferAsync( PEAK.value ).then( ( buf ) => {

				cb( new Uint32Array( buf )[ 0 ] / 1000 );

			} ).catch( () => {} ).finally( () => {

				peakPending = false;

			} );

		},
		dispose() {

			for ( const n of allNodes ) n.dispose?.();
			for ( const t of textures ) t.dispose();
			const attrs = renderer._attributes;
			for ( const b of [ P0, P1, DIV, PEAK ] ) {

				try {

					attrs?.delete( b.value );

				} catch ( e ) { /* already gone */ }

			}

		},
	};

}
