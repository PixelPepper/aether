// Silk — GPU XPBD cloth (small-steps), graph-coloured Gauss–Seidel constraint sweeps,
// analytic SDF sculptures with friction, sheet-vs-sheet / self collision through a
// GPU spatial hash built with an atomic counting sort, curl (Fourier-mode) turbulent wind
// with normal-dependent aerodynamic pressure, long-range attachments and GPU picking.
import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, Return, Continue, instanceIndex, uniform, uniformArray, storage,
	float, int, uint, vec2, vec3, vec4, ivec3, select, min, max, sqrt, length, normalize, cross, dot,
	abs, floor, sin, cos, atan, atomicAdd, atomicLoad, atomicStore, atomicMin, floatBitsToUint, smoothstep,
} from 'three/tsl';

const TAU = Math.PI * 2;

// deterministic PRNG for wind modes / perturbations
function mulberry32( a ) {

	return () => {

		a |= 0; a = a + 0x6D2B79F5 | 0;
		let t = Math.imul( a ^ a >>> 15, 1 | a );
		t = t + Math.imul( t ^ t >>> 7, 61 | t ) ^ t;
		return ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296;

	};

}

export function createClothSim( renderer, opt ) {

	const { N, S, size, tier, colliders } = opt;
	const NN = N * N;
	const P = S * NN;
	const spacing = size / ( N - 1 );
	const A = 2; // anchors per sheet (hang mode)
	const K = S * A;

	// ── buffers ───────────────────────────────────────────────────────────────
	const attrs = [];
	const vec4Attr = ( name ) => {

		const a = new THREE.StorageBufferAttribute( P, 4 );
		a.name = 'silk_' + name;
		attrs.push( a );
		return a;

	};

	const posA = vec4Attr( 'pos' ); // xyz + inverse mass (0 = kinematic) — also the render 'position'
	const prevA = vec4Attr( 'prev' );
	const velA = vec4Attr( 'vel' );
	const nrmA = vec4Attr( 'nrm' ); // render 'normal'
	const tanA = vec4Attr( 'tan' ); // render 'tangent' (weft direction) for anisotropic GGX
	const windA = vec4Attr( 'wind' );
	const dxA = vec4Attr( 'dx' ); // Jacobi self-collision correction
	const hangA = vec4Attr( 'hang' ); // hang-mode rest pose, w = pin id
	const dropA = vec4Attr( 'drop' ); // drop-mode start pose

	const pos = storage( posA, 'vec4', P );
	const prev = storage( prevA, 'vec4', P );
	const vel = storage( velA, 'vec4', P );
	const nrm = storage( nrmA, 'vec4', P );
	const tan = storage( tanA, 'vec4', P );
	const wind = storage( windA, 'vec4', P );
	const dxb = storage( dxA, 'vec4', P );
	const hang = storage( hangA, 'vec4', P ).toReadOnly();
	const drop = storage( dropA, 'vec4', P ).toReadOnly();

	// spatial hash (counting sort)
	const T = tier === 0 ? 1 << 14 : tier === 1 ? 1 << 17 : 1 << 18;
	const BLK = 256;
	const NB = T / BLK;
	const uAttr = ( n, name ) => {

		const a = new THREE.StorageBufferAttribute( new Uint32Array( n ), 1 );
		a.name = 'silk_' + name;
		attrs.push( a );
		return a;

	};

	const cellCountA = uAttr( T, 'cellCount' );
	const cellStartA = uAttr( T, 'cellStart' );
	const blockSumA = uAttr( NB, 'blockSum' );
	const cellOfA = uAttr( P, 'cellOf' );
	const rankOfA = uAttr( P, 'rankOf' );
	const sortedA = uAttr( P, 'sorted' );
	const pickA = uAttr( 8, 'pick' );

	const cellCount = storage( cellCountA, 'uint', T ).toAtomic();
	const cellStart = storage( cellStartA, 'uint', T );
	const blockSum = storage( blockSumA, 'uint', NB );
	const cellOf = storage( cellOfA, 'uint', P );
	const rankOf = storage( rankOfA, 'uint', P );
	const sorted = storage( sortedA, 'uint', P );
	const pick = storage( pickA, 'uint', 8 ).toAtomic();

	// ── uniforms ──────────────────────────────────────────────────────────────
	const U = {
		dt: uniform( 1 / 960 ),
		invDt: uniform( 960 ),
		gravity: uniform( 9.81 ),
		aStretch: uniform( 0 ),
		aShear: uniform( 0 ),
		aBend: uniform( 0 ),
		time: uniform( 0 ),
		windDir: uniform( new THREE.Vector3( 0.55, 0, 0.83 ).normalize() ),
		windSpeed: uniform( 3.5 ),
		turbulence: uniform( 0.8 ),
		aeroN: uniform( 6.0 ),
		aeroT: uniform( 0.25 ),
		damping: uniform( 0.15 ),
		thick: uniform( spacing * 0.85 ), // self-collision distance
		invCell: uniform( 1 / ( spacing * 0.85 ) ),
		clothR: uniform( Math.max( 0.012, spacing * 0.45 ) ), // distance kept from sculptures
		muS: uniform( 0.9 ),
		muK: uniform( 0.55 ),
		selfFric: uniform( 0.25 ),
		pinsOn: uniform( 1 ),
		grabOn: uniform( 0 ),
		grabIdx: uniform( 0, 'uint' ),
		grabSheet: uniform( 0, 'uint' ),
		grabI: uniform( 0 ),
		grabJ: uniform( 0 ),
		grabTarget: uniform( new THREE.Vector3() ),
		rayO: uniform( new THREE.Vector3() ),
		rayD: uniform( new THREE.Vector3( 0, 0, - 1 ) ),
		lraSlack: uniform( 1.0 ),
	};

	const anchorPos = uniformArray( Array.from( { length: K }, () => new THREE.Vector4() ), 'vec4' );
	const anchorRest = uniformArray( Array.from( { length: K }, () => new THREE.Vector4() ), 'vec4' );
	const anchorIJ = uniformArray( Array.from( { length: K }, () => new THREE.Vector4() ), 'vec4' );

	const u32 = ( x ) => uint( x );

	// ── sheet layouts ─────────────────────────────────────────────────────────
	const rnd = mulberry32( 1337 );
	const anchorsJS = []; // { sheet, ia, ja, base: Vector3, rest: Vector3, phase }
	function buildPoses() {

		const hang = hangA.array;
		const drop = dropA.array;
		const pr = Math.max( 1, Math.round( N / 96 ) ); // pin cluster radius (particles)
		const inset = Math.max( 1, Math.round( N * 0.025 ) );

		anchorsJS.length = 0;

		for ( let s = 0; s < S; s ++ ) {

			const L = opt.hangSheets[ s ];
			const a0 = new THREE.Vector3( ...L.a ), a1 = new THREE.Vector3( ...L.b );
			const ia0 = inset, ia1 = N - 1 - inset;
			const eu = a1.clone().sub( a0 );
			const chord = eu.length();
			eu.normalize();
			const scaleU = chord / ( ( ia1 - ia0 ) * spacing ); // < 1 → gathered top edge → curtain folds
			// hanging direction: down, leaning slightly into `lean`
			const ev = new THREE.Vector3( ...( L.lean || [ 0, 0, 0 ] ) ).add( new THREE.Vector3( 0, - 1, 0 ) );
			ev.sub( eu.clone().multiplyScalar( ev.dot( eu ) ) ).normalize();
			const en = new THREE.Vector3().crossVectors( eu, ev ).normalize();
			const ph = rnd() * TAU;

			for ( let j = 0; j < N; j ++ ) {

				for ( let i = 0; i < N; i ++ ) {

					const k = ( s * NN + j * N + i ) * 4;
					const u = ( i - ia0 ) * spacing * scaleU;
					const v = j * spacing;
					const fu = i / ( N - 1 ), fv = j / ( N - 1 );
					// gentle sail-like bulge + folds so the first frames already read as fabric
					const bulge = ( L.bulge ?? 0.35 ) * Math.sin( Math.PI * fu ) * Math.sin( Math.PI * fv * 0.85 ) +
						0.05 * Math.sin( fu * 9 * Math.PI + ph ) * fv;
					const p = a0.clone().addScaledVector( eu, u ).addScaledVector( ev, v ).addScaledVector( en, bulge );
					hang[ k ] = p.x; hang[ k + 1 ] = p.y; hang[ k + 2 ] = p.z; hang[ k + 3 ] = 0;

				}

			}

			[ [ ia0, 0, a0 ], [ ia1, 0, a1 ] ].forEach( ( [ ia, ja, base ], ai ) => {

				const id = s * A + ai;
				for ( let dj = 0; dj <= pr; dj ++ ) for ( let di = - pr; di <= pr; di ++ ) {

					const i = ia + di, j = ja + dj;
					if ( i < 0 || i >= N || j < 0 || j >= N ) continue;
					hang[ ( s * NN + j * N + i ) * 4 + 3 ] = id + 1;

				}

				const k = ( s * NN + ja * N + ia ) * 4;
				const rest = new THREE.Vector3( hang[ k ], hang[ k + 1 ], hang[ k + 2 ] );
				anchorsJS.push( { sheet: s, ia, ja, base: base.clone(), rest, phase: [ rnd() * TAU, rnd() * TAU, rnd() * TAU ], amp: L.sway ?? 0.3 } );
				anchorRest.array[ id ].set( rest.x, rest.y, rest.z, 0 );
				anchorIJ.array[ id ].set( ia, ja, 1, 0 );
				anchorPos.array[ id ].set( base.x, base.y, base.z, 0 );

			} );

			// drop pose: flat sheet above a sculpture, rotated, slightly tilted
			const D = opt.dropSheets[ s ];
			const c = new THREE.Vector3( ...D.c );
			const ry = D.ry, tilt = D.tilt || 0;
			const du = new THREE.Vector3( Math.cos( ry ), 0, - Math.sin( ry ) );
			const dv0 = new THREE.Vector3( Math.sin( ry ), 0, Math.cos( ry ) );
			const dv = dv0.clone().multiplyScalar( Math.cos( tilt ) ).add( new THREE.Vector3( 0, Math.sin( tilt ), 0 ) );
			for ( let j = 0; j < N; j ++ ) {

				for ( let i = 0; i < N; i ++ ) {

					const k = ( s * NN + j * N + i ) * 4;
					const u = ( i - ( N - 1 ) / 2 ) * spacing, v = ( j - ( N - 1 ) / 2 ) * spacing;
					const wob = 0.04 * Math.sin( i * 0.37 + s ) * Math.cos( j * 0.29 - s );
					const p = c.clone().addScaledVector( du, u ).addScaledVector( dv, v );
					drop[ k ] = p.x; drop[ k + 1 ] = p.y + wob; drop[ k + 2 ] = p.z; drop[ k + 3 ] = 0;

				}

			}

		}

	}

	buildPoses();

	// ── helpers (TSL) ─────────────────────────────────────────────────────────
	const hashCell = ( c ) => uint( c.x ).mul( u32( 73856093 ) )
		.bitXor( uint( c.y ).mul( u32( 19349663 ) ) )
		.bitXor( uint( c.z ).mul( u32( 83492791 ) ) )
		.bitAnd( u32( T - 1 ) );

	// ── reset kernels ─────────────────────────────────────────────────────────
	const makeReset = ( src ) => Fn( () => {

		const i = instanceIndex;
		const p = src.element( i );
		const w = select( p.w.greaterThan( 0.5 ), float( 0 ), float( 1 ) );
		pos.element( i ).assign( vec4( p.xyz, w ) );
		prev.element( i ).assign( vec4( p.xyz, w ) );
		vel.element( i ).assign( vec4( 0 ) );
		dxb.element( i ).assign( vec4( 0 ) );

	} )().compute( P ).setName( 'silk reset' );

	const kResetHang = makeReset( hang );
	const kResetDrop = makeReset( drop );

	// ── wind: divergence-free synthetic turbulence (sum of curl Fourier modes) ──
	const M = tier === 0 ? 5 : 9;
	const modes = [];
	{

		const r = mulberry32( 99 );
		for ( let m = 0; m < M; m ++ ) {

			const kmag = 0.35 * Math.pow( 6.0, m / ( M - 1 ) ); // 0.35 … 2.1 rad/m
			const th = r() * TAU, ph = Math.acos( 2 * r() - 1 );
			const kd = new THREE.Vector3( Math.sin( ph ) * Math.cos( th ), Math.cos( ph ) * 0.6, Math.sin( ph ) * Math.sin( th ) ).normalize();
			let e = new THREE.Vector3( r() - 0.5, r() - 0.5, r() - 0.5 ).normalize();
			e.sub( kd.clone().multiplyScalar( e.dot( kd ) ) ).normalize();
			const dir = new THREE.Vector3().crossVectors( kd, e ).normalize(); // velocity direction of curl( e sin(k·x) )
			const amp = Math.pow( kmag / 0.35, - 1 / 3 ) * 0.62;
			modes.push( { k: kd.multiplyScalar( kmag ), dir, amp, w: 0.25 + r() * 0.9, phi: r() * TAU } );

		}

	}

	const kWind = Fn( () => {

		const i = instanceIndex;
		const x = pos.element( i ).xyz;
		const t = U.time;
		const adv = x.sub( U.windDir.mul( U.windSpeed.mul( t ).mul( 0.85 ) ) ).toVar(); // frozen turbulence advected with the mean flow
		const turb = vec3( 0 ).toVar();
		for ( const m of modes ) {

			const arg = dot( adv, vec3( m.k.x, m.k.y, m.k.z ) ).add( t.mul( m.w ) ).add( m.phi );
			turb.addAssign( vec3( m.dir.x, m.dir.y, m.dir.z ).mul( cos( arg ).mul( m.amp ) ) );

		}

		// travelling gust fronts + boundary-layer height profile
		const along = dot( x, U.windDir );
		const gust = sin( along.mul( 0.55 ).sub( t.mul( 1.7 ) ) ).mul( 0.3 ).add( sin( along.mul( 0.23 ).sub( t.mul( 0.61 ) ).add( x.y.mul( 0.4 ) ) ).mul( 0.25 ) ).add( 1 );
		const prof = smoothstep( - 0.2, 4.5, x.y ).mul( 0.5 ).add( 0.5 );
		const w = U.windDir.mul( gust ).add( turb.mul( U.turbulence ) ).mul( U.windSpeed.mul( prof ) );
		wind.element( i ).assign( vec4( w, 0 ) );

	} )().compute( P ).setName( 'silk wind' );

	// ── integrate (predict) ───────────────────────────────────────────────────
	const kIntegrate = Fn( () => {

		const i = instanceIndex;
		const p = pos.element( i ).toVar();
		prev.element( i ).assign( p );
		const rest = hang.element( i );
		const pinId = rest.w;

		If( U.pinsOn.greaterThan( 0.5 ).and( pinId.greaterThan( 0.5 ) ), () => {

			const k = uint( pinId.add( 0.5 ) ).sub( u32( 1 ) ); // round(pinId) - 1
			const target = anchorPos.element( k ).xyz.add( rest.xyz ).sub( anchorRest.element( k ).xyz );
			pos.element( i ).assign( vec4( target, 0 ) );

		} ).ElseIf( U.grabOn.greaterThan( 0.5 ).and( i.equal( U.grabIdx ) ), () => {

			pos.element( i ).assign( vec4( U.grabTarget, 0 ) );

		} ).Else( () => {

			const v = vel.element( i ).xyz.toVar();
			v.assign( v.sub( vec3( 0, U.gravity.mul( U.dt ), 0 ) ) );

			// aerodynamic pressure on the local surface element (normal-dependent drag + lift),
			// integrated implicitly so strong gusts can never overshoot the relative velocity
			const n = nrm.element( i ).xyz;
			const vr = wind.element( i ).xyz.sub( v ).toVar();
			const vn = dot( vr, n ).toVar();
			const kn = U.aeroN.mul( abs( vn ) ).mul( U.dt );
			v.addAssign( n.mul( vn.mul( kn.div( kn.add( 1 ) ) ) ) );
			const vt = vr.sub( n.mul( vn ) );
			const kt = U.aeroT.mul( U.dt );
			v.addAssign( vt.mul( kt.div( kt.add( 1 ) ) ) );

			v.mulAssign( float( 1 ).sub( U.damping.mul( U.dt ) ) );
			pos.element( i ).assign( vec4( p.xyz.add( v.mul( U.dt ) ), 1 ) );

		} );

	} )().compute( P ).setName( 'silk integrate' );

	// ── graph-coloured distance constraints (stretch / shear / bend) ─────────
	// Each colour is a set of constraints sharing no particle, so a dispatch is race-free and
	// successive dispatches form a true Gauss–Seidel sweep.
	const colorKernels = [];
	let constraintCount = 0;
	const addColor = ( type, r, alpha, rest ) => {

		const bend = type === 'h2' || type === 'v2';
		const step = bend ? 2 : 1;
		const alongJS = ( k ) => bend ? 4 * ( k >> 1 ) + 2 * r + ( k & 1 ) : 2 * k + r;
		let Kc = 0;
		while ( alongJS( Kc ) + step <= N - 1 ) Kc ++;
		const C = ( type === 'd1' || type === 'd2' ) ? N - 1 : N;
		const count = S * C * Kc;
		constraintCount += count;

		const kernel = Fn( () => {

			const t = instanceIndex;
			const per = u32( C * Kc );
			const s = t.div( per );
			const rem = t.mod( per );
			const c = rem.div( u32( Kc ) );
			const k = rem.mod( u32( Kc ) );
			const a = bend ? k.shiftRight( u32( 1 ) ).mul( u32( 4 ) ).add( u32( 2 * r ) ).add( k.bitAnd( u32( 1 ) ) ) : k.mul( u32( 2 ) ).add( u32( r ) );
			const base = s.mul( u32( NN ) );
			let iA, iB;
			if ( type === 'h1' || type === 'h2' ) {

				iA = base.add( c.mul( u32( N ) ) ).add( a ); iB = iA.add( u32( step ) );

			} else if ( type === 'v1' || type === 'v2' ) {

				iA = base.add( a.mul( u32( N ) ) ).add( c ); iB = iA.add( u32( step * N ) );

			} else if ( type === 'd1' ) {

				iA = base.add( c.mul( u32( N ) ) ).add( a ); iB = iA.add( u32( N + 1 ) );

			} else {

				iA = base.add( c.add( u32( 1 ) ).mul( u32( N ) ) ).add( a ); iB = base.add( c.mul( u32( N ) ) ).add( a ).add( u32( 1 ) );

			}

			const ia = iA.toVar(), ib = iB.toVar();
			const pA = pos.element( ia ).toVar();
			const pB = pos.element( ib ).toVar();
			const W = pA.w.add( pB.w );
			If( W.greaterThan( 0 ), () => {

				const d = pA.xyz.sub( pB.xyz ).toVar();
				const L = length( d ).max( 1e-9 ).toVar();
				const dl = L.sub( rest ).negate().div( W.add( alpha ) );
				const corr = d.mul( dl.div( L ) ).toVar();
				pos.element( ia ).assign( vec4( pA.xyz.add( corr.mul( pA.w ) ), pA.w ) );
				pos.element( ib ).assign( vec4( pB.xyz.sub( corr.mul( pB.w ) ), pB.w ) );

			} );

		} )().compute( count ).setName( `silk ${type}${r}` );
		colorKernels.push( kernel );

	};

	addColor( 'h1', 0, U.aStretch, spacing ); addColor( 'h1', 1, U.aStretch, spacing );
	addColor( 'v1', 0, U.aStretch, spacing ); addColor( 'v1', 1, U.aStretch, spacing );
	addColor( 'd1', 0, U.aShear, spacing * Math.SQRT2 ); addColor( 'd1', 1, U.aShear, spacing * Math.SQRT2 );
	addColor( 'd2', 0, U.aShear, spacing * Math.SQRT2 ); addColor( 'd2', 1, U.aShear, spacing * Math.SQRT2 );
	addColor( 'h2', 0, U.aBend, spacing * 2 ); addColor( 'h2', 1, U.aBend, spacing * 2 );
	addColor( 'v2', 0, U.aBend, spacing * 2 ); addColor( 'v2', 1, U.aBend, spacing * 2 );

	// ── spatial hash: clear → count (atomicAdd gives rank) → 3-pass scan → scatter ──
	const kHashClear = Fn( () => {

		atomicStore( cellCount.element( instanceIndex ), u32( 0 ) );

	} )().compute( T ).setName( 'silk hash clear' );

	const kHashCount = Fn( () => {

		const i = instanceIndex;
		const c = ivec3( floor( pos.element( i ).xyz.mul( U.invCell ) ) );
		const h = hashCell( c ).toVar();
		cellOf.element( i ).assign( h );
		rankOf.element( i ).assign( atomicAdd( cellCount.element( h ), u32( 1 ) ) );

	} )().compute( P ).setName( 'silk hash count' );

	const kScanA = Fn( () => {

		const b = instanceIndex;
		const sum = uint( 0 ).toVar();
		Loop( { start: u32( 0 ), end: u32( BLK ), type: 'uint', condition: '<' }, ( { i } ) => {

			sum.addAssign( atomicLoad( cellCount.element( b.mul( u32( BLK ) ).add( i ) ) ) );

		} );
		blockSum.element( b ).assign( sum );

	} )().compute( NB ).setName( 'silk scan a' );

	const kScanB = Fn( () => {

		const run = uint( 0 ).toVar();
		Loop( { start: u32( 0 ), end: u32( NB ), type: 'uint', condition: '<' }, ( { i } ) => {

			const v = blockSum.element( i ).toVar();
			blockSum.element( i ).assign( run );
			run.addAssign( v );

		} );

	} )().compute( 1, [ 1 ] ).setName( 'silk scan b' );

	const kScanC = Fn( () => {

		const b = instanceIndex;
		const run = blockSum.element( b ).toVar();
		Loop( { start: u32( 0 ), end: u32( BLK ), type: 'uint', condition: '<' }, ( { i } ) => {

			const idx = b.mul( u32( BLK ) ).add( i ).toVar();
			const c = atomicLoad( cellCount.element( idx ) );
			cellStart.element( idx ).assign( run );
			run.addAssign( c );

		} );

	} )().compute( NB ).setName( 'silk scan c' );

	const kScatter = Fn( () => {

		const i = instanceIndex;
		sorted.element( cellStart.element( cellOf.element( i ) ).add( rankOf.element( i ) ) ).assign( i );

	} )().compute( P ).setName( 'silk hash scatter' );

	// ── self / sheet-vs-sheet collision (Jacobi, written to dx) ───────────────
	const MAXC = 24;
	const kSelf = Fn( () => {

		const i = instanceIndex;
		const pi = pos.element( i ).toVar();
		If( pi.w.equal( 0 ), () => {

			dxb.element( i ).assign( vec4( 0 ) );
			Return();

		} );

		const si = i.div( u32( NN ) );
		const ri = i.mod( u32( NN ) );
		const ii = int( ri.mod( u32( N ) ) ).toVar();
		const ji = int( ri.div( u32( N ) ) ).toVar();
		const xi = pi.xyz.toVar();
		const vi = xi.sub( prev.element( i ).xyz ).toVar();
		const c = ivec3( floor( xi.mul( U.invCell ) ) ).toVar();
		const acc = vec3( 0 ).toVar();
		const cnt = float( 0 ).toVar();
		const th = U.thick;
		const th2 = th.mul( th );

		Loop( { start: int( 0 ), end: int( 27 ), type: 'int', condition: '<', name: 'o' }, ( { o } ) => {

			const off = ivec3( o.mod( int( 3 ) ).sub( int( 1 ) ), o.div( int( 3 ) ).mod( int( 3 ) ).sub( int( 1 ) ), o.div( int( 9 ) ).sub( int( 1 ) ) );
			const h = hashCell( c.add( off ) ).toVar();
			const start = cellStart.element( h ).toVar();
			const n = min( atomicLoad( cellCount.element( h ) ), u32( MAXC ) ).toVar();

			Loop( { start: u32( 0 ), end: n, type: 'uint', condition: '<', name: 'q' }, ( { q } ) => {

				const j = sorted.element( start.add( q ) ).toVar();
				const sj = j.div( u32( NN ) );
				const rj = j.mod( u32( NN ) );
				const di = abs( int( rj.mod( u32( N ) ) ).sub( ii ) );
				const dj = abs( int( rj.div( u32( N ) ) ).sub( ji ) );
				const near = sj.equal( si ).and( di.lessThanEqual( int( 2 ) ) ).and( dj.lessThanEqual( int( 2 ) ) );
				If( near, () => {

					Continue();

				} );

				const pj = pos.element( j );
				const d = xi.sub( pj.xyz ).toVar();
				const d2 = dot( d, d ).toVar();
				If( d2.lessThan( th2 ).and( d2.greaterThan( 1e-14 ) ), () => {

					const dist = sqrt( d2 );
					const nn = d.div( dist ).toVar();
					// split the overlap between both particles (a kinematic partner takes none of it)
					const share = select( pj.w.equal( 0 ), float( 1 ), float( 0.5 ) );
					const corr = nn.mul( th.sub( dist ).mul( share ) );
					const vj = pj.xyz.sub( prev.element( j ).xyz );
					const rel = vi.sub( vj );
					const relT = rel.sub( nn.mul( dot( rel, nn ) ) );
					acc.addAssign( corr.sub( relT.mul( U.selfFric.mul( share ) ) ) );
					cnt.addAssign( 1 );

				} );

			} );

		} );

		const scale = min( float( 1 ), float( 1.5 ).div( cnt.max( 1 ) ) );
		dxb.element( i ).assign( vec4( acc.mul( scale ), cnt ) );

	} )().compute( P ).setName( 'silk self collide' );

	// ── colliders (analytic SDFs) ─────────────────────────────────────────────
	const contact = ( x, xp, n, pen ) => {

		x.addAssign( n.mul( pen ) );
		const disp = x.sub( xp );
		const dt = disp.sub( n.mul( dot( disp, n ) ) ).toVar();
		const lt = length( dt );
		const f = select( lt.lessThan( pen.mul( U.muS ) ), float( 1 ), min( pen.mul( U.muK ).div( lt.max( 1e-9 ) ), float( 1 ) ) );
		x.subAssign( dt.mul( f ) );

	};

	const { sphere, capsule, knot, plinth, cyc } = colliders;
	const knotM = new THREE.Matrix3().setFromMatrix4( new THREE.Matrix4().makeRotationFromQuaternion( knot.quat ) );
	const knotMi = knotM.clone().transpose();
	const knotRot = uniform( knotM );
	const knotRotInv = uniform( knotMi );
	const knotC = vec3( ...knot.pos.toArray() );

	// closest point on a (p,q) torus-knot centre curve: azimuth-seeded Gauss–Newton on |q - C(u)|²
	const KP = knot.p, KQ = knot.q, KR = knot.R;
	const kq = KQ / KP;
	const curve = ( u ) => {

		const cu = cos( u ), su = sin( u ), au = u.mul( kq );
		const ca = cos( au ), sa = sin( au );
		const Aq = ca.add( 2 );
		const C = vec3( Aq.mul( cu ), Aq.mul( su ), sa ).mul( KR * 0.5 );
		const dA = sa.mul( - kq );
		const Cp = vec3( dA.mul( cu ).sub( Aq.mul( su ) ), dA.mul( su ).add( Aq.mul( cu ) ), ca.mul( kq ) ).mul( KR * 0.5 );
		return { C, Cp };

	};

	const collideKnot = ( x, xp, r ) => {

		const q = knotRotInv.mul( x.sub( knotC ) ).toVar();
		If( length( q ).lessThan( KR * 1.5 + knot.tube + 0.15 ), () => {

			const phi = atan( q.y, q.x ).toVar();
			const best = vec4( 0, 0, 0, 1e9 ).toVar();
			for ( let c = 0; c < KP; c ++ ) {

				const u = phi.add( c * TAU ).toVar();
				for ( let it = 0; it < 3; it ++ ) {

					const { C, Cp } = curve( u );
					const cp = Cp.toVar();
					u.subAssign( dot( C.sub( q ), cp ).div( dot( cp, cp ) ) );

				}

				const Cf = curve( u ).C.toVar();
				const dd = q.sub( Cf );
				const d2 = dot( dd, dd );
				If( d2.lessThan( best.w ), () => {

					best.assign( vec4( Cf, d2 ) );

				} );

			}

			const dist = sqrt( best.w ).toVar();
			const sd = dist.sub( knot.tube );
			If( sd.lessThan( r ).and( dist.greaterThan( 1e-6 ) ), () => {

				const nl = q.sub( best.xyz ).div( dist );
				const nw = knotRot.mul( nl ).toVar();
				contact( x, xp, nw, r.sub( sd ) );

			} );

		} );

	};

	const sdPlinth = ( p ) => {

		const dxz = length( p.xz.sub( vec2( plinth.pos[ 0 ], plinth.pos[ 2 ] ) ) ).sub( plinth.r - plinth.round );
		const dy = abs( p.y.sub( plinth.h * 0.5 ) ).sub( plinth.h * 0.5 - plinth.round );
		const ox = max( dxz, 0 ), oy = max( dy, 0 );
		return min( max( dxz, dy ), 0 ).add( sqrt( ox.mul( ox ).add( oy.mul( oy ) ) ) ).sub( plinth.round );

	};

	const kFinalize = Fn( () => {

		const i = instanceIndex;
		const p = pos.element( i ).toVar();
		const x = p.xyz.toVar();
		const xp = prev.element( i ).xyz.toVar();
		const r = U.clothR;

		If( p.w.greaterThan( 0 ), () => {

			x.addAssign( dxb.element( i ).xyz );

			const s = i.div( u32( NN ) );
			const ri = i.mod( u32( NN ) );
			const fi = float( ri.mod( u32( N ) ) ), fj = float( ri.div( u32( N ) ) );

			// long-range attachments (Kim et al. 2012): never farther from a pin than the geodesic rest distance
			If( U.pinsOn.greaterThan( 0.5 ), () => {

				for ( let a = 0; a < A; a ++ ) {

					const k = s.mul( u32( A ) ).add( u32( a ) );
					const ij = anchorIJ.element( k );
					const L = length( vec3( fi.sub( ij.x ), fj.sub( ij.y ), 0 ) ).mul( spacing ).mul( U.lraSlack );
					const ap = anchorPos.element( k ).xyz;
					const d = x.sub( ap ).toVar();
					const dl = length( d ).toVar();
					If( dl.greaterThan( L ), () => {

						x.assign( ap.add( d.mul( L.div( dl ) ) ) );

					} );

				}

			} );

			If( U.grabOn.greaterThan( 0.5 ).and( s.equal( U.grabSheet ) ), () => {

				const L = length( vec3( fi.sub( U.grabI ), fj.sub( U.grabJ ), 0 ) ).mul( spacing * 1.02 );
				const d = x.sub( U.grabTarget ).toVar();
				const dl = length( d ).toVar();
				If( dl.greaterThan( L ), () => {

					x.assign( U.grabTarget.add( d.mul( L.div( dl ) ) ) );

				} );

			} );

			// sculptures
			{

				const q = x.sub( vec3( ...sphere.pos ) ).toVar();
				const L = length( q ).toVar();
				const sd = L.sub( sphere.r );
				If( sd.lessThan( r ), () => {

					contact( x, xp, q.div( L.max( 1e-6 ) ), r.sub( sd ) );

				} );

			}

			{

				const a = vec3( ...capsule.a ), b = vec3( ...capsule.b );
				const ba = b.sub( a );
				const h = dot( x.sub( a ), ba ).div( dot( ba, ba ) ).clamp( 0, 1 );
				const q = x.sub( a.add( ba.mul( h ) ) ).toVar();
				const L = length( q ).toVar();
				const sd = L.sub( capsule.r );
				If( sd.lessThan( r ), () => {

					contact( x, xp, q.div( L.max( 1e-6 ) ), r.sub( sd ) );

				} );

			}

			{

				const sd = sdPlinth( x ).toVar();
				If( sd.lessThan( r ), () => {

					const e = 0.002;
					const g = vec3(
						sdPlinth( x.add( vec3( e, 0, 0 ) ) ).sub( sdPlinth( x.sub( vec3( e, 0, 0 ) ) ) ),
						sdPlinth( x.add( vec3( 0, e, 0 ) ) ).sub( sdPlinth( x.sub( vec3( 0, e, 0 ) ) ) ),
						sdPlinth( x.add( vec3( 0, 0, e ) ) ).sub( sdPlinth( x.sub( vec3( 0, 0, e ) ) ) ) );
					contact( x, xp, normalize( g ), r.sub( sd ) );

				} );

			}

			collideKnot( x, xp, r );

			// cyclorama: floor + rounded cove + back wall
			{

				const zc = cyc.wallZ + cyc.cove;
				If( x.z.lessThan( zc ).and( x.y.lessThan( cyc.cove ) ), () => {

					const q = vec3( 0, x.y.sub( cyc.cove ), x.z.sub( zc ) ).toVar();
					const L = length( q ).toVar();
					const sd = float( cyc.cove ).sub( L );
					If( sd.lessThan( r ), () => {

						contact( x, xp, q.div( L.max( 1e-6 ) ).negate(), r.sub( sd ) );

					} );

				} ).Else( () => {

					If( x.y.lessThan( r ), () => {

						contact( x, xp, vec3( 0, 1, 0 ), r.sub( x.y ) );

					} );
					If( x.z.sub( cyc.wallZ ).lessThan( r ), () => {

						contact( x, xp, vec3( 0, 0, 1 ), r.sub( x.z.sub( cyc.wallZ ) ) );

					} );

				} );

			}

		} );

		dxb.element( i ).assign( vec4( 0 ) );
		const v = x.sub( xp ).mul( U.invDt ).toVar();
		const sp = length( v );
		v.mulAssign( min( float( 1 ), float( 30 ).div( sp.max( 1e-6 ) ) ) );
		vel.element( i ).assign( vec4( v, 0 ) );
		pos.element( i ).assign( vec4( x, p.w ) );

	} )().compute( P ).setName( 'silk finalize' );

	// ── normals + weft tangents for rendering / aerodynamics ──────────────────
	const kNormals = Fn( () => {

		const i = instanceIndex;
		const ri = i.mod( u32( NN ) );
		const ii = ri.mod( u32( N ) ), jj = ri.div( u32( N ) );
		const iL = select( ii.greaterThan( u32( 0 ) ), i.sub( u32( 1 ) ), i );
		const iR = select( ii.lessThan( u32( N - 1 ) ), i.add( u32( 1 ) ), i );
		const iD = select( jj.greaterThan( u32( 0 ) ), i.sub( u32( N ) ), i );
		const iU = select( jj.lessThan( u32( N - 1 ) ), i.add( u32( N ) ), i );
		const du = pos.element( iR ).xyz.sub( pos.element( iL ).xyz ).toVar();
		const dv = pos.element( iU ).xyz.sub( pos.element( iD ).xyz ).toVar();
		const n = normalize( cross( du, dv ) ).toVar();
		const tg = normalize( du.sub( n.mul( dot( n, du ) ) ) );
		nrm.element( i ).assign( vec4( n, 0 ) );
		tan.element( i ).assign( vec4( tg, 1 ) );

	} )().compute( P ).setName( 'silk normals' );

	// ── GPU picking: nearest particle to the pointer ray (angular bin, then depth) ──
	const MAXANG = 0.06;
	const pickKey = ( x ) => {

		const d = x.sub( U.rayO );
		const t = dot( d, U.rayD );
		const perp = length( d.sub( U.rayD.mul( t ) ) );
		const ang = perp.div( t.max( 1e-3 ) );
		const valid = t.greaterThan( 0.05 ).and( ang.lessThan( MAXANG ) );
		const key = uint( ang.mul( 1023 / MAXANG ) ).shiftLeft( u32( 22 ) ).bitOr( uint( t.div( 80 ).clamp( 0, 1 ).mul( 4194303 ) ) );
		return select( valid, key, u32( 0xffffffff ) );

	};

	const kPickClear = Fn( () => {

		atomicStore( pick.element( u32( 0 ) ), u32( 0xffffffff ) );
		atomicStore( pick.element( u32( 1 ) ), u32( 0xffffffff ) );

	} )().compute( 1, [ 1 ] ).setName( 'silk pick clear' );

	const kPickA = Fn( () => {

		const key = pickKey( pos.element( instanceIndex ).xyz ).toVar();
		If( key.notEqual( u32( 0xffffffff ) ), () => {

			atomicMin( pick.element( u32( 0 ) ), key );

		} );

	} )().compute( P ).setName( 'silk pick a' );

	const kPickB = Fn( () => {

		const key = pickKey( pos.element( instanceIndex ).xyz ).toVar();
		If( key.notEqual( u32( 0xffffffff ) ).and( key.equal( atomicLoad( pick.element( u32( 0 ) ) ) ) ), () => {

			atomicMin( pick.element( u32( 1 ) ), instanceIndex );

		} );

	} )().compute( P ).setName( 'silk pick b' );

	const kPickC = Fn( () => {

		const idx = atomicLoad( pick.element( u32( 1 ) ) ).toVar();
		If( idx.lessThan( u32( P ) ), () => {

			const p = pos.element( idx ).xyz;
			atomicStore( pick.element( u32( 2 ) ), floatBitsToUint( p.x ) );
			atomicStore( pick.element( u32( 3 ) ), floatBitsToUint( p.y ) );
			atomicStore( pick.element( u32( 4 ) ), floatBitsToUint( p.z ) );

		} );

	} )().compute( 1, [ 1 ] ).setName( 'silk pick c' );

	const pickList = [ kPickClear, kPickA, kPickB, kPickC ];

	// ── frame schedule ────────────────────────────────────────────────────────
	const hashList = [ kHashClear, kHashCount, kScanA, kScanB, kScanC, kScatter, kSelf ];
	let frameList = [];
	const state = { substeps: 0, selfEvery: 0, selfOn: true };
	function schedule( substeps, selfEvery, selfOn ) {

		state.substeps = substeps; state.selfEvery = selfEvery; state.selfOn = selfOn;
		frameList = [ kWind ];
		for ( let k = 0; k < substeps; k ++ ) {

			frameList.push( kIntegrate );
			// alternate sweep direction each substep to cancel Gauss–Seidel ordering bias
			const cols = ( k & 1 ) ? colorKernels.slice().reverse() : colorKernels;
			frameList.push( ...cols );
			if ( selfOn && ( k % selfEvery ) === selfEvery - 1 ) frameList.push( ...hashList );
			frameList.push( kFinalize );

		}

		frameList.push( kNormals );

	}

	const allKernels = [ kResetHang, kResetDrop, kWind, kIntegrate, ...colorKernels, ...hashList, kFinalize, kNormals, ...pickList ];

	return {
		N, S, P, NN, spacing, A, K, T,
		attrs: { posA, nrmA, tanA, pickA },
		U, anchorPos, anchorRest, anchorIJ, anchorsJS,
		constraintCount,
		get frameList() {

			return frameList;

		},
		state,
		schedule,
		kResetHang, kResetDrop, kNormals, pickList,
		dispose() {

			for ( const k of allKernels ) k.dispose();
			for ( const a of attrs ) {

				try {

					renderer._attributes?.delete( a );

				} catch ( e ) { /* already gone */ }

			}

		},
	};

}
