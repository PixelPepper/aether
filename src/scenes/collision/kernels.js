// GPU kernels for the merger: tiled direct N-body (shared-memory tiles), tracer integration in the
// N-body field (tiled again), CIC gas splat with integer atomics, sticky-gas dissipation, density/shock
// triggered star formation and per-star dust extinction ray-marched through the gas grid.
import {
	Fn, If, Loop, uniform, instancedArray, instanceIndex, invocationLocalIndex, workgroupArray, workgroupBarrier,
	vec3, vec4, float, int, uint, dot, length, max, min, clamp, mix, exp, sqrt, inverseSqrt, floor, smoothstep, select, hash, abs,
	atomicAdd, atomicStore, atomicLoad,
} from 'three/tsl';
import { T_GAS } from './ics.js';

export const FIX = 2048; // fixed-point scale for atomic float accumulation

export function createSim( cfg ) {

	const { nm, nt, nGas, wg, G } = cfg;

	// ── buffers ──
	const mPos = instancedArray( nm, 'vec4' ).setName( 'mPos' ); // xyz, mass
	const mVel = instancedArray( nm, 'vec4' ).setName( 'mVel' ); // xyz, kind (0 bh, 1 halo, 2 bulge, +10*galaxy, 3 idle, 4 satellite, 5 perturber)
	const mAcc = instancedArray( nm, 'vec4' ).setName( 'mAcc' );
	const tPos = instancedArray( nt, 'vec4' ).setName( 'tPos' ); // xyz, local gas density (gas only)
	const tVel = instancedArray( nt, 'vec4' ).setName( 'tVel' ); // xyz, birth time (gas)
	const tAcc = instancedArray( nt, 'vec4' ).setName( 'tAcc' );
	const tInfo = instancedArray( nt, 'vec4' ).setName( 'tInfo' ); // type, galaxy, lum, colour
	const tRen = instancedArray( nt, 'vec4' ).setName( 'tRen' ); // tau, lum, hii glow, colour
	const G3 = G * G * G;
	const gridI = instancedArray( G3 * 4, 'int' ).setName( 'gridI' ).toAtomic();
	const gridF = instancedArray( G3, 'vec4' ).setName( 'gridF' ); // rho_n, mean velocity
	const statI = instancedArray( 4, 'int' ).setName( 'statI' ).toAtomic();

	// ── uniforms ──
	const U = {
		hM: uniform( 0.0 ), // massive substep
		hT: uniform( 0.0 ), // tracer step
		eps2M: uniform( cfg.epsM * cfg.epsM ),
		eps2T: uniform( cfg.epsT * cfg.epsT ),
		time: uniform( 0.0 ),
		frame: uniform( 0, 'uint' ),
		gridMin: uniform( cfg.gridMin.clone() ),
		invCell: uniform( G / cfg.box ),
		densNorm: uniform( cfg.densNorm ), // 1 / reference CIC count
		drag: uniform( 0.12 ),
		sfBase: uniform( cfg.sfBase ),
		sfEff: uniform( 1.0 ),
		shockSF: uniform( 40.0 ),
		camPos: uniform( cfg.gridMin.clone() ),
		kappa: uniform( 0.35 ),
		tHII: uniform( 2.6 ),
		tYoung: uniform( 32.0 ),
		clusterLum: uniform( cfg.clusterLum ),
		gasLum: uniform( cfg.gasLum ),
		lumScale: uniform( 1.0 ),
	};

	// ───────────────────────── tiled gravity ─────────────────────────
	// One thread per body; the massive set is streamed through workgroup shared memory in tiles of `wg`.
	const nTiles = nm / wg;

	function gravityAt( p, eps2 ) {

		const tile = workgroupArray( 'vec4', wg );
		const acc = vec3( 0 ).toVar( 'acc' );
		Loop( { start: int( 0 ), end: int( nTiles ), type: 'int', condition: '<', name: 'tl' }, ( { tl } ) => {

			tile.element( invocationLocalIndex ).assign( mPos.element( uint( tl ).mul( wg ).add( invocationLocalIndex ) ) );
			workgroupBarrier();
			Loop( { start: int( 0 ), end: int( wg ), type: 'int', condition: '<', update: '+= 4', name: 'jj' }, ( { jj } ) => {

				for ( let u = 0; u < 4; u ++ ) {

					const q = tile.element( jj.add( u ) ).toVar();
					const d = q.xyz.sub( p ).toVar();
					const r2 = dot( d, d ).add( eps2 );
					const ir = inverseSqrt( r2 ).toVar();
					acc.addAssign( d.mul( q.w.mul( ir ).mul( ir ).mul( ir ) ) );

				}

			} );
			workgroupBarrier();

		} );
		return acc;

	}

	const mKickDrift = Fn( () => {

		const i = instanceIndex;
		const v = mVel.element( i ).toVar();
		const nv = v.xyz.add( mAcc.element( i ).xyz.mul( U.hM.mul( 0.5 ) ) ).toVar();
		const p = mPos.element( i ).toVar();
		mVel.element( i ).assign( vec4( nv, v.w ) );
		mPos.element( i ).assign( vec4( p.xyz.add( nv.mul( U.hM ) ), p.w ) );

	} )().compute( nm, [ wg ] ).setName( 'mKickDrift' );

	const mForceKick = Fn( () => {

		const i = instanceIndex;
		const p = mPos.element( i ).xyz.toVar();
		const a = gravityAt( p, U.eps2M );
		const v = mVel.element( i ).toVar();
		mAcc.element( i ).assign( vec4( a, 0 ) );
		mVel.element( i ).assign( vec4( v.xyz.add( a.mul( U.hM.mul( 0.5 ) ) ), v.w ) );

	} )().compute( nm, [ wg ] ).setName( 'mForceKick' );

	const tKickDrift = Fn( () => {

		const i = instanceIndex;
		const v = tVel.element( i ).toVar();
		const nv = v.xyz.add( tAcc.element( i ).xyz.mul( U.hT.mul( 0.5 ) ) ).toVar();
		const p = tPos.element( i ).toVar();
		tVel.element( i ).assign( vec4( nv, v.w ) );
		tPos.element( i ).assign( vec4( p.xyz.add( nv.mul( U.hT ) ), p.w ) );

	} )().compute( nt, [ wg ] ).setName( 'tKickDrift' );

	const tForceKick = Fn( () => {

		const i = instanceIndex;
		const p = tPos.element( i ).xyz.toVar();
		const a = gravityAt( p, U.eps2T );
		const v = tVel.element( i ).toVar();
		tAcc.element( i ).assign( vec4( a, 0 ) );
		tVel.element( i ).assign( vec4( v.xyz.add( a.mul( U.hT.mul( 0.5 ) ) ), v.w ) );

	} )().compute( nt, [ wg ] ).setName( 'tForceKick' );

	// ───────────────────────── gas grid ─────────────────────────
	const gClear = Fn( () => {

		const b = instanceIndex.mul( 4 );
		atomicStore( gridI.element( b ), int( 0 ) );
		atomicStore( gridI.element( b.add( 1 ) ), int( 0 ) );
		atomicStore( gridI.element( b.add( 2 ) ), int( 0 ) );
		atomicStore( gridI.element( b.add( 3 ) ), int( 0 ) );

	} )().compute( G3 ).setName( 'gClear' );

	const gridCoord = ( p ) => p.sub( U.gridMin ).mul( U.invCell ).sub( 0.5 );
	const inGrid = ( g ) => g.x.greaterThanEqual( 0 ).and( g.y.greaterThanEqual( 0 ) ).and( g.z.greaterThanEqual( 0 ) )
		.and( g.x.lessThan( G - 1 ) ).and( g.y.lessThan( G - 1 ) ).and( g.z.lessThan( G - 1 ) );

	// cloud-in-cell deposit of mass + momentum (fixed point i32 atomics)
	const gSplat = Fn( () => {

		const i = instanceIndex;
		const p = tPos.element( i ).xyz.toVar();
		const g = gridCoord( p ).toVar();
		If( inGrid( g ), () => {

			const v = tVel.element( i ).xyz.toVar();
			const b = floor( g ).toVar();
			const f = g.sub( b ).toVar();
			const base = int( b.x ).add( int( b.y ).mul( G ) ).add( int( b.z ).mul( G * G ) ).toVar();
			for ( let c = 0; c < 8; c ++ ) {

				const dx = c & 1, dy = ( c >> 1 ) & 1, dz = ( c >> 2 ) & 1;
				const w = ( dx ? f.x : f.x.oneMinus() ).mul( dy ? f.y : f.y.oneMinus() ).mul( dz ? f.z : f.z.oneMinus() ).toVar();
				const idx = base.add( dx + dy * G + dz * G * G ).mul( 4 ).toVar();
				const wf = w.mul( FIX );
				atomicAdd( gridI.element( idx ), int( wf ) );
				atomicAdd( gridI.element( idx.add( 1 ) ), int( wf.mul( v.x ) ) );
				atomicAdd( gridI.element( idx.add( 2 ) ), int( wf.mul( v.y ) ) );
				atomicAdd( gridI.element( idx.add( 3 ) ), int( wf.mul( v.z ) ) );

			}

		} );

	} )().compute( nGas ).setName( 'gSplat' );

	const gResolve = Fn( () => {

		const b = instanceIndex.mul( 4 );
		const n = float( atomicLoad( gridI.element( b ) ) ).toVar();
		const px = float( atomicLoad( gridI.element( b.add( 1 ) ) ) );
		const py = float( atomicLoad( gridI.element( b.add( 2 ) ) ) );
		const pz = float( atomicLoad( gridI.element( b.add( 3 ) ) ) );
		const vm = vec3( px, py, pz ).div( max( n, 1.0 ) );
		gridF.element( instanceIndex ).assign( vec4( n.mul( U.densNorm.div( FIX ) ), vm ) );

	} )().compute( G3 ).setName( 'gResolve' );

	// trilinear sample of (rho_n, v_mean)
	const sampleGrid = ( p ) => {

		const g = gridCoord( p ).toVar();
		const ok = inGrid( g );
		const gc = clamp( g, vec3( 0 ), vec3( G - 1.001 ) ).toVar();
		const b = floor( gc ).toVar();
		const f = gc.sub( b ).toVar();
		const base = int( b.x ).add( int( b.y ).mul( G ) ).add( int( b.z ).mul( G * G ) ).toVar();
		const c = [];
		for ( let k = 0; k < 8; k ++ ) c.push( gridF.element( base.add( ( k & 1 ) + ( ( k >> 1 ) & 1 ) * G + ( ( k >> 2 ) & 1 ) * G * G ) ) );
		const x0 = mix( c[ 0 ], c[ 1 ], f.x ), x1 = mix( c[ 2 ], c[ 3 ], f.x ), x2 = mix( c[ 4 ], c[ 5 ], f.x ), x3 = mix( c[ 6 ], c[ 7 ], f.x );
		const r = mix( mix( x0, x1, f.y ), mix( x2, x3, f.y ), f.z );
		return select( ok, r, vec4( 0 ) );

	};

	const sampleRho = ( p ) => {

		const g = gridCoord( p ).toVar();
		const ok = inGrid( g );
		const gc = clamp( g, vec3( 0 ), vec3( G - 1.001 ) ).toVar();
		const b = floor( gc ).toVar();
		const f = gc.sub( b ).toVar();
		const base = int( b.x ).add( int( b.y ).mul( G ) ).add( int( b.z ).mul( G * G ) ).toVar();
		const c = [];
		for ( let k = 0; k < 8; k ++ ) c.push( gridF.element( base.add( ( k & 1 ) + ( ( k >> 1 ) & 1 ) * G + ( ( k >> 2 ) & 1 ) * G * G ) ).x );
		const x0 = mix( c[ 0 ], c[ 1 ], f.x ), x1 = mix( c[ 2 ], c[ 3 ], f.x ), x2 = mix( c[ 4 ], c[ 5 ], f.x ), x3 = mix( c[ 6 ], c[ 7 ], f.x );
		return select( ok, mix( mix( x0, x1, f.y ), mix( x2, x3, f.y ), f.z ), float( 0 ) );

	};

	// sticky-gas dissipation toward the local mean flow + star formation (density law + shock trigger)
	const gasUpdate = Fn( () => {

		const i = instanceIndex;
		const info = tInfo.element( i );
		If( info.y.lessThan( 8.5 ), () => {

			const p = tPos.element( i ).toVar();
			const v = tVel.element( i ).toVar();
			const s = sampleGrid( p.xyz ).toVar();
			const rho = s.x;
			const rhoC = min( rho, 4.0 );
			const dv = s.yzw.sub( v.xyz ).toVar();
			const shock = smoothstep( 0.05, 0.16, length( dv ) ).toVar();
			const fr = float( 1 ).sub( exp( U.drag.mul( rhoC ).mul( U.hT ).negate() ) ).mul( shock ).toVar();
			const has = rho.greaterThan( 0.02 );
			const nv = select( has, v.xyz.add( dv.mul( fr ) ), v.xyz );
			// dissipated specific energy rate (per unit time) ~ |dv|² * drag * rho
			const diss = select( has, dot( dv, dv ).mul( U.drag ).mul( rhoC ).mul( shock ), float( 0 ) );
			// star formation: Schmidt-like density law above a threshold, plus shock-induced (Barnes 2004)
			const law = rho.mul( sqrt( rho ) ).mul( smoothstep( 0.18, 0.5, rho ) );
			const rate = U.sfBase.mul( U.sfEff ).mul( law.add( diss.mul( U.shockSF ) ) );
			const age = U.time.sub( v.w );
			const ready = age.greaterThan( U.tYoung );
			const rnd = hash( i.add( U.frame.mul( uint( 196613 ) ) ) );
			const fire = ready.and( rnd.lessThan( rate.mul( U.hT ) ) );
			const birth = select( fire, U.time, v.w );
			If( fire, () => {

				atomicAdd( statI.element( 0 ), int( 1 ) );

			} );
			tVel.element( i ).assign( vec4( nv, birth ) );
			tPos.element( i ).assign( vec4( p.xyz, rho ) );

		} );

	} )().compute( nGas ).setName( 'gasUpdate' );

	// ───────────────────────── per-star shading: dust extinction + population light ─────────────────────────
	const NSTEP = cfg.extSteps;
	const shade = Fn( () => {

		const i = instanceIndex;
		const p = tPos.element( i ).xyz.toVar();
		const info = tInfo.element( i ).toVar();
		// ray from the star to the camera, clipped to the dust grid
		const toCam = U.camPos.sub( p ).toVar();
		const dist = length( toCam ).toVar();
		const d = toCam.div( max( dist, 1e-4 ) ).toVar();
		const sd = vec3(
			select( abs( d.x ).lessThan( 1e-5 ), float( 1e-5 ), d.x ),
			select( abs( d.y ).lessThan( 1e-5 ), float( 1e-5 ), d.y ),
			select( abs( d.z ).lessThan( 1e-5 ), float( 1e-5 ), d.z ) );
		const inv = vec3( 1 ).div( sd ).toVar();
		const cell = float( 1 ).div( U.invCell );
		const bmin = U.gridMin.add( cell.mul( 0.5 ) ), bmax = U.gridMin.add( cell.mul( G - 0.5 ) );
		const t0 = bmin.sub( p ).mul( inv ).toVar(), t1 = bmax.sub( p ).mul( inv ).toVar();
		const tn = min( t0, t1 ), tf = max( t0, t1 );
		const tmin = max( max( max( tn.x, tn.y ), tn.z ), 0.0 ).toVar();
		const tmax = min( min( min( tf.x, tf.y ), tf.z ), dist ).toVar();
		const len = max( tmax.sub( tmin ), 0.0 );
		const ds = len.div( NSTEP ).toVar();
		const col = float( 0 ).toVar();
		If( len.greaterThan( 0.0 ), () => {

			const jit = hash( i.add( uint( 77 ) ) ).mul( 0.5 ).add( 0.25 );
			Loop( { start: int( 0 ), end: int( NSTEP ), type: 'int', condition: '<', name: 'st' }, ( { st } ) => {

				const s = tmin.add( float( st ).add( jit ).mul( ds ) );
				col.addAssign( sampleRho( p.add( d.mul( s ) ) ) );

			} );

		} );
		const tau = col.mul( ds ).mul( U.invCell ).mul( U.kappa );

		// light by population
		const type = info.x;
		const lum = info.z.mul( U.lumScale ).toVar();
		const glow = float( 0 ).toVar();
		const colour = info.w.toVar();
		If( type.lessThan( 0.5 ).and( info.y.lessThan( 8.5 ) ), () => {

			const age = U.time.sub( tVel.element( i ).w ).toVar();
			const young = age.lessThan( U.tYoung );
			// young cluster fades ~ t^-1.1 ; HII glow for the first few Myr (ionising O stars)
			const fade = float( 1 ).add( age.div( U.tHII ) ).pow( - 1.1 ).mul( smoothstep( 0.0, 0.15, age ) );
			const hii = smoothstep( 0.0, 0.25, age ).mul( float( 1 ).sub( smoothstep( U.tHII.mul( 0.4 ), U.tHII, age ) ) );
			lum.assign( select( young, U.clusterLum.mul( fade ), U.gasLum ).mul( info.z ) );
			glow.assign( select( young, hii, float( 0 ) ).mul( info.z ) );
			colour.assign( select( young, mix( float( 1.0 ), float( 0.8 ), age.div( U.tYoung ) ), float( 0.82 ) ) );

		} );
		tRen.element( i ).assign( vec4( tau, lum, glow, colour ) );

	} )().compute( nt ).setName( 'shade' );

	const zeroStats = Fn( () => {

		atomicStore( statI.element( instanceIndex ), int( 0 ) );

	} )().compute( 4 ).setName( 'zeroStats' );

	const kernels = [ mKickDrift, mForceKick, tKickDrift, tForceKick, gClear, gSplat, gResolve, gasUpdate, shade, zeroStats ];

	return {
		buffers: { mPos, mVel, mAcc, tPos, tVel, tAcc, tInfo, tRen, gridI, gridF, statI },
		U, kernels,
		mKickDrift, mForceKick, tKickDrift, tForceKick, gClear, gSplat, gResolve, gasUpdate, shade, zeroStats,
	};

}

export { T_GAS };
