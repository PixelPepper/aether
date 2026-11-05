// Aurora — GPU particle physics.
//
//  1. electrons: Boris pusher in an Earth dipole field (+ divergence-free local perturbation, + convection E),
//     continuous slowing-down in a two-scale-height thermosphere (range ∝ E^1.67). Energy lost each sub-step is
//     deposited with a fixed-point atomicAdd into an observer-centric 3D grid.
//  2. volume:   the integer grid *is* the fast excitation state (decays in place); a float grid low-passes it into
//     the metastable O(¹D) reservoir. Both are written as emissivity (÷ voxel volume) into an rgba16f 3D texture.
//  3. probe:    stratified Monte-Carlo integral of the volume's irradiance at the observer (colour + direction),
//     used by the landscape as a huge area light. No CPU readback.
import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, uniform, uniformArray, instancedArray, storage, instanceIndex, float, uint, vec3, vec4, uvec3,
	sin, cos, exp, log, sqrt, pow, max, min, clamp, abs, floor, atan, length, normalize, cross, dot, mix, smoothstep, select,
	atomicAdd, atomicLoad, atomicStore, textureStore,
} from 'three/tsl';
import { R_E, GRID, gU, gV, gW, invU, dPhidU, invV, dHdV, invW, dRhodW, airRho, colMass, rangeOf, energyOf } from './model.js';

const H_REF = 110; // km — the curtain geometry is defined where its lower border is
const H_INJ = 480; // km — injection altitude (top of the acceleration region)
const H_TOP = 540;
const V0 = 250; // km/s (sim units: electrons are slowed ~200× so the precipitation is watchable)
const OMEGA = 140; // gyro-frequency (rad/s at |B|=1) — resolved by the pusher, ω·dt ≈ 0.8
const L_HALF = 1500; // half-length of the arcs (km)
const R_ARC = 2200; // radius of curvature of the arcs (centred on the geomagnetic pole)
export const ARC_MODES = [ // Kelvin–Helmholtz-like fold modes: wavelength (km), cross-arc amplitude (km), phase speed
	{ lam: 760, amp: 70, c: 14 },
	{ lam: 240, amp: 26, c: - 22 },
	{ lam: 95, amp: 9, c: 31 },
];

// geomagnetic frame: observer at magnetic latitude 67° → dip angle 78°
const MLAT = 67 * Math.PI / 180;
const DIP_AXIS = new THREE.Vector3( 0, Math.sin( MLAT ), - Math.cos( MLAT ) ); // toward the geomagnetic north pole
const B_NORM = 1 / Math.sqrt( 1 + 3 * Math.sin( MLAT ) ** 2 );

// ── RNG (PCG) ──
const pcg = ( s ) => {

	const st = s.mul( 747796405 ).add( 2891336453 );
	const w = st.shiftRight( st.shiftRight( 28 ).add( 4 ) ).bitXor( st ).mul( 277803737 );
	return w.shiftRight( 22 ).bitXor( w );

};

const rnd = ( seed ) => {

	seed.assign( pcg( seed ) );
	return seed.toFloat().mul( 1 / 4294967296 ).toVar();

};

const gauss = ( seed ) => {

	const a = max( rnd( seed ), 1e-7 );
	const b = rnd( seed );
	return sqrt( log( a ).mul( - 2 ) ).mul( cos( b.mul( 2 * Math.PI ) ) ).toVar();

};

export class AuroraSim {

	constructor( renderer, cfg ) {

		this.renderer = renderer;
		this.cfg = cfg;
		const { NX, NY, NZ, maxElectrons, nSub } = cfg;
		this.NV = NX * NY * NZ;
		this.count = cfg.electrons;

		const u = this.u = {
			time: uniform( 0 ), // sheet time (scaled by the speed control)
			dt: uniform( 1 / 180 ), // electron sub-step
			frame: uniform( 0, 'uint' ),
			gain: uniform( 1 ), // fixed-point deposit scale
			decay: uniform( 0.97 ), // fast channel decay per frame
			kRed: uniform( 0.01 ), // slow channel low-pass factor per frame
			volNorm: uniform( 1 ),
			act: uniform( 0.6 ),
			pSub: uniform( 0 ),
			pDiff: uniform( 0.12 ),
			softFrac: uniform( 0.22 ),
			softW: uniform( 2.5 ),
			drift: uniform( new THREE.Vector3( 2, 0, 0 ) ),
			arcCum: uniform( new THREE.Vector3( 0.5, 0.8, 1 ) ),
			fold: uniform( new THREE.Vector3( 0.5, 0.8, 0.8 ) ), // κ per mode (≥1 means the sheet folds over)
			diffD: uniform( 260 ),
			subCum: uniform( new THREE.Vector4( 1, 1, 1, 1 ) ),
			initPhase: uniform( 0 ),
			lightK: uniform( 0.08 ),
		};
		// arcs: (distance north of the observer at 110 km, characteristic energy keV, brightness, phase seed)
		this.arcs = uniformArray( [ new THREE.Vector4( 300, 5, 1, 0.3 ), new THREE.Vector4( 400, 3.5, 0.8, 2.1 ), new THREE.Vector4( 620, 2.5, 0.6, 4.4 ) ], 'vec4' );
		// substorm blobs: A = (cx, cz, R, rotation), B = (strength, energy keV, age, phase)
		this.subA = uniformArray( [ 0, 1, 2, 3 ].map( () => new THREE.Vector4( 0, - 300, 50, 0 ) ), 'vec4' );
		this.subB = uniformArray( [ 0, 1, 2, 3 ].map( () => new THREE.Vector4( 0, 8, 0, 0 ) ), 'vec4' );

		// ── buffers ──
		this.ePos = instancedArray( maxElectrons, 'vec4' ); // xyz km, w residual range (column mass)
		this.eVel = instancedArray( maxElectrons, 'vec4' ); // xyz km/s, w macro-particle weight
		this.dep = instancedArray( this.NV, 'uint' ).toAtomic(); // fast state, fixed point
		this.slow = instancedArray( this.NV, 'float' );
		this.PROBE_T = 256;
		this.PROBE_S = cfg.probeSamples;
		this.partial = instancedArray( this.PROBE_T * 2, 'vec4' );
		this.light = instancedArray( 2, 'vec4' ); // [0] irradiance rgb, [1] mean direction
		this.lightRead = storage( this.light.value, 'vec4', 2 ).toReadOnly();
		this.attributes = [ this.ePos.value, this.eVel.value, this.dep.value, this.slow.value, this.partial.value, this.light.value ];

		const tex = this.volTex = new THREE.Storage3DTexture( NX, NY, NZ );
		tex.type = THREE.HalfFloatType;
		tex.format = THREE.RGBAFormat;
		tex.magFilter = tex.minFilter = THREE.LinearFilter;
		tex.wrapS = THREE.RepeatWrapping; // azimuth wraps
		tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
		tex.generateMipmaps = false;

		this.nSub = nSub;
		this.computes = [];
		this._build();

	}

	// ── fields ──
	bField( p ) {

		const r = p.add( vec3( 0, R_E, 0 ) );
		const rl = length( r );
		const rh = r.div( rl );
		const m = vec3( - DIP_AXIS.x, - DIP_AXIS.y, - DIP_AXIS.z );
		const md = dot( m, rh );
		const s = pow( float( R_E ).div( rl ), 3 ).mul( B_NORM );
		const Bd = rh.mul( md.mul( 3 ) ).sub( m ).mul( s );
		// local perturbation δB = ∇×(ψ ŷ) — horizontal, divergence-free, ~3 % of |B|: gently bends and twists flux tubes
		const t = this.u.time;
		const a1 = p.x.mul( 0.011 ).add( p.z.mul( 0.004 ) ).add( p.y.mul( 0.006 ) ).add( t.mul( 0.05 ) );
		const a2 = p.x.mul( - 0.005 ).add( p.z.mul( 0.013 ) ).sub( p.y.mul( 0.004 ) ).sub( t.mul( 0.035 ) );
		const c1 = cos( a1 ).mul( 0.03 ), c2 = cos( a2 ).mul( 0.022 );
		const dB = vec3( c1.mul( 0.34 ).add( c2.mul( 0.93 ) ), 0, c1.mul( - 0.94 ).add( c2.mul( 0.36 ) ) );
		return Bd.add( dB );

	}

	// ── injection sources (all positions are footprints at H_REF) ──
	arcSource( ref, E0, wt, seed ) {

		const u = this.u;
		const r1 = rnd( seed );
		const j = select( r1.lessThan( u.arcCum.x ), uint( 0 ), select( r1.lessThan( u.arcCum.y ), uint( 1 ), uint( 2 ) ) );
		const A = this.arcs.element( j ).toVar();
		const xi = rnd( seed ).mul( 2 ).sub( 1 ).mul( L_HALF ).toVar();
		const t = u.time;
		const eta = float( 0 ).toVar();
		const xs = xi.toVar();
		ARC_MODES.forEach( ( M, m ) => {

			const k = 2 * Math.PI / M.lam;
			const ph = xi.mul( k ).sub( t.mul( k * M.c ) ).add( A.w.mul( 1.3 + m * 0.71 ) );
			const env = sin( xi.mul( 1 / ( 300 + 140 * m ) ).add( t.mul( 0.05 + 0.03 * m ) ).add( A.w.mul( 2.1 ).add( m * 1.7 ) ) ).mul( 0.5 ).add( 0.5 );
			eta.addAssign( sin( ph ).mul( M.amp ).mul( u.act ).mul( env.mul( 0.7 ).add( 0.3 ) ) );
			const kap = m === 0 ? u.fold.x : m === 1 ? u.fold.y : u.fold.z;
			xs.addAssign( cos( ph ).mul( kap.mul( 1 / k ) ).mul( env ) );

		} );
		eta.addAssign( gauss( seed ).mul( 2.2 ) );
		ref.x.assign( xs );
		ref.z.assign( A.x.add( eta ).negate().sub( xs.mul( xs ).mul( 1 / ( 2 * R_ARC ) ) ) );
		// field-aligned rays: interference of drifting along-arc modes (rays move along the curtain)
		const s = sin( xi.mul( 0.47 ).add( t.mul( 0.9 ) ).add( A.w ) )
			.add( sin( xi.mul( 0.23 ).sub( t.mul( 0.6 ) ).add( A.w.mul( 2 ) ) ) )
			.add( sin( xi.mul( 0.121 ).add( t.mul( 0.31 ) ) ) ).mul( 1 / 3 );
		const big = sin( xi.mul( 1 / 420 ).add( t.mul( 0.04 ) ).add( A.w.mul( 3 ) ) ).mul( 0.5 ).add( 0.5 );
		wt.assign( A.z.mul( exp( s.mul( 2.3 ) ) ).mul( big.mul( 0.75 ).add( 0.25 ) ).mul( 0.6 ) );
		E0.assign( A.y.mul( sin( xi.mul( 1 / 170 ).add( t.mul( 0.08 ) ).add( A.w ) ).mul( 0.3 ).add( 1 ) ).mul( gauss( seed ).mul( 0.12 ).add( 1 ) ) );

	}

	diffuseSource( ref, E0, wt, seed ) {

		// diffuse aurora equatorward of the arcs, broken into pulsating patches
		const u = this.u;
		const x = rnd( seed ).mul( 2 ).sub( 1 ).mul( 1300 ).toVar();
		const z = u.diffD.sub( rnd( seed ).mul( 280 ) ).negate().toVar();
		ref.x.assign( x ); ref.z.assign( z );
		const pn = sin( x.mul( 1 / 55 ).add( sin( z.mul( 1 / 90 ) ).mul( 2 ) ) ).mul( sin( z.mul( 1 / 47 ).add( sin( x.mul( 1 / 120 ) ) ) ) );
		const cell = floor( x.mul( 1 / 70 ) ).mul( 1.7 ).add( floor( z.mul( 1 / 60 ) ).mul( 2.3 ) );
		const pulse = smoothstep( 0.2, 0.8, sin( u.time.mul( 1.1 ).add( sin( cell ).mul( 43758.5 ).fract().mul( 6.283 ) ) ).mul( 0.5 ).add( 0.5 ) );
		wt.assign( smoothstep( 0.0, 0.6, pn ).mul( pulse ).mul( 1.4 ).add( 0.25 ).mul( 0.4 ) );
		E0.assign( max( gauss( seed ).mul( 0.8 ).add( 3.2 ), 1.0 ) );

	}

	substormSource( ref, E0, wt, pitch, seed ) {

		const u = this.u;
		const r2 = rnd( seed );
		const k = select( r2.lessThan( u.subCum.x ), uint( 0 ), select( r2.lessThan( u.subCum.y ), uint( 1 ), select( r2.lessThan( u.subCum.z ), uint( 2 ), uint( 3 ) ) ) );
		const A = this.subA.element( k ).toVar();
		const B = this.subB.element( k ).toVar();
		const q = rnd( seed );
		const t = u.time;
		If( rnd( seed ).lessThan( 0.62 ), () => {

			// westward-travelling-surge spiral: a rolled-up curtain
			const th = A.w.add( q.mul( 2 * Math.PI * 1.7 ) );
			const r = A.z.mul( q.mul( 0.85 ).add( 0.12 ) );
			ref.x.assign( A.x.add( cos( th ).mul( r ).mul( 1.5 ) ) );
			ref.z.assign( A.y.add( sin( th ).mul( r ).mul( 0.8 ) ).add( gauss( seed ).mul( 2.5 ) ) );
			const s = sin( q.mul( 157 ).add( t.mul( 2.3 ) ).add( B.w ) ).add( sin( q.mul( 389 ).sub( t.mul( 3.1 ) ) ) ).mul( 0.5 );
			wt.assign( exp( s.mul( 2.0 ) ).mul( 1.6 ) );

		} ).Else( () => {

			// the brightened onset arc through the bulge
			const x = A.x.add( q.mul( 2 ).sub( 1 ).mul( A.z ).mul( 2.2 ) ).toVar();
			ref.x.assign( x );
			ref.z.assign( A.y.add( sin( x.mul( 1 / 45 ).add( t.mul( 0.6 ) ).add( B.w ) ).mul( A.z.mul( 0.12 ) ) ).add( gauss( seed ).mul( 2.5 ) ) );
			const s = sin( x.mul( 0.5 ).add( t.mul( 1.7 ) ) ).add( sin( x.mul( 0.19 ).sub( t.mul( 2.3 ) ).add( B.w ) ) ).mul( 0.5 );
			wt.assign( exp( s.mul( 2.2 ) ).mul( 1.4 ) );

		} );
		E0.assign( max( B.y.mul( gauss( seed ).mul( 0.15 ).add( 1 ) ), 0.5 ) );
		pitch.assign( 0.3 );

	}

	respawn( p, v, Q, w, seed, isInit ) {

		const u = this.u;
		const ref = vec3( 0, H_REF, 0 ).toVar();
		const E0 = float( 4 ).toVar();
		const wt = float( 1 ).toVar();
		const pitch = float( 0.5 ).toVar();
		const r0 = rnd( seed );
		If( r0.lessThan( u.pSub ), () => {

			this.substormSource( ref, E0, wt, pitch, seed );

		} ).ElseIf( r0.lessThan( u.pSub.add( u.pDiff ) ), () => {

			this.diffuseSource( ref, E0, wt, seed );

		} ).Else( () => {

			this.arcSource( ref, E0, wt, seed );

		} );
		// soft (≈100–500 eV) population: stops at 200–300 km and feeds the red O(¹D) tops
		If( rnd( seed ).lessThan( u.softFrac ), () => {

			E0.assign( log( max( rnd( seed ), 1e-4 ) ).mul( - 0.18 ).add( 0.07 ) );
			wt.mulAssign( u.softW );
			pitch.assign( 0.9 );

		} );
		const bh = normalize( this.bField( ref ) ).toVar();
		const up = bh.negate();
		const hs = float( H_INJ ).toVar();
		const Q0 = rangeOf( E0 ).toVar();
		if ( isInit ) {

			// spread the initial population along the flux tube so the sky is lit on the first frame
			const hStop = max( log( float( 0.976 ).div( Q0 ) ).mul( 6.5 ), log( float( 0.024 ).div( Q0 ) ).mul( 45 ) ).add( 100 );
			hs.assign( mix( clamp( hStop.add( 1 ), 75, H_INJ ), float( H_INJ ), rnd( seed ) ) );

		}

		p.assign( ref.add( up.mul( hs.sub( H_REF ).div( up.y ) ) ) );
		const a = min( abs( gauss( seed ) ).mul( pitch ), 1.45 ).toVar();
		const g = rnd( seed ).mul( 2 * Math.PI );
		const e1 = normalize( cross( bh, vec3( 1, 0, 0 ) ) ).toVar();
		const e2 = cross( bh, e1 );
		v.assign( bh.mul( cos( a ) ).add( e1.mul( cos( g ) ).add( e2.mul( sin( g ) ) ).mul( sin( a ) ) ).mul( V0 ) );
		if ( isInit ) Q0.assign( max( Q0.sub( colMass( hs ).sub( colMass( float( H_INJ ) ) ).div( cos( a ) ) ), Q0.mul( 0.05 ) ) );
		Q.assign( Q0 );
		w.assign( wt );

	}

	deposit( p, dE, seed ) {

		const { NX, NY, NZ } = this.cfg;
		// sub-voxel jitter for anti-aliasing; grows with altitude to model the diffusion of long-lived O(¹D)
		const jr = smoothstep( 170, 330, p.y ).mul( 7 ).add( 0.7 );
		const q = p.add( vec3( rnd( seed ).sub( 0.5 ), rnd( seed ).sub( 0.5 ).mul( 0.6 ), rnd( seed ).sub( 0.5 ) ).mul( jr.mul( 2 ) ) ).toVar();
		const rho = length( q.xz );
		If( q.y.greaterThan( GRID.h0 ).and( q.y.lessThan( GRID.h1 ) ).and( rho.lessThan( GRID.rhoMax ) ), () => {

			const phi = atan( q.x, q.z.negate() );
			const ix = min( uint( floor( gU( phi ).mul( NX ) ) ), uint( NX - 1 ) );
			const iy = min( uint( floor( gV( q.y ).mul( NY ) ) ), uint( NY - 1 ) );
			const iz = min( uint( floor( gW( rho ).mul( NZ ) ) ), uint( NZ - 1 ) );
			const idx = ix.add( iy.mul( NX ) ).add( iz.mul( NX * NY ) );
			atomicAdd( this.dep.element( idx ), uint( dE.mul( this.u.gain ).add( rnd( seed ) ) ) );

		} );

	}

	_build() {

		const u = this.u;
		const { NX, NY, NZ, maxElectrons } = this.cfg;
		const NV = this.NV;

		// ── init: spawn every electron somewhere along its flux tube ──
		this.initKernel = Fn( () => {

			const i = instanceIndex;
			const seed = uint( i ).mul( 2654435761 ).bitXor( uint( 0x5bd1e995 ) ).toVar();
			seed.assign( pcg( seed ) );
			const p = vec3( 0 ).toVar(), v = vec3( 0 ).toVar(), Q = float( 0 ).toVar(), w = float( 0 ).toVar();
			this.respawn( p, v, Q, w, seed, true );
			this.ePos.element( i ).assign( vec4( p, Q ) );
			this.eVel.element( i ).assign( vec4( v, w ) );

		} )().compute( maxElectrons ).setName( 'aurora.init' );

		// ── electron push ──
		this.stepKernel = Fn( () => {

			const i = instanceIndex;
			const P = this.ePos.element( i ).toVar();
			const Vv = this.eVel.element( i ).toVar();
			const p = P.xyz.toVar(), Q = P.w.toVar(), v = Vv.xyz.toVar(), w = Vv.w.toVar();
			const seed = uint( i ).mul( 2654435761 ).bitXor( u.frame.mul( 2246822519 ) ).toVar();
			seed.assign( pcg( seed ) );
			const dt = u.dt;
			const qmh = dt.mul( - OMEGA * 0.5 ); // (q/m)·dt/2 for an electron

			Loop( this.nSub, () => {

				const B = this.bField( p ).toVar();
				const E = cross( B, u.drift ); // E = −v_d × B → E×B drift = v_d
				// Boris: half electric kick, magnetic rotation, half electric kick
				const vm = v.add( E.mul( qmh ) ).toVar();
				const tv = B.mul( qmh ).toVar();
				const sv = tv.mul( float( 2 ).div( dot( tv, tv ).add( 1 ) ) );
				const vp = vm.add( cross( vm, tv ) );
				const vplus = vm.add( cross( vp, sv ) );
				v.assign( vplus.add( E.mul( qmh ) ) );
				p.addAssign( v.mul( dt ) );
				// continuous slowing down: residual range decreases with the traversed column mass
				const ds = length( v ).mul( dt );
				const Qn = Q.sub( airRho( p.y ).mul( ds ) ).toVar();
				const dE = energyOf( Q ).sub( energyOf( Qn ) ).mul( w );
				this.deposit( p, dE, seed );
				Q.assign( Qn );
				If( Qn.lessThan( 2e-5 ).or( p.y.lessThan( 72 ) ).or( p.y.greaterThan( H_TOP ) ).or( length( p.xz ).greaterThan( GRID.rhoMax * 1.3 ) ), () => {

					this.respawn( p, v, Q, w, seed, false );

				} );

			} );

			this.ePos.element( i ).assign( vec4( p, Q ) );
			this.eVel.element( i ).assign( vec4( v, w ) );

		} )().compute( maxElectrons ).setName( 'aurora.push' );
		this.stepKernel.count = this.count;

		// ── volume: decay, O(¹D) reservoir, write emissivity ──
		this.volumeKernel = Fn( () => {

			const i = instanceIndex;
			const ix = i.mod( NX ), iy = i.div( NX ).mod( NY ), iz = i.div( NX * NY );
			const c = atomicLoad( this.dep.element( i ) ).toFloat().toVar();
			const s = this.slow.element( i ).toVar();
			s.addAssign( c.sub( s ).mul( u.kRed ) );
			this.slow.element( i ).assign( s );
			const dither = pcg( i.bitXor( u.frame.mul( 1664525 ) ) ).toFloat().mul( 1 / 4294967296 );
			atomicStore( this.dep.element( i ), uint( c.mul( u.decay ).add( dither ) ) );
			// voxel volume (km³) from the grid Jacobian
			const fu = ix.toFloat().add( 0.5 ).div( NX ), fv = iy.toFloat().add( 0.5 ).div( NY ), fw = iz.toFloat().add( 0.5 ).div( NZ );
			const vol = invW( fw ).mul( dPhidU( fu ).div( NX ) ).mul( dRhodW( fw ).div( NZ ) ).mul( dHdV( fv ).div( NY ) );
			const k = u.volNorm.div( vol );
			textureStore( this.volTex, uvec3( ix, iy, iz ), vec4( c.mul( k ), s.mul( k ), 0, 1 ) );

		} )().compute( NV ).setName( 'aurora.volume' );

		this.computes.push( this.initKernel, this.stepKernel, this.volumeKernel );

	}

	// probe kernels need the emission model, built once the sky exists
	buildProbe( emit ) {

		const u = this.u;
		const { NX, NY, NZ } = this.cfg;
		const NV = this.NV;
		const T = this.PROBE_T, S = this.PROBE_S;
		const total = T * S;
		const stride = NV / total;

		this.probeKernel = Fn( () => {

			const t = instanceIndex;
			const acc = vec3( 0 ).toVar(), dacc = vec3( 0 ).toVar();
			const seed = uint( t ).mul( 2654435761 ).bitXor( u.frame.mul( 40503 ) ).toVar();
			Loop( S, ( { i } ) => {

				// stratified: sample k covers [k·stride, (k+1)·stride)
				const kk = uint( i ).mul( T ).add( t );
				const idx = min( uint( kk.toFloat().add( rnd( seed ) ).mul( stride ) ), uint( NV - 1 ) );
				const c = atomicLoad( this.dep.element( idx ) ).toFloat();
				const s = this.slow.element( idx );
				const ix = idx.mod( NX ), iy = idx.div( NX ).mod( NY ), iz = idx.div( NX * NY );
				const fu = ix.toFloat().add( 0.5 ).div( NX ), fv = iy.toFloat().add( 0.5 ).div( NY ), fw = iz.toFloat().add( 0.5 ).div( NZ );
				const phi = invU( fu ), h = invV( fv ), rho = invW( fw );
				// emitted power of the voxel ∝ counts (emissivity × volume); irradiance at the observer ∝ 1/r²
				const pos = vec3( sin( phi ).mul( rho ), h, cos( phi ).mul( rho ).negate() );
				const r2 = dot( pos, pos );
				const e = emit( c, s, h ).mul( u.volNorm.div( r2 ) );
				acc.addAssign( e );
				dacc.addAssign( pos.mul( inverseLen( r2 ) ).mul( e.y.add( e.x.mul( 0.5 ) ) ) );

			} );
			this.partial.element( t ).assign( vec4( acc.mul( stride ), 0 ) );
			this.partial.element( t.add( T ) ).assign( vec4( dacc.mul( stride ), 0 ) );

		} )().compute( T ).setName( 'aurora.probe' );

		this.probeResolve = Fn( () => {

			const acc = vec3( 0 ).toVar(), dacc = vec3( 0 ).toVar();
			Loop( T, ( { i } ) => {

				acc.addAssign( this.partial.element( i ).xyz );
				dacc.addAssign( this.partial.element( uint( i ).add( T ) ).xyz );

			} );
			const prev = this.light.element( 0 ).xyz;
			const k = u.lightK;
			this.light.element( 0 ).assign( vec4( mix( prev, acc, k ), 1 ) );
			const dn = dacc.div( max( length( dacc ), 1e-9 ) );
			const pd = this.light.element( 1 ).xyz;
			this.light.element( 1 ).assign( vec4( normalize( mix( pd, dn, k ).add( vec3( 0, 1e-4, 0 ) ) ), 1 ) );

		} )().compute( 1 ).setName( 'aurora.probeResolve' );

		this.computes.push( this.probeKernel, this.probeResolve );

	}

	async init() {

		await this.renderer.computeAsync( this.initKernel );

	}

	step() {

		this.renderer.compute( [ this.stepKernel, this.volumeKernel, this.probeKernel, this.probeResolve ] );

	}

	setCount( n ) {

		this.count = Math.min( n, this.cfg.maxElectrons );
		this.stepKernel.count = this.count;

	}

	dispose() {

		for ( const c of this.computes ) c.dispose();
		this.volTex.dispose();
		const attrs = this.renderer._attributes;
		for ( const a of this.attributes ) {

			try {

				attrs?.delete( a );

			} catch ( e ) { /* already gone */ }

		}

	}

}

const inverseLen = ( r2 ) => float( 1 ).div( sqrt( r2 ) );
export const _keep = { clamp, gW, invV };
