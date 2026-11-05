// Initial conditions for the galaxy merger (pure JS, no three.js — also runnable in Node for tuning).
//
// Units: G = 1. One length unit ≈ 3.5 kpc (≈ one disk scale length), one mass unit ≈ 2e11 M☉,
// so one velocity unit ≈ 600 km/s and one time unit ≈ 5.7 Myr.
//
// Each galaxy = Hernquist dark halo + Hernquist bulge + central SMBH (massive, self-gravitating N-body)
// plus massless tracers: exponential stellar disk (old + young arm population), extended gas disk,
// bulge stars and a faint stellar halo. Disk tracers get the circular velocity measured from the
// *actual* softened force of the sampled massive particles, so they start in equilibrium with the
// N-body potential they will feel on the GPU. Spheroids get isotropic Jeans dispersions.

export const KPC_PER_UNIT = 3.5;
export const MYR_PER_UNIT = 5.7;

// Particle type codes (tInfo.x)
export const T_GAS = 0, T_BULGE = 1, T_DISK = 2, T_YOUNG = 3, T_HALO = 4, T_SAT = 5;

export function mulberry32( seed ) {

	let a = seed >>> 0;
	return function () {

		a |= 0; a = a + 0x6D2B79F5 | 0;
		let t = Math.imul( a ^ a >>> 15, 1 | a );
		t = t + Math.imul( t ^ t >>> 7, 61 | t ) ^ t;
		return ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296;

	};

}

export function makeGauss( rng ) {

	let spare = null;
	return function () {

		if ( spare !== null ) {

			const s = spare; spare = null; return s;

		}

		let u, v, s;
		do {

			u = rng() * 2 - 1; v = rng() * 2 - 1; s = u * u + v * v;

		} while ( s >= 1 || s === 0 );

		const m = Math.sqrt( - 2 * Math.log( s ) / s );
		spare = v * m;
		return u * m;

	};

}

// ───────────────────────── galaxy model ─────────────────────────

export function galaxyModel( massScale = 1, ov = {} ) {

	const L = Math.sqrt( massScale ); // size ∝ M^½  →  v ∝ M^¼
	return {
		q: massScale, L,
		halo: { M: 1.0 * massScale, a: ( ov.haloA ?? 2.4 ) * L, rcut: ( ov.haloCut ?? 16 ) * L },
		bulge: { M: 0.13 * massScale, a: 0.2 * L, rcut: 3.0 * L },
		bh: 0.006 * massScale,
		disk: { Rd: 1.0 * L, Rmax: 5.0 * L, z0: 0.07 * L },
		young: { z0: 0.035 * L },
		gas: { Rg: 1.7 * L, Rmax: 7.0 * L, z0: 0.03 * L },
		shalo: { a: 1.6 * L, rcut: 8 * L },
	};

}

// Hernquist enclosed mass (untruncated profile parameterised by total M)
const hernM = ( M, a, r ) => M * r * r / ( ( r + a ) * ( r + a ) );
const hernRho = ( M, a, r ) => M * a / ( 2 * Math.PI * r * Math.pow( r + a, 3 ) );
// inverse CDF with truncation
function hernSampleR( a, rcut, u ) {

	const umax = Math.pow( rcut / ( rcut + a ), 2 );
	const s = Math.sqrt( u * umax );
	return a * s / ( 1 - s );

}

function massEnclosed( g, r ) {

	return hernM( g.halo.M, g.halo.a, Math.min( r, g.halo.rcut ) ) + hernM( g.bulge.M, g.bulge.a, Math.min( r, g.bulge.rcut ) ) + g.bh;

}

function potential( g, r ) {

	return - g.halo.M / ( r + g.halo.a ) - g.bulge.M / ( r + g.bulge.a ) - g.bh / Math.sqrt( r * r + 0.01 );

}

// isotropic Jeans dispersion table σ(r) for a Hernquist component in the total potential
function jeansTable( g, comp ) {

	const n = 400, rmin = 1e-3, rmax = 400;
	const rs = new Float64Array( n ), sig = new Float64Array( n );
	for ( let i = 0; i < n; i ++ ) rs[ i ] = rmin * Math.pow( rmax / rmin, i / ( n - 1 ) );
	// integrate from outside in: I(r) = ∫_r^∞ ρ M / r'^2 dr'
	let I = 0;
	for ( let i = n - 1; i >= 0; i -- ) {

		if ( i < n - 1 ) {

			const r0 = rs[ i ], r1 = rs[ i + 1 ];
			const f0 = hernRho( comp.M, comp.a, r0 ) * massEnclosed( g, r0 ) / ( r0 * r0 );
			const f1 = hernRho( comp.M, comp.a, r1 ) * massEnclosed( g, r1 ) / ( r1 * r1 );
			I += 0.5 * ( f0 + f1 ) * ( r1 - r0 );

		}

		sig[ i ] = Math.sqrt( Math.max( I / hernRho( comp.M, comp.a, rs[ i ] ), 0 ) );

	}

	return ( r ) => {

		if ( r <= rmin ) return sig[ 0 ];
		if ( r >= rmax ) return sig[ n - 1 ];
		const f = Math.log( r / rmin ) / Math.log( rmax / rmin ) * ( n - 1 );
		const i = Math.floor( f ), t = f - i;
		return sig[ i ] * ( 1 - t ) + sig[ i + 1 ] * t;

	};

}

function randDir( rng, out ) {

	const z = rng() * 2 - 1, p = rng() * Math.PI * 2, s = Math.sqrt( 1 - z * z );
	out[ 0 ] = s * Math.cos( p ); out[ 1 ] = z; out[ 2 ] = s * Math.sin( p );
	return out;

}

function sampleIsoVel( gauss, sigma, vesc, out ) {

	for ( let k = 0; k < 20; k ++ ) {

		out[ 0 ] = gauss() * sigma; out[ 1 ] = gauss() * sigma; out[ 2 ] = gauss() * sigma;
		if ( out[ 0 ] ** 2 + out[ 1 ] ** 2 + out[ 2 ] ** 2 < ( 0.92 * vesc ) ** 2 ) break;

	}

	return out;

}

// Massive particles of one galaxy, galaxy frame (disk plane = xz, spin axis = +y).
// Returns { pos: Float32Array(n*4) (xyz, mass), vel: Float32Array(n*4) (xyz, kind) }  kind: 0 bh, 1 halo, 2 bulge
export function sampleMassive( g, n, rng ) {

	const gauss = makeGauss( rng );
	const pos = new Float32Array( n * 4 ), vel = new Float32Array( n * 4 );
	const nb = Math.max( 4, Math.round( ( n - 1 ) * 0.22 ) ), nh = n - 1 - nb;
	const mh = hernM( g.halo.M, g.halo.a, g.halo.rcut ) / nh;
	const mb = hernM( g.bulge.M, g.bulge.a, g.bulge.rcut ) / nb;
	const sH = jeansTable( g, g.halo ), sB = jeansTable( g, g.bulge );
	const d = [ 0, 0, 0 ], v = [ 0, 0, 0 ];
	// 0: SMBH at rest at the centre
	pos[ 3 ] = g.bh; vel[ 3 ] = 0;
	for ( let i = 1; i < n; i ++ ) {

		const isB = i <= nb;
		const comp = isB ? g.bulge : g.halo;
		// stratified radii for a smoother realisation
		const k = isB ? i - 1 : i - 1 - nb, cnt = isB ? nb : nh;
		const u = ( k + rng() ) / cnt;
		const r = hernSampleR( comp.a, comp.rcut, u );
		randDir( rng, d );
		const vesc = Math.sqrt( - 2 * potential( g, r ) );
		sampleIsoVel( gauss, ( isB ? sB : sH )( r ), vesc, v );
		pos.set( [ d[ 0 ] * r, d[ 1 ] * r, d[ 2 ] * r, isB ? mb : mh ], i * 4 );
		vel.set( [ v[ 0 ], v[ 1 ], v[ 2 ], isB ? 2 : 1 ], i * 4 );

	}

	// remove net momentum / centre-of-mass offset of the realisation
	let M = 0; const cm = [ 0, 0, 0 ], cv = [ 0, 0, 0 ];
	for ( let i = 0; i < n; i ++ ) {

		const m = pos[ i * 4 + 3 ]; M += m;
		for ( let c = 0; c < 3; c ++ ) {

			cm[ c ] += m * pos[ i * 4 + c ]; cv[ c ] += m * vel[ i * 4 + c ];

		}

	}

	for ( let i = 0; i < n; i ++ ) for ( let c = 0; c < 3; c ++ ) {

		pos[ i * 4 + c ] -= cm[ c ] / M; vel[ i * 4 + c ] -= cv[ c ] / M;

	}

	return { pos, vel, mass: M, sigmaBulge: sB, sigmaHalo: sH };

}

// Circular velocity in the disk plane from the actual softened particle forces (azimuthally averaged).
export function rotationTable( m, eps, Rmax, nR = 96 ) {

	const vc = new Float32Array( nR );
	const n = m.pos.length / 4, e2 = eps * eps, NA = 12;
	for ( let k = 0; k < nR; k ++ ) {

		const R = Rmax * ( k + 0.5 ) / nR;
		let aR = 0;
		for ( let j = 0; j < NA; j ++ ) {

			const ph = ( j + 0.5 ) / NA * Math.PI * 2;
			const x = R * Math.cos( ph ), z = R * Math.sin( ph );
			let ax = 0, az = 0;
			for ( let i = 0; i < n; i ++ ) {

				const dx = m.pos[ i * 4 ] - x, dy = m.pos[ i * 4 + 1 ], dz = m.pos[ i * 4 + 2 ] - z;
				const r2 = dx * dx + dy * dy + dz * dz + e2;
				const f = m.pos[ i * 4 + 3 ] / ( r2 * Math.sqrt( r2 ) );
				ax += dx * f; az += dz * f;

			}

			aR += - ( ax * Math.cos( ph ) + az * Math.sin( ph ) );

		}

		vc[ k ] = Math.sqrt( Math.max( aR / NA * R, 0 ) );

	}

	return { vc, Rmax, nR, at( R ) {

		const f = Math.min( Math.max( R / Rmax * nR - 0.5, 0 ), nR - 1.001 );
		const i = Math.floor( f ), t = f - i;
		return vc[ i ] * ( 1 - t ) + vc[ i + 1 ] * t;

	} };

}

// exponential disk radius by Newton inversion of 1-(1+x)e^-x = u
function expDiskR( u ) {

	let x = u < 0.5 ? 1.0 : 2.5;
	for ( let k = 0; k < 12; k ++ ) {

		const F = 1 - ( 1 + x ) * Math.exp( - x ) - u;
		const dF = x * Math.exp( - x );
		x = Math.max( 1e-4, x - F / Math.max( dF, 1e-6 ) );

	}

	return x;

}

// Fill tracer particles for one galaxy (galaxy frame) into the given arrays at [offset, offset+count).
// pos4: xyz, w = 0;  vel4: xyz, w = birth time (gas: -1e3 → never burst yet);  info4: (type, galaxy, lum, colour)
export function sampleTracers( g, m, rot, counts, rng, galaxyId, out, seedPhase = 0 ) {

	const gauss = makeGauss( rng );
	const { pos, vel, info } = out;
	let o = counts.offsetGas;
	const pitch = 0.42; // ~24° arms
	const R0 = 0.6 * g.L;
	const armPhase = ( R ) => Math.log( Math.max( R, 0.15 ) / R0 ) / Math.tan( pitch ) + seedPhase;
	const cdfCut = ( Rmax, Rs ) => 1 - ( 1 + Rmax / Rs ) * Math.exp( - Rmax / Rs );

	const putDisk = ( idx, type, Rs, Rmax, z0, armFrac, armW, sigFrac, lum, col ) => {

		const R = expDiskR( rng() * cdfCut( Rmax, Rs ) ) * Rs;
		let ph;
		if ( rng() < armFrac ) {

			ph = armPhase( R ) + ( rng() < 0.5 ? 0 : Math.PI ) + gauss() * armW;

		} else ph = rng() * Math.PI * 2;

		// sech² vertical profile
		const uz = Math.min( Math.max( rng(), 1e-4 ), 1 - 1e-4 );
		const y = z0 * Math.atanh( 2 * uz - 1 );
		const c = Math.cos( ph ), s = Math.sin( ph );
		const vc = rot.at( R );
		const om = vc / Math.max( R, 0.05 );
		const sR = sigFrac * vc * Math.exp( - R / ( 4 * g.disk.Rd ) );
		const vR = gauss() * sR, vT = vc + gauss() * sR * 0.7, vY = gauss() * om * z0 * 0.9;
		// disk in xz plane; spin angular momentum along +y (same sense as the orbit → prograde encounter at incl 0)
		pos.set( [ R * c, y, R * s, 0 ], idx * 4 );
		vel.set( [ vR * c + vT * s, vY, vR * s - vT * c, type === T_GAS ? - 1e3 : 0 ], idx * 4 );
		info.set( [ type, galaxyId, lum, col ], idx * 4 );

	};

	const sB = m.sigmaBulge;
	const v = [ 0, 0, 0 ], d = [ 0, 0, 0 ];
	const putSpheroid = ( idx, type, a, rcut, lum, col, sig ) => {

		const r = hernSampleR( a, rcut, rng() );
		randDir( rng, d );
		const vesc = Math.sqrt( - 2 * potential( g, r ) );
		sampleIsoVel( gauss, sig( r ), vesc, v );
		// a little net rotation for the bulge (real bulges rotate)
		const rr = Math.hypot( d[ 0 ], d[ 2 ] ) * r;
		const vrot = type === T_BULGE ? 0.25 * rot.at( rr ) * Math.min( rr / ( 0.5 * g.L ), 1 ) : 0;
		const ph = Math.atan2( d[ 2 ], d[ 0 ] );
		pos.set( [ d[ 0 ] * r, d[ 1 ] * r * ( type === T_BULGE ? 0.8 : 1 ), d[ 2 ] * r, 0 ], idx * 4 );
		vel.set( [ v[ 0 ] + Math.sin( ph ) * vrot, v[ 1 ], v[ 2 ] - Math.cos( ph ) * vrot, 0 ], idx * 4 );
		info.set( [ type, galaxyId, lum, col ], idx * 4 );

	};

	const lq = g.q; // luminosity scales with galaxy mass
	for ( let i = 0; i < counts.gas; i ++ ) putDisk( o ++, T_GAS, g.gas.Rg, g.gas.Rmax, g.gas.z0, 0.7, 0.32, 0.04, 1.0 * lq, 1 );
	o = counts.offsetStars;
	for ( let i = 0; i < counts.bulge; i ++ ) putSpheroid( o ++, T_BULGE, g.bulge.a, 2.2 * g.L, 0.32 * lq / counts.bulge, 0.18 + rng() * 0.22, sB );
	for ( let i = 0; i < counts.disk; i ++ ) putDisk( o ++, T_DISK, g.disk.Rd, g.disk.Rmax, g.disk.z0, 0.25, 0.6, 0.12, 0.42 * lq / counts.disk, 0.35 + rng() * 0.35 );
	for ( let i = 0; i < counts.young; i ++ ) putDisk( o ++, T_YOUNG, g.disk.Rd * 1.25, g.disk.Rmax, g.young.z0, 0.82, 0.22, 0.05, 0.36 * lq / counts.young, 0.7 + rng() * 0.3 );
	const sHalo = ( r ) => 0.9 * m.sigmaHalo( r ) + 0.1 * sB( r );
	for ( let i = 0; i < counts.shalo; i ++ ) putSpheroid( o ++, T_HALO, g.shalo.a, g.shalo.rcut, 0.05 * lq / counts.shalo, 0.15 + rng() * 0.25, sHalo );

}

// ───────────────────────── orientation / orbit ─────────────────────────

// rotation matrix (row-major 3x3) : tilt spin axis (+y) by incl about x, then rotate by azimuth about y
export function orientation( inclDeg, azDeg ) {

	const i = inclDeg * Math.PI / 180, w = azDeg * Math.PI / 180;
	const ci = Math.cos( i ), si = Math.sin( i ), cw = Math.cos( w ), sw = Math.sin( w );
	// Rx(i) then Ry(w):  M = Ry * Rx
	const Rx = [ 1, 0, 0, 0, ci, - si, 0, si, ci ];
	const Ry = [ cw, 0, sw, 0, 1, 0, - sw, 0, cw ];
	const M = new Array( 9 ).fill( 0 );
	for ( let r = 0; r < 3; r ++ ) for ( let c = 0; c < 3; c ++ ) for ( let k = 0; k < 3; k ++ ) M[ r * 3 + c ] += Ry[ r * 3 + k ] * Rx[ k * 3 + c ];
	return M;

}

export function transformRange( pos, vel, start, count, M, P, V ) {

	for ( let i = start; i < start + count; i ++ ) {

		const o = i * 4;
		const x = pos[ o ], y = pos[ o + 1 ], z = pos[ o + 2 ];
		pos[ o ] = M[ 0 ] * x + M[ 1 ] * y + M[ 2 ] * z + P[ 0 ];
		pos[ o + 1 ] = M[ 3 ] * x + M[ 4 ] * y + M[ 5 ] * z + P[ 1 ];
		pos[ o + 2 ] = M[ 6 ] * x + M[ 7 ] * y + M[ 8 ] * z + P[ 2 ];
		const a = vel[ o ], b = vel[ o + 1 ], c = vel[ o + 2 ];
		vel[ o ] = M[ 0 ] * a + M[ 1 ] * b + M[ 2 ] * c + V[ 0 ];
		vel[ o + 1 ] = M[ 3 ] * a + M[ 4 ] * b + M[ 5 ] * c + V[ 1 ];
		vel[ o + 2 ] = M[ 6 ] * a + M[ 7 ] * b + M[ 8 ] * c + V[ 2 ];

	}

}

// Relative orbit in the xz plane (orbital angular momentum along +y): conic with pericentre q and eccentricity e,
// started at separation r0 on the incoming branch.
export function keplerStart( q, e, r0, mu ) {

	const p = q * ( 1 + e );
	let cosNu = ( p / r0 - 1 ) / e;
	cosNu = Math.max( - 1, Math.min( 1, cosNu ) );
	const nu = - Math.acos( cosNu );
	const h = Math.sqrt( mu * p );
	const vr = mu / h * e * Math.sin( nu ), vt = mu / h * ( 1 + e * Math.cos( nu ) );
	const r = p / ( 1 + e * Math.cos( nu ) );
	// counter-clockwise seen from +y in the x-z plane: position (cos ν, -sin ν) mapping x→x, y→-z
	const rx = Math.cos( nu ), rz = - Math.sin( nu );
	const tx = - Math.sin( nu ), tz = - Math.cos( nu );
	return {
		pos: [ r * rx, 0, r * rz ],
		vel: [ vr * rx + vt * tx, 0, vr * rz + vt * tz ],
	};

}

// ───────────────────────── full scene builder ─────────────────────────
//
// layout (massive): [0] SMBH A, [1] SMBH B, [2..2+nPert) BH perturbers, then satellites (nSat × satM), then galaxy A, galaxy B
// layout (tracers): [0, nGas) gas (A, B, satellites), then stars (A, B, satellites)

export function planLayout( tier, nMassive, nTracers ) {

	const nPert = 4, nSat = 4;
	const satM = tier === 0 ? 12 : tier === 1 ? 32 : 64;
	const satT = Math.max( 256, Math.round( nTracers * 0.008 / 64 ) * 64 );
	const satGas = Math.round( satT * 0.3 );
	const reservedM = 2 + nPert + nSat * satM;
	const perGalM = Math.floor( ( nMassive - reservedM ) / 2 );
	const galT = Math.floor( ( nTracers - nSat * satT ) / 2 );
	const gas = Math.round( galT * 0.26 ), bulge = Math.round( galT * 0.17 ), young = Math.round( galT * 0.2 ), shalo = Math.round( galT * 0.05 );
	const disk = galT - gas - bulge - young - shalo;
	const nGas = 2 * gas + nSat * satGas;
	const L = {
		nMassive, nTracers, nPert, nSat, satM, satT, satGas,
		pertStart: 2, satMStart: 2 + nPert,
		galM: [ reservedM, reservedM + perGalM ], perGalM,
		// massive particles left over by rounding sit idle (mass 0) at the end
		gal: [
			{ gas, bulge, young, shalo, disk, offsetGas: 0, offsetStars: nGas },
			{ gas, bulge, young, shalo, disk, offsetGas: gas, offsetStars: nGas + ( galT - gas ) },
		],
		nGas,
		satGasStart: 2 * gas,
		satStarStart: nGas + 2 * ( galT - gas ),
		used: 2 * galT + nSat * satT,
	};
	return L;

}

export function buildScene( opts ) {

	const { tier, layout: Lyt, massRatio = 1, inclA = 25, azA = - 30, inclB = 60, azB = 70, peri = 2.4, sep = 13, ecc = 1.0, seed = 7, eps = 0.15, vScale = 1 } = opts;
	const rng = mulberry32( seed );
	const mPos = new Float32Array( Lyt.nMassive * 4 ), mVel = new Float32Array( Lyt.nMassive * 4 );
	const tPos = new Float32Array( Lyt.nTracers * 4 ), tVel = new Float32Array( Lyt.nTracers * 4 ), tInfo = new Float32Array( Lyt.nTracers * 4 );

	// park everything unused far away, massless, invisible
	for ( let i = 0; i < Lyt.nMassive; i ++ ) mPos.set( [ 0, - 9000 - i, 0, 0 ], i * 4 );
	for ( let i = 0; i < Lyt.nTracers; i ++ ) {

		tPos.set( [ 0, - 9000, 0, 0 ], i * 4 );
		tInfo.set( [ i < Lyt.nGas ? T_GAS : T_SAT, 9, 0, 1 ], i * 4 );
		tVel[ i * 4 + 3 ] = - 1e3;

	}

	for ( let i = 0; i < Lyt.nMassive; i ++ ) mVel[ i * 4 + 3 ] = 3; // kind 3 = idle

	const gals = [ galaxyModel( 1, opts.model ), galaxyModel( massRatio, opts.model ) ];
	const mass = [];
	const parts = gals.map( ( g, gi ) => {

		const m = sampleMassive( g, Lyt.perGalM, rng );
		const rot = rotationTable( m, eps, g.gas.Rmax * 1.05 );
		mass.push( m.mass );
		return { g, m, rot };

	} );

	// orbit: effective μ from the mass enclosed within the starting separation (halos are extended)
	const muEff = gals.reduce( ( s, g ) => s + hernM( g.halo.M, g.halo.a, Math.min( sep, g.halo.rcut ) ) + g.bulge.M + g.bh, 0 );
	const orb = keplerStart( peri, ecc, sep, muEff );
	orb.vel = orb.vel.map( ( x ) => x * vScale );
	const MA = mass[ 0 ], MB = mass[ 1 ], Mt = MA + MB;
	const PA = orb.pos.map( ( x ) => - x * MB / Mt ), VA = orb.vel.map( ( x ) => - x * MB / Mt );
	const PB = orb.pos.map( ( x ) => x * MA / Mt ), VB = orb.vel.map( ( x ) => x * MA / Mt );
	const rots = [ orientation( inclA, azA ), orientation( inclB, azB ) ];
	const P = [ PA, PB ], V = [ VA, VB ];

	parts.forEach( ( { g, m, rot }, gi ) => {

		// massive: SMBH to slot gi, rest to galaxy block
		const start = Lyt.galM[ gi ];
		mPos.set( m.pos.subarray( 0, 4 ), gi * 4 ); mVel.set( m.vel.subarray( 0, 4 ), gi * 4 );
		mPos.set( m.pos.subarray( 4 ), start * 4 ); mVel.set( m.vel.subarray( 4 ), start * 4 );
		for ( let i = start; i < start + Lyt.perGalM - 1; i ++ ) mVel[ i * 4 + 3 ] += gi * 10; // kind + 10*galaxy
		mVel[ gi * 4 + 3 ] = gi * 10;
		transformRange( mPos, mVel, gi, 1, rots[ gi ], P[ gi ], V[ gi ] );
		transformRange( mPos, mVel, start, Lyt.perGalM - 1, rots[ gi ], P[ gi ], V[ gi ] );

		const c = Lyt.gal[ gi ];
		sampleTracers( g, m, rot, c, rng, gi, { pos: tPos, vel: tVel, info: tInfo }, gi * 1.3 );
		transformRange( tPos, tVel, c.offsetGas, c.gas, rots[ gi ], P[ gi ], V[ gi ] );
		transformRange( tPos, tVel, c.offsetStars, c.bulge + c.disk + c.young + c.shalo, rots[ gi ], P[ gi ], V[ gi ] );

	} );

	// gas luminosity weight normalised per particle
	return { mPos, mVel, tPos, tVel, tInfo, gals, orbit: orb, muEff, spin: rots.map( ( M ) => [ M[ 1 ], M[ 4 ], M[ 7 ] ] ) };

}

// Plummer-sphere dwarf satellite: returns arrays to splice into the buffers (positions relative to centre).
export function sampleSatellite( nM, nT, nGas, M, a, rng ) {

	const gauss = makeGauss( rng );
	const mp = new Float32Array( nM * 4 ), mv = new Float32Array( nM * 4 );
	const tp = new Float32Array( nT * 4 ), tv = new Float32Array( nT * 4 ), ti = new Float32Array( nT * 4 );
	const d = [ 0, 0, 0 ];
	const plum = ( aa, u ) => aa / Math.sqrt( Math.pow( Math.max( u, 1e-4 ) * 0.97, - 2 / 3 ) - 1 );
	const sig = ( r ) => Math.sqrt( M / ( 6 * Math.sqrt( r * r + a * a ) ) );
	for ( let i = 0; i < nM; i ++ ) {

		const r = plum( a, ( i + rng() ) / nM ); randDir( rng, d );
		const s = sig( r );
		mp.set( [ d[ 0 ] * r, d[ 1 ] * r, d[ 2 ] * r, M / nM ], i * 4 );
		mv.set( [ gauss() * s, gauss() * s, gauss() * s, 4 ], i * 4 );

	}

	for ( let i = 0; i < nT; i ++ ) {

		const isGas = i < nGas;
		const r = plum( a * ( isGas ? 1.2 : 0.8 ), rng() ); randDir( rng, d );
		const s = sig( r ) * 0.9;
		tp.set( [ d[ 0 ] * r, d[ 1 ] * r, d[ 2 ] * r, 0 ], i * 4 );
		tv.set( [ gauss() * s, gauss() * s, gauss() * s, isGas ? - 1e3 : 0 ], i * 4 );
		ti.set( [ isGas ? T_GAS : T_SAT, 2, isGas ? 1.0 : 1, 0.75 + rng() * 0.25 ], i * 4 );

	}

	return { mp, mv, tp, tv, ti };

}
