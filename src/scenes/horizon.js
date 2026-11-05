// Chapter 05 — Event Horizon.
//
// A real-time general-relativistic renderer of a Kerr black hole with a thin Novikov–Thorne accretion disk.
//
// Pipeline (all WebGPU compute, reduced resolution, then upscaled):
//   1. trace   — one null geodesic per pixel, RK4 on the Hamiltonian in Kerr–Schild coordinates with a geometric
//                adaptive step.  Every equatorial-plane crossing inside [r_isco, r_out] is shaded (front-to-back,
//                optically thick/thin slab): Page–Thorne temperature, Keplerian-sheared turbulence, injected hot spots,
//                exact redshift g = (p·u_obs)/(p·u_emit) → blackbody LUT evaluated at g·T (⇒ I_ν ∝ g³ beaming built in).
//                Writes disk radiance + transmittance and the asymptotic (lensed) escape direction.
//   2. compose — lensed sky (Milky-Way band + star layers) sampled by the escape direction with a ray-differential
//                footprint taken from the neighbouring rays, composited behind the disk → HDR storage texture.
//   3. streak  — thresholded horizontal glare (anamorphic flare) from the HDR image.
//   post()     — Catmull–Rom upscale + mip-chain bloom + streak; the shell adds grade + AgX.
//
// Units: G = c = M = 1 (1 world unit = 1 M).  Photons are integrated from the camera outward, i.e. time-reversed.
// The time reverse of Kerr(a) is Kerr(−a) (t → −t flips the sign of g_tφ), so the integrator uses a_trace = −a and
// the time-reversed disk; redshift scalars p·u are invariant under the map, so g is exact.

import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, Break, Return, uniform, uniformArray, instancedArray, globalId, vec2, vec3, vec4, uvec2, float, uint,
	texture, textureStore, screenUV, select, mix, clamp, max, abs, sqrt, pow, log, exp, length, normalize, dot,
	smoothstep, atan, floor, round, tanh,
} from 'three/tsl';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { createLUT, writeDiskProfile, LUT_N, T_MIN, T_MAX, R_MAX, relLogLum } from './horizon/blackbody.js';
import { horizonRadius, photonOrbit, staticTetrad, pickDisk, kerrR as kerrRcpu } from './horizon/kerr.js';
import { geoStep, kerrR, escapeDir, diskG, diskTurb, skyColor } from './horizon/shaders.js';

const TIERS = [
	{ scale: 0.5, steps: 150, k: 0.13, label: 'software' },
	{ scale: 0.5, steps: 260, k: 0.09, label: 'mobile' },
	{ scale: 0.72, steps: 420, k: 0.065, label: 'desktop' },
];
const NB = 8; // hot-spot slots
const MAX_PIXELS = 1.7e6;
const DEG = Math.PI / 180;

export default {
	id: 'horizon',
	title: 'Event Horizon',
	subtitle: 'A spinning (Kerr) black hole, ray-traced per pixel: null geodesics integrated in Kerr–Schild coordinates through a Novikov–Thorne disk, with exact Doppler and gravitational redshift, photon ring and a lensed sky.',

	async create( ctx ) {

		const { renderer, camera, controls } = ctx;
		// The shell's adapter sniffing can miss SwiftShader in headless Chromium (empty adapter info), so an
		// un-overridden headless capture is treated as the software tier.
		const tierIdx = Math.max( 0, Math.min( 2, ctx.params.has( 'tier' ) ? ctx.tier : ( ctx.shot || ctx.isSoftware ? 0 : ctx.tier ) ) );
		const tier = TIERS[ tierIdx ];
		const scene = new THREE.Scene(); // everything is drawn by compute + post(); the scene pass stays unused

		// ───────────────── camera ─────────────────
		camera.fov = 36; camera.near = 0.1; camera.far = 1000; camera.updateProjectionMatrix();
		{

			const dist = 32, el = 7 * DEG, az = 0.55;
			camera.position.set( Math.sin( az ) * Math.cos( el ) * dist, Math.sin( el ) * dist, Math.cos( az ) * Math.cos( el ) * dist );

		}

		const ctlSaved = {
			minDistance: controls.minDistance, maxDistance: controls.maxDistance, enablePan: controls.enablePan,
			autoRotateSpeed: controls.autoRotateSpeed, rotateSpeed: controls.rotateSpeed, zoomSpeed: controls.zoomSpeed,
		};
		controls.target.set( 0, 0, 0 );
		controls.minDistance = 4.5; // r ≳ 4.4 M: well outside the horizon and the ergosphere (static observer exists)
		controls.maxDistance = 70;
		controls.enablePan = false;
		controls.autoRotate = true;
		controls.autoRotateSpeed = - 0.35; // slow drift (~170 s / orbit)
		controls.rotateSpeed = 0.5;
		controls.zoomSpeed = 0.6;
		controls.update();

		// ───────────────── parameters ─────────────────
		const P = {
			spin: 0.9,
			tilt: 0,
			tPeak: 7000,
			redshift: 2,
			quality: tier.k,
			exposure: 0.8,
			rOut: 20,
			timeScale: 10,
			stars: 1.0,
			corona: 0.12,
		};

		const dbgNum = ( k, d ) => ( ctx.params.has( k ) ? parseFloat( ctx.params.get( k ) ) : d );
		P.corona = dbgNum( 'hzcorona', P.corona ); P.tPeak = dbgNum( 'hzT', P.tPeak ); P.exposure = dbgNum( 'hzexp', P.exposure );
		P.spin = dbgNum( 'hzspin', P.spin ); P.redshift = dbgNum( 'hzmode', P.redshift ); P.tilt = dbgNum( 'hztilt', P.tilt );
		const bloomStrength = dbgNum( 'hzbloom', 0.3 );

		const U = {
			W: uniform( 1, 'uint' ), H: uniform( 1, 'uint' ),
			uvScale: uniform( new THREE.Vector2( 1, 1 ) ),
			texSize: uniform( new THREE.Vector2( 1, 1 ) ),
			tan: uniform( new THREE.Vector2( 1, 1 ) ),
			cam: uniform( new THREE.Vector3() ),
			tU: uniform( new THREE.Vector4() ), tR: uniform( new THREE.Vector4() ),
			tUp: uniform( new THREE.Vector4() ), tF: uniform( new THREE.Vector4() ),
			aT: uniform( - P.spin ), aP: uniform( P.spin ),
			k: uniform( P.quality ), rp: uniform( horizonRadius( P.spin ) ), rEsc: uniform( 60 ),
			rIsco: uniform( 2.32 ), rOut: uniform( P.rOut ), steps: uniform( tier.steps, 'int' ),
			time: uniform( 0 ), tPeak: uniform( P.tPeak ), lum: uniform( 1 ), mode: uniform( P.redshift ),
			tau: uniform( 1.1 ), exposure: uniform( P.exposure ), b2w: uniform( new THREE.Matrix3() ),
			pixAng: uniform( 0.003 ), sky: uniform( new THREE.Vector4( 0.12, 1, 0, 0 ) ),
			tInsp: uniform( 650 ), period: uniform( 0.75 ), streak: uniform( 0.022 ), streakStride: uniform( 0.004 ),
			ilStep: uniform( 1, 'uint' ), ilOff: uniform( new THREE.Vector2( 0, 0 ) ),
			corona: uniform( P.corona ),
			// a bright blue giant almost exactly behind the hole along the default orbit: watch it split into two
			// images and close into an Einstein ring as the camera drifts past the alignment
			beacon: uniform( new THREE.Vector4( - Math.sin( 0.84 ) * Math.cos( 7 * DEG ), - Math.sin( 7 * DEG ), - Math.cos( 0.84 ) * Math.cos( 7 * DEG ), 4.0e-4 ) ),
		};
		// tier 0 (software rasteriser): trace a 2×2-interleaved quarter of the pixels per frame (progressive refinement)
		const IL = tierIdx === 0 ? 2 : 1;
		U.ilStep.value = IL;
		let ilFrame = 0;
		const blobs = Array.from( { length: NB }, () => new THREE.Vector4( 0, 0, 0, 0 ) );
		const uBlobs = uniformArray( blobs, 'vec4' );

		// ───────────────── LUT (blackbody + disk profile) ─────────────────
		const lut = createLUT();
		let rIsco = 0, rPeak = 0;
		const applySpin = () => {

			const a = THREE.MathUtils.clamp( P.spin, - 0.998, 0.998 );
			U.aP.value = a; U.aT.value = - a;
			U.rp.value = horizonRadius( a );
			( { rIsco, rPeak } = writeDiskProfile( lut, a ) );
			U.rIsco.value = rIsco;

		};

		applySpin();
		const applyTemp = () => {

			U.tPeak.value = P.tPeak;
			U.lum.value = 1.6 / Math.pow( 10, relLogLum( P.tPeak ) );

		};

		applyTemp();

		// ───────────────── GPU resources ─────────────────
		const scr = window.screen || { width: 0, height: 0 };
		let capW = Math.ceil( Math.max( scr.width || 0, innerWidth, ctx.size.x ) * tier.scale );
		let capH = Math.ceil( Math.max( scr.height || 0, innerHeight, ctx.size.y ) * tier.scale );
		const capFit = Math.min( 1, Math.sqrt( MAX_PIXELS / ( capW * capH ) ) );
		capW = Math.ceil( capW * capFit / 8 ) * 8; capH = Math.ceil( capH * capFit / 8 ) * 8;
		const cap = capW * capH;

		const bufCol = instancedArray( cap, 'vec4' );
		const bufDir = instancedArray( cap, 'vec4' );
		const mkTex = () => {

			const t = new THREE.StorageTexture( capW, capH );
			t.type = THREE.HalfFloatType;
			t.format = THREE.RGBAFormat;
			t.generateMipmaps = false;
			t.mipmapsAutoUpdate = false;
			t.minFilter = THREE.LinearFilter; t.magFilter = THREE.LinearFilter;
			t.colorSpace = THREE.NoColorSpace;
			return t;

		};

		const outTex = mkTex();
		const streakTex = mkTex();
		U.texSize.value.set( capW, capH );

		let W = 1, H = 1;
		const fitRes = () => {

			let w = Math.max( 16, Math.round( ctx.size.x * tier.scale ) );
			let h = Math.max( 16, Math.round( ctx.size.y * tier.scale ) );
			const s = Math.min( 1, capW / w, capH / h );
			W = Math.max( 8, Math.floor( w * s ) ); H = Math.max( 8, Math.floor( h * s ) );
			U.W.value = W; U.H.value = H;
			U.uvScale.value.set( W / capW, H / capH );
			U.streakStride.value = ( W / 110 ) / capW;

		};

		fitRes();

		// ───────────────── shading helpers (TSL) ─────────────────
		const lnTmin = Math.log( T_MIN ), lnTr = Math.log( T_MAX ) - lnTmin;
		const blackbody = ( T ) => {

			const s = clamp( log( max( T, 1.0 ) ).sub( lnTmin ).div( lnTr ), 0, 1 );
			const tx = texture( lut, vec2( s.mul( ( LUT_N - 1 ) / LUT_N ).add( 0.5 / LUT_N ), 0.25 ) ).level( 0 );
			return tx.rgb.mul( exp( tx.a.mul( Math.LN10 ) ).mul( U.lum ) );

		};

		const diskProfile = ( r ) => {

			const u = sqrt( clamp( r.sub( U.rIsco ).div( float( R_MAX ).sub( U.rIsco ) ), 0, 1 ) );
			return texture( lut, vec2( u.mul( ( LUT_N - 1 ) / LUT_N ).add( 0.5 / LUT_N ), 0.75 ) ).level( 0 ).r;

		};

		// ───────────────── kernel 1: geodesic trace ─────────────────
		const ilX = () => globalId.x.mul( U.ilStep ).add( uint( U.ilOff.x ) );
		const ilY = () => globalId.y.mul( U.ilStep ).add( uint( U.ilOff.y ) );

		const traceKernel = Fn( () => {

			const gx = ilX().toVar( 'gx' ), gy = ilY().toVar( 'gy' );
			If( gx.greaterThanEqual( U.W ).or( gy.greaterThanEqual( U.H ) ), () => {

				Return();

			} );
			const idx = gy.mul( U.W ).add( gx );

			const sx = float( gx ).add( 0.5 ).div( float( U.W ) ).mul( 2 ).sub( 1 );
			const sy = float( 1 ).sub( float( gy ).add( 0.5 ).div( float( U.H ) ).mul( 2 ) );
			const n = normalize( vec3( sx.mul( U.tan.x ), sy.mul( U.tan.y ), 1 ) );
			// p_μ = u_μ + nⁱ e_iμ  (covariant, xyz = spatial, w = t)
			const p4 = U.tU.add( U.tR.mul( n.x ) ).add( U.tUp.mul( n.y ) ).add( U.tF.mul( n.z ) );
			const E = p4.w.negate().toVar( 'E' );
			const x = vec3( U.cam ).toVar( 'x' );
			const p = p4.xyz.toVar( 'p' );
			const Lz = x.x.mul( p.y ).sub( x.y.mul( p.x ) ).toVar( 'Lz' );
			const prm = vec4( U.aT, E, U.k, U.rp ).toVar( 'prm' );

			const acc = vec3( 0 ).toVar( 'acc' );
			const trans = float( 1 ).toVar( 'trans' );
			const esc = float( 0 ).toVar( 'esc' );
			const vel = vec3( 0, 0, 1 ).toVar( 'vel' );

			Loop( U.steps, () => {

				const res = geoStep( x, p, prm ).toVar( 'res' );
				const xn = res.element( 0 ).toVar( 'xn' );
				const rn = res.element( 2 ).x;

				// ── equatorial plane crossing → thin disk ──
				If( x.z.mul( xn.z ).lessThan( 0 ), () => {

					const tc = x.z.div( x.z.sub( xn.z ) );
					const xh = mix( x, xn, tc ).toVar( 'xh' );
					const rh = kerrR( xh, U.aT ).toVar( 'rh' );

					If( rh.greaterThan( U.rIsco ).and( rh.lessThan( U.rOut ) ), () => {

						const seg = xn.sub( x );
						const mu = max( abs( seg.z ).div( length( seg ) ), 0.02 );
						const phi = atan( xh.y, xh.x ).toVar( 'phi' );
						const g = diskG( rh, E, Lz, U.aP, U.mode ).toVar( 'g' );
						const turb = diskTurb( rh, phi, U.time, U.aP, U.period ).toVar( 'turb' );
						const prof = diskProfile( rh );
						const Td = U.tPeak.mul( prof ).mul( turb.mul( 0.35 ).add( 0.82 ) );
						const edge = smoothstep( U.rIsco, U.rIsco.mul( 1.18 ), rh ).mul( smoothstep( U.rOut, U.rOut.mul( 0.55 ), rh ) );
						const tauD = U.tau.mul( edge ).mul( smoothstep( 0.18, 0.78, turb ).mul( 1.8 ).add( 0.06 ) ).toVar( 'tauD' );

						// hot spots: Lagrangian labels (R, Φ) of gas that drifts as d(r³)/dτ = −c and rotates with Ω ≈ r^{-3/2}
						const tauB = float( 0 ).toVar( 'tauB' );
						Loop( NB, ( { i } ) => {

							const b = uBlobs.element( i );
							If( b.w.greaterThan( 0.001 ), () => {

								const c = b.x.mul( b.x ).mul( b.x ).div( U.tInsp );
								const age = max( U.time.sub( b.z ), 0 );
								const R = pow( rh.mul( rh ).mul( rh ).add( c.mul( age ) ), 1 / 3 );
								const ph = phi.sub( R.mul( sqrt( R ) ).sub( rh.mul( sqrt( rh ) ) ).mul( 2 ).div( c ) ).sub( b.y );
								const dph = ph.sub( round( ph.div( 2 * Math.PI ) ).mul( 2 * Math.PI ) );
								const dR = R.sub( b.x ).div( 0.6 );
								const dA = dph.mul( b.x ).div( 1.3 );
								tauB.addAssign( b.w.mul( exp( dR.mul( dR ).add( dA.mul( dA ) ).negate() ) ) );

							} );

						} );

						const tauBb = tauB.mul( 3.0 );
						const tauT = tauD.add( tauBb );
						const alpha = float( 1 ).sub( exp( tauT.div( mu ).negate() ) );
						const Tb = U.tPeak.mul( 1.45 ).mul( pow( U.rIsco.mul( 2.2 ).div( rh ), 0.5 ) );
						const S = blackbody( Td.mul( g ) ).mul( tauD ).add( blackbody( Tb.mul( g ) ).mul( tauBb ) ).div( max( tauT, 1e-5 ) );
						acc.addAssign( S.mul( alpha ).mul( trans ) );
						trans.mulAssign( float( 1 ).sub( alpha ) );

					} );

				} );

				// ── optically thin warm corona / thick-disk atmosphere: Gaussian layer of scale height H = h·r,
				//    integrated exactly (erf) along the straight chord of this step so coarse steps cannot band ──
				If( U.corona.greaterThan( 0 ), () => {

					const mid = x.add( xn ).mul( 0.5 );
					const rc = length( mid.xy );
					const Hs = rc.mul( 0.11 ).add( 0.25 );
					If( abs( mid.z ).lessThan( Hs.mul( 3.5 ) ).and( rc.greaterThan( U.rIsco.mul( 1.1 ) ) ).and( rc.lessThan( U.rOut ) ), () => {

						const seg = xn.sub( x );
						const ds = length( seg );
						const inv = float( 1 ).div( Hs.mul( Math.SQRT2 ) );
						const z0 = x.z.mul( inv ), z1 = xn.z.mul( inv );
						const dz = z1.sub( z0 );
						const erfA = ( v ) => tanh( v.mul( v.mul( v ).mul( 0.1009 ).add( 1.1283791 ) ) );
						const exact = erfA( z1 ).sub( erfA( z0 ) ).div( dz ).mul( 0.8862269 );
						const colInt = select( abs( dz ).greaterThan( 1e-3 ), exact, exp( z0.mul( z0 ).negate() ) ).mul( ds ); // ∫ exp(−z²/2H²) ds
						const prof = smoothstep( U.rIsco.mul( 1.1 ), U.rIsco.mul( 2.6 ), rc ).mul( pow( U.rIsco.mul( 2.6 ).div( rc ), 1.3 ) ).mul( smoothstep( U.rOut, U.rOut.mul( 0.5 ), rc ) );
						const gC = diskG( max( rc, U.rIsco ), E, Lz, U.aP, U.mode );
						const Tc = U.tPeak.mul( diskProfile( max( rc, U.rIsco.mul( 1.3 ) ) ) ).mul( 1.15 );
						const dtau = colInt.mul( prof ).mul( U.corona );
						acc.addAssign( blackbody( Tc.mul( gC ) ).mul( dtau ).mul( trans ) );
						trans.mulAssign( exp( dtau.mul( - 0.25 ) ) );

					} );

				} );

				vel.assign( xn.sub( x ) );
				x.assign( xn );
				p.assign( res.element( 1 ) );

				If( rn.lessThan( U.rp.mul( 1.005 ) ), () => {

					Break();

				} );
				If( rn.greaterThan( U.rEsc ).and( dot( x, vel ).greaterThan( 0 ) ), () => {

					esc.assign( 1 );
					Break();

				} );
				If( trans.lessThan( 0.003 ), () => {

					Break();

				} );

			} );

			bufCol.element( idx ).assign( vec4( acc, trans ) );
			bufDir.element( idx ).assign( vec4( escapeDir( x, vel ), esc ) );

		} )().computeKernel( [ 8, 8, 1 ] );

		// ───────────────── kernel 2: lensed sky + composite ─────────────────
		const composeKernel = Fn( () => {

			const gx = ilX().toVar( 'gx' ), gy = ilY().toVar( 'gy' );
			If( gx.greaterThanEqual( U.W ).or( gy.greaterThanEqual( U.H ) ), () => {

				Return();

			} );
			const idx = gy.mul( U.W ).add( gx );
			const c = bufCol.element( idx ).toVar( 'c' );
			const d = bufDir.element( idx ).toVar( 'd' );
			const sky = vec3( 0 ).toVar( 'sky' );

			If( d.w.greaterThan( 0.5 ).and( c.w.greaterThan( 0.002 ) ), () => {

				const nb = ( j ) => {

					const e = bufDir.element( j );
					return select( e.w.greaterThan( 0.5 ), length( e.xyz.sub( d.xyz ) ), 0.0 );

				};

				const iL = select( gx.greaterThan( 0 ), idx.sub( 1 ), idx );
				const iR = select( gx.add( 1 ).lessThan( U.W ), idx.add( 1 ), idx );
				const iU = select( gy.greaterThan( 0 ), idx.sub( U.W ), idx );
				const iD = select( gy.add( 1 ).lessThan( U.H ), idx.add( U.W ), idx );
				const fp = max( max( nb( iL ), nb( iR ) ), max( nb( iU ), nb( iD ) ) ).toVar( 'fp' );
				fp.assign( select( fp.greaterThan( 0 ), fp, U.pixAng.mul( 3 ) ) );
				fp.assign( clamp( fp, U.pixAng.mul( 0.08 ), 0.6 ) );
				const dw = normalize( U.b2w.mul( d.xyz ) );
				sky.assign( skyColor( dw, fp, U.sky, U.beacon ) );

			} );

			const col = c.xyz.add( sky.mul( c.w ) ).mul( U.exposure );
			textureStore( outTex, uvec2( gx, gy ), vec4( col, 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] );

		// ───────────────── kernel 3: anamorphic streak ─────────────────
		const streakKernel = Fn( () => {

			// half resolution: streak texel (gx, gy) covers output pixels 2gx..2gx+1
			const gx = globalId.x, gy = globalId.y;
			If( gx.mul( 2 ).greaterThanEqual( U.W ).or( gy.mul( 2 ).greaterThanEqual( U.H ) ), () => {

				Return();

			} );
			const uv0 = vec2( float( gx ).mul( 2 ).add( 1 ), float( gy ).mul( 2 ).add( 1 ) ).div( U.texSize );
			const xmax = U.uvScale.x.sub( float( 0.5 ).div( U.texSize.x ) );
			const acc = vec3( 0 ).toVar( 'acc' );
			Loop( { start: - 20, end: 21, type: 'int' }, ( { i } ) => {

				const fi = float( i );
				const w = exp( abs( fi ).mul( - 0.16 ) );
				const uvs = vec2( clamp( uv0.x.add( fi.mul( U.streakStride ) ), 0, xmax ), uv0.y );
				const s = texture( outTex, uvs ).level( 0 ).xyz;
				const lum = dot( s, vec3( 0.2126, 0.7152, 0.0722 ) );
				acc.addAssign( s.mul( max( lum.sub( 1.8 ), 0 ).div( max( lum, 1e-4 ) ) ).mul( w ) );

			} );
			textureStore( streakTex, uvec2( gx, gy ), vec4( acc.mul( 0.16 ), 1 ) );

		} )().computeKernel( [ 8, 8, 1 ] );

		// ───────────────── post: Catmull–Rom upscale + bloom + streak ─────────────────
		let bloomNode = null;
		const catmullRom = ( tex, uvIn ) => {

			const size = U.texSize;
			const lo = vec2( 0.5 ), hi = U.uvScale.mul( size ).sub( 0.5 );
			const sp = uvIn.mul( size );
			const t1 = floor( sp.sub( 0.5 ) ).add( 0.5 );
			const f = sp.sub( t1 );
			const w0 = f.mul( f.mul( f.mul( - 0.5 ).add( 1.0 ) ).sub( 0.5 ) );
			const w1 = f.mul( f ).mul( f.mul( 1.5 ).sub( 2.5 ) ).add( 1.0 );
			const w2 = f.mul( f.mul( f.mul( - 1.5 ).add( 2.0 ) ).add( 0.5 ) );
			const w3 = f.mul( f ).mul( f.mul( 0.5 ).sub( 0.5 ) );
			const w12 = w1.add( w2 );
			const o12 = w2.div( w12 );
			const p0 = clamp( t1.sub( 1 ), lo, hi ).div( size );
			const p3 = clamp( t1.add( 2 ), lo, hi ).div( size );
			const p12 = clamp( t1.add( o12 ), lo, hi ).div( size );
			const s = ( u, v ) => texture( tex, vec2( u, v ) ).xyz;
			let r = s( p0.x, p0.y ).mul( w0.x.mul( w0.y ) );
			r = r.add( s( p12.x, p0.y ).mul( w12.x.mul( w0.y ) ) );
			r = r.add( s( p3.x, p0.y ).mul( w3.x.mul( w0.y ) ) );
			r = r.add( s( p0.x, p12.y ).mul( w0.x.mul( w12.y ) ) );
			r = r.add( s( p12.x, p12.y ).mul( w12.x.mul( w12.y ) ) );
			r = r.add( s( p3.x, p12.y ).mul( w3.x.mul( w12.y ) ) );
			r = r.add( s( p0.x, p3.y ).mul( w0.x.mul( w3.y ) ) );
			r = r.add( s( p12.x, p3.y ).mul( w12.x.mul( w3.y ) ) );
			r = r.add( s( p3.x, p3.y ).mul( w3.x.mul( w3.y ) ) );
			return max( r, vec3( 0 ) );

		};

		const post = () => {

			const uvT = screenUV.mul( U.uvScale );
			const base = catmullRom( outTex, uvT );
			bloomNode = bloom( texture( outTex, uvT ), bloomStrength, 0.5, 0.85 );
			const st = texture( streakTex, uvT.mul( 0.5 ) ).xyz.mul( U.streak.mul( 16.0 ) ).mul( vec3( 0.75, 0.85, 1.15 ) );
			return vec4( base.add( bloomNode.xyz ).add( st ), 1 );

		};

		// ───────────────── GUI ─────────────────
		const gui = ctx.gui;
		gui.add( P, 'spin', - 0.998, 0.998, 0.001 ).name( 'spin a / M' ).onChange( applySpin );
		gui.add( P, 'tilt', - 75, 75, 1 ).name( 'disk tilt (°)' );
		gui.add( P, 'tPeak', 2500, 30000, 100 ).name( 'disk peak T (K)' ).onChange( applyTemp );
		gui.add( P, 'redshift', { 'full GR (Doppler + gravity)': 2, 'gravity + time dilation only': 1, 'off (rest-frame colour)': 0 } ).name( 'redshift' ).onChange( ( v ) => U.mode.value = Number( v ) );
		gui.add( P, 'quality', 0.03, 0.2, 0.005 ).name( 'RK4 step (k·r)' ).onChange( ( v ) => U.k.value = v );
		gui.add( P, 'exposure', 0.1, 5, 0.01 ).name( 'exposure' ).onChange( ( v ) => U.exposure.value = v );
		gui.add( P, 'rOut', 8, 40, 0.5 ).name( 'disk outer r (M)' ).onChange( ( v ) => U.rOut.value = v );
		gui.add( P, 'corona', 0, 1.5, 0.01 ).name( 'corona (thick layer)' ).onChange( ( v ) => U.corona.value = v );
		gui.add( P, 'timeScale', 0, 40, 0.5 ).name( 'time (M / s)' );
		const actions = { drop: () => spawnRandom() };
		gui.add( actions, 'drop' ).name( 'inject plasma blob' );

		// ───────────────── hot spots (pointer) ─────────────────
		let tM = 0; // coordinate time in M
		const slots = Array.from( { length: NB }, () => null );
		const spawn = ( r, phi, age = 0 ) => {

			let k = slots.findIndex( ( s ) => s === null );
			if ( k < 0 ) k = slots.reduce( ( best, s, i ) => ( s.t0 < slots[ best ].t0 ? i : best ), 0 );
			slots[ k ] = { R0: r, phi0: phi, t0: tM - age };

		};

		const spawnRandom = () => {

			// on the near side of the disk, as seen from the camera
			const cp = camBH;
			const phiCam = Math.atan2( cp[ 1 ], cp[ 0 ] );
			spawn( THREE.MathUtils.lerp( rIsco * 2.2 + 2, Math.min( P.rOut * 0.75, 16 ), Math.random() ), phiCam + ( Math.random() - 0.5 ) * 2.0 );

		};

		const updateBlobs = () => {

			const tInsp = U.tInsp.value;
			for ( let i = 0; i < NB; i ++ ) {

				const s = slots[ i ];
				if ( ! s ) {

					blobs[ i ].set( 0, 0, 0, 0 ); continue;

				}

				const R3 = s.R0 ** 3, c = R3 / tInsp;
				const age = tM - s.t0;
				const tEnd = ( R3 - Math.pow( rIsco * 1.05, 3 ) ) / c;
				const w = THREE.MathUtils.smoothstep( age, 0, 12 ) * ( 1 - THREE.MathUtils.smoothstep( age, tEnd - 60, tEnd ) );
				if ( age > tEnd || s.R0 > P.rOut || s.R0 < rIsco ) {

					slots[ i ] = null; blobs[ i ].set( 0, 0, 0, 0 ); continue;

				}

				blobs[ i ].set( s.R0, s.phi0, s.t0, w );

			}

		};

		let pendingClick = null, downAt = null;
		const el = renderer.domElement;
		const onDown = ( e ) => {

			downAt = { x: e.clientX, y: e.clientY, t: performance.now() };

		};

		const onUp = ( e ) => {

			if ( ! downAt ) return;
			const moved = Math.hypot( e.clientX - downAt.x, e.clientY - downAt.y );
			if ( moved < 6 && performance.now() - downAt.t < 450 ) {

				const rect = el.getBoundingClientRect();
				pendingClick = [ ( ( e.clientX - rect.left ) / rect.width ) * 2 - 1, - ( ( e.clientY - rect.top ) / rect.height ) * 2 + 1 ];

			}

			downAt = null;

		};

		el.addEventListener( 'pointerdown', onDown );
		el.addEventListener( 'pointerup', onUp );

		// ───────────────── per-frame camera → tetrad ─────────────────
		const camBH = [ 0, 0, 0 ];
		let tet = null;
		const ex = new THREE.Vector3(), ey = new THREE.Vector3(), ez = new THREE.Vector3();
		const vR = new THREE.Vector3(), vU = new THREE.Vector3(), vB = new THREE.Vector3(), vP = new THREE.Vector3();
		const toBH = ( v ) => [ v.dot( ex ), v.dot( ey ), v.dot( ez ) ];

		const updateCamera = () => {

			const b = P.tilt * DEG;
			ex.set( 1, 0, 0 );
			ez.set( 0, Math.cos( b ), Math.sin( b ) );
			ey.crossVectors( ez, ex );
			U.b2w.value.set( ex.x, ey.x, ez.x, ex.y, ey.y, ez.y, ex.z, ey.z, ez.z );

			camera.updateMatrixWorld();
			const m = camera.matrixWorld;
			vR.setFromMatrixColumn( m, 0 ).normalize();
			vU.setFromMatrixColumn( m, 1 ).normalize();
			vB.setFromMatrixColumn( m, 2 ).normalize();
			vP.setFromMatrixPosition( m );
			const pos = toBH( vP ), fwd = toBH( vB.clone().negate() ), up = toBH( vU ), right = toBH( vR );
			camBH[ 0 ] = pos[ 0 ]; camBH[ 1 ] = pos[ 1 ]; camBH[ 2 ] = pos[ 2 ];
			tet = staticTetrad( pos, fwd, up, right, U.aT.value );
			const set4 = ( u, v ) => u.value.set( v[ 1 ], v[ 2 ], v[ 3 ], v[ 0 ] );
			set4( U.tU, tet.u ); set4( U.tR, tet.eR ); set4( U.tUp, tet.eU ); set4( U.tF, tet.eF );
			U.cam.value.set( pos[ 0 ], pos[ 1 ], pos[ 2 ] );
			const ty = Math.tan( camera.fov * DEG / 2 );
			U.tan.value.set( ty * camera.aspect, ty );
			U.pixAng.value = 2 * ty / H;
			const rc = kerrRcpu( pos[ 0 ], pos[ 1 ], pos[ 2 ], U.aT.value );
			U.rEsc.value = Math.max( 1.6 * rc, 60 );
			return rc;

		};

		const doPick = ( ndc ) => {

			const ty = Math.tan( camera.fov * DEG / 2 );
			let nx = ndc[ 0 ] * ty * camera.aspect, ny = ndc[ 1 ] * ty, nz = 1;
			const l = Math.hypot( nx, ny, nz ); nx /= l; ny /= l; nz /= l;
			const p = [ 0, 1, 2, 3 ].map( ( i ) => tet.u[ i ] + nx * tet.eR[ i ] + ny * tet.eU[ i ] + nz * tet.eF[ i ] );
			const hit = pickDisk( camBH, [ p[ 1 ], p[ 2 ], p[ 3 ] ], - p[ 0 ], U.aT.value, U.rp.value, rIsco * 1.3, P.rOut );
			if ( hit ) spawn( THREE.MathUtils.clamp( hit.r, rIsco * 1.6, P.rOut * 0.9 ), hit.phi );

		};

		// a couple of blobs already in flight so the first frame tells the story
		updateCamera();
		{

			const phiCam = Math.atan2( camBH[ 1 ], camBH[ 0 ] );
			spawn( 11.5, phiCam - 0.9, 120 );
			spawn( 15, phiCam + 2.6, 40 );

		}

		// warm-up compile (async so the first frame does not stall the page)
		await renderer.computeAsync( traceKernel, [ 1, 1, 1 ] );
		await renderer.computeAsync( composeKernel, [ 1, 1, 1 ] );
		await renderer.computeAsync( streakKernel, [ 1, 1, 1 ] );

		let statT = 0;
		const dbg = ctx.params.has( 'hzdebug' );
		const shotFrames = parseInt( ctx.params.get( 'frames' ) || '90' );
		let shotUpdates = 0;
		if ( dbg ) console.warn( `[horizon] tier ${tierIdx} (shell ${ctx.tier}) trace ${W}x${H} cap ${capW}x${capH}` );
		let lastT = performance.now();

		return {
			scene,
			post: () => post(),

			update( dt, t ) {

				// headless capture: the shell keeps calling update() while it waits for the read-back; stop queueing
				// GPU work once the requested number of frames has been submitted (SwiftShader would never catch up)
				if ( ctx.shot && ++ shotUpdates > shotFrames ) return;
				tM += dt * P.timeScale;
				U.time.value = tM;
				const rc = updateCamera();
				if ( pendingClick ) {

					doPick( pendingClick ); pendingClick = null;

				}

				updateBlobs();

				if ( IL > 1 ) {

					const k = ilFrame ++ % 4;
					U.ilOff.value.set( [ 0, 1, 1, 0 ][ k ], [ 0, 1, 0, 1 ][ k ] );

				}

				const disp = [ Math.ceil( W / IL / 8 ), Math.ceil( H / IL / 8 ), 1 ];
				const t0 = dbg ? performance.now() : 0;
				renderer.compute( traceKernel, disp );
				renderer.compute( composeKernel, disp );
				renderer.compute( streakKernel, [ Math.ceil( W / 16 ), Math.ceil( H / 16 ), 1 ] );
				if ( dbg ) console.warn( `[horizon] frame ${ilFrame} cpu ${( performance.now() - t0 ).toFixed( 1 )} ms, dt ${( performance.now() - lastT ).toFixed( 0 )} ms` );
				if ( dbg ) lastT = performance.now();

				statT -= dt;
				if ( statT <= 0 ) {

					statT = 0.5;
					const a = U.aP.value;
					ctx.setStats( [
						[ 'metric', 'Kerr · Kerr–Schild' ],
						[ 'spin a', a.toFixed( 3 ) ],
						[ 'r obs', rc.toFixed( 1 ) + ' M' ],
						[ 'r₊ / isco', `${U.rp.value.toFixed( 2 )} / ${rIsco.toFixed( 2 )} M` ],
						[ 'photon orbit', photonOrbit( a ).toFixed( 2 ) + ' M' ],
						[ 'steps', `≤ ${U.steps.value} RK4` ],
						[ 'trace', `${W}×${H}` ],
					] );

				}

			},

			resize() {

				fitRes();

			},

			dispose() {

				el.removeEventListener( 'pointerdown', onDown );
				el.removeEventListener( 'pointerup', onUp );
				controls.autoRotate = false;
				Object.assign( controls, ctlSaved );
				traceKernel.dispose(); composeKernel.dispose(); streakKernel.dispose();
				bufCol.value.dispose(); bufDir.value.dispose();
				outTex.dispose(); streakTex.dispose(); lut.dispose();
				bloomNode?.dispose();

			},
		};

	},
};
