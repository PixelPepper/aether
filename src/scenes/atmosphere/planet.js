// Planet surface, cloud shell and final composition for the Atmosphere chapter.
// One full-screen pass ray-traces the planet analytically (sphere intersections in km, planet-centred), so the
// camera can fly from orbit down to ~5 km without depth-buffer or scale problems.
import * as THREE from 'three/webgpu';
import {
	Fn, float, vec2, vec3, vec4, uvec2, uvec3, texture, texture3D, textureStore, instanceIndex, Loop, If, Break, int,
	select, exp, sqrt, max, min, abs, dot, normalize, length, cos, sin, asin, atan, cross, smoothstep, mix, floor, fract, pow,
	varying, positionGeometry, screenCoordinate,
} from 'three/tsl';
import { RB, RT, raySphere, hgPhase } from './atmo.js';
import { fbm, ridged, gnoise, fbmd, tgnoise, tworley, hash13, hash33, remap, interleavedGradientNoise } from './noise.js';

const PI = Math.PI;
export const CLOUD_BOTTOM = RB + 1.2;
export const CLOUD_TOP = RB + 9.0;
const ELEV_KM = 6.0; //       map height unit → km
const NOISE_KM = 36.0; //     tile size of the cloud detail volume
const SUN_ANG_R = 0.004675; // solar angular radius (rad)
const SUN_RADIANCE = 2600.0; // disk radiance relative to unit illuminance (physical ≈ 14600; tamed for bloom)

// rotation about the planet axis
const rotY = ( v, a ) => {

	const c = cos( a ), s = sin( a );
	return vec3( v.x.mul( c ).add( v.z.mul( s ) ), v.y, v.z.mul( c ).sub( v.x.mul( s ) ) );

};

const equirectUV = ( q ) => vec2( atan( q.z, q.x ).div( 2 * PI ).add( 0.5 ), asin( q.y.clamp( - 1, 1 ) ).div( PI ).add( 0.5 ) );

// deterministic pseudo-random cyclone set (centre, radius, twist)
function cyclones() {

	let s = 1234567;
	const rnd = () => ( ( s = ( s * 16807 ) % 2147483647 ) / 2147483647 );
	const out = [];
	for ( let i = 0; i < 16; i ++ ) {

		const tropical = i < 5;
		const hemi = rnd() < 0.5 ? - 1 : 1;
		const lat = hemi * ( tropical ? 10 + rnd() * 14 : 32 + rnd() * 30 ) * Math.PI / 180;
		const lon = rnd() * Math.PI * 2;
		const c = new THREE.Vector3( Math.cos( lat ) * Math.cos( lon ), Math.sin( lat ), Math.cos( lat ) * Math.sin( lon ) );
		const radius = tropical ? 0.035 + rnd() * 0.025 : 0.09 + rnd() * 0.1;
		const twist = ( tropical ? 7 + rnd() * 4 : 2.5 + rnd() * 2.5 ) * hemi;
		out.push( { c, radius, twist } );

	}

	return out;

}

export function createPlanet( U, atmo, cfg ) {

	const { mapW: MW, mapH: MH, cloudN: CN } = cfg;

	// ───────────── baked planet map: R height, G cloud weather, B moisture, A population ─────────────
	const map = new THREE.StorageTexture( MW, MH );
	map.type = THREE.HalfFloatType;
	map.generateMipmaps = false;
	map.minFilter = map.magFilter = THREE.LinearFilter;
	map.wrapS = THREE.RepeatWrapping;
	map.wrapT = THREE.ClampToEdgeWrapping;
	map.name = 'atm_planetMap';

	const vort = cyclones();
	const oc = ( n ) => int( Math.max( 2, Math.round( n * ( cfg.bakeOct ?? 1 ) ) ) );

	const bakeCompute = Fn( () => {

		const ix = instanceIndex.mod( MW ), iy = instanceIndex.div( MW );
		const lon = float( ix ).add( 0.5 ).div( MW ).mul( 2 * PI ).sub( PI );
		const lat = float( iy ).add( 0.5 ).div( MH ).mul( PI ).sub( PI / 2 ).toVar();
		const cl = cos( lat );
		const q = vec3( cl.mul( cos( lon ) ), sin( lat ), cl.mul( sin( lon ) ) ).toVar();

		// continents: domain-warped fBm + a very low frequency term for continental shelves
		const wq = q.mul( 1.7 );
		const warp = vec3( fbm( wq.add( vec3( 13.1, 2.7, 5.3 ) ), oc( 4 ) ), fbm( wq.add( vec3( 37.7, 11.2, 1.9 ) ), oc( 4 ) ), fbm( wq.add( vec3( 71.3, 23.9, 8.4 ) ), oc( 4 ) ) );
		const pc = q.add( warp.mul( 0.55 ) ).toVar();
		const cont = fbm( pc.mul( 1.15 ).add( 3.3 ), oc( 8 ) ).add( gnoise( pc.mul( 0.6 ).add( 9.1 ) ).mul( 0.42 ) ).sub( 0.1 ).toVar();
		const land = smoothstep( 0.0, 0.22, cont );
		const mtnMask = smoothstep( - 0.1, 0.45, gnoise( pc.mul( 1.9 ).add( 4.4 ) ) );
		const rid = ridged( pc.mul( 5.0 ).add( 1.3 ), oc( 6 ) );
		const height = cont.add( rid.mul( 0.5 ).mul( land ).mul( mtnMask.mul( 0.75 ).add( 0.25 ) ) ).toVar();
		height.assign( select( height.lessThan( 0.0 ), height.mul( 1.3 ), height ) );

		// clouds: cyclonic vortex warps, then latitude-banded streaky fBm (ITCZ, subtropical highs, storm tracks)
		const qw = q.toVar();
		for ( const v of vort ) {

			const c = vec3( v.c.x, v.c.y, v.c.z );
			const d = length( qw.sub( c ) );
			const ang = exp( d.div( v.radius ).pow( 2 ).negate() ).mul( v.twist ).toVar();
			const ca = cos( ang ), sa = sin( ang );
			qw.assign( qw.mul( ca ).add( cross( c, qw ).mul( sa ) ).add( c.mul( dot( c, qw ) ).mul( ca.oneMinus() ) ) );

		}

		const latDeg = abs( lat ).mul( 180 / PI ).toVar();
		const band = float( 0.5 )
			.add( exp( latDeg.div( 7 ).pow( 2 ).negate() ).mul( 0.22 ) )
			.sub( exp( latDeg.sub( 23 ).div( 9 ).pow( 2 ).negate() ).mul( 0.3 ) )
			.add( exp( latDeg.sub( 53 ).div( 13 ).pow( 2 ).negate() ).mul( 0.2 ) );
		const n1 = fbm( qw.mul( vec3( 3.0, 5.2, 3.0 ) ).add( 21.0 ), oc( 6 ) ).mul( 0.5 ).add( 0.5 );
		const n2 = fbm( qw.mul( 11.0 ).add( 3.0 ), oc( 4 ) );
		const weather = n1.mul( 1.35 ).sub( 0.18 ).add( band.sub( 0.5 ) ).add( n2.mul( 0.22 ) ).clamp( 0, 1 );

		// moisture (biomes) and population clustering (city lights)
		const moist = fbm( q.mul( 2.3 ).add( 5.5 ), oc( 5 ) ).mul( 0.6 ).add( 0.5 );
		const civ = smoothstep( - 0.1, 0.35, fbm( q.mul( 3.1 ).add( 17.0 ), oc( 3 ) ) );
		const pop = smoothstep( 0.42, 0.85, fbm( q.mul( 28.0 ).add( 9.1 ), oc( 4 ) ).mul( 0.5 ).add( 0.5 ) ).mul( civ );

		textureStore( map, uvec2( ix, iy ), vec4( height, weather, moist, pop ) );

	} )().compute( MW * MH ).setName( 'atm_bakePlanet' );

	// ───────────── tileable Perlin–Worley cloud volume (R: Perlin-Worley, GBA: Worley fBm at 3 frequencies) ─────────────
	const noise3D = new THREE.Storage3DTexture( CN, CN, CN );
	noise3D.generateMipmaps = false;
	noise3D.minFilter = noise3D.magFilter = THREE.LinearFilter;
	noise3D.wrapS = noise3D.wrapT = noise3D.wrapR = THREE.RepeatWrapping;
	noise3D.name = 'atm_cloudNoise';

	const cloudNoiseCompute = Fn( () => {

		const i = instanceIndex;
		const x = i.mod( CN ), y = i.div( CN ).mod( CN ), z = i.div( CN * CN );
		const p = vec3( float( x ), float( y ), float( z ) ).add( 0.5 ).div( CN ).toVar();
		const f = ( k ) => Math.min( k, CN / 2 );
		const per = tgnoise( p.mul( 4 ), float( 4 ) ).mul( 0.5 ).add( tgnoise( p.mul( 8 ), float( 8 ) ).mul( 0.25 ) ).add( tgnoise( p.mul( 16 ), float( 16 ) ).mul( 0.125 ) ).mul( 0.6 ).add( 0.5 );
		const W4 = tworley( p.mul( 4 ), float( 4 ) ).toVar();
		const W8 = tworley( p.mul( 8 ), float( 8 ) ).toVar();
		const W16 = tworley( p.mul( f( 16 ) ), float( f( 16 ) ) ).toVar();
		const W32 = tworley( p.mul( f( 32 ) ), float( f( 32 ) ) ).toVar();
		const W64 = tworley( p.mul( f( 64 ) ), float( f( 64 ) ) ).toVar();
		const w1 = W4.mul( 0.625 ).add( W8.mul( 0.25 ) ).add( W16.mul( 0.125 ) );
		const w2 = W8.mul( 0.625 ).add( W16.mul( 0.25 ) ).add( W32.mul( 0.125 ) );
		const w3 = W16.mul( 0.625 ).add( W32.mul( 0.25 ) ).add( W64.mul( 0.125 ) );
		const pw = remap( per, w1.sub( 1 ), 1, 0, 1 ).clamp( 0, 1 );
		textureStore( noise3D, uvec3( x, y, z ), vec4( pw, w1, w2, w3 ) );

	} )().compute( CN * CN * CN ).setName( 'atm_cloudNoise' );

	// ───────────── cloud density ─────────────
	const sigmaScale = U.cloudDensity.mul( 28.0 ); // extinction (1/km) at density 1

	// coverage × height profile only (one 2D fetch) — used for empty-space skipping, light march and ground shadows
	const cheapDensity = ( P ) => {

		const r = length( P ).toVar();
		const hf = r.sub( CLOUD_BOTTOM ).div( CLOUD_TOP - CLOUD_BOTTOM );
		const qc = rotY( P.div( r ), U.cloudRot );
		const w = texture( map, equirectUV( qc ) ).level( 0 ).y;
		const cov = U.coverage;
		const base = w.sub( cov.oneMinus() ).div( max( cov, 0.05 ) ).clamp( 0, 1 ).toVar();
		const top = base.mul( 0.75 ).add( 0.25 );
		const prof = smoothstep( 0.0, 0.08, hf ).mul( smoothstep( top, top.mul( 0.55 ), hf ) );
		return base.mul( prof );

	};

	const fullDensity = ( P, cheap ) => {

		const r = length( P );
		const pc = rotY( P, U.cloudRot ).div( NOISE_KM ).add( vec3( 0, 0, U.cloudTime ) );
		const n = texture3D( noise3D, pc ).level( 0 );
		const fb = n.y.mul( 0.625 ).add( n.z.mul( 0.25 ) ).add( n.w.mul( 0.125 ) );
		const shape = remap( n.x, fb.sub( 1 ), 1, 0, 1 ).clamp( 0, 1 );
		const d = remap( cheap, shape.oneMinus().mul( 0.75 ), 1, 0, 1 ).clamp( 0, 1 ).toVar();
		if ( cfg.cloudDetail ) {

			const n2 = texture3D( noise3D, pc.mul( 4.3 ) ).level( 0 ).w;
			d.assign( remap( d, n2.mul( 0.35 ).mul( d.oneMinus() ), 1, 0, 1 ).clamp( 0, 1 ) );

		}

		return d.mul( smoothstep( CLOUD_BOTTOM - 0.01, CLOUD_BOTTOM + 0.3, r ) );

	};

	const cloudShadow = ( P, sun ) => {

		const ti = raySphere( P, sun, float( CLOUD_BOTTOM ) ).y;
		const to = raySphere( P, sun, float( CLOUD_TOP ) ).y;
		const a = max( ti, 0.0 ).toVar();
		const b = min( max( to, a ), a.add( 45.0 ) ).toVar();
		const od = float( 0 ).toVar();
		const N = 3;
		const dt = b.sub( a ).div( N );
		for ( let k = 0; k < N; k ++ ) od.addAssign( cheapDensity( P.add( sun.mul( a.add( dt.mul( k + 0.5 ) ) ) ) ).mul( dt ) );
		return exp( od.mul( sigmaScale ).mul( - 0.35 ) );

	};

	// ───────────── surface ─────────────
	const shadeGround = ( P, rd, tHit, sun ) => {

		const n0 = normalize( P ).toVar();
		const q = rotY( n0, U.planetRot ).toVar();
		const uv = equirectUV( q ).toVar();
		const m = texture( map, uv ).level( 0 ).toVar();
		const du = 1 / MW, dv = 1 / MH;
		const hE = texture( map, uv.add( vec2( du, 0 ) ) ).level( 0 ).x;
		const hW = texture( map, uv.sub( vec2( du, 0 ) ) ).level( 0 ).x;
		const hN = texture( map, uv.add( vec2( 0, dv ) ) ).level( 0 ).x;
		const hS = texture( map, uv.sub( vec2( 0, dv ) ) ).level( 0 ).x;
		const cosLat = max( sqrt( max( float( 1 ).sub( q.y.mul( q.y ) ), 0.0 ) ), 0.03 ).toVar();
		const gE = hE.sub( hW ).div( cosLat.mul( 2 * du * 2 * PI * RB ) );
		const gN = hN.sub( hS ).div( 2 * dv * PI * RB );

		// per-pixel detail (value noise with analytic derivatives), faded by the pixel footprint
		const V = rd.negate();
		const fpKm = tHit.mul( U.pixelAngle ).div( max( dot( n0, V ), 0.12 ) ).toVar();
		const DS = 70.0;
		const det = fbmd( q.mul( RB / DS ), int( cfg.detailOctaves ), fpKm.div( DS ) ).toVar();
		const coastAmp = float( 0.11 );
		const h = m.x.add( det.x.mul( coastAmp ) ).toVar();
		const gDet = det.yzw.mul( coastAmp.div( DS ) );
		const gDetT = gDet.sub( q.mul( dot( gDet, q ) ) );

		const E = normalize( vec3( q.z.negate(), 0, q.x ).add( vec3( 1e-5, 0, 0 ) ) );
		const Nn = cross( E, q );
		const isLand = smoothstep( - 0.004, 0.004, h ).toVar();
		const elevKm = max( h, 0.0 ).mul( ELEV_KM ).toVar();
		const grad = E.mul( gE ).add( Nn.mul( gN ) ).add( gDetT );
		const nLand = normalize( q.sub( grad.mul( ELEV_KM * 1.0 ).mul( U.bump ) ) );
		// ocean: gentle swell normals from the same detail noise, much flatter
		const nSea = normalize( q.sub( gDetT.mul( ELEV_KM * 0.06 ) ) );
		const nq = normalize( mix( nSea, nLand, isLand ) );
		const nW = rotY( nq, U.planetRot.negate() ).toVar();

		// biomes
		const latDeg = asin( abs( q.y ) ).mul( 180 / PI ).toVar();
		const temp = cosLat.pow( 1.4 ).sub( elevKm.mul( 0.075 ) ).add( det.x.mul( 0.05 ) ).toVar();
		const moist = m.z
			.add( exp( latDeg.div( 11 ).pow( 2 ).negate() ).mul( 0.22 ) )
			.sub( exp( latDeg.sub( 24 ).div( 9 ).pow( 2 ).negate() ).mul( 0.38 ) )
			.add( exp( latDeg.sub( 55 ).div( 12 ).pow( 2 ).negate() ).mul( 0.12 ) ).toVar();
		const desertC = vec3( 0.46, 0.33, 0.19 ), savannaC = vec3( 0.17, 0.14, 0.075 ), forestC = vec3( 0.035, 0.062, 0.022 );
		const taigaC = vec3( 0.03, 0.045, 0.032 ), tundraC = vec3( 0.12, 0.11, 0.09 ), rockC = vec3( 0.15, 0.13, 0.11 ), snowC = vec3( 0.78, 0.82, 0.88 );
		const alb = mix( desertC, savannaC, smoothstep( 0.22, 0.42, moist ) ).toVar();
		alb.assign( mix( alb, forestC, smoothstep( 0.45, 0.68, moist ) ) );
		alb.assign( mix( alb, taigaC, smoothstep( 0.62, 0.45, temp ).mul( smoothstep( 0.3, 0.5, moist ) ) ) );
		alb.assign( mix( alb, tundraC, smoothstep( 0.42, 0.28, temp ) ) );
		alb.assign( mix( alb, rockC, smoothstep( 1.6, 3.2, elevKm ).mul( 0.8 ) ) );
		const snowLine = temp.mul( 6.5 ).sub( 0.6 );
		const snow = max( smoothstep( snowLine.sub( 0.25 ), snowLine.add( 0.25 ), elevKm ), smoothstep( 0.2, 0.08, temp ) ).toVar();
		const desertness = smoothstep( 0.4, 0.2, moist ).mul( smoothstep( 0.35, 0.6, temp ) );

		// ocean colour and polar sea ice
		const depth = max( h.negate(), 0.0 ).toVar();
		const seaC = mix( vec3( 0.018, 0.075, 0.085 ), vec3( 0.004, 0.014, 0.038 ), smoothstep( 0.0, 0.12, depth ) );
		const seaIce = smoothstep( 70, 75, latDeg.add( det.x.mul( 6 ) ).add( m.z.sub( 0.5 ).mul( 8 ) ) ).toVar();
		const ice = max( snow.mul( isLand ), seaIce ).toVar();
		const albedo = mix( mix( seaC, alb, isLand ), snowC, ice ).toVar();

		// lighting (unit sun illuminance)
		const muS = dot( n0, sun ).toVar();
		const rS = elevKm.add( RB );
		const sunT = atmo.transmittance( rS, muS ).mul( atmo.earthShadow( rS, muS, 0.012 ) ).toVar();
		const cs = cloudShadow( P, sun ).toVar();
		const Esun = sunT.mul( cs ).toVar();
		const Esky = atmo.irradiance( elevKm, muS ).mul( cs.mul( 0.5 ).add( 0.5 ) ).toVar();
		const NdL = max( dot( nW, sun ), 0.0 );
		const col = albedo.mul( 1 / PI ).mul( Esun.mul( NdL ).add( Esky ) ).toVar();

		// ocean: GGX sun glint + Fresnel sky reflection
		const water = isLand.oneMinus().mul( seaIce.oneMinus() ).toVar();
		If( water.greaterThan( 0.01 ), () => {

			const Hv = normalize( V.add( sun ) );
			const NdH = max( dot( nW, Hv ), 0.0 );
			const NdV = max( dot( nW, V ), 0.02 );
			const NdLw = max( dot( nW, sun ), 0.0 );
			const VdH = max( dot( V, Hv ), 0.0 );
			const a = float( 0.11 ).add( fpKm.mul( 0.0008 ).min( 0.12 ) ); // widen with footprint (unresolved waves)
			const a2 = a.mul( a );
			const dd = NdH.mul( NdH ).mul( a2.sub( 1 ) ).add( 1 );
			const D = a2.div( dd.mul( dd ).mul( PI ) );
			const k = a.mul( 0.5 );
			const G = NdV.div( NdV.mul( k.oneMinus() ).add( k ) ).mul( NdLw.div( NdLw.mul( k.oneMinus() ).add( k ) ) );
			const F = float( 0.02 ).add( float( 0.98 ).mul( pow( VdH.oneMinus(), 5 ) ) );
			const spec = D.mul( G ).mul( F ).div( NdV.mul( 4 ) );
			const Fv = float( 0.02 ).add( float( 0.98 ).mul( pow( NdV.oneMinus(), 5 ) ) );
			col.addAssign( Esun.mul( spec ).add( Esky.mul( Fv ).mul( 1 / PI ) ).mul( water ) );

		} );

		// city lights on the night side: clustered, coast-loving, two cell scales + footprint-averaged fallback
		const night = smoothstep( 0.06, - 0.1, muS ).toVar();
		If( night.greaterThan( 0.001 ), () => {

			const lowland = exp( elevKm.div( - 0.45 ) ).mul( 0.7 ).add( 0.3 );
			const pop = m.w.mul( isLand ).mul( ice.oneMinus() ).mul( desertness.mul( 0.85 ).oneMinus() ).mul( lowland ).toVar();
			const pk = q.mul( RB ).toVar();
			const town = ( cell, thr, rad ) => {

				const c = floor( pk.div( cell ) ).toVar();
				const hh = hash33( c );
				const ctr = c.add( hh.mul( 0.6 ).add( 0.2 ) ).mul( cell );
				const d2 = dot( pk.sub( ctr ), pk.sub( ctr ) ).div( cell * cell );
				const on = smoothstep( thr, thr - 0.08, hash13( c.add( 7.7 ) ).add( pop.oneMinus() ) );
				return exp( d2.div( hh.z.mul( rad ).add( rad * 0.5 ) ).negate() ).mul( on ).mul( hh.x.add( 0.4 ) );

			};

			const lights = town( 7.0, 1.0, 0.03 ).add( town( 2.3, 0.7, 0.06 ).mul( smoothstep( 0.45, 0.9, pop ) ).mul( 1.5 ) ).toVar();
			const avg = pop.mul( 0.09 ).add( pop.mul( pop ).mul( 0.12 ) );
			lights.assign( mix( lights, avg, smoothstep( 0.6, 2.2, fpKm.div( 2.3 ) ) ) );
			col.addAssign( vec3( 1.0, 0.56, 0.22 ).mul( lights ).mul( night ).mul( U.cityLights ) );

		} );

		return col;

	};

	// ───────────── cloud march (Beer–Powder, dual-lobe HG, multi-scatter octaves) ─────────────
	const marchClouds = ( ro, rd, hitG, tG, sun, jitter ) => {

		const out = { L: vec3( 0 ).toVar(), alpha: float( 0 ).toVar(), depth: float( 0 ).toVar() };
		const r0 = length( ro );
		const outer = raySphere( ro, rd, float( CLOUD_TOP ) ).toVar();
		const inner = raySphere( ro, rd, float( CLOUD_BOTTOM ) ).toVar();
		const t0 = float( 0 ).toVar(), t1 = float( 0 ).toVar();
		If( r0.greaterThan( CLOUD_TOP ), () => {

			t0.assign( outer.x );
			t1.assign( select( inner.x.greaterThan( 0.0 ), inner.x, outer.y ) );

		} ).ElseIf( r0.greaterThan( CLOUD_BOTTOM ), () => {

			t0.assign( 0 );
			t1.assign( select( inner.x.greaterThan( 0.0 ), inner.x, outer.y ) );

		} ).Else( () => {

			t0.assign( select( hitG, float( 1e9 ), inner.y ) );
			t1.assign( outer.y );

		} );
		t0.assign( max( t0, 0.0 ) );
		t1.assign( min( t1, t0.add( 420.0 ) ) );

		If( t1.greaterThan( t0 ), () => {

			const N = cfg.cloudSteps;
			const dt = t1.sub( t0 ).div( N ).toVar();
			const T = float( 1 ).toVar();
			const wSum = float( 0 ).toVar(), dSum = float( 0 ).toVar();
			const cosT = dot( rd, sun ).toVar();
			const ph = ( g ) => mix( hgPhase( float( g ).mul( 0.8 ), cosT ), hgPhase( float( g ).mul( - 0.25 ), cosT ), 0.25 );
			const ph0 = ph( 1.0 ).toVar(), ph1 = ph( 0.5 ).toVar(), ph2 = ph( 0.25 ).toVar();
			const powderAmt = cosT.mul( - 0.5 ).add( 0.5 ).mul( 0.7 ).toVar();
			const sig = sigmaScale;

			Loop( N, ( { i } ) => {

				const t = t0.add( dt.mul( float( i ).add( jitter ) ) ).toVar();
				const P = ro.add( rd.mul( t ) ).toVar();
				const cheap = cheapDensity( P ).toVar();
				If( cheap.greaterThan( 0.004 ), () => {

					const d = fullDensity( P, cheap ).toVar();
					If( d.greaterThan( 0.002 ), () => {

						const sigma = d.mul( sig ).toVar();
						const r = length( P ).toVar();
						const up = P.div( r );
						const muS = dot( up, sun ).toVar();
						const sunT = atmo.transmittance( r, muS ).mul( atmo.earthShadow( r, muS, 0.01 ) ).toVar();
						// light march toward the sun (cheap density), geometric step growth
						const od = float( 0 ).toVar();
						const LS = cfg.lightSteps;
						let tl = 0;
						for ( let k = 0; k < LS; k ++ ) {

							const seg = 0.35 * Math.pow( 2.2, k );
							od.addAssign( cheapDensity( P.add( sun.mul( tl + seg * 0.5 ) ) ).mul( seg ) );
							tl += seg;

						}

						od.mulAssign( sig );
						const beer = exp( od.negate() ).mul( ph0 )
							.add( exp( od.mul( - 0.5 ) ).mul( ph1 ).mul( 0.5 ) )
							.add( exp( od.mul( - 0.25 ) ).mul( ph2 ).mul( 0.25 ) );
						const powder = mix( float( 1 ), exp( od.mul( - 2 ).sub( sigma.mul( 0.4 ) ) ).oneMinus().mul( 0.9 ).add( 0.1 ), powderAmt );
						const hf = r.sub( CLOUD_BOTTOM ).div( CLOUD_TOP - CLOUD_BOTTOM ).clamp( 0, 1 );
						const amb = atmo.irradiance( r.sub( RB ), muS ).mul( hf.mul( 0.6 ).add( 0.4 ) ).mul( 0.16 );
						const S = sunT.mul( beer ).mul( powder ).mul( 1.4 ).add( amb ).mul( sigma );
						const sT = exp( sigma.mul( dt ).negate() ).toVar();
						out.L.addAssign( S.sub( S.mul( sT ) ).div( max( sigma, 1e-5 ) ).mul( T ) );
						const w = T.mul( sT.oneMinus() );
						wSum.addAssign( w );
						dSum.addAssign( w.mul( t ) );
						T.mulAssign( sT );
						If( T.lessThan( 0.015 ), () => {

							Break();

						} );

					} );

				} );

			} );

			out.alpha.assign( T.oneMinus().div( 0.985 ).clamp( 0, 1 ) );
			out.depth.assign( dSum.div( max( wSum, 1e-5 ) ) );

		} );

		return out;

	};

	// ───────────── per-pixel atmosphere march for views from space (Hillaire: LUT-assisted ray marching) ─────────────
	// samples are distributed quadratically on both sides of the ray's closest approach to the planet, where the
	// optical depth along limb rays is concentrated.
	const marchSky = ( ro, rd, t0, t1, sun, jitter ) => {

		const L = vec3( 0 ).toVar(), T = vec3( 1 ).toVar();
		const len = max( t1.sub( t0 ), 0.0 ).toVar();
		const tc = dot( ro, rd ).negate().clamp( t0, t1 ).toVar();
		const us = tc.sub( t0 ).div( max( len, 1e-3 ) ).clamp( 1e-4, 1 - 1e-4 ).toVar();
		const c = dot( rd, sun );
		const phR = float( 3 / ( 16 * PI ) ).mul( c.mul( c ).add( 1 ) ).toVar();
		const phM = hgPhase( U.mieG, c ).toVar();
		const mapT = ( u ) => {

			const a = float( 1 ).sub( u.div( us ) );
			const b = u.sub( us ).div( us.oneMinus() );
			return select( u.lessThan( us ), tc.sub( tc.sub( t0 ).mul( a.mul( a ) ) ), tc.add( t1.sub( tc ).mul( b.mul( b ) ) ) );

		};

		const N = cfg.skyPixelSteps;
		Loop( N, ( { i } ) => {

			const ta = mapT( float( i ).div( N ) ).toVar();
			const tb = mapT( float( i ).add( 1 ).div( N ) ).toVar();
			const dt = tb.sub( ta );
			const s = atmo.scatterSample( ro.add( rd.mul( ta.add( dt.mul( jitter ) ) ) ), sun, phR, phM, true );
			atmo.accumulate( L, T, s.S, s.ext, dt );

		} );
		return { L, T };

	};

	// analytic airglow: green 557.7 nm O(¹S) emission shell at ~94 km, brightest edge-on along the night limb
	const airglow = ( ro, rd, tEnd, sun ) => {

		const o = raySphere( ro, rd, float( RB + 100 ) ).toVar();
		const i = raySphere( ro, rd, float( RB + 88 ) ).toVar();
		const seg = ( s ) => max( min( s.y, tEnd ).sub( max( s.x, 0.0 ) ), 0.0 );
		const len = seg( o ).sub( seg( i ) );
		const tc = dot( ro, rd ).negate().clamp( max( o.x, 0.0 ), max( min( o.y, tEnd ), 0.0 ) );
		const muS = dot( normalize( ro.add( rd.mul( tc ) ) ), sun );
		const nightF = smoothstep( - 0.08, - 0.3, muS );
		return vec3( 0.3, 1.0, 0.42 ).mul( len.mul( nightF ).mul( U.airglow ) );

	};

	const sunDisk = ( rd, sun ) => {

		const sinA = length( cross( rd, sun ) );
		const front = dot( rd, sun ).greaterThan( 0.0 );
		const x = sinA.div( SUN_ANG_R ).toVar();
		const aa = U.pixelAngle.div( SUN_ANG_R ).mul( 0.8 ).max( 0.02 );
		const disk = smoothstep( float( 1 ).add( aa ), float( 1 ).sub( aa ), x );
		// limb darkening (power law per channel, 680/550/440 nm)
		const mu = sqrt( max( float( 1 ).sub( x.mul( x ) ), 0.0 ) );
		const ld = vec3( 1 ).sub( vec3( 1 ).sub( pow( vec3( mu.max( 0.02 ) ), vec3( 0.397, 0.503, 0.652 ) ) ) );
		const glare = exp( sinA.div( - 0.012 ) ).mul( 0.6 ).add( exp( sinA.div( - 0.0025 ) ).mul( 4 ) );
		return select( front, ld.mul( disk ).mul( SUN_RADIANCE ).add( glare ), vec3( 0 ) );

	};

	const stars = ( rd ) => {

		const SC = 240;
		const s = rd.mul( SC ).toVar();
		const c = floor( s ).toVar();
		const hh = hash33( c );
		const d = length( s.sub( c.add( hh.mul( 0.7 ).add( 0.15 ) ) ) ).div( SC ).div( U.pixelAngle );
		const mag = pow( hash13( c.add( 3.1 ) ), 22 ).mul( 0.9 );
		const tint = mix( vec3( 1.0, 0.75, 0.55 ), vec3( 0.7, 0.85, 1.0 ), hh.z );
		return tint.mul( exp( d.mul( d ).mul( - 1.3 ) ).mul( mag ) );

	};

	// ───────────── full-screen material ─────────────
	const vNdc = varying( positionGeometry.xy, 'v_atmNdc' );
	const material = new THREE.NodeMaterial();
	material.name = 'atm_planet';
	material.depthTest = false;
	material.depthWrite = false;
	material.vertexNode = vec4( positionGeometry.xy, 0, 1 );
	material.fragmentNode = Fn( () => {

		const ndc = vNdc;
		const sun = U.sunDir;
		const rd = atmo.camRay( ndc ).toVar();
		const ro = U.camPos;
		const r0 = length( ro ).toVar();
		const jitter = interleavedGradientNoise( screenCoordinate.xy ).toVar();

		const atm = raySphere( ro, rd, float( RT ) ).toVar();
		const gnd = raySphere( ro, rd, float( RB ) ).toVar();
		const hitG = gnd.x.greaterThan( 0.0 ).toVar();
		const tG = gnd.x.toVar();
		const tStart = max( atm.x, 0.0 ).toVar();
		const pStart = ro.add( rd.mul( tStart ) ).toVar();

		const clouds = marchClouds( ro, rd, hitG, tG, sun, jitter );

		const col = vec3( 0 ).toVar();
		If( hitG, () => {

			const P = ro.add( rd.mul( tG ) ).toVar();
			const G = shadeGround( P, rd, tG, sun );
			const ap = atmo.apSample( ndc, tG.sub( tStart ) );
			const Tg = atmo.segmentTransmittance( pStart, P, rd, hitG );
			col.assign( ap.add( Tg.mul( G ) ) );

		} ).Else( () => {

			const S = vec3( 0 ).toVar(), Ts = vec3( 1 ).toVar();
			If( r0.greaterThan( RT - 0.5 ), () => {

				If( atm.y.greaterThan( 0.0 ), () => {

					const res = marchSky( ro, rd, tStart, atm.y, sun, jitter );
					S.assign( res.L );
					Ts.assign( res.T );

				} );

			} ).Else( () => {

				S.assign( atmo.skyView( ro, rd, sun ) );
				Ts.assign( atmo.transmittance( r0, dot( ro, rd ).div( r0 ) ) );

			} );
			col.assign( S.add( Ts.mul( sunDisk( rd, sun ).mul( clouds.alpha.oneMinus() ).add( stars( rd ) ) ) ) );

		} );

		If( clouds.alpha.greaterThan( 0.0 ), () => {

			const Pc = ro.add( rd.mul( clouds.depth ) );
			const apc = atmo.apSample( ndc, clouds.depth.sub( tStart ) );
			const Tc = atmo.segmentTransmittance( pStart, Pc, rd, hitG );
			col.assign( mix( col, apc, clouds.alpha ).add( Tc.mul( clouds.L ) ) );

		} );

		col.addAssign( airglow( ro, rd, select( hitG, tG, float( 1e9 ) ), sun ).mul( clouds.alpha.oneMinus() ) );

		if ( cfg.debugView ) {

			// quadrants: TL planet map (height / weather), TR transmittance LUT, BL aerial perspective slice, BR multi-scatter LUT
			const quv = fract( ndc.mul( 0.5 ).add( 0.5 ).mul( 2 ) );
			const left = ndc.x.lessThan( 0.0 ), top = ndc.y.greaterThan( 0.0 );
			const mp = texture( map, quv ).level( 0 );
			const tlC = vec3( mp.x.mul( 2 ).clamp( 0, 1 ), mp.y, select( mp.x.lessThan( 0.0 ), float( 0.6 ), float( 0 ) ) );
			const trC = texture( atmo.transLUT, quv ).level( 0 ).rgb;
			const blC = texture3D( atmo.apLUT, vec3( quv, 0.6 ) ).level( 0 ).rgb.mul( 4 );
			const brC = texture( atmo.msLUT, quv ).level( 0 ).rgb.mul( 30 );
			col.assign( select( top, select( left, tlC, trC ), select( left, blC, brC ) ).div( U.exposure ) );

		}

		return vec4( col.mul( U.exposure ), 1 );

	} )();

	const geometry = new THREE.PlaneGeometry( 2, 2 );
	const mesh = new THREE.Mesh( geometry, material );
	mesh.frustumCulled = false;

	function dispose() {

		map.dispose();
		noise3D.dispose();
		bakeCompute.dispose?.();
		cloudNoiseCompute.dispose?.();
		material.dispose();
		geometry.dispose();

	}

	// CPU-side helpers mirror the shader's geometry (for click-to-fly picking)
	return { mesh, map, noise3D, bakeCompute, cloudNoiseCompute, dispose };

}
