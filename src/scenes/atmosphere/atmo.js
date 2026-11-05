// Physically based atmosphere after Sébastien Hillaire, "A Scalable and Production Ready Sky and Atmosphere
// Rendering Technique" (EGSR 2020). All lookup tables live in storage textures written by compute passes:
//   transmittance LUT     (256×64)    T(r, μ) to the top of the atmosphere, Bruneton parameterisation
//   multi-scattering LUT  (32×32)     Ψ_ms(r, μs) — infinite scattering orders via the isotropic 1/(1-f_ms) series
//   sky-irradiance LUT    (64×8)      cosine-weighted sky dome irradiance (ambient for ground and clouds)
//   sky-view LUT          (192×108)   in-atmosphere sky, non-linear latitude mapping around the horizon
//   aerial-perspective    (froxels)   camera-frustum volume of in-scattered luminance + mean transmittance
// Units: kilometres, planet centred at the origin, +Y = planet axis. Earth parameters.
import * as THREE from 'three/webgpu';
import {
	Fn, float, vec2, vec3, vec4, uvec2, uvec3, texture, texture3D, textureStore, instanceIndex, Loop, If, int,
	select, exp, sqrt, max, min, abs, dot, normalize, length, cos, sin, acos, cross, smoothstep,
} from 'three/tsl';

export const RB = 6360.0; // ground radius
export const RT = 6460.0; // top of atmosphere
const H_TOP = Math.sqrt( RT * RT - RB * RB );
const PI = Math.PI;

const RAYLEIGH = vec3( 5.802e-3, 13.558e-3, 33.1e-3 ); // 1/km, scale height 8 km
const MIE_SCAT = 3.996e-3, MIE_EXT = 4.40e-3; //          scale height 1.2 km
const OZONE = vec3( 0.650e-3, 1.881e-3, 0.085e-3 ); //    tent profile centred at 25 km, half-width 15 km
const GROUND_ALBEDO = 0.3;

export const raySphere = /*@__PURE__*/ Fn( ( [ ro, rd, r ] ) => {

	// numerically robust: distance from centre to the ray line instead of b² - c
	const b = dot( ro, rd );
	const perp = ro.sub( rd.mul( b ) );
	const disc = r.mul( r ).sub( dot( perp, perp ) );
	const s = sqrt( max( disc, 0.0 ) );
	return select( disc.lessThan( 0.0 ), vec2( - 1e9, - 1e9 ), vec2( b.negate().sub( s ), b.negate().add( s ) ) );

} ).setLayout( { name: 'atm_raySphere', type: 'vec2', inputs: [ { name: 'ro', type: 'vec3' }, { name: 'rd', type: 'vec3' }, { name: 'r', type: 'float' } ] } );

export const rayleighPhase = ( c ) => float( 3 / ( 16 * PI ) ).mul( c.mul( c ).add( 1 ) );
export const hgPhase = ( g, c ) => {

	const g2 = float( g ).mul( g );
	const d = g2.add( 1 ).sub( float( g ).mul( 2 ).mul( c ) ).max( 1e-4 );
	return g2.oneMinus().div( d.mul( d.sqrt() ).mul( 4 * PI ) );

};

const toSub = ( u, res ) => float( u ).add( 0.5 / res ).mul( res / ( res + 1 ) );
const fromSub = ( u, res ) => float( u ).sub( 0.5 / res ).mul( res / ( res - 1 ) );

export function createAtmosphere( U, cfg ) {

	const TW = 256, TH = 64, MS = 32, IW = 64, IH = 8;
	const { skyW: SW, skyH: SH, apW: AW, apH: AH, apD: AD, apSub } = cfg;

	const mk2D = ( w, h, name ) => {

		const t = new THREE.StorageTexture( w, h );
		t.type = THREE.HalfFloatType;
		t.generateMipmaps = false;
		t.minFilter = t.magFilter = THREE.LinearFilter;
		t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
		t.name = name;
		return t;

	};

	const transLUT = mk2D( TW, TH, 'atm_transmittance' );
	const msLUT = mk2D( MS, MS, 'atm_multiscatter' );
	const irrLUT = mk2D( IW, IH, 'atm_irradiance' );
	const skyLUT = mk2D( SW, SH, 'atm_skyview' );
	skyLUT.wrapS = THREE.ClampToEdgeWrapping;
	const apLUT = new THREE.Storage3DTexture( AW, AH, AD );
	apLUT.type = THREE.HalfFloatType;
	apLUT.generateMipmaps = false;
	apLUT.minFilter = apLUT.magFilter = THREE.LinearFilter;
	apLUT.wrapS = apLUT.wrapT = apLUT.wrapR = THREE.ClampToEdgeWrapping;
	apLUT.name = 'atm_aerial';

	// ───────────── participating medium ─────────────
	const medium = ( h ) => {

		const hh = max( h, 0.0 );
		const dR = exp( hh.mul( - 1 / 8 ) );
		const dM = exp( hh.mul( - 1 / 1.2 ) );
		const dO = max( float( 1 ).sub( abs( hh.sub( 25 ) ).div( 15 ) ), 0.0 );
		const sR = RAYLEIGH.mul( dR.mul( U.rayScale ) );
		const sM = dM.mul( U.mieScale ).mul( MIE_SCAT );
		const ext = sR.add( dM.mul( U.mieScale ).mul( MIE_EXT ) ).add( OZONE.mul( dO.mul( U.ozoneScale ) ) );
		return { sR, sM, ext };

	};

	// ───────────── LUT parameterisations ─────────────
	const transUV = ( r, mu ) => {

		const rho = sqrt( max( r.mul( r ).sub( RB * RB ), 0.0 ) );
		const disc = r.mul( r ).mul( mu.mul( mu ).sub( 1 ) ).add( RT * RT );
		const d = max( r.negate().mul( mu ).add( sqrt( max( disc, 0.0 ) ) ), 0.0 );
		const dmin = float( RT ).sub( r );
		const dmax = rho.add( H_TOP );
		return vec2( d.sub( dmin ).div( max( dmax.sub( dmin ), 1e-3 ) ), rho.div( H_TOP ) );

	};

	const transmittance = ( r, mu ) => texture( transLUT, transUV( r, mu ) ).level( 0 ).rgb;

	// soft Earth shadow: is the sun above the geometric horizon of a point at radius r?
	const earthShadow = ( r, muS, eps = 0.006 ) => {

		const muH = sqrt( max( float( 1 ).sub( float( RB * RB ).div( r.mul( r ) ) ), 0.0 ) ).negate();
		return smoothstep( muH.sub( eps ), muH.add( eps ), muS );

	};

	const msLookup = ( r, muS ) => {

		const u = toSub( muS.mul( 0.5 ).add( 0.5 ).clamp( 0, 1 ), MS );
		const v = toSub( r.sub( RB ).div( RT - RB ).clamp( 0, 1 ), MS );
		return texture( msLUT, vec2( u, v ) ).level( 0 ).rgb;

	};

	const irradiance = ( hKm, muS ) => {

		const u = toSub( muS.mul( 0.5 ).add( 0.5 ).clamp( 0, 1 ), IW );
		const v = toSub( hKm.div( 12 ).clamp( 0, 1 ), IH );
		return texture( irrLUT, vec2( u, v ) ).level( 0 ).rgb;

	};

	// one ray-march sample: in-scattered radiance S and extinction at P
	const scatterSample = ( P, sunDir, phR, phM, withMS ) => {

		const r = length( P ).toVar();
		const muS = dot( P.div( r ), sunDir ).toVar();
		const m = medium( r.sub( RB ) );
		const sun = transmittance( r, muS ).mul( earthShadow( r, muS ) );
		let S = sun.mul( m.sR.mul( phR ).add( m.sM.mul( phM ) ) );
		if ( withMS ) S = S.add( msLookup( r, muS ).mul( m.sR.add( m.sM ) ) );
		return { S, ext: m.ext, scat: m.sR.add( m.sM ) };

	};

	// analytic integration of a constant-medium segment (Hillaire 2015, Frostbite): returns updated L, T
	const accumulate = ( L, T, S, ext, dt ) => {

		const sT = exp( ext.mul( dt ).negate() ).toVar();
		L.addAssign( T.mul( S.sub( S.mul( sT ) ).div( max( ext, vec3( 1e-7 ) ) ) ) );
		T.mulAssign( sT );

	};

	// ───────────── transmittance LUT ─────────────
	const transCompute = Fn( () => {

		const ix = instanceIndex.mod( TW ), iy = instanceIndex.div( TW );
		const u = float( ix ).add( 0.5 ).div( TW ), v = float( iy ).add( 0.5 ).div( TH );
		const rho = v.mul( H_TOP ).toVar();
		const r = sqrt( rho.mul( rho ).add( RB * RB ) ).toVar();
		const dmin = float( RT ).sub( r ), dmax = rho.add( H_TOP );
		const d = dmin.add( u.mul( dmax.sub( dmin ) ) ).toVar();
		const mu = select( d.lessThan( 1e-4 ), float( 1 ), float( H_TOP * H_TOP ).sub( rho.mul( rho ) ).sub( d.mul( d ) ).div( r.mul( 2 ).mul( d ) ) ).clamp( - 1, 1 ).toVar();
		const od = vec3( 0 ).toVar();
		const N = 40;
		const dt = d.div( N ).toVar();
		Loop( N, ( { i } ) => {

			const t = float( i ).add( 0.5 ).mul( dt );
			const rr = sqrt( r.mul( r ).add( t.mul( t ) ).add( r.mul( mu ).mul( t ).mul( 2 ) ) );
			od.addAssign( medium( rr.sub( RB ) ).ext.mul( dt ) );

		} );
		textureStore( transLUT, uvec2( ix, iy ), vec4( exp( od.negate() ), 1 ) );

	} )().compute( TW * TH ).setName( 'atm_transmittanceLUT' );

	// ───────────── multiple-scattering LUT (Hillaire 2020 §5.5) ─────────────
	const msCompute = Fn( () => {

		const ix = instanceIndex.mod( MS ), iy = instanceIndex.div( MS );
		const u = fromSub( float( ix ).add( 0.5 ).div( MS ), MS ).clamp( 0, 1 );
		const v = fromSub( float( iy ).add( 0.5 ).div( MS ), MS ).clamp( 0, 1 );
		const cs = u.mul( 2 ).sub( 1 ).toVar();
		const sunDir = vec3( 0, cs, sqrt( max( float( 1 ).sub( cs.mul( cs ) ), 0.0 ) ) ).toVar();
		const r0 = float( RB + 0.01 ).add( v.mul( RT - RB - 0.02 ) ).toVar();
		const ro = vec3( 0, r0, 0 ).toVar();
		const Lsum = vec3( 0 ).toVar(), Fsum = vec3( 0 ).toVar();
		const SQ = 8, N = 20;
		Loop( { start: int( 0 ), end: int( SQ ), type: 'int', name: 'si' }, ( { si } ) => {

			Loop( { start: int( 0 ), end: int( SQ ), type: 'int', name: 'sj' }, ( { sj } ) => {

				const theta = float( si ).add( 0.5 ).div( SQ ).mul( 2 * PI );
				const phi = acos( float( 1 ).sub( float( sj ).add( 0.5 ).div( SQ ).mul( 2 ) ) );
				const rd = vec3( cos( theta ).mul( sin( phi ) ), cos( phi ), sin( theta ).mul( sin( phi ) ) ).toVar();
				const tB = raySphere( ro, rd, float( RB ) ).toVar();
				const tT = raySphere( ro, rd, float( RT ) ).toVar();
				const hitG = tB.x.greaterThan( 0.0 );
				const tMax = select( hitG, tB.x, tT.y ).toVar();
				const dt = tMax.div( N ).toVar();
				const L = vec3( 0 ).toVar(), T = vec3( 1 ).toVar(), F = vec3( 0 ).toVar();
				Loop( N, ( { i } ) => {

					const P = ro.add( rd.mul( float( i ).add( 0.3 ).mul( dt ) ) );
					const s = scatterSample( P, sunDir, 1 / ( 4 * PI ), 1 / ( 4 * PI ), false );
					const sT = exp( s.ext.mul( dt ).negate() ).toVar();
					const inv = float( 1 ).div( max( s.ext, vec3( 1e-7 ) ) );
					L.addAssign( T.mul( s.S.sub( s.S.mul( sT ) ).mul( inv ) ) );
					F.addAssign( T.mul( s.scat.sub( s.scat.mul( sT ) ).mul( inv ) ) );
					T.mulAssign( sT );

				} );
				If( hitG, () => {

					const P = ro.add( rd.mul( tB.x ) );
					const up = normalize( P );
					const muS = dot( up, sunDir );
					L.addAssign( transmittance( float( RB ), muS ).mul( T ).mul( muS.clamp( 0, 1 ) ).mul( GROUND_ALBEDO / PI ) );

				} );
				Lsum.addAssign( L );
				Fsum.addAssign( F );

			} );

		} );
		const inv = 1 / ( SQ * SQ );
		const fms = Fsum.mul( inv );
		const psi = Lsum.mul( inv ).div( vec3( 1 ).sub( fms ).max( 1e-3 ) );
		textureStore( msLUT, uvec2( ix, iy ), vec4( psi, 1 ) );

	} )().compute( MS * MS ).setName( 'atm_multiScatteringLUT' );

	// ───────────── sky irradiance LUT (cosine-weighted hemisphere of single + multiple scattering) ─────────────
	const irrCompute = Fn( () => {

		const ix = instanceIndex.mod( IW ), iy = instanceIndex.div( IW );
		const u = fromSub( float( ix ).add( 0.5 ).div( IW ), IW ).clamp( 0, 1 );
		const v = fromSub( float( iy ).add( 0.5 ).div( IH ), IH ).clamp( 0, 1 );
		const cs = u.mul( 2 ).sub( 1 ).toVar();
		const sunDir = vec3( 0, cs, sqrt( max( float( 1 ).sub( cs.mul( cs ) ), 0.0 ) ) ).toVar();
		const ro = vec3( 0, float( RB + 0.02 ).add( v.mul( 12 ) ), 0 ).toVar();
		const E = vec3( 0 ).toVar();
		const SQ = 6, N = 12;
		Loop( { start: int( 0 ), end: int( SQ ), type: 'int', name: 'si' }, ( { si } ) => {

			Loop( { start: int( 0 ), end: int( SQ ), type: 'int', name: 'sj' }, ( { sj } ) => {

				const u1 = float( si ).add( 0.5 ).div( SQ ), u2 = float( sj ).add( 0.5 ).div( SQ );
				const rr = sqrt( u1 ), ph = u2.mul( 2 * PI );
				const rd = vec3( rr.mul( cos( ph ) ), sqrt( float( 1 ).sub( u1 ) ), rr.mul( sin( ph ) ) ).toVar();
				const tMax = raySphere( ro, rd, float( RT ) ).y.toVar();
				const c = dot( rd, sunDir );
				const phR = rayleighPhase( c ), phM = hgPhase( U.mieG, c );
				const L = vec3( 0 ).toVar(), T = vec3( 1 ).toVar();
				Loop( N, ( { i } ) => {

					const a = float( i ).div( N ), b = float( i ).add( 1 ).div( N );
					const t0 = a.mul( a ).mul( tMax ), t1 = b.mul( b ).mul( tMax );
					const dt = t1.sub( t0 );
					const s = scatterSample( ro.add( rd.mul( t0.add( dt.mul( 0.3 ) ) ) ), sunDir, phR, phM, true );
					accumulate( L, T, s.S, s.ext, dt );

				} );
				E.addAssign( L );

			} );

		} );
		textureStore( irrLUT, uvec2( ix, iy ), vec4( E.mul( PI / ( SQ * SQ ) ), 1 ) );

	} )().compute( IW * IH ).setName( 'atm_irradianceLUT' );

	// ───────────── sky-view LUT (camera inside the atmosphere) ─────────────
	const skyCompute = Fn( () => {

		const ix = instanceIndex.mod( SW ), iy = instanceIndex.div( SW );
		const u = fromSub( float( ix ).add( 0.5 ).div( SW ), SW ).clamp( 0, 1 );
		const v = fromSub( float( iy ).add( 0.5 ).div( SH ), SH ).clamp( 0, 1 ).toVar();
		const viewR = U.viewR;
		const vH = sqrt( max( viewR.mul( viewR ).sub( RB * RB ), 0.0 ) );
		const beta = acos( vH.div( viewR ) ).toVar();
		const zha = float( PI ).sub( beta ).toVar();
		const vzc = float( 0 ).toVar();
		If( v.lessThan( 0.5 ), () => {

			const c = float( 1 ).sub( v.mul( 2 ) );
			vzc.assign( cos( zha.mul( float( 1 ).sub( c.mul( c ) ) ) ) );

		} ).Else( () => {

			const c = v.mul( 2 ).sub( 1 );
			vzc.assign( cos( zha.add( beta.mul( c.mul( c ) ) ) ) );

		} );
		const lvc = u.mul( u ).mul( 2 ).sub( 1 ).negate().toVar();
		const szc = U.viewSunZenith;
		const sunDir = vec3( sqrt( max( float( 1 ).sub( szc.mul( szc ) ), 0.0 ) ), szc, 0 ).toVar();
		const vzs = sqrt( max( float( 1 ).sub( vzc.mul( vzc ) ), 0.0 ) );
		const rd = vec3( vzs.mul( lvc ), vzc, vzs.mul( sqrt( max( float( 1 ).sub( lvc.mul( lvc ) ), 0.0 ) ) ) ).toVar();
		const ro = vec3( 0, viewR, 0 ).toVar();
		const tB = raySphere( ro, rd, float( RB ) ).toVar();
		const tMax = select( tB.x.greaterThan( 0.0 ), tB.x, raySphere( ro, rd, float( RT ) ).y ).toVar();
		const c = dot( rd, sunDir );
		const phR = rayleighPhase( c ), phM = hgPhase( U.mieG, c );
		const L = vec3( 0 ).toVar(), T = vec3( 1 ).toVar();
		const N = cfg.skySteps;
		Loop( N, ( { i } ) => {

			const a = float( i ).div( N ), b = float( i ).add( 1 ).div( N );
			const t0 = a.mul( a ).mul( tMax ), t1 = b.mul( b ).mul( tMax );
			const dt = t1.sub( t0 );
			const s = scatterSample( ro.add( rd.mul( t0.add( dt.mul( 0.3 ) ) ) ), sunDir, phR, phM, true );
			accumulate( L, T, s.S, s.ext, dt );

		} );
		textureStore( skyLUT, uvec2( ix, iy ), vec4( L, 1 ) );

	} )().compute( SW * SH ).setName( 'atm_skyViewLUT' );

	// sample the sky-view LUT for a world-space view ray from the camera (inside the atmosphere)
	const skyView = ( camPos, rd, sunDir ) => {

		const viewR = length( camPos ).toVar();
		const up = camPos.div( viewR ).toVar();
		const vzc = dot( rd, up ).toVar();
		const side = normalize( cross( up, rd ) );
		const fwd = normalize( cross( side, up ) );
		const lp = normalize( vec2( dot( sunDir, fwd ), dot( sunDir, side ) ).add( vec2( 1e-6, 0 ) ) );
		const lvc = lp.x;
		const vH = sqrt( max( viewR.mul( viewR ).sub( RB * RB ), 0.0 ) );
		const beta = acos( vH.div( viewR ) ).toVar();
		const zha = float( PI ).sub( beta );
		const hitG = raySphere( camPos, rd, float( RB ) ).x.greaterThan( 0.0 );
		const vUp = float( 1 ).sub( sqrt( max( float( 1 ).sub( acos( vzc.clamp( - 1, 1 ) ).div( zha ) ), 0.0 ) ) ).mul( 0.5 );
		const vDn = sqrt( max( acos( vzc.clamp( - 1, 1 ) ).sub( zha ).div( beta ), 0.0 ) ).mul( 0.5 ).add( 0.5 );
		const vv = select( hitG, vDn, vUp );
		const uu = sqrt( lvc.mul( - 0.5 ).add( 0.5 ).clamp( 0, 1 ) );
		return texture( skyLUT, vec2( toSub( uu, SW ), toSub( vv, SH ) ) ).level( 0 ).rgb;

	};

	// ───────────── aerial-perspective froxel volume ─────────────
	// Each thread owns one froxel column (camera ray through a screen cell) and marches it once, writing every
	// depth slice as it passes it. Slice k sits at distance Dmax·((k+.5)/D)² from the ray's atmosphere entry,
	// so from orbit the volume starts at the top of the atmosphere instead of at the camera.
	const camRay = ( ndc ) => normalize( U.camFwd.add( U.camRight.mul( ndc.x.mul( U.tanHalf.x ) ) ).add( U.camUp.mul( ndc.y.mul( U.tanHalf.y ) ) ) );

	const apCompute = Fn( () => {

		const ix = instanceIndex.mod( AW ), iy = instanceIndex.div( AW );
		const ndc = vec2( float( ix ).add( 0.5 ).div( AW ).mul( 2 ).sub( 1 ), float( iy ).add( 0.5 ).div( AH ).mul( 2 ).sub( 1 ) );
		const rd = camRay( ndc ).toVar();
		const ro = U.camPos;
		const atm = raySphere( ro, rd, float( RT ) ).toVar();
		const tStart = max( atm.x, 0.0 ).toVar();
		const g = raySphere( ro, rd, float( RB ) ).toVar();
		const tEnd = max( select( g.x.greaterThan( 0.0 ), min( g.x, atm.y ), atm.y ), tStart ).toVar();
		const c = dot( rd, U.sunDir );
		const phR = rayleighPhase( c ), phM = hgPhase( U.mieG, c );
		const L = vec3( 0 ).toVar(), T = vec3( 1 ).toVar();
		const dPrev = float( 0 ).toVar();
		Loop( { start: int( 0 ), end: int( AD ), type: 'int', name: 'k' }, ( { k } ) => {

			const w = float( k ).add( 0.5 ).div( AD );
			const dk = w.mul( w ).mul( U.apDmax ).toVar();
			const a = min( tStart.add( dPrev ), tEnd ).toVar();
			const b = min( tStart.add( dk ), tEnd ).toVar();
			const dt = b.sub( a ).div( apSub ).toVar();
			If( dt.greaterThan( 0.0 ), () => {

				Loop( apSub, ( { i } ) => {

					const P = ro.add( rd.mul( a.add( dt.mul( float( i ).add( 0.5 ) ) ) ) );
					const s = scatterSample( P, U.sunDir, phR, phM, true );
					accumulate( L, T, s.S, s.ext, dt );

				} );

			} );
			textureStore( apLUT, uvec3( ix, iy, k ), vec4( L, dot( T, vec3( 1 / 3 ) ) ) );
			dPrev.assign( dk );

		} );

	} )().compute( AW * AH ).setName( 'atm_aerialPerspective' );

	// in-scattering between the view ray's atmosphere entry and distance d beyond it (unit sun illuminance)
	const apSample = ( ndc, d ) => {

		const w = sqrt( max( d, 0.0 ).div( U.apDmax ) ).toVar();
		const w0 = 0.5 / AD;
		const weight = w.div( w0 ).clamp( 0, 1 );
		const uvw = vec3( ndc.mul( 0.5 ).add( 0.5 ), max( w, w0 ) );
		return texture3D( apLUT, uvw ).level( 0 ).rgb.mul( weight );

	};

	// transmittance between two points on a ray, from the transmittance LUT (Bruneton trick):
	// forward rays use T(a)/T(b) to the top; rays that end on the ground use the reversed direction.
	const segmentTransmittance = ( pa, pb, rd, hitsGround ) => {

		const ra = length( pa ).toVar(), rb = length( pb ).toVar();
		const mua = dot( pa, rd ).div( ra ), mub = dot( pb, rd ).div( rb );
		const fwd = transmittance( ra, mua ).div( max( transmittance( rb, mub ), vec3( 1e-6 ) ) );
		const rev = transmittance( rb, mub.negate() ).div( max( transmittance( ra, mua.negate() ), vec3( 1e-6 ) ) );
		return select( hitsGround, rev, fwd ).clamp( 0, 1 );

	};

	function dispose() {

		for ( const t of [ transLUT, msLUT, irrLUT, skyLUT, apLUT ] ) t.dispose();
		for ( const c of [ transCompute, msCompute, irrCompute, skyCompute, apCompute ] ) c.dispose?.();

	}

	return {
		transLUT, msLUT, irrLUT, skyLUT, apLUT,
		transCompute, msCompute, irrCompute, skyCompute, apCompute,
		medium, transmittance, earthShadow, msLookup, irradiance, scatterSample, accumulate,
		skyView, apSample, segmentTransmittance, camRay,
		sizes: { TW, TH, MS, SW, SH, AW, AH, AD, IW, IH },
		dispose,
	};

}
