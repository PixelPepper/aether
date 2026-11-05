// Aurora — the night sky seen from the fjord: aurora raymarch through the observer-centric emission volume,
// procedural star field (celestially rotating, 2×2-cell search so PSFs never clip), a precomputed Milky Way with
// dust lanes, van Rhijn-brightened airglow layers and atmospheric extinction.
import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, Break, uniform, texture, texture3D, textureStore, instanceIndex, float, uint, vec2, vec3, vec4, uvec2,
	sin, cos, exp, exp2, sqrt, pow, max, min, clamp, abs, floor, fract, atan, asin, length, normalize, dot, mix, smoothstep, select, hash,
	screenUV, screenCoordinate, positionLocal, mx_fractal_noise_float,
} from 'three/tsl';
import { R_E, GRID, gU, gV, gW, dHdV, dRhodW, extinction } from './model.js';

const LAT = 69.6 * Math.PI / 180; // Tromsø-like latitude: Polaris 70° up in the north

const radec = ( ra, dec ) => new THREE.Vector3( Math.cos( dec ) * Math.sin( ra ), Math.sin( dec ), Math.cos( dec ) * Math.cos( ra ) );
const D2R = Math.PI / 180;
const NGP = radec( 192.86 * D2R, 27.13 * D2R );
const GC = radec( 266.40 * D2R, - 28.94 * D2R );
const GY = new THREE.Vector3().crossVectors( NGP, GC ).normalize();

export function createSky( renderer, sim, emission, cfg, auroraRT ) {

	const u = {
		celestial: uniform( new THREE.Matrix3() ),
		time: uniform( 0 ),
		airglow: uniform( 1 ),
		starGain: uniform( 1 ),
		pixMain: uniform( 0.001 ),
		pixRefl: uniform( 0.002 ),
		invProj: uniform( new THREE.Matrix4() ),
		camWorld: uniform( new THREE.Matrix4() ),
		stepScale: uniform( cfg.stepScale ),
		frame: uniform( 0 ),
		emitGain: uniform( 1 ),
	};

	// ── sidereal rotation: local (x east, y up, z south) → celestial (y = north celestial pole) ──
	const qPole = new THREE.Quaternion().setFromUnitVectors( new THREE.Vector3( 0, Math.sin( LAT ), - Math.cos( LAT ) ), new THREE.Vector3( 0, 1, 0 ) );
	const qSpin = new THREE.Quaternion();
	const mat4 = new THREE.Matrix4();
	const setSidereal = ( lst ) => {

		qSpin.setFromAxisAngle( new THREE.Vector3( 0, 1, 0 ), lst ).multiply( qPole );
		mat4.makeRotationFromQuaternion( qSpin );
		u.celestial.value.setFromMatrix4( mat4 );

	};

	setSidereal( 0 );

	// ── Milky Way: computed once into an equirectangular (celestial) half-float texture ──
	const MW_W = cfg.mwW, MW_H = cfg.mwW / 2;
	const mwTex = new THREE.StorageTexture( MW_W, MW_H );
	mwTex.type = THREE.HalfFloatType;
	mwTex.format = THREE.RGBAFormat;
	mwTex.wrapS = THREE.RepeatWrapping;
	mwTex.wrapT = THREE.ClampToEdgeWrapping;
	mwTex.magFilter = mwTex.minFilter = THREE.LinearFilter;
	mwTex.generateMipmaps = false;

	const mwKernel = Fn( () => {

		const i = instanceIndex;
		const x = i.mod( MW_W ), y = i.div( MW_W );
		const lon = x.toFloat().add( 0.5 ).mul( 2 * Math.PI / MW_W ).sub( Math.PI );
		const lat = y.toFloat().add( 0.5 ).mul( Math.PI / MW_H ).sub( Math.PI / 2 );
		const c = vec3( cos( lat ).mul( sin( lon ) ), sin( lat ), cos( lat ).mul( cos( lon ) ) ).toVar();
		const b = asin( clamp( dot( c, vec3( NGP.x, NGP.y, NGP.z ) ), - 1, 1 ) ).toVar();
		const l = atan( dot( c, vec3( GY.x, GY.y, GY.z ) ), dot( c, vec3( GC.x, GC.y, GC.z ) ) ).toVar();
		const n1 = mx_fractal_noise_float( c.mul( 5.0 ), 5, 2.0, 0.55 );
		const n2 = mx_fractal_noise_float( c.mul( 13.0 ).add( 3.1 ), 4, 2.1, 0.5 );
		const thin = exp( b.div( 0.085 ).pow( 2 ).negate() );
		const disc = exp( b.div( 0.16 ).pow( 2 ).negate() ).mul( 0.7 ).add( exp( b.div( 0.42 ).pow( 2 ).negate() ).mul( 0.3 ) );
		const lonW = cos( l ).mul( 0.5 ).add( 0.6 ).max( 0.15 );
		const bulge = exp( l.div( 0.42 ).pow( 2 ).add( b.div( 0.22 ).pow( 2 ) ).negate() );
		const clouds = n1.mul( 0.8 ).add( 0.85 ).max( 0 ).mul( n2.mul( 0.35 ).add( 1 ) );
		const glow = disc.mul( lonW ).mul( clouds ).add( bulge.mul( 1.6 ) ).toVar();
		// dust: the Great Rift and filamentary lanes hugging the plane
		const dust = smoothstep( - 0.05, 0.45, mx_fractal_noise_float( c.mul( 8.0 ).add( 11.7 ), 5, 2.0, 0.55 ).add( thin.mul( 0.25 ) ) ).mul( exp( b.div( 0.11 ).pow( 2 ).negate() ) );
		const ext = float( 1 ).sub( dust.mul( 0.82 ) );
		const warm = vec3( 1.0, 0.82, 0.62 ), cool = vec3( 0.78, 0.86, 1.0 );
		const col = mix( cool, warm, clamp( bulge.mul( 1.5 ).add( dust.mul( 0.5 ) ), 0, 1 ) ).mul( glow ).mul( ext );
		// alpha: faint-star density (for the star layer)
		textureStore( mwTex, uvec2( x, y ), vec4( col, disc.mul( lonW ).mul( ext ) ) );

	} )().compute( MW_W * MW_H ).setName( 'aurora.milkyway' );

	// ── stars ──
	const cubeCell = ( c ) => {

		const a = abs( c );
		const isX = a.x.greaterThanEqual( a.y ).and( a.x.greaterThanEqual( a.z ) );
		const isY = isX.not().and( a.y.greaterThanEqual( a.z ) );
		const ma = select( isX, a.x, select( isY, a.y, a.z ) );
		const uv = select( isX, vec2( c.z, c.y ), select( isY, vec2( c.x, c.z ), vec2( c.x, c.y ) ) ).div( ma );
		const face = select( isX, select( c.x.greaterThan( 0 ), float( 0 ), float( 1 ) ), select( isY, select( c.y.greaterThan( 0 ), float( 2 ), float( 3 ) ), select( c.z.greaterThan( 0 ), float( 4 ), float( 5 ) ) ) );
		return { uv, face, ma };

	};

	const starColor = ( h ) => {

		// blackbody-ish: M (orange) … B (blue-white)
		const k = h;
		return mix( mix( vec3( 1.0, 0.62, 0.36 ), vec3( 1.0, 0.92, 0.82 ), smoothstep( 0.0, 0.35, k ) ), vec3( 0.72, 0.82, 1.0 ), smoothstep( 0.55, 1.0, k ) );

	};

	const starLayer = ( c, N, seedOff, magMin, magRange, gam, pix, density ) => {

		const { uv, face, ma } = cubeCell( c );
		const g = uv.mul( 0.5 ).add( 0.5 ).mul( N ).toVar();
		const base = floor( g.sub( 0.5 ) ).toVar();
		const sum = vec3( 0 ).toVar();
		const sig = max( pix.mul( 0.6 ), 0.00035 ).toVar();
		const angPerCell = ma.mul( ma ).mul( 2 / N );
		for ( let k = 0; k < 4; k ++ ) {

			const cell = base.add( vec2( k & 1, k >> 1 ) );
			const id = cell.x.add( cell.y.mul( N + 2 ) ).add( face.mul( ( N + 2 ) * ( N + 2 ) ) ).add( seedOff );
			const h1 = hash( id ), h2 = hash( id.add( 0.37 ).mul( 1.618 ) ), h3 = hash( id.mul( 3.1 ).add( 17.0 ) ), h4 = hash( id.mul( 0.71 ).add( 91.0 ) );
			const sp = cell.add( vec2( h1, h2 ).mul( 0.84 ).add( 0.08 ) );
			const d = length( g.sub( sp ) ).mul( angPerCell );
			const mag = pow( h3, gam ).mul( magRange ).add( magMin );
			const F = exp2( mag.mul( - 1.329 ) ).mul( density );
			sum.addAssign( starColor( h4 ).mul( F.mul( exp( d.mul( d ).div( sig.mul( sig ).mul( - 2 ) ) ) ) ) );

		}

		return sum.div( sig.mul( sig ).mul( 2 * Math.PI ) );

	};

	const celestialDir = ( d ) => u.celestial.mul( d );

	const mwSample = ( c ) => texture( mwTex, vec2( atan( c.x, c.z ).mul( 0.5 / Math.PI ).add( 0.5 ), asin( clamp( c.y, - 1, 1 ) ).mul( 1 / Math.PI ).add( 0.5 ) ) ).level( 0 );

	// van Rhijn factor for a thin emitting layer at altitude h
	const vanRhijn = ( sinE, h ) => {

		const k = ( R_E / ( R_E + h ) ) ** 2;
		return float( 1 ).div( sqrt( float( 1 ).sub( float( 1 ).sub( sinE.mul( sinE ) ).mul( k ) ) ) );

	};

	// sky without aurora (radiance, linear)
	const skyBase = ( d, pix ) => {

		const e = max( d.y, 0.0 ).toVar();
		const ext = extinction( e ).toVar();
		const c = celestialDir( d ).toVar();
		const mw = mwSample( c ).toVar();
		// airglow: O 557.7 at 95 km with gravity-wave banding, O 630.0 at 250 km, broadband continuum
		const lp = d.xz.div( max( e, 0.06 ) ).mul( 95 );
		const waves = sin( dot( lp, vec2( 0.031, 0.012 ) ).add( u.time.mul( 0.05 ) ) ).mul( sin( dot( lp, vec2( - 0.008, 0.021 ) ).sub( u.time.mul( 0.03 ) ) ) ).mul( 0.3 ).add( 1 );
		const ag = vec3( 0.10, 0.30, 0.07 ).mul( vanRhijn( e, 95 ) ).mul( waves ).mul( 0.0042 )
			.add( vec3( 0.32, 0.03, 0.025 ).mul( vanRhijn( e, 250 ) ).mul( 0.0016 ) )
			.add( vec3( 0.55, 0.50, 0.40 ).mul( vanRhijn( e, 90 ) ).mul( 0.0011 ) );
		const bg = vec3( 0.0005, 0.0008, 0.0016 );
		// aurora light scattered by the lower atmosphere: a faint tinted glow toward the horizon
		const La = sim.lightRead.element( 0 ).xyz;
		const scat = La.mul( pow( float( 1 ).sub( e ), 6 ).mul( 0.05 ).add( 0.004 ) );
		const stars = starLayer( c, 70, 11, - 0.5, 5.5, 2.2, pix, 1 )
			.add( starLayer( c, 230, 173, 3.4, 4.5, 1.1, pix, mw.a.mul( 2.2 ).add( 0.55 ) ) );
		const tw = sin( u.time.mul( 13 ).add( d.x.mul( 4000 ) ).add( d.z.mul( 3100 ) ) ).mul( float( 1 ).sub( e ).pow( 3 ).mul( 0.35 ) ).add( 1 );
		const celestialLight = stars.mul( u.starGain ).mul( tw ).add( mw.rgb.mul( 0.0055 ) ).mul( ext );
		const above = smoothstep( - 0.02, 0.0, d.y );
		return celestialLight.add( ag.mul( u.airglow ).mul( ext ) ).mul( above ).add( bg ).add( scat );

	};

	// ── aurora raymarch ──
	const { NY, NZ } = sim.cfg;
	const H0 = GRID.h0, H1 = GRID.h1;
	const TM = Math.tan( GRID.rhoMax / R_E );
	const march = ( d, jit, maxSteps ) => {

		const L = vec3( 0 ).toVar();
		const dy = d.y.toVar();
		const dh = max( length( d.xz ), 1e-5 ).toVar();
		If( dy.greaterThan( - 0.03 ), () => {

			const U = gU( atan( d.x, d.z.negate() ) ).toVar();
			const Rdy = dy.mul( R_E );
			// numerically stable ray/shell intersection: s = (2Rh + h²) / (sqrt(R²dy² + 2Rh + h²) + R·dy)
			const sAt = ( h ) => {

				const q = float( 2 * R_E * h + h * h );
				return q.div( sqrt( Rdy.mul( Rdy ).add( q ) ).add( Rdy ) );

			};

			const den = dh.sub( dy.mul( TM ) );
			const sR = select( den.greaterThan( 1e-5 ), float( R_E * TM ).div( den ), float( 1e9 ) );
			const s1 = min( sAt( H1 ), sR ).toVar();
			const s = sAt( H0 ).toVar();
			const first = float( 1 ).toVar();
			Loop( maxSteps, () => {

				If( s.greaterThanEqual( s1 ), () => {

					Break();

				} );
				const X = s.mul( dh ), Ys = s.mul( dy );
				const Y = Ys.add( R_E );
				const r2 = X.mul( X ).add( Y.mul( Y ) );
				const r = sqrt( r2 );
				const h = X.mul( X ).add( Ys.mul( Ys ) ).add( Ys.mul( 2 * R_E ) ).div( r.add( R_E ) ).toVar();
				const rho = atan( X, Y ).mul( R_E );
				const V = gV( h ).toVar(), W = gW( rho ).toVar();
				const cellH = dHdV( V ).mul( 1 / NY ).max( 0.6 );
				const cellR = dRhodW( W ).mul( 1 / NZ ).max( 0.8 );
				const dhds = X.mul( dh ).add( Y.mul( dy ) ).div( r );
				const drds = dh.mul( R_E * R_E ).div( r2 );
				const ds = clamp( min( cellH.div( max( dhds, 1e-3 ) ), cellR.div( max( drds, 1e-3 ) ) ).mul( u.stepScale ), 0.4, 80 ).toVar();
				If( first.greaterThan( 0 ), () => {

					// jittered first step (dither instead of banding)
					s.addAssign( ds.mul( jit ) );
					first.assign( 0 );

				} ).Else( () => {

					const smp = texture3D( sim.volTex, vec3( U, V, W ), float( 0 ) );
					L.addAssign( emission.emit( smp.x, smp.y, h ).mul( ds ) );
					s.addAssign( ds );

				} );

			} );
			L.mulAssign( extinction( max( dy, 0 ) ).mul( u.emitGain ) );

		} );
		return L;

	};

	const ign = ( fc ) => fract( fract( dot( fc, vec2( 0.06711056, 0.00583715 ) ) ).mul( 52.9829189 ).add( u.frame.mul( 0.61803 ) ) );

	// full-screen aurora pass (rendered into a reduced-resolution target with the main camera's matrices)
	const auroraMat = new THREE.NodeMaterial();
	auroraMat.depthTest = false; auroraMat.depthWrite = false;
	auroraMat.fragmentNode = Fn( () => {

		const ndc = vec2( screenUV.x.mul( 2 ).sub( 1 ), float( 1 ).sub( screenUV.y.mul( 2 ) ) );
		const vp = u.invProj.mul( vec4( ndc, 1, 1 ) );
		const dir = normalize( u.camWorld.mul( vec4( vp.xyz.div( vp.w ), 0 ) ).xyz );
		return vec4( march( dir, ign( screenCoordinate.xy ), cfg.maxSteps ), 1 );

	} )();

	// main sky dome: full-resolution stars + Milky Way + airglow, aurora from the reduced-res target
	const skyMainMat = new THREE.MeshBasicNodeMaterial( { side: THREE.BackSide, depthWrite: false, fog: false } );
	skyMainMat.colorNode = Fn( () => {

		const d = normalize( positionLocal );
		return skyBase( d, u.pixMain ).add( texture( auroraRT.texture, screenUV ).rgb );

	} )();

	// reflection sky dome: the mirrored camera raymarches the aurora itself (at the reflection's resolution)
	const skyReflMat = new THREE.MeshBasicNodeMaterial( { side: THREE.BackSide, depthWrite: false, fog: false } );
	skyReflMat.colorNode = Fn( () => {

		const d = normalize( positionLocal );
		return skyBase( d, u.pixRefl ).add( march( d, ign( screenCoordinate.xy ), cfg.maxStepsRefl ) );

	} )();

	const geo = new THREE.SphereGeometry( 100, 48, 24 );
	const skyMain = new THREE.Mesh( geo, skyMainMat );
	const skyRefl = new THREE.Mesh( geo, skyReflMat );
	for ( const m of [ skyMain, skyRefl ] ) {

		m.renderOrder = - 10; m.frustumCulled = false;

	}

	skyRefl.visible = false;

	const quad = new THREE.QuadMesh( auroraMat );

	return {
		u, skyMain, skyRefl, quad, setSidereal, skyBase,
		async init() {

			await renderer.computeAsync( mwKernel );

		},
		dispose() {

			geo.dispose(); skyMainMat.dispose(); skyReflMat.dispose(); auroraMat.dispose(); mwTex.dispose();
			mwKernel.dispose();

		},
	};

}

export const _keep = { exp2, min, uint };
