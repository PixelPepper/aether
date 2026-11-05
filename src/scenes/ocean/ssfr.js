// Screen-space fluid rendering (van der Laan et al. 2009 style):
//   1. particles → sphere-impostor linear view depth (r32float, hardware z-test with per-pixel frag depth)
//   2. particles → additive thickness + bioluminescence (half-res rgba16f)
//   3. separable bilateral smoothing of depth (screen-space radius ∝ 1/depth), N iterations
//   4. composite: normals from smoothed depth, Fresnel reflection of the baked nebula, refraction of the analytic
//      floor/sky through the fluid with Beer–Lambert absorption, moon specular, glow, analytic glass orb on top.
import * as THREE from 'three/webgpu';
import {
	Fn, uniform, uv, vec2, vec3, vec4, float, int, ivec2, texture, textureLoad, screenUV, positionView, cameraProjectionMatrix,
	Discard, If, select, dot, cross, normalize, reflect, refract, sqrt, max, min, abs, clamp, exp, pow, mix, smoothstep, floor, length,
} from 'three/tsl';
import { ORB_C, ORB_R, MOON_DIR, GLOW_COLOR, makeEnv, makeBackground, raySphere } from './env.js';

const TAPS = 7; // per side

export class FluidRenderer {

	constructor( renderer, camera, U, envRT, { fluidScale = 1, thickScale = 0.5, iterations = 2 } = {} ) {

		this.renderer = renderer;
		this.camera = camera;
		this.U = U;
		this.envRT = envRT;
		this.fluidScale = fluidScale;
		this.thickScaleRes = thickScale;
		this.iterations = iterations;
		this._state = {};

		const rtOpt = { type: THREE.FloatType, format: THREE.RedFormat, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false };
		this.depthRT = new THREE.RenderTarget( 1, 1, { ...rtOpt, depthBuffer: true } );
		this.blurA = new THREE.RenderTarget( 1, 1, rtOpt );
		this.blurB = new THREE.RenderTarget( 1, 1, rtOpt );
		this.thickRT = new THREE.RenderTarget( 1, 1, { type: THREE.HalfFloatType, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: false } );
		for ( const rt of [ this.depthRT, this.blurA, this.blurB, this.thickRT ] ) rt.texture.generateMipmaps = false;

		// view / projection uniforms (the composite runs in a fullscreen quad, so camera builtins are not usable there)
		this.u = {
			tan: uniform( new THREE.Vector2( 1, 1 ) ),
			camMat: uniform( new THREE.Matrix4() ),
			camPos: uniform( new THREE.Vector3() ),
			res: uniform( new THREE.Vector2( 1, 1 ) ),
			proj: uniform( 500 ), // pixels per world unit at view depth 1 (fluid RT resolution)
			radius: uniform( 0.02 ),
			map: uniform( new THREE.Vector4( 32, 0.03, 0, 0 ) ), // grid centre, dx (world)
			thickNorm: uniform( 1 ),
			filterWorld: uniform( 0.06 ),
			depthSigma: uniform( 0.05 ),
			maxPx: uniform( 18 ),
		};

		this.depthScene = new THREE.Scene();
		this.thickScene = new THREE.Scene();
		this.quad = new THREE.QuadMesh();

		this._buildBlur();

	}

	// ── particle splat materials (rebuilt whenever the simulation is rebuilt) ──
	setSim( sim ) {

		this._disposeSprites();
		const { u } = this;
		const orbC = vec3( ORB_C.x, ORB_C.y, ORB_C.z );
		u.map.value.set( sim.center, ORB_R / sim.radius, 0, 0 );
		const spacingW = sim.spacing * ORB_R / sim.radius;
		this.spacingW = spacingW;
		u.radius.value = spacingW * this.U.radiusMul.value;

		const posAttr = sim.pos.toAttribute();
		const velAttr = sim.vel.toAttribute();
		const wpos = posAttr.xyz.sub( u.map.x ).mul( u.map.y ).add( orbC );

		// depth
		const dm = new THREE.SpriteNodeMaterial( { transparent: false, depthTest: true, depthWrite: true } );
		dm.blending = THREE.NoBlending;
		dm.positionNode = wpos;
		dm.scaleNode = u.radius.mul( 2 );
		const c = uv().mul( 2 ).sub( 1 );
		const r2 = dot( c, c );
		const nz = sqrt( max( float( 1 ).sub( r2 ), 0.0 ) );
		const surf = vec3( positionView.xy, positionView.z.add( nz.mul( u.radius ) ) );
		const clip = cameraProjectionMatrix.mul( vec4( surf, 1 ) );
		dm.depthNode = clip.z.div( clip.w );
		dm.fragmentNode = Fn( () => {

			Discard( r2.greaterThan( 1.0 ) );
			return vec4( surf.z.negate(), 0, 0, 1 );

		} )();

		// thickness + glow (sphere chord length, additive)
		const tm = new THREE.SpriteNodeMaterial( { transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending } );
		const tScale = 1.35;
		tm.positionNode = wpos;
		tm.scaleNode = u.radius.mul( 2 * tScale );
		const glowA = velAttr.w;
		tm.fragmentNode = Fn( () => {

			const cc = uv().mul( 2 ).sub( 1 );
			const rr = dot( cc, cc );
			Discard( rr.greaterThan( 1.0 ) );
			const chord = sqrt( float( 1 ).sub( rr ) ).mul( u.radius.mul( 2 * tScale ) );
			return vec4( chord, chord.mul( glowA ), 0, 1 );

		} )();
		const rs = spacingW * this.U.radiusMul.value * tScale;
		u.thickNorm.value = ( spacingW ** 3 ) / ( 4 / 3 * Math.PI * rs ** 3 );

		this.depthSprite = new THREE.Sprite( dm );
		this.thickSprite = new THREE.Sprite( tm );
		for ( const s of [ this.depthSprite, this.thickSprite ] ) {

			s.count = sim.count;
			s.frustumCulled = false;

		}

		this.depthScene.add( this.depthSprite );
		this.thickScene.add( this.thickSprite );
		this.sim = sim;

	}

	_disposeSprites() {

		for ( const s of [ this.depthSprite, this.thickSprite ] ) {

			if ( ! s ) continue;
			s.removeFromParent();
			s.material.dispose();

		}

		this.depthSprite = this.thickSprite = null;

	}

	// ── bilateral depth smoothing ──
	_buildBlur() {

		const { u } = this;
		const make = ( srcTex, dir ) => {

			const m = new THREE.NodeMaterial();
			m.fragmentNode = Fn( () => {

				const tc = floor( uv().mul( u.res ) ).toVar();
				const c = textureLoad( srcTex, ivec2( tc ) ).x.toVar();
				const out = c.toVar();
				If( c.greaterThan( 0.0 ), () => {

					const radiusPx = clamp( u.filterWorld.mul( u.proj ).div( c ), 1.0, u.maxPx );
					const stepPx = radiusPx.div( TAPS );
					const sum = c.toVar(), wsum = float( 1 ).toVar();
					const inv = float( 1 ).div( u.depthSigma );
					for ( let i = - TAPS; i <= TAPS; i ++ ) {

						if ( i === 0 ) continue;
						const ws = Math.exp( - ( i * i ) / ( 2 * ( TAPS * 0.5 ) ** 2 ) );
						const p = clamp( tc.add( vec2( dir[ 0 ], dir[ 1 ] ).mul( stepPx.mul( i ) ) ), vec2( 0 ), u.res.sub( 1 ) );
						const s = textureLoad( srcTex, ivec2( p ) ).x;
						const dz = s.sub( c ).mul( inv );
						const w = select( s.greaterThan( 0.0 ), exp( dz.mul( dz ).negate() ).mul( ws ), 0.0 );
						sum.addAssign( s.mul( w ) );
						wsum.addAssign( w );

					}

					out.assign( sum.div( wsum ) );

				} );
				return vec4( out, 0, 0, 1 );

			} )();
			return m;

		};

		this.blurH0 = make( this.depthRT.texture, [ 1, 0 ] );
		this.blurV = make( this.blurA.texture, [ 0, 1 ] );
		this.blurHn = make( this.blurB.texture, [ 1, 0 ] );

	}

	setSize( w, h ) {

		const fw = Math.max( 2, Math.round( w * this.fluidScale ) ), fh = Math.max( 2, Math.round( h * this.fluidScale ) );
		this.depthRT.setSize( fw, fh );
		this.blurA.setSize( fw, fh );
		this.blurB.setSize( fw, fh );
		this.thickRT.setSize( Math.max( 2, Math.round( w * this.thickScaleRes ) ), Math.max( 2, Math.round( h * this.thickScaleRes ) ) );
		this.u.res.value.set( fw, fh );
		this.fh = fh;
		this.outH = h;

	}

	updateCamera() {

		const cam = this.camera;
		cam.updateMatrixWorld();
		const t = Math.tan( THREE.MathUtils.degToRad( cam.fov ) * 0.5 );
		this.u.tan.value.set( t * cam.aspect, t );
		this.u.camMat.value.copy( cam.matrixWorld );
		this.u.camPos.value.setFromMatrixPosition( cam.matrixWorld );
		this.u.proj.value = this.fh / ( 2 * t );
		this.U.pixAngle.value = ( 2 * t ) / Math.max( 1, this.outH );
		this.u.radius.value = this.spacingW * this.U.radiusMul.value;
		this.u.filterWorld.value = this.u.radius.value * this.U.smoothing.value;
		this.u.depthSigma.value = this.u.radius.value * 2.2;

	}

	render() {

		const r = this.renderer;
		this.updateCamera();
		this._state = THREE.RendererUtils.resetRendererState( r, this._state );
		r.setClearColor( 0x000000, 0 );

		r.setRenderTarget( this.depthRT );
		r.render( this.depthScene, this.camera );
		r.setRenderTarget( this.thickRT );
		r.render( this.thickScene, this.camera );

		for ( let i = 0; i < this.iterations; i ++ ) {

			this.quad.material = i === 0 ? this.blurH0 : this.blurHn;
			r.setRenderTarget( this.blurA );
			this.quad.render( r );
			this.quad.material = this.blurV;
			r.setRenderTarget( this.blurB );
			this.quad.render( r );

		}

		THREE.RendererUtils.restoreRendererState( r, this._state );

	}

	// ── final composite (vec4 HDR) ──
	compositeNode() {

		const { u, U } = this;
		const E = makeEnv( this.envRT.texture );
		const bgPrimary = makeBackground( E, U, true );
		const bgSecondary = makeBackground( E, U, false );
		const orbC = vec3( ORB_C.x, ORB_C.y, ORB_C.z );
		const moon = vec3( MOON_DIR.x, MOON_DIR.y, MOON_DIR.z );
		const glowC = vec3( GLOW_COLOR.r, GLOW_COLOR.g, GLOW_COLOR.b );
		const depthTex = this.blurB.texture;
		const thickTex = this.thickRT.texture;

		const viewPos = ( tc, d ) => {

			const q = tc.add( 0.5 ).div( u.res );
			return vec3( q.x.mul( 2 ).sub( 1 ).mul( u.tan.x ).mul( d ), q.y.mul( - 2 ).add( 1 ).mul( u.tan.y ).mul( d ), d.negate() );

		};

		const load = ( tc ) => textureLoad( depthTex, ivec2( clamp( tc, vec2( 0 ), u.res.sub( 1 ) ) ) ).x;
		const schlick = ( f0, cosT ) => float( f0 ).add( float( 1 - f0 ).mul( pow( float( 1 ).sub( clamp( cosT, 0, 1 ) ), 5.0 ) ) );

		return Fn( () => {

			const suv = screenUV;
			const ndc = vec2( suv.x.mul( 2 ).sub( 1 ), suv.y.mul( - 2 ).add( 1 ) );
			const rd = normalize( u.camMat.mul( vec4( ndc.x.mul( u.tan.x ), ndc.y.mul( u.tan.y ), - 1, 0 ) ).xyz ).toVar();
			const ro = u.camPos;
			const V = rd.negate();

			const col = bgPrimary( ro, rd ).toVar();
			const orb = raySphere( ro, rd, orbC, ORB_R ).toVar();
			const inOrb = orb.x.greaterThan( 0.0 );

			const tg = texture( thickTex, suv ).toVar();
			const T = tg.x.mul( u.thickNorm ).toVar();
			const Gl = tg.y.mul( u.thickNorm ).toVar();

			If( inOrb, () => {

				// back wall of the glass (seen from inside): faint tint + reflection
				const nb = normalize( ro.add( rd.mul( orb.y ) ).sub( orbC ) );
				const Fb = schlick( 0.04, abs( dot( rd, nb ) ) ).mul( 0.6 );
				col.assign( col.mul( vec3( 0.88, 0.95, 0.97 ) ).mul( float( 1 ).sub( Fb ) ).add( E.env( reflect( rd, nb.negate() ) ).mul( Fb ) ) );

				// fluid surface
				const tc = floor( suv.mul( u.res ) ).toVar();
				const d = load( tc ).toVar();
				If( d.greaterThan( 0.0 ), () => {

					const pc = viewPos( tc, d );
					const dl = load( tc.sub( vec2( 1, 0 ) ) ), dr = load( tc.add( vec2( 1, 0 ) ) );
					const du = load( tc.sub( vec2( 0, 1 ) ) ), dd = load( tc.add( vec2( 0, 1 ) ) );
					const dxr = viewPos( tc.add( vec2( 1, 0 ) ), dr ).sub( pc ), dxl = pc.sub( viewPos( tc.sub( vec2( 1, 0 ) ), dl ) );
					const dyd = viewPos( tc.add( vec2( 0, 1 ) ), dd ).sub( pc ), dyu = pc.sub( viewPos( tc.sub( vec2( 0, 1 ) ), du ) );
					const useR = dr.greaterThan( 0.0 ).and( dl.lessThanEqual( 0.0 ).or( abs( dxr.z ).lessThan( abs( dxl.z ) ) ) );
					const useD = dd.greaterThan( 0.0 ).and( du.lessThanEqual( 0.0 ).or( abs( dyd.z ).lessThan( abs( dyu.z ) ) ) );
					const ddx = select( useR, dxr, dxl );
					const ddy = select( useD, dyd, dyu );
					const nV = normalize( cross( ddy, ddx ) );
					const n = normalize( u.camMat.mul( vec4( nV, 0 ) ).xyz ).toVar();
					const P = u.camMat.mul( vec4( pc, 1 ) ).xyz.toVar();

					const NdV = clamp( dot( n, V ), 0.0, 1.0 );
					const F = schlick( 0.02, NdV ).toVar();
					const refl = E.env( reflect( rd, n ) ).toVar();

					// moon specular (sharp + broad lobe)
					const H = normalize( moon.add( V ) );
					const nh = max( dot( n, H ), 0.0 );
					const spec = vec3( 0.85, 0.93, 1.0 ).mul( pow( nh, 1400.0 ).mul( 28.0 ).add( pow( nh, 120.0 ).mul( 0.9 ) ) ).mul( schlick( 0.02, dot( V, H ) ).mul( 8.0 ).add( 0.15 ) );

					// refraction through the slab, Beer–Lambert absorption, teal in-scattering
					const tr = refract( rd, n, 1.0 / 1.333 );
					const bgR = bgSecondary( P.add( tr.mul( T.mul( 0.5 ) ) ), tr ).mul( vec3( 0.9, 0.96, 0.98 ) );
					const sigma = vec3( 2.6, 0.75, 0.55 ).mul( U.absorb );
					const absorb = exp( sigma.mul( T ).negate() );
					const scatter = vec3( 0.0, 0.020, 0.028 ).mul( float( 1 ).sub( exp( T.mul( - 1.2 ) ) ) )
						.add( glowC.mul( Gl ).mul( U.glow ).mul( 0.6 ) );
					const trans = bgR.mul( absorb ).add( scatter );

					col.assign( mix( trans, refl, F ).add( spec ) );

				} );

			} );

			// bioluminescence (volumetric, sits into bloom); foam whitens at high intensity
			const g = Gl.mul( U.glow ).mul( 3.0 );
			const glowCol = mix( glowC, vec3( 0.75, 1.0, 0.95 ), smoothstep( 0.6, 3.0, g ) ).mul( g ).div( T.mul( 0.8 ).add( 1.0 ) );
			col.addAssign( glowCol );

			If( inOrb, () => {

				// front glass shell
				const nf = normalize( ro.add( rd.mul( orb.x ) ).sub( orbC ) ).toVar();
				const cosF = abs( dot( rd, nf ) );
				const Ff = schlick( 0.04, cosF ).toVar();
				const gR = E.env( reflect( rd, nf ) );
				const Hg = normalize( moon.add( V ) );
				const gs = pow( max( dot( nf, Hg ), 0.0 ), 3000.0 ).mul( 60.0 ).add( pow( max( dot( nf, Hg ), 0.0 ), 300.0 ).mul( 0.6 ) );
				const rim = pow( float( 1 ).sub( cosF ), 6.0 ).mul( 0.06 );
				col.assign( col.mul( float( 1 ).sub( Ff ) ).add( gR.mul( Ff ) ).add( vec3( 0.85, 0.93, 1.0 ).mul( gs ) ).add( vec3( 0.12, 0.35, 0.45 ).mul( rim ) ) );

			} );

			return vec4( col, 1 );

		} )();

	}

	dispose() {

		this._disposeSprites();
		for ( const m of [ this.blurH0, this.blurV, this.blurHn ] ) m.dispose();
		for ( const rt of [ this.depthRT, this.blurA, this.blurB, this.thickRT ] ) rt.dispose();

	}

}
