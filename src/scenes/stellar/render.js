// Rendering for "Ignition":
//  • volume:  emission–absorption raymarch of the plasma grid in local thermodynamic equilibrium:
//             dI/ds = κ(ρ,T) · (B(T) − I),  B = Planck radiance from a precomputed blackbody LUT,
//             integrated exactly per segment, interleaved-gradient-noise jittered, clipped by the analytic star.
//             Rendered at reduced resolution into its own HalfFloat target, then temporally accumulated.
//  • sky:     analytic star (ray–sphere), Eddington-approximation limb darkening T(μ) = Teff·(¾(μ+⅔))^¼ fed
//             through the same Planck LUT (so the limb reddens physically), animated Worley granulation,
//             sunspot pair at the footpoints of the field, chromospheric rim, K-corona with streamers, stars.
import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, Break, float, int, vec2, vec3, vec4, uniform, texture, texture3D, screenUV, screenCoordinate,
	normalize, length, dot, max, min, clamp, exp, sqrt, pow, mix, smoothstep, fwidth, atan, fract, floor,
} from 'three/tsl';
import { worley, granuleShade, vnoise, hash21, ign } from './noise.js';
import { GRAN_SCALE, GRAN_SPEED } from './fluid.js';

export function createRender( ctx, fluid, cfg ) {

	const { renderer } = ctx;
	const { origin, size, starR, blackbody } = cfg;

	const U = {
		invVP: uniform( new THREE.Matrix4() ),
		camPos: uniform( new THREE.Vector3() ),
		frame: uniform( 0 ),
		steps: uniform( cfg.steps, 'int' ),
		exposure: uniform( 1.0 ),
		gamma: uniform( 0.62 ),
		absorb: uniform( 5.0 ),
		T0: uniform( 2300 ),
		Tk: uniform( 4300 ),
		surf: uniform( 0.55 ),
		blend: uniform( 0.35 ),
		simTime: uniform( 0 ),
		q1: fluid.U.q1,
		q2: fluid.U.q2,
		flash: uniform( 0 ),
	};

	const starC = vec3( 0, - starR, 0 );
	const bmin = vec3( origin.x, origin.y, origin.z );
	const bsize = vec3( size.x, size.y, size.z );
	const bmax = bmin.add( bsize );

	const viewRay = () => {

		const ndc = vec2( screenUV.x.mul( 2 ).sub( 1 ), screenUV.y.mul( - 2 ).add( 1 ) );
		const hp = U.invVP.mul( vec4( ndc, 0.5, 1 ) );
		return normalize( hp.xyz.div( hp.w ).sub( U.camPos ) );

	};

	// ───────────── volume raymarch ─────────────
	const volumeFn = Fn( () => {

		const ro = U.camPos;
		const rd = viewRay().toVar();
		const inv = vec3( 1 ).div( rd.add( 1e-7 ) );
		const ta = bmin.sub( ro ).mul( inv ), tb = bmax.sub( ro ).mul( inv );
		const tmn = min( ta, tb ), tmx = max( ta, tb );
		const tn = max( max( tmn.x, tmn.y ), max( tmn.z, 0.0 ) ).toVar();
		const tf = min( min( tmx.x, tmx.y ), tmx.z ).toVar();

		// clip by the photosphere
		const oc = ro.sub( starC );
		const b = dot( oc, rd );
		const disc = b.mul( b ).sub( dot( oc, oc ).sub( starR * starR ) );
		const tS = b.negate().sub( sqrt( max( disc, 0 ) ) );
		If( disc.greaterThan( 0 ).and( tS.greaterThan( 0 ) ), () => {

			tf.assign( min( tf, tS ) );

		} );

		const col = vec3( 0 ).toVar();
		const tr = float( 1 ).toVar();

		If( tf.greaterThan( tn ), () => {

			const ds = tf.sub( tn ).div( float( U.steps ) ).toVar();
			const jit = ign( screenCoordinate.add( vec2( U.frame.mul( 5.588238 ), U.frame.mul( 3.13 ) ) ) );
			const t = tn.add( ds.mul( jit ) ).toVar();

			Loop( { start: int( 0 ), end: U.steps, type: 'int' }, () => {

				const p = ro.add( rd.mul( t ) );
				const uvw = p.sub( bmin ).div( bsize ).toVar();
				const vt = texture3D( fluid.VT, uvw ).level( 0 );
				const rho = texture3D( fluid.DN, uvw ).level( 0 ).x;
				const T = vt.w.toVar();

				// soft fade at the lateral / top boundaries of the simulation domain
				const edge = smoothstep( 0.0, 0.09, uvw.x ).mul( smoothstep( 1.0, 0.91, uvw.x ) )
					.mul( smoothstep( 0.0, 0.12, uvw.z ) ).mul( smoothstep( 1.0, 0.88, uvw.z ) )
					.mul( smoothstep( 1.0, 0.86, uvw.y ) );

				// cool plasma is the better absorber (neutral H / H⁻ opacity): dark filaments against hot gas
				const kap = rho.mul( U.absorb ).mul( edge ).mul( mix( 1.0, 2.6, smoothstep( 1.0, 0.3, T ) ) );
				const K = T.mul( U.Tk ).add( U.T0 );
				const Bt = blackbody( K, U.gamma );
				const a = float( 1 ).sub( exp( kap.mul( ds ).negate() ) );
				col.addAssign( Bt.mul( a.mul( tr ) ) );
				tr.mulAssign( float( 1 ).sub( a ) );
				t.addAssign( ds );

				If( tr.lessThan( 0.004 ), () => {

					Break();

				} );

			} );

		} );

		return vec4( col.mul( U.exposure ), tr );

	} );

	const volMat = new THREE.NodeMaterial();
	volMat.name = 'stellar.volume';
	volMat.fragmentNode = volumeFn();
	volMat.depthTest = false;
	volMat.depthWrite = false;
	const volQuad = new THREE.QuadMesh( volMat );

	// temporal accumulation (exponential history, no reprojection: the camera drift is slow)
	const rtOpts = { type: THREE.HalfFloatType, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
	const volRT = new THREE.RenderTarget( 2, 2, rtOpts );
	const accRT = [ new THREE.RenderTarget( 2, 2, rtOpts ), new THREE.RenderTarget( 2, 2, rtOpts ) ];
	const curTex = texture( volRT.texture );
	const histTex = texture( accRT[ 0 ].texture );
	const accMat = new THREE.NodeMaterial();
	accMat.name = 'stellar.accumulate';
	accMat.fragmentNode = Fn( () => {

		const cur = curTex.sample( screenUV );
		const hist = histTex.sample( screenUV );
		return mix( hist, cur, U.blend );

	} )();
	accMat.depthTest = false;
	accMat.depthWrite = false;
	const accQuad = new THREE.QuadMesh( accMat );
	let accIndex = 0;
	let accValid = false;
	const outTex = texture( accRT[ 1 ].texture ); // sampled by the composite; .value swapped each frame

	// ───────────── sky / photosphere (scene background) ─────────────
	const skyFn = Fn( () => {

		const ro = U.camPos;
		const rd = viewRay().toVar();
		const oc = ro.sub( starC ).toVar();
		const dist = length( oc );
		const b = dot( oc, rd ).toVar();
		const disc = b.mul( b ).sub( dot( oc, oc ).sub( starR * starR ) );

		// analytic, anti-aliased limb: signed angular distance to the disk edge
		const cosT = dot( rd, oc.negate().div( dist ) );
		const sinA = float( starR ).div( dist );
		const cosA = sqrt( max( float( 1 ).sub( sinA.mul( sinA ) ), 0 ) );
		const e = cosT.sub( cosA );
		const fw = max( fwidth( e ), 1e-6 );
		const cover = smoothstep( fw.negate(), fw, e );

		// ── photosphere ──
		const tHit = b.negate().sub( sqrt( max( disc, 0 ) ) );
		const P = ro.add( rd.mul( tHit ) ).toVar();
		const N = normalize( P.sub( starC ) );
		const mu = clamp( dot( N, rd.negate() ), 0, 1 );

		const gp = P.xz.div( GRAN_SCALE ).toVar();
		const pxw = length( fwidth( gp ) );
		const aa = smoothstep( 0.7, 0.18, pxw ); // fade granulation when cells shrink below ~2 px
		const w = worley( gp, U.simTime.mul( GRAN_SPEED ) );
		const gran = granuleShade( w );
		const fine = vnoise( gp.mul( 4.3 ) ).add( vnoise( gp.mul( 9.1 ).add( 3.7 ) ).mul( 0.5 ) );
		const aa2 = smoothstep( 0.18, 0.04, pxw );

		// sunspot pair at the field footpoints
		const spot = ( q ) => {

			const d = P.xz.sub( vec2( q.x, q.z ) );
			const r = length( d );
			const ang = atan( d.y, d.x );
			const fil = vnoise( vec2( ang.mul( 14 ), r.mul( 2.0 ) ) );
			const pen = smoothstep( 0.62, 0.36, r.add( fil.sub( 0.5 ).mul( 0.12 ) ) );
			const umb = smoothstep( 0.26, 0.14, r );
			return vec2( pen.mul( fil.mul( 0.35 ).add( 0.65 ) ), umb );

		};

		const s1 = spot( U.q1 ), s2 = spot( U.q2 );
		const pen = max( s1.x, s2.x ), umb = max( s1.y, s2.y );
		const quiet = float( 1 ).sub( max( pen, umb ) );
		const tVar = gran.sub( 0.55 ).mul( 0.085 ).mul( aa ).add( fine.sub( 0.75 ).mul( 0.018 ).mul( aa2 ) ).mul( quiet );
		const Teff = float( 5800 ).mul( float( 1 ).add( tVar ) ).mul( float( 1 ).sub( pen.mul( 0.16 ) ).sub( umb.mul( 0.22 ) ) );
		const Ts = Teff.mul( pow( mu.add( 2 / 3 ).mul( 0.75 ), 0.25 ) );
		const surfCol = blackbody( Ts, U.gamma ).mul( U.surf ).mul( U.flash.mul( 0.25 ).add( 1 ) );

		// ── sky ──
		const tca = max( b.negate(), 0 );
		const Q = oc.add( rd.mul( tca ) );
		const alt = max( length( Q ).sub( starR ), 0 ).toVar();
		const az = atan( Q.z, Q.x );
		const streak = vnoise( vec2( az.mul( 46 ), alt.mul( 0.08 ).sub( U.simTime.mul( 0.01 ) ) ) );
		const streak2 = vnoise( vec2( az.mul( 160 ), alt.mul( 0.4 ) ) );
		const spic = vnoise( vec2( az.mul( 900 ), 0.5 ) );
		const chrom = exp( alt.div( - 0.09 ) ).mul( spic.mul( 0.6 ).add( 0.6 ) ).mul( 1.1 );
		const corona = exp( alt.div( - 1.1 ) ).mul( 0.16 ).add( exp( alt.div( - 5.0 ) ).mul( 0.06 ) )
			.mul( streak.mul( 0.9 ).add( 0.4 ).mul( streak2.mul( 0.4 ).add( 0.8 ) ) );
		const coronaCol = vec3( 1.0, 0.86, 0.72 ).mul( corona ).add( vec3( 1.0, 0.22, 0.30 ).mul( chrom ) );

		// stars
		const sph = vec2( atan( rd.z, rd.x ), rd.y ).mul( vec2( 260, 330 ) );
		const cell = floor( sph );
		const hs = hash21( cell );
		const fp = fract( sph ).sub( 0.5 );
		const star = smoothstep( 0.9975, 1.0, hs ).mul( smoothstep( 0.35, 0.0, length( fp ) ) ).mul( 0.5 );
		const space = vec3( 0.0006, 0.0009, 0.0018 ).add( vec3( 0.8, 0.85, 1.0 ).mul( star ) );

		const skyCol = space.add( coronaCol );
		return vec4( mix( skyCol, surfCol, cover ), 1 );

	} );

	const skyNode = skyFn();

	// ───────────── composite (used by the chapter's post()) ─────────────
	const composite = ( sceneColor ) => {

		const vol = outTex.sample( screenUV );
		return sceneColor.rgb.mul( vol.a ).add( vol.rgb );

	};

	let rtW = 2, rtH = 2;

	return {
		U, skyNode, composite,
		setSize( w, h ) {

			rtW = Math.max( 2, Math.round( w ) ); rtH = Math.max( 2, Math.round( h ) );
			volRT.setSize( rtW, rtH );
			accRT[ 0 ].setSize( rtW, rtH );
			accRT[ 1 ].setSize( rtW, rtH );
			accValid = false;

		},
		get size() {

			return [ rtW, rtH ];

		},
		updateCamera( camera ) {

			camera.updateMatrixWorld();
			U.invVP.value.multiplyMatrices( camera.matrixWorld, camera.projectionMatrixInverse );
			U.camPos.value.setFromMatrixPosition( camera.matrixWorld );

		},
		renderVolume() {

			const prev = renderer.getRenderTarget();
			renderer.setRenderTarget( volRT );
			volQuad.render( renderer );
			// accumulate: hist = accRT[accIndex], write to the other
			const src = accRT[ accIndex ], dst = accRT[ 1 - accIndex ];
			histTex.value = accValid ? src.texture : volRT.texture;
			renderer.setRenderTarget( dst );
			accQuad.render( renderer );
			renderer.setRenderTarget( prev );
			outTex.value = dst.texture;
			accIndex = 1 - accIndex;
			accValid = true;
			U.frame.value = ( U.frame.value + 1 ) % 64;

		},
		sync() {

			return renderer.readRenderTargetPixelsAsync( accRT[ 1 - accIndex ], 0, 0, 2, 2 );

		},
		resetHistory() {

			accValid = false;

		},
		dispose() {

			volMat.dispose(); accMat.dispose();
			volRT.dispose(); accRT[ 0 ].dispose(); accRT[ 1 ].dispose();

		},
	};

}

