// Morphogenesis — fullscreen isosurface raymarcher for the baked 3D Gray–Scott field.
//
// Per pixel: ray/box clip → brick-distance empty-space skipping (sphere-tracing-like, conservative Chebyshev
// bound) → fine trilinear stepping with a static interleaved-gradient jitter → binary refinement of the v = thr
// crossing → gradient normals. Shading: GGX key + rim, analytic studio softbox environment, field AO (marching the
// field along the normal), key-light self shadowing and back-light transmission through the field (subsurface
// approximation), local thickness glow, thin-film interference (nacre / obsidian sheen), "fresh growth" emission.
// The floor is an analytic glossy plane with sphere-approximated soft shadow / AO and a raymarched reflection.
import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, Break, uniform, float, vec2, vec3, vec4, texture3D, positionGeometry, varying, cameraPosition,
	cameraProjectionMatrixInverse, cameraWorldMatrix, normalize, dot, max, min, mix, clamp, smoothstep, exp, pow,
	abs, reflect, length, sqrt, fract, cos, sin, screenCoordinate, select, PI,
} from 'three/tsl';
import { BRICK } from './sim.js';

export const MATERIALS = {
	nacre: {
		base: [ 0.80, 0.76, 0.72 ], sss: [ 1.0, 0.62, 0.48 ], sssAmt: 0.55, rough: 0.22, irid: 1.0, film: [ 280, 720 ],
		metal: 0.0, gold: 0.0, glow: [ 1.0, 0.45, 0.25 ], spec: 1.0,
	},
	jade: {
		base: [ 0.10, 0.36, 0.20 ], sss: [ 0.30, 1.0, 0.48 ], sssAmt: 1.6, rough: 0.12, irid: 0.12, film: [ 400, 600 ],
		metal: 0.0, gold: 0.0, glow: [ 0.55, 1.0, 0.35 ], spec: 1.0,
	},
	'obsidian-gold': {
		base: [ 0.012, 0.012, 0.014 ], sss: [ 0.5, 0.2, 0.08 ], sssAmt: 0.1, rough: 0.07, irid: 0.35, film: [ 500, 900 ],
		metal: 0.0, gold: 1.0, glow: [ 1.0, 0.55, 0.15 ], spec: 1.0,
	},
};

export function createRaymarch( sim, opts ) {

	const { G, volTex, brickTex, pick } = sim;
	const S = opts.halfSize; // half extent of the simulation box in world units
	const vox = ( 2 * S ) / G;
	const MAXSTEPS = opts.maxSteps;
	const REFL_STEPS = opts.reflSteps;
	const fine = opts.stepVox * vox;

	const bmin = vec3( - S, - S, - S ), bmax = vec3( S, S, S );
	const invSize = float( 1 / ( 2 * S ) );

	const m0 = MATERIALS.nacre;
	const u = {
		thr: sim.u.thr,
		floorY: uniform( opts.floorY ),
		base: uniform( new THREE.Color( ...m0.base ) ),
		sss: uniform( new THREE.Color( ...m0.sss ) ),
		sssAmt: uniform( m0.sssAmt ),
		rough: uniform( m0.rough ),
		irid: uniform( m0.irid ),
		filmMin: uniform( m0.film[ 0 ] ),
		filmMax: uniform( m0.film[ 1 ] ),
		gold: uniform( m0.gold ),
		glow: uniform( new THREE.Color( ...m0.glow ) ),
		glowAmt: uniform( 1.0 ),
		keyDir: uniform( new THREE.Vector3( - 0.55, 0.75, 0.45 ).normalize() ),
		keyCol: uniform( new THREE.Color( 1.0, 0.9, 0.78 ).multiplyScalar( 3.2 ) ),
		rimDir: uniform( new THREE.Vector3( 0.5, 0.35, - 0.8 ).normalize() ),
		rimCol: uniform( new THREE.Color( 0.55, 0.7, 1.0 ).multiplyScalar( 2.4 ) ),
		hover: uniform( 0 ),
		press: uniform( 0 ),
		time: uniform( 0 ),
		exposure: uniform( 1 ),
	};

	const SAMPLE = ( p ) => texture3D( volTex, p.sub( bmin ).mul( invSize ), float( 0 ) );
	const Vs = ( p ) => SAMPLE( p ).x;

	// interleaved gradient noise (static) for step jitter
	const ign = ( c ) => fract( float( 52.9829189 ).mul( fract( dot( c, vec2( 0.06711056, 0.00583715 ) ) ) ) );

	// ── march: returns vec4(t, hit, steps, 0) ──
	const marchFn = ( maxSteps, name ) => Fn( ( [ ro, rd, tmax, jit ] ) => {

		const rdS = select( abs( rd ).lessThan( vec3( 1e-6 ) ), vec3( 1e-6 ), rd );
		const inv = vec3( 1 ).div( rdS );
		const t0 = bmin.sub( ro ).mul( inv ), t1 = bmax.sub( ro ).mul( inv );
		const tn = max( max( min( t0.x, t1.x ), min( t0.y, t1.y ) ), max( min( t0.z, t1.z ), 0 ) ).toConst();
		const tf = min( min( max( t0.x, t1.x ), max( t0.y, t1.y ) ), min( max( t0.z, t1.z ), tmax ) ).toConst();
		const res = vec4( 1e9, 0, 0, 0 ).toVar();

		If( tn.lessThan( tf ), () => {

			const t = tn.add( jit.mul( fine ) ).toVar();
			const tp = tn.toVar();
			Loop( maxSteps, ( { i } ) => {

				If( t.greaterThan( tf ), () => {

					Break();

				} );
				const p = ro.add( rd.mul( t ) ).toConst();
				const uvw = p.sub( bmin ).mul( invSize ).toConst();
				const bd = texture3D( brickTex, uvw, float( 0 ) ).x.toConst();
				If( bd.greaterThan( 1.5 ), () => {

					// conservative skip: every voxel within (d − 1) bricks is empty
					tp.assign( t );
					t.addAssign( bd.sub( 1 ).mul( BRICK * vox ) );

				} ).Else( () => {

					const v = texture3D( volTex, uvw, float( 0 ) ).x;
					If( v.greaterThan( u.thr ), () => {

						const a = tp.toVar(), b = t.toVar();
						Loop( 6, () => {

							const m = a.add( b ).mul( 0.5 ).toConst();
							If( Vs( ro.add( rd.mul( m ) ) ).greaterThan( u.thr ), () => {

								b.assign( m );

							} ).Else( () => {

								a.assign( m );

							} );

						} );
						res.assign( vec4( b, 1, float( i ), 0 ) );
						Break();

					} );
					tp.assign( t );
					t.addAssign( fine );

				} );

			} );

		} );

		return res;

	} ).setLayout( {
		name,
		type: 'vec4',
		inputs: [
			{ name: 'ro', type: 'vec3' },
			{ name: 'rd', type: 'vec3' },
			{ name: 'tmax', type: 'float' },
			{ name: 'jit', type: 'float' },
		],
	} );

	const march = marchFn( MAXSTEPS, 'morphoMarch' );
	const marchRefl = marchFn( REFL_STEPS, 'morphoMarchRefl' );

	const normalAt = ( p ) => {

		const e = vox * 0.9;
		const dx = Vs( p.add( vec3( e, 0, 0 ) ) ).sub( Vs( p.sub( vec3( e, 0, 0 ) ) ) );
		const dy = Vs( p.add( vec3( 0, e, 0 ) ) ).sub( Vs( p.sub( vec3( 0, e, 0 ) ) ) );
		const dz = Vs( p.add( vec3( 0, 0, e ) ) ).sub( Vs( p.sub( vec3( 0, 0, e ) ) ) );
		return normalize( vec3( dx, dy, dz ).negate().add( vec3( 0, 1e-7, 0 ) ) );

	};

	// ── studio environment: dark room, large overhead softbox, tall strip on the left, cool rim strip behind ──
	const env = Fn( ( [ d, rough ] ) => {

		const w = rough.mul( 0.6 ).add( 0.04 ).toConst();
		const base = mix( vec3( 0.010, 0.009, 0.008 ), vec3( 0.03, 0.03, 0.034 ), smoothstep( - 0.3, 0.6, d.y ) ).toVar();
		// overhead softbox (slightly front-left)
		const top = smoothstep( float( 0.80 ).sub( w ), float( 0.86 ).add( w.mul( 0.3 ) ), dot( d, normalize( vec3( - 0.25, 1, 0.15 ) ) ) );
		base.addAssign( vec3( 1.0, 0.95, 0.88 ).mul( top ).mul( 2.6 ) );
		// tall strip on the key side
		const sd = normalize( vec3( - 0.85, 0.15, 0.5 ) );
		const across = abs( dot( d, normalize( vec3( 0.5, 0, 0.85 ) ) ) );
		const strip = smoothstep( float( 0.10 ).add( w ), float( 0.06 ), across ).mul( smoothstep( 0.15, 0.4, dot( d, sd ) ) ).mul( smoothstep( 0.75, 0.35, abs( d.y.sub( 0.15 ) ) ) );
		base.addAssign( vec3( 1.0, 0.92, 0.82 ).mul( strip ).mul( 1.8 ) );
		// cool rim strip behind-right
		const rd2 = normalize( vec3( 0.55, 0.2, - 0.8 ) );
		const across2 = abs( dot( d, normalize( vec3( 0.82, 0, 0.56 ) ) ) );
		const rim = smoothstep( float( 0.08 ).add( w ), float( 0.04 ), across2 ).mul( smoothstep( 0.2, 0.5, dot( d, rd2 ) ) ).mul( smoothstep( 0.7, 0.3, abs( d.y.sub( 0.2 ) ) ) );
		base.addAssign( vec3( 0.55, 0.72, 1.0 ).mul( rim ).mul( 1.6 ) );
		// floor bounce
		base.addAssign( vec3( 0.02, 0.018, 0.016 ).mul( smoothstep( 0.0, - 0.5, d.y ) ) );
		return base;

	} );

	const background = ( rd ) => {

		const hz = exp( abs( rd.y.sub( 0.02 ) ).mul( - 5.0 ) );
		return mix( vec3( 0.0035, 0.0038, 0.0048 ), vec3( 0.022, 0.021, 0.024 ), hz );

	};

	// thin-film interference (3 wavelengths), returns reflectance tint in ~[0, 1]
	const thinFilm = ( cosV, dnm ) => {

		const nF = 1.42;
		const sin2 = float( 1 ).sub( cosV.mul( cosV ) ).div( nF * nF );
		const cosT = sqrt( max( float( 1 ).sub( sin2 ), 0.0 ) );
		const opd = dnm.mul( 2 * nF ).mul( cosT ); // optical path difference, nm
		const ph = vec3( opd ).div( vec3( 650, 532, 450 ) ).mul( 2 * Math.PI );
		return vec3( 0.5 ).sub( cos( ph ).mul( 0.5 ) );

	};

	const ggx = ( n, v, l, rough ) => {

		const h = normalize( v.add( l ) );
		const ndh = max( dot( n, h ), 0 ), ndl = max( dot( n, l ), 0 ), ndv = max( dot( n, v ), 1e-3 );
		const a = rough.mul( rough ), a2 = a.mul( a );
		const dd = ndh.mul( ndh ).mul( a2.sub( 1 ) ).add( 1 );
		const D = a2.div( dd.mul( dd ).mul( Math.PI ) );
		const k = a.mul( 0.5 );
		const vis = float( 0.25 ).div( ndl.mul( float( 1 ).sub( k ) ).add( k ).mul( ndv.mul( float( 1 ).sub( k ) ).add( k ) ) );
		return D.mul( vis ).mul( ndl );

	};

	const schlick = ( f0, c ) => f0.add( float( 1 ).sub( f0 ).mul( pow( float( 1 ).sub( c ), 5 ) ) );

	const pickWorld = () => pick.element( 0 ).xyz.mul( vox ).add( bmin );

	// ── full sculpture shading ──
	const shade = ( p, rd ) => {

		const n = normalAt( p ).toConst();
		const V = rd.negate().toConst();
		const ndv = max( dot( n, V ), 1e-3 ).toConst();
		const s4 = SAMPLE( p ).toConst();
		const fresh = s4.y, uSub = s4.z;

		// ambient occlusion from the field itself: material found along the normal occludes
		const occ = float( 0 ).toVar();
		[ [ 1.5, 0.30 ], [ 3.0, 0.24 ], [ 5.0, 0.18 ], [ 8.0, 0.14 ], [ 13.0, 0.10 ] ].forEach( ( [ d, w ] ) => {

			occ.addAssign( smoothstep( u.thr.mul( 0.4 ), u.thr.mul( 1.4 ), Vs( p.add( n.mul( d * vox ) ) ) ).mul( w ) );

		} );
		const ao = clamp( float( 1 ).sub( occ.mul( 1.35 ) ), 0, 1 ).toConst();

		// key-light self shadow: optical depth of the field towards the key light
		const L = u.keyDir;
		const depthK = float( 0 ).toVar();
		[ [ 2, 2 ], [ 4, 2 ], [ 6.5, 2.5 ], [ 9.5, 3 ], [ 13, 3.5 ], [ 18, 5 ], [ 25, 7 ] ].forEach( ( [ d, ds ] ) => {

			depthK.addAssign( smoothstep( u.thr.mul( 0.8 ), u.thr.mul( 1.3 ), Vs( p.add( n.mul( vox ) ).add( L.mul( d * vox ) ) ) ).mul( ds ) );

		} );
		const shadow = exp( depthK.mul( - 0.45 ) ).toConst();

		// back-light transmission (subsurface): optical depth through the body towards the rim light
		const R = u.rimDir;
		const depthR = float( 0 ).toVar();
		[ [ 1, 1 ], [ 2.5, 1.5 ], [ 4.5, 2 ], [ 7, 2.5 ], [ 10, 3 ] ].forEach( ( [ d, ds ] ) => {

			depthR.addAssign( smoothstep( u.thr.mul( 0.7 ), u.thr.mul( 1.4 ), Vs( p.add( R.mul( d * vox ) ) ) ).mul( ds ) );

		} );
		// local thickness along −n (thin fins glow)
		const depthN = float( 0 ).toVar();
		[ [ 1.5, 1.5 ], [ 3.5, 2 ], [ 6, 2.5 ], [ 9, 3 ] ].forEach( ( [ d, ds ] ) => {

			depthN.addAssign( smoothstep( u.thr.mul( 0.7 ), u.thr.mul( 1.4 ), Vs( p.sub( n.mul( d * vox ) ) ) ).mul( ds ) );

		} );

		// material
		const pn = p.div( S ).toConst();
		const goldMask = smoothstep( 0.05, - 0.25, pn.y.add( sin( pn.x.mul( 3.1 ).add( pn.z.mul( 2.3 ) ) ).mul( 0.08 ) ) ).mul( u.gold ).toConst();
		const filmD = mix( u.filmMin, u.filmMax, fract( uSub.mul( 1.7 ).add( pn.y.mul( 0.6 ) ).add( dot( pn, vec3( 0.21, 0.13, - 0.17 ) ) ) ) );
		const film = thinFilm( ndv, filmD );
		const iridTint = mix( vec3( 1 ), film.mul( 1.6 ).add( 0.15 ), u.irid ).toConst();
		const goldF0 = vec3( 1.0, 0.72, 0.32 );
		const f0 = mix( vec3( 0.045 ).mul( iridTint ), goldF0, goldMask ).toConst();
		const albedo = mix( vec3( u.base ), vec3( 0 ), goldMask ).toConst();
		const rough = mix( u.rough, float( 0.22 ), goldMask ).toConst();

		// diffuse
		const ndl = dot( n, L );
		const wrap = max( ndl.add( 0.25 ).div( 1.25 ), 0 );
		const amb = mix( vec3( 0.022, 0.02, 0.019 ), vec3( 0.07, 0.07, 0.075 ), n.y.mul( 0.5 ).add( 0.5 ) );
		const diffuse = albedo.mul( u.keyCol.mul( wrap ).mul( shadow ).add( amb.mul( ao ).mul( 2.2 ) ) ).toVar();

		// subsurface: back-lit transmission + thickness glow, tinted by the scattering colour
		const trans = exp( depthR.mul( - 0.28 ) );
		const phase = pow( max( dot( rd, R ), 0 ), 2 ).mul( 0.7 ).add( 0.3 );
		const sssLight = u.rimCol.mul( trans ).mul( phase ).mul( 1.2 )
			.add( u.keyCol.mul( exp( depthK.mul( - 0.12 ) ) ).mul( max( ndl.negate(), 0 ).mul( 0.6 ).add( 0.15 ) ).mul( 0.35 ) )
			.add( vec3( 0.12 ).mul( exp( depthN.mul( - 0.22 ) ) ).mul( ao ) );
		diffuse.addAssign( vec3( u.sss ).mul( sssLight ).mul( u.sssAmt ).mul( float( 1 ).sub( goldMask ) ) );

		// specular: key + rim lobes and the softbox environment
		const Fv = schlick( f0, ndv );
		const spec = Fv.mul( u.keyCol.mul( ggx( n, V, L, rough ) ).mul( shadow ) ).add( Fv.mul( u.rimCol.mul( ggx( n, V, R, rough ) ) ) ).toVar();
		const refl = reflect( rd, n );
		const specOcc = clamp( ao.mul( ao ).mul( 1.2 ), 0, 1 );
		spec.addAssign( env( refl, rough ).mul( Fv ).mul( specOcc ) );

		const col = diffuse.add( spec ).toVar();

		// fresh growth: bioluminescent emission that cools as the structure settles
		col.addAssign( vec3( u.glow ).mul( smoothstep( 0.0, 1.0, fresh ) ).mul( u.glowAmt ).mul( 2.5 ) );

		// pointer: soft halo where the activator will be injected
		const pw = pickWorld();
		const dp = length( p.sub( pw ) ).div( vox * 7 );
		const halo = exp( dp.mul( dp ).negate() ).mul( u.hover.mul( 0.35 ).add( u.press.mul( 1.4 ) ) );
		col.addAssign( vec3( u.glow ).mul( halo ).mul( 1.5 ) );

		return col;

	};

	// ── cheap shading for the floor reflection ──
	const shadeLite = ( p, rd ) => {

		const n = normalAt( p ).toConst();
		const s4 = SAMPLE( p ).toConst();
		const ndv = max( dot( n, rd.negate() ), 1e-3 );
		const pn = p.div( S );
		const goldMask = smoothstep( 0.05, - 0.25, pn.y.add( sin( pn.x.mul( 3.1 ).add( pn.z.mul( 2.3 ) ) ).mul( 0.08 ) ) ).mul( u.gold );
		const f0 = mix( vec3( 0.045 ), vec3( 1.0, 0.72, 0.32 ), goldMask );
		const albedo = mix( vec3( u.base ), vec3( 0 ), goldMask );
		const ao = clamp( float( 1 ).sub( smoothstep( u.thr.mul( 0.4 ), u.thr.mul( 1.4 ), Vs( p.add( n.mul( 4 * vox ) ) ) ) ), 0.2, 1 );
		const ndl = max( dot( n, u.keyDir ), 0 );
		const amb = mix( vec3( 0.02 ), vec3( 0.07 ), n.y.mul( 0.5 ).add( 0.5 ) );
		const c = albedo.mul( u.keyCol.mul( ndl ).add( amb.mul( 2 ) ) ).mul( ao )
			.add( vec3( u.sss ).mul( u.sssAmt ).mul( 0.1 ) )
			.add( env( reflect( rd, n ), u.rough ).mul( schlick( f0, ndv ) ).mul( ao ) )
			.add( vec3( u.glow ).mul( smoothstep( 0, 1, s4.y ) ).mul( u.glowAmt ).mul( 2.5 ) );
		return c;

	};

	// analytic soft shadow / occlusion of the sculpture's bounding sphere (iq)
	const sphC = vec3( 0, 0, 0 ), sphR = float( opts.sphereR );
	const sphSoftShadow = ( ro, rd, k ) => {

		const oc = ro.sub( sphC );
		const b = dot( oc, rd );
		const c = dot( oc, oc ).sub( sphR.mul( sphR ) );
		const h = b.mul( b ).sub( c );
		const d = sqrt( max( sphR.mul( sphR ).sub( h ), 0 ) ).sub( sphR );
		const t = b.negate().sub( sqrt( max( h, 0 ) ) );
		return select( t.lessThan( 0 ), float( 1 ), smoothstep( 0, 1, d.mul( k ).mul( 2.5 ).div( t ) ) );

	};
	const sphOcclusion = ( pos, nor ) => {

		const di = sphC.sub( pos );
		const l = length( di );
		const nl = dot( nor, di.div( l ) );
		const h = l.div( sphR );
		return max( nl, 0 ).div( h.mul( h ) );

	};

	const vNdc = varying( positionGeometry.xy, 'vMorphoNdc' );

	const fragment = Fn( () => {

		const vp = cameraProjectionMatrixInverse.mul( vec4( vNdc, 0.5, 1 ) );
		const rdView = normalize( vp.xyz.div( vp.w ) );
		const rd = normalize( cameraWorldMatrix.mul( vec4( rdView, 0 ) ).xyz ).toConst();
		const ro = cameraPosition.toConst();
		const jit = ign( screenCoordinate.xy ).toConst();

		const col = background( rd ).toVar();
		const tFloor = select( rd.y.lessThan( - 1e-4 ), u.floorY.sub( ro.y ).div( rd.y ), float( 1e9 ) ).toConst();
		const hit = march( ro, rd, tFloor, jit ).toConst();

		If( hit.y.greaterThan( 0.5 ), () => {

			col.assign( shade( ro.add( rd.mul( hit.x ) ).toConst(), rd ) );

		} ).ElseIf( tFloor.lessThan( 1e8 ), () => {

			const pf = ro.add( rd.mul( tFloor ) ).toConst();
			const up = vec3( 0, 1, 0 );
			const r = length( pf.xz );
			// lacquered dark floor: key light with a soft sphere shadow, sky occlusion under the sculpture
			const sh = sphSoftShadow( pf, u.keyDir, float( 3.0 ) );
			const ao = clamp( float( 1 ).sub( sphOcclusion( pf, up ).mul( 0.9 ) ), 0, 1 );
			const alb = vec3( 0.05, 0.048, 0.046 );
			const fl = alb.mul( u.keyCol.mul( u.keyDir.y ).mul( sh ).mul( 0.6 ).add( vec3( 0.05 ).mul( ao ) ) ).mul( ao ).toVar();
			// raymarched soft reflection of the sculpture
			const rr = reflect( rd, up );
			const rh = marchRefl( pf.add( up.mul( 0.002 ) ), rr, float( 8 ), jit ).toConst();
			const fres = schlick( float( 0.03 ), max( rd.y.negate(), 0 ) );
			If( rh.y.greaterThan( 0.5 ), () => {

				const ph = pf.add( rr.mul( rh.x ) );
				const height = ph.y.sub( u.floorY );
				const fadeR = exp( height.mul( - 1.6 ) ).mul( 0.85 ).add( 0.15 );
				fl.addAssign( shadeLite( ph.toConst(), rr ).mul( fres.mul( 3.0 ).add( 0.08 ) ).mul( fadeR ) );

			} ).Else( () => {

				fl.addAssign( background( rr ).mul( fres ) );

			} );
			col.assign( mix( fl, background( rd ), smoothstep( 2.5, 9.0, r ) ) );

		} );

		return vec4( col.mul( u.exposure ), 1 );

	} );

	const material = new THREE.MeshBasicNodeMaterial( { depthTest: false, depthWrite: false } );
	material.vertexNode = vec4( positionGeometry.xy, 0, 1 );
	material.fragmentNode = fragment();

	const geometry = new THREE.PlaneGeometry( 2, 2 );
	const mesh = new THREE.Mesh( geometry, material );
	mesh.frustumCulled = false;

	return {
		mesh, u, vox, S,
		dispose() {

			geometry.dispose();
			material.dispose();

		},
	};

}
