// Half-resolution volumetric raymarch + temporal resolve for the nebula.
import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, Break, vec2, vec3, vec4, float, int, screenUV, screenCoordinate,
	length, normalize, smoothstep, mix, max, min, exp, abs, dot, clamp, pow, sqrt, fract, sin, select, inverseSqrt,
	interleavedGradientNoise,
} from 'three/tsl';
import { CLUSTER, JETS, MAX_USER_STARS, ssr } from './volume.js';

const v3 = ( a ) => vec3( a[ 0 ], a[ 1 ], a[ 2 ] );

// Henyey–Greenstein, normalised so that the isotropic phase == 1
const hg = ( c, g ) => {

	const g2 = g * g;
	return float( 1 - g2 ).div( pow( float( 1 + g2 ).sub( c.mul( 2 * g ) ), 1.5 ) );

};

const phase2 = ( c, k ) => mix( hg( c, 0.6 * k ), hg( c, - 0.3 * k ), 0.22 );

export const PALETTES = {
	// [ natural, hubble ]
	// SHO mapping (S II→red, Hα→green, O III→blue), graded toward gold rims / teal gas
	ha: [ [ 1.0, 0.12, 0.30 ], [ 0.55, 0.85, 0.35 ] ],
	sii: [ [ 0.80, 0.06, 0.05 ], [ 1.0, 0.25, 0.05 ] ],
	oiii: [ [ 0.06, 0.60, 0.72 ], [ 0.05, 0.45, 1.0 ] ],
	scat: [ [ 0.50, 0.60, 0.80 ], [ 0.60, 0.40, 0.25 ] ],
	jet: [ [ 1.0, 0.20, 0.12 ], [ 1.0, 0.42, 0.14 ] ],
	amb: [ [ 0.020, 0.014, 0.030 ], [ 0.020, 0.018, 0.022 ] ],
};

const palCol = ( key, pal ) => mix( v3( PALETTES[ key ][ 0 ] ), v3( PALETTES[ key ][ 1 ] ), pal );

export function buildMarchMaterial( vol, U ) {

	const { densNode, lightNode, detailNode, bmin, bsize, invSize } = vol;
	const C = v3( CLUSTER.toArray() );
	const redden = vec3( 0.72, 1.0, 1.42 ); // dust extinction is steeper in the blue -> transmitted light reddens

	const marchFn = Fn( () => {

		const uv = screenUV;
		const ndc = vec2( uv.x.mul( 2 ).sub( 1 ), uv.y.mul( - 2 ).add( 1 ) );
		const vp = U.invProj.mul( vec4( ndc, 1.0, 1.0 ) );
		const rd = normalize( U.camWorld.mul( vec4( vp.xyz.div( vp.w ), 0.0 ) ).xyz ).toVar();
		const ro = U.camPos;

		const inv = vec3( 1.0 ).div( rd );
		const t0 = bmin.sub( ro ).mul( inv ), t1 = bmin.add( bsize ).sub( ro ).mul( inv );
		const tmn = min( t0, t1 ), tmx = max( t0, t1 );
		const tN = max( max( tmn.x, tmn.y ), max( tmn.z, 0.0 ) ).toVar();
		const tF = min( tmx.x, min( tmx.y, tmx.z ) ).toVar();

		const acc = vec3( 0 ).toVar();
		const T = float( 1 ).toVar();
		const dSum = float( 0 ).toVar();
		const wSum = float( 0 ).toVar();

		const pal = U.palette;
		const cHa = palCol( 'ha', pal ).toVar(), cS = palCol( 'sii', pal ).toVar(), cO = palCol( 'oiii', pal ).toVar();
		const cScat = palCol( 'scat', pal ).toVar(), cJet = palCol( 'jet', pal ).toVar(), cAmb = palCol( 'amb', pal ).toVar();
		const kEx = U.density.mul( U.extinction ).toVar();

		If( tF.greaterThan( tN ), () => {

			const dt = tF.sub( tN ).div( float( U.steps ) ).toVar();
			const jit = fract( interleavedGradientNoise( screenCoordinate.xy ).add( U.frame.mul( 0.61803398875 ) ) );

			Loop( { start: int( 0 ), end: U.steps, type: 'int', condition: '<' }, ( { i } ) => {

				const t = tN.add( float( i ).add( jit ).mul( dt ) ).toVar();
				const p = ro.add( rd.mul( t ) ).toVar();
				const uvw = p.sub( bmin ).mul( invSize ).toVar();
				const D = densNode.sample( uvw ).level( 0 ).toVar();
				const rho0 = D.r.mul( U.density ).toVar();

				If( rho0.greaterThan( 0.0012 ), () => {

					// sub-voxel detail: tileable noise advected slowly, erodes thin gas into wisps
					const det = detailNode.sample( p.mul( U.detailScale ).add( U.detailOff ) ).level( 0 ).toVar();
					const thin = float( 1 ).sub( smoothstep( 0.0, 1.3, rho0 ) );
					const ero = det.r.mul( 0.6 ).add( det.g.mul( 0.4 ) ).sub( 0.5 );
					const rho = rho0.mul( clamp( float( 1 ).add( ero.mul( U.detail ).mul( thin.mul( 2.4 ).add( 0.5 ) ) ), 0.0, 2.2 ) ).toVar();
					const sig = rho.mul( U.extinction ).toVar();

					const L4 = lightNode.sample( uvw ).level( 0 ).toVar();

					// ── cluster light: single + Wrenninge-style multiple scattering octaves
					const lv = C.sub( p );
					const r2 = dot( lv, lv ).toVar();
					const cosT = dot( rd, lv.mul( inverseSqrt( r2 ) ) ).toVar();
					const dil = U.light.mul( 2.5 ).div( r2.mul( 0.045 ).add( 0.4 ) ).toVar();
					const tauC = L4.x.mul( kEx ).toVar();
					let S = vec3( 0 );
					const oct = [ [ 1.0, 1.0, 1.0 ], [ 0.5, 0.45, 0.5 ], [ 0.25, 0.2, 0.25 ] ]; // a^i, b^i, c^i
					for ( const [ a, b, c ] of oct ) {

						S = S.add( exp( redden.mul( tauC.mul( b ) ).negate() ).mul( phase2( cosT, c ).mul( a ) ) );

					}

					const scat = S.mul( dil ).mul( vec3( 0.78, 0.86, 1.0 ) ).toVar();

					// ionising flux reaching this point (UV is absorbed ~2.4x more strongly)
					const xC = exp( tauC.mul( - 2.4 ) );
					const ion = xC.mul( dil.div( dil.add( 0.35 ) ) ).toVar();
					const oiii = xC.mul( xC ).mul( dil ).mul( exp( sqrt( r2 ).mul( - 0.05 ) ) ).toVar();

					// ── user-ignited stars
					const tauU = [ L4.y, L4.z, L4.w ];
					If( U.userOn.greaterThan( 0.5 ), () => {

						for ( let k = 0; k < MAX_USER_STARS; k ++ ) {

						const sp = U.starPos[ k ], sa = U.starPar[ k ];
						const lvk = sp.xyz.sub( p );
						const r2k = dot( lvk, lvk );
						const win = ssr( sa.z.mul( sa.z ), sa.z.mul( sa.z ).mul( 0.3 ), r2k );
						const dk = sa.x.mul( win ).div( r2k.mul( 0.9 ).add( 0.06 ) ).toVar();
						const ck = dot( rd, lvk.mul( inverseSqrt( r2k.add( 1e-6 ) ) ) );
						const tk = tauU[ k ].mul( kEx );
						const Tk = exp( redden.mul( tk ).negate() );
						const Tk2 = exp( redden.mul( tk.mul( 0.4 ) ).negate() );
						scat.addAssign( Tk.mul( phase2( ck, 1.0 ) ).add( Tk2.mul( phase2( ck, 0.5 ).mul( 0.5 ) ) ).mul( dk ).mul( U.userCol ) );
						const xk = exp( tk.mul( - 2.4 ) );
						ion.addAssign( xk.mul( dk.div( dk.add( 0.35 ) ) ) );
						oiii.addAssign( xk.mul( xk ).mul( dk ).mul( 0.8 ) );

						}

					} );

					// ── emission lines (recombination ∝ n_e n_p ~ ρ²; S II from the partially ionised skin)
					const ionF = clamp( ion, 0.0, 1.0 );
					const rhoE = rho.mul( min( rho, 1.25 ).add( 0.25 ) );
					const ha = rhoE.mul( ionF );
					const sii = rhoE.mul( ionF ).mul( float( 1 ).sub( ionF ) ).mul( D.b.mul( 1.6 ).add( 0.6 ) ).mul( 1.0 );
					const o3 = rho.mul( oiii ).mul( 2.0 );
					const em = cHa.mul( ha ).add( cS.mul( sii ) ).add( cO.mul( o3 ) ).mul( U.emission.mul( 5.0 ) ).toVar();

					// ── Herbig–Haro jets: knotty bipolar outflows with bow shocks (analytic emission)
					for ( let j = 0; j < JETS.length; j ++ ) {

						const J = JETS[ j ];
						const ax = normalize( v3( J.d ) );
						const v = p.sub( v3( J.o ) ).toVar();
						If( dot( v, v ).lessThan( ( J.len + 0.6 ) * ( J.len + 0.6 ) ), () => {

						const s = dot( v, ax );
						const as = abs( s );
						const rr2 = max( dot( v, v ).sub( s.mul( s ) ), 0.0 );
						const w = as.mul( 0.05 / J.len ).add( 0.05 );
						const radial = exp( rr2.div( w.mul( w ) ).negate() );
						const knots = pow( sin( as.mul( 6.5 ).sub( U.time.mul( 0.9 ) ).add( j * 2.1 ) ).mul( 0.5 ).add( 0.5 ), 5.0 ).mul( 1.8 ).add( 0.25 );
						const body = radial.mul( knots ).mul( ssr( J.len, J.len * 0.55, as ) ).mul( smoothstep( 0.0, 0.15, as ) );
						const td = as.sub( J.len );
						const bow = exp( td.mul( td ).add( rr2 ).div( 0.05 ).negate() );
						em.addAssign( cJet.mul( body.add( bow.mul( 0.7 ) ) ).mul( U.emission ).mul( U.jets ).mul( 3.0 ) );

						} );

					}

					const albedo = mix( 0.12, 0.3, D.g );
					const src = scat.mul( cScat ).mul( sig ).mul( albedo ).add( cAmb.mul( sig ).mul( U.light ) ).add( em );
					const Tst = exp( sig.mul( dt ).negate() ).toVar();
					const integ = src.mul( float( 1 ).sub( Tst ) ).div( max( sig, 1e-5 ) );
					acc.addAssign( integ.mul( T ) );
					const wgt = T.mul( float( 1 ).sub( Tst ) );
					dSum.addAssign( wgt.mul( t ) );
					wSum.addAssign( wgt );
					T.mulAssign( Tst );

				} );

				If( T.lessThan( 0.003 ), () => {

					Break();

				} );

			} );

		} );

		const depth = select( wSum.greaterThan( 1e-4 ), dSum.div( max( wSum, 1e-4 ) ), max( tF, 50.0 ) );
		return vec4( acc, depth );

	} );

	const mat = new THREE.NodeMaterial();
	mat.fragmentNode = marchFn();
	mat.depthTest = false;
	mat.depthWrite = false;
	mat.name = 'nebula.march';
	return mat;

}

// Temporal accumulation at half res: reproject history using the opacity-weighted depth, clamp to the
// current frame's 3x3 neighbourhood (variance clip) to avoid ghosting.
export function buildResolveMaterial( curNode, histNode, U ) {

	const fn = Fn( () => {

		const uv = screenUV;
		const cur = curNode.sample( uv ).toVar();
		const m1 = vec3( 0 ).toVar(), m2 = vec3( 0 ).toVar();
		for ( let y = - 1; y <= 1; y ++ ) {

			for ( let x = - 1; x <= 1; x ++ ) {

				const c = ( x === 0 && y === 0 ) ? cur.rgb : curNode.sample( uv.add( vec2( x, y ).mul( U.texel ) ) ).rgb;
				m1.addAssign( c );
				m2.addAssign( c.mul( c ) );

			}

		}

		const mean = m1.div( 9 );
		const sd = sqrt( max( m2.div( 9 ).sub( mean.mul( mean ) ), vec3( 0 ) ) );
		const lo = mean.sub( sd.mul( 1.4 ) ), hi = mean.add( sd.mul( 1.4 ) );

		const ndc = vec2( uv.x.mul( 2 ).sub( 1 ), uv.y.mul( - 2 ).add( 1 ) );
		const vp = U.invProj.mul( vec4( ndc, 1.0, 1.0 ) );
		const rd = normalize( U.camWorld.mul( vec4( vp.xyz.div( vp.w ), 0.0 ) ).xyz );
		const wp = U.camPos.add( rd.mul( cur.a ) );
		const pc = U.prevViewProj.mul( vec4( wp, 1.0 ) );
		const pn = pc.xy.div( pc.w );
		const puv = vec2( pn.x.mul( 0.5 ).add( 0.5 ), pn.y.mul( - 0.5 ).add( 0.5 ) ).toVar();
		const inside = puv.x.greaterThan( 0 ).and( puv.x.lessThan( 1 ) ).and( puv.y.greaterThan( 0 ) ).and( puv.y.lessThan( 1 ) ).and( pc.w.greaterThan( 0 ) );
		const hist = clamp( histNode.sample( puv ).rgb, lo, hi );
		const a = select( inside, U.blend, float( 1.0 ) );
		return vec4( mix( hist, cur.rgb, a ), cur.a );

	} );

	const mat = new THREE.NodeMaterial();
	mat.fragmentNode = fn();
	mat.depthTest = false;
	mat.depthWrite = false;
	mat.name = 'nebula.resolve';
	return mat;

}
