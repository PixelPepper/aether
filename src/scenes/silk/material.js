// Silk BRDF: Charlie sheen + anisotropic GGX (per-thread warp/weft direction) + thin-film iridescence
// (three's PhysicalLightingModel), extended with thin-sheet diffuse transmission and a forward-scatter
// lobe so back-lit silk glows. Colour is "shot silk": warp and weft dyed differently, the visible mix
// depending on the viewing direction relative to the threads. A procedural plain-weave height field
// perturbs the normal (faded out by screen-space LOD) and weft slubs add dupioni streaks.
import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, vec2, vec3, vec4, uv, attribute, normalView, tangentView, bitangentView,
	positionViewDirection, fwidth, floor, fract, cos, sin, abs, mix, smoothstep, clamp, dot, normalize,
	max, mx_noise_float, select, PI,
} from 'three/tsl';

class SilkLightingModel extends THREE.PhysicalLightingModel {

	constructor( transNode, fwdNode ) {

		super( false, true, true, true, false, false, false );
		this.transNode = transNode;
		this.fwdNode = fwdNode;

	}

	direct( { lightDirection, lightColor, reflectedLight }, builder ) {

		// light arriving on the far side of the (thin) sheet: Lambertian diffuse transmission
		const back = normalView.dot( lightDirection ).negate().clamp();
		// forward scattering through the fibres toward the viewer
		const fwd = positionViewDirection.dot( lightDirection.negate() ).clamp().pow( 5 ).mul( back.sqrt() );
		const t = this.transNode.mul( back.mul( 1 / Math.PI ).add( fwd.mul( this.fwdNode ) ) );
		reflectedLight.directDiffuse.addAssign( t.mul( lightColor ) );

		super.direct( { lightDirection, lightColor, reflectedLight }, builder );

	}

}

class SilkNodeMaterial extends THREE.MeshPhysicalNodeMaterial {

	static get type() {

		return 'SilkNodeMaterial';

	}

	constructor( params ) {

		super( params );
		this.transNode = vec3( 0.3 );
		this.fwdNode = float( 0.5 );

	}

	setupLightingModel() {

		return new SilkLightingModel( this.transNode, this.fwdNode );

	}

}

export const PALETTES = {
	pearl: {
		warp: [ 0.86, 0.82, 0.78 ], weft: [ 0.70, 0.76, 0.86 ], sheen: [ 1.0, 0.96, 0.93 ],
		trans: [ 1.0, 0.86, 0.74 ], irid: 0.75, tMin: 260, tMax: 520, rough: 0.36, transAmt: 0.55,
	},
	gold: {
		warp: [ 0.80, 0.50, 0.14 ], weft: [ 0.62, 0.22, 0.07 ], sheen: [ 1.0, 0.78, 0.40 ],
		trans: [ 1.0, 0.55, 0.15 ], irid: 0.35, tMin: 480, tMax: 700, rough: 0.32, transAmt: 0.6,
	},
	'oil-slick': {
		warp: [ 0.035, 0.03, 0.06 ], weft: [ 0.02, 0.05, 0.06 ], sheen: [ 0.55, 0.62, 1.0 ],
		trans: [ 0.30, 0.16, 0.45 ], irid: 1.0, tMin: 280, tMax: 1150, rough: 0.28, transAmt: 0.5,
	},
};

export function createSilkMaterial( { threads = 700 } = {} ) {

	const c3 = ( a ) => new THREE.Color( a[ 0 ], a[ 1 ], a[ 2 ] );
	const P0 = PALETTES.pearl;
	const U = {
		warp: uniform( c3( P0.warp ) ),
		weft: uniform( c3( P0.weft ) ),
		sheen: uniform( c3( P0.sheen ) ),
		trans: uniform( c3( P0.trans ) ),
		transAmt: uniform( P0.transAmt ),
		irid: uniform( P0.irid ),
		tMin: uniform( P0.tMin ),
		tMax: uniform( P0.tMax ),
		rough: uniform( P0.rough ),
	};

	const sheetId = attribute( 'sheet', 'float' );
	const st = uv().mul( threads );

	// screen-space LOD: fade the thread-level structure once a thread is smaller than ~1.5 px
	const lod = Fn( () => {

		const w = max( fwidth( st.x ), fwidth( st.y ) );
		return clamp( float( 1.6 ).sub( w.mul( 1.2 ) ), 0, 1 );

	} )().toVar( 'weaveLod' );

	// plain weave: warp (along v) over weft (along u) in a checkerboard
	const cell = floor( st );
	const f = fract( st );
	const warpTop = fract( cell.x.add( cell.y ).mul( 0.5 ) ).lessThan( 0.25 );

	// weft slubs (dupioni): irregular thicker weft threads → streaks running along u
	const slub = Fn( () => {

		const v = uv().y.mul( threads * 0.5 ).add( sheetId.mul( 31.7 ) );
		const n1 = mx_noise_float( vec2( v, sheetId.mul( 3.1 ) ) );
		const n2 = mx_noise_float( vec2( uv().x.mul( 6.0 ), v.mul( 0.5 ) ) );
		return smoothstep( 0.25, 0.75, n1 ).mul( smoothstep( - 0.2, 0.6, n2 ) );

	} )().toVar( 'slub' );

	const mat = new SilkNodeMaterial( { side: THREE.DoubleSide } );

	// view direction projected into the weave frame
	const tW = dot( positionViewDirection, tangentView ); // along weft (u)
	const bW = dot( positionViewDirection, bitangentView ); // along warp (v)
	const nW = dot( positionViewDirection, normalView ).abs();

	mat.colorNode = Fn( () => {

		// changeant: looking along the warp shows more weft, and vice versa (stronger at grazing)
		const tt = tW.mul( tW ), bb = bW.mul( bW );
		const orient = bb.sub( tt ).div( tt.add( bb ).add( 1e-4 ) );
		const shift = orient.mul( float( 1 ).sub( nW ) ).mul( 0.5 ).add( 0.5 );
		const far = mix( U.warp, U.weft, shift.mul( 0.8 ).add( 0.1 ) );
		const near = select( warpTop, U.warp, U.weft );
		const base = mix( far, mix( far, near, 0.55 ), lod );
		const sl = slub.mul( 0.16 ).add( 0.94 );
		return vec4( base.mul( sl ), 1 );

	} )();

	mat.roughnessNode = U.rough.mul( float( 1 ).sub( slub.mul( 0.12 ) ) );
	mat.metalnessNode = float( 0 );
	mat.specularIntensityNode = float( 1 );
	mat.iorNode = float( 1.55 ); // fibroin

	// anisotropy perpendicular to the fibre on top: warp floats → spread along u, weft floats → along v
	mat.anisotropyNode = Fn( () => {

		const near = select( warpTop, vec2( 1, 0 ), vec2( 0, 1 ) );
		return mix( vec2( 1, 0 ), near, lod ).mul( 0.62 );

	} )();

	mat.sheenNode = U.sheen;
	mat.sheenRoughnessNode = float( 0.38 );

	mat.iridescenceNode = U.irid;
	mat.iridescenceIORNode = float( 1.38 );
	mat.iridescenceThicknessNode = Fn( () => {

		const n = mx_noise_float( vec3( uv().mul( 2.2 ), sheetId.mul( 1.7 ) ) ).mul( 0.5 ).add( 0.5 );
		const n2 = mx_noise_float( vec3( uv().mul( 7.0 ), sheetId.add( 4.0 ) ) ).mul( 0.15 );
		return mix( U.tMin, U.tMax, n.add( n2 ).clamp( 0, 1 ) );

	} )();

	// woven micro-relief (analytic gradient of thread cross-sections), in the tangent frame
	mat.normalNode = Fn( () => {

		const sx = sin( f.x.mul( PI ) ), sy = sin( f.y.mul( PI ) );
		const cx = cos( f.x.mul( PI ) ), cy = cos( f.y.mul( PI ) );
		// warp-top cell: ridge across u, crimp along v; weft-top: the opposite
		const gu = select( warpTop, cx.mul( sy.mul( 0.7 ).add( 0.3 ) ), sx.mul( cy ).mul( 0.0 ).add( cx.mul( sy ).mul( 0.25 ) ) );
		const gv = select( warpTop, cy.mul( sx ).mul( 0.25 ), cy.mul( sx.mul( 0.7 ).add( 0.3 ) ) );
		const k = lod.mul( 0.22 );
		const n = normalView.sub( tangentView.mul( gu.mul( k ) ) ).sub( bitangentView.mul( gv.mul( k ) ) );
		return normalize( n );

	} )();

	mat.transNode = Fn( () => U.trans.mul( U.transAmt ) )();
	mat.fwdNode = float( 0.9 );

	function setPalette( name ) {

		const p = PALETTES[ name ] || PALETTES.pearl;
		U.warp.value.setRGB( ...p.warp );
		U.weft.value.setRGB( ...p.weft );
		U.sheen.value.setRGB( ...p.sheen );
		U.trans.value.setRGB( ...p.trans );
		U.transAmt.value = p.transAmt;
		U.irid.value = p.irid;
		U.tMin.value = p.tMin;
		U.tMax.value = p.tMax;
		U.rough.value = p.rough;

	}

	return { material: mat, U, setPalette };

}
