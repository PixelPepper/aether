// Sprite renderers for the merger. Every sprite is flux-conserving: peak = L / (k·size²), with a minimum
// on-screen size so sub-pixel stars never alias — surface brightness stays constant under zoom.
import * as THREE from 'three/webgpu';
import {
	Fn, uniform, vec2, vec3, vec4, float, dot, max, min, exp, mix, smoothstep, select, uv, abs, pow,
	modelViewMatrix, vertexStage, instancedArray, clamp,
} from 'three/tsl';

const PROFILE_NORM = 0.175; // ∫ exp(-4.5 r²) over the quad, in units of size²

export const starColor = Fn( ( [ c ] ) => {

	const warm = vec3( 1.0, 0.56, 0.28 ), gold = vec3( 1.0, 0.78, 0.5 ), white = vec3( 1.0, 0.94, 0.86 ), blue = vec3( 0.56, 0.72, 1.0 );
	const lo = mix( warm, gold, smoothstep( 0.0, 0.35, c ) );
	const hi = mix( white, blue, smoothstep( 0.62, 1.0, c ) );
	return mix( lo, hi, smoothstep( 0.32, 0.66, c ) );

} );

const extinction = ( tau ) => exp( vec3( 0.62, 0.85, 1.25 ).mul( tau ).negate() );

function additive( mat ) {

	mat.transparent = true;
	mat.depthWrite = false;
	mat.depthTest = false;
	mat.blending = THREE.AdditiveBlending;
	return mat;

}

const gaussProfile = ( k = 4.5 ) => {

	const q = uv().sub( 0.5 ).mul( 2 );
	const r2 = dot( q, q );
	return max( exp( r2.mul( - k ) ).sub( exp( float( - k ) ) ), 0.0 );

};

export function createRenderers( sim, cfg ) {

	const B = sim.buffers;
	const RU = {
		pxWorld: uniform( 0.002 ), // world size of one pixel at unit depth
		exposure: uniform( cfg.exposure ),
		starSize: uniform( cfg.starSize ),
		minPx: uniform( cfg.minPx ),
		glowSize: uniform( cfg.glowSize ),
		glowGain: uniform( cfg.glowGain ),
		dmGain: uniform( 1.0 ),
		time: sim.U.time,
	};

	const objects = [];
	const materials = [];

	// ── stars (all tracers) ──
	{

		const pos = B.tPos.toAttribute();
		const ren = B.tRen.toAttribute();
		const info = B.tInfo.toAttribute();
		const mat = additive( new THREE.SpriteNodeMaterial() );
		const depth = max( modelViewMatrix.mul( vec4( pos.xyz, 1 ) ).z.negate(), 0.01 );
		const type = info.x;
		const kind = select( type.lessThan( 0.5 ), float( 0.75 ), select( type.lessThan( 1.5 ), float( 1.15 ), select( type.lessThan( 3.5 ), float( 1.0 ), float( 1.3 ) ) ) );
		const size = max( RU.starSize.mul( kind ), depth.mul( RU.pxWorld ).mul( RU.minPx ) ).toVar();
		const lum = ren.y;
		const peak = lum.mul( RU.exposure ).div( size.mul( size ).mul( PROFILE_NORM ) );
		const col = vertexStage( starColor( ren.w ).mul( extinction( ren.x ) ).mul( peak ) );
		mat.positionNode = pos.xyz;
		mat.scaleNode = select( lum.greaterThan( 0.0 ), size, float( 0 ) );
		mat.colorNode = vec4( col.mul( gaussProfile( 4.5 ) ), 1 );
		const sp = new THREE.Sprite( mat );
		sp.count = cfg.nt;
		sp.frustumCulled = false;
		sp.renderOrder = 2;
		objects.push( sp ); materials.push( mat );

	}

	// ── HII regions: pink Hα/[NII] knots around freshly-triggered gas ──
	{

		const pos = B.tPos.toAttribute();
		const ren = B.tRen.toAttribute();
		const vel = B.tVel.toAttribute();
		const mat = additive( new THREE.SpriteNodeMaterial() );
		const age = RU.time.sub( vel.w );
		const g = ren.z;
		const depth = max( modelViewMatrix.mul( vec4( pos.xyz, 1 ) ).z.negate(), 0.01 );
		const size = max( RU.glowSize.mul( clamp( age.div( 2.6 ), 0.0, 1.0 ).mul( 0.9 ).add( 0.55 ) ), depth.mul( RU.pxWorld ).mul( 3.0 ) ).toVar();
		const peak = g.mul( RU.glowGain ).div( size.mul( size ).mul( PROFILE_NORM ) );
		// Hα pink with a hot whitish-violet core
		const tint = mix( vec3( 1.0, 0.22, 0.5 ), vec3( 1.0, 0.3, 0.42 ), smoothstep( 0.0, 2.6, age ) );
		const col = vertexStage( tint.mul( extinction( ren.x.mul( 0.8 ) ) ).mul( peak ) );
		mat.positionNode = pos.xyz;
		mat.scaleNode = select( g.greaterThan( 0.002 ), size, float( 0 ) );
		const q = uv().sub( 0.5 ).mul( 2 );
		const r2 = dot( q, q );
		const prof = max( exp( r2.mul( - 3.2 ) ).sub( 0.04 ), 0.0 ).add( exp( r2.mul( - 40 ) ).mul( 0.6 ) );
		mat.colorNode = vec4( col.mul( prof ).add( vec3( 0.25, 0.2, 0.35 ).mul( col.g ).mul( exp( r2.mul( - 30 ) ) ) ), 1 );
		const sp = new THREE.Sprite( mat );
		sp.count = cfg.nGas;
		sp.frustumCulled = false;
		sp.renderOrder = 3;
		objects.push( sp ); materials.push( mat );

	}

	// ── dark matter (massive halo + bulge bodies), faint violet haze; toggled ──
	let dmSprite;
	{

		const pos = B.mPos.toAttribute();
		const vel = B.mVel.toAttribute();
		const mat = additive( new THREE.SpriteNodeMaterial() );
		const kind = vel.w.mod( 10 );
		const isHalo = kind.greaterThan( 0.5 ).and( kind.lessThan( 2.5 ) ).or( abs( vel.w.sub( 4 ) ).lessThan( 0.5 ) );
		const size = float( cfg.dmSize );
		const peak = pos.w.mul( RU.dmGain ).mul( cfg.dmExposure ).div( size.mul( size ).mul( PROFILE_NORM ) );
		mat.positionNode = pos.xyz;
		mat.scaleNode = select( isHalo.and( pos.w.greaterThan( 0 ) ), size, float( 0 ) );
		const col = vertexStage( vec3( 0.32, 0.36, 1.0 ).mul( peak ) );
		mat.colorNode = vec4( col.mul( gaussProfile( 3.0 ) ), 1 );
		dmSprite = new THREE.Sprite( mat );
		dmSprite.count = cfg.nm;
		dmSprite.frustumCulled = false;
		dmSprite.renderOrder = 1;
		dmSprite.visible = false;
		objects.push( dmSprite ); materials.push( mat );

	}

	// ── black-hole perturbers (massive slots 0..5): blazing quasar points with diffraction spikes ──
	{

		const pos = B.mPos.toAttribute();
		const vel = B.mVel.toAttribute();
		const mat = additive( new THREE.SpriteNodeMaterial() );
		const isPert = abs( vel.w.sub( 5 ) ).lessThan( 0.5 ).and( pos.w.greaterThan( 0 ) );
		const depth = max( modelViewMatrix.mul( vec4( pos.xyz, 1 ) ).z.negate(), 0.01 );
		const size = depth.mul( RU.pxWorld ).mul( 70 );
		mat.positionNode = pos.xyz;
		mat.scaleNode = select( isPert, size, float( 0 ) );
		const q = uv().sub( 0.5 ).mul( 2 );
		const r2 = dot( q, q );
		const spikes = exp( abs( q.x ).mul( - 160 ) ).add( exp( abs( q.y ).mul( - 160 ) ) ).mul( pow( max( float( 1 ).sub( r2 ), 0 ), 3 ) );
		const core = exp( r2.mul( - 900 ) ).mul( 30 ).add( exp( r2.mul( - 60 ) ).mul( 1.2 ) ).add( exp( r2.mul( - 8 ) ).mul( 0.12 ) );
		mat.colorNode = vec4( vec3( 0.75, 0.82, 1.0 ).mul( core.add( spikes.mul( 0.8 ) ) ), 1 );
		const sp = new THREE.Sprite( mat );
		sp.count = 6;
		sp.frustumCulled = false;
		sp.renderOrder = 4;
		objects.push( sp ); materials.push( mat );

	}

	// ── sky: foreground Milky-Way stars with 4-point spikes + faint distant background galaxies ──
	const skyBuffers = [];
	{

		const rng = mulberry( 99 );
		const nStars = 220, nGal = 360, n = nStars + nGal;
		const P = new Float32Array( n * 4 ), C = new Float32Array( n * 4 ), S = new Float32Array( n * 4 );
		for ( let i = 0; i < n; i ++ ) {

			const z = rng() * 2 - 1, ph = rng() * Math.PI * 2, s = Math.sqrt( 1 - z * z ), R = 700;
			P.set( [ s * Math.cos( ph ) * R, z * R, s * Math.sin( ph ) * R, i < nStars ? 0 : 1 ], i * 4 );
			if ( i < nStars ) {

				const m = Math.pow( rng(), 6 ); // few bright, many faint
				const t = rng();
				const c = t < 0.5 ? [ 1, 0.86, 0.7 ] : t < 0.8 ? [ 1, 0.95, 0.92 ] : [ 0.75, 0.85, 1 ];
				const b = 0.25 + m * 9;
				C.set( [ c[ 0 ] * b, c[ 1 ] * b, c[ 2 ] * b, m ], i * 4 );
				S.set( [ 5 + m * 60, 5 + m * 60, 0, m > 0.35 ? 1 : 0 ], i * 4 );

			} else {

				const red = rng() < 0.5;
				const c = red ? [ 1, 0.7, 0.45 ] : [ 0.75, 0.82, 1 ];
				const b = 0.05 + Math.pow( rng(), 3 ) * 0.25;
				C.set( [ c[ 0 ] * b, c[ 1 ] * b, c[ 2 ] * b, 0 ], i * 4 );
				const sz = 3 + Math.pow( rng(), 4 ) * 9, el = 0.3 + rng() * 0.7;
				S.set( [ sz, sz * el, rng() * Math.PI, 2 ], i * 4 );

			}

		}

		const pb = instancedArray( P, 'vec4' ), cb = instancedArray( C, 'vec4' ), sb = instancedArray( S, 'vec4' );
		skyBuffers.push( pb, cb, sb );
		const pos = pb.toAttribute(), col = cb.toAttribute(), sz = sb.toAttribute();
		const mat = additive( new THREE.SpriteNodeMaterial() );
		const depth = max( modelViewMatrix.mul( vec4( pos.xyz, 1 ) ).z.negate(), 0.01 );
		mat.positionNode = pos.xyz;
		mat.scaleNode = sz.xy.mul( depth.mul( RU.pxWorld ) );
		mat.rotationNode = sz.z;
		const q = uv().sub( 0.5 ).mul( 2 );
		const r2 = dot( q, q );
		const isSpiky = sz.w.greaterThan( 0.5 ).and( sz.w.lessThan( 1.5 ) );
		const isGal = sz.w.greaterThan( 1.5 );
		const starP = exp( r2.mul( - 60 ) ).add( exp( r2.mul( - 9 ) ).mul( 0.04 ) );
		const spike = exp( abs( q.x ).mul( - 120 ) ).add( exp( abs( q.y ).mul( - 120 ) ) ).mul( pow( max( float( 1 ).sub( r2 ), 0 ), 4 ) ).mul( 0.5 );
		const galP = exp( r2.mul( - 7 ) ).add( exp( r2.mul( - 40 ) ).mul( 0.8 ) );
		const prof = select( isGal, galP, starP.add( select( isSpiky, spike, float( 0 ) ) ) );
		mat.colorNode = vec4( col.xyz.mul( prof ), 1 );
		const sp = new THREE.Sprite( mat );
		sp.count = n;
		sp.frustumCulled = false;
		sp.renderOrder = 0;
		objects.push( sp ); materials.push( mat );

	}

	return { objects, materials, RU, dmSprite, skyBuffers };

}

function mulberry( seed ) {

	let a = seed >>> 0;
	return () => {

		a |= 0; a = a + 0x6D2B79F5 | 0;
		let t = Math.imul( a ^ a >>> 15, 1 | a );
		t = t + Math.imul( t ^ t >>> 7, 61 | t ) ^ t;
		return ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296;

	};

}

export { min };
