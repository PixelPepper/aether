// Star-filter sparkles: a few thousand candidate facets scattered over the crystal. In the vertex stage each one
// reads the relief at its site, tilts the local normal by a random microfacet perturbation, and evaluates a very
// narrow specular lobe against the studio lights. Only facets that are momentarily aligned with a light's half-vector
// get a non-zero size, and are drawn as six-pointed diffraction stars (a hexagonal aperture's spikes) — so the
// flake twinkles as it precesses, like a macro photograph through a star filter.
import * as THREE from 'three/webgpu';
import {
	Fn, uniform, instanceIndex, float, vec2, vec3, vec4, hash, uv, abs, max, mix, smoothstep, sin, cos, exp, sqrt,
	pow, texture, normalize, dot, transformNormalToView, varying, select, modelViewMatrix,
} from 'three/tsl';

export function createSparkles( studio, tex, P, count ) {

	const { S } = studio;
	const mat = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
	const rnd = ( k ) => hash( float( instanceIndex ).mul( 13.0 ).add( k ).add( P.seed.mul( 7.0 ) ) );

	// site on the flake (uv over the hex domain)
	const r = sqrt( rnd( 1 ) ).mul( 0.95 ), a = rnd( 2 ).mul( 6.2831853 );
	const suv = vec2( cos( a ), sin( a ) ).mul( r ).mul( 0.5 ).add( 0.5 );
	const s = texture( tex, suv ).level( 0 );
	const hgt = s.y.mul( smoothstep( 0.5, 0.85, s.x ) );
	const inside = s.x.greaterThan( 0.55 );

	// local relief normal + microfacet tilt
	const e = S.texel.mul( 2.0 );
	const hOf = ( q ) => q.y.mul( smoothstep( 0.5, 0.85, q.x ) );
	const hx = hOf( texture( tex, suv.add( vec2( e, 0 ) ) ).level( 0 ) ).sub( hOf( texture( tex, suv.sub( vec2( e, 0 ) ) ).level( 0 ) ) );
	const hy = hOf( texture( tex, suv.add( vec2( 0, e ) ) ).level( 0 ) ).sub( hOf( texture( tex, suv.sub( vec2( 0, e ) ) ).level( 0 ) ) );
	const k = S.thick.mul( S.bump ).div( e.mul( 4.0 ) );
	const tilt = vec2( rnd( 3 ).sub( 0.5 ), rnd( 4 ).sub( 0.5 ) ).mul( 1.3 );
	const nL = normalize( vec3( hx.mul( k ).negate().add( tilt.x ), hy.mul( k ).negate().add( tilt.y ), 1 ) );
	const nV = normalize( transformNormalToView( nL ) );

	const pLocal = vec3( suv.mul( 2 ).sub( 1 ), hgt.mul( S.thick ).add( 0.003 ) );
	mat.positionNode = pLocal;

	// view direction at the site (vertex stage: from the site's view-space position)
	const pView = modelViewMatrix.mul( vec4( pLocal, 1 ) ).xyz;
	const V = normalize( pView.negate() );
	const lobe = ( L, p ) => pow( max( dot( nV, normalize( L.add( V ) ) ), 0 ), p );
	const inten = lobe( S.lKey, 3000.0 ).mul( 1.0 ).add( lobe( S.lSpark, 4000.0 ).mul( 0.8 ) ).add( lobe( S.lFill, 3000.0 ).mul( 0.5 ) )
		.mul( select( inside, float( 1 ), float( 0 ) ) ).mul( float( 1 ).sub( P.fade ) );
	const vI = varying( inten, 'vSparkle' );
	const vTint = varying( mix( vec3( 1.0, 0.9, 0.78 ), vec3( 0.75, 0.88, 1.0 ), rnd( 5 ) ), 'vSparkleTint' );

	// size ∝ intensity (a brighter flare spreads further); zero size culls it
	mat.scaleNode = select( inten.greaterThan( 0.02 ), S.sparkSize.mul( sqrt( inten ).mul( 0.8 ).add( 0.2 ) ), float( 0 ) );

	mat.colorNode = Fn( () => {

		const p = uv().mul( 2 ).sub( 1 );
		const rr = p.length();
		let rays = float( 0 );
		for ( let i = 0; i < 3; i ++ ) {

			const ang = 0.26 + i * Math.PI / 3;
			const d = vec2( Math.cos( ang ), Math.sin( ang ) );
			const along = abs( dot( p, d ) );
			const perp = abs( dot( p, vec2( - d.y, d.x ) ) );
			rays = rays.add( exp( perp.mul( perp ).mul( - 2500.0 ) ).mul( pow( max( float( 1 ).sub( along ), 0 ), 2.5 ) ) );

		}

		const core = exp( rr.mul( rr ).mul( - 160.0 ) ).mul( 2.0 ).add( exp( rr.mul( rr ).mul( - 18.0 ) ).mul( 0.25 ) );
		const c = vTint.mul( rays.mul( 0.9 ).add( core ) ).mul( vI.min( 2.0 ) ).mul( S.sparkGain );
		return vec4( c, 1 );

	} )();

	const sprite = new THREE.Sprite( mat );
	sprite.count = count;
	sprite.frustumCulled = false;
	sprite.renderOrder = 3;

	return { sprite, mat };

}
