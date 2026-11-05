// Out-of-focus snowflakes falling between the lens and the velvet: instanced sprites whose size is the thin-lens
// circle of confusion for their depth (focus is on the hero flake). Far/near flakes become hexagonal aperture discs
// with a bright rim; flakes near the focal plane keep a hint of their six arms.
import * as THREE from 'three/webgpu';
import {
	Fn, uniform, instanceIndex, float, vec2, vec3, vec4, hash, uv, abs, max, min, mix, smoothstep, sin, cos, fract,
	exp, length, cameraWorldMatrix, pow, atan, clamp,
} from 'three/tsl';

export function createBokeh( count ) {

	const u = {
		time: uniform( 0 ),
		focus: uniform( 2.6 ), // distance to the hero flake
		aperture: uniform( 0.075 ),
		gain: uniform( 1.0 ),
		wind: uniform( 0 ),
	};

	const mat = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );

	const rnd = ( k ) => hash( float( instanceIndex ).mul( 7.0 ).add( k ) );

	// view-space position: a slab of air from just in front of the lens to far behind the flake
	const depth = mix( float( 0.7 ), float( 12.0 ), pow( rnd( 1 ), 1.6 ) );
	const halfH = depth.mul( 0.5 ).add( 0.4 );
	const halfW = halfH.mul( 1.9 );
	const speed = mix( float( 0.05 ), float( 0.16 ), rnd( 3 ) );
	const fall = fract( rnd( 4 ).sub( u.time.mul( speed ).div( halfH.mul( 2 ) ) ) );
	const sway = sin( u.time.mul( mix( 0.3, 0.9, rnd( 5 ) ) ).add( rnd( 6 ).mul( 6.28 ) ) ).mul( 0.06 ).mul( depth.mul( 0.15 ).add( 0.4 ) );
	const pv = vec3(
		rnd( 2 ).mul( 2 ).sub( 1 ).mul( halfW ).add( sway ).add( u.wind.mul( depth ).mul( 0.05 ) ),
		fall.mul( 2 ).sub( 1 ).mul( halfH ),
		depth.negate(),
	);
	mat.positionNode = cameraWorldMatrix.mul( vec4( pv, 1 ) ).xyz;

	// circle of confusion (world units at that depth)
	const coc = u.aperture.mul( abs( depth.sub( u.focus ) ) ).div( u.focus );
	const core = mix( float( 0.006 ), float( 0.016 ), rnd( 7 ) );
	const radius = max( coc, core );
	mat.scaleNode = radius.mul( 2.0 );

	const sharp = smoothstep( core.mul( 2.5 ), core.mul( 0.8 ), coc );

	mat.colorNode = Fn( () => {

		const p = uv().mul( 2 ).sub( 1 );
		const ap = abs( p );
		// hexagonal aperture (6 blades)
		const hexd = max( ap.x.mul( 0.8660254 ).add( ap.y.mul( 0.5 ) ), ap.y ).div( 0.92 );
		const disc = smoothstep( 1.0, 0.9, hexd );
		const rimL = smoothstep( 0.55, 0.97, hexd ).mul( 0.6 ).add( 0.55 );
		const bokehShape = disc.mul( rimL );
		// in-focus-ish: a tiny six-armed star
		const r = length( p );
		const a = atan( p.y, p.x ).add( rnd( 8 ).mul( 6.28 ) );
		const arms = pow( abs( cos( a.mul( 3.0 ) ) ), 24.0 ).mul( smoothstep( 1.0, 0.1, r ) );
		const star = arms.add( exp( r.mul( r ).mul( - 30.0 ) ) ).mul( 0.9 );
		const shape = mix( bokehShape, star, sharp );
		// energy conservation: brightness ∝ 1/area, clamped so the far discs remain a faint presence
		const e = clamp( core.mul( core ).div( radius.mul( radius ) ).mul( 2.5 ), 0.012, 1.0 );
		const tint = mix( vec3( 0.55, 0.75, 1.0 ), vec3( 1.0, 0.86, 0.7 ), smoothstep( 0.8, 1.0, rnd( 9 ) ) );
		const near = smoothstep( 1.0, 2.2, depth ); // fade out the ones about to touch the lens
		return vec4( tint.mul( shape ).mul( e ).mul( u.gain ).mul( near ).mul( 1.6 ), 1 );

	} )();

	const sprite = new THREE.Sprite( mat );
	sprite.count = count;
	sprite.frustumCulled = false;
	sprite.renderOrder = 2;

	return { sprite, u, dispose() {

		mat.dispose(); // the sprite quad geometry is shared by three.js — never dispose it

	} };

}
