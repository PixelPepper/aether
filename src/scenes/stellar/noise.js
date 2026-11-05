// Small procedural-noise kit shared by the fluid forcing (granulation source) and the photosphere shader,
// so the convection cells driving the simulation line up with the granules drawn on the limb.
import { vec2, vec4, float, floor, fract, sin, dot, min, max, select, sqrt, mix, smoothstep } from 'three/tsl';

export const hash22 = ( p ) => fract( sin( vec2( dot( p, vec2( 127.1, 311.7 ) ), dot( p, vec2( 269.5, 183.3 ) ) ) ).mul( 43758.5453 ) );
export const hash21 = ( p ) => fract( sin( dot( p, vec2( 12.9898, 78.233 ) ) ).mul( 43758.5453 ) );

// Animated 2D Worley noise. Returns vec4( F1, F2, toNearest.xy ) where toNearest points from p to the nearest feature
// point (the granule centre). Feature points orbit their cell slowly: granules are born, drift and fragment.
// Must be called inside an Fn() (uses toVar()).
export function worley( p, t ) {

	const ip = floor( p ).toVar();
	const fp = fract( p ).toVar();
	const f1 = float( 9 ).toVar();
	const f2 = float( 9 ).toVar();
	const nv = vec2( 0 ).toVar();

	for ( let j = - 1; j <= 1; j ++ ) {

		for ( let i = - 1; i <= 1; i ++ ) {

			const gcell = vec2( i, j );
			const o = hash22( ip.add( gcell ) );
			const pt = gcell.add( vec2( 0.5 ).add( sin( o.mul( 6.2831 ).add( t.mul( o.add( 0.5 ) ) ) ).mul( 0.4 ) ) );
			const d = pt.sub( fp ).toVar();
			const dd = dot( d, d ).toVar();
			nv.assign( select( dd.lessThan( f1 ), d, nv ) );
			f2.assign( min( f2, max( f1, dd ) ) );
			f1.assign( min( f1, dd ) );

		}

	}

	return vec4( sqrt( f1 ), sqrt( f2 ), nv );

}

// granule brightness in [0,1]: bright domed cell interiors, narrow dark intergranular lanes
export function granuleShade( w ) {

	const lane = smoothstep( 0.0, 0.32, w.y.sub( w.x ) );
	return lane.mul( float( 1 ).sub( w.x.mul( 0.45 ) ) );

}

// value noise (2D), smooth
export function vnoise( p ) {

	const i = floor( p ).toVar();
	const f = fract( p ).toVar();
	const u = f.mul( f ).mul( f.mul( - 2 ).add( 3 ) );
	const a = hash21( i );
	const b = hash21( i.add( vec2( 1, 0 ) ) );
	const c = hash21( i.add( vec2( 0, 1 ) ) );
	const d = hash21( i.add( vec2( 1, 1 ) ) );
	return mix( mix( a, b, u.x ), mix( c, d, u.x ), u.y );

}

export const ign = ( px ) => fract( float( 52.9829189 ).mul( fract( dot( px, vec2( 0.06711056, 0.00583715 ) ) ) ) );

