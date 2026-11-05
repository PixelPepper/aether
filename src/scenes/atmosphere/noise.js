// Procedural noise for the Atmosphere chapter (TSL). Everything here is evaluated on the GPU:
// gradient / value noise with analytic derivatives, fBm, and *tileable* Perlin + Worley noise used to
// build the Perlin–Worley cloud volume (Schneider / Nubis style) in a compute pass.
import { Fn, float, vec3, vec4, floor, fract, dot, mix, min, max, abs, sqrt, Loop, int } from 'three/tsl';

// Dave Hoskins' sine-free hashes (stable on all GPUs for the coordinate ranges used here)
export const hash13 = /*@__PURE__*/ Fn( ( [ p_immutable ] ) => {

	const p3 = fract( vec3( p_immutable ).mul( 0.1031 ) ).toVar();
	p3.addAssign( dot( p3, p3.zyx.add( 31.32 ) ) );
	return fract( p3.x.add( p3.y ).mul( p3.z ) );

} ).setLayout( { name: 'atm_hash13', type: 'float', inputs: [ { name: 'p', type: 'vec3' } ] } );

export const hash33 = /*@__PURE__*/ Fn( ( [ p_immutable ] ) => {

	const p3 = fract( vec3( p_immutable ).mul( vec3( 0.1031, 0.1030, 0.0973 ) ) ).toVar();
	p3.addAssign( dot( p3, p3.yxz.add( 33.33 ) ) );
	return fract( p3.xxy.add( p3.yxx ).mul( p3.zyx ) );

} ).setLayout( { name: 'atm_hash33', type: 'vec3', inputs: [ { name: 'p', type: 'vec3' } ] } );

// floor-mod (WGSL % truncates toward zero)
const fmod3 = ( a, b ) => a.sub( b.mul( floor( a.div( b ) ) ) );

// 3D gradient noise (iq), quintic fade, range ≈ [-1, 1]
export const gnoise = /*@__PURE__*/ Fn( ( [ x ] ) => {

	const i = floor( x ).toVar();
	const f = fract( x ).toVar();
	const u = f.mul( f ).mul( f ).mul( f.mul( f.mul( 6.0 ).sub( 15.0 ) ).add( 10.0 ) ).toVar();
	const g = ( o ) => hash33( i.add( o ) ).mul( 2.0 ).sub( 1.0 );
	const v = ( o ) => dot( g( o ), f.sub( o ) );
	const va = v( vec3( 0, 0, 0 ) ), vb = v( vec3( 1, 0, 0 ) ), vc = v( vec3( 0, 1, 0 ) ), vd = v( vec3( 1, 1, 0 ) );
	const ve = v( vec3( 0, 0, 1 ) ), vf = v( vec3( 1, 0, 1 ) ), vg = v( vec3( 0, 1, 1 ) ), vh = v( vec3( 1, 1, 1 ) );
	const x0 = mix( mix( va, vb, u.x ), mix( vc, vd, u.x ), u.y );
	const x1 = mix( mix( ve, vf, u.x ), mix( vg, vh, u.x ), u.y );
	return mix( x0, x1, u.z ).mul( 1.6 );

} ).setLayout( { name: 'atm_gnoise', type: 'float', inputs: [ { name: 'x', type: 'vec3' } ] } );

// fBm of gradient noise with a runtime octave loop (keeps the WGSL small)
export const fbm = /*@__PURE__*/ Fn( ( [ p_immutable, octaves ] ) => {

	const p = vec3( p_immutable ).toVar();
	const sum = float( 0 ).toVar();
	const amp = float( 0.5 ).toVar();
	Loop( { start: int( 0 ), end: octaves, type: 'int' }, () => {

		sum.addAssign( gnoise( p ).mul( amp ) );
		// rotate-ish + scale between octaves to break lattice alignment
		p.assign( vec3( p.y.mul( 1.6 ).add( p.z.mul( 1.2 ) ), p.x.mul( - 1.6 ).add( p.z.mul( 0.72 ) ), p.x.mul( - 1.2 ).sub( p.y.mul( 0.96 ) ) ).mul( 0.98 ).add( vec3( 1.7, 9.2, 3.1 ) ) );
		amp.mulAssign( 0.5 );

	} );
	return sum;

} ).setLayout( { name: 'atm_fbm', type: 'float', inputs: [ { name: 'p', type: 'vec3' }, { name: 'octaves', type: 'int' } ] } );

// ridged multifractal (mountain chains)
export const ridged = /*@__PURE__*/ Fn( ( [ p_immutable, octaves ] ) => {

	const p = vec3( p_immutable ).toVar();
	const sum = float( 0 ).toVar();
	const amp = float( 0.5 ).toVar();
	const prev = float( 1 ).toVar();
	Loop( { start: int( 0 ), end: octaves, type: 'int' }, () => {

		const n = float( 1 ).sub( abs( gnoise( p ) ) ).toVar();
		n.assign( n.mul( n ) );
		sum.addAssign( n.mul( amp ).mul( prev ) );
		prev.assign( n );
		p.assign( vec3( p.y.mul( 1.6 ).add( p.z.mul( 1.2 ) ), p.x.mul( - 1.6 ).add( p.z.mul( 0.72 ) ), p.x.mul( - 1.2 ).sub( p.y.mul( 0.96 ) ) ).mul( 1.04 ).add( vec3( 5.3, 1.1, 7.7 ) ) );
		amp.mulAssign( 0.5 );

	} );
	return sum;

} ).setLayout( { name: 'atm_ridged', type: 'float', inputs: [ { name: 'p', type: 'vec3' }, { name: 'octaves', type: 'int' } ] } );

// 3D value noise with analytic derivatives (iq). returns vec4( value in [-1,1], d/dx, d/dy, d/dz )
export const vnoised = /*@__PURE__*/ Fn( ( [ x ] ) => {

	const i = floor( x ).toVar();
	const w = fract( x ).toVar();
	const u = w.mul( w ).mul( w ).mul( w.mul( w.mul( 6.0 ).sub( 15.0 ) ).add( 10.0 ) ).toVar();
	const du = w.mul( w ).mul( w.mul( w.sub( 2.0 ) ).add( 1.0 ) ).mul( 30.0 ).toVar();
	const a = hash13( i ), b = hash13( i.add( vec3( 1, 0, 0 ) ) ), c = hash13( i.add( vec3( 0, 1, 0 ) ) ), d = hash13( i.add( vec3( 1, 1, 0 ) ) );
	const e = hash13( i.add( vec3( 0, 0, 1 ) ) ), f = hash13( i.add( vec3( 1, 0, 1 ) ) ), g = hash13( i.add( vec3( 0, 1, 1 ) ) ), h = hash13( i.add( vec3( 1, 1, 1 ) ) );
	const k0 = a.toVar(), k1 = b.sub( a ).toVar(), k2 = c.sub( a ).toVar(), k3 = e.sub( a ).toVar();
	const k4 = a.sub( b ).sub( c ).add( d ).toVar(), k5 = a.sub( c ).sub( e ).add( g ).toVar();
	const k6 = a.sub( b ).sub( e ).add( f ).toVar();
	const k7 = a.negate().add( b ).add( c ).sub( d ).add( e ).sub( f ).sub( g ).add( h ).toVar();
	const val = k0.add( k1.mul( u.x ) ).add( k2.mul( u.y ) ).add( k3.mul( u.z ) ).add( k4.mul( u.x ).mul( u.y ) )
		.add( k5.mul( u.y ).mul( u.z ) ).add( k6.mul( u.z ).mul( u.x ) ).add( k7.mul( u.x ).mul( u.y ).mul( u.z ) );
	const dv = du.mul( vec3(
		k1.add( k4.mul( u.y ) ).add( k6.mul( u.z ) ).add( k7.mul( u.y ).mul( u.z ) ),
		k2.add( k5.mul( u.z ) ).add( k4.mul( u.x ) ).add( k7.mul( u.z ).mul( u.x ) ),
		k3.add( k6.mul( u.x ) ).add( k5.mul( u.y ) ).add( k7.mul( u.x ).mul( u.y ) ) ) );
	return vec4( val.mul( 2.0 ).sub( 1.0 ), dv.mul( 2.0 ) );

} ).setLayout( { name: 'atm_vnoised', type: 'vec4', inputs: [ { name: 'x', type: 'vec3' } ] } );

// fBm with derivatives and footprint-based octave fade (anti-aliasing): fp = pixel footprint in noise units
export const fbmd = /*@__PURE__*/ Fn( ( [ p_immutable, octaves, fp ] ) => {

	const p = vec3( p_immutable ).toVar();
	const acc = vec4( 0 ).toVar();
	const amp = float( 0.5 ).toVar();
	const freq = float( 1 ).toVar();
	Loop( { start: int( 0 ), end: octaves, type: 'int' }, () => {

		// fade octaves whose wavelength approaches the pixel footprint
		const fade = float( 1 ).sub( fp.mul( freq ).mul( 1.5 ).clamp( 0, 1 ) );
		const n = vnoised( p.mul( freq ).add( freq.mul( 3.7 ) ) );
		acc.addAssign( vec4( n.x, n.yzw.mul( freq ) ).mul( amp.mul( fade ) ) );
		amp.mulAssign( 0.5 );
		freq.mulAssign( 2.03 );

	} );
	return acc;

} ).setLayout( { name: 'atm_fbmd', type: 'vec4', inputs: [ { name: 'p', type: 'vec3' }, { name: 'octaves', type: 'int' }, { name: 'fp', type: 'float' } ] } );

// ───────────── tileable noise for the cloud volume (period in lattice cells) ─────────────

export const tgnoise = /*@__PURE__*/ Fn( ( [ x, period ] ) => {

	const i = floor( x ).toVar();
	const f = fract( x ).toVar();
	const u = f.mul( f ).mul( f ).mul( f.mul( f.mul( 6.0 ).sub( 15.0 ) ).add( 10.0 ) ).toVar();
	const P = vec3( period );
	const g = ( o ) => hash33( fmod3( i.add( o ), P ) ).mul( 2.0 ).sub( 1.0 );
	const v = ( o ) => dot( g( o ), f.sub( o ) );
	const va = v( vec3( 0, 0, 0 ) ), vb = v( vec3( 1, 0, 0 ) ), vc = v( vec3( 0, 1, 0 ) ), vd = v( vec3( 1, 1, 0 ) );
	const ve = v( vec3( 0, 0, 1 ) ), vf = v( vec3( 1, 0, 1 ) ), vg = v( vec3( 0, 1, 1 ) ), vh = v( vec3( 1, 1, 1 ) );
	const x0 = mix( mix( va, vb, u.x ), mix( vc, vd, u.x ), u.y );
	const x1 = mix( mix( ve, vf, u.x ), mix( vg, vh, u.x ), u.y );
	return mix( x0, x1, u.z ).mul( 1.6 );

} ).setLayout( { name: 'atm_tgnoise', type: 'float', inputs: [ { name: 'x', type: 'vec3' }, { name: 'period', type: 'float' } ] } );

// tileable Worley (F1), returns 1 - distance (billowy cells)
export const tworley = /*@__PURE__*/ Fn( ( [ x, period ] ) => {

	const i = floor( x ).toVar();
	const f = fract( x ).toVar();
	const P = vec3( period );
	const d = float( 8 ).toVar();
	Loop( { start: int( - 1 ), end: int( 2 ), type: 'int', name: 'oz' }, ( { oz } ) => {

		Loop( { start: int( - 1 ), end: int( 2 ), type: 'int', name: 'oy' }, ( { oy } ) => {

			Loop( { start: int( - 1 ), end: int( 2 ), type: 'int', name: 'ox' }, ( { ox } ) => {

				const o = vec3( float( ox ), float( oy ), float( oz ) );
				const fp = o.add( hash33( fmod3( i.add( o ), P ) ) ).sub( f );
				d.assign( min( d, dot( fp, fp ) ) );

			} );

		} );

	} );
	return float( 1 ).sub( sqrt( d ) ).clamp( 0, 1 );

} ).setLayout( { name: 'atm_tworley', type: 'float', inputs: [ { name: 'x', type: 'vec3' }, { name: 'period', type: 'float' } ] } );

// remap v from [a,b] to [c,d] (no clamping)
export const remap = ( v, a, b, c, d ) => float( v ).sub( a ).div( float( b ).sub( a ) ).mul( float( d ).sub( c ) ).add( c );

export const interleavedGradientNoise = ( xy ) => fract( fract( xy.x.mul( 0.06711056 ).add( xy.y.mul( 0.00583715 ) ) ).mul( 52.9829189 ) );

export { max, abs };
