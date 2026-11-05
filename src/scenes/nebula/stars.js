// Star sprites: ionising cluster, embedded young stars / protostars, background field.
// Each star's column density through the nebula (camera → star) is computed per frame on the GPU,
// then the sprite is extinguished + reddened accordingly, so embedded stars glow through thin gas
// and vanish behind pillars without any depth compositing.
import * as THREE from 'three/webgpu';
import { Fn, vec2, vec3, vec4, float, uv, exp, abs, dot, max, pow, smoothstep, instancedArray, clamp, mix, select } from 'three/tsl';
import { CLUSTER, PILLARS, JETS, BOX_MIN, BOX_SIZE, MAX_USER_STARS } from './volume.js';

function rng( seed ) {

	let s = seed >>> 0;
	return () => {

		s = ( s + 0x6D2B79F5 ) >>> 0;
		let t = s;
		t = Math.imul( t ^ ( t >>> 15 ), t | 1 );
		t ^= t + Math.imul( t ^ ( t >>> 7 ), t | 61 );
		return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

	};

}

// approximate blackbody tint (linear sRGB, normalised to max 1)
function bbColor( tempK ) {

	const t = tempK / 100;
	let r, g, b;
	if ( t <= 66 ) {

		r = 255; g = 99.47 * Math.log( t ) - 161.12;
		b = t <= 19 ? 0 : 138.52 * Math.log( t - 10 ) - 305.04;

	} else {

		r = 329.7 * Math.pow( t - 60, - 0.1332 ); g = 288.12 * Math.pow( t - 60, - 0.0755 ); b = 255;

	}

	const c = [ r, g, b ].map( ( v ) => Math.pow( Math.min( Math.max( v, 0 ), 255 ) / 255, 2.2 ) );
	const m = Math.max( ...c );
	return c.map( ( v ) => v / m );

}

export function buildStars( ctx, vol, U ) {

	const tier = ctx.tier;
	const R = rng( 1337 );
	const gauss = () => Math.sqrt( - 2 * Math.log( R() + 1e-9 ) ) * Math.cos( 6.2831853 * R() );
	const list = []; // [x,y,z,kind, r,g,b,flux]

	// 0. reserved slots for user-ignited stars (kind 3+k); parked far below until ignited
	for ( let k = 0; k < MAX_USER_STARS; k ++ ) list.push( [ 0, - 5000, 0, 3 + k, ...bbColor( 26000 ), 60 ] );

	// 1. the ionising cluster: Plummer-ish clump of hot O/B stars
	const nCluster = tier === 0 ? 28 : 64;
	for ( let i = 0; i < nCluster; i ++ ) {

		const rr = 0.85 * Math.pow( R(), 1.6 );
		const th = Math.acos( 2 * R() - 1 ), ph = 6.2831853 * R();
		const x = CLUSTER.x + rr * Math.sin( th ) * Math.cos( ph );
		const y = CLUSTER.y + rr * Math.cos( th ) * 0.8;
		const z = CLUSTER.z + rr * Math.sin( th ) * Math.sin( ph );
		const c = bbColor( 9000 + 30000 * R() );
		const flux = i < 5 ? 40 + 70 * R() : 3 + 18 * Math.pow( R(), 2 );
		list.push( [ x, y, z, 1, ...c, flux ] );

	}

	// 2. embedded young stars & protostars (pillar heads, jet sources, scattered through the gas)
	for ( const J of JETS ) list.push( [ ...J.o, 2, ...bbColor( 3800 ), 22 ] );
	for ( const pl of PILLARS ) {

		for ( let i = 0; i < 5; i ++ ) {

			const s = pl.rb * 1.3;
			list.push( [ pl.b[ 0 ] + gauss() * s, pl.b[ 1 ] + gauss() * s, pl.b[ 2 ] + gauss() * s, 2, ...bbColor( 3500 + 4000 * R() ), 3 + 10 * R() ] );

		}

	}

	const nEmb = tier === 0 ? 260 : 1400;
	for ( let i = 0; i < nEmb; i ++ ) {

		const x = BOX_MIN.x + BOX_SIZE.x * R(), y = BOX_MIN.y + BOX_SIZE.y * R(), z = BOX_MIN.z + BOX_SIZE.z * R();
		const c = bbColor( 3200 + 12000 * Math.pow( R(), 2 ) );
		list.push( [ x, y, z, 2, ...c, 0.4 + 3.5 * Math.pow( R(), 5 ) ] );

	}

	// 3. background field on a far sphere (power-law luminosity function)
	const nBg = tier === 0 ? 3500 : 22000;
	for ( let i = 0; i < nBg; i ++ ) {

		const u = 2 * R() - 1, ph = 6.2831853 * R();
		const s = Math.sqrt( 1 - u * u ), rad = 420;
		const temp = 2800 + 9000 * Math.pow( R(), 1.8 );
		const flux = 0.05 + 0.9 * Math.pow( R(), 9 ) * 6 + 0.25 * R();
		list.push( [ s * Math.cos( ph ) * rad, u * rad, s * Math.sin( ph ) * rad, 0, ...bbColor( temp ), flux ] );

	}

	const N = list.length;
	const posArr = new Float32Array( N * 4 ), colArr = new Float32Array( N * 4 );
	list.forEach( ( s, i ) => {

		posArr.set( [ s[ 0 ], s[ 1 ], s[ 2 ], s[ 3 ] ], i * 4 );
		colArr.set( [ s[ 4 ], s[ 5 ], s[ 6 ], s[ 7 ] ], i * 4 );

	} );

	const pos = instancedArray( posArr, 'vec4' );
	const col = instancedArray( colArr, 'vec4' );
	const tau = instancedArray( N, 'float' );
	const tauKernel = vol.makeStarTauKernel( pos, tau, N, tier === 0 ? 24 : 48 );

	// user-ignited stars rendered by the same material from a small uniform-driven buffer
	const mat = makeStarMaterial( pos.toAttribute(), col.toAttribute(), tau.toAttribute(), U );
	const sprite = new THREE.Sprite( mat );
	sprite.count = N;
	sprite.frustumCulled = false;
	sprite.renderOrder = 1;

	// move a user slot (only the first MAX_USER_STARS entries are touched → tiny partial upload)
	const setUserStar = ( k, p ) => {

		const a = pos.value;
		a.array[ k * 4 ] = p.x; a.array[ k * 4 + 1 ] = p.y; a.array[ k * 4 + 2 ] = p.z;
		a.clearUpdateRanges?.();
		a.addUpdateRange?.( 0, MAX_USER_STARS * 4 );
		a.needsUpdate = true;

	};

	return {
		sprite, count: N, tauKernel, setUserStar,
		dispose() {

			mat.dispose();
			for ( const b of [ pos, col, tau ] ) b.value?.dispose?.();
			tauKernel.dispose?.();

		},
	};

}

// JWST-like PSF: gaussian core + faint halo + 6 hexagonal-mirror spikes + 2 faint strut spikes
export function makeStarMaterial( posA, colA, tauA, U, opts = {} ) {

	const mat = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending } );
	mat.sizeAttenuation = false;
	mat.positionNode = posA.xyz;

	const kind = posA.w; // 0 bg, 1 cluster, 2 embedded, 3+k user slot k
	let userGain = float( 1 );
	for ( let k = 0; k < MAX_USER_STARS; k ++ ) userGain = select( kind.equal( 3 + k ), U.starPar[ k ].x.mul( 0.5 ), userGain );
	const flux = colA.w.mul( userGain );
	const bright = flux.mul( U.starGain );
	// spikes only for bright stars; quad grows to hold them
	const spikeLen = clamp( pow( max( bright.sub( 1.2 ), 0.0 ), 0.5 ).mul( 9.0 ).mul( U.pxRef ), 0.0, U.pxRef.mul( 150.0 ) );
	const sizePx = spikeLen.mul( 2.0 ).add( U.pxRef.mul( 7.0 ) ).add( 4.0 );
	mat.scaleNode = sizePx.mul( U.pxScale );

	mat.colorNode = Fn( () => {

		const q = uv().sub( 0.5 ).mul( sizePx ).toVar(); // pixels from centre
		const r2 = dot( q, q );
		const core = exp( r2.mul( - 0.5 / ( 0.85 * 0.85 ) ) );
		const halo = float( 1 ).div( r2.div( U.pxRef.mul( U.pxRef ).mul( 6.0 ) ).add( 1.0 ) ).mul( 0.035 );
		let sp = float( 0 );
		const L = max( spikeLen, 1.0 );
		const dirs = [ [ 0, 1, 1.0 ], [ 0.8660254, 0.5, 1.0 ], [ - 0.8660254, 0.5, 1.0 ], [ 1, 0, 0.45 ] ];
		for ( const [ dx, dy, wgt ] of dirs ) {

			const along = abs( dot( q, vec2( dx, dy ) ) );
			const perp = abs( dot( q, vec2( - dy, dx ) ) );
			const len = wgt < 1 ? L.mul( 0.55 ) : L;
			const fall = pow( max( float( 1 ).sub( along.div( len ) ), 0.0 ), 2.2 );
			sp = sp.add( exp( perp.mul( perp ).mul( - 0.9 ) ).mul( fall ).mul( wgt ) );

		}

		const spikes = sp.mul( smoothstep( 1.2, 6.0, bright ) ).mul( 0.22 );
		// interstellar extinction + reddening from the GPU-integrated column density
		const T = exp( vec3( 0.55, 0.8, 1.2 ).mul( tauA.mul( U.density ).mul( U.extinction ) ).negate() );
		const I = core.add( halo ).add( spikes ).mul( bright );
		const bgDim = mix( float( 1.0 ), U.bgStars, select( kind.lessThan( 0.5 ), 1.0, 0.0 ) );
		return vec4( colA.rgb.mul( T ).mul( I ).mul( bgDim ), 1.0 );

	} )();

	return mat;

}
