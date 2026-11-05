// GPU-generated nebula volume: density field + light (column density) volume + tileable detail noise,
// all written by compute shaders into Storage3DTextures.
import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, Break, uniform, vec3, vec4, float, uvec3, globalId, instanceIndex, textureStore, texture3D,
	length, normalize, smoothstep, mix, max, min, exp, abs, dot, clamp, pow, select, instancedArray,
} from 'three/tsl';
import { fbm, ridged, fbmP, ridgedP } from './noise.js';

// world-space volume box (pillars stand along +Y, camera looks down -Z)
export const BOX_MIN = new THREE.Vector3( - 8, - 7, - 6 );
export const BOX_MAX = new THREE.Vector3( 8, 7, 4 );
export const BOX_SIZE = new THREE.Vector3().subVectors( BOX_MAX, BOX_MIN );

// the ionising cluster (NGC 6611-like), up and behind the pillars
export const CLUSTER = new THREE.Vector3( 2.7, 4.7, - 3.3 );

// three "Pillars of Creation"-like columns: base, mid, head + radii (world units)
export const PILLARS = [
	{ a: [ - 3.5, - 7.8, - 0.9 ], m: [ - 3.05, - 1.6, - 1.25 ], b: [ - 2.35, 3.3, - 1.7 ], ra: 1.9, rm: 1.15, rb: 0.95 },
	{ a: [ 0.55, - 7.8, - 1.6 ], m: [ 0.9, - 3.4, - 1.8 ], b: [ 1.3, 0.25, - 2.05 ], ra: 1.45, rm: 0.92, rb: 0.78 },
	{ a: [ 4.4, - 7.8, - 0.5 ], m: [ 4.05, - 4.7, - 0.65 ], b: [ 3.75, - 2.15, - 0.85 ], ra: 1.15, rm: 0.78, rb: 0.62 },
];

// Herbig–Haro jets (origin, axis, half-length) from protostars buried in the pillar heads
export const JETS = [
	{ o: [ - 2.2, 3.0, - 1.35 ], d: [ 0.93, 0.22, 0.30 ], len: 2.3 },
	{ o: [ 1.25, - 0.05, - 1.8 ], d: [ - 0.88, 0.38, 0.28 ], len: 1.5 },
];

export const MAX_USER_STARS = 3;

const v3 = ( a ) => vec3( a[ 0 ], a[ 1 ], a[ 2 ] );

const smin = ( a, b, k ) => {

	const h = max( float( k ).sub( abs( a.sub( b ) ) ), 0.0 ).div( k );
	return min( a, b ).sub( h.mul( h ).mul( k * 0.25 ) );

};

// tapered capsule between a and b
const sdTaper = ( p, a, b, ra, rb ) => {

	const pa = p.sub( a ), ba = b.sub( a );
	const h = clamp( dot( pa, ba ).div( dot( ba, ba ) ), 0.0, 1.0 );
	return length( pa.sub( ba.mul( h ) ) ).sub( mix( float( ra ), float( rb ), h ) );

};

function make3D( nx, ny, nz, name, wrap = THREE.ClampToEdgeWrapping ) {

	const t = new THREE.Storage3DTexture( nx, ny, nz );
	t.name = name;
	t.type = THREE.HalfFloatType;
	t.format = THREE.RGBAFormat;
	t.generateMipmaps = false;
	t.minFilter = THREE.LinearFilter;
	t.magFilter = THREE.LinearFilter;
	t.wrapS = t.wrapT = t.wrapR = wrap;
	return t;

}

export class NebulaVolume {

	constructor( renderer, cfg, U ) {

		this.renderer = renderer;
		this.cfg = cfg; // { res:[nx,ny,nz], detailRes, lightSteps, userSteps, slab }
		this.U = U; // shared uniforms
		const [ nx, ny, nz ] = cfg.res;
		this.res = cfg.res;

		this.densityTex = make3D( nx, ny, nz, 'nebula.density' );
		this.lightTex = make3D( nx, ny, nz, 'nebula.light' );
		this.detailTex = make3D( cfg.detailRes, cfg.detailRes, cfg.detailRes, 'nebula.detail', THREE.RepeatWrapping );

		// sampler nodes (base nodes; .sample() clones share one binding)
		this.densNode = texture3D( this.densityTex );
		this.lightNode = texture3D( this.lightTex );
		this.detailNode = texture3D( this.detailTex );

		this.bmin = vec3( BOX_MIN.x, BOX_MIN.y, BOX_MIN.z );
		this.bsize = vec3( BOX_SIZE.x, BOX_SIZE.y, BOX_SIZE.z );
		this.invSize = vec3( 1 / BOX_SIZE.x, 1 / BOX_SIZE.y, 1 / BOX_SIZE.z );
		this.voxel = vec3( BOX_SIZE.x / nx, BOX_SIZE.y / ny, BOX_SIZE.z / nz );

		this.pickBuf = instancedArray( 1, 'vec4' );
		this.kernels = {};

	}

	// ───────── density field ─────────
	densityField( p ) {

		const U = this.U;
		const t = U.drift;
		// large-scale, slowly evolving domain warp
		const q = p.mul( 0.16 ).add( vec3( t.mul( 0.011 ), t.mul( - 0.007 ), t.mul( 0.017 ) ) );
		const w = vec3( fbm( q, 3 ), fbm( q.add( vec3( 5.2, 1.3, 2.8 ) ), 3 ), fbm( q.add( vec3( 9.1, 4.7, 7.3 ) ), 3 ) ).toVar();
		const pw = p.add( w.mul( 1.25 ) ).toVar();

		// pillars (two tapered segments + bulbous head each)
		let dP = null;
		for ( const pl of PILLARS ) {

			const a = v3( pl.a ), m = v3( pl.m ), b = v3( pl.b );
			const d1 = sdTaper( pw, a, m, pl.ra, pl.rm );
			const d2 = sdTaper( pw, m, b, pl.rm, pl.rb );
			const d3 = length( pw.sub( b ) ).sub( pl.rb * 1.22 );
			const d = smin( smin( d1, d2, 0.6 ), d3, 0.55 );
			dP = dP ? smin( dP, d, 0.9 ) : d;

		}

		// bottom molecular cloud bank, rising toward the back
		const hB = float( - 4.6 ).add( fbm( vec3( pw.x.mul( 0.21 ), 1.7, pw.z.mul( 0.21 ) ), 3 ).mul( 1.5 ) )
			.add( smoothstep( - 1.0, - 6.0, pw.z ).mul( 1.6 ) );
		const dBank = pw.y.sub( hB ).mul( 0.75 );

		// turbulent erosion of every surface + striations along the pillars
		const n1 = fbm( p.mul( 0.55 ).add( w.mul( 0.6 ) ), 5 ).toVar();
		const str = ridged( pw.mul( vec3( 0.95, 0.32, 0.95 ) ).add( 2.0 ), 3 );
		const d = smin( dP, dBank, 1.3 ).add( n1.mul( 0.62 ) ).sub( str.mul( 0.28 ) ).toVar();

		const core = smoothstep( 0.3, - 0.8, d ).toVar();
		const rhoCore = core.mul( n1.mul( 0.55 ).add( 1.05 ) ).mul( 1.7 );

		// back wall: a lumpy sheet of gas behind the cluster, lit face-on
		const zW = float( - 5.25 ).add( fbm( vec3( pw.x.mul( 0.17 ), pw.y.mul( 0.17 ), 3.1 ), 3 ).mul( 1.1 ) );
		const dWall = abs( pw.z.sub( zW ) ).sub( 0.75 ).add( n1.mul( 0.7 ) );
		const wall = smoothstep( 0.45, - 0.55, dWall ).mul( n1.mul( 0.4 ).add( 0.6 ) ).mul( 0.75 );

		// diffuse HII gas + ridged filaments/sheets (where light shafts show up)
		const nd = fbm( p.mul( 0.21 ).add( w.mul( 0.75 ) ).add( vec3( 0, 0, 7.7 ) ), 4 ).mul( 0.5 ).add( 0.5 ).toVar();
		const rhoDiff = float( 0.018 ).add( pow( nd, 3.0 ).mul( 0.16 ) );
		const rf = ridged( pw.mul( 0.27 ).add( vec3( 3.0, 1.0, 5.0 ) ), 4 );
		const rhoFil = pow( rf, 9.0 ).mul( 0.42 );

		const rho = rhoCore.add( wall ).add( rhoDiff ).add( rhoFil ).toVar();

		// the cluster has blown a cavity around itself
		const dc = length( p.sub( v3( CLUSTER.toArray() ) ) );
		rho.mulAssign( mix( 0.06, 1.0, smoothstep( 0.6, 3.2, dc.add( n1.mul( 0.6 ) ) ) ) );

		// user-ignited stars carve expanding ionisation bubbles with compressed shells
		for ( let k = 0; k < MAX_USER_STARS; k ++ ) {

			const sp = U.starPos[ k ], sa = U.starPar[ k ];
			const r = length( p.sub( sp.xyz ) );
			const R = max( sp.w.mul( n1.mul( 0.22 ).add( 1.0 ) ), 0.02 );
			const inside = smoothstep( R.mul( 0.55 ), R, r );
			const shellW = R.mul( 0.22 ).add( 0.08 );
			const x = r.sub( R ).div( shellW );
			const shell = exp( x.mul( x ).negate() );
			const carve = inside.mul( shell.mul( 1.5 ).add( 1.0 ) ).add( 0.015 );
			rho.mulAssign( mix( float( 1.0 ), carve, sa.y ) );

		}

		// soft fade at the box faces (no hard slabs)
		const e0 = p.sub( this.bmin ), e1 = this.bmin.add( this.bsize ).sub( p );
		const edge = min( min( min( e0.x, e1.x ), min( e0.y, e1.y ) ), min( e0.z, e1.z ) );
		rho.mulAssign( smoothstep( 0.0, 1.1, edge ) );

		const dust = clamp( core.add( wall.mul( 0.6 ) ), 0.0, 1.0 );
		return vec4( rho, dust, nd, 1.0 );

	}

	_inBounds( vf ) {

		const [ nx, ny, nz ] = this.res;
		return vf.x.greaterThanEqual( 0 ).and( vf.y.greaterThanEqual( 0 ) ).and( vf.z.greaterThanEqual( 0 ) )
			.and( vf.x.lessThan( nx ) ).and( vf.y.lessThan( ny ) ).and( vf.z.lessThan( nz ) );

	}

	makeDensityKernel( offset, dispatch ) {

		return Fn( () => {

			const vf = vec3( globalId ).add( offset ).toVar();
			If( this._inBounds( vf ), () => {

				const p = this.bmin.add( vf.add( 0.5 ).mul( this.voxel ) ).toVar();
				textureStore( this.densityTex, uvec3( vf ), this.densityField( p ) );

			} );

		} )().compute( dispatch, [ 4, 4, 4 ] ).setName( 'nebula.density' );

	}

	// optical depth (column density, in density·world units) from p toward target
	marchTau( p, target, steps ) {

		const dv = target.sub( p );
		const dist = max( length( dv ), 1e-4 );
		const dir = dv.div( dist ).toVar();
		const ds = dist.div( steps ).toVar();
		const tau = float( 0 ).toVar();
		Loop( steps, ( { i } ) => {

			const s = float( i ).add( 0.5 ).mul( ds );
			const q = p.add( dir.mul( s ) );
			tau.addAssign( this.densNode.sample( q.sub( this.bmin ).mul( this.invSize ) ).level( 0 ).r );

		} );
		return tau.mul( ds );

	}

	makeLightKernel( offset, dispatch ) {

		const U = this.U;
		const { lightSteps, userSteps } = this.cfg;
		return Fn( () => {

			const vf = vec3( globalId ).add( offset ).toVar();
			If( this._inBounds( vf ), () => {

				const p = this.bmin.add( vf.add( 0.5 ).mul( this.voxel ) ).toVar();
				const tauC = this.marchTau( p, v3( CLUSTER.toArray() ), lightSteps );
				const tu = [];
				for ( let k = 0; k < MAX_USER_STARS; k ++ ) {

					const sp = U.starPos[ k ], sa = U.starPar[ k ];
					const t = float( 1000.0 ).toVar();
					If( sa.y.greaterThan( 0.001 ).and( length( p.sub( sp.xyz ) ).lessThan( sa.z ) ), () => {

						t.assign( this.marchTau( p, sp.xyz, userSteps ) );

					} );
					tu.push( t );

				}

				textureStore( this.lightTex, uvec3( vf ), vec4( tauC, tu[ 0 ], tu[ 1 ], tu[ 2 ] ) );

			} );

		} )().compute( dispatch, [ 4, 4, 4 ] ).setName( 'nebula.light' );

	}

	makeDetailKernel() {

		const N = this.cfg.detailRes;
		const P = 4; // lattice period
		return Fn( () => {

			const p = vec3( globalId ).add( 0.5 ).mul( P / N ).toVar();
			const a = fbmP( p, P, 4 ).mul( 0.5 ).add( 0.5 );
			const b = ridgedP( p.add( vec3( 1, 2, 3 ) ), P, 3 );
			const c = fbmP( p.add( vec3( 2, 0, 1 ) ), P, 3 ).mul( 0.5 ).add( 0.5 );
			textureStore( this.detailTex, uvec3( globalId ), vec4( a, b, c, 1.0 ) );

		} )().compute( [ N / 4, N / 4, N / 4 ], [ 4, 4, 4 ] ).setName( 'nebula.detail' );

	}

	// Single-thread ray pick: first point along the pointer ray where the cloud becomes optically thick.
	makePickKernel() {

		const U = this.U;
		const out = this.pickBuf;
		return Fn( () => {

			const ro = U.pickO, rd = U.pickD;
			const inv = vec3( 1.0 ).div( rd );
			const t0 = this.bmin.sub( ro ).mul( inv ), t1 = this.bmin.add( this.bsize ).sub( ro ).mul( inv );
			const tmn = min( t0, t1 ), tmx = max( t0, t1 );
			const tN = max( max( tmn.x, tmn.y ), max( tmn.z, 0.0 ) ).toVar();
			const tF = min( tmx.x, min( tmx.y, tmx.z ) ).toVar();
			const res = vec4( 0, 0, 0, - 1 ).toVar();
			If( tF.greaterThan( tN ), () => {

				const n = 320;
				const dt = tF.sub( tN ).div( n ).toVar();
				const tau = float( 0 ).toVar();
				const best = float( 0 ).toVar();
				const bestP = vec3( 0 ).toVar();
				Loop( n, ( { i } ) => {

					const p = ro.add( rd.mul( tN.add( float( i ).add( 0.5 ).mul( dt ) ) ) ).toVar();
					const rho = this.densNode.sample( p.sub( this.bmin ).mul( this.invSize ) ).level( 0 ).r.mul( U.density );
					tau.addAssign( rho.mul( dt ).mul( U.extinction ) );
					If( rho.greaterThan( best ), () => {

						best.assign( rho ); bestP.assign( p );

					} );
					If( tau.greaterThan( 1.6 ), () => {

						res.assign( vec4( p, 1.0 ) );
						Break();

					} );

				} );
				If( res.w.lessThan( 0.0 ).and( best.greaterThan( 0.03 ) ), () => {

					res.assign( vec4( bestP, 0.5 ) );

				} );
				If( res.w.lessThan( 0.0 ), () => {

					res.assign( vec4( ro.add( rd.mul( mix( tN, tF, 0.5 ) ) ), 0.25 ) );

				} );

			} );
			out.element( 0 ).assign( res );

		} )().compute( 1 ).setName( 'nebula.pick' );

	}

	// per-star column density camera → star (stars are then attenuated + reddened in their sprite shader)
	makeStarTauKernel( starPos, starTau, count, steps ) {

		const U = this.U;
		return Fn( () => {

			const s = starPos.element( instanceIndex );
			const ro = U.camPos;
			const dv = s.xyz.sub( ro );
			const L = length( dv );
			const rd = dv.div( L ).toVar();
			const inv = vec3( 1.0 ).div( rd );
			const t0 = this.bmin.sub( ro ).mul( inv ), t1 = this.bmin.add( this.bsize ).sub( ro ).mul( inv );
			const tmn = min( t0, t1 ), tmx = max( t0, t1 );
			const tN = max( max( tmn.x, tmn.y ), max( tmn.z, 0.0 ) ).toVar();
			const tF = min( min( tmx.x, min( tmx.y, tmx.z ) ), L ).toVar();
			const tau = float( 0 ).toVar();
			If( tF.greaterThan( tN ), () => {

				const dt = tF.sub( tN ).div( steps ).toVar();
				Loop( steps, ( { i } ) => {

					const p = ro.add( rd.mul( tN.add( float( i ).add( 0.5 ).mul( dt ) ) ) );
					tau.addAssign( this.densNode.sample( p.sub( this.bmin ).mul( this.invSize ) ).level( 0 ).r );

				} );
				tau.mulAssign( dt );

			} );
			starTau.element( instanceIndex ).assign( tau );

		} )().compute( count ).setName( 'nebula.starTau' );

	}

	dispose() {

		this.densityTex.dispose();
		this.lightTex.dispose();
		this.detailTex.dispose();
		for ( const k in this.kernels ) this.kernels[ k ]?.dispose?.();
		this.pickBuf.value?.dispose?.();

	}

}

export { select };
