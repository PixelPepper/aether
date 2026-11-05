// Morphogenesis — GPU 3D Gray–Scott reaction–diffusion.
//
// State lives in two ping-pong storage buffers of vec2(u, v), one per voxel of a G³ grid.
// The step kernel runs in 8×8×4 workgroups: each workgroup cooperatively loads its 10×10×6 tile
// (block + 1-voxel halo, edges and corners included) into workgroup shared memory, barriers, and then
// evaluates a 19-point isotropic Laplacian (faces + edges) entirely from shared memory.
// After the iterations a bake pass writes the (shape-masked) v field, a "fresh growth" glow channel and u
// into an rgba16float 3D texture for hardware-trilinear raymarching, and two brick passes build a coarse
// Chebyshev distance field (4³-voxel bricks) used for empty-space skipping by the raymarcher.
import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, Break, uniform, instancedArray, workgroupArray, workgroupBarrier, workgroupId, localId, globalId,
	instanceIndex, uint, int, float, vec2, vec3, vec4, ivec3, uvec3, max, min, abs, clamp, smoothstep, mix, exp,
	length, dot, textureStore, hash, floor, sin, cos, select,
} from 'three/tsl';

export const WG = [ 8, 8, 4 ];
const TX = WG[ 0 ] + 2, TY = WG[ 1 ] + 2, TZ = WG[ 2 ] + 2, TILE = TX * TY * TZ, WGN = WG[ 0 ] * WG[ 1 ] * WG[ 2 ];
export const BRICK = 4;
const BRICK_R = 3; // brick distance search radius (bricks); distances saturate at BRICK_R + 1

export function createSim( renderer, G ) {

	const N = G * G * G;
	const BG = G / BRICK;
	const NB = BG * BG * BG;

	const bufA = instancedArray( N, 'vec2' ).setName( 'rdA' );
	const bufB = instancedArray( N, 'vec2' ).setName( 'rdB' );
	const aux = instancedArray( N, 'vec2' ).setName( 'rdAux' ); // (v at last bake, fresh-growth glow)
	const occ = instancedArray( NB, 'float' ).setName( 'rdOcc' );
	const pick = instancedArray( 1, 'vec4' ).setName( 'rdPick' ); // xyz = grid coords of pointer hit, w = hit flag

	const volTex = new THREE.Storage3DTexture( G, G, G );
	volTex.type = THREE.HalfFloatType;
	volTex.format = THREE.RGBAFormat;
	volTex.minFilter = volTex.magFilter = THREE.LinearFilter;
	volTex.wrapS = volTex.wrapT = volTex.wrapR = THREE.ClampToEdgeWrapping;
	volTex.generateMipmaps = false;

	const brickTex = new THREE.Storage3DTexture( BG, BG, BG );
	brickTex.type = THREE.HalfFloatType;
	brickTex.format = THREE.RGBAFormat;
	brickTex.minFilter = brickTex.magFilter = THREE.NearestFilter;
	brickTex.wrapS = brickTex.wrapT = brickTex.wrapR = THREE.ClampToEdgeWrapping;
	brickTex.generateMipmaps = false;

	const u = {
		feed: uniform( 0.037 ),
		kill: uniform( 0.06 ),
		spread: uniform( 1 ), // amount of spatial parameter variation
		dFeed: uniform( 0.0 ), // parameter-space direction of the variation (per preset)
		dKill: uniform( 0.004 ),
		gradDir: uniform( new THREE.Vector3( 0, 1, 0 ) ),
		du: uniform( 0.1 ),
		ratio: uniform( 0.5 ),
		thr: uniform( 0.22 ), // isosurface threshold (render side, used for occupancy + picking)
		shapeR: uniform( 0.43 ), // radius of the sculpture mask, fraction of the grid
		shapeSoft: uniform( 0.09 ),
		shapeT: uniform( 0 ), // time for the slowly breathing silhouette
		fade: uniform( 0.97 ), // per-frame decay of the growth glow
		glowGain: uniform( 6.0 ),
		seedPos: uniform( new THREE.Vector3() ),
		seedR: uniform( 4.0 ),
		seedAmt: uniform( 0.6 ),
		seedSalt: uniform( 0 ),
		seedDensity: uniform( 0.5 ),
		seedClear: uniform( 1 ),
		rayO: uniform( new THREE.Vector3() ), // pointer ray in grid coords
		rayD: uniform( new THREE.Vector3( 0, 0, 1 ) ),
	};

	const Gf = float( G );
	const gIndex = ( g ) => g.x.add( g.y.mul( G ) ).add( g.z.mul( G * G ) );

	// Sculpture silhouette: a slightly squashed, gently lobed sphere, in normalized coords p ∈ [-1, 1]³.
	// Returns a 0..1 mask (1 deep inside). Shared by sim (kill ramp outside) and bake (render mask).
	const shapeDist = ( p ) => {

		const q = p.mul( vec3( 1.0, 1.08, 1.0 ) );
		const r = length( q );
		const lobe = sin( q.x.mul( 2.3 ).add( u.shapeT ) ).mul( sin( q.y.mul( 1.9 ).sub( u.shapeT.mul( 0.7 ) ) ) ).mul( sin( q.z.mul( 2.1 ).add( 1.3 ) ) ).mul( 0.06 );
		return r.sub( u.shapeR.mul( 2 ).add( lobe ) );

	};
	const shapeMask = ( p, soft ) => smoothstep( float( 0 ), soft.negate(), shapeDist( p ) );

	// Spatially varying feed / kill: a linear gradient across the volume along gradDir (spots → worms → mazes → holes)
	const paramsAt = ( p ) => {

		const s = dot( p, u.gradDir ).mul( u.spread );
		const F = u.feed.add( u.dFeed.mul( s ) );
		// outside the sculpture region the kill rate rises so nothing grows in the far corners (saves nothing, but
		// keeps the brick field empty there → the raymarcher skips it)
		const outside = smoothstep( 0.04, 0.2, shapeDist( p ) );
		const K = u.kill.add( u.dKill.mul( s ) ).add( outside.mul( 0.03 ) );
		return { F, K };

	};

	// ── reaction–diffusion step with shared-memory tiling ──
	function makeStep( src, dst ) {

		const tile = workgroupArray( 'vec2', TILE );

		return Fn( () => {

			const base = ivec3( workgroupId.mul( uvec3( WG[ 0 ], WG[ 1 ], WG[ 2 ] ) ) ).sub( 1 ).toConst();
			const li = localId.x.add( localId.y.mul( WG[ 0 ] ) ).add( localId.z.mul( WG[ 0 ] * WG[ 1 ] ) ).toConst();

			// cooperative load: 256 invocations fill a 600-cell tile (≈2.3 loads each)
			Loop( { start: li, end: uint( TILE ), type: 'uint', condition: '<', update: WGN }, ( { i } ) => {

				const t = ivec3( int( i.mod( TX ) ), int( i.div( TX ).mod( TY ) ), int( i.div( TX * TY ) ) );
				const g = clamp( base.add( t ), ivec3( 0 ), ivec3( G - 1 ) );
				tile.element( i ).assign( src.element( gIndex( g ) ) );

			} );

			workgroupBarrier();

			const c = int( localId.x.add( 1 ).add( localId.y.add( 1 ).mul( TX ) ).add( localId.z.add( 1 ).mul( TX * TY ) ) ).toConst();
			const T = ( o ) => tile.element( c.add( o ) );
			const X = 1, Y = TX, Z = TX * TY;

			const faces = T( X ).add( T( - X ) ).add( T( Y ) ).add( T( - Y ) ).add( T( Z ) ).add( T( - Z ) );
			const edges = T( X + Y ).add( T( X - Y ) ).add( T( - X + Y ) ).add( T( - X - Y ) )
				.add( T( X + Z ) ).add( T( X - Z ) ).add( T( - X + Z ) ).add( T( - X - Z ) )
				.add( T( Y + Z ) ).add( T( Y - Z ) ).add( T( - Y + Z ) ).add( T( - Y - Z ) );
			const cc = T( 0 ).toConst();
			// 19-point isotropic Laplacian: (2·Σfaces + Σedges − 24·c) / 6
			const lap = faces.mul( 2 ).add( edges ).sub( cc.mul( 24 ) ).div( 6 ).toConst();

			const gid = globalId.toConst();
			const p = vec3( gid ).add( 0.5 ).div( Gf ).mul( 2 ).sub( 1 );
			const { F, K } = paramsAt( p );

			const U = cc.x, V = cc.y;
			const uvv = U.mul( V ).mul( V ).toConst();
			const nu = U.add( u.du.mul( lap.x ) ).sub( uvv ).add( F.mul( float( 1 ).sub( U ) ) );
			const nv = V.add( u.du.mul( u.ratio ).mul( lap.y ) ).add( uvv ).sub( F.add( K ).mul( V ) );

			dst.element( gIndex( gid ) ).assign( clamp( vec2( nu, nv ), vec2( 0 ), vec2( 1 ) ) );

		} )().compute( [ G / WG[ 0 ], G / WG[ 1 ], G / WG[ 2 ] ], WG );

	}

	const stepAB = makeStep( bufA, bufB ).setName( 'morpho.stepAB' );
	const stepBA = makeStep( bufB, bufA ).setName( 'morpho.stepBA' );

	const coordOf = ( idx ) => uvec3( idx.mod( G ), idx.div( G ).mod( G ), idx.div( G * G ) );

	// ── (re)seeding: u = 1, v = 0 plus clumps of v in hashed 3³ cells inside the sculpture ──
	const seedKernel = Fn( () => {

		const g = coordOf( instanceIndex ).toConst();
		const p = vec3( g ).add( 0.5 ).div( Gf ).mul( 2 ).sub( 1 );
		const cell = floor( vec3( g ).div( 3 ) );
		const h = hash( cell.x.add( cell.y.mul( 97 ) ).add( cell.z.mul( 9311 ) ).add( u.seedSalt.mul( 7919 ) ) );
		const inside = shapeMask( p, float( 0.2 ) );
		const st = bufA.element( instanceIndex );
		const cur = mix( st, vec2( 1, 0 ), u.seedClear ).toVar();
		If( h.lessThan( u.seedDensity.mul( 0.04 ).mul( inside ) ), () => {

			const j = hash( instanceIndex.add( u.seedSalt.mul( 131 ) ) );
			cur.assign( vec2( 0.5, j.mul( 0.2 ).add( 0.25 ) ) );

		} );
		st.assign( cur );

	} )().compute( N ).setName( 'morpho.seed' );

	// ── pointer pick: march the pointer ray through the live field (1 invocation, nearest lookups) ──
	const MAXPICK = Math.ceil( G * 1.8 * 2 );
	const pickKernel = Fn( () => {

		const ro = u.rayO, rd = u.rayD;
		const inv = vec3( 1 ).div( rd );
		const t0 = vec3( 0 ).sub( ro ).mul( inv ), t1 = vec3( Gf ).sub( ro ).mul( inv );
		const tn = max( max( min( t0.x, t1.x ), min( t0.y, t1.y ) ), max( min( t0.z, t1.z ), 0 ) ).toConst();
		const tf = min( min( max( t0.x, t1.x ), max( t0.y, t1.y ) ), max( t0.z, t1.z ) ).toConst();
		// default: closest point on the ray to the sculpture's centre
		const cpos = vec3( Gf.mul( 0.5 ) );
		const tc = max( dot( cpos.sub( ro ), rd ), 0 );
		const res = vec4( ro.add( rd.mul( tc ) ), 0 ).toVar();
		const t = tn.toVar();
		Loop( MAXPICK, () => {

			If( t.greaterThan( tf ), () => {

				Break();

			} );
			const q = ro.add( rd.mul( t ) );
			const gi = clamp( ivec3( floor( q ) ), ivec3( 0 ), ivec3( G - 1 ) );
			const p = vec3( gi ).add( 0.5 ).div( Gf ).mul( 2 ).sub( 1 );
			const v = bufA.element( gIndex( gi ) ).y.mul( shapeMask( p, u.shapeSoft ) );
			If( v.greaterThan( u.thr ), () => {

				res.assign( vec4( q.add( rd.mul( 1.5 ) ), 1 ) );
				Break();

			} );
			t.addAssign( 0.5 );

		} );
		pick.element( 0 ).assign( res );

	} )().compute( 1, [ 1 ] ).setName( 'morpho.pick' );

	// ── pointer seeding: inject activator around the picked point ──
	const brushKernel = Fn( () => {

		const g = coordOf( instanceIndex ).toConst();
		const P = pick.element( 0 ).xyz;
		const d = length( vec3( g ).add( 0.5 ).sub( P ) );
		const n = hash( instanceIndex.add( u.seedSalt.mul( 17 ) ) );
		const w = smoothstep( u.seedR, u.seedR.mul( 0.35 ), d.mul( n.mul( 0.5 ).add( 0.75 ) ) ).mul( u.seedAmt );
		If( w.greaterThan( 0.001 ), () => {

			const st = bufA.element( instanceIndex );
			st.assign( mix( st, vec2( 0.42, 0.33 ), w ) );

		} );

	} )().compute( N ).setName( 'morpho.brush' );

	// ── bake: masked v + growth glow + u → rgba16f 3D texture ──
	const bake = Fn( () => {

		const g = coordOf( instanceIndex ).toConst();
		const p = vec3( g ).add( 0.5 ).div( Gf ).mul( 2 ).sub( 1 );
		const st = bufA.element( instanceIndex ).toConst();
		const m = shapeMask( p, u.shapeSoft );
		const v = st.y.mul( m ).toConst();
		const a = aux.element( instanceIndex );
		const prev = a.toConst();
		const growth = max( v.sub( prev.x ), 0 );
		const fresh = min( prev.y.mul( u.fade ).add( growth.mul( u.glowGain ) ), 1.5 );
		a.assign( vec2( v, fresh ) );
		textureStore( volTex, g, vec4( v, fresh, st.x, 1 ) );

	} )().compute( N ).setName( 'morpho.bake' );

	// ── bricks: occupancy (max of masked v over brick + 1-voxel apron) ──
	const occKernel = Fn( () => {

		const b = uvec3( instanceIndex.mod( BG ), instanceIndex.div( BG ).mod( BG ), instanceIndex.div( BG * BG ) );
		const o = ivec3( b.mul( BRICK ) ).sub( 1 ).toConst();
		const m = float( 0 ).toVar();
		Loop( { start: 0, end: BRICK + 2, name: 'z' }, ( { z } ) => {

			Loop( { start: 0, end: BRICK + 2, name: 'y' }, ( { y } ) => {

				Loop( { start: 0, end: BRICK + 2, name: 'x' }, ( { x } ) => {

					const g = clamp( o.add( ivec3( x, y, z ) ), ivec3( 0 ), ivec3( G - 1 ) );
					m.assign( max( m, aux.element( gIndex( g ) ).x ) );

				} );

			} );

		} );
		occ.element( instanceIndex ).assign( select( m.greaterThan( u.thr.mul( 0.85 ) ), float( 1 ), float( 0 ) ) );

	} )().compute( NB ).setName( 'morpho.occ' );

	// ── bricks: Chebyshev distance (in bricks) to the nearest occupied brick, saturating at BRICK_R + 1 ──
	const distKernel = Fn( () => {

		const b = ivec3( uvec3( instanceIndex.mod( BG ), instanceIndex.div( BG ).mod( BG ), instanceIndex.div( BG * BG ) ) ).toConst();
		const d = float( BRICK_R + 1 ).toVar();
		Loop( { start: - BRICK_R, end: BRICK_R + 1, name: 'z' }, ( { z } ) => {

			Loop( { start: - BRICK_R, end: BRICK_R + 1, name: 'y' }, ( { y } ) => {

				Loop( { start: - BRICK_R, end: BRICK_R + 1, name: 'x' }, ( { x } ) => {

					const q = b.add( ivec3( x, y, z ) );
					const inb = q.x.greaterThanEqual( 0 ).and( q.y.greaterThanEqual( 0 ) ).and( q.z.greaterThanEqual( 0 ) )
						.and( q.x.lessThan( BG ) ).and( q.y.lessThan( BG ) ).and( q.z.lessThan( BG ) );
					If( inb, () => {

						const qc = clamp( q, ivec3( 0 ), ivec3( BG - 1 ) );
						If( occ.element( qc.x.add( qc.y.mul( BG ) ).add( qc.z.mul( BG * BG ) ) ).greaterThan( 0.5 ), () => {

							d.assign( min( d, float( max( max( abs( x ), abs( y ) ), abs( z ) ) ) ) );

						} );

					} );

				} );

			} );

		} );
		textureStore( brickTex, uvec3( b ), vec4( d, 0, 0, 1 ) );

	} )().compute( NB ).setName( 'morpho.dist' );

	return {
		G, N, BG, u, volTex, brickTex, pick, bufA, bufB, aux, occ,
		stepAB, stepBA, seedKernel, pickKernel, brushKernel, bake, occKernel, distKernel,
		kernels: [ stepAB, stepBA, seedKernel, pickKernel, brushKernel, bake, occKernel, distKernel ],
		dispose( renderer ) {

			for ( const k of this.kernels ) k.dispose();
			for ( const b of [ bufA, bufB, aux, occ, pick ] ) {

				try {

					renderer._attributes?.delete( b.value );

				} catch ( e ) { /* already gone */ }

			}
			volTex.dispose();
			brickTex.dispose();

		},
	};

}
