// Deep Time — GPU landscape evolution.
//
// Hydraulic erosion with the virtual-pipe shallow-water model of Mei, Decaudin & Hu (2007), "Fast Hydraulic Erosion
// Simulation and Visualization on GPU", plus thermal (talus) erosion and stream-power-style tectonic uplift.
// Everything lives in storage buffers on an N×N grid and is advanced by three compute dispatches per iteration:
//
//   1. flux     outflow flux f = (L, R, B, T) through 4 virtual pipes:  f' = max(0, f·damp + g·A·Δh/l),
//               then scaled by K = min(1, d·l² / (Σf·Δt)) so a cell can never ship more water than it holds.
//   2. water    Δd = Δt(Σf_in − Σf_out)/l²; velocity from the net flux through the cell (ΔW / (l·d̄));
//               transport capacity C = Kc·sin(α)·|v|·depthFactor; erode Ks(C−s) (soft regolith first, rock by
//               stratum hardness) or deposit Kd(s−C) as fresh soft sediment.
//   3. advect   semi-Lagrangian sediment transport s(x) ← s(x − vΔt) (bilinear); thermal erosion as a symmetric,
//               mass-conserving pairwise exchange over 8 neighbours wherever slope exceeds the talus angle;
//               evaporation d·(1−Ke); tectonic uplift; sea level as an open boundary (base level); rainfall.
//
// Units: horizontal cell size = 1, heights stored in cell units (world height / cellSize), so slopes, talus angles,
// CFL limits and all rate constants are resolution independent. World height = grid height × cellSize.
//
// After the iterations a "prep" pass bakes render data into storage textures: exact f32 height/water for vertex
// displacement, filterable normals + heightfield ray-marched soft sun shadows + AO, and flow/wetness/turbidity.

import * as THREE from 'three/webgpu';
import {
	Fn, If, Loop, Break, uniform, instancedArray, instanceIndex, float, int, uint, vec2, vec3, vec4, uvec2,
	clamp, min, max, abs, sqrt, exp, mix, smoothstep, floor, length, dot, sin, cos, normalize, textureStore,
	mx_noise_float, select,
} from 'three/tsl';

export const LOW = 128; // CPU-side height copy (for pointer picking)

export class ErosionSim {

	constructor( renderer, { N, size, tier } ) {

		this.renderer = renderer;
		this.N = N;
		this.size = size;
		this.cellW = size / N; // world size of one cell
		this.cells = N * N;
		this.lowN = Math.min( LOW, N );
		this.tier = tier;
		this.ticks = 0;

		const C = this.cells;
		const grid = N / size; // world → grid height scale

		// ── state ─────────────────────────────────────────────
		this.stateA = instancedArray( C, 'vec4' ); // b (terrain), d (water), s (suspended sediment), r (soft regolith thickness)
		this.stateB = instancedArray( C, 'vec4' );
		this.flux = instancedArray( C, 'vec4' ); // outflow L(−x) R(+x) B(−y) T(+y)
		this.vel = instancedArray( C, 'vec4' ); // u, v, wetness (EMA), discharge (EMA)
		this.tect = instancedArray( C, 'vec4' ); // uplift field, cumulative uplift (strata offset), –, –
		this.low = instancedArray( this.lowN * this.lowN, 'float' );
		this.buffers = [ this.stateA, this.stateB, this.flux, this.vel, this.tect, this.low ];

		// ── render textures ───────────────────────────────────
		const mk = ( type, filter ) => {

			const t = new THREE.StorageTexture( N, N );
			t.type = type;
			t.format = THREE.RGBAFormat;
			t.magFilter = t.minFilter = filter;
			t.generateMipmaps = false;
			t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
			return t;

		};

		this.texGeo = mk( THREE.FloatType, THREE.NearestFilter ); // world b, world d, world soft, world strata coord (textureLoad only)
		this.texLit = mk( THREE.HalfFloatType, THREE.LinearFilter ); // normal.x, normal.z, sun shadow, ambient occlusion
		this.texFlow = mk( THREE.HalfFloatType, THREE.LinearFilter ); // speed, wetness, turbidity, discharge
		this.textures = [ this.texGeo, this.texLit, this.texFlow ];

		// ── parameters (grid units, per iteration) ────────────
		this.u = {
			seed: uniform( 0 ),
			gridScale: uniform( grid ),
			cellW: uniform( this.cellW ),
			sea: uniform( 0 ),
			hMax: uniform( 1.4 * grid ), // for orographic rain
			rain: uniform( 0.0010 ),
			oro: uniform( 1.6 ),
			storm: uniform( new THREE.Vector3( N * 0.5, N * 0.5, 0 ) ), // grid x, grid y, rate
			stormR: uniform( N * 0.07 ),
			push: uniform( new THREE.Vector3( 0, 0, 0 ) ), // grid x, grid y, amount per iteration
			pushR: uniform( N * 0.06 ),
			gain: uniform( 0.45 ), // g·A·Δt/l
			damp: uniform( 0.995 ),
			Kc: uniform( 0.9 ),
			Ks: uniform( 0.035 ),
			Kd: uniform( 0.06 ),
			Ke: uniform( 0.0016 ),
			dRef: uniform( 0.6 ),
			vmax: uniform( 1.2 ),
			minTilt: uniform( 0.035 ),
			softRef: uniform( 1.5 ),
			talus: uniform( Math.tan( 48 * Math.PI / 180 ) ),
			Kt: uniform( 0.12 ),
			uplift: uniform( 0.0035 ),
			strataFreq: uniform( 0.9 / ( 0.06 * grid ) ), // one hard/soft bed pair every ~0.06 world units
			sunDir: uniform( new THREE.Vector3( - 0.8, 0.16, - 0.5 ).normalize() ),
		};

		this._build();

	}

	// ───────────────────────── kernels ─────────────────────────
	_build() {

		const { N, u } = this;
		const NI = int( N ), NM = int( N - 1 );
		const sA = this.stateA, sB = this.stateB, flux = this.flux, vel = this.vel, tect = this.tect;

		const cx = ( v ) => clamp( v, int( 0 ), NM );
		const at = ( x, y ) => cx( y ).mul( NI ).add( cx( x ) );
		const coords = () => {

			const x = int( instanceIndex.mod( uint( N ) ) ).toVar( 'gx' );
			const y = int( instanceIndex.div( uint( N ) ) ).toVar( 'gy' );
			return [ x, y ];

		};

		// rainfall: orographic base rate (wetter on the mountains) + a storm cell following the pointer
		const rainAt = ( x, y, b ) => {

			const p = vec2( float( x ), float( y ) );
			const dd = p.sub( vec2( u.storm.x, u.storm.y ) );
			const storm = u.storm.z.mul( exp( dot( dd, dd ).negate().div( u.stormR.mul( u.stormR ) ) ) );
			const oro = float( 0.35 ).add( u.oro.mul( smoothstep( u.sea, u.hMax, b ) ) );
			return select( b.greaterThan( u.sea ), u.rain.mul( oro ).add( storm ), float( 0 ) );

		};

		// ── initial terrain: domain-warped ridged multifractal on a continent with a curved mountain spine ──
		const fbm = ( p, oct ) => {

			let s = float( 0 );
			let q = p;
			let a = 0.5;
			for ( let i = 0; i < oct; i ++ ) {

				s = s.add( mx_noise_float( q ).mul( a ) ).toVar();
				q = q.mul( 2.03 ).add( vec2( 1.7 + i, 9.2 - i ) ).toVar();
				a *= 0.5;

			}

			return s;

		};

		const ridged = ( p, oct ) => {

			let s = float( 0 );
			let w = float( 1 );
			let q = p;
			let a = 0.55;
			for ( let i = 0; i < oct; i ++ ) {

				let n = float( 1 ).sub( abs( mx_noise_float( q ) ) );
				n = n.mul( n ).mul( w ).toVar();
				w = clamp( n.mul( 1.8 ), 0, 1 ).toVar();
				s = s.add( n.mul( a ) ).toVar();
				q = q.mul( 2.07 ).add( vec2( 3.1 * i + 0.3, 1.3 - 2.1 * i ) ).toVar();
				a *= 0.5;

			}

			return s;

		};

		this.initNode = Fn( () => {

			const [ x, y ] = coords();
			const seed = u.seed;
			const so = vec2( seed.mul( 17.13 ), seed.mul( - 9.71 ) );
			const p = vec2( float( x ), float( y ) ).add( 0.5 ).div( N ).mul( 2 ).sub( 1 ).toVar(); // [-1, 1]

			// domain warp
			const w1 = vec2( fbm( p.mul( 1.2 ).add( so ), 4 ), fbm( p.mul( 1.2 ).add( so ).add( vec2( 5.2, 1.3 ) ), 4 ) ).toVar();
			const q = p.add( w1.mul( 0.38 ) ).toVar();

			// continent mask (land in the middle, sea floor at the edges)
			const coast = length( q.mul( vec2( 0.95, 1.08 ) ) ).add( fbm( q.mul( 2.6 ).add( so.mul( 1.7 ) ), 3 ).mul( 0.18 ) );
			const land = smoothstep( 0.98, 0.52, coast ).toVar();

			// curved mountain spine
			const th = seed.mul( 2.399 ).add( 0.55 );
			const dir = vec2( cos( th ), sin( th ) );
			const nrm = vec2( sin( th ).negate(), cos( th ) );
			const along = dot( q, dir ).toVar();
			const across = dot( q, nrm ).sub( sin( along.mul( 2.3 ).add( seed ) ).mul( 0.22 ) );
			const spine = exp( across.mul( across ).negate().div( 0.17 ) ).mul( smoothstep( 1.05, 0.35, abs( along ) ) ).toVar();

			const r = ridged( q.mul( 2.4 ).add( so.mul( 0.7 ) ), this.tier === 0 ? 6 : 8 );
			const hills = fbm( q.mul( 4.0 ).add( so.mul( 1.3 ) ), 4 );
			const base = mix( float( - 0.42 ), float( 0.06 ), land ).add( hills.mul( 0.06 ).mul( land ) );
			const mountains = r.mul( float( 0.12 ).add( spine.mul( 1.55 ) ) ).mul( land ).mul( land );
			const h = base.add( mountains ).add( spine.mul( land ).mul( 0.18 ) ); // world units

			const i = instanceIndex;
			sA.element( i ).assign( vec4( h.mul( u.gridScale ), 0, 0, 0 ) );
			sB.element( i ).assign( vec4( h.mul( u.gridScale ), 0, 0, 0 ) );
			flux.element( i ).assign( vec4( 0 ) );
			vel.element( i ).assign( vec4( 0 ) );
			const upl = land.mul( spine.mul( 0.85 ).add( 0.15 ) ).mul( hills.mul( 0.6 ).add( 0.8 ) );
			tect.element( i ).assign( vec4( max( upl, 0 ), 0, 0, 0 ) );

		} )().compute( this.cells );

		// ── pass 1: outflow flux ──
		this.fluxNode = Fn( () => {

			const [ x, y ] = coords();
			const i = instanceIndex;
			const s = sA.element( i );
			const H = s.x.add( s.y ).toVar();
			const hn = ( dx, dy ) => {

				const n = sA.element( at( x.add( dx ), y.add( dy ) ) );
				return n.x.add( n.y );

			};

			const f = flux.element( i ).toVar();
			const fL = max( f.x.mul( u.damp ).add( u.gain.mul( H.sub( hn( - 1, 0 ) ) ) ), 0 );
			const fR = max( f.y.mul( u.damp ).add( u.gain.mul( H.sub( hn( 1, 0 ) ) ) ), 0 );
			const fB = max( f.z.mul( u.damp ).add( u.gain.mul( H.sub( hn( 0, - 1 ) ) ) ), 0 );
			const fT = max( f.w.mul( u.damp ).add( u.gain.mul( H.sub( hn( 0, 1 ) ) ) ), 0 );
			const nf = vec4( fL, fR, fB, fT ).toVar();
			const sum = nf.x.add( nf.y ).add( nf.z ).add( nf.w );
			const K = min( float( 1 ), s.y.div( sum.add( 1e-6 ) ) );
			flux.element( i ).assign( nf.mul( K ) );

		} )().compute( this.cells );

		// ── pass 2: water volume, velocity, erosion / deposition ──
		this.waterNode = Fn( () => {

			const [ x, y ] = coords();
			const i = instanceIndex;
			const s = sA.element( i ).toVar();
			const f = flux.element( i ).toVar();
			const t = tect.element( i );

			const mL = select( x.greaterThan( 0 ), float( 1 ), float( 0 ) );
			const mR = select( x.lessThan( NM ), float( 1 ), float( 0 ) );
			const mB = select( y.greaterThan( 0 ), float( 1 ), float( 0 ) );
			const mT = select( y.lessThan( NM ), float( 1 ), float( 0 ) );
			const inL = flux.element( at( x.sub( 1 ), y ) ).y.mul( mL ).toVar(); // left neighbour's R pipe
			const inR = flux.element( at( x.add( 1 ), y ) ).x.mul( mR ).toVar();
			const inB = flux.element( at( x, y.sub( 1 ) ) ).w.mul( mB ).toVar();
			const inT = flux.element( at( x, y.add( 1 ) ) ).z.mul( mT ).toVar();

			const fin = inL.add( inR ).add( inB ).add( inT );
			const fout = f.x.add( f.y ).add( f.z ).add( f.w );
			const d1 = s.y;
			const d2 = max( d1.add( fin ).sub( fout ), 0 ).toVar();
			const dbar = max( d1.add( d2 ).mul( 0.5 ), 0.02 );
			const wx = inL.sub( f.x ).add( f.y ).sub( inR ).mul( 0.5 );
			const wy = inB.sub( f.z ).add( f.w ).sub( inT ).mul( 0.5 );
			const v = vec2( wx, wy ).div( dbar ).toVar();
			const sp = length( v ).toVar();
			If( sp.greaterThan( u.vmax ), () => {

				v.mulAssign( u.vmax.div( sp ) );
				sp.assign( u.vmax );

			} );

			// local slope from the terrain
			const bL = sA.element( at( x.sub( 1 ), y ) ).x;
			const bR = sA.element( at( x.add( 1 ), y ) ).x;
			const bB = sA.element( at( x, y.sub( 1 ) ) ).x;
			const bT = sA.element( at( x, y.add( 1 ) ) ).x;
			const g = vec2( bR.sub( bL ), bT.sub( bB ) ).mul( 0.5 ).toVar();
			const g2 = dot( g, g );
			const sinA = sqrt( g2.div( g2.add( 1 ) ) );
			const bmin = min( min( bL, bR ), min( bB, bT ) );

			const b = s.x.toVar();
			const sed = s.z.toVar();
			const soft = s.w.toVar();
			const depthF = clamp( d2.div( u.dRef ), 0, 1 );
			const cap = u.Kc.mul( max( sinA, u.minTilt ) ).mul( sp ).mul( depthF ).toVar();

			// layered bedrock: alternating hard / soft strata (offset by cumulative uplift so beds rise with the rock)
			const strata = b.sub( t.y );
			const band = sin( strata.mul( u.strataFreq ).add( sin( float( x ).mul( 0.013 ).add( float( y ).mul( 0.009 ) ) ).mul( 2.0 ) ) );
			const rockE = mix( float( 1.0 ), float( 0.3 ), smoothstep( - 0.2, 0.5, band ) );
			const erod = mix( rockE, float( 2.2 ), clamp( soft.div( u.softRef ), 0, 1 ) );

			If( cap.greaterThan( sed ), () => {

				const amt = min( u.Ks.mul( cap.sub( sed ) ).mul( erod ), max( b.sub( bmin ), 0 ).mul( 0.6 ).add( 0.004 ) );
				b.subAssign( amt );
				sed.addAssign( amt );
				soft.assign( max( soft.sub( amt ), 0 ) );

			} ).Else( () => {

				const amt = u.Kd.mul( sed.sub( cap ) );
				b.addAssign( amt );
				sed.subAssign( amt );
				soft.addAssign( amt );

			} );

			const vo = vel.element( i );
			const wet = mix( vo.z, clamp( d2.mul( 3 ), 0, 1 ), 0.02 );
			const dis = mix( vo.w, sp.mul( d2 ), 0.03 );
			sB.element( i ).assign( vec4( b, d2, sed, soft ) );
			vel.element( i ).assign( vec4( v, wet, dis ) );

		} )().compute( this.cells );

		// ── pass 3: sediment advection, thermal erosion, evaporation, uplift, sea level, rain ──
		this.advectNode = Fn( () => {

			const [ x, y ] = coords();
			const i = instanceIndex;
			const s = sB.element( i ).toVar();
			const v = vel.element( i ).xy;

			// semi-Lagrangian back-trace, bilinear
			const p = clamp( vec2( float( x ), float( y ) ).sub( v ), vec2( 0 ), vec2( N - 1 ) ).toVar();
			const p0 = floor( p );
			const fr = p.sub( p0 ).toVar();
			const x0 = int( p0.x ).toVar(), y0 = int( p0.y ).toVar();
			const s00 = sB.element( at( x0, y0 ) ).z;
			const s10 = sB.element( at( x0.add( 1 ), y0 ) ).z;
			const s01 = sB.element( at( x0, y0.add( 1 ) ) ).z;
			const s11 = sB.element( at( x0.add( 1 ), y0.add( 1 ) ) ).z;
			const sed = mix( mix( s00, s10, fr.x ), mix( s01, s11, fr.x ), fr.y ).toVar();

			// thermal erosion: symmetric exchange with 8 neighbours where the slope exceeds the talus angle
			const b = s.x;
			const softSelf = clamp( s.w.div( u.softRef ), 0, 1 );
			const rateSelf = mix( float( 0.12 ), float( 1.0 ), softSelf ).toVar();
			const delta = float( 0 ).toVar();
			const offs = [ [ 1, 0, 1 ], [ - 1, 0, 1 ], [ 0, 1, 1 ], [ 0, - 1, 1 ], [ 1, 1, Math.SQRT2 ], [ - 1, 1, Math.SQRT2 ], [ 1, - 1, Math.SQRT2 ], [ - 1, - 1, Math.SQRT2 ] ];
			for ( const [ dx, dy, dist ] of offs ) {

				const n = sB.element( at( x.add( dx ), y.add( dy ) ) );
				const diff = b.sub( n.x );
				const T = u.talus.mul( dist );
				const rateN = mix( float( 0.12 ), float( 1.0 ), clamp( n.w.div( u.softRef ), 0, 1 ) );
				const out = max( diff.sub( T ), 0 ).mul( rateSelf );
				const inn = max( diff.negate().sub( T ), 0 ).mul( rateN );
				delta.addAssign( inn.sub( out ).mul( 1 / dist ) );

			}

			delta.mulAssign( u.Kt.mul( 0.5 ) );

			const t = tect.element( i ).toVar();
			const pp = vec2( float( x ), float( y ) ).sub( vec2( u.push.x, u.push.y ) );
			const up = u.uplift.mul( t.x ).add( u.push.z.mul( exp( dot( pp, pp ).negate().div( u.pushR.mul( u.pushR ) ) ) ) );

			const bn = b.add( delta ).add( up ).toVar();
			const soft = max( s.w.add( delta ), 0 ).toVar();
			const d = s.y.mul( float( 1 ).sub( u.Ke ) ).toVar();

			// sea = open boundary at base level: water surface pinned to sea level
			If( bn.lessThan( u.sea ), () => {

				d.assign( u.sea.sub( bn ) );

			} );

			// water gone → drop what it carried
			If( d.lessThan( 1e-4 ), () => {

				bn.addAssign( sed );
				soft.addAssign( sed );
				sed.assign( 0 );

			} );

			d.addAssign( rainAt( x, y, bn ) );
			sA.element( i ).assign( vec4( bn, d, max( sed, 0 ), soft ) );
			tect.element( i ).assign( vec4( t.x, t.y.add( up ), t.z, t.w ) );

		} )().compute( this.cells );

		// ── render prep: bake textures ──
		const shadowSteps = this.tier === 2 ? 56 : this.tier === 1 ? 44 : 32;
		const reach = N * 0.7;
		const stepB = ( reach - shadowSteps ) / ( shadowSteps * shadowSteps * 0.5 );
		const aoR1 = Math.max( 2, Math.round( N / 96 ) ), aoR2 = Math.max( 6, Math.round( N / 24 ) );

		this.prepNode = Fn( () => {

			const [ x, y ] = coords();
			const i = instanceIndex;
			const s = sA.element( i ).toVar();
			const vv = vel.element( i ).toVar();
			const t = tect.element( i );
			const cw = u.cellW;

			const bL = sA.element( at( x.sub( 1 ), y ) ).x;
			const bR = sA.element( at( x.add( 1 ), y ) ).x;
			const bB = sA.element( at( x, y.sub( 1 ) ) ).x;
			const bT = sA.element( at( x, y.add( 1 ) ) ).x;
			const n = normalize( vec3( bL.sub( bR ).mul( 0.5 ), 1, bB.sub( bT ).mul( 0.5 ) ) ).toVar();

			// heightfield ray-marched soft shadow toward the sun (grid space)
			const sd = u.sunDir;
			const hz = length( vec2( sd.x, sd.z ) );
			const dir = vec2( sd.x, sd.z ).div( hz ).toVar();
			const tanE = sd.y.div( hz ).toVar();
			const h0 = s.x.add( s.y ).add( 0.6 ).toVar();
			const res = float( 1 ).toVar();
			const tt = float( 1 ).toVar();
			Loop( shadowSteps, ( { i: k } ) => {

				tt.addAssign( float( k ).mul( stepB ).add( 1 ) );
				const q = vec2( float( x ), float( y ) ).add( dir.mul( tt ) );
				If( q.x.lessThan( 0 ).or( q.y.lessThan( 0 ) ).or( q.x.greaterThan( N - 1 ) ).or( q.y.greaterThan( N - 1 ) ), () => {

					Break();

				} );
				const o = sA.element( at( int( q.x ), int( q.y ) ) );
				const dh = h0.add( tt.mul( tanE ) ).sub( o.x.add( o.y ) );
				res.assign( min( res, dh.mul( 10 ).div( tt ) ) );

			} );
			const shadow = smoothstep( 0, 1, clamp( res, 0, 1 ) );

			// two-scale height-difference ambient occlusion
			const ring = ( r ) => {

				let acc = float( 0 );
				const dirs = [ [ 1, 0 ], [ - 1, 0 ], [ 0, 1 ], [ 0, - 1 ], [ 0.7, 0.7 ], [ - 0.7, 0.7 ], [ 0.7, - 0.7 ], [ - 0.7, - 0.7 ] ];
				for ( const [ dx, dy ] of dirs ) acc = acc.add( sA.element( at( x.add( Math.round( dx * r ) ), y.add( Math.round( dy * r ) ) ) ).x );
				return acc.mul( 1 / 8 ).sub( s.x ).div( r );

			};

			const occ = clamp( ring( aoR1 ), 0, 2 ).mul( 0.55 ).add( clamp( ring( aoR2 ), 0, 2 ).mul( 0.45 ) );
			const ao = clamp( float( 1 ).sub( occ.mul( 1.1 ) ), 0.15, 1 );

			const uvI = uvec2( uint( x ), uint( y ) );
			textureStore( this.texGeo, uvI, vec4( s.x.mul( cw ), s.y.mul( cw ), s.w.mul( cw ), s.x.sub( t.y ).mul( cw ) ) ).toWriteOnly();
			textureStore( this.texLit, uvI, vec4( n.x, n.z, shadow, ao ) ).toWriteOnly();
			const sp = length( vv.xy );
			textureStore( this.texFlow, uvI, vec4( sp, vv.z, s.z.div( max( s.y, 0.05 ) ), vv.w ) ).toWriteOnly();

		} )().compute( this.cells );

		// ── downsampled surface height for CPU picking ──
		const L = this.lowN, stride = N / L;
		this.lowNode = Fn( () => {

			const lx = int( instanceIndex.mod( uint( L ) ) );
			const ly = int( instanceIndex.div( uint( L ) ) );
			const s = sA.element( at( lx.mul( stride ).add( stride >> 1 ), ly.mul( stride ).add( stride >> 1 ) ) );
			this.low.element( instanceIndex ).assign( max( s.x, u.sea ).mul( u.cellW ) );

		} )().compute( L * L );

		this.iteration = [ this.fluxNode, this.waterNode, this.advectNode ];
		this.computes = [ this.initNode, this.fluxNode, this.waterNode, this.advectNode, this.prepNode, this.lowNode ];

	}

	// ───────────────────────── driving ─────────────────────────
	async init( seed ) {

		this.u.seed.value = seed;
		this.ticks = 0;
		await this.renderer.computeAsync( this.initNode );

	}

	/** n iterations in one compute pass (plus the render bake when `prep`). */
	step( n, prep = true ) {

		if ( n <= 0 && ! prep ) return;
		const list = [];
		for ( let k = 0; k < n; k ++ ) list.push( this.fluxNode, this.waterNode, this.advectNode );
		if ( prep ) list.push( this.prepNode );
		this.renderer.compute( list );
		this.ticks += n;

	}

	async prerun( total, batch ) {

		for ( let done = 0; done < total; done += batch ) {

			const n = Math.min( batch, total - done );
			const list = [];
			for ( let k = 0; k < n; k ++ ) list.push( this.fluxNode, this.waterNode, this.advectNode );
			await this.renderer.computeAsync( list );
			this.ticks += n;

		}

		await this.renderer.computeAsync( this.prepNode );

	}

	async readLow() {

		this.renderer.compute( this.lowNode );
		const buf = await this.renderer.getArrayBufferAsync( this.low.value );
		return new Float32Array( buf );

	}

	dispose() {

		for ( const c of this.computes ) c.dispose();
		for ( const t of this.textures ) t.dispose();
		const attrs = this.renderer._attributes;
		for ( const b of this.buffers ) {

			try {

				attrs?.delete( b.value );

			} catch ( e ) { /* already gone */ }

		}

	}

}
