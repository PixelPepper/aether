// Genesis — 3D particle life on the GPU with an O(N) spatial hash grid rebuilt every frame.
//
// Per step (all compute, no CPU readback):
//   1. clear      cellCount[c] = 0                                   (atomicStore, every padded cell)
//   2. count      cell_i = cell(p_i); rank_i = atomicAdd(cellCount[cell_i], 1)
//   3. scan       cellStart = exclusive prefix sum of cellCount     (multi-level workgroup scan, see scan.js)
//   4. scatter    k = cellStart[cell_i] + rank_i; sorted[k] = state_i   (atomic-free, deterministic counting sort)
//   5. force      for each sorted particle k: walk the 3×3 neighbouring cell *rows* — because x is the fastest grid axis,
//                 cells (x-1..x+1, y, z) are contiguous in sorted order, so 27 cells collapse into 9 contiguous ranges
//                 [cellStart[row+x0], cellStart[row+x1+1]). Integrate and write back into the primary buffers at k.
// The primary buffers therefore end every frame spatially sorted, which keeps the next frame's reads cache-coherent.

import { StorageBufferAttribute } from 'three/webgpu';
import {
	Fn, If, Loop, uniform, storage, instancedArray, instanceIndex, atomicAdd, atomicStore,
	float, int, uint, vec3, vec4, ivec3, dot, max, min, clamp, floor, fract, sqrt, length, normalize, cross, select, exp, hash, TWO_PI,
} from 'three/tsl';
import { GPUScan, SCAN_BLOCK, roundUp } from './scan.js';
import { MAX_SPECIES } from './rules.js';

export class ParticleLife {

	constructor( renderer, { count, bound, rMax, rMaxMin } ) {

		this.renderer = renderer;
		this.count = count;
		this.attributes = [];
		this.computes = [];

		// ── uniforms ─────────────────────────────────────────────
		const u = this.u = {
			bound: uniform( bound ),
			rMax: uniform( rMax ),
			beta: uniform( 0.3 ),
			force: uniform( 10.0 ),
			friction: uniform( 0.7 ), // per-step velocity retention, derived from half-life
			dt: uniform( 0.02 ),
			wall: uniform( 30.0 ),
			centerPull: uniform( 0.05 ),
			vmax: uniform( 20.0 ),
			species: uniform( 6, 'int' ),
			seed: uniform( 0 ),
			// grid
			dim: uniform( 8, 'int' ),
			dimF: uniform( 8 ),
			origin: uniform( vec3( 0 ) ),
			invCell: uniform( 1 ),
			// pointer
			time: uniform( 0 ),
			ptr: uniform( vec3( 0 ) ),
			ptrVel: uniform( vec3( 0 ) ),
			rayO: uniform( vec3( 0 ) ),
			rayD: uniform( vec3( 0, 0, - 1 ) ),
			viewDir: uniform( vec3( 0, 0, - 1 ) ),
			attract: uniform( 0 ),
			repel: uniform( 0 ),
			current: uniform( 0 ),
			ptrRadius: uniform( 3 ),
			shock: uniform( vec4( 0, 0, 0, - 100 ) ),
			shockAmp: uniform( 0 ),
		};

		// ── grid sizing: allocate for the smallest interaction radius the GUI allows ──
		this.gridHalf = bound * 1.2;
		this.maxDim = Math.max( 4, Math.min( 96, Math.floor( 2 * this.gridHalf / rMaxMin ) ) );
		this.maxCells = this.maxDim ** 3;
		this.cellsPadded = roundUp( this.maxCells + 1, SCAN_BLOCK ); // +1 so cellStart[last + 1] == N is always readable

		// ── buffers ──────────────────────────────────────────────
		const N = count;
		this.pos = instancedArray( N, 'vec4' ).setName( 'gPos' ); // xyz, w = species + seed fraction
		this.vel = instancedArray( N, 'vec4' ).setName( 'gVel' ); // xyz, w = local density
		this.sPos = instancedArray( N, 'vec4' ).setName( 'gSortedPos' );
		this.sVel = instancedArray( N, 'vec4' ).setName( 'gSortedVel' );
		this.cellOf = instancedArray( N, 'uint' ).setName( 'gCell' );
		this.rank = instancedArray( N, 'uint' ).setName( 'gRank' );
		for ( const b of [ this.pos, this.vel, this.sPos, this.sVel, this.cellOf, this.rank ] ) this.attributes.push( b.value );

		const countAttr = new StorageBufferAttribute( new Uint32Array( this.cellsPadded ), 1, Uint32Array );
		const startAttr = new StorageBufferAttribute( new Uint32Array( this.cellsPadded ), 1, Uint32Array );
		this.matrixAttr = new StorageBufferAttribute( new Float32Array( MAX_SPECIES * MAX_SPECIES ), 1, Float32Array );
		this.attributes.push( countAttr, startAttr, this.matrixAttr );
		this.countAttr = countAttr;
		this.startAttr = startAttr;

		const countAtomic = storage( countAttr, 'uint', this.cellsPadded ).toAtomic();
		const countRead = storage( countAttr, 'uint', this.cellsPadded ).toReadOnly();
		const startRW = storage( startAttr, 'uint', this.cellsPadded );
		const startRead = storage( startAttr, 'uint', this.cellsPadded ).toReadOnly();
		const matrix = storage( this.matrixAttr, 'float', MAX_SPECIES * MAX_SPECIES ).toReadOnly();

		// ── helpers ──────────────────────────────────────────────
		const cellCoord = ( p ) => ivec3( clamp( floor( p.sub( u.origin ).mul( u.invCell ) ), vec3( 0 ), vec3( u.dimF.sub( 1 ) ) ) );
		const cellLinear = ( c ) => c.x.add( u.dim.mul( c.y.add( u.dim.mul( c.z ) ) ) );

		// ── init: primordial soup, uniform in the ball ──────────────
		this.initKernel = Fn( () => {

			const i = instanceIndex;
			const s = u.seed.mul( 7919.0 ).add( float( i ).mul( 4.0 ) );
			const a = hash( s ), b = hash( s.add( 1 ) ), c = hash( s.add( 2 ) ), d = hash( s.add( 3 ) );
			const z = b.mul( 2 ).sub( 1 );
			const rxy = sqrt( max( float( 1 ).sub( z.mul( z ) ), 0 ) );
			const th = a.mul( TWO_PI );
			const dir = vec3( rxy.mul( th.cos() ), z, rxy.mul( th.sin() ) );
			const rad = c.pow( 1 / 3 ).mul( u.bound ).mul( 0.92 );
			const sd = d.mul( 0.999 );
			const sp = floor( sd.mul( float( u.species ) ) );
			this.pos.element( i ).assign( vec4( dir.mul( rad ), sp.add( sd ) ) );
			this.vel.element( i ).assign( vec4( 0 ) );

		} )().compute( N ).setName( 'genesis.init' );

		// reassign species after the species count changes (keeps positions, re-derives species from the seed fraction)
		this.speciesKernel = Fn( () => {

			const P = this.pos.element( instanceIndex );
			const sd = fract( P.w );
			P.w.assign( floor( sd.mul( float( u.species ) ) ).add( sd ) );

		} )().compute( N ).setName( 'genesis.species' );

		// ── 1. clear ─────────────────────────────────────────────
		this.clearKernel = Fn( () => {

			atomicStore( countAtomic.element( instanceIndex ), uint( 0 ) );

		} )().compute( this.cellsPadded, [ 256 ] ).setName( 'genesis.clear' );

		// ── 2. count ─────────────────────────────────────────────
		this.countKernel = Fn( () => {

			const i = instanceIndex;
			const c = cellLinear( cellCoord( this.pos.element( i ).xyz ) ).toUint().toVar( 'c' );
			this.cellOf.element( i ).assign( c );
			const r = atomicAdd( countAtomic.element( c ), uint( 1 ) ).toVar( 'r' );
			this.rank.element( i ).assign( r );

		} )().compute( N, [ 256 ] ).setName( 'genesis.count' );

		// ── 3. scan ──────────────────────────────────────────────
		this.scan = new GPUScan( countRead, startRW, this.cellsPadded );
		for ( const a of this.scan.attributes ) this.attributes.push( a );

		// ── 4. scatter ───────────────────────────────────────────
		this.scatterKernel = Fn( () => {

			const i = instanceIndex;
			const k = startRead.element( this.cellOf.element( i ) ).add( this.rank.element( i ) ).toVar( 'k' );
			this.sPos.element( k ).assign( this.pos.element( i ) );
			this.sVel.element( k ).assign( this.vel.element( i ) );

		} )().compute( N, [ 256 ] ).setName( 'genesis.scatter' );

		// ── 5. forces + integration ──────────────────────────────
		this.forceKernel = Fn( () => {

			const k = instanceIndex;
			const P = this.sPos.element( k ).toVar( 'P' );
			const V = this.sVel.element( k ).toVar( 'V' );
			const p = P.xyz.toVar( 'p' );
			const v = V.xyz.toVar( 'v' );
			const row0 = P.w.toInt().mul( MAX_SPECIES ).toVar( 'row0' );

			const cc = cellCoord( p ).toVar( 'cc' );
			const lo = max( cc.sub( 1 ), ivec3( 0 ) ).toVar( 'lo' );
			const hi = min( cc.add( 1 ), ivec3( u.dim.sub( 1 ) ) ).toVar( 'hi' );

			const acc = vec3( 0 ).toVar( 'acc' );
			const dens = float( 0 ).toVar( 'dens' );
			const r2max = u.rMax.mul( u.rMax ).toVar( 'r2max' );
			const invR = float( 1 ).div( u.rMax ).toVar( 'invR' );
			const invRamp = float( 1 ).div( float( 1 ).sub( u.beta ) ).toVar( 'invRamp' );

			Loop( { start: lo.z, end: hi.z, type: 'int', condition: '<=', name: 'cz' }, ( { cz } ) => {

				Loop( { start: lo.y, end: hi.y, type: 'int', condition: '<=', name: 'cy' }, ( { cy } ) => {

					const row = cz.mul( u.dim ).add( cy ).mul( u.dim ).toVar( 'row' );
					const j0 = startRead.element( row.add( lo.x ) ).toVar( 'j0' );
					const j1 = startRead.element( row.add( hi.x ).add( 1 ) ).toVar( 'j1' );

					Loop( { start: j0, end: j1, type: 'uint', condition: '<', name: 'j' }, ( { j } ) => {

						const Q = this.sPos.element( j ).toVar( 'Q' );
						const d = Q.xyz.sub( p ).toVar( 'd' );
						const r2 = dot( d, d ).toVar( 'r2' );

						If( r2.lessThan( r2max ).and( r2.greaterThan( 1e-10 ) ), () => {

							const r = sqrt( r2 );
							const x = r.mul( invR ).toVar( 'x' );
							const a = matrix.element( row0.add( Q.w.toInt() ) );
							const ramp = a.mul( float( 1 ).sub( x.mul( 2 ).sub( 1 ).sub( u.beta ).abs().mul( invRamp ) ) );
							const f = select( x.lessThan( u.beta ), x.div( u.beta ).sub( 1 ), ramp );
							acc.addAssign( d.mul( f.div( r ) ) );
							const w = float( 1 ).sub( x );
							dens.addAssign( w.mul( w ) );

						} );

					} );

				} );

			} );

			acc.mulAssign( u.rMax.mul( u.force ) );

			// soft containing sphere + a faint central well
			const rr = length( p ).toVar( 'rr' );
			acc.subAssign( p.mul( max( rr.sub( u.bound ), 0 ).mul( u.wall ).div( max( rr, 1e-4 ) ).add( u.centerPull ) ) );

			// ── pointer ──
			const toP = u.ptr.sub( p ).toVar( 'toP' );
			const dP = length( toP ).add( 1e-4 ).toVar( 'dP' );
			const nP = toP.div( dP ).toVar( 'nP' );
			const R = u.ptrRadius;

			// left: gravity well with a vortex around the view axis
			const pull = dP.div( dP.add( R.mul( 0.35 ) ) ).mul( float( 1 ).div( float( 1 ).add( dP.div( R.mul( 3 ) ) ) ) );
			const swirl = cross( u.viewDir, nP ).mul( exp( dP.div( R ).negate() ) ).mul( 0.9 );
			acc.addAssign( nP.mul( pull ).add( swirl ).mul( u.attract ) );

			// right: repulsor bubble
			const q = dP.div( R.mul( 0.8 ) );
			acc.subAssign( nP.mul( exp( q.mul( q ).negate() ).mul( u.repel ) ) );

			// shockwave ring expanding from the click point
			const toS = p.sub( u.shock.xyz ).toVar( 'toS' );
			const dS = length( toS ).add( 1e-4 );
			const age = u.time.sub( u.shock.w );
			const front = age.mul( u.bound ).mul( 1.4 );
			const band = dS.sub( front ).div( u.rMax.mul( 1.5 ) );
			const ring = exp( band.mul( band ).negate() ).mul( max( float( 1 ).sub( age.div( 1.3 ) ), 0 ) );
			acc.addAssign( toS.div( dS ).mul( ring.mul( u.shockAmp ) ) );

			// idle: the cursor's motion drags a column of fluid along the view ray
			const w = p.sub( u.rayO );
			const perp = w.sub( u.rayD.mul( dot( w, u.rayD ) ) );
			const c = dot( perp, perp ).div( R.mul( R ).mul( 0.5 ) );
			acc.addAssign( u.ptrVel.sub( v ).mul( exp( c.negate() ).mul( u.current ) ) );

			// semi-implicit Euler with exponential friction
			v.assign( v.mul( u.friction ).add( acc.mul( u.dt ) ) );
			const sp = length( v );
			v.mulAssign( min( float( 1 ), u.vmax.div( max( sp, 1e-6 ) ) ) );
			p.addAssign( v.mul( u.dt ) );

			this.pos.element( k ).assign( vec4( p, P.w ) );
			this.vel.element( k ).assign( vec4( v, dens ) );

		} )().compute( N, [ 128 ] ).setName( 'genesis.force' );

		this.computes.push( this.initKernel, this.speciesKernel, this.clearKernel, this.countKernel, this.scatterKernel, this.forceKernel );
		this.stepList = [ this.clearKernel, this.countKernel, ...this.scan.passes, this.scatterKernel, this.forceKernel ];

		this.setGrid();

	}

	// Recompute grid resolution for the current interaction radius / bound (cell size must be >= rMax).
	setGrid() {

		const u = this.u;
		const half = this.gridHalf;
		const cell = Math.max( u.rMax.value, 2 * half / this.maxDim );
		const dim = Math.max( 3, Math.min( this.maxDim, Math.floor( 2 * half / cell ) ) );
		const ext = dim * cell;
		u.dim.value = dim;
		u.dimF.value = dim;
		u.invCell.value = 1 / cell;
		u.origin.value.setScalar( - ext / 2 );
		this.dim = dim;

	}

	setMatrix( m ) {

		this.matrixAttr.array.set( m );
		this.matrixAttr.needsUpdate = true;

	}

	setFrictionHalfLife( halfLife ) {

		this.halfLife = halfLife;
		this.u.friction.value = Math.pow( 0.5, this.u.dt.value / Math.max( halfLife, 1e-4 ) );

	}

	reset( seed ) {

		this.u.seed.value = seed % 1000;
		this.renderer.compute( this.initKernel );

	}

	reassignSpecies() {

		this.renderer.compute( this.speciesKernel );

	}

	step( dt ) {

		const u = this.u;
		if ( u.dt.value !== dt ) {

			u.dt.value = dt;
			this.setFrictionHalfLife( this.halfLife ?? 0.04 );

		}

		this.renderer.compute( this.stepList );

	}

	dispose() {

		for ( const c of this.computes ) c.dispose();
		this.scan.dispose();
		const attrs = this.renderer._attributes;
		for ( const a of this.attributes ) {

			try {

				attrs?.delete( a );

			} catch ( e ) { /* already gone */ }

		}

	}

}
