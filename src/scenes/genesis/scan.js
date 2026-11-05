// Genesis — multi-level parallel exclusive prefix sum (scan) on the GPU.
//
// Level k: every workgroup (256 threads) loads a block of 1024 uints (4 per thread), each thread reduces its 4 values
// serially, the 256 partial sums are scanned in workgroup shared memory (Hillis–Steele, log2(256) = 8 rounds of
// workgroupBarrier-separated read/add), and the block total is written to a sums buffer. If more than one block exists,
// the sums buffer is scanned recursively (in place) and a final pass adds each block's offset back.
// Two levels cover 1 M elements, three cover 1 G.
//
// All buffers are padded to a multiple of BLOCK so no bounds checks are needed (the barriers also make compute()'s
// automatic early-return unavailable, so we must never rely on it here).

import { StorageBufferAttribute } from 'three/webgpu';
import { Fn, If, uint, storage, instanceIndex, workgroupArray, workgroupBarrier, workgroupId, invocationLocalIndex } from 'three/tsl';

export const SCAN_WG = 256;
export const SCAN_PER = 4;
export const SCAN_BLOCK = SCAN_WG * SCAN_PER;

export const roundUp = ( n, m ) => Math.ceil( n / m ) * m;

function scanBlocksKernel( inNode, outNode, sumsNode, numBlocks, label ) {

	const shared = workgroupArray( 'uint', SCAN_WG );

	return Fn( () => {

		const lid = invocationLocalIndex;
		const block = workgroupId.x;
		const base = block.mul( SCAN_BLOCK ).add( lid.mul( SCAN_PER ) ).toVar( 'base' );

		const v = [];
		for ( let e = 0; e < SCAN_PER; e ++ ) v.push( inNode.element( base.add( e ) ).toVar( 'v' + e ) );

		const local = v.reduce( ( a, b ) => a.add( b ) ).toVar( 'local' );
		shared.element( lid ).assign( local );
		workgroupBarrier();

		for ( let off = 1; off < SCAN_WG; off <<= 1 ) {

			const t = uint( 0 ).toVar( 't' + off );
			If( lid.greaterThanEqual( uint( off ) ), () => {

				t.assign( shared.element( lid.sub( uint( off ) ) ) );

			} );
			workgroupBarrier();
			shared.element( lid ).assign( shared.element( lid ).add( t ) );
			workgroupBarrier();

		}

		const inclusive = shared.element( lid ).toVar( 'inclusive' );
		const run = inclusive.sub( local ).toVar( 'run' );
		for ( let e = 0; e < SCAN_PER; e ++ ) {

			outNode.element( base.add( e ) ).assign( run );
			if ( e < SCAN_PER - 1 ) run.addAssign( v[ e ] );

		}

		If( lid.equal( uint( SCAN_WG - 1 ) ), () => {

			sumsNode.element( block ).assign( inclusive );

		} );

	} )().compute( numBlocks * SCAN_WG, [ SCAN_WG ] ).setName( label );

}

function addOffsetsKernel( outNode, sumsNode, n, label ) {

	return Fn( () => {

		const i = instanceIndex;
		outNode.element( i ).addAssign( sumsNode.element( i.div( SCAN_BLOCK ) ) );

	} )().compute( n, [ 256 ] ).setName( label );

}

export class GPUScan {

	/**
	 * @param {Node} inNode  read view of the input (uint), length n
	 * @param {Node} outNode writable view of the output (uint), length n (may alias inNode)
	 * @param {number} n     element count, must be a multiple of SCAN_BLOCK
	 */
	constructor( inNode, outNode, n ) {

		if ( n % SCAN_BLOCK !== 0 ) throw new Error( 'GPUScan: length must be a multiple of ' + SCAN_BLOCK );
		this.passes = [];
		this.attributes = [];
		this.levels = 0;
		this._build( inNode, outNode, n, 0 );

	}

	_build( inNode, outNode, n, level ) {

		this.levels = Math.max( this.levels, level + 1 );
		const numBlocks = n / SCAN_BLOCK;
		const sumsLen = roundUp( numBlocks, SCAN_BLOCK );
		const sumsAttr = new StorageBufferAttribute( new Uint32Array( sumsLen ), 1, Uint32Array );
		this.attributes.push( sumsAttr );
		const sumsNode = storage( sumsAttr, 'uint', sumsLen );

		this.passes.push( scanBlocksKernel( inNode, outNode, sumsNode, numBlocks, 'genesis.scan' + level ) );

		if ( numBlocks > 1 ) {

			this._build( sumsNode, sumsNode, sumsLen, level + 1 );
			this.passes.push( addOffsetsKernel( outNode, sumsNode, n, 'genesis.scanAdd' + level ) );

		}

	}

	dispose() {

		for ( const p of this.passes ) p.dispose();

	}

}
