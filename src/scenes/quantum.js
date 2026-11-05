// PLACEHOLDER — replaced by the Genesis agent. Minimal compute example proving the contract works.
import * as THREE from 'three/webgpu';
import { Fn, instancedArray, instanceIndex, vec3, float, hash, time, sin, cos, color, uniform } from 'three/tsl';

export default {
	id: 'quantum',
	title: 'Quantum',
	subtitle: 'placeholder',
	async create( ctx ) {

		const { renderer, camera, controls } = ctx;
		const COUNT = ctx.tier === 0 ? 20000 : 200000;
		const scene = new THREE.Scene();
		camera.position.set( 0, 0, 6 );
		controls.target.set( 0, 0, 0 );

		const pos = instancedArray( COUNT, 'vec3' );
		const init = Fn( () => {

			const i = float( instanceIndex );
			const p = pos.element( instanceIndex );
			p.assign( vec3( hash( i ), hash( i.add( 1 ) ), hash( i.add( 2 ) ) ).sub( 0.5 ).mul( 4 ) );

		} )().compute( COUNT );
		await renderer.computeAsync( init );

		const upd = Fn( () => {

			const p = pos.element( instanceIndex );
			const a = float( 0.01 );
			p.assign( vec3( p.x.mul( cos( a ) ).sub( p.z.mul( sin( a ) ) ), p.y, p.x.mul( sin( a ) ).add( p.z.mul( cos( a ) ) ) ) );

		} )().compute( COUNT );

		const mat = new THREE.SpriteNodeMaterial( { blending: THREE.AdditiveBlending, depthWrite: false, transparent: true } );
		mat.positionNode = pos.toAttribute();
		mat.colorNode = color( 0x88bbff ).mul( 2 );
		mat.scaleNode = float( 0.02 );
		const sprites = new THREE.Sprite( mat );
		sprites.count = COUNT;
		sprites.frustumCulled = false;
		scene.add( sprites );

		const u = { speed: uniform( 1 ) };
		ctx.gui.add( u.speed, 'value', 0, 4 ).name( 'speed' );

		return {
			scene,
			bloom: { strength: 1.2, radius: 0.5, threshold: 0 },
			update() {

				renderer.compute( upd );
				ctx.setStats( [ [ 'particles', COUNT.toLocaleString() ] ] );

			},
			dispose() {

				mat.dispose();

			},
		};

	},
};
