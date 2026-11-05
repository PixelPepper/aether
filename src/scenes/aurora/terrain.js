// Aurora — landscape: snowy fjord walls (observer-centric polar mesh: constant angular vertex density),
// mirror water with ripples and shore ice, and a small cabin for scale. All lit by the aurora's integrated
// irradiance (computed on the GPU by the probe pass), starlight/airglow, and the cabin's window light.
import * as THREE from 'three/webgpu';
import {
	Fn, If, uniform, texture, float, vec2, vec3, vec4, sin, cos, exp, sqrt, pow, max, min, clamp, abs, length, normalize, dot,
	mix, smoothstep, select, saturate, positionWorld, normalWorld, cameraPosition, screenUV, uv, Discard, mx_noise_float, reflect,
} from 'three/tsl';
import { height, channel, EYE } from './height.js';

export function createLandscape( { tier, sim, sky, reflRT, eyeY } ) {

	const u = {
		time: uniform( 0 ),
		reflOn: uniform( 1 ),
		pix: uniform( 0.001 ),
		cabinPos: uniform( new THREE.Vector3() ),
		cabinGlow: uniform( 1 ),
		ripple: uniform( 1 ),
	};
	const La = sim.lightRead.element( 0 ).xyz;
	const Ld = sim.lightRead.element( 1 ).xyz;

	// ── terrain mesh: rings at geometrically growing radius around the observer ──
	const NR = tier >= 2 ? 560 : tier === 1 ? 360 : 200;
	const NT = tier >= 2 ? 960 : tier === 1 ? 600 : 320;
	const R0 = 0.004, R1 = 95, TH = 2.45;
	const pos = new Float32Array( NR * NT * 3 );
	for ( let i = 0; i < NR; i ++ ) {

		const r = R0 * Math.pow( R1 / R0, i / ( NR - 1 ) );
		for ( let j = 0; j < NT; j ++ ) {

			const th = ( j / ( NT - 1 ) * 2 - 1 ) * TH;
			const x = r * Math.sin( th ), z = - r * Math.cos( th );
			const k = ( i * NT + j ) * 3;
			pos[ k ] = x; pos[ k + 1 ] = height( x, z ); pos[ k + 2 ] = z;

		}

	}

	const idx = new Uint32Array( ( NR - 1 ) * ( NT - 1 ) * 6 );
	let n = 0;
	for ( let i = 0; i < NR - 1; i ++ ) for ( let j = 0; j < NT - 1; j ++ ) {

		const a = i * NT + j, b = a + 1, c = a + NT, d = c + 1;
		idx[ n ++ ] = a; idx[ n ++ ] = c; idx[ n ++ ] = b;
		idx[ n ++ ] = b; idx[ n ++ ] = c; idx[ n ++ ] = d;

	}

	const tGeo = new THREE.BufferGeometry();
	tGeo.setAttribute( 'position', new THREE.BufferAttribute( pos, 3 ) );
	tGeo.setIndex( new THREE.BufferAttribute( idx, 1 ) );
	tGeo.computeVertexNormals();
	// make sure the normals point up (winding independent)
	const nrm = tGeo.attributes.normal.array;
	if ( nrm[ 1 ] < 0 ) for ( let i = 0; i < nrm.length; i ++ ) nrm[ i ] = - nrm[ i ];

	const cabin = placeCabin();
	u.cabinPos.value.set( cabin.x + 0.004, cabin.y + 0.002, cabin.z + 0.003 );

	const ambient = vec3( 0.004, 0.0052, 0.0075 ); // starlight + airglow irradiance (÷π applied below)
	const cabinCol = vec3( 1.0, 0.55, 0.22 );

	const lightSurface = ( p, nIn, albedo, rimAmt ) => {

		const nn = normalize( nIn );
		const V = normalize( cameraPosition.sub( p ) );
		const wrap = saturate( dot( nn, Ld ).mul( 0.62 ).add( 0.38 ) );
		const rim = pow( float( 1 ).sub( saturate( dot( nn, V ) ) ), 4 ).mul( saturate( dot( nn, Ld ).add( 0.45 ) ) ).mul( rimAmt );
		const sky = ambient.mul( nn.y.mul( 0.4 ).add( 0.6 ) );
		const dc = u.cabinPos.sub( p );
		const d2 = dot( dc, dc );
		const cab = cabinCol.mul( u.cabinGlow ).mul( 1.6e-6 ).div( d2.add( 2e-6 ) ).mul( saturate( dot( nn, dc.div( sqrt( d2 ) ) ).mul( 0.7 ).add( 0.3 ) ) );
		return albedo.mul( La.mul( wrap.add( rim ) ).add( sky ).add( cab ) ).mul( 1 / Math.PI );

	};

	// aerial perspective toward the dim, aurora-tinted night air
	const haze = ( col, p ) => {

		const d = length( p.sub( cameraPosition ) );
		const f = float( 1 ).sub( exp( d.mul( - 1 / 38 ) ) ).mul( 0.85 );
		return mix( col, vec3( 0.0035, 0.0045, 0.007 ).add( La.mul( 0.012 ) ), f );

	};

	const tMat = new THREE.MeshBasicNodeMaterial();
	tMat.colorNode = Fn( () => {

		const p = positionWorld;
		Discard( p.y.lessThan( - 0.0006 ) ); // the reflection camera must not see the fjord bed
		const nn = normalize( normalWorld ).toVar();
		const dn = mx_noise_float( p.xz.mul( 2.3 ) ).mul( 0.5 ).add( mx_noise_float( p.xz.mul( 11 ) ).mul( 0.25 ) );
		// snow sticks where it is flat enough; wind-scoured rock on steep faces and arêtes
		const snow = smoothstep( 0.5, 0.72, nn.y.add( dn.mul( 0.22 ) ).add( smoothstep( 0.2, 1.2, p.y ).mul( 0.06 ) ) );
		const near = smoothstep( 0.0005, 0.0025, p.y ); // dark wet rock at the waterline
		const rock = vec3( 0.035, 0.034, 0.037 ).mul( dn.mul( 0.5 ).add( 1 ) );
		const snowC = vec3( 0.80, 0.85, 0.93 );
		const albedo = mix( rock, snowC, snow.mul( near.mul( 0.8 ).add( 0.2 ) ) );
		const col = lightSurface( p, nn, albedo, snow.mul( 0.9 ).add( 0.15 ) );
		return haze( col, p );

	} )();
	const terrain = new THREE.Mesh( tGeo, tMat );
	terrain.frustumCulled = false;

	// ── water ──
	const chan = ( z ) => {

		const nn = max( z.negate(), 0 );
		const xc = sin( nn.mul( 0.13 ).add( 0.4 ) ).mul( 0.55 ).mul( min( nn.div( 4 ), 1 ) ).add( sin( nn.mul( 0.045 ).sub( 0.6 ) ).mul( 0.9 ).mul( min( nn.div( 12 ), 1 ) ) ).sub( 0.15 );
		const hw = sin( nn.mul( 0.21 ) ).mul( 0.25 ).add( nn.mul( 0.075 ) ).add( 0.55 ).mul( float( 1 ).sub( smoothstep( 20, 30, nn ).mul( 0.85 ) ) );
		return { xc, hw };

	};

	const WAVES = [ // direction, wavelength (km), slope amplitude, speed
		[ 0.8, 0.6, 0.0007, 0.050, 1.1 ], [ - 0.3, 0.95, 0.0012, 0.040, 1.4 ], [ 0.95, - 0.31, 0.0023, 0.030, 2.0 ],
		[ - 0.7, 0.71, 0.004, 0.022, 2.6 ], [ 0.2, 0.98, 0.009, 0.010, 4.0 ],
	];
	const wMat = new THREE.MeshBasicNodeMaterial();
	wMat.colorNode = Fn( () => {

		const p = positionWorld;
		const toC = cameraPosition.sub( p );
		const dist = length( toC );
		const V = toC.div( dist );
		const grazing = max( V.y, 0.015 );
		const sx = float( 0 ).toVar(), sz = float( 0 ).toVar();
		for ( const [ dx, dz, lam, amp, spd ] of WAVES ) {

			const k = 2 * Math.PI / lam;
			// fade waves whose wavelength falls below ~4 pixels in the (foreshortened) footprint
			const fp = dist.mul( u.pix ).div( grazing ).mul( 1 / lam );
			const fade = float( 1 ).sub( smoothstep( 0.08, 0.35, fp ) );
			const ph = p.x.mul( k * dx ).add( p.z.mul( k * dz ) ).add( u.time.mul( spd ) );
			const s = cos( ph ).mul( amp ).mul( fade ).mul( u.ripple );
			sx.addAssign( s.mul( dx ) ); sz.addAssign( s.mul( dz ) );

		}

		const cat = mx_noise_float( vec3( p.xz.mul( 220 ), u.time.mul( 0.15 ) ) ).mul( 0.012 ).mul( float( 1 ).sub( smoothstep( 0.1, 0.4, dist.mul( u.pix ).div( grazing ).mul( 220 ) ) ) );
		const nn = normalize( vec3( sx.add( cat ).negate(), 1, sz.add( cat.mul( 0.7 ) ).negate() ) ).toVar();
		// shore ice: thin ice and slush along the margins of the bay
		const { xc, hw } = chan( p.z );
		const dd = abs( p.x.sub( xc ) ).div( hw );
		const iceN = mx_noise_float( p.xz.mul( 70 ) ).mul( 0.5 ).add( mx_noise_float( p.xz.mul( 300 ) ).mul( 0.25 ) );
		const ice = smoothstep( 0.86, 0.95, dd.add( iceN.mul( 0.12 ) ) ).toVar();
		const crack = float( 1 ).sub( smoothstep( 0.0, 0.04, abs( mx_noise_float( p.xz.mul( 140 ) ) ) ) );
		const nIce = normalize( vec3( iceN.mul( 0.05 ), 1, mx_noise_float( p.xz.mul( 90 ).add( 3 ) ).mul( 0.05 ) ) );
		nn.assign( normalize( mix( nn, nIce, ice ) ) );
		const cosT = saturate( dot( nn, V ) );
		const F = pow( float( 1 ).sub( cosT ), 5 ).mul( 0.98 ).add( 0.02 ).mul( float( 1 ).sub( ice.mul( 0.55 ) ) );
		const refl = vec3( 0 ).toVar();
		If( u.reflOn.greaterThan( 0.5 ), () => {

			const off = vec2( nn.x.mul( 0.9 ), nn.z.mul( 2.4 ) ).mul( 0.09 ).mul( ice.mul( 3 ).add( 1 ) );
			refl.assign( texture( reflRT.texture, vec2( float( 1 ).sub( screenUV.x ), screenUV.y ).add( off ) ).rgb );

		} ).Else( () => {

			refl.assign( sky.skyBase( reflect( V.negate(), nn ), u.pix ) );

		} );
		const body = vec3( 0.00025, 0.0005, 0.0008 ).add( La.mul( 0.0015 ) );
		const iceAlb = mix( vec3( 0.32, 0.40, 0.46 ), vec3( 0.75, 0.80, 0.88 ), smoothstep( 0.0, 0.5, iceN.add( 0.25 ) ) ).mul( float( 1 ).sub( crack.mul( 0.6 ) ) );
		const iceCol = lightSurface( p, nIce, iceAlb, float( 0.3 ) );
		const water = mix( body, refl, F );
		return haze( mix( water, iceCol.add( refl.mul( F ) ), ice.mul( 0.92 ) ), p );

	} )();
	const wGeo = new THREE.CircleGeometry( 140, 96 );
	wGeo.rotateX( - Math.PI / 2 );
	const water = new THREE.Mesh( wGeo, wMat );
	water.frustumCulled = false;

	// ── cabin ──
	const cabinGroup = new THREE.Group();
	const wood = new THREE.MeshBasicNodeMaterial();
	wood.colorNode = Fn( () => lightSurface( positionWorld, normalWorld, vec3( 0.09, 0.035, 0.022 ), float( 0.2 ) ) )();
	const roofM = new THREE.MeshBasicNodeMaterial();
	roofM.colorNode = Fn( () => lightSurface( positionWorld, normalWorld, vec3( 0.78, 0.82, 0.9 ), float( 0.4 ) ) )();
	const winM = new THREE.MeshBasicNodeMaterial();
	winM.colorNode = Fn( () => vec3( 2.6, 1.35, 0.5 ).mul( u.cabinGlow ).mul( sin( u.time.mul( 7.3 ) ).mul( 0.03 ).add( sin( u.time.mul( 2.1 ) ).mul( 0.04 ) ).add( 1 ) ) )();
	const body = new THREE.BoxGeometry( 0.008, 0.0042, 0.0055 );
	const roof = new THREE.CylinderGeometry( 0.0001, 0.0045, 0.0026, 4, 1 );
	roof.rotateY( Math.PI / 4 );
	roof.scale( 1.35, 1, 0.95 );
	const win = new THREE.PlaneGeometry( 0.0011, 0.0011 );
	const bodyMesh = new THREE.Mesh( body, wood );
	bodyMesh.position.y = 0.0021;
	const roofMesh = new THREE.Mesh( roof, roofM );
	roofMesh.position.y = 0.0042 + 0.0013;
	cabinGroup.add( bodyMesh, roofMesh );
	for ( const [ x, z, ry ] of [ [ - 0.0018, 0.00276, 0 ], [ 0.0016, 0.00276, 0 ], [ 0.00401, 0.0, Math.PI / 2 ] ] ) {

		const w = new THREE.Mesh( win, winM );
		w.position.set( x, 0.0022, z ); w.rotation.y = ry;
		cabinGroup.add( w );

	}

	cabinGroup.position.set( cabin.x, cabin.y - 0.0004, cabin.z );
	cabinGroup.rotation.y = 0.35;
	// window glow (bloom does the rest) — a soft sprite so the light reads even when the cabin is sub-pixel
	const glowM = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
	glowM.colorNode = Fn( () => {

		const r = length( uv().sub( 0.5 ) ).mul( 2 );
		return vec4( vec3( 1.0, 0.5, 0.18 ).mul( exp( r.mul( r ).mul( - 9 ) ).mul( 0.5 ).add( exp( r.mul( - 14 ) ).mul( 0.9 ) ) ).mul( u.cabinGlow ), 1 );

	} )();
	const glow = new THREE.Sprite( glowM );
	glow.scale.setScalar( 0.03 );
	glow.position.set( cabin.x + 0.002, cabin.y + 0.0025, cabin.z + 0.004 );

	const group = new THREE.Group();
	group.add( terrain, water, cabinGroup, glow );

	return {
		group, terrain, water, u, cabin,
		dispose() {

			tGeo.dispose(); tMat.dispose(); wGeo.dispose(); wMat.dispose(); wood.dispose(); roofM.dispose(); winM.dispose();
			body.dispose(); roof.dispose(); win.dispose(); glowM.dispose();

		},
	};

}

// a flat-ish spot just above the west shore, ~2.4 km up the fjord
function placeCabin() {

	const z = - 2.4;
	const [ xc, hw ] = channel( z );
	let best = null;
	for ( let x = xc - hw * 0.8; x > xc - hw * 2.5; x -= 0.004 ) {

		const h = height( x, z );
		if ( h > 0.006 ) {

			const s = Math.abs( height( x + 0.01, z ) - height( x - 0.01, z ) ) + Math.abs( height( x, z + 0.01 ) - height( x, z - 0.01 ) );
			if ( s < 0.012 ) {

				best = { x: x - 0.006, y: height( x - 0.006, z ), z };
				break;

			}

		}

	}

	return best || { x: xc - hw * 1.1, y: Math.max( 0.004, height( xc - hw * 1.1, z ) ), z };

}

export const _keep = { EYE, select, clamp, If };
