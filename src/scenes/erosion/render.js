// Deep Time — rendering: displaced terrain with slope/height/sediment/moisture material blending, a separate water
// surface (depth absorption, turbidity, sky reflection, sun glitter, foam, faint sunset glow), an open ocean ring
// to the horizon, an analytic sunset sky, height fog / aerial perspective, and a storm cell with rain.
// All lighting is evaluated by hand (GGX + Lambert + hemispherical sky term) so the heightfield shadows and AO
// baked by the sim's prep pass can be applied exactly.

import * as THREE from 'three/webgpu';
import {
	Fn, uniform, float, int, uint, vec2, vec3, vec4, ivec2, clamp, min, max, abs, sqrt, exp, pow, mix, smoothstep,
	floor, fract, length, dot, normalize, reflect, sin, cos, texture, textureLoad, varying, positionGeometry,
	cameraPosition, positionWorldDirection, hash, instanceIndex, uv, Discard, select,
} from 'three/tsl';

// ───────────────────────── shared uniforms ─────────────────────────
export function makeRenderUniforms( sim ) {

	return {
		sunDir: sim.u.sunDir,
		sunCol: uniform( new THREE.Color( 1.0, 0.56, 0.31 ).multiplyScalar( 3.6 ) ),
		time: uniform( 0 ),
		snow: uniform( 1.02 ),
		waterVis: uniform( 0.06 ), // grid-unit depth below which water is just a film
		fogDen: uniform( 0.055 ),
		fogFall: uniform( 1.35 ),
		storm: uniform( new THREE.Vector3( 0, 2.1, 0 ) ),
		stormVis: uniform( 0 ),
		stormRain: uniform( 0 ),
		stormRW: uniform( 0.7 ),
		glow: uniform( 1 ),
	};

}

// ───────────────────────── small noise ─────────────────────────
const hash2 = ( p ) => hash( uint( int( p.x ).add( 4096 ) ).mul( uint( 73856093 ) ).bitXor( uint( int( p.y ).add( 4096 ) ).mul( uint( 19349663 ) ) ) );

export const vnoise = /*@__PURE__*/ Fn( ( [ p_immutable ] ) => {

	const p = vec2( p_immutable ).toVar();
	const i = floor( p );
	const f = fract( p );
	const w = f.mul( f ).mul( f.mul( - 2 ).add( 3 ) );
	const a = hash2( i );
	const b = hash2( i.add( vec2( 1, 0 ) ) );
	const c = hash2( i.add( vec2( 0, 1 ) ) );
	const d = hash2( i.add( vec2( 1, 1 ) ) );
	return mix( mix( a, b, w.x ), mix( c, d, w.x ), w.y );

} ).setLayout( { name: 'deepTimeNoise', type: 'float', inputs: [ { name: 'p', type: 'vec2' } ] } );

const fbm3 = ( p ) => vnoise( p ).mul( 0.55 ).add( vnoise( p.mul( 2.13 ).add( 7.1 ) ).mul( 0.3 ) ).add( vnoise( p.mul( 4.37 ).add( 3.3 ) ).mul( 0.15 ) );

// ───────────────────────── sky & atmosphere ─────────────────────────
export function skyColor( dir, U, disk = true ) {

	const y = dir.y;
	const mu = max( dot( dir, U.sunDir ), 0 );
	const t = clamp( y, 0, 1 );
	const hor = vec3( 1.10, 0.50, 0.27 );
	const mid = vec3( 0.36, 0.30, 0.46 );
	const zen = vec3( 0.040, 0.075, 0.20 );
	let c = mix( hor, mid, smoothstep( 0.0, 0.16, t ) );
	c = mix( c, zen, smoothstep( 0.1, 0.75, t ) );
	// warm Mie glow around the sun, stronger near the horizon
	c = c.add( U.sunCol.mul( pow( mu, 6 ).mul( 0.16 ).add( pow( mu, 48 ).mul( 0.32 ) ) ) );
	c = c.add( vec3( 0.9, 0.32, 0.12 ).mul( pow( mu, 2.5 ) ).mul( exp( t.mul( - 7 ) ) ).mul( 0.55 ) );
	if ( disk ) c = c.add( U.sunCol.mul( smoothstep( 0.99965, 0.99985, mu ) ).mul( 14 ) );
	// below the horizon: dim warm haze
	return mix( c, hor.mul( 0.42 ), smoothstep( 0.0, - 0.12, y ) );

}

// height fog with sun in-scattering (exponential density, closed-form integral along the view ray)
export function applyFog( col, P, U ) {

	const ro = cameraPosition;
	const rv = P.sub( ro );
	const dist = length( rv );
	const rd = rv.div( dist );
	const b = U.fogFall;
	const ry = rd.y.mul( b );
	const safe = select( abs( ry ).lessThan( 1e-3 ), float( 1e-3 ), ry );
	const integral = exp( ro.y.mul( b ).negate() ).mul( float( 1 ).sub( exp( dist.mul( safe ).negate() ) ) ).div( safe );
	const fogAmt = clamp( float( 1 ).sub( exp( U.fogDen.mul( integral ).negate() ) ), 0, 1 );
	const haze = skyColor( normalize( vec3( rd.x, 0.035, rd.z ) ), U, false );
	const mu = max( dot( rd, U.sunDir ), 0 );
	const fogCol = haze.add( U.sunCol.mul( pow( mu, 10 ) ).mul( 0.25 ) );
	return mix( col, fogCol, fogAmt );

}

// ───────────────────────── PBR helpers ─────────────────────────
function ggxSpec( n, v, l, rough, f0 ) {

	const h = normalize( v.add( l ) );
	const a = max( rough.mul( rough ), 0.002 );
	const a2 = a.mul( a );
	const nh = max( dot( n, h ), 0 );
	const nl = max( dot( n, l ), 0 );
	const nv = max( dot( n, v ), 1e-3 );
	const dd = nh.mul( nh ).mul( a2.sub( 1 ) ).add( 1 );
	const D = a2.div( dd.mul( dd ).mul( Math.PI ) );
	const Vis = float( 0.5 ).div( nl.mul( sqrt( nv.mul( nv ).mul( float( 1 ).sub( a2 ) ).add( a2 ) ) ).add( nv.mul( sqrt( nl.mul( nl ).mul( float( 1 ).sub( a2 ) ).add( a2 ) ) ) ).add( 1e-5 ) );
	const F = f0.add( float( 1 ).sub( f0 ).mul( pow( float( 1 ).sub( max( dot( v, h ), 0 ) ), 5 ) ) );
	return D.mul( Vis ).mul( F ).mul( nl );

}

function skyAmbient( n, U ) {

	const up = n.y.mul( 0.5 ).add( 0.5 );
	const sky = vec3( 0.20, 0.23, 0.36 );
	const ground = vec3( 0.11, 0.075, 0.055 );
	const sunSide = max( dot( normalize( vec3( n.x, 0, n.z ) ), normalize( vec3( U.sunDir.x, 0, U.sunDir.z ) ) ), 0 );
	return mix( ground, sky, up ).add( vec3( 0.16, 0.08, 0.04 ).mul( sunSide ).mul( up ) );

}

// storm cell shadow projected along the sun direction
function cloudShade( P, U ) {

	const s = U.storm;
	const k = s.y.sub( P.y ).div( max( U.sunDir.y, 0.05 ) );
	const c = vec2( s.x, s.z ).sub( vec2( U.sunDir.x, U.sunDir.z ).mul( k ) );
	const d = P.xz.sub( c );
	const r = U.stormRW;
	const under = exp( dot( d, d ).negate().div( r.mul( r ).mul( 1.6 ) ) );
	const rainDark = exp( dot( P.xz.sub( vec2( s.x, s.z ) ), P.xz.sub( vec2( s.x, s.z ) ) ).negate().div( r.mul( r ) ) );
	return float( 1 ).sub( under.mul( U.stormVis ).mul( 0.75 ) ).mul( float( 1 ).sub( rainDark.mul( U.stormRain ).mul( 0.25 ) ) );

}

// ───────────────────────── terrain material model ─────────────────────────
// P world pos, n normal, soft/strata world, wet 0..1, ao, sim params
function terrainAlbedo( P, n, softW, strataW, wet, ao, sim, U ) {

	const gs = sim.u.gridScale;
	const slope = float( 1 ).sub( n.y );
	const nz = fbm3( P.xz.mul( 7.0 ) ).toVar();
	const nzf = vnoise( P.xz.mul( 41.0 ) );

	// bedrock strata — same banding as the sim's hardness, so cliff-forming beds are the pale ones
	const N = sim.N, S = sim.size;
	const gx = P.x.div( S ).add( 0.5 ).mul( N ), gy = P.z.div( S ).add( 0.5 ).mul( N );
	const st = strataW.mul( gs ).mul( sim.u.strataFreq ).add( sin( gx.mul( 0.013 ).add( gy.mul( 0.009 ) ) ).mul( 2.0 ) );
	const hard = smoothstep( - 0.2, 0.5, sin( st ) );
	const red = vec3( 0.33, 0.15, 0.075 );
	const pale = vec3( 0.47, 0.40, 0.31 );
	const shale = vec3( 0.12, 0.10, 0.09 );
	let rock = mix( red, pale, hard );
	rock = mix( rock, shale, smoothstep( 0.55, 0.95, sin( st.mul( 0.37 ).add( 1.7 ) ) ).mul( float( 1 ).sub( hard ) ) );
	rock = rock.mul( nz.mul( 0.5 ).add( 0.75 ) ).mul( nzf.mul( 0.3 ).add( 0.85 ) );

	const softG = softW.mul( gs );
	const sed = smoothstep( 0.15, 2.5, softG ).toVar();
	const soil = vec3( 0.15, 0.095, 0.055 );
	const sand = vec3( 0.50, 0.40, 0.27 );
	const veg = vec3( 0.050, 0.085, 0.030 );
	const snowC = vec3( 0.86, 0.89, 0.97 );

	const rockW = smoothstep( 0.16, 0.40, slope.add( nz.sub( 0.5 ).mul( 0.12 ) ) ).mul( float( 1 ).sub( sed.mul( 0.7 ) ) );
	let col = mix( soil.mul( nz.mul( 0.4 ).add( 0.8 ) ), sand.mul( nzf.mul( 0.2 ).add( 0.9 ) ), sed );

	const h = P.y;
	const moist = clamp( wet.mul( 1.5 ).add( float( 1 ).sub( ao ).mul( 0.9 ) ).add( nz.sub( 0.45 ).mul( 0.7 ) ).add( smoothstep( 0.55, 0.05, h ).mul( 0.25 ) ), 0, 1 );
	const vegW = smoothstep( 0.32, 0.08, slope ).mul( smoothstep( 0.25, 0.65, moist ) ).mul( smoothstep( U.snow.sub( 0.1 ), U.snow.sub( 0.45 ), h ) )
		.mul( float( 1 ).sub( sed.mul( 0.75 ) ) ).mul( smoothstep( 0.0, 0.03, h ) );
	col = mix( col, veg.mul( nzf.mul( 0.5 ).add( 0.75 ) ), vegW.mul( 0.9 ) );
	col = mix( col, rock, rockW );

	const snowW = smoothstep( U.snow.sub( 0.05 ), U.snow.add( 0.08 ), h.add( nz.sub( 0.5 ).mul( 0.25 ) ).sub( slope.mul( 0.35 ) ) ).mul( smoothstep( 0.62, 0.38, slope ) );

	// wet darkening (moisture held in the soil, not standing water)
	const wetD = clamp( wet.mul( 1.6 ), 0, 1 ).mul( float( 1 ).sub( snowW ) );
	col = col.mul( mix( float( 1 ), float( 0.42 ), wetD ) );
	col = mix( col, snowC, snowW );
	const rough = mix( mix( float( 0.88 ), float( 0.3 ), wetD ), float( 0.55 ), snowW );
	return { albedo: col, rough, snowW };

}

function shadeSurface( P, n, albedo, rough, shadow, ao, U, f0 = 0.04 ) {

	const v = normalize( cameraPosition.sub( P ) );
	const l = U.sunDir;
	const cs = cloudShade( P, U );
	const nl = max( dot( n, l ), 0 );
	const sun = U.sunCol.mul( shadow.mul( cs ) );
	const diff = albedo.mul( sun.mul( nl ).add( skyAmbient( n, U ).mul( ao ).mul( cs.mul( 0.4 ).add( 0.6 ) ) ) );
	const spec = sun.mul( ggxSpec( n, v, l, rough, float( f0 ) ) );
	return diff.add( spec );

}

// ───────────────────────── geometry helpers ─────────────────────────
function gridLookup( sim ) {

	const N = sim.N, S = sim.size;
	const gp = positionGeometry;
	const fx = gp.x.div( S ).add( 0.5 ).mul( N - 1 );
	const fz = gp.z.div( S ).add( 0.5 ).mul( N - 1 );
	const ti = ivec2( int( fx.add( 0.5 ) ), int( fz.add( 0.5 ) ) ).toVar();
	return { gp, ti };

}

const texUV = ( P, sim ) => P.xz.div( sim.size ).add( 0.5 ).mul( ( sim.N - 1 ) / sim.N ).add( 0.5 / sim.N );

// ───────────────────────── terrain ─────────────────────────
export function makeTerrain( sim, U, geometry ) {

	const mat = new THREE.MeshBasicNodeMaterial();
	const { gp, ti } = gridLookup( sim );
	const geo = textureLoad( sim.texGeo, ti );
	const pos = vec3( gp.x, geo.x, gp.z );
	mat.positionNode = pos;

	const vP = varying( pos, 'vTerrP' );
	const vG = varying( geo, 'vTerrG' );
	const tuv = texUV( vP, sim );
	const lit = texture( sim.texLit, tuv );
	const flow = texture( sim.texFlow, tuv );
	const n = normalize( vec3( lit.x, sqrt( max( float( 1 ).sub( lit.x.mul( lit.x ) ).sub( lit.y.mul( lit.y ) ), 0.0 ) ), lit.y ) ).toVar();
	const shadow = lit.z, ao = lit.w;
	const wet = flow.y;

	mat.colorNode = Fn( () => {

		const m = terrainAlbedo( vP, n, vG.z, vG.w, wet, ao, sim, U );
		const c = shadeSurface( vP, n, m.albedo, m.rough, shadow, ao, U );
		return vec4( applyFog( c, vP, U ), 1 );

	} )();

	const mesh = new THREE.Mesh( geometry, mat );
	mesh.frustumCulled = false;
	return mesh;

}

// ───────────────────────── water shading (rivers, lakes, sea) ─────────────────────────
function waterColor( P, nBase, depthW, flowV, speed, turb, discharge, bed, shadow, U, sim ) {

	const t = U.time;
	// two-phase flow-advected ripples (flow-map style) so detail streams along the current without stretching
	const ph0 = fract( t.mul( 0.35 ) );
	const ph1 = fract( t.mul( 0.35 ).add( 0.5 ) );
	const wgt = abs( ph0.mul( 2 ).sub( 1 ) );
	const flowW = flowV.mul( 0.035 );
	const rip = ( q ) => {

		const e = 0.07;
		const h0 = fbm3( q );
		return vec2( fbm3( q.add( vec2( e, 0 ) ) ).sub( h0 ), fbm3( q.add( vec2( 0, e ) ) ).sub( h0 ) ).div( e );

	};

	const q = P.xz.mul( 26 );
	const g0 = rip( q.sub( flowW.mul( ph0 ).mul( 26 ) ).add( t.mul( 0.15 ) ) );
	const g1 = rip( q.sub( flowW.mul( ph1 ).mul( 26 ) ).add( 11.3 ).sub( t.mul( 0.13 ) ) );
	const g = mix( g0, g1, wgt );
	const amp = clamp( speed.mul( 0.25 ).add( 0.035 ), 0, 0.3 );
	const n = normalize( nBase.add( vec3( g.x, 0, g.y ).mul( amp.negate() ) ) ).toVar();

	const v = normalize( cameraPosition.sub( P ) );
	const nv = max( dot( n, v ), 0.02 );
	const F = float( 0.02 ).add( pow( float( 1 ).sub( nv ), 5 ).mul( 0.98 ) );
	const r = reflect( v.negate(), n );
	const refl = skyColor( normalize( vec3( r.x, max( r.y, 0.02 ), r.z ) ), U, false );
	const cs = cloudShade( P, U );
	const sunSpec = U.sunCol.mul( ggxSpec( n, v, U.sunDir, clamp( float( 0.07 ).add( speed.mul( 0.05 ) ), 0.05, 0.25 ), float( 0.02 ) ) ).mul( shadow ).mul( cs );

	// absorption + scattering (turbid water is brown, clear deep water teal)
	const tb = clamp( turb.mul( 2.0 ), 0, 1 );
	const path = depthW.mul( float( 1 ).add( float( 1 ).div( max( nv, 0.12 ) ) ) );
	const sigma = mix( vec3( 7.5, 2.6, 1.8 ), vec3( 9, 7, 6 ), tb );
	const T = exp( sigma.mul( path ).negate() );
	const amb = skyAmbient( vec3( 0, 1, 0 ), U ).mul( 0.6 ).add( U.sunCol.mul( shadow ).mul( U.sunDir.y ).mul( 0.6 ) ).mul( cs );
	const scat = mix( vec3( 0.010, 0.050, 0.060 ), vec3( 0.20, 0.13, 0.06 ), tb ).mul( amb );
	const under = bed.mul( T ).add( scat.mul( float( 1 ).sub( T ) ) );
	let col = mix( under, refl.mul( cs.mul( 0.3 ).add( 0.7 ) ), F ).add( sunSpec );

	// foam on fast water and along breaking shallows
	const fn = vnoise( P.xz.mul( 70 ).sub( flowV.mul( t ).mul( 0.4 ) ) );
	const foam = smoothstep( 0.75, 1.15, speed ).mul( smoothstep( 0.35, 0.75, fn ) ).mul( 0.8 );
	const foamLit = U.sunCol.mul( shadow ).mul( max( U.sunDir.y, 0 ) ).add( skyAmbient( vec3( 0, 1, 0 ), U ) ).mul( 0.75 );
	col = mix( col, foamLit, foam.mul( cs ) );

	// faint sunset glow along concentrated flow
	const q2 = smoothstep( 0.02, 0.5, discharge );
	col = col.add( vec3( 1.0, 0.38, 0.16 ).mul( q2 ).mul( 0.16 ).mul( U.glow ).mul( smoothstep( 0.0, 0.01, depthW ) ) );
	return col;

}

export function makeWater( sim, U, geometry ) {

	const N = sim.N;
	const mat = new THREE.MeshBasicNodeMaterial();
	const { gp, ti } = gridLookup( sim );
	const geo = textureLoad( sim.texGeo, ti );
	const cw = sim.cellW;
	const nb = ( dx, dy ) => {

		const g = textureLoad( sim.texGeo, ivec2( clamp( ti.x.add( dx ), int( 0 ), int( N - 1 ) ), clamp( ti.y.add( dy ), int( 0 ), int( N - 1 ) ) ) );
		return g.x.add( g.y );

	};

	const dG = geo.y.div( cw );
	const wet = dG.greaterThan( U.waterVis );
	const surf = geo.x.add( geo.y );
	const y = select( wet, surf, geo.x.sub( cw * 2.5 ) );
	const pos = vec3( gp.x, y, gp.z );
	mat.positionNode = pos;
	const nW = normalize( vec3( nb( - 1, 0 ).sub( nb( 1, 0 ) ), cw * 2, nb( 0, - 1 ).sub( nb( 0, 1 ) ) ) );

	const vP = varying( pos, 'vWatP' );
	const vD = varying( select( wet, geo.y.sub( U.waterVis.mul( cw ) ), float( 0 ) ), 'vWatD' );
	const vN = varying( nW, 'vWatN' );
	const vG = varying( geo, 'vWatG' );

	mat.colorNode = Fn( () => {

		const tuv = texUV( vP, sim );
		const lit = texture( sim.texLit, tuv );
		const flow = texture( sim.texFlow, tuv );
		const nT = normalize( vec3( lit.x, sqrt( max( float( 1 ).sub( lit.x.mul( lit.x ) ).sub( lit.y.mul( lit.y ) ), 0.0 ) ), lit.y ) );
		const depth = max( vD, 0 );
		// bed: terrain material under water, lit through the water column
		const bedP = vec3( vP.x, vG.x, vP.z );
		const m = terrainAlbedo( bedP, nT, vG.z, vG.w, float( 1 ), lit.w, sim, U );
		const sunT = exp( vec3( 7.5, 2.6, 1.8 ).mul( depth ).div( max( U.sunDir.y, 0.1 ) ).negate() );
		const bed = m.albedo.mul( U.sunCol.mul( max( dot( nT, U.sunDir ), 0 ) ).mul( lit.z ).mul( sunT ).add( skyAmbient( nT, U ).mul( lit.w ) ) );
		// flow direction is not stored; approximate it from the surface slope
		const fdir = normalize( vec2( vN.x, vN.z ).add( 1e-5 ) ).mul( flow.x );
		const c = waterColor( vP, normalize( vN ), depth, fdir, flow.x, flow.z, flow.w, bed, lit.z, U, sim );
		return vec4( applyFog( c, vP, U ), 1 );

	} )();

	const mesh = new THREE.Mesh( geometry, mat );
	mesh.frustumCulled = false;
	mesh.renderOrder = 1;
	return mesh;

}

// open ocean outside the simulated square, out to the horizon
export function makeOcean( sim, U ) {

	const S = sim.size / 2, R = 900;
	const quads = [
		[ - R, - R, - S, R ], [ S, - R, R, R ], [ - S, - R, S, - S ], [ - S, S, S, R ],
	];
	const pos = [];
	for ( const [ x0, z0, x1, z1 ] of quads ) pos.push( x0, 0, z0, x0, 0, z1, x1, 0, z1, x0, 0, z0, x1, 0, z1, x1, 0, z0 );
	const g = new THREE.BufferGeometry();
	g.setAttribute( 'position', new THREE.Float32BufferAttribute( pos, 3 ) );

	const mat = new THREE.MeshBasicNodeMaterial( { side: THREE.DoubleSide } );
	const vP = varying( positionGeometry, 'vOcP' );
	mat.colorNode = Fn( () => {

		const c = waterColor( vP, vec3( 0, 1, 0 ), float( 2 ), vec2( 0 ), float( 0 ), float( 0 ), float( 0 ), vec3( 0 ), float( 1 ), U, sim );
		return vec4( applyFog( c, vP, U ), 1 );

	} )();
	const mesh = new THREE.Mesh( g, mat );
	mesh.frustumCulled = false;
	return mesh;

}

export function makeSky( U ) {

	return Fn( () => skyColor( positionWorldDirection, U, true ) )();

}

// ───────────────────────── storm cell ─────────────────────────
export function makeStorm( U, tier ) {

	const group = new THREE.Group();
	const puffs = tier === 2 ? 90 : tier === 1 ? 60 : 36;
	const drops = tier === 2 ? 3000 : tier === 1 ? 1500 : 500;

	// cloud: lit billboard puffs
	const cm = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false } );
	const id = float( instanceIndex );
	const r1 = hash( id.mul( 3 ).add( 1 ) ), r2 = hash( id.mul( 3 ).add( 2 ) ), r3 = hash( id.mul( 3 ).add( 3 ) ), r4 = hash( id.add( 711 ) );
	const ang = r1.mul( Math.PI * 2 );
	const rad = sqrt( r2 ).mul( U.stormRW ).mul( 1.05 );
	const off = vec3( cos( ang ).mul( rad ), r3.mul( r3 ).mul( 0.45 ).mul( float( 1 ).sub( rad.div( U.stormRW ).mul( 0.6 ) ) ), sin( ang ).mul( rad ) );
	const drift = vec3( sin( U.time.mul( 0.21 ).add( id ) ), 0, cos( U.time.mul( 0.17 ).add( id.mul( 1.3 ) ) ) ).mul( 0.04 );
	cm.positionNode = U.storm.add( off ).add( drift );
	cm.scaleNode = vec2( r4.mul( 0.35 ).add( 0.4 ).mul( U.stormRW ).mul( 0.95 ).mul( U.stormVis.mul( 0.4 ).add( 0.6 ) ) );
	const pu = uv().sub( 0.5 );
	const rr = length( pu ).mul( 2 );
	const nse = fbm3( pu.mul( 3.2 ).add( id.mul( 7.31 ) ).add( U.time.mul( 0.03 ) ) );
	const alpha = smoothstep( 1.0, 0.25, rr.add( nse.sub( 0.5 ).mul( 0.7 ) ) ).mul( U.stormVis ).mul( 0.78 );
	const toSun = clamp( dot( normalize( off.add( vec3( 0, 0.12, 0 ) ) ), U.sunDir ).mul( 0.5 ).add( 0.5 ), 0, 1 );
	const height = clamp( off.y.div( 0.45 ), 0, 1 );
	const lit = pow( toSun, 2.0 ).mul( height.mul( 0.6 ).add( 0.4 ) );
	const ccol = mix( vec3( 0.045, 0.05, 0.07 ), U.sunCol.mul( 0.32 ), lit ).add( vec3( 0.05, 0.06, 0.09 ).mul( height ) );
	cm.colorNode = vec4( ccol, alpha );
	const cloud = new THREE.Sprite( cm );
	cloud.count = puffs;
	cloud.frustumCulled = false;
	cloud.renderOrder = 3;
	group.add( cloud );

	// rain streaks
	const rm = new THREE.SpriteNodeMaterial( { transparent: true, depthWrite: false, blending: THREE.AdditiveBlending } );
	const q1 = hash( id.mul( 5 ).add( 11 ) ), q2 = hash( id.mul( 5 ).add( 12 ) ), q3 = hash( id.mul( 5 ).add( 13 ) );
	const ra = q1.mul( Math.PI * 2 ), rd = sqrt( q2 ).mul( U.stormRW ).mul( 0.85 );
	const fall = fract( U.time.mul( 0.9 ).add( q3 ) );
	const top = U.storm.y.sub( 0.05 );
	rm.positionNode = vec3( U.storm.x.add( cos( ra ).mul( rd ) ), top.sub( fall.mul( top.add( 0.1 ) ) ), U.storm.z.add( sin( ra ).mul( rd ) ) );
	rm.scaleNode = vec2( 0.0045, 0.11 );
	const ru = uv();
	const streak = smoothstep( 0.5, 0.0, abs( ru.x.sub( 0.5 ) ) ).mul( ru.y );
	rm.colorNode = vec4( vec3( 0.45, 0.5, 0.62 ).mul( streak ).mul( U.stormRain ).mul( 0.5 ), 1 );
	const rain = new THREE.Sprite( rm );
	rain.count = drops;
	rain.frustumCulled = false;
	rain.renderOrder = 2;
	group.add( rain );

	group.userData.dispose = () => {

		cm.dispose(); rm.dispose();

	};

	return group;

}
