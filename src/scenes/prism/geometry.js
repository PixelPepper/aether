// Analytic scene description: the room, plinth, glass objects (convex polyhedra as plane lists, spheres),
// ambient lights, beam presets and the caustic light-map atlas layout. Everything here is baked into WGSL.

export const ROOM = { x0: - 4.2, x1: 4.6, y1: 3.8, z0: - 3.4, z1: 5.0 };
export const PLINTH = { h: 0.92, w: 0.3 }; // height, half width

// Ambient ceiling lights (rectangles facing down): [cx, cz, halfX, halfZ, relative intensity]
export const CEIL_LIGHTS = [
	[ 0.2, - 1.6, 3.2, 0.05, 1.0 ],
	[ 0.2, 2.6, 3.2, 0.05, 1.0 ],
	[ 0.0, 0.0, 0.26, 0.26, 1.6 ],
];

// ── convex solids (object space, y up, origin at the base / centre) ──
export function prismPlanes( R = 0.21, H = 0.46 ) {

	// equilateral triangular prism, vertical axis, apex at +z. Inradius = R/2.
	const planes = [];
	for ( let k = 0; k < 3; k ++ ) {

		const a = - Math.PI / 2 + k * ( 2 * Math.PI / 3 ); // face normals at -90°, 30°, 150° (apex at +90°)
		planes.push( [ Math.cos( a ), 0, Math.sin( a ), R / 2 ] );

	}

	planes.push( [ 0, 1, 0, H ] );
	planes.push( [ 0, - 1, 0, 0 ] );
	return planes;

}

// Round brilliant cut (Tolkowsky-like proportions), girdle radius r. Returns outward planes [nx,ny,nz,d].
export function brilliantPlanes( r = 0.2 ) {

	const P = [];
	const rad = Math.PI / 180;
	const gth = 0.022; // girdle half thickness (in units of r)
	const crownA = 34.5 * rad, pavA = 40.75 * rad, starA = 21.5 * rad, ugA = 41.5 * rad, lgA = 42.2 * rad;
	const hc = ( 1 - 0.56 ) * Math.tan( crownA ); // crown height for a 56 % table
	const yt = gth + hc;
	const add = ( n, p ) => {

		const l = Math.hypot( n[ 0 ], n[ 1 ], n[ 2 ] );
		const nn = n.map( ( v ) => v / l );
		P.push( [ nn[ 0 ], nn[ 1 ], nn[ 2 ], ( nn[ 0 ] * p[ 0 ] + nn[ 1 ] * p[ 1 ] + nn[ 2 ] * p[ 2 ] ) * r ] );

	};

	const up = ( phi, a ) => [ Math.sin( a ) * Math.cos( phi ), Math.cos( a ), Math.sin( a ) * Math.sin( phi ) ];
	const dn = ( phi, a ) => [ Math.sin( a ) * Math.cos( phi ), - Math.cos( a ), Math.sin( a ) * Math.sin( phi ) ];

	add( [ 0, 1, 0 ], [ 0, yt, 0 ] ); // table
	for ( let k = 0; k < 8; k ++ ) {

		const phi = k * Math.PI / 4;
		add( up( phi, crownA ), [ Math.cos( phi ), gth, Math.sin( phi ) ] ); // bezel
		const ps = phi + Math.PI / 8;
		const rt = 0.56 * Math.cos( Math.PI / 8 );
		add( up( ps, starA ), [ rt * Math.cos( ps ), yt, rt * Math.sin( ps ) ] ); // star
		add( dn( phi, pavA ), [ Math.cos( phi ), - gth, Math.sin( phi ) ] ); // pavilion main

	}

	for ( let k = 0; k < 16; k ++ ) {

		const ps = k * Math.PI / 8 + Math.PI / 16;
		add( up( ps, ugA ), [ Math.cos( ps ), gth, Math.sin( ps ) ] ); // upper girdle
		add( dn( ps, lgA ), [ Math.cos( ps ), - gth, Math.sin( ps ) ] ); // lower girdle
		const pg = k * Math.PI / 8;
		add( [ Math.cos( pg ), 0, Math.sin( pg ) ], [ Math.cos( pg ), 0, Math.sin( pg ) ] ); // girdle

	}

	return { planes: P, radius: r * Math.hypot( 1, Math.tan( pavA ) + gth ) * 1.02 };

}

// Glass spheres for the "spheres" preset: [x, y, z, r]
export const SPHERES = [
	[ - 0.55, 0.36, 0.25, 0.36 ],
	[ 0.42, 0.24, - 0.25, 0.24 ],
	[ 0.18, 0.13, 0.62, 0.13 ],
];

// ── presets ──
// beam: target point, direction (normalised later), slit half sizes (w across, h along "up").
export const PRESETS = {
	prism: {
		id: 0, plinth: true, objPos: [ 0, PLINTH.h, 0.9 ], yaw: 0.72, pitch: 0, bound: 0.4,
		beam: { target: [ 0, PLINTH.h + 0.23, 0.97 ], dir: [ 1, 0, - 0.103 ], hw: 0.011, hh: 0.075 },
		cam: { pos: [ 2.2, 1.55, 3.9 ], target: [ - 0.2, 1.05, - 0.8 ] },
	},
	gem: {
		id: 1, plinth: true, objPos: [ 0, PLINTH.h + 0.24, 0.6 ], yaw: 0.25, pitch: - 0.55, bound: 0.3,
		beam: { target: [ 0, PLINTH.h + 0.24, 0.6 ], dir: [ 0.75, - 0.5, - 0.2 ], hw: 0.07, hh: 0.07 },
		cam: { pos: [ 0.9, 1.5, 3.9 ], target: [ 0.0, 1.0, - 0.4 ] },
	},
	spheres: {
		id: 2, plinth: false, objPos: [ 0, 0, 0 ], yaw: 0, pitch: 0, bound: 1.0,
		beam: { target: [ 0, 0.2, 0.2 ], dir: [ 0.8, - 0.45, - 0.25 ], hw: 0.55, hh: 0.4 },
		cam: { pos: [ 0.6, 1.9, 3.6 ], target: [ 0.0, 0.25, 0.0 ] },
	},
};

// Caustic light-map atlas. Surfaces are addressed by material id (1..11).
// Each entry: id, extent U (m), extent V (m), texels per metre weight.
export function buildAtlas( density ) {

	const R = ROOM, W = R.x1 - R.x0, D = R.z1 - R.z0, H = R.y1;
	const pw = 2 * PLINTH.w, ph = PLINTH.h;
	const surf = [
		{ id: 1, U: W, V: H, k: 1.0 }, // back wall z0
		{ id: 2, U: D, V: H, k: 0.55 }, // left wall x0
		{ id: 3, U: D, V: H, k: 0.7 }, // right wall x1
		{ id: 4, U: W, V: D, k: 0.4 }, // ceiling
		{ id: 5, U: W, V: H, k: 0.4 }, // front wall z1
		{ id: 6, U: W, V: D, k: 0.9 }, // floor
		{ id: 7, U: pw, V: pw, k: 2.0 }, // plinth top
		{ id: 8, U: pw, V: ph, k: 1.0 }, // plinth +x
		{ id: 9, U: pw, V: ph, k: 1.0 }, // plinth -x
		{ id: 10, U: pw, V: ph, k: 1.0 }, // plinth +z
		{ id: 11, U: pw, V: ph, k: 1.0 }, // plinth -z
	];
	for ( const s of surf ) {

		s.w = Math.max( 4, Math.round( s.U * density * s.k ) );
		s.h = Math.max( 4, Math.round( s.V * density * s.k ) );

	}

	// shelf packing
	const atlasW = Math.max( ...surf.map( ( s ) => s.w ) );
	const order = [ ...surf ].sort( ( a, b ) => b.h - a.h );
	let x = 0, y = 0, rowH = 0;
	for ( const s of order ) {

		if ( x + s.w > atlasW ) {

			x = 0; y += rowH; rowH = 0;

		}

		s.x = x; s.y = y; x += s.w; rowH = Math.max( rowH, s.h );

	}

	const atlasH = y + rowH;
	return { surf, W: atlasW, H: atlasH };

}
