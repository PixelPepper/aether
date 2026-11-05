// Genesis — interaction rules, ecosystems and palettes.
//
// The attraction matrix is always stored as MAX_SPECIES × MAX_SPECIES floats (row = the particle feeling the force,
// column = the neighbour exerting it). Values live in [-1, 1]: positive = "I am drawn towards you".
// The matrix is deliberately asymmetric (a[i][j] != a[j][i]): that broken Newton's third law is what lets
// particle life produce chasing, orbiting and self-propelling structures instead of settling into crystals.

export const MAX_SPECIES = 8;

// Small, fast, seedable PRNG (mulberry32).
export function rng( seed ) {

	let a = seed >>> 0;
	return () => {

		a = ( a + 0x6D2B79F5 ) >>> 0;
		let t = a;
		t = Math.imul( t ^ ( t >>> 15 ), t | 1 );
		t ^= t + Math.imul( t ^ ( t >>> 7 ), t | 61 );
		return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

	};

}

// Random asymmetric matrix. Self-affinity is biased positive (cohesive "tissue"), cross terms uniform in [-1, 1].
export function randomMatrix( seed ) {

	const r = rng( seed );
	const m = new Float32Array( MAX_SPECIES * MAX_SPECIES );
	for ( let i = 0; i < MAX_SPECIES; i ++ ) {

		for ( let j = 0; j < MAX_SPECIES; j ++ ) {

			m[ i * MAX_SPECIES + j ] = i === j ? 0.15 + 0.85 * r() : r() * 2 - 1;

		}

	}

	return m;

}

function fromRows( rows ) {

	const m = new Float32Array( MAX_SPECIES * MAX_SPECIES );
	rows.forEach( ( row, i ) => row.forEach( ( v, j ) => m[ i * MAX_SPECIES + j ] = v ) );
	return m;

}

// Hand-built "cell" ecology: nuclei (0,3) cohere, membranes (1,4) wrap them and repel each other,
// cytoplasm (2) fills the gap, predators (5) hunt membranes which flee them.
function cellsMatrix() {

	return fromRows( [
		//  nuc   mem   cyto  nuc2  mem2  pred
		[ 0.95, 0.10, 0.20, - 0.35, 0.00, - 0.20 ],
		[ 0.65, - 0.30, 0.15, 0.00, - 0.40, - 0.60 ],
		[ 0.45, 0.40, 0.25, 0.00, 0.10, 0.00 ],
		[ - 0.35, 0.00, 0.00, 0.90, 0.15, - 0.20 ],
		[ 0.00, - 0.40, 0.10, 0.65, - 0.30, - 0.60 ],
		[ 0.10, 0.70, 0.20, 0.10, 0.70, 0.35 ],
	] );

}

// Cyclic pursuit: each species chases the next and flees the previous — orbiting, spiralling clusters.
function cycleMatrix( n ) {

	const m = new Float32Array( MAX_SPECIES * MAX_SPECIES );
	for ( let i = 0; i < n; i ++ ) {

		m[ i * MAX_SPECIES + i ] = 0.7;
		m[ i * MAX_SPECIES + ( i + 1 ) % n ] = 0.85;
		m[ i * MAX_SPECIES + ( i + n - 1 ) % n ] = - 0.55;

	}

	return m;

}

// Ecosystems. `seed` drives random rules; `matrix` (fn) overrides with designed rules.
// rMax is relative to the base interaction radius, friction is the velocity half-life in sim seconds.
export const PRESETS = [
	{ name: 'Abyssal bloom', seed: 1, species: 6, rMax: 1.0, friction: 0.04, force: 1.0, palette: 'abyss' },
	{ name: 'Membranes', matrix: cellsMatrix, species: 6, rMax: 1.0, friction: 0.05, force: 1.0, palette: 'abyss', seed: 7 },
	{ name: 'Orbiters', matrix: () => cycleMatrix( 5 ), species: 5, rMax: 1.0, friction: 0.04, force: 1.0, palette: 'iris', seed: 3 },
	{ name: 'Tidepool', seed: 2, species: 7, rMax: 1.0, friction: 0.04, force: 1.0, palette: 'reef' },
	{ name: 'Siphonophore', seed: 3, species: 8, rMax: 1.0, friction: 0.04, force: 1.0, palette: 'iris' },
];

// Palettes: per species a deep "core" tint (dense interior) and a bright "rim" tint (sparse, fast edges).
// Linear-ish HDR-friendly values; they are multiplied by intensity in the shader.
const hex = ( h ) => [ ( ( h >> 16 ) & 255 ) / 255, ( ( h >> 8 ) & 255 ) / 255, ( h & 255 ) / 255 ].map( ( c ) => Math.pow( c, 2.2 ) );

export const PALETTES = {
	// deep-sea bioluminescence: teal → azure → indigo → violet, one warm amber lure, one pale ghost
	abyss: [
		[ 0x0a8f8a, 0x5ff7e0 ], // teal
		[ 0x1846c9, 0x7fb4ff ], // azure
		[ 0x5a2ad1, 0xb69bff ], // violet
		[ 0xb3246f, 0xff8fc4 ], // magenta
		[ 0xc06a12, 0xffcf7a ], // amber lure
		[ 0x0d6bb0, 0x9ff0ff ], // ice
		[ 0x2a9d5c, 0xb8ffcf ], // sea-glass
		[ 0x7a3fa8, 0xf0c8ff ], // lilac
	],
	// iridescent cells: thin-film blues, pinks and golds
	iris: [
		[ 0x2350d8, 0x8fd1ff ],
		[ 0xa0287f, 0xff9ad6 ],
		[ 0x1f9a8a, 0x9effe3 ],
		[ 0xb8781a, 0xffe2a0 ],
		[ 0x5b33c8, 0xc8b0ff ],
		[ 0x147ac0, 0xa8fbff ],
		[ 0xc23f3f, 0xffb59a ],
		[ 0x3a8f2e, 0xd6ff9e ],
	],
	// coral reef at night: coral, rose, sea-foam, kelp-gold
	reef: [
		[ 0xc2412b, 0xffa58a ],
		[ 0x0f8f9a, 0x8ff6ff ],
		[ 0xa13a88, 0xffa6e8 ],
		[ 0xb08a1c, 0xfff0a0 ],
		[ 0x2a5bd0, 0x9cc0ff ],
		[ 0x1b9a63, 0xa6ffd0 ],
		[ 0x6c3ac2, 0xd2b8ff ],
		[ 0xc06030, 0xffd0a0 ],
	],
};

export function paletteColors( name ) {

	const p = PALETTES[ name ] || PALETTES.abyss;
	return { core: p.map( ( c ) => hex( c[ 0 ] ) ), rim: p.map( ( c ) => hex( c[ 1 ] ) ) };

}

export function presetMatrix( preset, seed ) {

	return preset.matrix ? preset.matrix() : randomMatrix( seed ?? preset.seed );

}
