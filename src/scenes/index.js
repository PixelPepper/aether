// Chapter registry. Each entry lazily imports its module so chapters are code-split.
// Order here is the order in the UI. Each module's default export must follow CONTRACT.md.
export const chapters = [
	{ id: 'genesis', label: 'Genesis', load: () => import( './genesis.js' ) },
	{ id: 'nebula', label: 'Nebula', load: () => import( './nebula.js' ) },
	{ id: 'collision', label: 'Collision', load: () => import( './collision.js' ) },
	{ id: 'ocean', label: 'Ocean', load: () => import( './ocean.js' ) },
	{ id: 'horizon', label: 'Horizon', load: () => import( './horizon.js' ) },
];
