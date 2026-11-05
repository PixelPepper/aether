// Chapter registry. Each entry lazily imports its module so chapters are code-split.
// Order here is the order in the UI: a journey from quantum fluctuation to event horizon.
// Each module's default export must follow CONTRACT.md.
export const chapters = [
	{ id: 'quantum', label: 'Fluctuation', load: () => import( './quantum.js' ) },
	{ id: 'cosmicweb', label: 'Cosmic Web', load: () => import( './cosmicweb.js' ) },
	{ id: 'collision', label: 'Collision', load: () => import( './collision.js' ) },
	{ id: 'nebula', label: 'Nursery', load: () => import( './nebula.js' ) },
	{ id: 'stellar', label: 'Ignition', load: () => import( './stellar.js' ) },
	{ id: 'atmosphere', label: 'Atmosphere', load: () => import( './atmosphere.js' ) },
	{ id: 'erosion', label: 'Deep Time', load: () => import( './erosion.js' ) },
	{ id: 'ocean', label: 'Sea', load: () => import( './ocean.js' ) },
	{ id: 'morpho', label: 'Morphogenesis', load: () => import( './morpho.js' ) },
	{ id: 'genesis', label: 'Emergence', load: () => import( './genesis.js' ) },
	{ id: 'crystal', label: 'Ice', load: () => import( './crystal.js' ) },
	{ id: 'aurora', label: 'Aurora', load: () => import( './aurora.js' ) },
	{ id: 'silk', label: 'Silk', load: () => import( './silk.js' ) },
	{ id: 'prism', label: 'Spectrum', load: () => import( './prism.js' ) },
	{ id: 'horizon', label: 'Horizon', load: () => import( './horizon.js' ) },
];
