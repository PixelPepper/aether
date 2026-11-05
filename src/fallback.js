// No-WebGPU fallback: what the visitor is missing, which browsers work, and a gallery of stills.
// Deliberately free of three.js so it renders instantly even where WebGPU (or the GPU) is absent.
// Stills live at public/stills/<id>.jpg; missing ones are hidden, so the gallery degrades to a list of titles.

const CHAPTERS = [
	[ 'quantum', 'Fluctuation', 'One particle, many paths: the Schrödinger equation solved live with GPU Fourier transforms.' ],
	[ 'cosmicweb', 'Cosmic Web', 'Dark matter pulled by gravity from a near-uniform field into filaments, sheets, voids and halos.' ],
	[ 'collision', 'Collision', 'Two spiral galaxies in a live N-body merger, tidal tails flung from their disks.' ],
	[ 'nebula', 'Stellar Nursery', 'Ultraviolet light from a newborn cluster erodes towering columns of molecular gas.' ],
	[ 'stellar', 'Ignition', 'Buoyant plasma rising from the photosphere of a young star, every voxel a Planck blackbody.' ],
	[ 'atmosphere', 'Atmosphere', 'Rayleigh, Mie and ozone light transport through 100 km of air over a procedural planet.' ],
	[ 'erosion', 'Deep Time', 'Rain carves a rising mountain range into rivers, fans and deltas, a million years every few seconds.' ],
	[ 'ocean', 'Primordial Sea', 'A material-point-method ocean whose strained water flashes like plankton.' ],
	[ 'morpho', 'Morphogenesis', 'Turing’s reaction–diffusion grown in three dimensions into coral, folds and dividing cells.' ],
	[ 'genesis', 'Emergence', 'A quarter-million particles obeying one asymmetric table of likes and dislikes.' ],
	[ 'crystal', 'Ice', 'A snow crystal freezing out of supersaturated vapour, one lattice site at a time.' ],
	[ 'aurora', 'Aurora', 'Solar-wind electrons spiral down Earth’s magnetic field and set the upper air glowing.' ],
	[ 'silk', 'Silk', 'Sheets of shot silk under XPBD dynamics, billowing in curl-noise wind.' ],
	[ 'prism', 'Spectrum', 'A spectral path tracer: white light, Sellmeier glass, every rainbow computed per wavelength.' ],
	[ 'horizon', 'Event Horizon', 'A spinning black hole, ray-traced per pixel through a Novikov–Thorne accretion disk.' ],
];

const REASONS = {
	'no-webgpu': 'Your browser does not expose WebGPU.',
	'init-failed': 'Your browser has WebGPU, but it could not start a GPU device.',
	'no-adapter': 'Your browser has WebGPU, but no suitable GPU adapter was found.',
};

const esc = ( s ) => String( s ).replace( /[&<>"]/g, ( c ) => ( { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' } )[ c ] );

export function showFallback( reason = 'no-webgpu', detail = '' ) {

	document.getElementById( 'boot' )?.remove();
	document.getElementById( 'hud' )?.remove();
	document.getElementById( 'fade' )?.remove();
	document.getElementById( 'gl' )?.remove();
	document.documentElement.classList.add( 'fallback-mode' );

	const root = document.getElementById( 'fallback' );
	const base = import.meta.env.BASE_URL || './';

	root.innerHTML = `
		<header class="fb-head">
			<div class="mark">AETHER</div>
			<div class="tag">from quantum foam to event horizon</div>
		</header>
		<section class="fb-intro">
			<h1>This journey runs on your GPU.</h1>
			<p class="fb-reason">${esc( REASONS[ reason ] || REASONS[ 'no-webgpu' ] )}</p>
			<p>Aether is fifteen physical simulations — quantum wave packets, galaxy mergers, erosion, fluids, light, a black hole —
			computed live, every frame, with WebGPU compute shaders. There is no video here: without WebGPU there is nothing to compute it with.</p>
			<div class="fb-browsers" aria-label="Supported browsers">
				<div><b>Chrome / Edge</b><span>113 or newer</span></div>
				<div><b>Safari</b><span>26 or newer</span></div>
				<div><b>Firefox</b><span>141 or newer</span></div>
			</div>
			<p class="fb-note">A desktop or laptop with a dedicated or recent integrated GPU works best. If your browser is recent, check that hardware acceleration is enabled.</p>
			${detail ? `<p class="fb-detail">${esc( detail )}</p>` : ''}
		</section>
		<section class="fb-gallery" aria-label="Chapters">
			${CHAPTERS.map( ( [ id, title, sub ], i ) => `
			<figure class="fb-card">
				<div class="fb-img"><img src="${base}stills/${id}.jpg" alt="" loading="lazy" decoding="async"></div>
				<figcaption>
					<span class="num">${String( i + 1 ).padStart( 2, '0' )}</span>
					<h2>${esc( title )}</h2>
					<p>${esc( sub )}</p>
				</figcaption>
			</figure>` ).join( '' )}
		</section>`;

	// stills are added later: a missing one collapses its frame instead of showing a broken image
	root.querySelectorAll( '.fb-img img' ).forEach( ( img ) => {

		const fail = () => img.parentElement.classList.add( 'missing' );
		img.addEventListener( 'error', fail, { once: true } );
		img.addEventListener( 'load', () => img.parentElement.classList.add( 'loaded' ), { once: true } );
		if ( img.complete && img.naturalWidth === 0 && img.src ) fail();

	} );

	root.hidden = false;

}
