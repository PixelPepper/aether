// Failure handling for the shell: GPU device loss and chapters that throw at runtime.
// Both surfaces are created on demand so index.html stays untouched.

let lostShown = false;

// A lost device (driver reset, GPU process crash, tab backgrounded too long on some mobiles) leaves the canvas frozen.
// Show a calm overlay with a reload instead. Intentional destruction (reason 'destroyed') is ignored.
export function watchDeviceLoss( renderer, { shot = false, onLost } = {} ) {

	const device = renderer.backend?.device;
	if ( ! device?.lost ) return;

	device.lost.then( ( info ) => {

		if ( info?.reason === 'destroyed' || lostShown ) return;
		lostShown = true;
		console.warn( 'WebGPU device lost:', info?.reason, info?.message );
		if ( shot ) window.__error = `device lost: ${info?.message || info?.reason || 'unknown'}`;
		onLost?.( info );
		showDeviceLost( info );

	} );

}

function showDeviceLost( info ) {

	const el = document.createElement( 'div' );
	el.id = 'lost';
	el.setAttribute( 'role', 'alertdialog' );
	el.setAttribute( 'aria-labelledby', 'lost-title' );
	el.innerHTML = `
		<div class="lost-card">
			<div class="mark">AETHER</div>
			<h2 id="lost-title">The GPU stepped away.</h2>
			<p>Your graphics device was reset — this happens after a driver update, a GPU crash elsewhere, or when the system reclaims memory.
			Nothing is broken; the simulations just need to start again.</p>
			<button class="enter primary" type="button">Reload</button>
			<p class="lost-detail"></p>
		</div>`;
	el.querySelector( '.lost-detail' ).textContent = info?.message || '';
	el.querySelector( 'button' ).onclick = () => location.reload();
	document.body.appendChild( el );
	requestAnimationFrame( () => el.classList.add( 'on' ) );
	el.querySelector( 'button' ).focus();

}

// Inline notice for a chapter that threw in create() or update(): the rest of the app keeps running.
export function showChapterNotice( title, err, onNext ) {

	hideChapterNotice();
	const el = document.createElement( 'div' );
	el.id = 'ch-error';
	el.setAttribute( 'role', 'status' );
	el.innerHTML = `
		<div><b></b> stopped with an error on this device.</div>
		<code></code>
		<button type="button">Next chapter →</button>`;
	el.querySelector( 'b' ).textContent = title;
	el.querySelector( 'code' ).textContent = String( err?.message || err ).slice( 0, 160 );
	el.querySelector( 'button' ).onclick = onNext;
	document.getElementById( 'hud' )?.appendChild( el );

}

export function hideChapterNotice() {

	document.getElementById( 'ch-error' )?.remove();

}
