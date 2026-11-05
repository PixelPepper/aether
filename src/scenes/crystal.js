// Chapter 11 — Ice.
// A snow crystal grown in real time by the Gravner–Griffeath mesoscopic lattice model (vapour diffusion, freezing into a
// quasi-liquid boundary layer, anisotropic attachment, melting) in compute shaders on a hexagonal lattice, then rendered
// as a refracting, dispersive ice slab whose relief is the accumulated crystal mass. See crystal/sim.js and crystal/ice.js.
import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import { SnowLattice } from './crystal/sim.js';
import { PRESETS, KEYS, layerLabel, jitter } from './crystal/presets.js';
import { createStudio, createIceMaterial, createHaloMaterial } from './crystal/ice.js';
import { createBokeh } from './crystal/bokeh.js';

const TIERS = [
	{ N: 160, bake: 320, mesh: 160, maxSteps: 8, bokeh: 70 },
	{ N: 512, bake: 768, mesh: 256, maxSteps: 24, bokeh: 180 },
	{ N: 768, bake: 1024, mesh: 384, maxSteps: 64, bokeh: 320 },
];

const GROW_TIME = 34; // seconds per flake at speed 1
const HOLD_TIME = 3.5; // admire the finished flake
const FADE_TIME = 3.2; // crossfade to the next one
const CYCLE = 'Cycle (each flake different)';

function mulberry32( a ) {

	return () => {

		a |= 0; a = a + 0x6D2B79F5 | 0;
		let t = Math.imul( a ^ a >>> 15, 1 | a );
		t = t + Math.imul( t ^ t >>> 7, 61 | t ) ^ t;
		return ( ( t ^ t >>> 14 ) >>> 0 ) / 4294967296;

	};

}

export default {
	id: 'crystal',
	title: 'Ice',
	subtitle: 'A snow crystal freezing out of supersaturated vapour, one lattice site at a time: diffusion, a quasi-liquid skin and anisotropic attachment on a hexagonal grid (Gravner–Griffeath), every flake a record of the cloud layers it fell through.',

	async create( ctx ) {

		const { renderer, camera, controls, params, tier } = ctx;
		const cfg = { ...TIERS[ Math.max( 0, Math.min( 2, tier ) ) ] };
		if ( params.has( 'lattice' ) ) {

			cfg.N = Math.max( 64, parseInt( params.get( 'lattice' ) ) & ~ 1 );
			cfg.bake = Math.max( cfg.bake, Math.round( cfg.N * 1.33 / 8 ) * 8 );
			cfg.mesh = Math.max( cfg.mesh, cfg.N >> 1 );

		}
		const rnd = mulberry32( ctx.shot ? 1234 + ( parseInt( params.get( 'seed' ) || '0' ) ) : ( Math.random() * 1e9 ) | 0 );

		const scene = new THREE.Scene();
		const studio = createStudio();
		const S = studio.S;
		S.texel.value = 1 / cfg.bake;
		S.glintRes.value = Math.round( cfg.bake / 2 );
		scene.backgroundNode = studio.background;

		// ── camera: slightly above, looking at the flake whose normal precesses ~20° off the view axis ──
		camera.position.set( 0, 0.32, 2.65 );
		controls.target.set( 0, 0, 0 );
		controls.enablePan = false;
		controls.minDistance = 1.1;
		controls.maxDistance = 6;
		controls.autoRotate = false;
		controls.update();

		// ── simulation ──
		const lattice = new SnowLattice( renderer, { N: cfg.N, bakeSize: cfg.bake } );
		const RD = lattice.RD;
		const R_TARGET = Math.floor( RD * 0.88 );
		lattice.u.sigma.value = params.has( 'sigma' ) ? parseFloat( params.get( 'sigma' ) ) : 0.0008;

		// ── flake slots (current + retiring) ──
		const geo = new THREE.PlaneGeometry( 2, 2, cfg.mesh, cfg.mesh );
		const haloGeo = new THREE.PlaneGeometry( 2, 2, 1, 1 );
		const slots = [ 0, 1 ].map( ( k ) => {

			const P = { fade: uniform( 0 ), now: uniform( 0 ), seed: uniform( 0 ), ring: uniform( 0 ) };
			const mat = createIceMaterial( studio, lattice.textures[ k ], P, params.get( 'dbgview' ) || '' );
			const halo = createHaloMaterial( studio, lattice.textures[ k ], P );
			const group = new THREE.Group();
			const top = new THREE.Mesh( geo, mat );
			const bottom = new THREE.Mesh( geo, mat );
			bottom.scale.z = - 1; // mirrored copy: the slab's lower face
			const haloMesh = new THREE.Mesh( haloGeo, halo );
			haloMesh.renderOrder = 1;
			for ( const m of [ top, bottom, haloMesh ] ) m.frustumCulled = false;
			const spinner = new THREE.Group();
			spinner.add( top, bottom, haloMesh );
			group.add( spinner );
			group.visible = false;
			scene.add( group );
			return {
				k, P, mat, halo, group, spinner, state: 'idle', t: 0, scale: 3, spin0: 0, prec0: 0, tilt: 0.36,
				retireFrom: new THREE.Vector3(),
			};

		} );

		const bokeh = createBokeh( cfg.bokeh );
		scene.add( bokeh.sprite );
		const hide = ( params.get( 'hide' ) || '' ).split( ',' );
		if ( hide.includes( 'bokeh' ) ) bokeh.sprite.visible = false;
		if ( hide.includes( 'bg' ) ) scene.backgroundNode = null;
		for ( const s of slots ) {

			if ( hide.includes( 'ice' ) ) s.spinner.children[ 0 ].visible = s.spinner.children[ 1 ].visible = false;
			if ( hide.includes( 'halo' ) ) s.spinner.children[ 2 ].visible = false;

		}

		// ── state ──
		const st = {
			presetIdx: 0, preset: PRESETS[ 0 ], params: { ...PRESETS[ 0 ] }, target: { ...PRESETS[ 0 ] },
			growT: 0, phase: 'grow', holdT: 0, layerT: 9, label: layerLabel( PRESETS[ 0 ] ),
			vEst: 0.02, acc: 0, rPrev: 0, sPrev: 0, flash: 0, ringR: 0, pollT: 0, statT: 0, stepsLast: 0, active: 0,
			lastPresetIdx: - 1, flakes: 0,
		};
		const gp = {
			preset: CYCLE, rho: 0.6, beta: 1.6, kappa: 0.0025,
			drift: params.has( 'drift' ) ? params.get( 'drift' ) !== '0' : true,
			speed: params.has( 'speed' ) ? parseFloat( params.get( 'speed' ) ) : 1,
			newFlake: () => newFlake(),
		};

		const applyParams = () => {

			lattice.setParams( st.params );
			lattice.setRho( st.params.rho );
			gp.rho = st.params.rho; gp.beta = st.params.beta; gp.kappa = st.params.kappa;

		};

		const pickPreset = () => {

			if ( st.flakes === 0 && ( params.has( 'preset' ) || ctx.shot ) ) return parseInt( params.get( 'preset' ) || '0' ) % PRESETS.length;
			if ( gp.preset !== CYCLE ) return PRESETS.findIndex( ( p ) => p.name === gp.preset );
			let i;
			do {

				i = Math.floor( rnd() * PRESETS.length );

			} while ( i === st.lastPresetIdx && PRESETS.length > 1 );
			return i;

		};

		// ── caption for cloud-layer changes (makes the click obvious) ──
		let caption = null, captionTimer = 0;
		if ( ! ctx.shot ) {

			caption = document.createElement( 'div' );
			caption.style.cssText = 'position:fixed;left:50%;top:15%;transform:translateX(-50%);pointer-events:none;'
				+ 'font:400 12px/1.6 "JetBrains Mono",monospace;letter-spacing:.18em;text-transform:uppercase;'
				+ 'color:rgba(205,228,255,.92);text-shadow:0 0 18px rgba(80,160,255,.6);opacity:0;transition:opacity .6s ease;'
				+ 'text-align:center;white-space:nowrap;z-index:5';
			document.body.appendChild( caption );

		}

		const showCaption = ( text ) => {

			if ( ! caption ) return;
			caption.textContent = text;
			caption.style.opacity = '1';
			captionTimer = 3.2;

		};

		function newFlake() {

			// retire the current flake: it falls away and sublimates while the next one nucleates
			const old = slots[ st.active ];
			if ( old.state === 'grow' ) {

				old.state = 'retire';
				old.t = 0;
				old.retireFrom.copy( old.group.position );

			}

			st.active = 1 - st.active;
			lattice.current = st.active;
			const slot = slots[ st.active ];

			st.presetIdx = pickPreset();
			st.lastPresetIdx = st.presetIdx;
			st.preset = PRESETS[ st.presetIdx ];
			st.params = jitter( st.preset, rnd, st.flakes === 0 && ctx.shot ? 0 : 0.04 );
			st.target = { ...st.params };
			st.label = layerLabel( st.preset );
			st.growT = 0;
			st.phase = 'grow';
			st.holdT = 0;
			st.layerT = 8 + rnd() * 5;
			st.vEst = st.preset.rate;
			st.acc = 0;
			st.rPrev = 0;
			st.sPrev = 0;
			st.flash = 0;
			st.flakes ++;
			lattice.reset( st.params );
			gp.rho = st.params.rho; gp.beta = st.params.beta; gp.kappa = st.params.kappa;

			slot.state = 'grow';
			slot.t = 0;
			slot.P.fade.value = 0;
			slot.P.now.value = 0;
			slot.P.ring.value = 0;
			slot.P.seed.value = Math.floor( rnd() * 997 );
			slot.spin0 = rnd() * Math.PI * 2;
			slot.prec0 = rnd() * Math.PI * 2;
			slot.scale = 3.0;
			slot.group.position.set( 0, 0, 0 );
			slot.group.visible = true;
			lattice.bake();

		}

		// a different cloud layer: supersaturation / temperature jump → new growth regime mid-flake
		function cloudJump( strength = 1 ) {

			const cur = st.params;
			const dendritic = cur.rho > 0.47;
			const pool = PRESETS.map( ( p, i ) => i ).filter( ( i ) => i !== st.presetIdx && ( ( PRESETS[ i ].rho > 0.47 ) !== dendritic ) );
			const i = pool.length ? pool[ Math.floor( rnd() * pool.length ) ] : Math.floor( rnd() * PRESETS.length );
			const p = jitter( PRESETS[ i ], rnd );
			st.presetIdx = i;
			st.params = { ...p };
			st.target = { ...p };
			st.label = layerLabel( PRESETS[ i ] );
			applyParams();
			st.vEst = PRESETS[ i ].rate;
			st.rPrev = lattice.radius; st.sPrev = lattice.steps;
			st.flash = strength;
			st.ringR = lattice.radius / RD;
			S.flashCol.value.setRGB( ...( p.rho > 0.47 ? [ 1.0, 0.72, 0.4 ] : [ 0.45, 0.62, 1.0 ] ) );
			st.layerT = 8 + rnd() * 5;
			showCaption( `cloud layer · ${st.label}` );

		}

		// drift: glide towards a new layer every ~10 s
		function drift( dt ) {

			st.layerT -= dt;
			if ( st.layerT <= 0 ) {

				st.layerT = 7 + rnd() * 6;
				let i;
				do {

					i = Math.floor( rnd() * PRESETS.length );

				} while ( i === st.presetIdx );
				st.presetIdx = i;
				st.target = jitter( PRESETS[ i ], rnd );
				st.label = layerLabel( PRESETS[ i ] );
				st.vEst = Math.max( st.vEst * 0.5, Math.min( st.vEst * 2, PRESETS[ i ].rate ) );
				st.flash = Math.max( st.flash, 0.35 );
				st.ringR = lattice.radius / RD;
				S.flashCol.value.setRGB( ...( PRESETS[ i ].rho > 0.47 ? [ 1.0, 0.72, 0.4 ] : [ 0.45, 0.62, 1.0 ] ) );
				showCaption( `drifting into · ${st.label}` );

			}

			const k = 1 - Math.exp( - dt / 2.0 );
			let changed = false;
			for ( const key of KEYS ) {

				const d = st.target[ key ] - st.params[ key ];
				if ( Math.abs( d ) > 1e-7 ) {

					st.params[ key ] += d * k; changed = true;

				}

			}

			if ( changed ) applyParams();

		}

		// growth-rate controller: choose steps/frame so the flake reaches R_TARGET in ~GROW_TIME/speed seconds
		function stepsForFrame( dt ) {

			const T = GROW_TIME / Math.max( 0.05, gp.speed );
			const f = ( x ) => R_TARGET * Math.pow( Math.min( 1, Math.max( 0, x ) / T ), 0.85 );
			const want = f( st.growT + dt ) - f( st.growT );
			const err = f( st.growT ) - lattice.radius;
			const cells = Math.max( 0, want + err * dt / 1.2 );
			st.acc += cells / Math.max( st.vEst, 1e-4 );
			const n = Math.min( cfg.maxSteps, Math.floor( st.acc ) );
			st.acc = Math.min( st.acc - n, 2 );
			return n;

		}

		function onRadius( res ) {

			if ( ! res ) return;
			const dr = res.radius - st.rPrev, ds = res.steps - st.sPrev;
			if ( dr >= 3 && ds > 0 ) {

				st.vEst = st.vEst * 0.6 + ( dr / ds ) * 0.4;
				st.rPrev = res.radius; st.sPrev = res.steps;

			} else if ( ds > 6 / Math.max( st.vEst, 1e-4 ) ) {

				st.vEst *= 0.75; // slower than we thought
				st.sPrev = res.steps; st.rPrev = res.radius;

			}

		}

		// ── pointer: drag orbits (OrbitControls), a click changes the cloud layer ──
		const canvas = renderer.domElement;
		let down = null;
		const onDown = ( e ) => {

			down = { x: e.clientX, y: e.clientY, t: performance.now() };

		};

		const onUp = ( e ) => {

			if ( ! down ) return;
			const moved = Math.hypot( e.clientX - down.x, e.clientY - down.y );
			if ( moved < 6 && performance.now() - down.t < 350 ) cloudJump();
			down = null;

		};

		canvas.addEventListener( 'pointerdown', onDown );
		canvas.addEventListener( 'pointerup', onUp );

		// ── GUI ──
		const gui = ctx.gui;
		gui.add( gp, 'preset', [ CYCLE, ...PRESETS.map( ( p ) => p.name ) ] ).name( 'preset' ).onChange( () => newFlake() );
		gui.add( gp, 'rho', 0.25, 0.95, 0.001 ).name( 'supersaturation ρ' ).listen().onChange( ( v ) => {

			st.params.rho = st.target.rho = v; lattice.setRho( v );

		} );
		gui.add( gp, 'beta', 1.0, 3.0, 0.01 ).name( 'anisotropy β' ).listen().onChange( ( v ) => {

			st.params.beta = st.target.beta = v; lattice.setParams( st.params );

		} );
		gui.add( gp, 'kappa', 0.0, 0.2, 0.0005 ).name( 'direct freezing κ' ).listen().onChange( ( v ) => {

			st.params.kappa = st.target.kappa = v; lattice.setParams( st.params );

		} );
		gui.add( gp, 'drift' ).name( 'drifting conditions' );
		gui.add( gp, 'speed', 0.25, 3, 0.05 ).name( 'growth speed' );
		gui.add( gp, 'newFlake' ).name( 'nucleate new flake' );

		// ── first flake (+ optional pre-growth for captures / debugging) ──
		newFlake();
		{

			const warpSteps = params.has( 'warp' ) ? parseInt( params.get( 'warp' ) ) : 0;
			const warpFrac = params.has( 'warpr' ) ? parseFloat( params.get( 'warpr' ) ) : ( ctx.shot && ! warpSteps ? 0.6 : 0 );
			if ( warpSteps > 0 || warpFrac > 0 ) {

				const chunk = tier === 0 ? 100 : 250;
				const maxSteps = warpSteps > 0 ? warpSteps : ( tier === 0 ? 4000 : 60000 );
				const t0 = performance.now();
				const T = GROW_TIME / Math.max( 0.05, gp.speed );
				let done = 0;
				while ( done < maxSteps ) {

					const n = Math.min( chunk, maxSteps - done );
					lattice.u.time.value = 1 + st.growT;
					lattice.step( n );
					done += n;
					const res = await lattice.poll();
					onRadius( res );
					st.growT = T * Math.pow( Math.min( 1, lattice.radius / R_TARGET ), 1 / 0.85 );
					if ( warpFrac > 0 && lattice.radius >= warpFrac * R_TARGET ) break;

				}

				if ( params.has( 'dbg' ) ) console.warn( `[crystal] warp ${done} steps → r=${lattice.radius}/${R_TARGET} in ${( performance.now() - t0 ).toFixed( 0 )} ms` );

				lattice.bake();

			}

		}

		const qTilt = new THREE.Quaternion(), qSpin = new THREE.Quaternion(), Z = new THREE.Vector3( 0, 0, 1 ), nrm = new THREE.Vector3();

		function orient( slot, t ) {

			if ( params.has( 'dbgcam' ) ) {

				slot.group.quaternion.identity(); slot.scale = 1; slot.group.scale.setScalar( 1 ); return;

			}

			const tilt = 0.36 + 0.07 * Math.sin( 0.19 * t + slot.prec0 );
			const phi = slot.prec0 + 0.11 * t;
			nrm.set( Math.sin( tilt ) * Math.cos( phi ), Math.sin( tilt ) * Math.sin( phi ), Math.cos( tilt ) );
			qTilt.setFromUnitVectors( Z, nrm );
			qSpin.setFromAxisAngle( Z, slot.spin0 + 0.035 * t );
			slot.group.quaternion.copy( qTilt ).multiply( qSpin );

		}

		const resize = () => {

			S.aspect.value = camera.aspect;
			S.tanHalf.value = Math.tan( THREE.MathUtils.degToRad( camera.fov / 2 ) );

		};

		resize();
		const dbg = params.has( 'dbg' );
		let frameNo = 0;
		if ( dbg ) console.warn( `[crystal] create done: N=${cfg.N} tier=${tier}` );

		return {
			scene,
			bloom: { strength: 0.6, radius: 0.3, threshold: 0.45 },

			update( dt, t ) {

				if ( dbg && frameNo ++ < 4 ) console.warn( `[crystal] frame ${frameNo} t=${t.toFixed( 2 )} r=${lattice.radius} steps=${lattice.steps}` );
				resize();
				const slot = slots[ st.active ];

				// conditions
				if ( st.phase === 'grow' ) {

					if ( gp.drift ) drift( dt );
					st.growT += dt;
					lattice.u.time.value = 1 + st.growT;
					const n = stepsForFrame( dt );
					st.stepsLast = n;
					lattice.step( n );
					lattice.bake();

					st.pollT -= dt;
					if ( st.pollT <= 0 ) {

						st.pollT = 0.15;
						lattice.poll()?.then( onRadius );

					}

					const T = GROW_TIME / Math.max( 0.05, gp.speed );
					if ( lattice.radius >= R_TARGET || st.growT > T * 1.3 ) {

						st.phase = 'hold'; st.holdT = 0; st.stepsLast = 0;

					}

				} else if ( st.phase === 'hold' ) {

					st.holdT += dt;
					if ( st.holdT > HOLD_TIME ) newFlake();

				}

				// glow & shock ring after a cloud-layer change
				st.flash *= Math.exp( - dt / 1.4 );
				S.flash.value = st.flash;
				st.ringR += dt * 0.45;
				slot.P.ring.value = st.ringR;
				slot.P.now.value = st.growT;

				// flakes: precession + spin; the hero is magnified while small so growth stays legible
				for ( const s of slots ) {

					if ( s.state === 'idle' ) continue;
					orient( s, t );
					if ( s.state === 'grow' ) {

						const frac = Math.max( lattice.radius, 6 ) / RD;
						const target = THREE.MathUtils.clamp( 0.62 / frac, 1.0, 3.0 );
						if ( ! params.has( 'dbgcam' ) ) s.scale += ( target - s.scale ) * ( 1 - Math.exp( - dt / 1.2 ) );
						s.group.scale.setScalar( s.scale );

					} else if ( s.state === 'retire' ) {

						s.t += dt;
						const u = Math.min( 1, s.t / FADE_TIME );
						s.P.fade.value = u * u * ( 3 - 2 * u );
						s.group.position.set( s.retireFrom.x + 0.25 * u * u, s.retireFrom.y - 0.9 * u * u, s.retireFrom.z - 1.6 * u );
						s.spinner.rotation.z += dt * 0.6 * u;
						if ( u >= 1 ) {

							s.state = 'idle'; s.group.visible = false; s.spinner.rotation.z = 0;

						}

					}

				}

				bokeh.u.time.value = t;
				bokeh.u.focus.value = camera.position.distanceTo( controls.target );

				if ( captionTimer > 0 ) {

					captionTimer -= dt;
					if ( captionTimer <= 0 && caption ) caption.style.opacity = '0';

				}

				st.statT -= dt;
				if ( st.statT <= 0 ) {

					st.statT = 0.5;
					ctx.setStats( [
						[ 'lattice', `${cfg.N}² hex · ${( cfg.N * cfg.N / 1000 ).toFixed( 0 )}k sites` ],
						[ 'steps', `${lattice.steps.toLocaleString()} · ${st.stepsLast}/frame` ],
						[ 'mass', `Σc ${Math.round( lattice.mass ).toLocaleString()} · ${lattice.cells.toLocaleString()} cells` ],
						[ 'radius', `${lattice.radius} / ${R_TARGET}` ],
						[ 'layer', st.label ],
					] );

				}

			},

			resize,

			onKey( e ) {

				if ( e.key === 'n' || e.key === 'N' ) newFlake();
				if ( e.key === 'c' || e.key === 'C' ) cloudJump();

			},

			dispose() {

				canvas.removeEventListener( 'pointerdown', onDown );
				canvas.removeEventListener( 'pointerup', onUp );
				caption?.remove();
				controls.enablePan = true;
				controls.minDistance = 0;
				controls.maxDistance = Infinity;
				lattice.dispose();
				geo.dispose();
				haloGeo.dispose();
				for ( const s of slots ) {

					s.mat.dispose(); s.halo.dispose();

				}

				bokeh.dispose();
				scene.backgroundNode = null;

			},
		};

	},
};
