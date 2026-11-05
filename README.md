# AETHER

Fifteen real-time simulations in one WebGPU experience, from quantum foam to the event horizon.
Built with three.js r186 (`three/webgpu` + TSL compute shaders). Everything is generated procedurally; no assets are downloaded.

| # | Chapter | Simulation |
|---|---|---|
| 01 | Fluctuation | Time-dependent Schrödinger equation, split-step Fourier with a GPU FFT |
| 02 | Cosmic Web | Particle-mesh N-body structure formation from Zel'dovich initial conditions |
| 03 | Collision | Galaxy merger: tiled direct N-body halos and up to ~1M star particles |
| 04 | Nursery | Volumetric emission nebula, GPU-generated density and light volumes |
| 05 | Ignition | 3D Eulerian plasma with blackbody emission (solar prominence) |
| 06 | Atmosphere | Hillaire 2020 sky/atmosphere LUTs, procedural planet, clouds |
| 07 | Deep Time | Virtual-pipe hydraulic + thermal erosion with tectonic uplift |
| 08 | Sea | MLS-MPM fluid with screen-space fluid rendering |
| 09 | Morphogenesis | 3D Gray–Scott reaction–diffusion, ray-marched isosurface |
| 10 | Emergence | 3D particle life with a GPU spatial hash (atomic counting sort + scan) |
| 11 | Ice | Gravner–Griffeath snowflake growth, dispersive ice rendering |
| 12 | Aurora | Boris-pusher electrons in a dipole field, altitude-correct emission |
| 13 | Silk | XPBD cloth with graph-coloured constraints and sheen/iridescence |
| 14 | Spectrum | Spectral path tracer with Sellmeier dispersion and denoising |
| 15 | Horizon | Kerr/Schwarzschild geodesic ray tracing, Novikov–Thorne disk |

## Run

```
npm install
npm run dev        # http://localhost:5173
```

Requires a browser with WebGPU (recent Chrome/Edge, Safari 26+, Firefox 141+) and a real GPU.

Keys: `←` `→` chapters · `H` hide UI · `G` controls · `M` sound · `T` hands-free tour.
URL params: `?c=<chapter id>` start chapter, `?tier=0|1|2` quality tier, `?q=<pixel ratio>` fixed resolution.

## Structure

- `src/main.js` – shell: renderer, post chain, chapter lifecycle, adaptive resolution, HUD
- `src/core/audio.js` – generative WebAudio score, one mood per chapter
- `src/scenes/<id>.js` – one module per chapter, see `CONTRACT.md`
- `tools/shot.mjs` – headless capture harness (Chromium + SwiftShader WebGPU)
