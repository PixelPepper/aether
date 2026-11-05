# AETHER — chapter contract

AETHER is a WebGPU-only three.js (r186.1, `three/webgpu` + TSL) experience made of chapters.
The shell (`src/main.js`) owns the renderer, camera, OrbitControls, pointer, GUI, post chain, HUD, chapter transitions.
Each chapter is ONE module `src/scenes/<id>.js` (it may add helpers under `src/scenes/<id>/`). Do not edit
`src/main.js`, `index.html`, `src/style.css`, `src/scenes/index.js` or other chapters — tell the lead if the shell needs a change.

## Module shape

```js
import * as THREE from 'three/webgpu';
import { Fn, ... } from 'three/tsl';

export default {
  id: 'nebula',
  title: 'Stellar Nursery',            // shown large in the HUD
  subtitle: 'One sentence, poetic but technically honest about what is being simulated.',
  async create( ctx ) {
    // build everything here; await renderer.computeAsync(...) for init passes is fine
    return {
      scene,                    // THREE.Scene (required)
      camera,                   // optional; defaults to ctx.camera (shared PerspectiveCamera + ctx.controls)
      bloom: { strength, radius, threshold },     // optional, for default post chain
      post( ctx, scenePass ) {}, // optional: return a vec4 TSL node to replace the default post chain
                                 // (scenePass = pass(scene, camera); shell adds vignette+grain after, then tone mapping AgX + sRGB)
      update( dt, t, ctx ) {},  // per frame, BEFORE render. Run renderer.compute(...) here.
      resize( w, h, dpr ) {},   // optional
      onKey( e ) {},            // optional
      dispose() {},             // REQUIRED: free geometries, materials, render targets, storage buffers, listeners
    };
  },
};
```

## ctx

| field | meaning |
|---|---|
| `THREE` | `three/webgpu` namespace |
| `renderer` | `WebGPURenderer`, already `init()`ed, WebGPU backend guaranteed. Tone mapping AgX. |
| `camera`, `controls` | shared camera + OrbitControls (damping on). Shell calls `controls.reset()` before your chapter; set `camera.position`, `controls.target`, min/maxDistance, `controls.autoRotate` etc. yourself. Set `controls.enabled = false` if you drive the camera. |
| `pointer` | `{ ndc: Vector2, velocity: Vector2 (ndc/s), down: bool, button, ray: Ray (updated each frame from ndc + active camera), inside: bool }` |
| `gui` | a lil-gui folder for this chapter (destroyed on exit). Bind TSL `uniform()`s via `ctx.gui.add(u, 'value', min, max).name('…')`. |
| `tier` | 0 = software GPU (SwiftShader, headless tests), 1 = mobile, 2 = desktop discrete. **Scale particle counts / steps by tier.** `?tier=N` overrides. |
| `shot` | true in headless capture mode (dt fixed at 1/60). |
| `params` | URLSearchParams of the page (for your own debug flags). |
| `size` | Vector2 CSS px size. |
| `setStats( [[label, value], ...] )` | extra lines in the top-right stats panel (call ≤ 2×/s, not every frame). |

## Quality bar

- This is meant to be at the edge of what real-time web graphics can do. Real GPU compute (`instancedArray`, `storage`,
  `atomicAdd`, `workgroupBarrier`, storage textures, indirect draws, etc.), physically-motivated simulation, serious shading.
  Not a tutorial demo.
- Must look stunning at tier 2 (desktop GPU, 60 fps target at 1080p) and still render *correctly* at tier 0.
- Interactive: the pointer must do something meaningful. Add 3–6 tasteful GUI controls.
- Cinematic default camera framing; gentle idle motion (autoRotate or a scripted drift) so a screenshot is never empty.
- Colour: HDR values into the AgX tone mapper + bloom. Avoid clipped flat white blobs. Stay in a coherent palette.
- No external network assets (no textures/models/fonts fetched). Generate everything procedurally.
- No new npm dependencies.
- Clean dispose: switching chapters 10× must not leak or throw.

## Testing (WebGPU works headless via SwiftShader — slow, CPU)

A Vite dev server should already be running at http://127.0.0.1:5173 (the harness starts one if not).

```
cd /home/user/aether
NODE_PATH=/opt/node-tools/node_modules node tools/shot.mjs <id> --frames 60 --out shots/<id>-1.png      # tier 0 auto
NODE_PATH=/opt/node-tools/node_modules node tools/shot.mjs <id> --extra "tier=2" --frames 20 --out shots/<id>-t2.png
```

Note: headless SwiftShader cannot present a WebGPU canvas, so in `?shot=1` mode the shell renders the final post chain into an
offscreen RenderTarget and reads it back into a 2D canvas (see `blitShot` in main.js). Your chapter code does not need to care,
but never call `renderer.setRenderTarget(null)` + render to screen yourself inside `update()` — render into your own RTs and restore
the previous target (`const prev = renderer.getRenderTarget(); … renderer.setRenderTarget(prev)`).
Expect ~3–6 s per frame on SwiftShader for a full chapter at 640×360; budget your captures.

It prints console errors/warnings + page errors and exits non-zero on any error. Then **look at the PNG with the Read tool**.
SwiftShader is ~100× slower than a real GPU: a run can take 1–4 min; use `--frames` small, `--w 960 --h 540` to iterate faster,
and run long commands in the background. Shader compile errors (WGSL) show up as console errors — read them.
Other agents share this machine (15 agents, 4 CPU cores). `tools/shot.mjs` holds a global semaphore (3 concurrent captures);
it prints `[slot] waited Ns` when it had to queue. Never run more than one capture at a time yourself, keep captures small
(`--w 480 --h 270` while iterating, few frames), and do useful work (reading three.js source, writing code) while one runs in
the background instead of waiting idle. Kill any capture/browser you started if you abandon it.

Reference: three.js source is in `node_modules/three/src` (TSL: `src/nodes/`), addons in `node_modules/three/examples/jsm/`.
Read the source rather than guessing APIs — TSL changes fast. If network allows, raw examples are at
https://raw.githubusercontent.com/mrdoob/three.js/r186/examples/webgpu_compute_particles.html etc.
