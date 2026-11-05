# Review C — crystal, aurora, silk, prism, horizon (PARTIAL)

**Status: partial.** The review was stopped early. Only **crystal** got reviewed: `crystal.js` and `crystal/sim.js` were read in full; `crystal/ice.js`, `crystal/bokeh.js` and `crystal/presets.js` were not read.
**Not reviewed:** aurora, silk, prism, horizon. All four have real code: aurora.js 504 lines + 5 helpers, silk.js 447 + 3, prism.js 717 + 3 (incl. raw WGSL in prism/wgsl.js), horizon.js 743 + 3.

Build: `npx vite build` passes at main 46be9cd (all chunks emitted, no errors). No GPU captures were run.

## crystal (Gravner–Griffeath snow crystal on a hex lattice, then a refracting ice slab)

Crash checks passed in sim.js. Workgroups are 8×8 = 64 / 64. Storage bindings per kernel stay at 3 or fewer (K2: S0, S1, stats). Atomics are on a `uint` buffer. The bake writes `rgba16float` storage write-only, which is a core format and needs no feature. K1 and K2 ping-pong S0→S1→S0, so there is no read/write race inside a dispatch. Neighbour reads at `hd == RD` stay in bounds because `RD = C − 2`. Stats readback is async with at most one in flight and a serial guard. No P0 found in the code that was read.

### P1
1. **`crystal/sim.js:312-320` + `crystal.js:270-283` — a full-lattice `kRescale` dispatch runs almost every frame.**
   `drift()` eases every param toward its target with `k = 1−exp(−dt/2)` (≈0.008 per frame) and stops only when `|d| < 1e-7`. From a typical gap of ~0.3 that takes about ln(3e6)/0.008 ≈ 1900 frames (~30 s), and a new target arrives every 7–13 s. So `applyParams()` → `setRho()` runs on nearly every frame. Its relative early-out (`1e-5`) almost never triggers, so each frame launches an extra N² dispatch over both buffers: 590k sites at tier 2, 262k at tier 1. It also multiplies vapour by a ratio ≈ 1 over and over, which builds up float error.
   *Fix:* snap a param to its target once the gap is within 1e-3 of the param's scale. Batch rho changes: either call `setRho` only when the cumulative ratio passes ~0.5 %, or move ρ into K1 as a far-field Dirichlet value and delete `kRescale` from the drift path. The explicit cloud jump can keep it.
2. **`crystal.js:457` — `lattice.bake()` runs every frame, including the `hold` phase and frames where `n = 0`.**
   At tier 2 that is a 1024² bake with 3 storage reads per texel when nothing has changed. *Fix:* bake only when `n > 0`, or on a newFlake/cloudJump event.
3. **`crystal/sim.js:338-358` — up to 128 dispatches per frame at tier 2 (`maxSteps: 64` × K1 + K2).**
   Each one rewrites the uniform group. That should be fine on discrete GPUs, but it is the main cost to profile. The active window is `R = radius + margin`, so late in growth it is the whole domain. *Suggestion:* cap tier 2 at ~32 steps, or merge K1+K2 into one kernel that uses a workgroup tile with a halo (shared memory, `workgroupBarrier`). That halves global traffic.

### P2
4. **`crystal/sim.js:347` — `u.seed = Math.random()` on every `step()` call, and σ defaults to 0.0008 > 0 (`crystal.js:74`).**
   Shot-mode captures are therefore not deterministic, even though `rnd` is seeded in shot mode. *Fix:* derive the seed from `this.steps` (or from `rnd`) when `ctx.shot` is set.
5. **`crystal/sim.js:404-409` — `dispose()` frees storage buffers through the private `renderer._attributes.delete()`.**
   This works in r186 (`Renderer.js:829`, `Attributes.js:46`), but it is a private API that could break on a three upgrade. Keep it, but leave a comment (or flag it to the lead) so it is reviewed when three is bumped.
6. **`crystal.js:371-397` — the warp pre-growth `await`s a GPU readback every 100–250 steps.**
   In shot mode at tier 0 that is up to 40 serial round trips on SwiftShader. It is only debug/shot cost, but it adds to capture time. Polling every ~1000 steps would do.
7. **`crystal.js:323-339` — the click detector uses raw DOM listeners rather than `ctx.pointer`.**
   It works and is cleaned up in `dispose`. Note that a quick tap also starts an OrbitControls rotate, which is harmless. The `crystal.js:552-571` dispose also resets controls pan/min/max (good). Not verified: whether `createStudio()` (ice.js) owns textures or RTs that need disposing — `studio` is never disposed.

## Not reviewed (high-value checks for the next reviewer)
- **prism/wgsl.js** (raw WGSL): check uniformity analysis (textureSample or derivatives inside loops or branches), f32/i32 mixing, loop-length / TDR risk at tier 2.
- **horizon**: NaN near r₊ and in the sqrt terms of the Kerr metric, per-pixel step count per tier, camera handling (the shell only updates `inst.camera.aspect`).
- **silk** (907-line sim): constraint-solver races, i.e. Jacobi vs. concurrent writes to shared particles; storage bindings per stage ≤ 8.
- **aurora**: storage buffer count per stage, per-frame allocations, RT restore.
