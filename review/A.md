# Review A — quantum, cosmicweb, collision, nebula, stellar  (PARTIAL)

Reviewed at main `46be9cd` (07:52 UTC autosave). Stopped early at the lead's request.

- **Reviewed in full:** `quantum.js` + `quantum/{solver,render,scenarios}.js`, `cosmicweb.js` + `cosmicweb/{pm,cosmology}.js`
- **Not reviewed:** `collision*`, `nebula*`, `stellar*`. All three have real code (527 / 434 / 302 lines plus helpers). The cross-cutting item X1 very likely applies to them as well.
- `npx vite build`: **passes** (all 15 chapters bundle; build checks syntax and imports only, not WGSL).

Priority: **P0** = crash or wrong output on real GPUs · **P1** = leak or serious quality/perf problem · **P2** = minor/cleanup.

---

## Cross-cutting

### X1 (P1) — `instancedArray(...).value.dispose()` frees no GPU memory in r186
`BufferAttribute.dispose()` (`three/src/core/BufferAttribute.js:684`) only dispatches a `'dispose'` event. For a storage buffer that is only bound to compute or used via `toAttribute()`, nothing listens for that event:
- `Bindings._destroyBindings` destroys uniform buffers and samplers only.
- `Geometries` listens on *geometries*, not storage attributes.

So these calls are no-ops for GPU memory:
- `quantum/solver.js:550` (`b.value?.dispose?.()`)
- `cosmicweb/pm.js:396-400`

The buffers are freed only when the JS objects are garbage-collected, whenever that happens. Cosmicweb at tier 2 holds about **150 MB per visit**: C 16 MB, Dk 16 MB, F 32 MB, grid 8 MB, rho 8 MB, pos/vel 64 MB. Switching chapters 10× can stack well over 1 GB before GC. On Android that means an OOM or device loss.

**Fix:** call `renderer._attributes.delete( buf.value )` for each storage buffer. That calls `backend.destroyAttribute` → `GPUBuffer.destroy()`. Do it **after** disposing the compute nodes. `cosmicweb.js:583-591` already does this. Quantum (`solver.js:550`, plus `counterAttr` at `:551`) needs the same. The lead should consider adding a `ctx.freeBuffer(attr)` helper to the shell so every chapter does it the same way. Collision, nebula and stellar most likely have the same pattern but I did not check them.

### X2 (P2) — never dispose the shared Sprite geometry
`THREE.Sprite` shares one module-level `_geometry` (`three/src/objects/Sprite.js:12,93`).
- `quantum/render.js:271` calls `sprites.geometry?.dispose?.()`.
- `quantum.js:559` calls `mist.geometry?.dispose?.()`.

Both dispose that shared geometry. Every other chapter's sprites then lose their buffers, and the buffers have to be re-uploaded. Cosmicweb correctly avoids this (comment at `cosmicweb.js:579`). **Fix:** delete those two calls.

### X3 (P2) — workgroup-barrier kernels have no bounds guard
`ComputeNode` drops its `if (instanceIndex >= count) return` guard whenever a barrier is present (`BarrierNode.js:31` sets `allowEarlyReturns = false`). If a dispatch goes over `maxComputeWorkgroupsPerDimension` (65535), three splits it into 2D with `ceil()` (`WebGPUBackend.js:1963`). The extra workgroups then run unguarded.
- **Cosmicweb FFT:** `compute(N*N*H,[H])` gives N² workgroups, which is fine at N ≤ 128. `?mesh=256` is rejected by `pow2()` today, so this is safe for now. Keep N ≤ 128, or add a manual `If(line < N*N)` guard around everything except the barriers (keep the barriers unconditional).
- **Quantum:** uses `workgroupId.x` with N ≤ 512 workgroups, so it is fine.

---

## quantum (Fluctuation)

Overall solid. I checked the following and they are correct:
- DIF forward / DIT inverse twiddles and indexing (`solver.js:121-156`)
- the bit-reversed kinetic table
- the 1/N² folding
- Strang ordering
- barriers all in uniform control flow
- workgroup size H = N/2 = 256 at tier 2, which is exactly at the WebGPU default limit (OK)
- shared memory: 4 KB
- at most 5 storage buffers per stage (detect kernel)
- `rgba16float` storage texture (core format, filterable)

| # | P | where | issue | fix |
|---|---|---|---|---|
| Q1 | P1 | `solver.js:550-551` | X1: storage buffers (psi, pot, ktab, rowSum, rng, hits, mist, counter) never destroyed. ~2 MB psi + 2 MB pot + 1 MB mist at tier 2. | `renderer._attributes.delete(b.value)` / `delete(this.counterAttr)` after the node disposes |
| Q2 | P2 | `render.js:271`, `quantum.js:559` | X2: disposes the shared Sprite geometry | remove |
| Q3 | P2 | `solver.js:220,253` | Tier 2 column pass reads psi with stride N (512×8 B), which is uncoalesced, and it runs 5–10× per frame. It works, but it is likely the main GPU cost on mobile/integrated GPUs. | Optional: transpose once per step, or do the column FFT on a transposed buffer layout |
| Q4 | P2 | `quantum.js:212-218` | `frameList()` does `slice()` + pushes, allocating a new array every frame | cache per `(steps, mist)` key like `stepList` |
| Q5 | P2 | `solver.js:367` | `slot.bitAnd(MAXH-1)` assumes the hit count is a power of two. True for all tiers today; worth an assert. | `console.assert` in the constructor |
| Q6 | P2 | `quantum.js:394-397` | Capture-phase window listeners set `controls.enabled=false` before OrbitControls sees `pointerdown`. That works, but a press on the plane that *starts* outside `inside()` still orbits, while one starting inside never does. Intended; just confirm the UX. | — |

Physics sanity:
- Tier 0 `timeScale=2.5` → dt = 2.5e-5, so V_WALL·dt ≈ 3 rad per step at the walls. Split-step stays unitary, but splitting error at walls grows. Acceptable.
- kMax is under Nyquist (πN) on every tier.

## cosmicweb (Cosmic Web)

I checked the following and they are correct:
- Stockham indexing and twiddles (`pm.js:147-176`)
- the fused z-Poisson with the discrete-Laplacian Green function and the 1/N³ folded into `green`
- the sign of φ (−1.5 Ωm/a)
- CIC periodic wrap
- KDK with Simpson kick/drift integrals
- Zel'dovich s = ikδ/k² with Nyquist zeroed
- 2–3 storage buffers per stage plus one storage 3D texture

| # | P | where | issue | fix |
|---|---|---|---|---|
| C1 | P1 | `cosmicweb.js:431` | If `float32-blendable` is missing (most Android/Mali/Adreno and some Intel), the accumulator falls back to **HalfFloat** with additive blending of ~260k–2M tiny splats. fp16 has an 11-bit mantissa, so small per-splat values stop adding once Σ ≳ 2–4: voids and filaments band and halos saturate, and the log/colour-index post (`acc.g/acc.r`) gets noisy. | On the fallback, pre-scale: multiply `flux` by a tier constant so typical Σ stays ~O(1), or splat into `rgba16float` at reduced resolution. Better: deposit projected density with a compute atomic (fixed-point u32) into a 2D buffer instead of blending. |
| C2 | P1 | `pm.js:36,227` | The fixed-point grid is i32 with FIX = 65536, so a cell overflows at ρ/ρ̄ ≈ 32768 (mass = 1 when NL = N). PM halos at 128³ / 1.25 Mpc/h cells reach δ ~ 10³–10⁴ at z = 0, which is close. With `?np=128&mesh=32`, mass = 64 and it overflows at ρ ≈ 512, which *will* wrap negative and blow up forces. | Make FIX depend on mass: `FIX = floor(2^31 / (mass * 8 * 4096))`, or use u32 (deposits are non-negative), which doubles the headroom. |
| C3 | P1 | `pm.js:396-400` | X1: about 150 MB at tier 2. Already mitigated by `cosmicweb.js:583-591`, but that only runs from chapter `dispose`, which is fine. Keep it, and add `volTex` (it is disposed correctly). | — |
| C4 | P2 | `cosmicweb.js:405` | Capture mode jumps to z = 1.2 with `advance(aT, 100000, …)`. At SEEK_DLNA 0.045 that is about 85 full PM steps (9 kernels each) submitted inside `create()` with no await in between. That is fine on GPU. On SwiftShader it can trip the shot timeout; if captures are flaky, add an `await renderer.computeAsync` every ~10 steps. | — |
| C5 | P2 | `cosmicweb.js:528` | `camera.position.clone()` allocates every frame | reuse a scratch `Vector3` |
| C6 | P2 | `pm.js:276` | Writes full `rgba16float` (8 B per voxel) to store one channel | Fine. `r32float` is a core storage format with half the voxel count of rgba16 bytes, but it is not filterable without `float32-filterable`, so leave as is. |
| C7 | P2 | `pm.js:193` | `select(k2 > 1e-9, …)` zeroes the DC mode. Correct. Note that `kToComplex` writes `r - 1`, so the mean is already ~0. OK. | — |
| C8 | P2 | `cosmicweb.js:94` | The debug path reads `renderer._pipelines`, a private API, behind `?cwdump` only. | Fine, but strip it before shipping. |

## collision, nebula, stellar
**Not reviewed** (stopped early). Suggested quick checks for whoever picks them up:
- X1 (storage-buffer destroy on dispose)
- X2 (shared Sprite geometry)
- storage buffers per stage ≤ 8
- workgroup sizes ≤ 256
- storage texture formats in the core list (no `r16float`, `rg11b10ufloat` or `bgra8unorm` storage without features)
- render-target restore around any offscreen `renderer.render`
