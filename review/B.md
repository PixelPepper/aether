# Review B — atmosphere, erosion, ocean, morpho, genesis (PARTIAL)

**Status: partial.** The lead stopped this review before any per-chapter findings were done.
**Chapters NOT reviewed:** atmosphere, erosion, ocean, morpho, genesis. None of their code has had a line-by-line review yet.
What follows is only what I checked across the whole project, against `main` @ `46be9cd` (07:52 UTC) and three r186.1 source.

## Build

- `vite build` passes for the whole project (77 modules, no errors). I ran it on a copy of the repo with `three@0.186.1` / `vite@6.4.5` installed in a scratch dir. The repo itself has no `node_modules`, so `npx vite build` in `/home/user/aether` fails with `Cannot find package 'vite'`.
  Fix (lead): run `npm install` once in the repo, or document that the build has to run where deps are installed.

## Cross-cutting findings (apply to every chapter)

### P0-risk — the device only has the default WebGPU limits, even on desktop
- `src/main.js:30` passes `requiredLimits: undefined`. `WebGPUBackend.js:97` turns that into `{}`, and `WebGPUBackend.js:246-247` requests the device with no raised limits. So every GPU, the RTX 4090 included, runs with the **spec defaults**:
  - `maxStorageBuffersPerShaderStage = 8`
  - `maxStorageTexturesPerShaderStage = 4`
  - `maxStorageBufferBindingSize = 128 MiB`, `maxBufferSize = 256 MiB`
  - `maxComputeWorkgroupStorageSize = 16 KiB`, `maxComputeInvocationsPerWorkgroup = 256`
- Consequence: any chapter kernel that binds a 9th storage buffer, or any buffer over 128 MiB at tier 2, fails pipeline creation on **every** machine. "It works on my desktop" does not count as evidence. Each chapter agent should count the distinct `instancedArray`/`storage` nodes reached from each `Fn` (`.toAttribute()` does not count; it is a vertex attribute).
- Optional shell fix: request the adapter's real limits for the few that matter, e.g.
  `requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 10), maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize }`.
  This needs the adapter before `init()`. Even so, chapters should stay within the defaults so they run on Android.

### OK — large compute counts are handled
- `WebGPUBackend.js:1959-1966`: when `ceil(count / workgroupSize) > maxComputeWorkgroupsPerDimension` (65535), three splits the dispatch into `[min(n, max), ceil(n / max), 1]`. `compute(count)` with a few million invocations is therefore safe. **Caveat:** a kernel that does its own index arithmetic from `globalId.x` / `workgroupId.x` instead of `instanceIndex` breaks once the dispatch goes 2-D. Kernels also need an `if (instanceIndex >= count) return` guard, because the last workgroup over-dispatches.

### Note — `float32-filterable`
- three requests every supported feature (`WebGPUBackend.js:246`), and `WebGPUBindingUtils.js:551` only declares float32 textures as filterable when the feature exists. On devices without it (many Android, and some iOS), a `FloatType` texture sampled with `LinearFilter` silently gets unfilterable-float bindings, or validation errors. Chapters that sample float32 sim textures (erosion heightmap, morpho R-D field, atmosphere LUTs, likely candidates) should use `HalfFloatType`, or do manual bilinear filtering through `textureLoad`.

## Not done (to pick up in a follow-up review)
For each of the five chapters: count storage buffers and textures per stage; check atomics, barriers and uniformity; look at numerical stability (MPM in ocean, explicit diffusion in morpho, hydraulic erosion CFL, the particle-life r→0 case in genesis); check per-frame allocations and graph rebuilds, dispose, render-target restore, and tier scaling.
