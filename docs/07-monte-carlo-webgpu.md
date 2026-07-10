# 07 — Monte Carlo & WebGPU

The showpiece. Instead of one backtest line, run **10⁴–10⁵ randomized
re-runs on the GPU** and show distributions: an equity fan, a drawdown
histogram, a probability of ruin. `packages/sim` owns all of it.

## 1. What we simulate (three modes)

| Mode | Input | Question it answers |
|---|---|---|
| **A. Bootstrap resampling** | `returnsPerBar` from a backtest `RunResult` (docs/06) | "If the strategy's returns are real but their *order* is luck, what's the range of outcomes?" — resample per-bar returns with replacement (IID) or in blocks of length L (default 20, preserves autocorrelation) into N paths of horizon H |
| **B. Parametric paths (GBM / jump-diffusion)** | μ, σ (estimated from history or user-set), optional jump params (λ, μⱼ, σⱼ) | "What does the *price* do?" — feeds option payoff pricing (docs/09) and what-if fan charts on any symbol |
| **C. Strategy-on-synthetic** *(v2, CPU-only at first)* | Mode-B paths + a strategy | "Does the strategy survive worlds that didn't happen?" — runs the full backtest loop per synthetic path on the worker pool |

Modes A and B are embarrassingly parallel per-path float math → perfect GPU
targets. Mode C reuses the CPU engine on the worker pool (GPU port only if we
ever need it).

## 2. One interface, two backends

```ts
// packages/sim/src/backend.ts
export interface SimBackend {
  readonly kind: 'webgpu' | 'cpu';
  run(req: SimRequest, onProgress?: (pct: number) => void): Promise<SimResult>;
}

export interface SimRequest {
  mode: 'bootstrap' | 'gbm';
  paths: number;              // default 50_000, max 500_000
  horizon: number;            // bars
  seed: number;               // master seed — REPRODUCIBLE, stored in result
  // bootstrap:
  sourceReturns?: Float32Array; blockLen?: number;   // 1 = IID
  // gbm:
  mu?: number; sigma?: number; dt?: number; s0?: number;
  jumps?: { lambda: number; muJ: number; sigmaJ: number };
  // reductions computed on-device (see §4):
  percentiles: number[];      // default [5, 25, 40, 50, 60, 75, 95]
}

export interface SimResult {
  runId: string; req: SimRequest; backend: 'webgpu' | 'cpu'; elapsedMs: number;
  fan: Float32Array;          // [percentiles.length × horizon] equity/price percentiles per step
  terminal: Histogram;        // distribution of end values (128 bins)
  maxDrawdownDist: Histogram; // distribution of per-path max DD
  ruinProb: number;           // P(path ever ≤ ruinLevel, default 50% of start)
  ghostPaths: Float32Array;   // 60 sample paths for the 4%-alpha overlay (docs/02 §5.7)
}
```

`createSimBackend()` detects WebGPU (`navigator.gpu?.requestAdapter()`), falls
back to the CPU worker, and records the choice in `useSettingsStore` → "GPU"
or "CPU mode" badge in the Sim Lab. **Identical `SimRequest` semantics on both
backends**; results agree within tolerance (§6).

## 3. RNG (the part people get wrong)

- **Philox4x32-10 counter-based RNG.** No state to carry: `random =
  philox(key=seed, counter=(pathId, step, substream))`. Every thread derives
  its stream independently → no correlation between paths, bit-reproducible
  for a given seed on both backends, and trivially parallel. (This is the
  standard choice for GPU Monte Carlo; do not substitute `Math.random` or a
  shared-state xorshift.)
- Uniform → normal via **Box-Muller** (uses 2 uniforms → 2 normals; polar
  rejection is branch-divergent and worse on GPU).
- The CPU backend implements the *same* Philox + Box-Muller in TS, so CPU and
  GPU draw **identical random sequences**. Divergence between backends is then
  only float accumulation order — testable and small (§6).

## 4. WebGPU design

WGSL compute shaders in `packages/sim/src/wgsl/` (imported as strings via Vite
`?raw`). Two-kernel pipeline per batch:

**Kernel 1 — `simulate.wgsl`** (one invocation = one path)

```
@group(0) @binding(0) var<uniform>            cfg      : SimCfg;
@group(0) @binding(1) var<storage, read>      returns  : array<f32>;  // bootstrap source (mode A)
@group(0) @binding(2) var<storage, read_write> terminal: array<f32>;  // per-path end value
@group(0) @binding(3) var<storage, read_write> maxDD   : array<f32>;  // per-path max drawdown
@group(0) @binding(4) var<storage, read_write> ruinHit : array<u32>;
@group(0) @binding(5) var<storage, read_write> fanSamp : array<f32>;  // path value at K checkpoints
@compute @workgroup_size(256)
```

Each thread walks its path `horizon` steps, keeping running value / peak /
maxDD in registers, writing only per-path aggregates plus its value at **K = 64
checkpoint steps** (evenly spaced) for fan estimation. Paths are `f32`
(sufficient for path values; see §6), reductions in `f32` with per-workgroup
partials.

**Kernel 2 — `reduce.wgsl`**: histogram terminal values and maxDD (atomic adds
into 128 bins), done on-device so the readback is kilobytes, not
`paths × horizon` floats.

**Percentiles for the fan:** readback the `paths × 64` checkpoint matrix
(50k paths → 12.8 MB, fine) and select percentiles per checkpoint on the CPU
with `Float32Array` nth-element — exact, simple, and off the hot path. If we
push to 500k paths, switch to on-device P² estimation (noted as future work).

Batching & limits:

- Paths dispatched in batches ≤ 65 535 workgroups; multiple `dispatchWorkgroups`
  per command buffer, `onProgress` after each submit via `queue.onSubmittedWorkDone()`.
- Respect `maxStorageBufferBindingSize` from adapter limits; degrade `paths`
  with a toast if the request exceeds device capability.
- All GPU work behind `device.lost.then(...)` → auto-retry once on CPU backend.

## 5. CPU fallback

Same algorithm, same Philox streams, in a Web Worker with a plain `for` loop
over paths (chunked, `onProgress` per chunk, transfers results back). Target:
50k × 252 in a few seconds — slower than GPU but usable. One worker only
(sims don't get the whole pool; backtests share it).

## 6. Validation & performance (CI-enforced where possible)

- **Statistical golden tests** (Vitest, CPU backend): GBM with known μ, σ ⇒
  E[S_T], Var[ln S_T] match closed form within 3σ of the MC standard error;
  bootstrap of constant returns ⇒ deterministic curve, exactly.
- **CPU/GPU parity** (Playwright, real Chromium with WebGPU): same seed ⇒ fan
  percentiles agree within 0.1%, ruinProb within 0.5pp, histograms within
  per-bin tolerance. Run headless in CI with `--enable-unsafe-webgpu` where
  available; skipped (not failed) if no adapter.
- **Perf budgets** (dev-mode assert, logged to console.table): 50k paths ×
  252 steps ≤ **300ms** GPU on a 2021+ laptop, ≤ 5s CPU. UI stays interactive
  throughout (all off main thread).

## 7. Sim Lab UI (the full-page view)

- Left rail: mode picker (A/B), source selector (a backtest run, or
  symbol+range for GBM parameter estimation), params with sensible defaults,
  paths/horizon/seed, block length, **Run** button with GPU/CPU badge.
- Main: `mc-fan` widget (fan chart per docs/02 §5.7 — layered single-hue bands,
  median line, realized path overlaid in primary ink), StatTile row
  (median terminal, P5, P95, max-DD median, **ruin probability** with `--warn`
  styling ≥ 10%), `mc-histogram` widgets for terminal value & max DD.
- Every result: "pin to dashboard" (publishes to `selectedRuns` context,
  docs/03 §5) and "copy seed/config" for reproducing a run in a bug report.
- Empty state teaches: "Run a backtest first, then stress-test it here."
