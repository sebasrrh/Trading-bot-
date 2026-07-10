# 01 — Architecture

## Shape of the system

Two runnable apps, seven shared packages, one direction of dependencies.

```
                    ┌──────────────────────────────────────────────┐
                    │                 apps/web (React)             │
                    │  ┌────────────┐ ┌──────────┐ ┌────────────┐  │
                    │  │ Dashboard  │ │ Backtest │ │   Paper    │  │
                    │  │ shell +    │ │ & sim UI │ │  trading   │  │
                    │  │ widgets    │ │          │ │  UI        │  │
                    │  └─────┬──────┘ └────┬─────┘ └─────┬──────┘  │
                    │        │             │             │         │
                    │  ┌─────┴─────────────┴─────────────┴──────┐  │
                    │  │ Web Workers: backtest runs, CPU sims   │  │
                    │  │ WebGPU: Monte Carlo kernels             │  │
                    │  └────────────────────────────────────────┘  │
                    └───────────────────────┬──────────────────────┘
                                            │ HTTPS (typed JSON)
                    ┌───────────────────────┴──────────────────────┐
                    │              apps/api (Node + Hono)          │
                    │   provider adapters → normalize → SQLite     │
                    │   cache → rate limiter → REST endpoints      │
                    └───────────────────────┬──────────────────────┘
                                            │
                       Alpaca / Polygon / Stooq / Yahoo (docs/04)
```

## Monorepo layout

pnpm workspaces. Node ≥ 22. Every package is ESM, strict TypeScript,
built with `tsc` (packages) or Vite (web).

```
apps/
  web/                  # the dashboard SPA
    src/
      app/              #   shell: routing, layout grid, workspace persistence
      widgets/          #   widget implementations + registry (docs/03)
      views/            #   full-page views: Backtest Lab, Sim Lab, Paper, Settings
      charts/           #   chart wrappers (lightweight-charts, uPlot, SVG)
      workers/          #   Web Worker entry points (backtest, cpu-sim)
      state/            #   Zustand stores: workspace, global context, settings
      lib/              #   web-only helpers (IndexedDB, formatters)
  api/                  # data proxy
    src/
      providers/        #   one adapter per data vendor (docs/04)
      cache/            #   SQLite bar cache
      routes/           #   /bars /quote /search /chain /leaderboard
packages/
  core/                 # domain types + shared utils — depends on nothing
  data/                 # DataClient (browser) + provider interface (shared)
  indicators/           # pure incremental indicators
  strategies/           # Strategy interface + built-ins + registry
  backtest/             # event-driven engine, metrics
  sim/                  # SimBackend interface, WGSL kernels, CPU fallback
  paper/                # paper accounts, fill simulator, journal
  ui/                   # design tokens (CSS vars), primitives, chart chrome
```

## Dependency rules (enforced by review + `dependency-cruiser` later)

```
core ← indicators ← strategies ← backtest ← (web workers)
core ← data ← (web, api)
core ← sim ← (web)
core ← paper ← (web, api leaderboard route)
ui   ← (web only)
```

- `packages/*` never import from `apps/*`.
- `core` imports nothing from the workspace. It is types + pure functions.
- `strategies`, `backtest`, `sim`, `paper` are **environment-agnostic**: no DOM,
  no `fetch`, no Node APIs. They run in the main thread, a Worker, or Node tests
  unchanged. All I/O is injected.
- Only `apps/api` holds secrets. The browser never sees a provider API key.

## Where computation runs

| Work | Where | Why |
|---|---|---|
| UI, charts | Main thread | It's a UI. |
| Backtest runs | Web Worker (pool of `navigator.hardwareConcurrency - 1`) | A 10-year sweep must not jank the grid. |
| Monte Carlo | WebGPU compute pass; CPU Worker fallback | docs/07. |
| Indicator precompute for charts | Main thread (incremental) or Worker (bulk) | Indicators are O(n) incremental. |
| Data fetch + cache | `apps/api` (SQLite) + browser IndexedDB | Two-level cache, docs/04. |

**Rule:** anything that can take >16ms goes off the main thread. Workers receive
`Float64Array`/`Float32Array` buffers via transfer, not JSON bars.

## Typed contracts

- All cross-boundary payloads (API responses, worker messages, workspace JSON,
  strategy params, widget configs) have **zod schemas in `packages/core`**
  (`core/src/schemas/`). Types are inferred from schemas — never hand-written
  twice.
- Worker protocol: request/response with `id`, plus streamed `progress` events:
  ```ts
  type WorkerReq  = { id: string; kind: 'backtest' | 'sim'; payload: ... };
  type WorkerMsg  = { id: string; type: 'progress'; pct: number }
                  | { id: string; type: 'result'; payload: ... }
                  | { id: string; type: 'error'; message: string };
  ```
- API routes are defined with Hono's typed client (`hono/client`) so `apps/web`
  gets end-to-end types without codegen.

## Bar data representation (performance-critical)

JSON bars are for the wire only. In memory, a series is **columnar**:

```ts
// packages/core — the universal in-memory format
interface BarSeries {
  symbol: string;
  timeframe: Timeframe;
  t: Float64Array;  // epoch ms, ascending, no duplicates
  o: Float64Array;
  h: Float64Array;
  l: Float64Array;
  c: Float64Array;
  v: Float64Array;
  length: number;
}
```

Columnar buffers transfer to Workers zero-copy, map directly into GPU buffers,
and make indicator loops cache-friendly. `core` provides
`toBarSeries(json)` / `sliceBars(series, from, to)` / `mergeBars(a, b)`.

## State management (apps/web)

- **Zustand stores**, one per concern, no global god-store:
  - `useWorkspaceStore` — grid layouts, widget instances/configs (persisted, docs/03)
  - `useContextStore` — global symbol/timeframe/date-range + link channels (docs/03)
  - `useRunStore` — in-flight backtest/sim runs, progress, results index
  - `useSettingsStore` — theme prefs (density), data provider choice, seeds
- **TanStack Query** owns all server data (bars, quotes, search, chain) with
  IndexedDB persistence for bar history. Query keys:
  `['bars', symbol, timeframe, from, to]`, `['quote', symbol]`, etc.
- Backtest/sim **results** are content-addressed: `runId = hash(strategyId,
  paramsJSON, symbol, timeframe, range, costModel, seed)`. Stored in IndexedDB;
  the compare view (docs/03) reads them by id. Re-running identical inputs is a
  cache hit.

## Error handling posture

- Provider failures degrade: api tries providers in priority order and reports
  which one served (`X-Data-Source` header + UI badge, docs/04).
- WebGPU unavailability is not an error: capability-detect at startup, store in
  `useSettingsStore`, show "CPU mode" badge in the Sim Lab (docs/07).
- Workers crash independently: the pool restarts a dead worker and re-queues its
  job once; a second failure surfaces as a run error toast with the message.

## Testing strategy

| Layer | Tool | What |
|---|---|---|
| packages | Vitest | Unit tests; indicators/backtest have golden-file tests (known input → exact expected output) |
| sim | Vitest | CPU vs GPU parity within tolerance (docs/07 §validation) |
| api | Vitest + recorded fixtures | Provider adapters tested against saved responses, never live APIs in CI |
| web | Playwright | Smoke: load dashboard, add widget, run canned backtest offline (fixture data) |

Determinism rule: every random process takes an explicit seed; tests pin seeds.

## Build & tooling

- pnpm workspaces; root scripts: `pnpm dev` (api + web concurrently),
  `pnpm test`, `pnpm build`, `pnpm lint`.
- ESLint + Prettier, strict TS (`noUncheckedIndexedAccess: true` — index into
  Float64Arrays is checked).
- CI (GitHub Actions): lint + typecheck + unit tests on PR. No Docker publishing
  until we actually deploy somewhere.
- Dev ports: web `5173`, api `8787`.
