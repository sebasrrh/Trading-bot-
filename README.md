# Tradeboard

A modular trading research dashboard for our group — build strategies as plugins,
backtest them, stress-test them with GPU-accelerated Monte Carlo sims, and race
each other in paper trading. **No real money. Ever.** (See non-goals in
[docs/00-overview.md](docs/00-overview.md).)

> **Status: spec phase.** This repo currently contains the full written
> specification and the empty package skeleton. Implementation follows the
> phases in [docs/10-roadmap.md](docs/10-roadmap.md).

## The stack (decided)

| Concern | Choice |
|---|---|
| Language | TypeScript everywhere |
| Frontend | React 19 + Vite |
| Compute | WebGPU (WGSL compute shaders) with a Web-Worker CPU fallback |
| Charts | `lightweight-charts` (candles), `uPlot` (dense lines), custom SVG (tiles/heatmaps) |
| State / data | Zustand + TanStack Query, IndexedDB bar cache |
| API | Small Node (Hono) proxy for market data + SQLite cache |
| Monorepo | pnpm workspaces |
| Markets | US stocks/ETFs, basic options (modeled) |
| Trading | Paper + backtest only |
| Theme | Dark-only "modern dark fintech" (validated palette in docs/02) |

## Repo layout

```
apps/
  web/         React dashboard (widget shell, charts, all UI)
  api/         Node data proxy: market-data providers, cache, rate limiting
packages/
  core/        Shared domain types (Bar, Signal, Order, Position...), time & math utils
  data/        Provider adapters + client-side data access (shared by web and api)
  indicators/  Technical indicators as pure, incremental functions
  strategies/  Strategy plugin interface + built-in strategies
  backtest/    Event-driven backtest engine (CPU reference implementation)
  sim/         Monte Carlo engine: WebGPU kernels + CPU fallback, one interface
  paper/       Paper-trading engine: accounts, fills, P&L, leaderboard
  ui/          Design system: tokens, primitives, chart chrome
docs/          The full spec — start at 00-overview.md
```

## The docs (read in order)

| Doc | What it specifies |
|---|---|
| [00-overview.md](docs/00-overview.md) | Vision, goals, non-goals, glossary |
| [01-architecture.md](docs/01-architecture.md) | Monorepo, package boundaries, data flow |
| [02-design-system.md](docs/02-design-system.md) | **Color palette**, typography, spacing, components, chart rules |
| [03-dashboard-shell.md](docs/03-dashboard-shell.md) | Modular widget system, layout grid, linking |
| [04-data-layer.md](docs/04-data-layer.md) | Providers, schemas, caching, rate limits |
| [05-strategy-framework.md](docs/05-strategy-framework.md) | Strategy plugin API — how to add one |
| [06-backtest-engine.md](docs/06-backtest-engine.md) | Event loop, fills, costs, metrics |
| [07-monte-carlo-webgpu.md](docs/07-monte-carlo-webgpu.md) | WGSL kernels, RNG, fan charts, CPU fallback |
| [08-paper-trading.md](docs/08-paper-trading.md) | Paper accounts, order sim, leaderboard |
| [09-options.md](docs/09-options.md) | Black-Scholes, Greeks, chains, payoff graphs |
| [10-roadmap.md](docs/10-roadmap.md) | Build order, phase acceptance criteria |
| [11-deployment.md](docs/11-deployment.md) | Docker image, publish workflow, hosting the leaderboard |

## Ground rules for contributors (that's us)

1. **New strategies are plugins.** You never touch the engine to add one — see docs/05.
2. **New dashboard views are widgets.** You never touch the shell to add one — see docs/03.
3. **Determinism.** Backtests and sims are seeded and reproducible; same inputs → same outputs, bit-for-bit on CPU, tolerance-checked on GPU.
4. **Colors come from tokens.** No hex in components; the palette in docs/02 is validated — re-run the validator if you change it.
5. **No secret keys in the frontend.** Provider API keys live only in `apps/api` env.
