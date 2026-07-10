# 00 — Overview

## What this is

A **trading research playground** for a small group of friends. It is a web app
where each of us can:

1. Watch US stocks/ETFs (and dabble in options) on a good-looking dashboard we
   can rearrange like a workbench.
2. Write trading strategies as small TypeScript plugins.
3. Backtest those strategies against historical data with honest cost modeling.
4. Stress-test results with Monte Carlo simulation — running on the GPU via
   WebGPU so 100k simulated paths render as a live fan chart, not a progress bar.
5. Run strategies forward in paper trading with fake money and a leaderboard.

## Goals

- **Modular first.** Adding a strategy or a dashboard widget must require zero
  changes to engine or shell code. Both are plugin registries.
- **Fast feedback.** A 10-year daily backtest completes in under a second; a
  100k-path Monte Carlo in about a second on GPU. Iteration speed is the product.
- **Honest numbers.** Fills at next bar's open, slippage and commission modeled,
  no look-ahead bias, walk-forward validation available. We are trying to
  disprove our strategies, not flatter them.
- **Looks good.** A single, committed dark theme with a validated palette
  (docs/02). If it looks like a school project we won't use it.
- **Cheap/free to run.** Free-tier data providers, local-first persistence,
  one small Node process. No paid infra required.

## Non-goals (hard boundaries)

- **No live trading. No broker keys. No real money.** The architecture has no
  order-routing layer and we are not adding one. Paper only.
- **No financial advice.** Signals are outputs of our own toy models.
- **No accounts/auth service in v1.** Local-first; the shared leaderboard is a
  tiny JSON sync via the API (docs/08). If it ever grows, revisit.
- **No mobile app.** Responsive enough to glance at on a phone, designed for
  a laptop/desktop with a real GPU.
- **No microservices.** One web app, one API process, shared packages.

## Who uses it

Us — a handful of technically comfortable friends. That means:

- Adding a strategy should feel like writing a ~50-line file and refreshing.
- Docs live in the repo; there is no external wiki.
- Breaking changes are fine if announced in the group chat; there are no
  external users.

## Core concepts (glossary)

| Term | Meaning here |
|---|---|
| **Bar** | One OHLCV candle: `{ t, o, h, l, c, v }`, UTC epoch-ms timestamp, for a symbol + timeframe. |
| **Timeframe** | Bar duration: `1m 5m 15m 1h 1D 1W`. Daily is the primary citizen. |
| **Signal** | A strategy's desire: target position (e.g. `+1` = fully long) with optional metadata. |
| **Order** | A concrete instruction produced from signals: market/limit/stop, qty, side. |
| **Fill** | The simulated execution of an order: price, qty, fees, timestamp. |
| **Position** | Current holding for a symbol: qty, average price, unrealized P&L. |
| **Strategy** | A plugin implementing the `Strategy` interface (docs/05). Pure, deterministic, serializable params. |
| **Backtest** | Replaying history bar-by-bar through a strategy + portfolio simulator (docs/06). |
| **Monte Carlo sim** | Thousands of randomized re-runs (bootstrapped returns or model-generated paths) producing distributions instead of point estimates (docs/07). |
| **Widget** | A dashboard panel implementing the `Widget` interface (docs/03): chart, table, tile, anything. |
| **Workspace** | A named arrangement of widgets on the grid, saved and shareable as JSON. |
| **Paper account** | A fake-money portfolio with simulated fills, tracked over time (docs/08). |

## Decisions already made (don't re-litigate casually)

| Decision | Why |
|---|---|
| TypeScript everywhere, no Python | One language, sims run client-side on WebGPU, everyone can touch everything. The old Python/Streamlit app is deleted (it's in git history). |
| Sims run **in the browser** | WebGPU gives us each a free GPU; no shared compute server to pay for or queue on. |
| CPU fallback is mandatory | WebGPU isn't universal; every sim has a Web-Worker CPU path with identical results semantics (docs/07). |
| Dark-only theme | One committed look done well beats two half-done. Palette is CVD-validated (docs/02). |
| Daily bars are the primary timeframe | Free data is reliable at daily resolution; intraday is supported but best-effort. |
| Local-first persistence | IndexedDB + localStorage; the API process is a stateless-ish data proxy with a cache. Nothing breaks if the API restarts. |

## Reading order for implementers

Architecture (01) → Design system (02) → Dashboard shell (03) → Data layer (04)
→ Strategy framework (05) → Backtest (06) → Monte Carlo (07) → Paper (08) →
Options (09) → Roadmap (10). The roadmap defines what "phase 1 done" means —
build in that order.
