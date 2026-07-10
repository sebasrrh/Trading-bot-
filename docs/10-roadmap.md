# 10 — Roadmap

Build order with acceptance criteria. Each phase is demoable to the group chat;
don't start a phase until the previous one's checklist is green. Estimates
assume nights-and-weekends pace.

## Phase 0 — Skeleton (a weekend)

Scaffold the monorepo exactly as docs/01: pnpm workspaces, TS strict configs,
Vite app boots, Hono api boots, `packages/ui` tokens.css from docs/02, ESLint/
Prettier/Vitest wired, CI running lint+typecheck+test.

- [ ] `pnpm dev` starts web (5173) + api (8787); web renders the app shell
      (topbar, nav rail, empty grid) with the design tokens applied
- [ ] `pnpm test` green in CI on a PR
- [ ] `core` schemas: Bar, Quote, Timeframe, BarSeries + golden tests

## Phase 1 — Data + first real dashboard (1–2 weeks)

docs/04 data layer (Stooq + Yahoo keyless first, Alpaca behind env), docs/03
grid with the first four widgets: `candles`, `watchlist`, `stat-tile`,
`quote-strip`. Workspaces persist. ⌘K symbol search.

- [ ] Clone → `pnpm dev` with **zero API keys** shows a live SPY dashboard
- [ ] Bars cached: second load of a chart is instant, offline reload works
- [ ] Widgets: add/remove/drag/resize/configure; layout survives refresh
- [ ] Link channels work: click a watchlist row, candles follow
- [ ] Data-source badge truthful (provider + freshness, warn on fallback)

## Phase 2 — Strategies + backtesting (2–3 weeks)

docs/05 + docs/06 complete: indicators package, strategy registry with the five
built-ins, backtest engine on the worker pool, Backtest Lab view, strategy
widgets (`equity-curve`, `drawdown`, `run-metrics`), signal markers on candles.

- [ ] 10y daily backtest of `sma-cross` on SPY completes < 1s, UI never jams
- [ ] Golden-run tests + conformance suite green; runs content-addressed &
      cached; identical re-run is instant
- [ ] Every result screen shows buy-and-hold comparison + cost model used
- [ ] Param sweep heatmap (fast × slow) renders with the sequential ramp
- [ ] Walk-forward report with the "likely overfit" badge logic

## Phase 3 — Monte Carlo on WebGPU (2–3 weeks)

docs/07: Philox RNG (TS + WGSL), bootstrap + GBM kernels, CPU fallback worker,
Sim Lab view, `mc-fan` + `mc-histogram` widgets, parity + statistical tests.

- [ ] 50k × 252 bootstrap ≤ 300ms on GPU, ≤ 5s CPU fallback, progress streamed
- [ ] Same seed ⇒ CPU/GPU parity within documented tolerance (CI-checked)
- [ ] Fan chart to spec (docs/02 §5.7); ruin-prob StatTile; pin-to-dashboard
- [ ] GBM E[S_T]/Var golden test green — permanent
- [ ] Chrome-without-WebGPU demo path: badge says CPU, everything still works

## Phase 4 — Paper trading + leaderboard (2 weeks)

docs/08: paper account, order ticket, quote-driven fills, reconciliation,
daily marks, autopilot bindings, leaderboard api + widget.

- [ ] Place market/limit orders; costs applied; journal append-only
- [ ] Overnight limit order fills correctly on reopen (reconciliation test)
- [ ] Autopilot `sma-cross` places the same orders a backtest would (semantics test)
- [ ] Leaderboard shows ≥ 2 real friends' accounts with sparklines
- [ ] Docker image + publish workflow live (docs/11) and one hosted instance
      running the shared leaderboard

## Phase 5 — Options (2 weeks)

docs/09: BS math + golden tests, chain endpoint + widget, payoff builder,
MC probability-of-profit, paper option legs with model marks.

- [ ] BS vs MC cross-validation test permanent in CI
- [ ] Build a covered call; payoff chart with breakevens; paper-trade it to expiry settlement
- [ ] Chain gracefully degrades to model mode with visible badge

## Phase 6 — Polish & stretch (ongoing)

In rough priority order:

1. In-browser strategy editor (esbuild-wasm sandbox, docs/05 §6)
2. Mode C sims: strategy-on-synthetic-paths (docs/07)
3. Multi-symbol portfolio backtests (rotation universe > 1 holding)
4. Intraday autopilot via api-side scheduler
5. Workspace share-links hosted by api instead of paste-strings
6. Light theme (tokens are role-based, so this is a token sheet + validator run,
   not a redesign)
7. GPU param sweeps (backtest kernel in WGSL — only if sweep sizes ever hurt)

## Standing rules

- Every phase PR updates the relevant doc if reality diverged from spec —
  **docs are the contract**, stale docs are bugs.
- Never merge red CI. Never commit an API key. Never add a live-trading code path.
