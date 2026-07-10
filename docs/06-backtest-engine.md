# 06 — Backtest Engine

Event-driven, bar-by-bar, no look-ahead, honest costs. Lives in
`packages/backtest`, runs in a Web Worker (docs/01), pure TypeScript, fully
deterministic. This CPU engine is the **reference implementation** the GPU sims
are validated against (docs/07).

## 1. The loop

```
for i in warmup..N-1:
  1. mark portfolio to bar[i] close                 (equity curve point)
  2. check engine-managed stops/TPs against bar[i]   (H/L touch ⇒ exit fill *within* bar i, see §3)
  3. signal = strategy.onBar(ctx at i)               (bar i has CLOSED)
  4. if signal → generate order(s), queue them
  5. queued orders FILL AT BAR i+1 OPEN (+ slippage)  ← the no-look-ahead law
```

A signal computed on bar *i*'s close can never be filled at bar *i*'s prices.
Everything downstream (metrics, charts, paper mode) inherits honesty from this
one rule.

## 2. Run configuration

```ts
interface BacktestConfig {
  strategyId: string;
  params: unknown;                  // validated by the strategy's schema
  symbol: string;  timeframe: Timeframe;
  from: number; to: number;         // epoch ms; engine clamps to covered data & reports
  initialCash: number;              // default 100_000
  costModel: CostModel;
  sizing: SizingModel;
  allowShort: boolean;              // default false (cash account)
  seed: number;                     // for any stochastic component (slippage jitter off by default)
}

interface CostModel {
  commissionPerOrder: number;       // default 0 (retail zero-commission era)
  spreadBps: number;                // default 2 — half-spread paid on each side
  slippageBps: number;              // default 3 — adverse, always against you
}                                   // effective: fill = nextOpen * (1 ± (spreadBps/2 + slippageBps)/1e4)

type SizingModel =
  | { kind: 'all-in' }                          // target × full equity (default)
  | { kind: 'fixed-fraction'; fraction: number }// target × fraction × equity
  | { kind: 'fixed-dollar'; dollars: number }
  | { kind: 'volatility'; targetAnnualVol: number };  // scale by realized ATR-vol
```

Positions are whole shares (floor), cash can't go negative, shorts (when
enabled) reserve 150% margin — simple but not fantasy.

## 3. Fills

| Order | Fill rule |
|---|---|
| Market (from signal) | next bar open ± cost adjustment |
| Stop-loss (engine-managed) | if bar low ≤ stop (long): fill at `min(open, stop)` ± costs — gap-through fills at the gap, not the stop price |
| Take-profit | mirror of stop with high ≥ tp |

If both stop and TP are touched in one bar, the **stop wins** (pessimistic).
All pessimism biases are deliberate and documented here — when in doubt, the
engine under-reports performance.

## 4. Outputs (the `RunResult` artifact)

Content-addressed by `runId` (docs/01), stored in IndexedDB, schema in `core`:

```ts
interface RunResult {
  runId: string; config: BacktestConfig; startedAt: number; dataCoverage: { from: number; to: number };
  equity: Float64Array;        // per bar, marked at close
  returnsPerBar: Float64Array; // log returns of equity  ← Monte Carlo input (docs/07)
  drawdown: Float64Array;
  trades: Trade[];             // { entryT, exitT, side, qty, entryPx, exitPx, pnl, pnlPct, bars, reason }
  markers: SignalMarker[];     // for chart overlay: { t, side, reason }
  metrics: Metrics;
  logs: LogLine[];             // strategy ctx.log output, capped 5k lines
}
```

## 5. Metrics (definitions pinned here so nobody argues later)

Annualization uses 252 (daily) / by-timeframe factors from the calendar module.
Risk-free rate: 0 in v1 (compare strategies to each other and to buy-and-hold,
not to T-bills).

| Metric | Definition |
|---|---|
| Total return | equity_end / equity_start − 1 |
| CAGR | (end/start)^(1/years) − 1 |
| Volatility | stdev(returnsPerBar) × √annFactor |
| Sharpe | mean(returnsPerBar)/stdev(returnsPerBar) × √annFactor |
| Sortino | same with downside deviation (returns < 0 only) |
| Max drawdown | min over t of equity/cummax(equity) − 1 |
| Calmar | CAGR / |maxDD| |
| Win rate | winning trades / total trades |
| Profit factor | gross profits / gross losses |
| Avg trade | mean(trade.pnlPct) |
| Exposure | fraction of bars with a position |
| Turnover | annualized traded notional / avg equity |
| Trade count | — |

Every result view shows the strategy **and** `buy-and-hold` on the same range
with the same cost model. That comparison is not optional.

## 6. Parameter sweeps

- Grid sweep over any numeric params (bounds from the zod schema): UI picks
  ranges/steps, engine enumerates configs, worker pool executes (docs/01),
  results stream in and paint a **heatmap** (2 params) or small-multiple table.
- Budget guard: > 2 000 combos requires confirmation; > 20 000 refused (that's
  what the coarse→fine workflow is for).
- Sweep results feed **overfitting honesty**: the heatmap uses the sequential
  ramp so you *see* whether a peak is a plateau (robust) or a spike (curve-fit).

## 7. Walk-forward validation

- Split range into K folds (default 5): optimize params on fold train window
  (in-sample), evaluate best params on the following out-of-sample window,
  stitch OOS segments into one OOS equity curve.
- Report: IS vs OOS metric table + stitched curve vs full-sample-optimized
  curve. If OOS Sharpe < ½ × IS Sharpe, the UI badges the run "likely overfit"
  (`--warn`). Blunt, but it's the lesson we're here to learn.

## 8. Testing the engine itself

- Golden runs: fixture bars + `sma-cross` defaults → exact expected trades,
  equity, and metrics committed as JSON. Any diff fails CI.
- Property tests: costs ≥ 0 ⇒ equity ≤ zero-cost equity; buy-and-hold equity
  matches closed-form; no trade entry/exit outside data range; determinism
  (two runs bit-identical).
- Pathological fixtures: gaps, halts (missing bars), splits (adjusted data),
  one-bar data, all-NaN warmup.
