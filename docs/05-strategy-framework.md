# 05 — Strategy Framework

The whole point of the rebuild: **a strategy is a small plugin file**. The
engine, dashboard, backtester, Monte Carlo, and paper trader all consume the
same interface. Nobody edits engine code to add an idea.

## 1. The contract

```ts
// packages/strategies/src/types.ts
import type { ZodType } from 'zod';
import type { BarSeries, Signal } from '@tradeboard/core';

export interface StrategyDef<P> {
  id: string;                 // 'sma-cross' — stable forever, kebab-case
  name: string;               // 'SMA Crossover'
  description: string;        // one paragraph: the idea + when it should work
  paramsSchema: ZodType<P>;   // zod schema WITH .default() on every field and
                              // .min/.max so the UI can render sliders/sweeps
  warmupBars: (params: P) => number;   // bars consumed before first signal
  create: (params: P) => StrategyInstance;
}

export interface StrategyInstance {
  /** Called once per bar, in order, AFTER the bar has closed.
   *  Return the desired target exposure, or null for "no opinion" (hold). */
  onBar(ctx: StrategyContext): Signal | null;
  /** Optional: serialize internal state (for inspection/debugging only). */
  snapshot?(): Record<string, unknown>;
}

export interface StrategyContext {
  readonly i: number;                    // current bar index
  readonly bars: BarSeries;              // FULL series, but…
  readonly close: (offset?: number) => number;  // ctx.close(0)=current, (1)=prev — guarded
  readonly indicator: IndicatorAccessor; // memoized: ctx.indicator.sma(20), .rsi(14),
                                         // .ema(50), .atr(14), .bbands(20,2), …
  readonly position: PositionView;       // current qty, avgPrice, unrealizedPnl
  readonly equity: number;               // current account equity
  readonly log: (msg: string, data?: unknown) => void;  // shows in run inspector
}

export type Signal = {
  target: number;        // desired exposure: +1 full long … 0 flat … -1 full short
                         // (fractional fine: 0.5 = half the allowed allocation)
  reason?: string;       // 'fast crossed above slow' — shown on chart markers
  stopLoss?: number;     // optional protective price levels the engine manages
  takeProfit?: number;
};
```

### Hard rules (the engine enforces what it can)

1. **No look-ahead.** `ctx` only exposes bars `≤ i`. Accessing `bars.c[i+1]`
   is physically prevented (the context hands out a length-capped view in dev
   mode; sampled asserts in release).
2. **Deterministic.** No `Date.now`, no `Math.random` (a seeded `ctx.random()`
   can be added if a strategy truly needs noise), no `fetch`, no globals. Same
   bars + params ⇒ same signals, always. This is what makes runs
   content-addressable (docs/01) and Monte Carlo meaningful.
3. **Params are data.** Everything tunable is in `paramsSchema` — that is what
   the auto-generated config UI, the sweep runner, and the run hash consume.
4. **Signals, not orders.** Strategies express *desired exposure*; sizing,
   order generation, and fills belong to the portfolio layer (docs/06). This
   keeps strategies comparable and lets one cost model serve all.

## 2. Registry

```ts
// packages/strategies/src/registry.ts
export const strategyRegistry = defineStrategies([
  smaCross, rsiMeanRevert, momentumRotation, donchianBreakout, buyAndHold,
]);
```

- `buyAndHold` is required forever — it is the benchmark every result screen
  shows alongside your strategy, uninvited.
- UI (Backtest Lab, widget configs) lists strategies from the registry with
  auto-generated param forms (zod schema → sliders/number inputs, defaults
  pre-filled).
- **Adding a strategy** (this goes in the README too):
  1. `packages/strategies/src/builtin/my-idea.ts` — export a `StrategyDef`.
  2. Add to `registry.ts` (one line).
  3. `pnpm test --filter strategies` — the shared conformance suite runs your
     def automatically (schema has defaults, warmup respected, deterministic
     across two runs, no NaN signals).
  4. It now appears everywhere: Backtest Lab dropdown, chart signal overlays,
     sweeps, Monte Carlo, paper trading.

## 3. Built-ins (v1) — also the documentation-by-example

| id | Idea | Params (defaults) |
|---|---|---|
| `buy-and-hold` | Benchmark | — |
| `sma-cross` | Long when fast SMA > slow SMA | fast 20, slow 50 |
| `rsi-mean-revert` | Long when RSI < low, exit at mid | period 14, low 30, exit 50 |
| `donchian-breakout` | Long on N-day-high break, trail stop at M-day low | entry 55, exit 20 |
| `momentum-rotation` | Hold the strongest of a symbol list by k-bar return, monthly | lookback 126, top 1 |

`sma-cross` in full, as the canonical example:

```ts
export const smaCross: StrategyDef<{ fast: number; slow: number }> = {
  id: 'sma-cross',
  name: 'SMA Crossover',
  description: 'Long when the fast SMA is above the slow SMA, flat otherwise. Trend-following; whipsaws sideways markets.',
  paramsSchema: z.object({
    fast: z.number().int().min(2).max(200).default(20),
    slow: z.number().int().min(5).max(400).default(50),
  }).refine(p => p.fast < p.slow, 'fast must be < slow'),
  warmupBars: p => p.slow,
  create: p => ({
    onBar(ctx) {
      const fast = ctx.indicator.sma(p.fast);
      const slow = ctx.indicator.sma(p.slow);
      const target = fast > slow ? 1 : 0;
      if (target !== (ctx.position.qty > 0 ? 1 : 0))
        return { target, reason: target ? 'fast>slow' : 'fast<slow' };
      return null;
    },
  }),
};
```

## 4. Indicators (`packages/indicators`)

- Pure incremental implementations over columnar `BarSeries`; each returns a
  `Float64Array` aligned to bars with `NaN` during warmup: `sma ema wma rsi
  macd atr bbands stochastic donchian obv vwap zscore returns`.
- `IndicatorAccessor` in the strategy context memoizes per (indicator, params)
  per run, so ten strategies asking for `sma(20)` compute it once.
- Golden tests: fixtures with hand-verified expected values; parity-checked
  against a reference implementation output committed to the repo.
- Chart overlays reuse the exact same functions (no separate "UI indicator"
  math — one implementation, one truth).

## 5. Multi-symbol strategies

`momentum-rotation` needs several series. The context generalizes:

- A strategy's def may declare `universe: 'single' | 'list'`. For `list`, the
  runner supplies `ctx.symbols: string[]` and `ctx.barsFor(symbol)` (aligned on
  the union calendar; missing bars = NaN, and `ctx.aligned(i)` tells you if all
  symbols have data at bar i).
- Signals become per-symbol: `{ targets: Record<string, number>, reason? }`.
- v1 keeps this minimal — enough for rotation strategies; pairs/stat-arb later.

## 6. Where user strategies live

- v1: in-repo under `packages/strategies/src/builtin/` — we're all committers;
  a strategy PR is the review-and-trash-talk mechanism.
- v2 (roadmap): "custom strategy" editor in the UI compiling TS in-browser
  (esbuild-wasm) into a sandboxed Worker, stored in IndexedDB, same interface.
  The interface above is designed so this needs no engine change.
