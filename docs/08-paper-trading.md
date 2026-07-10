# 08 — Paper Trading

Fake money, real discipline, public shame. `packages/paper` runs strategies (or
manual clicks) forward against live-ish quotes, tracks P&L honestly, and feeds a
leaderboard so the group can compete.

## 1. Model

```ts
interface PaperAccount {
  id: string; owner: string;            // display name, e.g. 'bodhi'
  createdAt: number; startingCash: number;   // fixed 100_000 for leaderboard fairness
  cash: number;
  positions: Position[];                // { symbol, qty, avgPrice }
  openOrders: PaperOrder[];
  history: Fill[];                      // every fill ever, append-only
  equityMarks: { t: number; equity: number }[];  // daily close marks
  autoStrategies: AutoStrategyBinding[];         // §4
}

type PaperOrder = {
  id: string; symbol: string; side: 'buy' | 'sell';
  qty: number;
  type: 'market' | 'limit' | 'stop';
  limitPrice?: number; stopPrice?: number;
  tif: 'day' | 'gtc';
  placedAt: number; note?: string;      // journal note at order time
};
```

- Long-only cash account in v1 (matches backtester default). No margin, no
  shorting — keeps leaderboard math honest and the fill sim simple.
- Whole shares. Cash floor at 0.

## 2. Fill simulation (client-side, quote-driven)

- Market orders fill immediately at `quote.price` **plus the same cost model
  as backtests** (docs/06 §2 defaults) — half-spread + slippage against you.
  Market-closed market orders queue and fill at next session's first quote.
- Limit/stop orders are checked against each quote poll (5s, docs/04) while
  the app is open, and reconciled against the session's bar range
  (high/low) on next app open — so an order you left overnight fills if the
  bar range crossed it, at the limit price (or gap price for stops, same
  pessimism rule as docs/06 §3).
- Every fill appends to `history` with price, costs, and the quote timestamp
  it filled against. No edits, no deletes — the journal is append-only
  (mistakes are data).

## 3. Persistence & the "offline problem"

- Local-first: the account lives in IndexedDB. Closing the laptop just pauses
  order checking; reconciliation on reopen (using cached daily bars) catches up
  fills and appends missed daily equity marks.
- Daily equity mark: first app-open after each market close computes
  mark-to-market equity from that day's close prices.

## 4. Strategy auto-trading (paper autopilot)

```ts
interface AutoStrategyBinding {
  strategyId: string; params: unknown; symbol: string;
  allocation: number;        // fraction of account this binding may use, 0–1
  enabled: boolean;
}
```

- On each **completed daily bar** (first open after close), the binding runs
  `strategy.onBar` over the updated series and converts the signal into paper
  orders (same signal→order logic as the backtester, `allocation` as the
  sizing fraction) that fill at next open — i.e. **paper autopilot has exactly
  backtest semantics**, one bar behind reality, by construction.
- The dashboard `orders-journal` widget tags autopilot fills with the strategy
  badge so manual and bot trades are distinguishable.
- Daily-bars only in v1 (intraday autopilot needs an always-on process —
  explicitly out of scope; noted for v2 as an api-side scheduler).

## 5. Leaderboard (the social feature)

- `apps/api` gets two tiny routes backed by a `leaderboard` SQLite table:
  - `POST /api/leaderboard` — `{ owner, accountId, equityMarks (last 90d), totalReturn, maxDD, sharpe }`,
    pushed opportunistically whenever the app computes a new daily mark.
    Shared-secret header (env `LEADERBOARD_KEY`) when hosted; open on LAN.
  - `GET /api/leaderboard` — everyone's latest summary + sparkline series.
- `leaderboard` widget (docs/03): rank by total return since account creation,
  DeltaChip for the day, 90-day sparkline, `--warn` badge if maxDD < −30%
  ("degen flag"). Ties broken by Sharpe, because we're classy.
- Trust model: friends. No auth beyond the shared secret; we can see each
  other's numbers, that's the point. Resetting your account restarts your
  `createdAt` — the widget shows account age so a fresh reset is visible.

## 6. UI (Paper view)

- Header StatTiles: equity, day P&L, total return vs SPY-since-inception,
  cash, buying power.
- Order ticket: symbol (linked to context channel A), side toggle
  (buy = `--gain` fill, sell = `--loss` fill), qty with "25/50/100%" chips,
  type, limit/stop price, note field. Confirm shows estimated cost incl.
  modeled slippage.
- Positions + orders-journal widgets (docs/03), fixed arrangement plus the
  leaderboard rail.
- Kill switch per autopilot binding, and a global "flatten all" (market-sell
  everything) behind a confirm dialog.
