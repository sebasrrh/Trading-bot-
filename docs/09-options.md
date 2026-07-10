# 09 — Options (dabble-grade)

"A little bit of options": price them, see Greeks, build simple positions, and
paper-trade defined-risk structures. **Model-first** — free options *market*
data is unreliable, so v1 leans on Black-Scholes with real underlying data, and
treats live chain quotes as best-effort garnish.

## 1. Scope

| In (v1) | Out (explicitly) |
|---|---|
| Long calls/puts, covered calls, cash-secured puts, verticals | Naked short options, ratio/exotic structures |
| Black-Scholes pricing + Greeks, IV solving | American early-exercise premium (note the bias, price European) |
| Payoff diagrams (expiry + T+n model curves) | Realistic assignment simulation |
| MC pricing via docs/07 GBM paths (path-dependent payoffs later) | Vol surface fitting |
| Paper trading the above with model marks | Options backtesting (needs historical chains = paid data) |

## 2. Math (`packages/core/src/options/`)

Pure functions, golden-tested against published table values:

```ts
bsPrice({ S, K, T, r, sigma, type }): number            // Black-Scholes-Merton, q=0 v1
bsGreeks(...): { delta, gamma, theta, vega, rho }       // closed-form
impliedVol({ price, S, K, T, r, type }): number | null  // Newton w/ vega, bisection fallback,
                                                        // null if outside no-arb bounds
realizedVol(series: BarSeries, lookback = 21): number   // close-to-close, annualized — default sigma
```

- `r`: 13-week T-bill constant in config (`0.04` default), user-overridable in
  Settings. `T` in year fractions from the market calendar.
- MC cross-check: GBM engine (docs/07 mode B) pricing a vanilla within the MC
  standard error of `bsPrice` is a permanent CI test — it validates both.

## 3. Chain data

- `GET /api/chain?underlying=SPY` via Yahoo-unofficial adapter (docs/04):
  expiries, strikes, bid/ask/last, OI, volume. Cached 15 min, clearly badged
  with fetch time; missing/stale chain ⇒ UI falls back to **model mode**
  (strikes generated ±30% around spot, prices from BS at realized vol) with a
  visible "modeled" badge — never silently mix real and modeled quotes.
- Schema: `OptionsChain { underlying, spot, fetchedAt, expiries: [{ expiry, strikes: [{ K, call: OptionQuote|null, put: OptionQuote|null }] }] }`.

## 4. UI

**Chain widget** (`analysis` category): expiry tabs, strikes table centered on
ATM, calls left / puts right, IV column (solved from mid; falls back to model
vol), delta shading via the sequential ramp at 25% alpha for moneyness. Row
click adds the leg to the position builder.

**Payoff widget** (`payoff`, docs/03): the star.

- Legs list (≤ 4): type, strike, expiry, qty, debit/credit each.
- Chart: P&L vs underlying price at expiry (solid, 2px, gain/loss area fills
  above/below zero at 10% alpha) + T+0 and T+½ model curves (dashed,
  `--series-1`/`--series-2`). Breakevens, max profit/loss as labeled reference
  lines; current spot as a vertical marker.
- StatTile row: net debit/credit, max loss, max profit, breakeven(s),
  position Greeks (sum of legs).
- "Probability of profit" via docs/07 GBM sim (paths ending beyond breakeven),
  with seed shown — our two showpieces composing.

## 5. Paper integration

- Paper orders may carry option legs (`symbol` becomes an OCC-style contract
  id, e.g. `SPY 2026-09-18 C 550`). Fills at chain mid ± modeled half-spread
  (options `spreadBps` default 100 — wide, honest) or BS price in model mode.
- Marks: positions marked daily at model price with IV frozen at entry unless
  a fresh chain quote exists (badged which). Expiry: auto-settle intrinsic at
  underlying close; covered-call assignment removes shares at strike.
- Leaderboard equity includes option marks; the "modeled marks %" of an
  account is visible on the leaderboard tooltip (so a friend gaming model
  marks gets clowned appropriately).
