# 04 — Data Layer

Free-tier market data, normalized once, cached twice. The browser talks only to
`apps/api`; `apps/api` talks to providers. **Provider keys never reach the
frontend.**

## 1. Provider strategy

One interface, several adapters, priority-ordered with fallback. All in
`apps/api/src/providers/` implementing:

```ts
// packages/data/src/provider.ts
export interface MarketDataProvider {
  id: 'alpaca' | 'polygon' | 'stooq' | 'yahoo';
  capabilities: Set<'bars-1D' | 'bars-intraday' | 'quote' | 'search' | 'options-chain'>;
  getBars(req: BarsRequest): Promise<Bar[]>;        // normalized, ascending, UTC
  getQuotes(symbols: string[]): Promise<Quote[]>;   // batched!
  search?(q: string): Promise<SymbolInfo[]>;
  getOptionsChain?(underlying: string): Promise<OptionsChain>;
}
```

| Priority | Provider | Free tier | Used for |
|---|---|---|---|
| 1 | **Alpaca** (Market Data, IEX feed) | 200 req/min, real-time-ish IEX quotes, bars all timeframes | primary: bars, quotes |
| 2 | **Polygon.io** | 5 req/min, EOD + 2y intraday history | backfill, options chain snapshots |
| 3 | **Stooq** | unlimited CSV, EOD only | deep daily history (20y+), zero-key fallback |
| 4 | **Yahoo (unofficial)** | unofficial JSON endpoints | last-resort fallback + options chains; expect breakage, keep adapter isolated |

- Keys via env: `ALPACA_KEY_ID/ALPACA_SECRET`, `POLYGON_KEY` — `.env` in
  `apps/api`, documented in `apps/api/.env.example`. Everything must still work
  key-less (Stooq daily + Yahoo) so a friend can clone and run with zero setup.
- The router picks the highest-priority provider whose `capabilities` cover the
  request and whose rate budget has headroom; on error/timeout (3s) it falls
  through. Response carries `X-Data-Source: alpaca` and `X-Data-Fresh-At`; the
  UI badge (docs/03 §1) shows it, `--warn` when serving stale cache or a
  fallback provider.

## 2. Canonical schemas (zod, in `packages/core`)

```ts
Bar        { t: number /* epoch ms UTC, bar OPEN time */, o, h, l, c, v: number }
Quote      { symbol, price, ts, change, changePct, prevClose }         // delayed ok
SymbolInfo { symbol, name, exchange, type: 'stock'|'etf' }
Timeframe  '1m'|'5m'|'15m'|'1h'|'1D'|'1W'
BarsRequest{ symbol, timeframe, from, to /* epoch ms */ }
OptionsChain — see docs/09
```

Normalization rules (enforced in one place, `packages/data/normalize.ts`, with
golden tests per provider fixture):

- Timestamps → UTC epoch ms of bar open. Daily bars stamped at 00:00 ET → UTC.
- **Adjusted prices only** (splits + dividends) for anything the backtester
  consumes. Store `adjusted: true` on the series; if a provider only returns
  unadjusted, apply its adjustment factors or reject to fallback. A backtest on
  unadjusted data is silently wrong — this is non-negotiable.
- Ascending `t`, dedup on `t` (last write wins), gaps left as gaps (no fill).
- Symbols uppercased; class shares normalized to dot form (`BRK.B`).

## 3. API surface (`apps/api`, Hono)

```
GET /api/bars?symbol=SPY&timeframe=1D&from=…&to=…   → { source, adjusted, bars: Bar[] }
GET /api/quotes?symbols=SPY,QQQ,AAPL                 → { source, quotes: Quote[] }   (batched only)
GET /api/search?q=app                                → SymbolInfo[]  (cached 24h)
GET /api/chain?underlying=SPY&expiry=…               → OptionsChain (docs/09)
GET /api/health                                      → provider status + budgets
POST/GET /api/leaderboard                            → docs/08
```

- All responses zod-validated on the way out (dev mode) and on the way in
  (browser, always). Hono typed client gives `apps/web` full types.
- CORS locked to the web origin. No auth in v1 (LAN/local usage); the
  leaderboard route gets a shared secret in env when we host it (docs/08).

## 4. Server cache (SQLite via `better-sqlite3`)

```
bars(symbol, timeframe, t, o, h, l, c, v, source, adjusted, PRIMARY KEY(symbol, timeframe, t))
meta(symbol, timeframe, first_t, last_t, fetched_at)
symbols(symbol, name, exchange, type, updated_at)
```

- Read path: serve from SQLite; fetch only the missing head/tail range from the
  provider ("range subtraction"), upsert, return merged. Historical bars are
  immutable — cache forever. The current (partial) bar is never cached.
- Quotes cached 5s in-memory (LRU). Search cached 24h in `symbols`.
- Rate limiting: token bucket per provider (config per adapter), plus a global
  in-flight dedup so 4 widgets asking for SPY 1D concurrently = 1 provider call.
- DB file `apps/api/data/cache.db`, gitignored. `pnpm api:nuke-cache` drops it.

## 5. Browser data access (`packages/data` client + TanStack Query)

- Hooks: `useBars(symbol, tf, range)`, `useQuotes(symbols)`, `useSymbolSearch(q)`,
  `useOptionsChain(underlying)`.
- `useBars` returns a **`BarSeries`** (columnar, docs/01) — conversion happens
  once in the query `select`.
- IndexedDB persistence (TanStack Query persister) for bar queries only —
  dashboard renders instantly offline/from cache while revalidating. Quotes are
  never persisted.
- Polling: quotes refetch every 5s while any consuming widget is visible
  (`refetchInterval` + page-visibility pause). Bars for the live chart refetch
  the current bar every 30s intraday / on demand for daily.
- Batching: a tiny request-collapser gathers all `useQuotes` symbols mounted in
  the same tick into one `/api/quotes` call.

## 6. Historical depth targets

| Timeframe | Target depth | Source reality |
|---|---|---|
| 1D | 20+ years | Stooq/Alpaca fine |
| 1h | 2 years | Alpaca/Polygon |
| 1m–15m | 30–90 days | best-effort, Alpaca |

Backtests validate that the requested range is actually covered and **fail
loudly with the covered range** rather than silently testing a shorter window.

## 7. Market calendar

`packages/core/src/calendar.ts`: NYSE sessions, half-days, holidays (static
table, updated yearly). Used for: x-axis gap collapsing, "is market open" badge,
scheduling paper-trading marks (docs/08), and annualization constants
(252 trading days) in metrics (docs/06).
