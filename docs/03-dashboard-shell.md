# 03 — Dashboard Shell & Widget System

The dashboard is a **workbench**: a grid of widgets the user arranges, resizes,
configures, and saves as named workspaces. Adding a new view to the product =
writing one widget module + registering it. The shell never changes.

## 1. App shell

```
┌──────────────────────────────────────────────────────────────┐
│ Topbar: logo · workspace switcher · symbol search (⌘K) ·      │
│         global timeframe/range · data-source badge · settings │
├────────┬─────────────────────────────────────────────────────┤
│ Nav    │                                                     │
│ rail   │              Widget grid (this doc)                 │
│ 56px   │                                                     │
│        │                                                     │
│ 📊 Dash│                                                     │
│ 🧪 Lab │   (Backtest Lab, Sim Lab, Paper, Settings are       │
│ 📈 Sim │    full-page views on the same shell)               │
│ 💰 Papr│                                                     │
└────────┴─────────────────────────────────────────────────────┘
```

- **Nav rail** (icons + tooltips): Dashboard, Backtest Lab, Sim Lab, Paper
  Trading, Settings. Full-page views may still embed widgets (e.g. the Backtest
  Lab results area is a fixed widget arrangement).
- **⌘K command palette**: symbol search, "add widget…", "switch workspace…",
  "run backtest…". Single entry point for everything keyboard.
- **Data-source badge**: which provider served current data + freshness
  (docs/04); turns `--warn` when stale/fallback.

## 2. The grid

- Library: `react-grid-layout` (12 columns, `rowHeight: 32px`, gutter 12px,
  vertical compaction). Breakpoints: `lg ≥ 1200px` 12 cols, `md ≥ 768px` 8 cols,
  `sm < 768px` single column stack (read-only layout on mobile).
- Widgets declare min/max/default spans (in grid units) in their manifest; the
  grid enforces them.
- Edit interactions: drag by widget header, resize by corner handle (visible on
  hover), remove & duplicate in the widget's `⋯` menu. Layout changes persist
  immediately (debounced 500ms).

## 3. Widget interface (the contract)

```ts
// packages/core/src/widget.ts — the shape; registry lives in apps/web/src/widgets
import type { ZodType } from 'zod';

export interface WidgetManifest<Config> {
  id: string;                    // 'candles', unique, kebab-case, stable forever
  name: string;                  // 'Price Chart'
  description: string;           // one sentence for the Add Widget gallery
  icon: LucideIconName;
  category: 'markets' | 'portfolio' | 'strategy' | 'analysis';
  size: {
    min: { w: number; h: number };      // grid units
    default: { w: number; h: number };
    max?: { w: number; h: number };
  };
  configSchema: ZodType<Config>;        // parsed on load; defaults via .default()
  ConfigPanel?: React.FC<ConfigPanelProps<Config>>; // optional custom form;
                                        // else auto-generated from the schema
  Component: React.FC<WidgetProps<Config>>;
}

export interface WidgetProps<Config> {
  instanceId: string;
  config: Config;
  setConfig: (patch: Partial<Config>) => void;
  context: WidgetContext;        // resolved symbol/timeframe/range (see §5)
  size: { wPx: number; hPx: number }; // live pixel size — widgets adapt, not media-query
}
```

Rules:

- A widget gets data **only** through the shared hooks (`useBars`, `useQuote`,
  `useRunResult`, …) — never raw `fetch`. That keeps caching/dedup central.
- A widget renders skeleton / empty / error states itself (ui package provides
  them); the shell only draws the card frame and header.
- Widgets are lazy-loaded (`React.lazy`) so the dashboard bundle stays small.
- `id` and `configSchema` are versioned-compatible: adding optional fields is
  fine; renames require a `migrate(oldConfig)` export.

Registration:

```ts
// apps/web/src/widgets/registry.ts
export const widgetRegistry = defineWidgets([
  candlesWidget, statTileWidget, watchlistWidget, /* ... */
]);
```

Adding a widget = one folder in `apps/web/src/widgets/<id>/` + one line here.

## 4. Built-in widgets (v1 set)

| id | Name | Category | Default span (w×h) | Notes |
|---|---|---|---|---|
| `candles` | Price Chart | markets | 8×10 | lightweight-charts; candles+volume; indicator overlays (SMA/EMA/BB) & signal markers from a chosen strategy run; config: symbol-or-linked, indicators, timeframe override |
| `watchlist` | Watchlist | markets | 4×10 | DataTable: last, Δ%, sparkline (30d); rows set the link-channel symbol on click; config: symbol list, channel |
| `stat-tile` | Stat Tile | markets | 3×4 | One StatTile: price, market cap, P&L of an account, or any metric of a run; config: metric picker |
| `quote-strip` | Quote Strip | markets | 12×2 | Horizontal ticker chips across the top |
| `equity-curve` | Equity Curve | strategy | 6×8 | uPlot; one or more run results indexed to 100; ≤ 6 series, palette slots in order |
| `drawdown` | Drawdown | strategy | 6×6 | Underwater plot (area, loss hue at 25% alpha) for selected run(s) |
| `run-metrics` | Run Metrics | strategy | 4×8 | Metric table for selected runs, side-by-side columns, best value per row subtly accent-washed |
| `mc-fan` | Monte Carlo Fan | analysis | 8×10 | Fan chart per docs/02 §5.7 + ruin/DD probability StatTiles; config: sim result, horizon |
| `mc-histogram` | Outcome Histogram | analysis | 4×8 | Terminal-value / max-DD distribution; percentile markers |
| `corr-heatmap` | Correlation Matrix | analysis | 6×8 | Diverging ramp, per-cell tooltip, table toggle |
| `positions` | Positions | portfolio | 8×8 | Paper account positions: qty, avg, last, unrealized P&L (DeltaChip) |
| `orders-journal` | Orders & Journal | portfolio | 8×8 | Paper order history + notes column |
| `leaderboard` | Leaderboard | portfolio | 4×8 | Friends' paper accounts by return; sparkline each (docs/08) |
| `payoff` | Options Payoff | analysis | 6×8 | Payoff-at-expiry + T+n curves for a position builder (docs/09) |
| `news-notes` | Notes | markets | 4×6 | Shared markdown scratchpad per workspace (local) |

Every chart widget: table toggle + PNG export + "copy data as CSV" in the `⋯` menu.

## 5. Global context & linking

TradingView-style **link channels** so widgets follow each other:

- Channels: `A` (accent-colored), `B`, `C`, and `∅` (unlinked). A widget's
  header shows its channel chip; clicking cycles it.
- The **context store** holds per-channel `{ symbol, timeframe, dateRange }`.
  Topbar controls write to channel `A` by default.
- A widget's manifest declares which context fields it consumes. Resolution
  order: explicit config override → its channel's context → app defaults
  (`SPY`, `1D`, last 1y).
- Clicking a row in `watchlist`/`positions`/`leaderboard` publishes the symbol
  to that widget's channel — this is the core "click a ticker, every linked
  chart follows" interaction.
- Run-scoped widgets (`equity-curve`, `mc-fan`, …) additionally subscribe to a
  `selectedRuns: RunId[]` context, set from the Backtest/Sim Labs ("open in
  dashboard").

## 6. Workspaces

```ts
interface Workspace {
  version: 1;
  id: string; name: string; icon?: string;
  layouts: Record<'lg' | 'md', Layout[]>;   // react-grid-layout format
  widgets: Record<string, { widgetId: string; channel: 'A'|'B'|'C'|null; config: unknown }>;
  context: ContextSnapshot;                  // channels' symbol/tf/range
}
```

- Persisted to `localStorage` (`tradeboard.workspaces.v1`), autosaved.
- **Ship 3 presets** (seed data, also the demo): *Markets* (quote-strip,
  candles, watchlist, stat tiles), *Strategy Lab* (equity-curve, drawdown,
  run-metrics, candles-with-signals), *Risk* (mc-fan, mc-histogram,
  corr-heatmap, drawdown).
- Export/import as a `.json` file, and "copy share string" (base64 of the JSON)
  so we can paste workspaces to each other in chat. On import, unknown widget
  ids are skipped with a toast; configs are re-parsed through their zod schemas
  (invalid → widget's defaults).
- "Reset to preset" always available; deleting the last workspace recreates
  *Markets*.

## 7. Add-widget flow

Topbar `+ Widget` (and ⌘K) opens a gallery modal: category tabs, live preview
card per widget (rendered with canned fixture data — every widget ships a
`fixture.ts`), one-click add → lands in the first free grid slot → its config
popover opens if the widget requires a choice (e.g. stat-tile metric).

## 8. Performance budgets

- Dashboard cold load (cached data): **< 1.5s** to interactive on a mid laptop.
- Widget add/drag/resize: no frame > 32ms (charts pause re-render during drag;
  re-render once on settle).
- A workspace with 12 widgets at 1s quote polling stays under 30% of one core.
  Quote polling is one batched request for all visible symbols (docs/04), fanned
  out via the query cache — never per-widget requests.
