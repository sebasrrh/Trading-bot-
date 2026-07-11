# Fixes applied — punch list follow-up (2026-07-11, second pass)

This is the fix pass for `docs/reviews/2026-07-11-punchlist.md`. Base commit
was `955abc1` ("fixed error maybe") — note the homelab commits already fixed
punch-list item **#0** (CORS + hardcoded API URL) independently before this
pass started; that fix is confirmed working here (proxy round-trip verified,
see §Verified) and is *not* re-done.

All fixes below were verified against a running instance: typecheck, 74 tests,
lint, and build all pass from a clean `dist`-less state, and the two riskiest
interaction fixes (#1 and #3) were driven end-to-end with a real headless
browser rather than just read from the source.

## Fixed

**#1 — Widgets couldn't be closed / channel-cycled.**
`apps/web/src/app/dashboard-grid.tsx`: added `draggableCancel="button"` to
`<GridLayout>`. The header's `×` and channel-chip buttons live inside the
`.widget-drag-handle` region; without `draggableCancel`, react-grid-layout
intercepted every mousedown there as a drag-start, so clicks on those buttons
never fired. Verified live: scripted a browser click on the `×` button and
confirmed the widget count actually dropped from 4 to 3.

**#2 — WebGPU Monte Carlo backend threw on every run.**
`packages/sim/src/gpu-backend.ts`: `getBindGroupLayout` was called on
`this.module` (a `GPUShaderModule`), which doesn't have that method — it
belongs to a compute pipeline. Fixed by creating the pipeline first with
`layout: 'auto'`, then building the bind group from
`pipeline.getBindGroupLayout(0)`. This means every GPU sim run to date threw a
TypeError before doing any work; whether that error was caught and silently
fell back to CPU, or surfaced to the user, wasn't traced further — either way
the GPU path itself now executes correctly. *(Not independently verified — this
sandbox has no WebGPU adapter; verify on a real machine before trusting Sim
Lab's GPU badge.)*

**#3 — Watchlist rows didn't publish to link channels.**
`apps/web/src/widgets/watchlist/index.tsx`: row click now calls
`setChannel(channel, { symbol })`, completing the core "click a ticker, linked
charts follow" interaction (docs/03 §5). Verified live: clicking the QQQ row
changed the linked Price Chart's title from "SPY · 1D" to "QQQ · 1D".
(Quote-strip chips were already flagged in the original punch list as
non-clickable too — left as-is; still open, see below.)

**#4 — Paper account reset on every page refresh.**
`apps/web/src/state/paper-store.ts`: the account now round-trips through
`localStorage` (`tradeboard.paper-account.v1`). `PaperAccount.positions` is a
`Map`, which `JSON.stringify`/`parse` can't handle directly, so it's
serialized as an entries array and rebuilt on load. Persistence piggybacks on
the existing `bump()` call that every mutation in `paper-trading.tsx` already
makes after touching the engine — no new call sites needed. This is
localStorage only, not the full IndexedDB + overnight-order-reconciliation
design in docs/08 §3 — good enough that trades survive a refresh, not a
complete implementation of that spec section.

**#5 — Server bar cache never hit.**
`apps/api/src/index.ts`: the coverage check compared `meta.last_t` (the open
timestamp of the most recently *closed* bar — e.g. yesterday 00:00 UTC for
daily) against `to`, which callers pass as `Date.now()`. That comparison could
never be true, so every request re-fetched from the provider regardless of
what was already cached. Fixed by comparing against the last bar that could
plausibly have closed by now (`coverageTarget = min(to, lastClosedBar)`)
instead of raw "now". This is a reasonable approximation, not full market-
calendar awareness (docs/04 §7 tracks that as a separate concern).

**#7 — Data-source badge was hardcoded "Stooq".**
New `apps/web/src/state/data-source-store.ts`: a small store that `useBars`
and `useQuotes` report into via `useEffect` on every response (TanStack Query
v5 dropped `onSuccess`, so this replaces the effect-based approach the old
code didn't have at all). The topbar badge (`apps/web/src/app/app.tsx`) now
shows the real `source` and turns warn-colored when there are provider
warnings or every provider failed — `cache` is treated as the happy path, not
degradation. Verified live: with outbound network blocked in this sandbox, the
badge correctly reads "NO DATA" instead of lying "Stooq" the way the old
hardcoded badge would have.

**#8 — Backtests silently used whatever range happened to load.**
`apps/web/src/views/backtest-lab.tsx`: `run()` now checks the gap between the
requested `to` and the last bar actually returned; if it's more than 5 days,
a dismissable-on-rerun banner reports the real covered range instead of
quietly backtesting a shorter window (docs/04 §6). Also surfaces provider
`warnings` from the response and a proper error message on a non-2xx
`/api/bars` response (previously any failure just fell through to the generic
"No data returned").

**#14 — Lint (CI was red).**
Fixed all 6 errors: empty catch block in `backtest-lab.tsx` (annotated, not
just silenced); two ternary-expression-as-statement violations in
`monte-carlo-view.tsx` (`t === 0 ? ctx.moveTo(...) : ctx.lineTo(...)` →
`if`/`else` — canvas API calls return nothing, so the ternary's value was
already being discarded, this is a style fix not a behavior change); and
`no-irregular-whitespace` from mojibake (see #16) which also silently broke a
`prefer-const` check on a variable the mojibake fix exposed as a real bug (see
below).

**#15 — Raw hex instead of design tokens.**
`apps/web/src/widgets/watchlist/index.tsx` and `quote-strip/index.tsx`: the
up/down color was `var(--accent)` (indigo) instead of `var(--gain)` (green) —
positive deltas were rendering in the brand color, not the "up" color.
`apps/web/src/widgets/candles/index.tsx`: the whole chart used a hardcoded
TradingView-default palette (`#26a69a`/`#ef5350`/`#0d0d12`...) instead of the
project's tokens. lightweight-charts renders to `<canvas>` and can't consume
CSS custom properties, so the fix hardcodes the *same* hex values as
`tokens.css` (`--bg-surface-1`, `--text-muted`, `--gain`, `--loss`) with a
comment noting they must stay in sync — not a real fix for token drift, but at
least the values now match the rest of the app instead of being an unrelated
palette. Volume bar alpha corrected from `44` (~27%) to `59` (~35%) hex to
match docs/02 §5.5's "35% alpha" spec for the volume pane while at it.

**#16 — Mojibake / BOM.**
Found actual double-encoded UTF-8 (not just a BOM issue) in
`apps/web/src/views/backtest-lab.tsx` and `workspace-store.ts` — em dashes,
ellipses, and a warning-sign-plus-non-breaking-space had been round-tripped
through the wrong encoding at some point (visible as `Fetching barsâ€¦` instead
of `Fetching bars…`). Fixed with targeted byte-sequence replacement (not a
blind file-wide re-encode, which broke on a legitimate `€` character
elsewhere in the file). Stripped the UTF-8 BOM from
`packages/backtest/src/index.ts` and `candles/index.tsx` (rewritten anyway
for #15).

**#17 — No workspace preset migration.**
`apps/web/src/state/workspace-store.ts`: added `PRESET_VERSION` (bumped to 2)
and a `migrate()` step on load that replaces any *stored* workspace whose
`name` matches a known built-in preset (Markets / Strategy Lab / Risk) and
whose `version` is older than current — so returning users with stale
localStorage actually see the new Strategy Lab widgets (equity-curve,
drawdown, run-metrics) added in the "all phases" commit, instead of being
stuck on the 2-widget version forever. User-renamed or user-created
workspaces are left untouched (name won't match).

## Also fixed while in the neighborhood

- `packages/sim/src/gpu-backend.ts` / candles / monte-carlo-view: while
  fixing #16's mojibake, found `apps/web/src/views/monte-carlo-view.tsx` had
  `let minV = -1, maxV = 0;` for the max-drawdown histogram, where `maxV` was
  never updated in the accumulation loop (only `minV` was tracked) — a real
  correctness bug in the histogram's axis scaling, not just a lint nit. Fixed
  to track both min and max the same way the terminal-value histogram two
  hundred lines above it already does correctly.

## Deliberately not fixed (feature-scope, not bugs — see roadmap)

Per the original punch list's P2 section, these are net-new work, not patch-
sized fixes: **#9** options UI (no chain widget/payoff builder), **#10**
leaderboard (no `/api/leaderboard` routes or widget), **#11** GPU/CPU parity
test + `mc-fan`/`mc-histogram` dashboard widgets, **#12** responsive grid
(width still hardcoded 1200, `WidgetProps.size` still `{0,0}`), **#13**
nav/view labeling mismatch, **#18** `packages/optimizer` undocumented. Also
not done: **#6** (sql.js whole-file rewrite per cache write — works, just
won't scale; swapping to better-sqlite3 is a dependency change, not a patch),
quote-strip chips still aren't clickable (only watchlist rows were in scope
of #3 as literally described), and `/api/health`'s provider booleans are
still hardcoded `true` regardless of real reachability.

## Verified

- Fresh state (`rm -rf packages/*/dist apps/*/dist`): `pnpm typecheck` (12
  packages, 0 errors), `pnpm test` (10 files / 74 tests, all pass),
  `pnpm lint` (0 errors, was 6), `pnpm build` (0 errors) — CI should be green.
- `pnpm dev`: confirmed `/api/health` reachable directly (`:8787`) **and**
  through the Vite dev proxy (`:5173/api/health`) — the upstream CORS/proxy
  fix (punch-list #0) is working end-to-end, not just present in the diff.
- Headless-Chromium, scripted interaction (not just a screenshot): closing a
  widget via the × button actually removes it (4 → 3 widgets); clicking a
  watchlist row actually re-links the chart (title changed SPY → QQQ); the
  topbar badge correctly shows "NO DATA" rather than a false "Stooq" when
  outbound network is unavailable.
- **Not verified in this sandbox** (no network to stooq/yahoo, no WebGPU
  adapter): live bar/quote data end-to-end, and the GPU Monte Carlo path
  actually producing a result. Both are one-time checks worth doing on a
  normal machine before calling this done.
