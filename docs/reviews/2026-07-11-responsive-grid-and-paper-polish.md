# Responsive grid + paper trading polish (2026-07-11, third pass)

Two user-reported issues, fixed and verified live (headless-browser scripted
interaction, not just read from source). Base commit `3ca6edc` (the previous
punch-list-fixes patch, already applied to `main`).

## 1. Grid didn't fill the screen

**Root cause:** `apps/web/src/app/dashboard-grid.tsx` used the plain
`react-grid-layout` `<GridLayout>` with a hardcoded `width={1200}`. On any
viewport wider than 1200px the grid stopped there, leaving a visible empty
gap on the right — this was punch-list item #12.

**Fix:** switched to `WidthProvider(Responsive)`, which measures its actual
container width via `ResizeObserver` instead of a fixed number, and added the
three breakpoints from docs/03 §2 (`lg` ≥1200px/12 cols, `md` ≥768px/8 cols,
`sm` <768px/1 col). `sm` additionally sets `isDraggable={false}
isResizable={false}` — a phone-width screen gets a readable stacked view, not
a drag-resize surface, matching "read-only layout on mobile" from the spec.

Per-breakpoint layouts are derived from the stored `lg` layout when a
breakpoint hasn't been customized yet (`md`/`sm` clamp `w`/`x` into that
breakpoint's column count; `sm` additionally re-stacks into a single column in
top-to-bottom reading order), and are persisted per-breakpoint once the user
actually drags/resizes at that width — `onLayoutChange(current, all)` now
saves the full `{lg, md, sm}` object instead of only ever writing `lg`.

**Also fixed while touching this file (punch-list #12's other half):**
`WidgetProps.size` was declared in the type but permanently hardcoded to
`{ wPx: 0, hPx: 0 }` — no widget could ever know its real rendered size. Each
widget frame now has a `ResizeObserver` reporting live pixel dimensions
(rounded to whole px and only updating state on an actual change, so a
sub-pixel jiggle during drag doesn't trigger a re-render storm across every
widget on the board). `stat-tile` was updated to actually consume it — the
label and delta chip drop out and the value shrinks when the tile is resized
very small/narrow, instead of clipping — as a concrete, verifiable proof the
plumbing works, not just dead code sitting unused again.

**Verified:** scripted viewport resize across 1920/1600/900/500px; at every
width the grid's rendered width tracks the container (`main width − 24px
padding`), confirmed by comparing bounding boxes — no more fixed 1200px
ceiling. Screenshot at 1920px shows widgets spanning the full window.

## 2. Paper trading — no reset, "clunky"

**"I can't even start over" — there was genuinely no reset control anywhere.**
Added one:

- `apps/web/src/state/paper-store.ts`: new `resetAccount()` action —
  recreates the engine with a fresh $100k account and clears the persisted
  `localStorage` entry (the persistence added in the previous pass meant a
  reset had to explicitly clear storage too, or the wiped account would just
  reload itself back from the old save on next visit).
- `apps/web/src/views/paper-trading.tsx`: a "Reset" button next to the Paper
  Trading header, two-step confirm (click once → "Confirm reset?" for 4s,
  click again → actually resets) rather than a jarring native `confirm()`
  popup or, worse, no confirmation at all for a destructive action.

**Verified live, including the part that actually matters — that it's not
just clearing in-memory state:** placed an order, confirmed it shows in Open
Orders; clicked Reset once and confirmed the order was *still there*
(the first click only asks, doesn't act); clicked again and confirmed Open
Orders went 1 → 0 with cash back to $100,000.00. Then, separately, reloaded
the page after placing an order (order survived — persistence works) and
reloaded again after a confirmed reset (stayed at 0 orders — the reset
genuinely wiped `localStorage`, it didn't just reset a variable that would've
silently reloaded the old account on the next visit).

**The rest of "clunky," fixed alongside it:**

- **No order validation, no feedback.** You could submit qty 0, a negative
  qty, or a limit/stop order with no price, and it would silently queue
  something broken with zero indication anything happened. Now: inline
  validation disables the submit button with a reason shown (`Quantity must
  be greater than 0`, `Limit orders need a price`, `No SPY position to
  sell.`), and a successful submit shows a green confirmation
  (`Buy order queued — fills on the next bar feed.`) and clears the price
  field so a second click can't accidentally resubmit the same limit price.
- **No quick-size buttons**, despite docs/08 §6 specifying "qty with 25/50/100%
  chips." Added them — buy-side sizes off available cash at the last quote,
  sell-side sizes off the quantity actually held in that symbol.
- **No "flatten all."** Added a "Flatten All Positions" button (same
  two-step confirm pattern as Reset) that queues an opposite-side market
  order for every open position — also spec'd in docs/08 §6, also just
  missing.
- **Confusing fill timing.** Placing an order doesn't fill it — it only fills
  the next time bars are fed (matches the backtest engine's next-bar-open
  semantics, by design, docs/08 §4) — but nothing on screen said so, so a
  placed order that visibly did nothing looked broken. Added one line of
  copy under the feed button explaining it, plus the queued-order
  confirmation message says the same thing.
- **Wrong color semantics — the same bug I'd already fixed elsewhere,
  present here too.** Return/unrealized-P&L/buy-side/fill-side/fill-P&L all
  used `var(--accent)` (brand indigo) for positive/buy, instead of
  `var(--gain)` (green). Fixed all five occurrences — this file hadn't been
  touched in the earlier color-token pass.

## Verified

- Fresh state (`rm -rf packages/*/dist apps/*/dist`): typecheck (12
  packages), 74 tests, lint, and build all pass.
- Headless-Chromium, scripted (not just screenshotted): grid width tracks
  viewport at 4 sizes; paper-trading reset requires two clicks and only the
  second one clears state; reset survives a page reload (localStorage
  actually cleared, not just the in-memory engine).
- **Not verified in this sandbox:** no network to real quote providers, so
  the 25/50/100% buy-side sizing (which needs a live quote price) couldn't be
  exercised against real numbers — the sell-side sizing (off held qty, no
  quote needed) was implicitly exercised via the flatten-all logic sharing
  the same code path. Worth a quick manual check on a machine with real data.
