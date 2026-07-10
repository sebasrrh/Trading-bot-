# 02 — Design System

One committed look: **modern dark fintech**. Dark slate surfaces, generous
spacing, rounded cards, one indigo accent, and disciplined chart color. Think
Linear-meets-Robinhood-dark, not Bloomberg-terminal-dense. Dark-only in v1.

Every value below ships as a CSS custom property in `packages/ui/src/tokens.css`.
**Components never contain raw hex** — they reference tokens. That file is the
single source of truth; this doc is its specification.

## 1. Color

### 1.1 Surfaces & structure

| Token | Hex | Use |
|---|---|---|
| `--bg-page` | `#0b0d12` | App background, behind everything |
| `--bg-surface-1` | `#12151c` | Cards, widgets, panels — **the chart surface** |
| `--bg-surface-2` | `#1a1e27` | Elevated: dropdowns, modals, hover rows |
| `--bg-surface-3` | `#232834` | Pressed/active fills, selected rows |
| `--border-hairline` | `rgba(255,255,255,0.08)` | Card borders, dividers |
| `--border-strong` | `rgba(255,255,255,0.14)` | Input borders, focused card ring base |

Elevation is expressed by surface step + hairline border, **not** heavy
drop-shadows. Modals may add `box-shadow: 0 16px 48px rgba(0,0,0,0.5)`.

### 1.2 Text (ink)

| Token | Hex | Contrast on surface-1 | Use |
|---|---|---|---|
| `--text-primary` | `#eef1f8` | 16.2:1 | Values, headings |
| `--text-secondary` | `#9aa3b5` | 7.2:1 | Labels, descriptions |
| `--text-muted` | `#5c6577` | 3.1:1 | Axis ticks, timestamps, disabled |

**Text always wears text tokens, never a series/semantic color** — a colored dot
or delta chip sits *beside* neutral text to carry identity/direction. The single
exception: P&L deltas and prices may use `--gain`/`--loss` directly (it's a
trading app; green/red numbers are the vernacular).

### 1.3 Accent (brand)

| Token | Hex | Use |
|---|---|---|
| `--accent` | `#7b8cff` | Primary buttons, active tabs, focus rings, links, selection |
| `--accent-hover` | `#8f9dff` | Hover state |
| `--accent-pressed` | `#6a7be8` | Pressed |
| `--accent-soft` | `rgba(123,140,255,0.12)` | Selected-item washes, badges |

6.1:1 on surface-1. One accent, used sparingly — if everything glows indigo,
nothing does.

### 1.4 Semantic market colors (reserved)

| Token | Hex | Contrast (surface-1) | Use |
|---|---|---|---|
| `--gain` | `#3dd68c` | 9.7:1 | Up moves, positive P&L, buy side, up candles |
| `--gain-soft` | `rgba(61,214,140,0.12)` | — | Positive backgrounds/chips |
| `--loss` | `#f2555a` | 5.4:1 | Down moves, negative P&L, sell side, down candles |
| `--loss-soft` | `rgba(242,85,90,0.12)` | — | Negative backgrounds/chips |
| `--warn` | `#f5b83d` | 10.3:1 | Stale data, degraded provider, risk warnings |
| `--info` | `#7b8cff` | 6.1:1 | (= accent) informational banners |

**Reserved means reserved:** gain/loss/warn are never used as chart series
colors. A strategy that happens to be plotted green would silently read as
"profitable" — that's why the categorical palette below contains no pure green
or red. Status/direction is additionally always encoded by sign, arrow, or
icon — never color alone.

### 1.5 Categorical chart series (validated)

Fixed-order palette for "which series is which" (strategy overlays, comparison
curves, multi-symbol charts). **Assign in slot order, never cycle, never skip.**
Color follows the *entity*: if series 2 is removed from a chart, series 3 keeps
its color.

| Slot | Token | Hex | Hue |
|---|---|---|---|
| 1 | `--series-1` | `#3d8be8` | blue |
| 2 | `--series-2` | `#12a5bd` | cyan |
| 3 | `--series-3` | `#c48312` | amber |
| 4 | `--series-4` | `#8f7ff0` | violet |
| 5 | `--series-5` | `#d5589a` | magenta |
| 6 | `--series-6` | `#d96b31` | orange |

Validation (dataviz-skill `validate_palette.js`, mode dark, surface `#12151c`):
**all checks pass** — lightness band L 0.48–0.67 ✓, chroma floor ✓, worst
adjacent CVD ΔE **32.4** (target ≥ 12) ✓, contrast ≥ 3:1 vs surface ✓.

Rules that come with it:

- **Max 6 concurrent series.** A 7th folds into "Other" or the view becomes
  small multiples. No generated hues, ever.
- ≥ 2 series ⇒ a legend is always present; ≤ 4 series are also direct-labeled
  at line end. One series ⇒ no legend (the title names it).
- **If the palette changes, re-run the validator** before merging:
  `node validate_palette.js "<hexes>" --mode dark --surface "#12151c"`.

### 1.6 Sequential ramp (magnitude: heatmaps, exposure maps)

Single hue (blue), 7 steps, monotonic luminance (0.061 → 0.516), darkest step
recedes toward the surface = "near zero":

| Step | 100 | 200 | 300 | 400 | 500 | 600 | 700 |
|---|---|---|---|---|---|---|---|
| Hex | `#35475f` | `#496383` | `#5a7ba2` | `#688ebb` | `#749fd1` | `#80b0e7` | `#8ec2ff` |

If two sequential contexts appear at once, the second uses a cyan ramp built the
same way from `--series-2`. Never a rainbow.

### 1.7 Diverging ramp (polarity: correlation heatmaps, return maps)

Poles are the **gain/loss hues** (this is the one chart context where they're
correct — the data *is* signed P&L/return), midpoint is a neutral surface gray,
never a hue:

```
--loss-pole #f2555a ← #b04a55 ← #6e3a44 ← MID #2a2e38 → #2c5e4c → #2f9a6e → #3dd68c --gain-pole
```

Equal steps per arm. For non-P&L polarity (e.g. above/below benchmark) use
blue↔orange (`--series-1`/`--series-6`) with the same gray midpoint.

## 2. Typography

| Token | Value |
|---|---|
| `--font-ui` | `"Inter", system-ui, -apple-system, "Segoe UI", sans-serif` |
| `--font-mono` | `"JetBrains Mono", ui-monospace, "SF Mono", monospace` |

Inter self-hosted via `@fontsource-variable/inter` (no CDN). Mono is for code
(strategy params JSON, seeds) only — **prices and P&L are Inter with
`font-variant-numeric: tabular-nums`**, not mono.

| Style | Size/line | Weight | Use |
|---|---|---|---|
| `display` | 28/34 | 650 | Hero numbers on stat tiles |
| `title` | 18/24 | 600 | View titles |
| `heading` | 14/20 | 600 | Widget headers, table headers |
| `body` | 13/20 | 450 | Default UI text |
| `label` | 12/16 | 500 | Field labels, legend text |
| `micro` | 11/14 | 500, `letter-spacing: 0.04em`, uppercase | Axis ticks, tiny badges |

Numeric columns (tables, axis ticks, tickers) always set `tabular-nums`.
Percent/delta chips: `label` size, 600 weight.

## 3. Space, radius, motion

- **Spacing scale:** 4px base — `4 8 12 16 20 24 32 40 48 64`. Token names
  `--space-1` … `--space-10`.
- **Widget grid gutter:** 12px. Card internal padding: 16px (20px for stat tiles).
- **Radius:** `--radius-s: 6px` (inputs, chips), `--radius-m: 10px` (cards,
  widgets, modals), `--radius-full` (pills, avatar). Nothing sharp-cornered.
- **Motion:** 120ms ease-out for hovers, 200ms cubic-bezier(0.2, 0, 0, 1) for
  panel/layout transitions. Numbers **never** animate by tweening value (fake);
  price flashes use a 300ms `--gain-soft`/`--loss-soft` background pulse.
  Respect `prefers-reduced-motion`: disable pulses and layout animation.

## 4. Core components (`packages/ui`)

All primitives are headless-ish React + tokens. Radix UI primitives underneath
for a11y (dropdown, dialog, tooltip, tabs); styled with vanilla CSS modules —
no Tailwind, no CSS-in-JS runtime.

| Component | Spec |
|---|---|
| `Card` | surface-1, hairline border, radius-m. Optional header row: heading + actions right-aligned. |
| `Button` | variants: `primary` (accent fill, `#0b0d12` text), `secondary` (surface-2 + border), `ghost`, `danger` (loss fill). Heights 32/28. |
| `Tabs` | underline style, accent indicator, label typography. |
| `Select`, `Combobox` | surface-2 popover, check-mark selection, keyboard nav. Symbol search combobox shows ticker + name + exchange. |
| `Input`, `NumberInput` | surface-1 inset, border-strong, accent focus ring (2px). NumberInput has step arrows + unit suffix slot (%, $, bps). |
| `Badge` | soft-color chips: accent-soft / gain-soft / loss-soft / warn at 12%; `micro` type. |
| `DeltaChip` | signed value + arrow: `▲ 2.4%` in gain, `▼ 1.1%` in loss; arrow is the color-independent channel. |
| `StatTile` | label (secondary), display-size value (primary, tabular), optional DeltaChip + 40px sparkline. |
| `DataTable` | 36px rows, hairline row dividers, sticky header (surface-1 + blur), numeric columns right-aligned tabular. Row hover surface-2. Sortable headers. |
| `Toast` | bottom-right, surface-2, semantic left border 3px. |
| `EmptyState` | muted icon + one sentence + one action. Every widget must ship one. |
| `Skeleton` | shimmer on surface-2; charts get a skeleton axis frame, not a spinner. |

Focus ring everywhere: `outline: 2px solid var(--accent); outline-offset: 2px`.
All interactive targets ≥ 28×28px.

## 5. Chart chrome rules (bind for every chart, every widget)

These implement the dataviz-skill method for our system:

1. **Grid/axes are recessive.** Gridlines `rgba(255,255,255,0.06)` hairlines,
   horizontal only by default; axis line `rgba(255,255,255,0.14)`; tick labels
   `--text-muted` `micro`. The data is the loudest thing on the chart.
2. **One y-axis.** Never dual-axis. Comparing two scales ⇒ index both to 100 at
   range start (the compare view does this) or small multiples.
3. **Lines 2px**, no drop shadows, no gradients under lines except the single
   hero area chart (portfolio equity) which may use a 8%→0% fill of its own hue.
4. **Bars/histograms:** 4px rounded *data-end* only (baseline corners square),
   2px gap between adjacent bars, gain/loss coloring only when the measure is
   signed.
5. **Candles:** up = `--gain`, down = `--loss`, 1px wicks, no border,
   ≥ 1px body gap. Volume pane uses the same hues at 35% alpha.
6. **Tooltips by default:** crosshair + shared tooltip on time-series (shows all
   visible series, values tabular-num, series dot + neutral text); per-mark
   tooltip on bars/cells. Tooltip = surface-2, hairline border, radius-s.
7. **Monte Carlo fan:** percentile bands as layered fills of ONE hue
   (`--series-1` at 8/14/22% alpha for P5–P95/P25–P75/P40–P60), median line 2px
   solid, realized/backtest path in `--text-primary` 2px. Never rainbow spaghetti;
   individual sample paths only in "ghost" mode ≤ 60 paths at 4% alpha.
8. **Direct labels selectively** — line-end labels for ≤ 4 series; never a value
   on every point. Hero numbers live in StatTiles, not on the plot.
9. **Every chart widget has a table toggle** (⌸ icon in widget header) rendering
   the same data as a DataTable — the accessibility escape hatch and the
   copy-paste path.
10. **Density:** default comfortable; a workspace-level "compact" switch drops
    paddings one step and row height to 30px. That's the whole density story.

## 6. Iconography & voice

- Icons: [Lucide](https://lucide.dev) 16/20px, 1.75px stroke, `--text-secondary`
  default. No emoji in UI chrome.
- Numbers: always localized, `$1,234.56`, percents 1 decimal (`+2.4%`), bps for
  costs. Timestamps in the user's locale, UTC on hover.
- Empty/error copy is one sentence, plain, no exclamation marks. ("No data for
  this range. Try a wider one.")

## 7. Definition of done for any new UI

- [ ] No raw hex — tokens only.
- [ ] New chart passed the §5 checklist; if it introduced new series colors, the
      validator was re-run and results pasted in the PR.
- [ ] Keyboard reachable; focus visible; hover targets ≥ mark size.
- [ ] Loading skeleton + empty state + error state all exist.
- [ ] Looks right at widget sizes from min (docs/03) to full-screen.
