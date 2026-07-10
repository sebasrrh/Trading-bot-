# @tradeboard/web

The React dashboard SPA: app shell, widget grid, charts, labs (Backtest, Sim, Paper).

- Spec: [docs/03-dashboard-shell.md](../../docs/03-dashboard-shell.md) (shell & widgets), [docs/02-design-system.md](../../docs/02-design-system.md) (look)
- Stack: React 19 + Vite, Zustand, TanStack Query, react-grid-layout, lightweight-charts, uPlot
- Source layout is specified in [docs/01-architecture.md](../../docs/01-architecture.md) § Monorepo layout
- Heavy work never runs on the main thread — see `src/workers/`
