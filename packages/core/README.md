# @tradeboard/core

Shared domain types, zod schemas, and pure utilities. **Depends on nothing in the workspace; imports no DOM/Node APIs.**

- `Bar`, `Quote`, `Timeframe`, `BarSeries` (columnar), `Signal`, `Order`, `Position`, run/result schemas
- Market calendar (NYSE sessions), time utils, options math (`src/options/`, [docs/09](../../docs/09-options.md))
- Schema definitions per [docs/01-architecture.md](../../docs/01-architecture.md) § Typed contracts
