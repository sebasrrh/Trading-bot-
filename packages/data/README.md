# @tradeboard/data

Market-data access shared by web and api: the `MarketDataProvider` interface, normalization, and the browser `DataClient` + TanStack Query hooks.

- Spec: [docs/04-data-layer.md](../../docs/04-data-layer.md)
- Normalization (UTC, adjusted-only, ascending, dedup) lives here in ONE place with golden tests per provider fixture.
