# @tradeboard/api

Small Node (Hono) data proxy: provider adapters, SQLite bar cache, rate limiting, leaderboard.

- Spec: [docs/04-data-layer.md](../../docs/04-data-layer.md), leaderboard routes in [docs/08-paper-trading.md](../../docs/08-paper-trading.md) §5
- Holds ALL provider API keys (`.env`, see `.env.example` once scaffolded). The browser never sees a key.
- Must run useful (Stooq/Yahoo) with zero keys configured.
