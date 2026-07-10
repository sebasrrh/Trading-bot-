# @tradeboard/strategies

The `StrategyDef` plugin interface, the registry, and the built-in strategies.

- Spec: [docs/05-strategy-framework.md](../../docs/05-strategy-framework.md) — read §1 rules before writing one
- Add a strategy: one file in `src/builtin/` + one registry line + green conformance suite. Nothing else.
- Environment-agnostic: no DOM, no fetch, no Date.now, no Math.random.
