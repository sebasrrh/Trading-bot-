# 11 — Deployment (Docker + GitHub Actions)

Carries forward the deployment pattern from the original repo (push to `main` ⇒
Docker image on Docker Hub, using the `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`
repo secrets that are already configured) — adapted to the TS monorepo.

## 1. One image, not two

The old setup shipped separate backend and frontend images. The new stack
doesn't need that: the web app is a **static Vite build**, so the api container
serves it.

```
tradeboard image
├── node:22-slim
├── apps/api (compiled)         ← Hono server on :8787
│   └── serves /api/* routes
│   └── serves apps/web/dist as static files (SPA fallback to index.html)
└── volume: /data               ← SQLite cache + leaderboard (apps/api/data)
```

One `Dockerfile` at the repo root, multi-stage:

1. **build stage** — `pnpm install --frozen-lockfile`, `pnpm build`
   (packages → api → web; Turborepo/workspace topological order).
2. **runtime stage** — copy `apps/api/dist`, `apps/web/dist`, production
   `node_modules` (pnpm deploy --prod). Non-root user. `EXPOSE 8787`.
   `HEALTHCHECK` hits `/api/health` (docs/04 §3).

Config via env: `PORT`, `ALPACA_KEY_ID/SECRET`, `POLYGON_KEY`,
`LEADERBOARD_KEY` — all optional; the container must run useful with none
(keyless providers, docs/04 §1).

## 2. Workflow: `.github/workflows/publish.yml`

Same trigger and secrets as the original, plus CI gating and better tags:

```yaml
name: Build & Push to Docker Hub
on:
  push:
    branches: [main]
jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with: { node-version: 22, cache: pnpm }
      - run: pnpm install --frozen-lockfile
      - run: pnpm lint && pnpm typecheck && pnpm test   # never publish a red build
      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            madpods/tradeboard:latest
            madpods/tradeboard:${{ github.sha }}
```

Differences from the old workflow, on purpose:

- **Tests gate publishing.** The old one pushed whatever was on main.
- **SHA tag alongside `latest`** so a bad deploy can be rolled back by pinning
  the previous SHA instead of "hope `latest` was cached somewhere."
- One image instead of two (see §1).

The separate CI workflow (lint/test on PRs, docs/01 §Build & tooling) stays
independent — publishing runs only on main.

## 3. Running it (the friend-server)

Whoever hosts it (a home server, an old laptop, a $5 VPS) runs:

```yaml
# docker-compose.yml (deploy copy, not in-repo config)
services:
  tradeboard:
    image: madpods/tradeboard:latest
    ports: ["8787:8787"]
    env_file: .env            # provider keys, LEADERBOARD_KEY
    volumes: ["tradeboard-data:/data"]
    restart: unless-stopped
volumes:
  tradeboard-data:
```

- The hosted instance is what makes the **shared leaderboard** (docs/08 §5)
  real for the group; everything else also works fully local via `pnpm dev`.
- Updating = `docker compose pull && docker compose up -d`. No migrations in
  v1 (SQLite schema created on boot; cache is disposable).
- WebGPU note: sims run in each visitor's **browser**, so the server needs no
  GPU — hosting stays $5-tier no matter how many paths we simulate.

## 4. When to build this

Phase 1 exit (docs/10) is the right moment — that's when there's a dashboard
worth hosting. Until then the workflow would just publish an empty shell.
Docker Hub repo to create once: `madpods/tradeboard`.
