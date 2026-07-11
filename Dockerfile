FROM node:22-alpine
RUN corepack enable
WORKDIR /app

COPY pnpm-lock.yaml pnpm-workspace.yaml package.json tsconfig.base.json ./

COPY packages/core/package.json packages/core/tsconfig.json ./packages/core/
COPY packages/data/package.json packages/data/tsconfig.json ./packages/data/
COPY packages/ui/package.json packages/ui/tsconfig.json ./packages/ui/
COPY packages/indicators/package.json packages/indicators/tsconfig.json ./packages/indicators/
COPY packages/strategies/package.json packages/strategies/tsconfig.json ./packages/strategies/
COPY packages/backtest/package.json packages/backtest/tsconfig.json ./packages/backtest/
COPY packages/sim/package.json packages/sim/tsconfig.json ./packages/sim/
COPY packages/paper/package.json packages/paper/tsconfig.json ./packages/paper/
COPY packages/optimizer/package.json packages/optimizer/tsconfig.json ./packages/optimizer/
COPY apps/api/package.json apps/api/tsconfig.json ./apps/api/
COPY apps/web/package.json apps/web/tsconfig.json apps/web/vite.config.ts apps/web/index.html ./apps/web/

RUN pnpm install --frozen-lockfile

COPY . .

EXPOSE 8787 5173
CMD ["pnpm", "dev"]