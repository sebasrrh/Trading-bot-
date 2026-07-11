import { z } from 'zod';
import type { StrategyDef } from '../types';

type Params = { lookback: number; topN: number };

export const momentumRotation: StrategyDef<Params> = {
  id: 'momentum-rotation',
  name: 'Momentum Rotation',
  description: 'Hold the strongest asset by lookback return. Rebalance monthly. Single-symbol simplified.',
  paramsSchema: z.object({
    lookback: z.number().int().min(20).max(252).default(126),
    topN: z.number().int().min(1).max(5).default(1),
  }),
  warmupBars: (p) => p.lookback,
  create: (p) => {
    let rebalanceBar = 0;
    return {
      onBar(ctx) {
        if (ctx.i < rebalanceBar) return null;
        rebalanceBar = ctx.i + 21;
        if (ctx.i < p.lookback) return null;
        const ret = ctx.close(p.lookback) > 0 ? ctx.close() / ctx.close(p.lookback) - 1 : 0;
        return ret > 0 ? { target: 1, reason: `momentum ${(ret * 100).toFixed(1)}%` } : { target: 0, reason: 'no positive momentum' };
      },
    };
  },
};