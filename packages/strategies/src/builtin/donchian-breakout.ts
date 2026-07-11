import { z } from 'zod';
import type { StrategyDef } from '../types';

type Params = { entry: number; exit: number };

export const donchianBreakout: StrategyDef<Params> = {
  id: 'donchian-breakout',
  name: 'Donchian Breakout',
  description: 'Long when price breaks above N-day high, trail stop at M-day low.',
  paramsSchema: z.object({
    entry: z.number().int().min(10).max(120).default(55),
    exit: z.number().int().min(5).max(60).default(20),
  }),
  warmupBars: (p) => p.entry,
  create: (p) => {
    let stopPrice = 0;
    return {
      onBar(ctx) {
        const dc = ctx.indicator.donchian(p.entry);
        const upper = dc.upper[ctx.i]!;
        const lower = dc.lower[ctx.i]!;
        if (isNaN(upper) || isNaN(lower)) return null;
        const inPos = ctx.position.qty > 0;
        if (!inPos) {
          if (ctx.close() > upper) {
            const trailDc = ctx.indicator.donchian(p.exit);
            stopPrice = trailDc.lower[ctx.i]!;
            return { target: 1, stopLoss: stopPrice, reason: `breakout > ${upper.toFixed(2)}` };
          }
        } else {
          const trailDc = ctx.indicator.donchian(p.exit);
          stopPrice = trailDc.lower[ctx.i]!;
          if (ctx.close() <= stopPrice) return { target: 0, reason: `trail stop ${stopPrice.toFixed(2)}` };
        }
        return null;
      },
    };
  },
};