import { z } from 'zod';
import type { StrategyDef } from '../types';

type Params = { fast: number; slow: number };

export const smaCross: StrategyDef<Params> = {
  id: 'sma-cross',
  name: 'SMA Crossover',
  description: 'Long when the fast SMA is above the slow SMA, flat otherwise. Trend-following; whipsaws sideways markets.',
  paramsSchema: z.object({
    fast: z.number().int().min(2).max(200).default(20),
    slow: z.number().int().min(5).max(400).default(50),
  }),
  warmupBars: (p: Params) => p.slow,
  create: (p: Params) => ({
    onBar(ctx) {
      const fast = ctx.indicator.sma(p.fast);
      const slow = ctx.indicator.sma(p.slow);
      const target = fast > slow ? 1 : 0;
      if (target !== (ctx.position.qty > 0 ? 1 : 0))
        return { target, reason: target ? 'fast>slow' : 'fast<slow' };
      return null;
    },
  }),
};