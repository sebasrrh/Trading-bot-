import { z } from 'zod';
import type { StrategyDef } from '../types';

type Params = { period: number; low: number; exit: number };

export const rsiMeanRevert: StrategyDef<Params> = {
  id: 'rsi-mean-revert',
  name: 'RSI Mean Revert',
  description: 'Long when RSI crosses below oversold threshold, exit when it crosses back above neutral.',
  paramsSchema: z.object({
    period: z.number().int().min(5).max(30).default(14),
    low: z.number().min(10).max(40).default(30),
    exit: z.number().min(40).max(70).default(50),
  }),
  warmupBars: (p) => p.period,
  create: (p) => ({
    onBar(ctx) {
      const vals = ctx.indicator.rsi(p.period);
      const r = vals[ctx.i]!;
      if (isNaN(r)) return null;
      const inPos = ctx.position.qty > 0;
      if (!inPos && r < p.low) return { target: 1, reason: `rsi=${r.toFixed(1)}<${p.low}` };
      if (inPos && r >= p.exit) return { target: 0, reason: `rsi=${r.toFixed(1)}>=${p.exit}` };
      return null;
    },
  }),
};