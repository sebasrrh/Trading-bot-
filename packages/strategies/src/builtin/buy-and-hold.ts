import { z } from 'zod';
import type { StrategyDef } from '../types';

export const buyAndHold: StrategyDef<Record<string, never>> = {
  id: 'buy-and-hold',
  name: 'Buy & Hold',
  description: 'Always fully long. The benchmark every strategy is compared against.',
  paramsSchema: z.object({}).default({}),
  warmupBars: () => 0,
  create: () => ({
    onBar: () => ({ target: 1, reason: 'always long' }),
  }),
};