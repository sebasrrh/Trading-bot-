import { z } from 'zod';

export const SignalSchema = z.object({
  target: z.number(),
  reason: z.string().optional(),
  stopLoss: z.number().optional(),
  takeProfit: z.number().optional(),
});
export type Signal = z.infer<typeof SignalSchema>;

export const OrderSchema = z.object({
  symbol: z.string(),
  side: z.enum(['buy', 'sell']),
  qty: z.number().int(),
  type: z.enum(['market', 'limit', 'stop']),
  limitPrice: z.number().optional(),
  stopPrice: z.number().optional(),
});
export type Order = z.infer<typeof OrderSchema>;

export const PositionSchema = z.object({
  symbol: z.string(),
  qty: z.number().int(),
  avgPrice: z.number(),
  unrealizedPnl: z.number(),
});
export type Position = z.infer<typeof PositionSchema>;