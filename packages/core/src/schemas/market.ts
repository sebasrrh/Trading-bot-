import { z } from 'zod';

export const TimeframeSchema = z.enum(['1m', '5m', '15m', '1h', '1D', '1W']);
export type Timeframe = z.infer<typeof TimeframeSchema>;

export const BarSchema = z.object({
  t: z.number(),
  o: z.number(),
  h: z.number(),
  l: z.number(),
  c: z.number(),
  v: z.number(),
});
export type Bar = z.infer<typeof BarSchema>;

export const QuoteSchema = z.object({
  symbol: z.string(),
  price: z.number(),
  ts: z.number(),
  change: z.number(),
  changePct: z.number(),
  prevClose: z.number(),
});
export type Quote = z.infer<typeof QuoteSchema>;

export const SymbolInfoSchema = z.object({
  symbol: z.string(),
  name: z.string(),
  exchange: z.string(),
  type: z.enum(['stock', 'etf']),
});
export type SymbolInfo = z.infer<typeof SymbolInfoSchema>;

export const BarsRequestSchema = z.object({
  symbol: z.string(),
  timeframe: TimeframeSchema,
  from: z.number(),
  to: z.number(),
});
export type BarsRequest = z.infer<typeof BarsRequestSchema>;