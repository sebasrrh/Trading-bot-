import { describe, it, expect } from 'vitest';
import { BarSchema, QuoteSchema, TimeframeSchema, SignalSchema, OrderSchema, PositionSchema } from '../index';

describe('core schemas', () => {
  it('parses a valid Bar', () => {
    const bar = { t: 1700000000000, o: 100, h: 101, l: 99, c: 100.5, v: 10000 };
    expect(BarSchema.parse(bar)).toEqual(bar);
  });

  it('rejects invalid Bar', () => {
    expect(() => BarSchema.parse({ t: 'bad', o: 100 })).toThrow();
  });

  it('parses a valid Quote', () => {
    const q = { symbol: 'SPY', price: 500, ts: 1700000000000, change: 2.5, changePct: 0.5, prevClose: 497.5 };
    expect(QuoteSchema.parse(q)).toEqual(q);
  });

  it('accepts all Timeframe values', () => {
    const tfs = ['1m', '5m', '15m', '1h', '1D', '1W'] as const;
    for (const tf of tfs) expect(TimeframeSchema.parse(tf)).toBe(tf);
  });

  it('parses a valid Signal', () => {
    const s = { target: 1, reason: 'buy signal', stopLoss: 95 };
    expect(SignalSchema.parse(s)).toEqual(s);
  });

  it('parses a valid Order', () => {
    const o = { symbol: 'AAPL', side: 'buy', qty: 100, type: 'market' as const };
    expect(OrderSchema.parse(o)).toEqual(o);
  });

  it('parses a valid Position', () => {
    const p = { symbol: 'AAPL', qty: 100, avgPrice: 150, unrealizedPnl: 500 };
    expect(PositionSchema.parse(p)).toEqual(p);
  });
});