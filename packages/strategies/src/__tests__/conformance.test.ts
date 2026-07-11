import { describe, it, expect } from 'vitest';
import { allStrategies } from '../registry';
import type { StrategyDef, StrategyContext, IndicatorAccessor } from '../types';
import type { BarSeries } from '@tradeboard/core';

function makeBars(n: number): BarSeries {
  const t = new Float64Array(n);
  const o = new Float64Array(n);
  const h = new Float64Array(n);
  const l = new Float64Array(n);
  const c = new Float64Array(n);
  const v = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    t[i] = 86_400_000 * i;
    o[i] = 100 + Math.sin(i * 0.1) * 10;
    h[i] = o[i]! + 2;
    l[i] = o[i]! - 2;
    c[i] = o[i]! + Math.sin(i * 0.15) * 5;
    v[i] = 1_000_000 + Math.random() * 500_000;
  }
  return { symbol: 'TEST', timeframe: '1D', t, o, h, l, c, v, length: n };
}

function makeIndicatorAccessor(bars: BarSeries): IndicatorAccessor {
  const cache = new Map<string, any>();
  const nanArr = () => { const a = new Float64Array(bars.length); a.fill(NaN); return a; };
  return {
    sma: (p) => { const k = `sma:${p}`; let v = cache.get(k); if (v) return v;
      v = new Float64Array(bars.length); let s = 0;
      for (let i = 0; i < bars.length; i++) { s += bars.c[i]!; if (i >= p) s -= bars.c[i - p]!; v[i] = i >= p - 1 ? s / p : NaN; }
      cache.set(k, v); return v; },
    ema: (p) => { const k = `ema:${p}`; let v = cache.get(k); if (v) return v;
      v = new Float64Array(bars.length); const kf = 2 / (p + 1); let prev = NaN;
      for (let i = 0; i < bars.length; i++) { if (i < p - 1) { v[i] = NaN; continue; } if (i === p - 1) { let su = 0; for (let j = 0; j < p; j++) su += bars.c[j]!; prev = su / p; v[i] = prev; } else { prev = (bars.c[i]! - prev) * kf + prev; v[i] = prev; } }
      cache.set(k, v); return v; },
    rsi: () => nanArr(),
    atr: () => nanArr(),
    wma: () => nanArr(),
    macd: () => ({ macd: nanArr(), signal: nanArr(), histogram: nanArr() }),
    bbands: () => ({ upper: nanArr(), middle: nanArr(), lower: nanArr() }),
    donchian: () => ({ upper: nanArr(), middle: nanArr(), lower: nanArr() }),
    stoch: () => ({ k: nanArr(), d: nanArr() }),
  };
}

function makeCtx(bars: BarSeries, i: number, qty = 0): StrategyContext {
  const accessor = makeIndicatorAccessor(bars);
  return {
    i, bars,
    close: (offset = 0) => {
      const idx = i - offset;
      return idx >= 0 && idx < bars.length ? bars.c[idx]! : NaN;
    },
    indicator: accessor,
    position: { qty, avgPrice: 100, unrealizedPnl: 0 },
    equity: 100_000,
    log: () => {},
  };
}

describe('strategy conformance', () => {
  const strategies = allStrategies();
  const bars = makeBars(300);

  it.each(strategies)(`$id has default params`, (def: StrategyDef) => {
    const schema = def.paramsSchema;
    const parsed = schema.safeParse({});
    expect(parsed.success).toBe(true);
  });

  it.each(strategies)(`$id warmup respected`, (def: StrategyDef) => {
    const params = def.paramsSchema.parse({});
    const warmup = def.warmupBars(params);
    const inst = def.create(params);
    for (let i = 0; i < Math.min(warmup, bars.length); i++) {
      const signal = inst.onBar(makeCtx(bars, i));
      expect(signal).toBeNull();
    }
  });

  it.each(strategies)(`$id is deterministic`, (def: StrategyDef) => {
    const params = def.paramsSchema.parse({});
    const warmup = def.warmupBars(params);
    const inst1 = def.create(params);
    const inst2 = def.create(params);
    const signals1: any[] = [];
    const signals2: any[] = [];
    for (let i = warmup; i < bars.length; i++) {
      signals1.push(inst1.onBar(makeCtx(bars, i)));
      signals2.push(inst2.onBar(makeCtx(bars, i)));
    }
    expect(signals1).toEqual(signals2);
  });

  it.each(strategies)(`$id produces no NaN target`, (def: StrategyDef) => {
    const params = def.paramsSchema.parse({});
    const warmup = def.warmupBars(params);
    const inst = def.create(params);
    for (let i = warmup; i < bars.length; i++) {
      const signal = inst.onBar(makeCtx(bars, i));
      if (signal) {
        expect(Number.isNaN(signal.target)).toBe(false);
        expect(Number.isFinite(signal.target)).toBe(true);
      }
    }
  });
});