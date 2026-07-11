import { describe, it, expect } from 'vitest';
import { BacktestEngine, type BacktestConfig } from '../index';
import { allStrategies } from '@tradeboard/strategies';
import type { BarSeries } from '@tradeboard/core';

function spyBars(): BarSeries {
  const n = 500;
  const t = new Float64Array(n);
  const o = new Float64Array(n);
  const h = new Float64Array(n);
  const l = new Float64Array(n);
  const c = new Float64Array(n);
  const v = new Float64Array(n);
  let price = 100;
  for (let i = 0; i < n; i++) {
    t[i] = 86_400_000 * i;
    const ret = (Math.random() - 0.48) * 0.02;
    price = price * (1 + ret);
    o[i] = price; h[i] = price * 1.01; l[i] = price * 0.99;
    c[i] = price * (1 + (Math.random() - 0.5) * 0.005);
    v[i] = 1_000_000;
  }
  return { symbol: 'SPY', timeframe: '1D', t, o, h, l, c, v, length: n };
}

describe('backtest engine', () => {
  it('deterministic: same input produces same output', () => {
    const bars = spyBars();
    const smaCross = allStrategies().find(s => s.id === 'sma-cross')!;
    const config: BacktestConfig = { strategyId: 'sma-cross', params: { fast: 20, slow: 50 }, symbol: 'SPY', timeframe: '1D', from: bars.t[0]!, to: bars.t[bars.length - 1]!, initialCash: 100_000, costModel: { commissionPerOrder: 0, spreadBps: 2, slippageBps: 3 }, sizing: { kind: 'all-in' }, allowShort: false, seed: 42 };
    const engine1 = new BacktestEngine(config, smaCross);
    const engine2 = new BacktestEngine(config, smaCross);
    const r1 = engine1.run(bars);
    const r2 = engine2.run(bars);
    expect(r1.metrics.sharpe).toBe(r2.metrics.sharpe);
    expect(r1.metrics.totalReturn).toBe(r2.metrics.totalReturn);
    expect(r1.trades.length).toBe(r2.trades.length);
  });

  it('buy-and-hold tracks equity correctly', () => {
    const bars = spyBars();
    const bah = allStrategies().find(s => s.id === 'buy-and-hold')!;
    const config: BacktestConfig = { strategyId: 'buy-and-hold', params: {}, symbol: 'SPY', timeframe: '1D', from: bars.t[0]!, to: bars.t[bars.length - 1]!, initialCash: 100_000, costModel: { commissionPerOrder: 0, spreadBps: 0, slippageBps: 0 }, sizing: { kind: 'all-in' }, allowShort: false, seed: 42 };
    const engine = new BacktestEngine(config, bah);
    const result = engine.run(bars);
    expect(result.equity.length).toBe(bars.length);
    expect(result.metrics.totalReturn).not.toBeNaN();
    expect(result.metrics.sharpe).not.toBeNaN();
  });

  it('costs reduce returns', () => {
    const bars = spyBars();
    const smaCross = allStrategies().find(s => s.id === 'sma-cross')!;
    const baseConfig: BacktestConfig = { strategyId: 'sma-cross', params: { fast: 20, slow: 50 }, symbol: 'SPY', timeframe: '1D', from: bars.t[0]!, to: bars.t[bars.length - 1]!, initialCash: 100_000, costModel: { commissionPerOrder: 0, spreadBps: 2, slippageBps: 3 }, sizing: { kind: 'all-in' }, allowShort: false, seed: 42 };
    const noCostConfig = { ...baseConfig, costModel: { commissionPerOrder: 0, spreadBps: 0, slippageBps: 0 } };
    const rCost = new BacktestEngine(baseConfig, smaCross).run(bars);
    const rNoCost = new BacktestEngine(noCostConfig, smaCross).run(bars);
    expect(rCost.metrics.totalReturn).toBeLessThanOrEqual(rNoCost.metrics.totalReturn);
  });
});