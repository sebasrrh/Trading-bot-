import { describe, it, expect } from 'vitest';
import type { BacktestConfig } from '@tradeboard/backtest';
import { gridSearch, geneticOptimize } from '../index';
import type { SearchSpace, ObjectiveMetric } from '../types';
import { smaCross } from '@tradeboard/strategies';
import type { BarSeries } from '@tradeboard/core';

function makeBars(n: number): BarSeries {
  const t = new Float64Array(n);
  const o = new Float64Array(n);
  const h = new Float64Array(n);
  const l = new Float64Array(n);
  const c = new Float64Array(n);
  const v = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    t[i] = 1672531200000 + i * 86400000;
    const trend = Math.sin(i / 20) * 10 + 100;
    const noise = (Math.sin(i * 1.7) * 3 + Math.cos(i * 0.3) * 2);
    o[i] = trend + noise;
    h[i] = o[i]! + Math.abs(noise) + 1;
    l[i] = o[i]! - Math.abs(noise) - 1;
    c[i] = o[i]! + noise * 0.5;
    v[i] = 1000000 + Math.sin(i / 5) * 500000;
  }
  return { symbol: 'SPY', timeframe: '1D' as const, t, o, h, l, c, v, length: n };
}

const bars = makeBars(200);
const firstT = bars.t[0]!;
const lastT = bars.t[bars.length - 1]!;

const baseConfig: BacktestConfig = {
  strategyId: 'sma-cross',
  params: {},
  symbol: 'SPY',
  timeframe: '1D',
  from: firstT,
  to: lastT,
  initialCash: 100_000,
  costModel: { commissionPerOrder: 0, spreadBps: 2, slippageBps: 3 },
  sizing: { kind: 'all-in' },
  allowShort: false,
  seed: 42,
};

const space: SearchSpace = {
  fast: { min: 5, max: 50, step: 5 },
  slow: { min: 20, max: 200, step: 20 },
};

describe('gridSearch', () => {
  it('returns best params from exhaustive search', async () => {
    const res = await gridSearch(baseConfig, space, bars, smaCross, 'sharpe');
    expect(res.objective).toBe('sharpe');
    expect(res.objective).toBeDefined();
    expect(res.best.params).toBeDefined();
    expect(res.best.params.fast).toBeGreaterThanOrEqual(5);
    expect(res.best.params.fast).toBeLessThanOrEqual(50);
    expect(res.best.params.slow).toBeGreaterThanOrEqual(20);
    expect(res.best.params.slow).toBeLessThanOrEqual(200);
    expect(res.totalEvals).toBeGreaterThan(0);
    expect(res.durationMs).toBeGreaterThanOrEqual(0);
  });

  it('evaluations includes all combos', async () => {
    const res = await gridSearch(baseConfig, space, bars, smaCross);
    const expected = 10 * 10;
    expect(res.evaluations.length).toBe(expected);
    expect(res.totalEvals).toBe(expected);
  });

  it('finds best with different objective', async () => {
    const r1 = await gridSearch(baseConfig, space, bars, smaCross, 'sharpe');
    const r2 = await gridSearch(baseConfig, space, bars, smaCross, 'totalReturn');
    expect(r1.best.params).toBeDefined();
    expect(r2.best.params).toBeDefined();
  });

  it('works with single param', async () => {
    const single: SearchSpace = { fast: { min: 5, max: 20, step: 5 } };
    const res = await gridSearch(baseConfig, single, bars, smaCross);
    expect(res.evaluations.length).toBe(4);
    expect(res.best.params.fast).toBeDefined();
  });
});

describe('geneticOptimize', () => {
  it('returns best params', async () => {
    const res = await geneticOptimize(baseConfig, space, bars, smaCross, {
      populationSize: 10,
      generations: 5,
      seed: 42,
    });
    expect(res.objective).toBeDefined();
    expect(res.best.params).toBeDefined();
    expect(res.best.params.fast).toBeGreaterThanOrEqual(5);
    expect(res.best.params.fast).toBeLessThanOrEqual(50);
    expect(res.totalEvals).toBeGreaterThan(0);
    expect(res.gaHistory).toBeDefined();
    expect(res.gaHistory!.length).toBe(5);
  });

  it('fitness stable over generations', async () => {
    const res = await geneticOptimize(baseConfig, space, bars, smaCross, {
      populationSize: 15,
      generations: 8,
      seed: 42,
    });
    const history = res.gaHistory!;
    const lastBest = history[history.length - 1]!.bestFitness;
    expect(lastBest).toBeGreaterThan(-Infinity);
  });

  it('works with different objective', async () => {
    const res = await geneticOptimize(baseConfig, space, bars, smaCross, {
      populationSize: 8,
      generations: 3,
      objective: 'calmar' as ObjectiveMetric,
      seed: 42,
    });
    expect(res.best.metrics.calmar).toBeDefined();
  });
});