import type { BacktestConfig } from '@tradeboard/backtest';
import { BacktestEngine } from '@tradeboard/backtest';

import type { BarSeries } from '@tradeboard/core';
import type { SearchSpace, OptimizerResult, ParamEval, ObjectiveMetric } from './types';

function extractMetric(m: { sharpe: number; sortino: number; calmar: number; cagr: number; profitFactor: number; totalReturn: number }, key: ObjectiveMetric): number {
  return m[key];
}

function cartesianProduct(arrays: number[][]): number[][] {
  if (arrays.length === 0) return [[]];
  const [first, ...rest] = arrays;
  const restProduct = cartesianProduct(rest);
  const result: number[][] = [];
  for (const f of first!) {
    for (const r of restProduct) {
      result.push([f, ...r]);
    }
  }
  return result;
}

export async function gridSearch(
  config: BacktestConfig,
  space: SearchSpace,
  bars: BarSeries,
  def_: any,
  objective?: ObjectiveMetric,
): Promise<OptimizerResult> {
  const start = Date.now();
  const keys = Object.keys(space);
  const ranges = keys.map(k => {
    const r = space[k]!;
    const values: number[] = [];
    const step = r.step ?? Math.max(1, Math.floor((r.max - r.min) / 10));
    for (let v = r.min; v <= r.max; v += step) values.push(v);
    if (!values.includes(r.min)) values.unshift(r.min);
    if (!values.includes(r.max)) values.push(r.max);
    return values;
  });

  const obj = objective ?? 'sharpe';
  const combos = cartesianProduct(ranges);
  const evaluations: ParamEval[] = [];

  for (const combo of combos) {
    const params: Record<string, number> = {};
    for (let i = 0; i < keys.length; i++) params[keys[i]!] = combo[i]!;
    const cfg: BacktestConfig = { ...config, params };
    const engine = new BacktestEngine(cfg, def_);
    const result = engine.run(bars);
    evaluations.push({ params, metrics: result.metrics });
  }

  let best = evaluations[0]!;
  for (const ev of evaluations) {
    if (extractMetric(ev.metrics, obj) > extractMetric(best.metrics, obj)) best = ev;
  }

  return { best, evaluations, totalEvals: evaluations.length, durationMs: Date.now() - start, objective: obj };
}