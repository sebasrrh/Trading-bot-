import type { BacktestConfig } from '@tradeboard/backtest';
import { BacktestEngine } from '@tradeboard/backtest';

import type { BarSeries } from '@tradeboard/core';
import type { SearchSpace, OptimizerResult, ParamEval, GAOptions, ObjectiveMetric, GAPopulation } from './types';

function extractMetric(m: { sharpe: number; sortino: number; calmar: number; cagr: number; profitFactor: number; totalReturn: number }, key: ObjectiveMetric): number {
  return m[key];
}

function createRng(seed: number) {
  let s = seed | 0;
  return (): number => {
    s = (s * 1664525 + 1013904223) | 0;
    return (s >>> 0) / 4294967296;
  };
}

function randomParams(rng: () => number, space: SearchSpace): Record<string, number> {
  const p: Record<string, number> = {};
  for (const [k, r] of Object.entries(space)) {
    const val = r.min + rng() * (r.max - r.min);
    p[k] = r.step ? Math.round(val / r.step) * r.step : val;
  }
  return p;
}

function crossover(a: Record<string, number>, b: Record<string, number>, rng: () => number): Record<string, number> {
  const child: Record<string, number> = {};
  for (const k of Object.keys(a)) {
    child[k] = rng() < 0.5 ? a[k]! : b[k]!;
  }
  return child;
}

function blendCrossover(a: Record<string, number>, b: Record<string, number>, rng: () => number): Record<string, number> {
  const child: Record<string, number> = {};
  for (const k of Object.keys(a)) {
    const low = Math.min(a[k]!, b[k]!);
    const high = Math.max(a[k]!, b[k]!);
    const ext = (high - low) * 0.25;
    child[k] = low - ext + rng() * (high - low + 2 * ext);
  }
  return child;
}

function mutate(params: Record<string, number>, space: SearchSpace, rate: number, rng: () => number): Record<string, number> {
  const p = { ...params };
  for (const k of Object.keys(p)) {
    if (rng() < rate) {
      const r = space[k]!;
      const delta = (r.max - r.min) * 0.1 * (rng() * 2 - 1);
      const val = p[k]! + delta;
      p[k] = Math.max(r.min, Math.min(r.max, val));
      if (r.step) p[k] = Math.round(p[k]! / r.step) * r.step;
    }
  }
  return p;
}

export async function geneticOptimize(
  baseConfig: BacktestConfig,
  space: SearchSpace,
  bars: BarSeries,
  def_: any,
  options?: GAOptions,
): Promise<OptimizerResult> {
  const start = Date.now();
  const popSize = options?.populationSize ?? 30;
  const generations = options?.generations ?? 20;
  const mutationRate = options?.mutationRate ?? 0.15;
  const crossoverRate = options?.crossoverRate ?? 0.8;
  const tournamentSize = options?.tournamentSize ?? 3;
  const elitismCount = options?.elitismCount ?? 2;
  const seed = options?.seed ?? 42;
  const obj = options?.objective ?? 'sharpe';

  const rng = createRng(seed);

  function evaluate(params: Record<string, number>): ParamEval {
    const cfg: BacktestConfig = { ...baseConfig, params };
    const engine = new BacktestEngine(cfg, def_);
    const result = engine.run(bars);
    return { params, metrics: result.metrics };
  }

  function tournament(pop: ParamEval[], fitness: (ev: ParamEval) => number): ParamEval {
    let best: ParamEval | null = null;
    let bestF = -Infinity;
    for (let i = 0; i < tournamentSize; i++) {
      const idx = Math.floor(rng() * pop.length);
      const f = fitness(pop[idx]!);
      if (f > bestF) { bestF = f; best = pop[idx]!; }
    }
    return best!;
  }

  // Initialize population
  let population: ParamEval[] = [];
  for (let i = 0; i < popSize; i++) {
    population.push(evaluate(randomParams(rng, space)));
  }

  const history: GAPopulation[] = [];

  for (let gen = 0; gen < generations; gen++) {
    const fitness = (ev: ParamEval) => extractMetric(ev.metrics, obj);

    // Sort by fitness
    population.sort((a, b) => fitness(b) - fitness(a));

    const bestF = fitness(population[0]!);
    const avgF = population.reduce((s, p) => s + fitness(p), 0) / population.length;

    history.push({
      generation: gen,
      individuals: [...population],
      bestFitness: bestF,
      avgFitness: avgF,
    });

    // Build next generation
    const next: ParamEval[] = [];

    // Elitism
    for (let i = 0; i < elitismCount && i < population.length; i++) {
      next.push(population[i]!);
    }

    while (next.length < popSize) {
      const parentA = tournament(population, fitness);
      const parentB = tournament(population, fitness);
      let childParams: Record<string, number>;

      if (rng() < crossoverRate) {
        childParams = blendCrossover(parentA.params, parentB.params, rng);
      } else {
        childParams = crossover(parentA.params, parentB.params, rng);
      }

      childParams = mutate(childParams, space, mutationRate, rng);
      next.push(evaluate(childParams));
    }

    population = next;
  }

  // Final sort
  const fitness = (ev: ParamEval) => extractMetric(ev.metrics, obj);
  population.sort((a, b) => fitness(b) - fitness(a));

  return {
    best: population[0]!,
    evaluations: population,
    totalEvals: popSize + generations * (popSize - Math.min(elitismCount, popSize)),
    durationMs: Date.now() - start,
    objective: obj,
    gaHistory: history,
  };
}