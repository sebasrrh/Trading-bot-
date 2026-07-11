import type { Metrics } from '@tradeboard/backtest';

export interface ParamRange {
  min: number;
  max: number;
  step?: number;
}

export interface SearchSpace {
  [param: string]: ParamRange;
}

export type ObjectiveMetric = 'sharpe' | 'sortino' | 'calmar' | 'cagr' | 'profitFactor' | 'totalReturn';

export interface ParamEval {
  params: Record<string, number>;
  metrics: Metrics;
}

export interface GAPopulation {
  generation: number;
  individuals: ParamEval[];
  bestFitness: number;
  avgFitness: number;
}

export interface OptimizerResult {
  best: ParamEval;
  evaluations: ParamEval[];
  totalEvals: number;
  durationMs: number;
  objective: ObjectiveMetric;
  gaHistory?: GAPopulation[];
}

export interface GAOptions {
  populationSize?: number;
  generations?: number;
  mutationRate?: number;
  crossoverRate?: number;
  tournamentSize?: number;
  elitismCount?: number;
  seed?: number;
  objective?: ObjectiveMetric;
}