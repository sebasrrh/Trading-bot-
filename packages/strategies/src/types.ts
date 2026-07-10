import type { BarSeries, Signal } from '@tradeboard/core';
import type { z } from 'zod';

export interface IndicatorAccessor {
  sma(period: number): Float64Array;
  ema(period: number): Float64Array;
  rsi(period: number): Float64Array;
}

export interface StrategyContext {
  readonly i: number;
  readonly bars: BarSeries;
  readonly close: (offset?: number) => number;
  readonly indicator: IndicatorAccessor;
  readonly position: { qty: number; avgPrice: number; unrealizedPnl: number };
  readonly equity: number;
  readonly log: (msg: string, data?: unknown) => void;
}

export interface StrategyDef<P = unknown> {
  id: string;
  name: string;
  description: string;
  paramsSchema: z.ZodType<P, any, any>;
  warmupBars: (params: P) => number;
  create: (params: P) => StrategyInstance;
}

export interface StrategyInstance {
  onBar(ctx: StrategyContext): Signal | null;
  snapshot?(): Record<string, unknown>;
}