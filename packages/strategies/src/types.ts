import type { BarSeries, Signal } from '@tradeboard/core';
import type { z } from 'zod';

export interface IndicatorAccessor {
  sma(period: number): Float64Array;
  ema(period: number): Float64Array;
  rsi(period: number): Float64Array;
  atr(period: number): Float64Array;
  wma(period: number): Float64Array;
  macd(fast?: number, slow?: number, signal?: number): { macd: Float64Array; signal: Float64Array; histogram: Float64Array };
  bbands(period?: number, stdDev?: number): { upper: Float64Array; middle: Float64Array; lower: Float64Array };
  donchian(period: number): { upper: Float64Array; middle: Float64Array; lower: Float64Array };
  stoch(kPeriod?: number, dPeriod?: number): { k: Float64Array; d: Float64Array };
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