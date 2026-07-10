import type { BarSeries } from '@tradeboard/core';
import type { StrategyInstance } from '@tradeboard/strategies';

export interface CostModel {
  commissionPerOrder: number;
  spreadBps: number;
  slippageBps: number;
}

export type SizingModel =
  | { kind: 'all-in' }
  | { kind: 'fixed-fraction'; fraction: number }
  | { kind: 'fixed-dollar'; dollars: number }
  | { kind: 'volatility'; targetAnnualVol: number };

export interface BacktestConfig {
  strategyId: string;
  params: unknown;
  symbol: string;
  timeframe: '1D' | '1h' | '5m' | '15m' | '1m' | '1W';
  from: number;
  to: number;
  initialCash: number;
  costModel: CostModel;
  sizing: SizingModel;
  allowShort: boolean;
  seed: number;
}

export interface Trade {
  entryT: number;
  exitT: number;
  side: 'long' | 'short';
  qty: number;
  entryPx: number;
  exitPx: number;
  pnl: number;
  pnlPct: number;
  bars: number;
  reason: string;
}

export interface SignalMarker {
  t: number;
  side: 'buy' | 'sell';
  reason: string;
}

export interface Metrics {
  totalReturn: number;
  cagr: number;
  volatility: number;
  sharpe: number;
  sortino: number;
  maxDrawdown: number;
  calmar: number;
  winRate: number;
  profitFactor: number;
  avgTrade: number;
  exposure: number;
  turnover: number;
  tradeCount: number;
}

export interface LogLine {
  bar: number;
  message: string;
  data?: unknown;
}

export interface RunResult {
  runId: string;
  config: BacktestConfig;
  startedAt: number;
  dataCoverage: { from: number; to: number };
  equity: Float64Array;
  returnsPerBar: Float64Array;
  drawdown: Float64Array;
  trades: Trade[];
  markers: SignalMarker[];
  metrics: Metrics;
  logs: LogLine[];
}

export class BacktestEngine {
  private readonly config: BacktestConfig;

  constructor(config: BacktestConfig, _strategy: StrategyInstance) {
    this.config = config;
  }

  run(_bars: BarSeries): RunResult {
    const runId = Math.random().toString(36).slice(2) + Date.now().toString(36);
    return {
      runId,
      config: this.config,
      startedAt: Date.now(),
      dataCoverage: { from: 0, to: 0 },
      equity: new Float64Array(0),
      returnsPerBar: new Float64Array(0),
      drawdown: new Float64Array(0),
      trades: [],
      markers: [],
      metrics: {
        totalReturn: 0, cagr: 0, volatility: 0, sharpe: 0, sortino: 0,
        maxDrawdown: 0, calmar: 0, winRate: 0, profitFactor: 0,
        avgTrade: 0, exposure: 0, turnover: 0, tradeCount: 0,
      },
      logs: [],
    };
  }
}