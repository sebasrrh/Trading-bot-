import { BacktestEngine } from '@tradeboard/backtest';
import { allStrategies } from '@tradeboard/strategies';
import type { BacktestConfig, RunResult } from '@tradeboard/backtest';
import type { BarSeries } from '@tradeboard/core';

interface WorkerRequest {
  id: string;
  kind: 'backtest';
  config: BacktestConfig;
  bars: {
    symbol: string;
    timeframe: string;
    t: number[]; o: number[]; h: number[]; l: number[]; c: number[]; v: number[];
  };
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const { id, config, bars } = e.data;

  const def = allStrategies().find(s => s.id === config.strategyId);
  if (!def) { self.postMessage({ id, type: 'error', message: `unknown strategy: ${config.strategyId}` }); return; }

  const series: BarSeries = {
    symbol: bars.symbol, timeframe: bars.timeframe as any,
    t: new Float64Array(bars.t), o: new Float64Array(bars.o), h: new Float64Array(bars.h),
    l: new Float64Array(bars.l), c: new Float64Array(bars.c), v: new Float64Array(bars.v),
    length: bars.t.length,
  };

  try {
    const engine = new BacktestEngine(config, def as any);
    const result = engine.run(series);

    self.postMessage({
      id, type: 'result',
      payload: {
        runId: result.runId, config: result.config, startedAt: result.startedAt,
        dataCoverage: result.dataCoverage,
        equity: Array.from(result.equity),
        returnsPerBar: Array.from(result.returnsPerBar),
        drawdown: Array.from(result.drawdown),
        trades: result.trades,
        markers: result.markers,
        metrics: result.metrics,
        logs: result.logs,
      } as unknown as RunResult,
    }, undefined as any);
  } catch (err) {
    self.postMessage({ id, type: 'error', message: String(err) });
  }
};