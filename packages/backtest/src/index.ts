import type { BarSeries } from '@tradeboard/core';
import type { StrategyDef, StrategyContext, IndicatorAccessor } from '@tradeboard/strategies';
import { sma, ema, rsi, atr, wma, macd, bbands, donchian, stoch } from '@tradeboard/indicators';

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

function computeMetrics(equity: Float64Array, returns: Float64Array, trades: Trade[], nBars: number, annFactor: number): Metrics {
  const n = equity.length;
  const startEq = equity[0]!;
  const endEq = equity[n - 1]!;
  const totalReturn = startEq > 0 ? endEq / startEq - 1 : 0;
  const years = nBars / annFactor;
  const cagr = years > 0 && startEq > 0 ? Math.pow(endEq / startEq, 1 / years) - 1 : 0;

  let sumR = 0, sumSq = 0, sumDSq = 0, negCount = 0;
  for (let i = 0; i < n; i++) {
    const r = returns[i]!;
    if (!isNaN(r) && isFinite(r)) { sumR += r; sumSq += r * r; if (r < 0) { sumDSq += r * r; negCount++; } }
  }
  const meanR = sumR / n;
  const variance = sumSq / n - meanR * meanR;
  const vol = Math.sqrt(Math.max(0, variance)) * Math.sqrt(annFactor);
  const downsideVariance = negCount > 0 ? sumDSq / negCount : 0;
  const sharpe = vol > 0 ? meanR / Math.sqrt(Math.max(0, variance)) * Math.sqrt(annFactor) : 0;
  const sortino = Math.sqrt(downsideVariance) > 0 ? meanR / Math.sqrt(downsideVariance) * Math.sqrt(annFactor) : 0;

  let maxDD = 0, peak = equity[0]!;
  for (let i = 0; i < n; i++) {
    if (equity[i]! > peak) peak = equity[i]!;
    const dd = peak > 0 ? (equity[i]! / peak - 1) : 0;
    if (dd < maxDD) maxDD = dd;
  }
  const calmar = maxDD < 0 ? Math.abs(cagr / maxDD) : 0;

  const winners = trades.filter(t => t.pnl > 0);
  const losers = trades.filter(t => t.pnl <= 0);
  const winRate = trades.length > 0 ? winners.length / trades.length : 0;
  const grossProfit = winners.reduce((s, t) => s + t.pnl, 0);
  const grossLoss = Math.abs(losers.reduce((s, t) => s + t.pnl, 0));
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;
  const avgTrade = trades.length > 0 ? trades.reduce((s, t) => s + t.pnlPct, 0) / trades.length : 0;

  const barsWithPosition = returns.filter((_, i) => equity[i] !== startEq).length;
  const exposure = n > 0 ? barsWithPosition / n : 0;
  const turnover = 0;

  return { totalReturn, cagr, volatility: vol, sharpe, sortino, maxDrawdown: maxDD, calmar, winRate, profitFactor, avgTrade, exposure, turnover, tradeCount: trades.length };
}

export class BacktestEngine {
  private readonly config: BacktestConfig;
  private readonly def: StrategyDef;
  private readonly annFactor: number;

  constructor(config: BacktestConfig, def: StrategyDef) {
    this.config = config;
    this.def = def;
    this.annFactor = config.timeframe === '1D' ? 252 : config.timeframe === '1h' ? 252 * 6.5 : 252 * 390;
  }

  run(bars: BarSeries): RunResult {
    const cfg = this.config;
    const strategy = this.def.create(cfg.params as any);
    const warmup = this.def.warmupBars(cfg.params as any);
    const n = bars.length;
    const configDigest = JSON.stringify({
      strategyId: cfg.strategyId, params: cfg.params,
      symbol: cfg.symbol, timeframe: cfg.timeframe,
      from: cfg.from, to: cfg.to, cost: cfg.costModel, seed: cfg.seed,
    });
    let hash = 0;
    for (let i = 0; i < configDigest.length; i++) {
      const c = configDigest.charCodeAt(i);
      hash = ((hash << 5) - hash) + c;
      hash |= 0;
    }
    const runId = Math.abs(hash).toString(36).padStart(8, '0');

    const equity = new Float64Array(n);
    const returns = new Float64Array(n);
    const dd = new Float64Array(n);
    const trades: Trade[] = [];
    const markers: SignalMarker[] = [];
    const logs: LogLine[] = [];

    let cash = cfg.initialCash;
    let qty = 0;
    let avgPrice = 0;
    let entryBar = 0;
    let pendingQty = 0;
    let pendingSide: 'buy' | 'sell' | null = null;
    let pendingReason = '';
    let stopPrice = 0;
    

    const indicatorCache = new Map<string, any>();
    const indicatorAccessor: IndicatorAccessor = {
      sma: (p) => cacheOrCompute(indicatorCache, `sma:${p}`, () => sma(bars, p)),
      ema: (p) => cacheOrCompute(indicatorCache, `ema:${p}`, () => ema(bars, p)),
      rsi: (p) => cacheOrCompute(indicatorCache, `rsi:${p}`, () => rsi(bars, p)),
      atr: (p) => cacheOrCompute(indicatorCache, `atr:${p}`, () => atr(bars, p)),
      wma: (p) => cacheOrCompute(indicatorCache, `wma:${p}`, () => wma(bars, p)),
      macd: (f, s, sig) => cacheOrCompute(indicatorCache, `macd:${f}:${s}:${sig}`, () => macd(bars, f, s, sig)),
      bbands: (p, sd) => cacheOrCompute(indicatorCache, `bbands:${p}:${sd}`, () => bbands(bars, p, sd)),
      donchian: (p) => cacheOrCompute(indicatorCache, `donchian:${p}`, () => donchian(bars, p)),
      stoch: (k, d) => cacheOrCompute(indicatorCache, `stoch:${k}:${d}`, () => stoch(bars, k, d)),
    };

    function cacheOrCompute<T>(cache: Map<string, T>, key: string, fn: () => T): T {
      const cached = cache.get(key);
      if (cached) return cached;
      const result = fn();
      cache.set(key, result);
      return result;
    }

    for (let i = 0; i < n; i++) {
      const open = bars.o[i]!;
      
      const low = bars.l[i]!;
      const close = bars.c[i]!;
      const t = bars.t[i]!;

      // Mark to market
      const posValue = qty * close;
      const eq = cash + posValue;
      equity[i] = eq;
      returns[i] = i > 0 && equity[i - 1]! > 0 ? Math.log(eq / equity[i - 1]!) : 0;

      // Check stops
      if (qty > 0 && stopPrice > 0 && low <= stopPrice) {
        const fillPx = Math.min(open, stopPrice);
        const cost = cfg.costModel.commissionPerOrder + fillPx * qty * (cfg.costModel.spreadBps / 2 + cfg.costModel.slippageBps) / 10000;
        const fillQty = qty;
        cash += fillQty * fillPx - cost;
        const pnl = (fillPx - avgPrice) * fillQty - cost;
        trades.push({ entryT: bars.t[entryBar]!, exitT: t, side: 'long', qty: fillQty, entryPx: avgPrice, exitPx: fillPx, pnl, pnlPct: avgPrice > 0 ? (fillPx - avgPrice) / avgPrice : 0, bars: i - entryBar, reason: 'stop-loss' });
        qty = 0; avgPrice = 0; stopPrice = 0; pendingQty = 0; pendingSide = null;
      }

      // Fill pending orders at open
      if (pendingQty !== 0 && pendingSide) {
        const isBuy = pendingSide === 'buy';
        const fillPx = open * (1 + (isBuy ? 1 : -1) * (cfg.costModel.spreadBps / 2 + cfg.costModel.slippageBps) / 10000);
        const cost = cfg.costModel.commissionPerOrder;
        const needed = fillPx * pendingQty;
        if (isBuy && cash >= needed) {
          cash -= needed + cost;
          avgPrice = (avgPrice * qty + fillPx * pendingQty) / (qty + pendingQty);
          qty += pendingQty;
          entryBar = i;
          markers.push({ t: bars.t[i - 1]!, side: 'buy', reason: pendingReason });
        } else if (!isBuy && qty >= pendingQty) {
          cash += fillPx * pendingQty - cost;
          const pnl = (fillPx - avgPrice) * pendingQty - cost;
          trades.push({ entryT: bars.t[entryBar]!, exitT: t, side: 'long', qty: pendingQty, entryPx: avgPrice, exitPx: fillPx, pnl, pnlPct: avgPrice > 0 ? (fillPx - avgPrice) / avgPrice : 0, bars: i - entryBar, reason: pendingReason });
          qty -= pendingQty;
          markers.push({ t: bars.t[i - 1]!, side: 'sell', reason: pendingReason });
          if (qty === 0) avgPrice = 0;
        }
        pendingQty = 0; pendingSide = null; pendingReason = '';
      }

      // Compute drawdown
      let peak = eq;
      for (let j = 0; j <= i; j++) if (equity[j]! > peak) peak = equity[j]!;
      dd[i] = peak > 0 ? eq / peak - 1 : 0;

      // Strategy signal
      if (i >= warmup) {
        const position = { qty, avgPrice, unrealizedPnl: qty * (close - avgPrice) };
        const ctx: StrategyContext = {
          i, bars,
          close: (offset = 0) => {
            const idx = i - offset;
            return idx >= 0 && idx < n ? bars.c[idx]! : NaN;
          },
          indicator: indicatorAccessor,
          position,
          equity: eq,
          log: (msg, data) => { logs.push({ bar: i, message: msg, data }); },
        };
        const signal = strategy.onBar(ctx);
        if (signal) {
          const isBuy = signal.target > 0;
          const sizing = isBuy ? cfg.initialCash * 0.95 : qty; // all-in long
          const desiredQty = Math.floor(sizing / open);
          if (desiredQty > 0 && (isBuy ? true : cfg.allowShort)) {
            pendingQty = isBuy ? desiredQty - qty : qty;
            pendingSide = isBuy ? 'buy' : 'sell';
            pendingReason = signal.reason ?? '';
            stopPrice = signal.stopLoss ?? 0;
            
          }
        }
      }
    }

    // Compute metrics
    const metrics = computeMetrics(equity, returns, trades, n, this.annFactor);

    return {
      runId, config: cfg, startedAt: Date.now(),
      dataCoverage: { from: bars.t[0]!, to: bars.t[n - 1]! },
      equity, returnsPerBar: returns, drawdown: dd,
      trades, markers, metrics, logs,
    };
  }
}

