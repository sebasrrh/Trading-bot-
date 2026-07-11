import type { Position } from '@tradeboard/core';
import type { StrategyDef, StrategyContext, IndicatorAccessor } from '@tradeboard/strategies';
import { sma, ema, rsi, atr, wma, macd, bbands, donchian, stoch } from '@tradeboard/indicators';

export interface AutoStrategyBinding {
  id: string;
  strategyId: string;
  params: unknown;
  symbol: string;
  allocation: number;
  enabled: boolean;
}

export interface PaperAccount {
  id: string;
  owner: string;
  createdAt: number;
  startingCash: number;
  cash: number;
  positions: Map<string, Position>;
  openOrders: PaperOrder[];
  fills: Fill[];
  equityMarks: { t: number; equity: number }[];
  autoStrategies: AutoStrategyBinding[];
}

export type OrderType = 'market' | 'limit' | 'stop';
export type OrderSide = 'buy' | 'sell';
export type TIF = 'day' | 'gtc';

export interface NewOrder {
  symbol: string;
  side: OrderSide;
  qty: number;
  type: OrderType;
  limitPrice?: number;
  stopPrice?: number;
  tif?: TIF;
  note?: string;
}

export interface PaperOrder extends NewOrder {
  id: string;
  placedAt: number;
  status: 'open' | 'filled' | 'cancelled' | 'expired';
  filledQty: number;
  filledAvgPrice?: number;
}

export interface Fill {
  id: string;
  t: number;
  symbol: string;
  side: OrderSide;
  qty: number;
  price: number;
  costs: number;
  orderId: string;
  pnl?: number;
}

interface BarBuffer {
  t: number[]; o: number[]; h: number[]; l: number[]; c: number[]; v: number[];
}

export class PaperEngine {
  account: PaperAccount;
  private strategies: Map<string, StrategyDef> = new Map();
  private barBuf: Map<string, BarBuffer> = new Map();
  private oid = 0;

  constructor(account: PaperAccount, defs?: StrategyDef[]) {
    this.account = account;
    if (defs) for (const d of defs) this.strategies.set(d.id, d);
  }

  registerStrategy(def: StrategyDef): void { this.strategies.set(def.id, def); }

  private nextOid(): string { return `po_${++this.oid}_${Date.now().toString(36)}`; }
  private nextFid(): string { return `f_${Date.now().toString(36)}_${(++this.oid).toString(36)}`; }

  placeOrder(n: NewOrder): PaperOrder {
    const o: PaperOrder = { ...n, id: this.nextOid(), placedAt: Date.now(), status: 'open', filledQty: 0, tif: n.tif ?? 'day' };
    this.account.openOrders.push(o);
    return o;
  }

  cancelOrder(id: string): boolean {
    const i = this.account.openOrders.findIndex(o => o.id === id);
    if (i < 0) return false;
    this.account.openOrders[i]!.status = 'cancelled';
    this.account.openOrders.splice(i, 1);
    return true;
  }

  processBar(symbol: string, open: number, high: number, low: number, close: number, volume: number, t: number): Fill[] {
    const fills: Fill[] = [];

    for (const o of this.account.openOrders) {
      if (o.symbol !== symbol || o.status !== 'open') continue;
      if (o.side === 'sell') {
        const pos = this.account.positions.get(symbol);
        const avail = pos?.qty ?? 0;
        if (o.qty > avail) o.qty = avail;
        if (o.qty <= 0) continue;
      }
      const f = this.tryFill(o, open, high, low, close, t);
      if (f) { fills.push(f); this.account.fills.push(f); this.applyFill(f, o); }
    }

    for (const pos of this.account.positions.values()) {
      const curPx = pos.symbol === symbol ? close : pos.avgPrice;
      pos.unrealizedPnl = pos.qty * curPx - pos.qty * pos.avgPrice;
    }

    const af = this.runAuto(symbol, open, high, low, close, volume, t);
    fills.push(...af);

    this.account.equityMarks.push({ t, equity: this.equity().total });
    return fills;
  }

  processBars(symbol: string, bars: { t: number; o: number; h: number; l: number; c: number; v: number }[]): Fill[] {
    const all: Fill[] = [];
    for (const b of bars) {
      all.push(...this.processBar(symbol, b.o, b.h, b.l, b.c, b.v, b.t));
    }
    return all;
  }

  private tryFill(o: PaperOrder, open: number, high: number, low: number, _close: number, t: number): Fill | null {
    const buy = o.side === 'buy';
    let px = 0, ok = false;
    switch (o.type) {
      case 'market': px = open; ok = true; break;
      case 'limit': {
        const lp = o.limitPrice!;
        if (buy && low <= lp) { px = Math.min(open, lp); ok = true; }
        else if (!buy && high >= lp) { px = Math.max(open, lp); ok = true; }
        break;
      }
      case 'stop': {
        const sp = o.stopPrice!;
        if (buy && high >= sp) { px = Math.max(open, sp); ok = true; }
        else if (!buy && low <= sp) { px = Math.min(open, sp); ok = true; }
        break;
      }
    }
    if (!ok) return null;
    const cost = o.qty * px * 0.0005;
    o.status = 'filled'; o.filledQty = o.qty; o.filledAvgPrice = px;
    this.account.openOrders = this.account.openOrders.filter(x => x.id !== o.id);
    return { id: this.nextFid(), t, symbol: o.symbol, side: o.side, qty: o.qty, price: px, costs: cost, orderId: o.id };
  }

  private applyFill(f: Fill, _o: PaperOrder): void {
    if (f.side === 'buy') {
      this.account.cash -= f.qty * f.price + f.costs;
      const pos = this.account.positions.get(f.symbol);
      if (pos) {
        const tq = pos.qty + f.qty;
        pos.avgPrice = (pos.qty * pos.avgPrice + f.qty * f.price) / tq;
        pos.qty = tq;
      } else {
        this.account.positions.set(f.symbol, { symbol: f.symbol, qty: f.qty, avgPrice: f.price, unrealizedPnl: 0 });
      }
    } else {
      this.account.cash += f.qty * f.price - f.costs;
      const pos = this.account.positions.get(f.symbol);
      if (pos) {
        f.pnl = (f.price - pos.avgPrice) * f.qty;
        pos.qty -= f.qty;
        if (pos.qty <= 0) this.account.positions.delete(f.symbol);
      }
    }
  }

  private runAuto(symbol: string, open: number, high: number, low: number, close: number, volume: number, t: number): Fill[] {
    const fills: Fill[] = [];
    let buf = this.barBuf.get(symbol);
    if (!buf) { buf = { t: [], o: [], h: [], l: [], c: [], v: [] }; this.barBuf.set(symbol, buf); }
    buf.t.push(t); buf.o.push(open); buf.h.push(high); buf.l.push(low); buf.c.push(close); buf.v.push(volume);
    if (buf.t.length > 500) { buf.t.shift(); buf.o.shift(); buf.h.shift(); buf.l.shift(); buf.c.shift(); buf.v.shift(); }

    const bars = {
      symbol, timeframe: '' as any,
      t: new Float64Array(buf.t), o: new Float64Array(buf.o), h: new Float64Array(buf.h),
      l: new Float64Array(buf.l), c: new Float64Array(buf.c), v: new Float64Array(buf.v), length: buf.t.length,
    };

    const ic = new Map<string, any>();
    const ia: IndicatorAccessor = {
      sma: (p: number) => cacheOr(ic, `sma${p}`, () => sma(bars, p)),
      ema: (p: number) => cacheOr(ic, `ema${p}`, () => ema(bars, p)),
      rsi: (p: number) => cacheOr(ic, `rsi${p}`, () => rsi(bars, p)),
      atr: (p: number) => cacheOr(ic, `atr${p}`, () => atr(bars, p)),
      wma: (p: number) => cacheOr(ic, `wma${p}`, () => wma(bars, p)),
      macd: (f: number, s: number, sig: number) => cacheOr(ic, `macd${f}${s}${sig}`, () => macd(bars, f, s, sig)),
      bbands: (p: number, sd: number) => cacheOr(ic, `bb${p}${sd}`, () => bbands(bars, p, sd)),
      donchian: (p: number) => cacheOr(ic, `don${p}`, () => donchian(bars, p)),
      stoch: (k: number, d: number) => cacheOr(ic, `st${k}${d}`, () => stoch(bars, k, d)),
    };

    for (const b of this.account.autoStrategies) {
      if (b.symbol !== symbol || !b.enabled) continue;
      const def = this.strategies.get(b.strategyId);
      if (!def || buf.t.length < def.warmupBars(b.params)) continue;

      const inst = def.create(b.params);
      const i = buf.t.length - 1;
      const pos = this.account.positions.get(symbol) ?? { symbol, qty: 0, avgPrice: 0, unrealizedPnl: 0 };
      const ctx: StrategyContext = {
        i, bars, close: (off = 0) => buf.t.length > off ? buf.c[buf.t.length - 1 - off]! : NaN,
        indicator: ia, position: pos, equity: this.equity().total,
        log: (_msg: string, _data?: unknown) => {},
      };
      const sig = inst.onBar(ctx);
      if (!sig || sig.target === 0) continue;

      const buy = sig.target > 0;
      const alloc = this.account.cash * b.allocation;
      const dq = buy ? Math.max(1, Math.floor(alloc / close)) : (pos?.qty ?? 0);
      if (dq <= 0) continue;

      const o = this.placeOrder({ symbol, side: buy ? 'buy' : 'sell', qty: dq, type: 'market', note: sig.reason });
      const f = this.tryFill(o, open, high, low, close, t);
      if (f) { this.account.fills.push(f); this.applyFill(f, o); fills.push(f); }
    }
    return fills;
  }

  equity(): { cash: number; posValue: number; total: number; ret: number } {
    let pv = 0;
    for (const p of this.account.positions.values()) pv += p.qty * p.avgPrice;
    const total = this.account.cash + pv;
    return { cash: this.account.cash, posValue: pv, total, ret: this.account.startingCash > 0 ? total / this.account.startingCash - 1 : 0 };
  }
}

function cacheOr<T>(cache: Map<string, T>, k: string, fn: () => T): T {
  const c = cache.get(k);
  if (c) return c;
  const r = fn();
  cache.set(k, r);
  return r;
}
export function createAccount(owner: string, startingCash = 100_000): PaperAccount {
  return {
    id: Math.random().toString(36).slice(2) + Date.now().toString(36),
    owner,
    createdAt: Date.now(),
    startingCash,
    cash: startingCash,
    positions: new Map(),
    openOrders: [],
    fills: [],
    equityMarks: [],
    autoStrategies: [],
  };
}
