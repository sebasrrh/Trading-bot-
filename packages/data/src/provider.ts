import type { Bar, BarsRequest, Timeframe, Quote, SymbolInfo } from '@tradeboard/core';

export interface MarketDataProvider {
  id: 'alpaca' | 'polygon' | 'stooq' | 'yahoo';
  capabilities: Set<'bars-1D' | 'bars-intraday' | 'quote' | 'search' | 'options-chain'>;
  getBars(req: BarsRequest): Promise<Bar[]>;
  getQuotes(symbols: string[]): Promise<Quote[]>;
  search?(q: string): Promise<SymbolInfo[]>;
  getOptionsChain?(underlying: string): Promise<unknown>;
}

export function normalizeTimestamp(t: number, timeframe: Timeframe): number {
  switch (timeframe) {
    case '1D': {
      const d = new Date(t);
      return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
    }
    default: return t;
  }
}

export function dedupBars(bars: Bar[]): Bar[] {
  const seen = new Set<number>();
  return bars.filter(b => {
    if (seen.has(b.t)) return false;
    seen.add(b.t);
    return true;
  });
}

export function sortBars(bars: Bar[]): Bar[] {
  return [...bars].sort((a, b) => a.t - b.t);
}