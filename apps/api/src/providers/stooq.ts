import { BarSchema } from '@tradeboard/core';
import type { Bar, Quote, SymbolInfo } from '@tradeboard/core';
import type { ProviderAdapter, BarsRequest, ProviderId, Capability } from './types';

function abortAfterMs(ms: number): AbortSignal {
  const c = new AbortController();
  setTimeout(() => c.abort(), ms);
  return c.signal;
}

const STOOQ_URL = 'https://stooq.com/q/d/l/';

function parseStooqDate(dateStr: string): number {
  const parts = dateStr.split(/[-/]/);
  return Date.UTC(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2]));
}

export const stooqAdapter: ProviderAdapter = {
  id: 'stooq' as ProviderId,
  capabilities: new Set<Capability>(['bars-1D', 'search']),

  async getBars(req: BarsRequest): Promise<Bar[]> {
    const interval = req.timeframe === '1W' ? 'w' : 'd';
    // Stooq needs the market suffix for US tickers: spy.us, not spy.
    const sym = req.symbol.toLowerCase();
    const stooqSym = sym.includes('.') ? sym : `${sym}.us`;
    const url = `${STOOQ_URL}?s=${stooqSym}&i=${interval}`;

    const res = await fetch(url, { signal: abortAfterMs(8_000) });
    if (!res.ok) throw new Error(`stooq HTTP ${res.status} for ${stooqSym}`);
    const csv = await res.text();
    const lines = csv.trim().split('\n').slice(1);
    const bars: Bar[] = [];

    for (const line of lines) {
      const parts = line.split(',');
      if (parts.length < 6) continue;
      const [dateStr, oStr, hStr, lStr, cStr, vStr] = parts as string[];
      if (!dateStr || !oStr || !hStr || !lStr || !cStr || !vStr) continue;
      const t = parseStooqDate(dateStr);
      if (!t || t < req.from || t > req.to) continue;
      bars.push(BarSchema.parse({
        t, o: Number(oStr), h: Number(hStr), l: Number(lStr),
        c: Number(cStr), v: Number(vStr),
      }));
    }

    bars.sort((a, b) => a.t - b.t);
    return bars;
  },

  async getQuotes(_symbols: string[]): Promise<Quote[]> {
    return [];
  },

  async search(_q: string): Promise<SymbolInfo[]> {
    return [];
  },
};