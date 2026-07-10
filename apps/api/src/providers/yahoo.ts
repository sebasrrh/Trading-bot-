import { BarSchema } from '@tradeboard/core';
import type { Bar, Quote, SymbolInfo } from '@tradeboard/core';
import type { ProviderAdapter, BarsRequest, ProviderId, Capability } from './types';

function abortAfterMs(ms: number): AbortSignal {
  const c = new AbortController();
  setTimeout(() => c.abort(), ms);
  return c.signal;
}

const YAHOO_CHART = 'https://query1.finance.yahoo.com/v8/finance/chart';
// Yahoo rejects default fetch user agents with 403/429.
const UA = { 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36' };

export const yahooAdapter: ProviderAdapter = {
  id: 'yahoo' as ProviderId,
  capabilities: new Set<Capability>(['bars-1D', 'quote', 'search', 'options-chain']),

  async getBars(req: BarsRequest): Promise<Bar[]> {
    const range = Math.ceil((req.to - req.from) / 86_400_000) > 365 ? 'max' : '1y';
    const url = `${YAHOO_CHART}/${req.symbol}?interval=1d&range=${range}`;
    const res = await fetch(url, { signal: abortAfterMs(8_000), headers: UA });
    if (!res.ok) throw new Error(`yahoo HTTP ${res.status} for ${req.symbol}`);
    const json = await res.json() as any;
    const result = json?.chart?.result?.[0];
    if (!result) return [];
    const { timestamp, indicators } = result;
    const { quote, adjclose } = indicators;
    if (!quote?.[0]) return [];
    const q = quote[0];
    const ac = adjclose?.[0]?.adjclose ?? null;
    const bars: Bar[] = [];
    for (let i = 0; i < timestamp.length; i++) {
      const t = timestamp[i]! * 1000;
      if (t < req.from || t > req.to) continue;
      const o = q.open[i]!;
      const h = q.high[i]!;
      const l = q.low[i]!;
      const c = q.close[i]!;
      const v = q.volume[i]!;
      if (o == null || h == null || l == null || c == null || v == null) continue;
      // Split/dividend adjustment: scale every field by adjclose/close.
      const f = ac?.[i] != null && c !== 0 ? ac[i]! / c : 1;
      bars.push(BarSchema.parse({
        t, o: o * f, h: h * f, l: l * f, c: c * f, v,
      }));
    }
    bars.sort((a, b) => a.t - b.t);
    return bars;
  },

  async getQuotes(symbols: string[]): Promise<Quote[]> {
    const results = await Promise.all(symbols.map(async (symbol) => {
      try {
        const url = `${YAHOO_CHART}/${symbol}?interval=1d&range=5d`;
        const res = await fetch(url, { signal: abortAfterMs(5_000), headers: UA });
        if (!res.ok) return null;
        const json = await res.json() as any;
        const result = json?.chart?.result?.[0];
        if (!result) return null;
        const meta = result.meta;
        const q = result.indicators?.quote?.[0];
        if (!meta || !q) return null;
        const close = q.close?.filter((v: number | null) => v != null) as number[] | undefined;
        const prevClose = close?.[close.length! - 2] ?? meta.previousClose ?? close?.[0] ?? 0;
        const price = meta.regularMarketPrice ?? close?.[close.length! - 1] ?? 0;
        const change = price - prevClose;
        const changePct = prevClose ? ((change / prevClose) * 100) : 0;
        return { symbol, price, ts: Date.now(), change, changePct, prevClose };
      } catch { return null; }
    }));
    return results.filter((q): q is Quote => q !== null);
  },

  async search(q: string): Promise<SymbolInfo[]> {
    if (q.length < 1) return [];
    const url = `https://query1.finance.yahoo.com/v1/finance/search?q=${encodeURIComponent(q)}`;
    try {
      const res = await fetch(url, { signal: abortAfterMs(5_000), headers: UA });
      if (!res.ok) return [];
      const json = await res.json() as any;
      return (json.quotes ?? []).map((item: any) => ({
        symbol: item.symbol,
        name: item.shortname ?? item.longname ?? item.symbol,
        exchange: item.exchange ?? '',
        type: (item.quoteType ?? '').toLowerCase() === 'etf' ? 'etf' as const : 'stock' as const,
      }));
    } catch { return []; }
  },
};