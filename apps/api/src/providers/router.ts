import type { Bar, Quote, SymbolInfo } from '@tradeboard/core';
import type { ProviderAdapter, BarsRequest, ProviderId } from './types';
import { stooqAdapter } from './stooq';
import { yahooAdapter } from './yahoo';

const providers: ProviderAdapter[] = [
  stooqAdapter,
  yahooAdapter,
];

export interface RoutedResponse<T> {
  source: ProviderId;
  data: T;
}

export async function routeBars(req: BarsRequest): Promise<RoutedResponse<Bar[]>> {
  for (const p of providers) {
    if (!p.capabilities.has('bars-1D')) continue;
    try {
      const bars = await p.getBars(req);
      if (bars.length > 0) return { source: p.id, data: bars };
    } catch { continue; }
  }
  return { source: 'stooq', data: [] };
}

export async function routeQuotes(symbols: string[]): Promise<RoutedResponse<Quote[]>> {
  for (const p of providers) {
    if (!p.capabilities.has('quote')) continue;
    try {
      const quotes = await p.getQuotes(symbols);
      if (quotes.length > 0) return { source: p.id, data: quotes };
    } catch { continue; }
  }
  return { source: 'stooq', data: [] };
}

export async function routeSearch(q: string): Promise<SymbolInfo[]> {
  for (const p of providers) {
    if (!p.search) continue;
    try {
      const results = await p.search(q);
      if (results.length > 0) return results;
    } catch { continue; }
  }
  return [];
}