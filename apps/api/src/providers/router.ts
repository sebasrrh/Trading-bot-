import type { Bar, Quote, SymbolInfo } from '@tradeboard/core';
import type { ProviderAdapter, BarsRequest, ProviderId } from './types';
import { stooqAdapter } from './stooq';
import { yahooAdapter } from './yahoo';

const providers: ProviderAdapter[] = [
  stooqAdapter,
  yahooAdapter,
];

export interface RoutedResponse<T> {
  /** null = every capable provider failed or returned nothing */
  source: ProviderId | null;
  data: T;
  /** one entry per provider that errored, for the response body + logs */
  warnings: string[];
}

/** All capable providers threw — the caller should surface a 502, not empty data. */
export class AllProvidersFailedError extends Error {
  constructor(public warnings: string[]) {
    super(`all providers failed: ${warnings.join('; ')}`);
  }
}

async function route<T>(
  capable: (p: ProviderAdapter) => boolean,
  call: (p: ProviderAdapter) => Promise<T>,
  nonEmpty: (data: T) => boolean,
): Promise<RoutedResponse<T | null>> {
  const warnings: string[] = [];
  let tried = 0;
  for (const p of providers) {
    if (!capable(p)) continue;
    tried++;
    try {
      const data = await call(p);
      if (nonEmpty(data)) return { source: p.id, data, warnings };
    } catch (err) {
      const msg = `${p.id}: ${err instanceof Error ? err.message : String(err)}`;
      console.warn(`[provider] ${msg}`);
      warnings.push(msg);
    }
  }
  // Every capable provider errored (vs. legitimately returning nothing).
  if (tried > 0 && warnings.length === tried) throw new AllProvidersFailedError(warnings);
  return { source: null, data: null, warnings };
}

export async function routeBars(req: BarsRequest): Promise<RoutedResponse<Bar[]>> {
  const r = await route(
    (p) => p.capabilities.has('bars-1D'),
    (p) => p.getBars(req),
    (bars) => bars.length > 0,
  );
  return { ...r, data: r.data ?? [] };
}

export async function routeQuotes(symbols: string[]): Promise<RoutedResponse<Quote[]>> {
  const r = await route(
    (p) => p.capabilities.has('quote'),
    (p) => p.getQuotes(symbols),
    (quotes) => quotes.length > 0,
  );
  return { ...r, data: r.data ?? [] };
}

export async function routeSearch(q: string): Promise<SymbolInfo[]> {
  const r = await route(
    (p) => p.capabilities.has('search') && !!p.search,
    (p) => p.search!(q),
    (results) => results.length > 0,
  );
  return r.data ?? [];
}
