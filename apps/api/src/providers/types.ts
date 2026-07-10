import type { Bar, Quote, SymbolInfo } from '@tradeboard/core';
import type { Timeframe } from '@tradeboard/core';

export interface BarsRequest {
  symbol: string;
  timeframe: Timeframe;
  from: number; // epoch ms
  to: number;   // epoch ms
}

export type ProviderId = 'alpaca' | 'polygon' | 'stooq' | 'yahoo';
export type Capability = 'bars-1D' | 'bars-intraday' | 'quote' | 'search' | 'options-chain';

export interface ProviderAdapter {
  id: ProviderId;
  capabilities: Set<Capability>;
  getBars(req: BarsRequest): Promise<Bar[]>;
  getQuotes(symbols: string[]): Promise<Quote[]>;
  search?(q: string): Promise<SymbolInfo[]>;
}