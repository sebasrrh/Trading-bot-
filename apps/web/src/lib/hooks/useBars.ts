import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../api';
import type { Bar, Timeframe } from '@tradeboard/core';

interface BarsResponse {
  source: string; bars: Bar[]; adjusted: boolean; symbol: string; timeframe: Timeframe;
}

export function useBars(symbol: string, timeframe: Timeframe, from: number, to: number) {
  return useQuery<BarsResponse>({
    queryKey: ['bars', symbol, timeframe, from, to],
    queryFn: () => apiFetch<BarsResponse>(`/api/bars?symbol=${encodeURIComponent(symbol)}&timeframe=${timeframe}&from=${from}&to=${to}`),
    enabled: !!symbol,
    staleTime: 30_000,
  });
}