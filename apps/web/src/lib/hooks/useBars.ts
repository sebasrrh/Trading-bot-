import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../api';
import { useDataSourceStore } from '../../state/data-source-store';
import type { Bar, Timeframe } from '@tradeboard/core';

interface BarsResponse {
  source: string | null; warnings: string[]; bars: Bar[]; adjusted: boolean; symbol: string; timeframe: Timeframe;
}

export function useBars(symbol: string, timeframe: Timeframe, from: number, to: number) {
  const report = useDataSourceStore((s) => s.report);
  const query = useQuery<BarsResponse>({
    queryKey: ['bars', symbol, timeframe, from, to],
    queryFn: () => apiFetch<BarsResponse>(`/api/bars?symbol=${encodeURIComponent(symbol)}&timeframe=${timeframe}&from=${from}&to=${to}`),
    enabled: !!symbol,
    staleTime: 30_000,
  });

  useEffect(() => {
    if (query.data) report(query.data.source, query.data.warnings ?? []);
  }, [query.data, report]);

  return query;
}
