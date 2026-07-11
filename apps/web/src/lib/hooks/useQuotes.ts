import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../api';
import { useDataSourceStore } from '../../state/data-source-store';

interface QuoteData { symbol: string; price: number; ts: number; change: number; changePct: number; prevClose: number; }
interface QuotesResponse { source: string | null; warnings: string[]; quotes: QuoteData[]; }

export function useQuotes(symbols: string[], intervalMs?: number) {
  const report = useDataSourceStore((s) => s.report);
  const query = useQuery<QuotesResponse>({
    queryKey: ['quotes', [...symbols].sort()],
    queryFn: () => apiFetch<QuotesResponse>(`/api/quotes?symbols=${symbols.map(s => encodeURIComponent(s)).join(',')}`),
    enabled: symbols.length > 0,
    refetchInterval: intervalMs ?? 5_000,
    staleTime: Math.max(1_000, (intervalMs ?? 5_000) - 1_000),
  });

  useEffect(() => {
    if (query.data) report(query.data.source, query.data.warnings ?? []);
  }, [query.data, report]);

  return query;
}

export function useQuote(symbol: string, intervalMs?: number) {
  const q = useQuotes(symbol ? [symbol] : [], intervalMs);
  return { ...q, data: q.data?.quotes?.[0] ?? null };
}
