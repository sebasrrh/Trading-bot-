import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../api';

interface QuoteData { symbol: string; price: number; ts: number; change: number; changePct: number; prevClose: number; }
interface QuotesResponse { source: string; quotes: QuoteData[]; }

export function useQuotes(symbols: string[], intervalMs?: number) {
  return useQuery<QuotesResponse>({
    queryKey: ['quotes', [...symbols].sort()],
    queryFn: () => apiFetch<QuotesResponse>(`/api/quotes?symbols=${symbols.map(s => encodeURIComponent(s)).join(',')}`),
    enabled: symbols.length > 0,
    refetchInterval: intervalMs ?? 5_000,
    staleTime: Math.max(1_000, (intervalMs ?? 5_000) - 1_000),
  });
}

export function useQuote(symbol: string, intervalMs?: number) {
  const q = useQuotes(symbol ? [symbol] : [], intervalMs);
  return { ...q, data: q.data?.quotes?.[0] ?? null };
}