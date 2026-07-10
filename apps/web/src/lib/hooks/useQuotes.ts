import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../api';

interface QuoteData { symbol: string; price: number; ts: number; change: number; changePct: number; prevClose: number; }
interface QuotesResponse { source: string; quotes: QuoteData[]; }

export function useQuotes(symbols: string[]) {
  return useQuery<QuotesResponse>({
    queryKey: ['quotes', [...symbols].sort()],
    queryFn: () => apiFetch<QuotesResponse>(`/api/quotes?symbols=${symbols.map(s => encodeURIComponent(s)).join(',')}`),
    enabled: symbols.length > 0,
    refetchInterval: 5_000,
    staleTime: 4_000,
  });
}

export function useQuote(symbol: string) {
  const q = useQuotes(symbol ? [symbol] : []);
  return { ...q, data: q.data?.quotes?.[0] ?? null };
}