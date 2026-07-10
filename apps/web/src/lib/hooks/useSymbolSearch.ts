import { useQuery } from '@tanstack/react-query';
import { apiFetch } from '../api';

interface SymbolInfo { symbol: string; name: string; exchange: string; type: 'stock' | 'etf'; }

export function useSymbolSearch(q: string) {
  return useQuery<SymbolInfo[]>({
    queryKey: ['symbol-search', q],
    queryFn: () => apiFetch<SymbolInfo[]>(`/api/search?q=${encodeURIComponent(q)}`),
    enabled: q.length >= 1,
    staleTime: 60_000 * 60 * 24,
  });
}