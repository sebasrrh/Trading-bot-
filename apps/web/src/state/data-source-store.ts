import { create } from 'zustand';

export interface DataSourceState {
  source: string | null;
  warnings: string[];
  updatedAt: number;
  report: (source: string | null, warnings: string[]) => void;
}

// Populated by useBars/useQuotes on every response so the topbar badge
// (apps/web/src/app/app.tsx) reflects who actually served the last request
// instead of a hardcoded provider name.
export const useDataSourceStore = create<DataSourceState>((set) => ({
  source: null,
  warnings: [],
  updatedAt: 0,
  report: (source, warnings) => set({ source, warnings, updatedAt: Date.now() }),
}));
