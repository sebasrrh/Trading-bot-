import { create } from 'zustand';
import { PaperEngine, createAccount } from '@tradeboard/paper';
import type { PaperAccount } from '@tradeboard/paper';
import { allStrategies } from '@tradeboard/strategies';

const KEY = 'tradeboard.paper-account.v1';

// PaperAccount.positions is a Map, which JSON.stringify/parse can't round-trip
// directly — serialize it as entries and rebuild the Map on load.
function serializeAccount(acct: PaperAccount): string {
  return JSON.stringify({ ...acct, positions: Array.from(acct.positions.entries()) });
}

function deserializeAccount(raw: string): PaperAccount | null {
  try {
    const parsed = JSON.parse(raw);
    return { ...parsed, positions: new Map(parsed.positions ?? []) };
  } catch {
    return null;
  }
}

function persistAccount(acct: PaperAccount): void {
  try { localStorage.setItem(KEY, serializeAccount(acct)); } catch { /* storage full or unavailable */ }
}

function freshAccount(): PaperAccount {
  const acct = createAccount('Trader', 100_000);
  acct.autoStrategies.push({
    id: 's1', strategyId: 'sma-cross', params: { fast: 10, slow: 30 }, symbol: 'SPY', allocation: 0.5, enabled: false,
  });
  acct.autoStrategies.push({
    id: 's2', strategyId: 'rsi-mean-revert', params: { period: 14, oversold: 30, overbought: 70 }, symbol: 'SPY', allocation: 0.3, enabled: false,
  });
  return acct;
}

let _engine: PaperEngine | null = null;
let _initialized = false;

function getEngine(): PaperEngine {
  if (!_initialized) {
    _initialized = true;
    let acct: PaperAccount | null = null;
    try {
      const saved = localStorage.getItem(KEY);
      if (saved) acct = deserializeAccount(saved);
    } catch { /* storage unavailable */ }

    _engine = new PaperEngine(acct ?? freshAccount(), allStrategies());
  }
  return _engine!;
}

interface PaperStoreState {
  refreshKey: number;
  bump: () => void;
  getEngine: () => PaperEngine;
  /** Wipes the account back to $100k cash, no positions, no history. Cannot be undone. */
  resetAccount: () => void;
}

export const usePaperStore = create<PaperStoreState>((set) => ({
  refreshKey: 0,
  // Every mutation to the engine's account goes through here so persistence
  // never falls out of sync with what the UI is showing.
  bump: () => set((s) => { persistAccount(getEngine().account); return { refreshKey: s.refreshKey + 1 }; }),
  getEngine,
  resetAccount: () => set((s) => {
    _engine = new PaperEngine(freshAccount(), allStrategies());
    try { localStorage.removeItem(KEY); } catch { /* storage unavailable */ }
    return { refreshKey: s.refreshKey + 1 };
  }),
}));
