import { create } from 'zustand';
import { PaperEngine, createAccount } from '@tradeboard/paper';
import { allStrategies } from '@tradeboard/strategies';

let _engine: PaperEngine | null = null;
let _initialized = false;

function getEngine(): PaperEngine {
  if (!_initialized) {
    _initialized = true;
    const acct = createAccount('Trader', 100_000);
    const eng = new PaperEngine(acct, allStrategies());
    eng.account.autoStrategies.push({
      id: 's1', strategyId: 'sma-cross', params: { fast: 10, slow: 30 }, symbol: 'SPY', allocation: 0.5, enabled: false,
    });
    eng.account.autoStrategies.push({
      id: 's2', strategyId: 'rsi-mean-revert', params: { period: 14, oversold: 30, overbought: 70 }, symbol: 'SPY', allocation: 0.3, enabled: false,
    });
    _engine = eng;
  }
  return _engine!;
}

interface PaperStoreState {
  refreshKey: number;
  bump: () => void;
  getEngine: () => PaperEngine;
}

export const usePaperStore = create<PaperStoreState>((set) => ({
  refreshKey: 0,
  bump: () => set((s) => ({ refreshKey: s.refreshKey + 1 })),
  getEngine,
}));