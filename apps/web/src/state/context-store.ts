import { create } from 'zustand';
import type { Timeframe } from '@tradeboard/core';

export interface LinkChannel {
  symbol: string;
  timeframe: Timeframe;
  dateRange: { from: number; to: number };
}

export type ChannelId = 'A' | 'B' | 'C';

export interface ContextState {
  channels: Record<ChannelId, LinkChannel>;
  defaultChannel: ChannelId;
  setChannel: (ch: ChannelId, update: Partial<LinkChannel>) => void;
  getChannel: (ch: ChannelId | null) => LinkChannel;
  activeView: 'dashboard' | 'lab' | 'sim' | 'paper' | 'settings';
  setActiveView: (v: ContextState['activeView']) => void;
}

const oneYear = 365 * 86_400_000;

export const useContextStore = create<ContextState>((set, get) => {
  const now = Date.now();
  return {
    channels: {
      A: { symbol: 'SPY', timeframe: '1D', dateRange: { from: now - oneYear, to: now } },
      B: { symbol: 'QQQ', timeframe: '1D', dateRange: { from: now - oneYear, to: now } },
      C: { symbol: 'AAPL', timeframe: '1D', dateRange: { from: now - oneYear, to: now } },
    },
    defaultChannel: 'A',
    setChannel: (ch, update) => set((s) => ({
      channels: { ...s.channels, [ch]: { ...s.channels[ch], ...update } },
    })),
    getChannel: (ch) => {
      if (ch && ch in get().channels) return get().channels[ch];
      return get().channels[get().defaultChannel];
    },
    activeView: 'dashboard',
    setActiveView: (v) => set({ activeView: v }),
  };
});