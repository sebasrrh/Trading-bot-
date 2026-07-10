import type { Position } from '@tradeboard/core';

export interface AutoStrategyBinding {
  strategyId: string;
  params: unknown;
  symbol: string;
  allocation: number;
  enabled: boolean;
}

export interface PaperAccount {
  id: string;
  owner: string;
  createdAt: number;
  startingCash: number;
  cash: number;
  positions: Position[];
  openOrders: PaperOrder[];
  history: Fill[];
  equityMarks: { t: number; equity: number }[];
  autoStrategies: AutoStrategyBinding[];
}

export type PaperOrder = {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  qty: number;
  type: 'market' | 'limit' | 'stop';
  limitPrice?: number;
  stopPrice?: number;
  tif: 'day' | 'gtc';
  placedAt: number;
  note?: string;
};

export interface Fill {
  t: number;
  symbol: string;
  side: 'buy' | 'sell';
  qty: number;
  price: number;
  costs: number;
  orderId: string;
}

export function createAccount(owner: string, startingCash = 100_000): PaperAccount {
  return {
    id: Math.random().toString(36).slice(2) + Date.now().toString(36),
    owner,
    createdAt: Date.now(),
    startingCash,
    cash: startingCash,
    positions: [],
    openOrders: [],
    history: [],
    equityMarks: [],
    autoStrategies: [],
  };
}