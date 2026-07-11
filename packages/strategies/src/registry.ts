import { smaCross } from './builtin/sma-cross';
import { buyAndHold } from './builtin/buy-and-hold';
import { rsiMeanRevert } from './builtin/rsi-mean-revert';
import { donchianBreakout } from './builtin/donchian-breakout';
import { momentumRotation } from './builtin/momentum-rotation';
import type { StrategyDef } from './types';

const _all: StrategyDef<any>[] = [
  buyAndHold, smaCross, rsiMeanRevert, donchianBreakout, momentumRotation,
];

export function allStrategies(): StrategyDef<any>[] {
  return _all;
}