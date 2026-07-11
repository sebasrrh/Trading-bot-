export type { StrategyDef, StrategyInstance, StrategyContext, IndicatorAccessor } from './types';
export { smaCross } from './builtin/sma-cross';
export { buyAndHold } from './builtin/buy-and-hold';
export { rsiMeanRevert } from './builtin/rsi-mean-revert';
export { donchianBreakout } from './builtin/donchian-breakout';
export { momentumRotation } from './builtin/momentum-rotation';
export { allStrategies } from './registry';