import { describe, it, expect } from 'vitest';
import { PaperEngine, createAccount } from '../index';

const testStrategy = {
  id: 'test-buy',
  name: 'Test Always Buy',
  description: 'Buys every bar',
  paramsSchema: { parse: (x: any) => x } as any,
  warmupBars: () => 0,
  create: () => ({
    onBar: () => ({ target: 1, reason: 'test' }),
  }),
};

function makeEngine(cash = 100_000): PaperEngine {
  return new PaperEngine(createAccount('test', cash), [testStrategy]);
}

describe('PaperEngine', () => {
  it('creates account with correct initial state', () => {
    const acct = createAccount('alice', 50_000);
    expect(acct.owner).toBe('alice');
    expect(acct.startingCash).toBe(50_000);
    expect(acct.cash).toBe(50_000);
    expect(acct.positions.size).toBe(0);
    expect(acct.openOrders.length).toBe(0);
    expect(acct.fills.length).toBe(0);
  });

  it('places and fills a market order', () => {
    const eng = makeEngine();
    eng.placeOrder({ symbol: 'SPY', side: 'buy', qty: 10, type: 'market' });
    expect(eng.account.openOrders.length).toBe(1);

    const fills = eng.processBar('SPY', 100, 105, 95, 102, 1000, 1000);
    expect(fills.length).toBe(1);
    expect(fills[0]!.price).toBe(100);
    expect(fills[0]!.qty).toBe(10);
    expect(fills[0]!.side).toBe('buy');
    expect(eng.account.openOrders.length).toBe(0);
    expect(eng.account.cash).toBeLessThan(100_000);
    expect(eng.account.positions.get('SPY')!.qty).toBe(10);
    expect(eng.account.positions.get('SPY')!.avgPrice).toBe(100);
  });

  it('fills limit buy when low crosses limit', () => {
    const eng = makeEngine();
    eng.placeOrder({ symbol: 'AAPL', side: 'buy', qty: 5, type: 'limit', limitPrice: 150 });
    let fills = eng.processBar('AAPL', 155, 160, 152, 158, 2000, 2000);
    expect(fills.length).toBe(0);
    expect(eng.account.openOrders.length).toBe(1);

    fills = eng.processBar('AAPL', 152, 155, 148, 150, 2000, 2001);
    expect(fills.length).toBe(1);
    expect(fills[0]!.price).toBe(150);
    expect(eng.account.positions.get('AAPL')!.qty).toBe(5);
  });

  it('fills limit sell when high crosses limit', () => {
    const eng = makeEngine();
    eng.placeOrder({ symbol: 'MSFT', side: 'buy', qty: 10, type: 'market' });
    eng.processBar('MSFT', 200, 205, 198, 202, 3000, 3000);
    eng.placeOrder({ symbol: 'MSFT', side: 'sell', qty: 10, type: 'limit', limitPrice: 210 });

    let fills = eng.processBar('MSFT', 205, 208, 203, 207, 3000, 3001);
    expect(fills.length).toBe(0);

    fills = eng.processBar('MSFT', 208, 212, 207, 211, 3000, 3002);
    expect(fills.length).toBe(1);
    expect(fills[0]!.price).toBe(210);
    expect(eng.account.positions.size).toBe(0);
  });

  it('fills stop buy when high crosses stop', () => {
    const eng = makeEngine();
    eng.placeOrder({ symbol: 'TSLA', side: 'buy', qty: 3, type: 'stop', stopPrice: 250 });
    let fills = eng.processBar('TSLA', 240, 248, 238, 245, 5000, 5000);
    expect(fills.length).toBe(0);

    fills = eng.processBar('TSLA', 247, 252, 246, 251, 5000, 5001);
    expect(fills.length).toBe(1);
    expect(fills[0]!.price).toBe(250);
  });

  it('cancels an open order', () => {
    const eng = makeEngine();
    const o = eng.placeOrder({ symbol: 'GOOG', side: 'buy', qty: 1, type: 'limit', limitPrice: 100 });
    expect(eng.cancelOrder('nonexistent')).toBe(false);
    expect(eng.cancelOrder(o.id)).toBe(true);
    expect(eng.account.openOrders.length).toBe(0);
  });

  it('tracks equity marks', () => {
    const eng = makeEngine();
    expect(eng.equity().total).toBe(100_000);
    expect(eng.equity().ret).toBe(0);

    eng.placeOrder({ symbol: 'SPY', side: 'buy', qty: 100, type: 'market' });
    eng.processBar('SPY', 100, 105, 95, 110, 1000, 1000);
    expect(eng.account.equityMarks.length).toBe(1);
    expect(eng.equity().ret).toBeLessThan(0);
  });

  it('runs auto strategy', () => {
    const eng = makeEngine();
    eng.account.autoStrategies.push({ id: 'a1', strategyId: 'test-buy', params: {}, symbol: 'SPY', allocation: 1, enabled: true });
    const fills = eng.processBar('SPY', 100, 105, 95, 102, 1000, 1000);
    expect(fills.length).toBeGreaterThan(0);
    expect(eng.account.positions.get('SPY')!.qty).toBeGreaterThan(0);
  });

  it('caps overselling to available position qty', () => {
    const eng = makeEngine();
    eng.placeOrder({ symbol: 'SPY', side: 'buy', qty: 10, type: 'market' });
    eng.processBar('SPY', 100, 105, 95, 102, 1000, 1000);
    // Sell 20 when only 10 owned — should cap to 10
    eng.placeOrder({ symbol: 'SPY', side: 'sell', qty: 20, type: 'market' });
    const fills = eng.processBar('SPY', 101, 106, 100, 103, 1000, 1001);
    expect(fills.length).toBe(1);
    expect(fills[0]!.qty).toBe(10);
    expect(eng.account.positions.size).toBe(0);
    expect(eng.account.cash).toBeGreaterThan(0); // got cash back
  });

  it('processBars handles multiple bars', () => {
    const eng = makeEngine();
    eng.account.autoStrategies.push({ id: 'a1', strategyId: 'test-buy', params: {}, symbol: 'SPY', allocation: 1, enabled: true });
    const bars = [
      { o: 100, h: 105, l: 95, c: 102, v: 1000, t: 1000 },
      { o: 103, h: 108, l: 101, c: 107, v: 1000, t: 1001 },
      { o: 106, h: 110, l: 104, c: 109, v: 1000, t: 1002 },
    ];
    expect(eng.processBars('SPY', bars).length).toBe(bars.length);
  });
});
