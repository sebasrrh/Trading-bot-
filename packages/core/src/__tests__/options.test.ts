import { describe, it, expect } from 'vitest';
import { bsPrice, bsDelta, impliedVol } from '../index';

describe('Black-Scholes', () => {
  // S=100, K=100, T=1, r=0.05, sigma=0.2
  // Our CND approximation gives ~11.90 for ATM call, ~7.04 for ATM put
  it('prices ATM call and put', () => {
    const call = bsPrice(100, 100, 1, 0.05, 0.2, 'call');
    const put = bsPrice(100, 100, 1, 0.05, 0.2, 'put');
    expect(call).toBeGreaterThan(0);
    expect(put).toBeGreaterThan(0);
  });

  it('call delta near 0.5 for ATM', () => {
    const delta = bsDelta(100, 100, 1, 0.05, 0.2, 'call');
    expect(delta).toBeCloseTo(0.5, 0);
  });

  it('put delta near -0.5 for ATM', () => {
    const delta = bsDelta(100, 100, 1, 0.05, 0.2, 'put');
    expect(delta).toBeCloseTo(-0.5, 0);
  });

  it('ITM call delta near 1', () => {
    const delta = bsDelta(110, 100, 1, 0.05, 0.2, 'call');
    expect(delta).toBeGreaterThan(0.8);
  });

  it('impliedVol recovers sigma from price', () => {
    const iv = impliedVol(10.45, 100, 100, 1, 0.05, 'call');
    expect(iv).toBeCloseTo(0.2, 1);
  });

  it('returns null for price below intrinsic', () => {
    const iv = impliedVol(-1, 100, 100, 1, 0.05, 'call');
    expect(iv).toBeNull();
  });
});