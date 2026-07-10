import { describe, it, expect } from 'vitest';
import { toBarSeries } from '@tradeboard/core';
import { sma, ema, rsi } from '../index';

describe('indicators', () => {
  const series = toBarSeries('TEST', '1D', Array.from({ length: 100 }, (_, i) => ({
    t: i * 86400000, o: 100 + i, h: 101 + i, l: 99 + i, c: 100 + i, v: 1000,
  })));

  it('sma returns correct length', () => {
    const result = sma(series, 20);
    expect(result.length).toBe(100);
    expect(result[18]).toBeNaN();
    expect(result[19]).not.toBeNaN();
  });

  it('sma first valid value equals average of first 20 closes', () => {
    const result = sma(series, 20);
    let sum = 0;
    for (let i = 0; i < 20; i++) sum += series.c[i]!;
    expect(result[19]).toBeCloseTo(sum / 20, 6);
  });

  it('ema returns correct length', () => {
    const result = ema(series, 14);
    expect(result.length).toBe(100);
    expect(result[12]).toBeNaN();
    expect(result[13]).not.toBeNaN();
  });

  it('rsi returns values between 0 and 100', () => {
    const result = rsi(series, 14);
    for (let i = 14; i < 100; i++) {
      expect(result[i]).toBeGreaterThanOrEqual(0);
      expect(result[i]).toBeLessThanOrEqual(100);
    }
  });
});