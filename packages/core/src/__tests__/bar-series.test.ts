import { describe, it, expect } from 'vitest';
import { toBarSeries, sliceBars, mergeBars } from '../index';

const makeBars = () => toBarSeries('SPY', '1D', [
  { t: 1, o: 100, h: 101, l: 99, c: 100, v: 1000 },
  { t: 2, o: 101, h: 102, l: 100, c: 101, v: 1100 },
  { t: 3, o: 102, h: 103, l: 101, c: 102, v: 1200 },
  { t: 4, o: 103, h: 104, l: 102, c: 103, v: 1300 },
]);

describe('BarSeries', () => {
  it('toBarSeries sorts and creates columnar arrays', () => {
    const s = makeBars();
    expect(s.length).toBe(4);
    expect(s.t[0]).toBe(1);
    expect(s.c[3]).toBe(103);
  });

  it('sliceBars returns correct range', () => {
    const s = makeBars();
    const sliced = sliceBars(s, 2, 3);
    expect(sliced.length).toBe(2);
    expect(sliced.t[0]).toBe(2);
  });

  it('mergeBars deduplicates on t and keeps ascending', () => {
    const a = toBarSeries('SPY', '1D', [
      { t: 1, o: 100, h: 101, l: 99, c: 100, v: 1000 },
      { t: 2, o: 101, h: 102, l: 100, c: 101, v: 1100 },
    ]);
    const b = toBarSeries('SPY', '1D', [
      { t: 2, o: 200, h: 201, l: 199, c: 200, v: 2000 },
      { t: 3, o: 102, h: 103, l: 101, c: 102, v: 1200 },
    ]);
    const merged = mergeBars(a, b);
    expect(merged.length).toBe(3);
    expect(merged.c[1]).toBe(200); // b wins at t=2
    expect(merged.t[2]).toBe(3);
  });

  it('returns empty series from empty input', () => {
    const s = toBarSeries('SPY', '1D', []);
    expect(s.length).toBe(0);
  });
});