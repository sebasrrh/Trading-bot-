import type { Timeframe } from './market';

export interface BarSeries {
  symbol: string;
  timeframe: Timeframe;
  t: Float64Array;
  o: Float64Array;
  h: Float64Array;
  l: Float64Array;
  c: Float64Array;
  v: Float64Array;
  length: number;
}

export function toBarSeries(
  symbol: string,
  timeframe: Timeframe,
  bars: { t: number; o: number; h: number; l: number; c: number; v: number }[],
): BarSeries {
  bars.sort((a, b) => a.t - b.t);
  const n = bars.length;
  const t = new Float64Array(n);
  const o = new Float64Array(n);
  const h = new Float64Array(n);
  const l = new Float64Array(n);
  const c = new Float64Array(n);
  const v = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const bar = bars[i]!;
    t[i] = bar.t;
    o[i] = bar.o;
    h[i] = bar.h;
    l[i] = bar.l;
    c[i] = bar.c;
    v[i] = bar.v;
  }
  return { symbol, timeframe, t, o, h, l, c, v, length: n };
}

export function sliceBars(series: BarSeries, from: number, to: number): BarSeries {
  let start = 0;
  let end = series.length;
  for (let i = 0; i < series.length; i++) {
    if (series.t[i]! < from) start = i + 1;
    if (series.t[i]! <= to) end = i + 1;
  }
  const len = end - start;
  return {
    symbol: series.symbol,
    timeframe: series.timeframe,
    t: series.t.slice(start, end) as Float64Array,
    o: series.o.slice(start, end) as Float64Array,
    h: series.h.slice(start, end) as Float64Array,
    l: series.l.slice(start, end) as Float64Array,
    c: series.c.slice(start, end) as Float64Array,
    v: series.v.slice(start, end) as Float64Array,
    length: len,
  };
}

export function mergeBars(a: BarSeries, b: BarSeries): BarSeries {
  if (a.length === 0) return b;
  if (b.length === 0) return a;
  const merged: { t: number; o: number; h: number; l: number; c: number; v: number }[] = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a.t[i]! < b.t[j]!) {
      merged.push({ t: a.t[i]!, o: a.o[i]!, h: a.h[i]!, l: a.l[i]!, c: a.c[i]!, v: a.v[i]! });
      i++;
    } else if (a.t[i]! > b.t[j]!) {
      merged.push({ t: b.t[j]!, o: b.o[j]!, h: b.h[j]!, l: b.l[j]!, c: b.c[j]!, v: b.v[j]! });
      j++;
    } else {
      merged.push({ t: b.t[j]!, o: b.o[j]!, h: b.h[j]!, l: b.l[j]!, c: b.c[j]!, v: b.v[j]! });
      i++; j++;
    }
  }
  while (i < a.length) {
    merged.push({ t: a.t[i]!, o: a.o[i]!, h: a.h[i]!, l: a.l[i]!, c: a.c[i]!, v: a.v[i]! });
    i++;
  }
  while (j < b.length) {
    merged.push({ t: b.t[j]!, o: b.o[j]!, h: b.h[j]!, l: b.l[j]!, c: b.c[j]!, v: b.v[j]! });
    j++;
  }
  return toBarSeries(a.symbol, a.timeframe, merged);
}