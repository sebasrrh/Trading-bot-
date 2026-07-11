import type { BarSeries } from '@tradeboard/core';

export function sma(series: BarSeries, period: number): Float64Array {
  const result = new Float64Array(series.length);
  let sum = 0;
  for (let i = 0; i < series.length; i++) {
    sum += series.c[i]!;
    if (i >= period) sum -= series.c[i - period]!;
    if (i >= period - 1) result[i] = sum / period;
    else result[i] = NaN;
  }
  return result;
}

export function ema(series: BarSeries, period: number): Float64Array {
  const result = new Float64Array(series.length);
  const k = 2 / (period + 1);
  let prev = NaN;
  for (let i = 0; i < series.length; i++) {
    if (i < period - 1) { result[i] = NaN; continue; }
    if (i === period - 1) {
      let sum = 0;
      for (let j = 0; j < period; j++) sum += series.c[j]!;
      prev = sum / period;
      result[i] = prev;
    } else {
      prev = (series.c[i]! - prev) * k + prev;
      result[i] = prev;
    }
  }
  return result;
}

export function rsi(series: BarSeries, period: number): Float64Array {
  const result = new Float64Array(series.length);
  let gain = 0, loss = 0;
  for (let i = 0; i < series.length; i++) {
    if (i < period) { result[i] = NaN;
      if (i > 0) { const d = series.c[i]! - series.c[i - 1]!; if (d > 0) gain += d; else loss -= d; }
      if (i === period - 1) { const ag = gain / period, al = loss / period; result[i] = al === 0 ? 100 : 100 - 100 / (1 + ag / al); }
    } else {
      const d = series.c[i]! - series.c[i - 1]!;
      gain = (gain * (period - 1) + (d > 0 ? d : 0)) / period;
      loss = (loss * (period - 1) + (d < 0 ? -d : 0)) / period;
      result[i] = loss === 0 ? 100 : 100 - 100 / (1 + gain / loss);
    }
  }
  return result;
}

export function atr(series: BarSeries, period: number): Float64Array {
  const result = new Float64Array(series.length);
  let trSum = 0;
  for (let i = 0; i < series.length; i++) {
    if (i === 0) { result[i] = NaN; continue; }
    const hl = series.h[i]! - series.l[i]!;
    const hc = Math.abs(series.h[i]! - series.c[i - 1]!);
    const lc = Math.abs(series.l[i]! - series.c[i - 1]!);
    const tr = Math.max(hl, hc, lc);
    if (i < period - 1) { trSum += tr; result[i] = NaN; }
    else if (i === period - 1) { trSum += tr; result[i] = trSum / period; }
    else result[i] = (result[i - 1]! * (period - 1) + tr) / period;
  }
  return result;
}

export function wma(series: BarSeries, period: number): Float64Array {
  const result = new Float64Array(series.length);
  const weightSum = period * (period + 1) / 2;
  for (let i = 0; i < series.length; i++) {
    if (i < period - 1) { result[i] = NaN; continue; }
    let sum = 0;
    for (let j = 0; j < period; j++) sum += series.c[i - j]! * (period - j);
    result[i] = sum / weightSum;
  }
  return result;
}

export function macd(series: BarSeries, fast = 12, slow = 26, signal = 9): { macd: Float64Array; signal: Float64Array; histogram: Float64Array } {
  const efast = ema(series, fast);
  const eslow = ema(series, slow);
  const m = new Float64Array(series.length);
  for (let i = 0; i < series.length; i++) m[i] = efast[i]! - eslow[i]!;
  const msignal = ema({ ...series, c: m }, signal);
  const histogram = new Float64Array(series.length);
  for (let i = 0; i < series.length; i++) histogram[i] = m[i]! - msignal[i]!;
  return { macd: m, signal: msignal, histogram };
}

export function bbands(series: BarSeries, period = 20, stdDev = 2): { upper: Float64Array; middle: Float64Array; lower: Float64Array } {
  const middle = sma(series, period);
  const upper = new Float64Array(series.length);
  const lower = new Float64Array(series.length);
  for (let i = 0; i < series.length; i++) {
    if (i < period - 1) { upper[i] = NaN; lower[i] = NaN; continue; }
    let sumSq = 0;
    for (let j = i - period + 1; j <= i; j++) sumSq += (series.c[j]! - middle[i]!) ** 2;
    const std = Math.sqrt(sumSq / period);
    upper[i] = middle[i]! + stdDev * std;
    lower[i] = middle[i]! - stdDev * std;
  }
  return { upper, middle, lower };
}

export function donchian(series: BarSeries, period: number): { upper: Float64Array; middle: Float64Array; lower: Float64Array } {
  const upper = new Float64Array(series.length);
  const lower = new Float64Array(series.length);
  for (let i = 0; i < series.length; i++) {
    if (i < period - 1) { upper[i] = NaN; lower[i] = NaN; continue; }
    let hi = -Infinity, lo = Infinity;
    for (let j = i - period + 1; j <= i; j++) { hi = Math.max(hi, series.h[j]!); lo = Math.min(lo, series.l[j]!); }
    upper[i] = hi; lower[i] = lo;
  }
  const middle = new Float64Array(series.length);
  for (let i = 0; i < series.length; i++) middle[i] = upper[i]! !== undefined && !isNaN(upper[i]!) ? (upper[i]! + lower[i]!) / 2 : NaN;
  return { upper, middle, lower };
}

export function stoch(series: BarSeries, kPeriod = 14, dPeriod = 3): { k: Float64Array; d: Float64Array } {
  const rawK = new Float64Array(series.length);
  for (let i = 0; i < series.length; i++) {
    if (i < kPeriod - 1) { rawK[i] = NaN; continue; }
    let hi = -Infinity, lo = Infinity;
    for (let j = i - kPeriod + 1; j <= i; j++) { hi = Math.max(hi, series.h[j]!); lo = Math.min(lo, series.l[j]!); }
    rawK[i] = hi === lo ? 50 : ((series.c[i]! - lo) / (hi - lo)) * 100;
  }
  const d = sma({ ...series, c: rawK }, dPeriod);
  return { k: rawK, d };
}