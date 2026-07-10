import type { BarSeries } from '@tradeboard/core';

export function sma(series: BarSeries, period: number): Float64Array {
  const result = new Float64Array(series.length);
  let sum = 0;
  for (let i = 0; i < series.length; i++) {
    sum += series.c[i]!;
    if (i >= period - 1) {
      if (i >= period) sum -= series.c[i - period]!;
      result[i] = sum / period;
    } else {
      result[i] = NaN;
    }
  }
  return result;
}

export function ema(series: BarSeries, period: number): Float64Array {
  const result = new Float64Array(series.length);
  const k = 2 / (period + 1);
  let prev = NaN;
  for (let i = 0; i < series.length; i++) {
    if (i < period - 1) {
      result[i] = NaN;
    } else if (i === period - 1) {
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
    if (i < period) {
      result[i] = NaN;
      if (i > 0) {
        const d = series.c[i]! - series.c[i - 1]!;
        if (d > 0) gain += d; else loss -= d;
      }
      if (i === period - 1) {
        const avgGain = gain / period;
        const avgLoss = loss / period;
        result[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
      }
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
    if (i === 0) {
      result[i] = NaN;
      continue;
    }
    const hl = series.h[i]! - series.l[i]!;
    const hc = Math.abs(series.h[i]! - series.c[i - 1]!);
    const lc = Math.abs(series.l[i]! - series.c[i - 1]!);
    const tr = Math.max(hl, hc, lc);
    if (i < period) {
      trSum += tr;
      result[i] = NaN;
      if (i === period - 1) result[i] = trSum / period;
    } else {
      result[i] = (result[i - 1]! * (period - 1) + tr) / period;
    }
  }
  return result;
}