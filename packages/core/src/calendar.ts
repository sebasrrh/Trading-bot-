export const NYSE_HOLIDAYS_2025 = new Set([
  new Date('2025-01-01').getTime(),
  new Date('2025-01-20').getTime(),
  new Date('2025-02-17').getTime(),
  new Date('2025-04-18').getTime(),
  new Date('2025-05-26').getTime(),
  new Date('2025-06-19').getTime(),
  new Date('2025-07-04').getTime(),
  new Date('2025-09-01').getTime(),
  new Date('2025-11-27').getTime(),
  new Date('2025-12-25').getTime(),
]);

const MS_PER_DAY = 86_400_000;

export function isTradingDay(epochMs: number): boolean {
  const d = new Date(epochMs);
  const day = d.getUTCDay();
  if (day === 0 || day === 6) return false;
  const dateStart = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
  return !NYSE_HOLIDAYS_2025.has(dateStart.getTime());
}

export function nextTradingDay(epochMs: number): number {
  let t = epochMs + MS_PER_DAY;
  while (!isTradingDay(t)) t += MS_PER_DAY;
  return t;
}

export function prevTradingDay(epochMs: number): number {
  let t = epochMs - MS_PER_DAY;
  while (!isTradingDay(t)) t -= MS_PER_DAY;
  return t;
}

export function countTradingDays(from: number, to: number): number {
  let count = 0;
  let t = from;
  while (t <= to) {
    if (isTradingDay(t)) count++;
    t += MS_PER_DAY;
  }
  return count;
}

export function annualizationFactor(timeframe: string): number {
  switch (timeframe) {
    case '1D': return 252;
    case '1W': return 52;
    case '1h': return 252 * 6.5;
    case '15m': return 252 * 6.5 * 4;
    case '5m': return 252 * 6.5 * 12;
    case '1m': return 252 * 6.5 * 60;
    default: return 252;
  }
}