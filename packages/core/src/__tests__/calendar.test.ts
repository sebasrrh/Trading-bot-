import { describe, it, expect } from 'vitest';
import { isTradingDay, countTradingDays, annualizationFactor } from '../index';

describe('calendar', () => {
  it('weekends are not trading days', () => {
    const sat = new Date('2025-01-04').getTime();
    const sun = new Date('2025-01-05').getTime();
    expect(isTradingDay(sat)).toBe(false);
    expect(isTradingDay(sun)).toBe(false);
  });

  it('weekdays are trading days', () => {
    const wed = new Date('2025-01-08').getTime();
    expect(isTradingDay(wed)).toBe(true);
  });

  it('countTradingDays returns expected count', () => {
    const from = new Date('2025-01-06').getTime(); // Mon
    const to = new Date('2025-01-10').getTime();   // Fri
    expect(countTradingDays(from, to)).toBe(5);
  });

  it('annualizationFactor for daily is 252', () => {
    expect(annualizationFactor('1D')).toBe(252);
  });
});