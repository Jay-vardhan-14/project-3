import { describe, it, expect } from 'vitest';
import { pct, num } from './chart';

describe('formatting helpers', () => {
  it('formats fractions as percentages', () => {
    expect(pct(0.8496)).toBe('85.0%');
    expect(pct(0)).toBe('0.0%');
  });

  it('renders a dash for null/undefined', () => {
    expect(pct(null)).toBe('—');
    expect(num(undefined)).toBe('—');
  });

  it('formats numbers with the requested precision', () => {
    expect(num(1234)).toBe('1,234');
    expect(num(2.375, 1)).toBe('2.4');
  });
});
