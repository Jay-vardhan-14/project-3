// Monochrome chart palette — color is reserved for alert states only.
export const CHART = {
  data: '#18181b', // zinc-900 bars/lines
  grid: '#e4e4e7', // zinc-200 gridlines
  axis: '#71717a', // zinc-500 axis labels
  threshold: '#dc2626', // red-600 drift threshold line
  seriesStrong: '#3f3f46', // zinc-700 (e.g. positive)
  seriesWeak: '#d4d4d8', // zinc-300 (e.g. negative)
} as const;

export const AXIS_TICK = { fill: CHART.axis, fontSize: 12 } as const;

export function pct(value: number | null | undefined): string {
  if (value == null) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

export function num(value: number | null | undefined, digits = 0): string {
  if (value == null) return '—';
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}
