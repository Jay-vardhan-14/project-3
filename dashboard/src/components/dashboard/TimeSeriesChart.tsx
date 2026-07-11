import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { AXIS_TICK, CHART } from '../../lib/chart';

interface Props<T extends object> {
  data: T[];
  xKey: keyof T & string;
  yKey: keyof T & string;
  kind?: 'area' | 'line';
  threshold?: number;
  yDomain?: [number, number];
  xFormatter?: (value: string) => string;
  height?: number;
}

const tooltipStyle = {
  fontSize: 12,
  borderRadius: 6,
  border: '1px solid #e4e4e7',
  boxShadow: 'none',
  color: '#18181b',
} as const;

export function TimeSeriesChart<T extends object>({
  data,
  xKey,
  yKey,
  kind = 'area',
  threshold,
  yDomain,
  xFormatter,
  height = 220,
}: Props<T>) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      {kind === 'area' ? (
        <AreaChart data={data} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
          <CartesianGrid stroke={CHART.grid} vertical={false} />
          <XAxis dataKey={xKey} tick={AXIS_TICK} tickFormatter={xFormatter} tickLine={false} axisLine={{ stroke: CHART.grid }} />
          <YAxis tick={AXIS_TICK} tickLine={false} axisLine={false} width={44} allowDecimals={false} />
          <Tooltip contentStyle={tooltipStyle} labelFormatter={xFormatter} />
          <Area type="monotone" dataKey={yKey} stroke={CHART.data} strokeWidth={1.5} fill={CHART.data} fillOpacity={0.1} isAnimationActive={false} />
        </AreaChart>
      ) : (
        <LineChart data={data} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
          <CartesianGrid stroke={CHART.grid} vertical={false} />
          <XAxis dataKey={xKey} tick={AXIS_TICK} tickFormatter={xFormatter} tickLine={false} axisLine={{ stroke: CHART.grid }} />
          <YAxis tick={AXIS_TICK} tickLine={false} axisLine={false} width={44} domain={yDomain} />
          <Tooltip contentStyle={tooltipStyle} labelFormatter={xFormatter} />
          {threshold != null && (
            <ReferenceLine y={threshold} stroke={CHART.threshold} strokeDasharray="4 3" strokeWidth={1} />
          )}
          <Line type="monotone" dataKey={yKey} stroke={CHART.data} strokeWidth={1.5} dot={{ r: 2, fill: CHART.data }} isAnimationActive={false} />
        </LineChart>
      )}
    </ResponsiveContainer>
  );
}
