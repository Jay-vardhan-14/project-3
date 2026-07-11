import { useCallback } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { client } from '../api/client';
import { usePoll } from '../hooks/usePoll';
import { AppLayout } from '../components/layout/AppLayout';
import { Card, Panel, Skeleton, EmptyState } from '../components/ui/primitives';
import { AXIS_TICK, CHART, num, pct } from '../lib/chart';

export function PredictionsPage() {
  const distribution = usePoll(useCallback(() => client.distribution(30), []));
  const confidence = usePoll(useCallback(() => client.confidence(30), []));
  const latency = usePoll(useCallback(() => client.latency(30), []));
  const recent = usePoll(useCallback(() => client.recentPredictions(20), []));

  const dist = distribution.data ?? [];
  const total = dist.reduce((sum, d) => sum + Number(d.count), 0);
  const positive = Number(dist.find((d) => d.predicted_sentiment === 'positive')?.count ?? 0);
  const negative = Number(dist.find((d) => d.predicted_sentiment === 'negative')?.count ?? 0);
  const confData = (confidence.data ?? []).map((c) => ({
    label: `${Number(c.bucket).toFixed(1)}–${(Number(c.bucket) + 0.1).toFixed(1)}`,
    count: Number(c.count),
  }));
  const lat = latency.data;

  return (
    <AppLayout title="Predictions">
      <div className="mb-5 grid grid-cols-1 gap-3 md:grid-cols-2">
        <Panel title="Sentiment distribution">
          {distribution.loading ? (
            <Skeleton className="h-16" />
          ) : total === 0 ? (
            <EmptyState message="No predictions yet." />
          ) : (
            <div>
              <div className="flex h-6 w-full overflow-hidden rounded">
                <div className="bg-zinc-700" style={{ width: `${(positive / total) * 100}%` }} />
                <div className="bg-zinc-300" style={{ width: `${(negative / total) * 100}%` }} />
              </div>
              <div className="mt-2 flex justify-between text-xs text-zinc-500">
                <span>Positive {pct(positive / total)}</span>
                <span>Negative {pct(negative / total)}</span>
              </div>
            </div>
          )}
        </Panel>

        <Panel title="Latency percentiles">
          {latency.loading || !lat ? (
            <Skeleton className="h-16" />
          ) : lat.count === 0 ? (
            <EmptyState message="No predictions yet." />
          ) : (
            <div className="grid grid-cols-3 gap-2">
              {(['p50', 'p95', 'p99'] as const).map((k) => (
                <div key={k}>
                  <div className="text-lg font-medium text-zinc-900">
                    {num(lat[k], 1)}
                    <span className="text-xs text-zinc-500"> ms</span>
                  </div>
                  <div className="text-xs uppercase text-zinc-500">{k}</div>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>

      <div className="mb-5">
        <Panel title="Confidence distribution">
          {confidence.loading ? (
            <Skeleton className="h-[200px]" />
          ) : confData.length === 0 ? (
            <EmptyState message="No predictions yet." />
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={confData} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
                <CartesianGrid stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="label" tick={AXIS_TICK} tickLine={false} axisLine={{ stroke: CHART.grid }} />
                <YAxis tick={AXIS_TICK} tickLine={false} axisLine={false} width={44} allowDecimals={false} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 6, border: '1px solid #e4e4e7', boxShadow: 'none' }} />
                <Bar dataKey="count" fill={CHART.data} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Panel>
      </div>

      <Panel title="Recent predictions">
        {recent.loading ? (
          <Skeleton className="h-24" />
        ) : (recent.data ?? []).length === 0 ? (
          <EmptyState message="No predictions recorded." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="text-left text-xs text-zinc-500">
                  <th className="pb-2 font-medium">Time</th>
                  <th className="pb-2 font-medium">Sentiment</th>
                  <th className="pb-2 font-medium">Confidence</th>
                  <th className="pb-2 font-medium">Latency (ms)</th>
                  <th className="pb-2 font-medium">Model</th>
                </tr>
              </thead>
              <tbody>
                {(recent.data ?? []).map((r, i) => (
                  <tr key={i} className="border-t border-zinc-100">
                    <td className="py-2 text-zinc-500">{new Date(r.created_at).toLocaleString()}</td>
                    <td className="py-2 text-zinc-700">{r.predicted_sentiment}</td>
                    <td className="py-2 text-zinc-900">{pct(Number(r.confidence))}</td>
                    <td className="py-2 text-zinc-700">{r.latency_ms}</td>
                    <td className="py-2 text-zinc-500">v{r.model_version}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </AppLayout>
  );
}
