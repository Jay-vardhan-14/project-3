import { useCallback } from 'react';
import { client } from '../api/client';
import { usePoll } from '../hooks/usePoll';
import { AppLayout } from '../components/layout/AppLayout';
import { TimeSeriesChart } from '../components/dashboard/TimeSeriesChart';
import { Card, Panel, Skeleton, EmptyState, StatusDot } from '../components/ui/primitives';
import { num } from '../lib/chart';

const DRIFT_THRESHOLD = 0.15;
const shortDate = (value: string) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });

export function DriftPage() {
  const drift = usePoll(useCallback(() => client.drift(30), []));
  const rows = drift.data ?? [];
  const latest = rows.length > 0 ? rows[rows.length - 1] : null;
  const status = latest ? (latest.drift_score > DRIFT_THRESHOLD * 2 ? 'critical' : latest.drift_score > DRIFT_THRESHOLD ? 'warning' : 'healthy') : 'healthy';
  const chartData = rows.map((r) => ({ ...r, drift_score: Number(r.drift_score) }));

  return (
    <AppLayout title="Drift">
      <div className="mb-5 grid grid-cols-1 gap-3 md:grid-cols-3">
        <Card className="p-4">
          <div className="text-xs text-zinc-500">Current drift score</div>
          <div className="mt-1 text-2xl font-medium text-zinc-900">{latest ? num(Number(latest.drift_score), 3) : '—'}</div>
        </Card>
        <Card className="p-4">
          <div className="text-xs text-zinc-500">Status</div>
          <div className="mt-1 flex items-center gap-1.5 text-base text-zinc-900">
            <StatusDot status={status} />
            {status === 'critical' ? 'Critical' : status === 'warning' ? 'Warning' : 'Healthy'}
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-xs text-zinc-500">Last check</div>
          <div className="mt-1 text-base text-zinc-900">{latest ? new Date(latest.report_date).toLocaleDateString() : '—'}</div>
        </Card>
      </div>

      <div className="mb-5">
        <Panel title={`Drift score trend (threshold ${DRIFT_THRESHOLD})`}>
          {drift.loading ? (
            <Skeleton className="h-[220px]" />
          ) : chartData.length === 0 ? (
            <EmptyState message="No drift reports yet. The drift_detection DAG populates this daily." />
          ) : (
            <TimeSeriesChart
              data={chartData}
              xKey="report_date"
              yKey="drift_score"
              kind="line"
              threshold={DRIFT_THRESHOLD}
              yDomain={[0, 1]}
              xFormatter={shortDate}
            />
          )}
        </Panel>
      </div>

      <Panel title="Drift reports">
        {drift.loading ? (
          <Skeleton className="h-24" />
        ) : rows.length === 0 ? (
          <EmptyState message="No drift reports recorded." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="text-left text-xs text-zinc-500">
                  <th className="pb-2 font-medium">Date</th>
                  <th className="pb-2 font-medium">Drift</th>
                  <th className="pb-2 font-medium">Score</th>
                  <th className="pb-2 font-medium">Features drifted</th>
                  <th className="pb-2 font-medium">Prediction drift</th>
                </tr>
              </thead>
              <tbody>
                {[...rows].reverse().map((r, i) => (
                  <tr key={i} className="border-t border-zinc-100">
                    <td className="py-2 text-zinc-700">{new Date(r.report_date).toLocaleDateString()}</td>
                    <td className="py-2">
                      <span className="flex items-center gap-1.5 text-zinc-600">
                        <StatusDot status={r.dataset_drift_detected ? 'critical' : 'healthy'} />
                        {r.dataset_drift_detected ? 'Yes' : 'No'}
                      </span>
                    </td>
                    <td className="py-2 text-zinc-900">{num(Number(r.drift_score), 3)}</td>
                    <td className="py-2 text-zinc-700">
                      {r.features_drifted}/{r.total_features}
                    </td>
                    <td className="py-2 text-zinc-700">{r.prediction_drift_detected ? 'Yes' : 'No'}</td>
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
