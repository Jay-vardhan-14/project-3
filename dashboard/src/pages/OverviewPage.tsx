import { useCallback } from 'react';
import { client } from '../api/client';
import { usePoll } from '../hooks/usePoll';
import { AppLayout } from '../components/layout/AppLayout';
import { AlertBanner } from '../components/dashboard/AlertBanner';
import { KPICard } from '../components/dashboard/KPICard';
import { TimeSeriesChart } from '../components/dashboard/TimeSeriesChart';
import { Panel, Skeleton, EmptyState, StatusDot } from '../components/ui/primitives';
import { pct, num } from '../lib/chart';

const shortDate = (value: string) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
const driftLabel = { healthy: 'Healthy', warning: 'Warning', critical: 'Alert' } as const;

export function OverviewPage() {
  const summary = usePoll(useCallback(() => client.summary(), []));
  const volume = usePoll(useCallback(() => client.volume(14), []));
  const alerts = usePoll(useCallback(() => client.alerts(), []));

  const s = summary.data;
  const recentAlerts = (alerts.data ?? []).slice(0, 5);

  return (
    <AppLayout title="Overview">
      {alerts.data && <AlertBanner alerts={alerts.data} />}

      <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-4">
        {summary.loading || !s ? (
          Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-[76px]" />)
        ) : (
          <>
            <KPICard label="Predictions today" value={num(s.predictions_today)} />
            <KPICard label="Avg latency" value={num(s.avg_latency_ms, 1)} suffix="ms" />
            <KPICard label="Model F1" value={pct(s.model_f1)} />
            <KPICard
              label={driftLabel[s.drift_status]}
              value={<span className="text-base">{driftLabel[s.drift_status]}</span>}
              status={s.drift_status}
            />
          </>
        )}
      </div>

      <div className="mb-5">
        <Panel title="Prediction volume">
          {volume.loading ? (
            <Skeleton className="h-[220px]" />
          ) : (volume.data ?? []).length === 0 ? (
            <EmptyState message="No predictions recorded yet." />
          ) : (
            <TimeSeriesChart data={volume.data ?? []} xKey="bucket" yKey="count" kind="area" xFormatter={shortDate} />
          )}
        </Panel>
      </div>

      <Panel title="Recent alerts">
        {alerts.loading ? (
          <Skeleton className="h-24" />
        ) : recentAlerts.length === 0 ? (
          <EmptyState message="No alerts. System healthy." />
        ) : (
          <table className="w-full text-[13px]">
            <thead>
              <tr className="text-left text-xs text-zinc-500">
                <th className="pb-2 font-medium">Type</th>
                <th className="pb-2 font-medium">Severity</th>
                <th className="pb-2 font-medium">Message</th>
                <th className="pb-2 text-right font-medium">Time</th>
              </tr>
            </thead>
            <tbody>
              {recentAlerts.map((a) => (
                <tr key={a.id} className="border-t border-zinc-100">
                  <td className="py-2 text-zinc-700">{a.alert_type}</td>
                  <td className="py-2">
                    <span className="flex items-center gap-1.5 text-zinc-600">
                      <StatusDot status={a.severity} />
                      {a.severity}
                    </span>
                  </td>
                  <td className="py-2 text-zinc-700">{a.message}</td>
                  <td className="py-2 text-right text-zinc-500">{new Date(a.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </AppLayout>
  );
}
