import { useCallback } from 'react';
import { client } from '../api/client';
import { usePoll } from '../hooks/usePoll';
import { AppLayout } from '../components/layout/AppLayout';
import { Panel, Skeleton, EmptyState, StatusDot } from '../components/ui/primitives';

const duration = (seconds: number | null) => (seconds == null ? '—' : `${Math.round(seconds)}s`);

export function PipelinePage() {
  const runs = usePoll(useCallback(() => client.pipelineRuns(), []));
  const rows = runs.data ?? [];

  return (
    <AppLayout title="Pipeline">
      <Panel title="Pipeline runs">
        {runs.loading ? (
          <Skeleton className="h-24" />
        ) : rows.length === 0 ? (
          <EmptyState message="No pipeline runs recorded yet." />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="text-left text-xs text-zinc-500">
                  <th className="pb-2 font-medium">DAG</th>
                  <th className="pb-2 font-medium">Status</th>
                  <th className="pb-2 font-medium">Started</th>
                  <th className="pb-2 font-medium">Duration</th>
                  <th className="pb-2 font-medium">Run ID</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id} className="border-t border-zinc-100">
                    <td className="py-2 text-zinc-700">{r.dag_id}</td>
                    <td className="py-2">
                      <span className="flex items-center gap-1.5 text-zinc-600">
                        <StatusDot status={r.status} />
                        {r.status}
                      </span>
                    </td>
                    <td className="py-2 text-zinc-500">{new Date(r.started_at).toLocaleString()}</td>
                    <td className="py-2 text-zinc-700">{duration(r.duration_seconds)}</td>
                    <td className="py-2 text-zinc-500">{r.run_id}</td>
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
