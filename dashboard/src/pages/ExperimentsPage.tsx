import { useCallback } from 'react';
import { client } from '../api/client';
import { usePoll } from '../hooks/usePoll';
import { AppLayout } from '../components/layout/AppLayout';
import { RunsTable } from '../components/experiments/RunsTable';
import { Panel, Skeleton, EmptyState } from '../components/ui/primitives';

export function ExperimentsPage() {
  const experiments = usePoll(useCallback(() => client.experiments(), []));

  return (
    <AppLayout title="Experiments">
      <Panel title="MLflow runs">
        {experiments.loading ? (
          <Skeleton className="h-40" />
        ) : experiments.error ? (
          <EmptyState message={`Could not load experiments: ${experiments.error}`} />
        ) : (experiments.data ?? []).length === 0 ? (
          <EmptyState message="No MLflow runs found." />
        ) : (
          <RunsTable runs={experiments.data ?? []} />
        )}
      </Panel>
    </AppLayout>
  );
}
