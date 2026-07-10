import type { AlertRow } from '../../api/client';

// Full-width banner shown only when unresolved alerts exist.
export function AlertBanner({ alerts }: { alerts: AlertRow[] }) {
  const active = alerts.filter((a) => !a.is_resolved);
  if (active.length === 0) return null;

  const critical = active.find((a) => a.severity === 'critical');
  const lead = critical ?? active[0];
  const tone = critical
    ? 'border-red-200 bg-red-50 text-red-800'
    : 'border-amber-200 bg-amber-50 text-amber-800';

  return (
    <div className={`mb-5 rounded-md border p-3 text-sm ${tone}`}>
      <span className="font-medium">{lead.message}</span>
      {active.length > 1 && <span className="ml-1 opacity-80">(+{active.length - 1} more)</span>}
      <span className="ml-2 opacity-70">Last checked: {new Date(lead.created_at).toLocaleString()}</span>
    </div>
  );
}
