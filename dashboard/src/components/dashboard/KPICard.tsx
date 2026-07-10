import type { ReactNode } from 'react';
import { Card, StatusDot } from '../ui/primitives';

export function KPICard({
  label,
  value,
  suffix,
  status,
}: {
  label: string;
  value: ReactNode;
  suffix?: string;
  status?: string;
}) {
  return (
    <Card className="p-4">
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-medium text-zinc-900">{value}</span>
        {suffix && <span className="text-sm text-zinc-500">{suffix}</span>}
      </div>
      <div className="mt-1 flex items-center gap-1.5 text-xs text-zinc-500">
        {status && <StatusDot status={status} />}
        {label}
      </div>
    </Card>
  );
}
