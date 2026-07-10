import type { ReactNode } from 'react';

// Card: 1px zinc border, no shadow, small radius — the base surface everywhere.
export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-md border border-zinc-200 bg-white ${className}`}>{children}</div>;
}

// Panel: a Card with a compact header row.
export function Panel({ title, action, children }: { title: string; action?: ReactNode; children: ReactNode }) {
  return (
    <Card>
      <div className="flex items-center justify-between border-b border-zinc-100 px-4 py-2.5">
        <h2 className="text-[13px] font-medium text-zinc-900">{title}</h2>
        {action}
      </div>
      <div className="p-4">{children}</div>
    </Card>
  );
}

const DOT: Record<string, string> = {
  healthy: 'bg-emerald-500',
  success: 'bg-emerald-500',
  ok: 'bg-emerald-500',
  warning: 'bg-amber-500',
  info: 'bg-blue-500',
  critical: 'bg-red-500',
  failed: 'bg-red-500',
  running: 'bg-zinc-400',
};

export function StatusDot({ status }: { status: string }) {
  return <span className={`inline-block h-2 w-2 shrink-0 rounded-full ${DOT[status] ?? 'bg-zinc-300'}`} />;
}

export function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-zinc-200 ${className}`} />;
}

export function EmptyState({ message }: { message: string }) {
  return <div className="py-8 text-center text-[13px] text-zinc-400">{message}</div>;
}

// Muted zinc chip for labels like model stage. Color only used for alert severity elsewhere.
export function Badge({ children, tone = 'zinc' }: { children: ReactNode; tone?: 'zinc' | 'emerald' | 'amber' | 'red' }) {
  const tones: Record<string, string> = {
    zinc: 'text-zinc-600 bg-zinc-100 border-zinc-200',
    emerald: 'text-emerald-700 bg-emerald-50 border-emerald-200',
    amber: 'text-amber-700 bg-amber-50 border-amber-200',
    red: 'text-red-700 bg-red-50 border-red-200',
  };
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[11px] font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}
