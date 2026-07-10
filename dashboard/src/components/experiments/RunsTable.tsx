import { useMemo, useState } from 'react';
import type { ExperimentRow } from '../../api/client';
import { pct, num } from '../../lib/chart';

interface Row {
  id: string;
  name: string;
  model: string;
  f1: number;
  accuracy: number;
  precision: number;
  recall: number;
  latency: number;
  duration: number;
  date: number;
}

type Key = keyof Omit<Row, 'id' | 'name' | 'model'>;

const COLUMNS: Array<{ key: Key | 'name' | 'model'; label: string; kind: 'text' | 'pct' | 'num' | 'date' }> = [
  { key: 'name', label: 'Run', kind: 'text' },
  { key: 'model', label: 'Model', kind: 'text' },
  { key: 'f1', label: 'F1', kind: 'pct' },
  { key: 'accuracy', label: 'Accuracy', kind: 'pct' },
  { key: 'precision', label: 'Precision', kind: 'pct' },
  { key: 'recall', label: 'Recall', kind: 'pct' },
  { key: 'latency', label: 'Latency (ms)', kind: 'num' },
  { key: 'duration', label: 'Duration (s)', kind: 'num' },
  { key: 'date', label: 'Date', kind: 'date' },
];

function toRow(r: ExperimentRow): Row {
  const m = r.metrics ?? {};
  return {
    id: r.run_id,
    name: r.run_name,
    model: r.experiment,
    f1: m.f1_macro ?? 0,
    accuracy: m.accuracy ?? 0,
    precision: m.precision_macro ?? 0,
    recall: m.recall_macro ?? 0,
    latency: m.inference_latency_ms ?? 0,
    duration: m.training_time ?? 0,
    date: r.start_time,
  };
}

export function RunsTable({ runs }: { runs: ExperimentRow[] }) {
  const [sortKey, setSortKey] = useState<Key | 'name' | 'model'>('f1');
  const [dir, setDir] = useState<'asc' | 'desc'>('desc');

  const rows = useMemo(() => {
    const mapped = runs.map(toRow);
    mapped.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      const cmp = typeof av === 'number' && typeof bv === 'number' ? av - bv : String(av).localeCompare(String(bv));
      return dir === 'asc' ? cmp : -cmp;
    });
    return mapped;
  }, [runs, sortKey, dir]);

  const bestF1 = useMemo(() => Math.max(0, ...rows.map((r) => r.f1)), [rows]);

  const onSort = (key: Key | 'name' | 'model') => {
    if (key === sortKey) setDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else {
      setSortKey(key);
      setDir('desc');
    }
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[13px]">
        <thead>
          <tr className="text-left text-xs text-zinc-500">
            {COLUMNS.map((c) => (
              <th key={c.key} className="pb-2 font-medium">
                <button
                  className="inline-flex items-center gap-1 hover:text-zinc-700"
                  onClick={() => onSort(c.key)}
                >
                  {c.label}
                  {sortKey === c.key && <span className="text-zinc-400">{dir === 'asc' ? '↑' : '↓'}</span>}
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className={`border-t border-zinc-100 ${r.f1 === bestF1 && bestF1 > 0 ? 'bg-blue-50' : ''}`}>
              <td className="py-2 text-zinc-700">{r.name}</td>
              <td className="py-2 text-zinc-500">{r.model}</td>
              <td className="py-2 text-zinc-900">{pct(r.f1)}</td>
              <td className="py-2 text-zinc-700">{pct(r.accuracy)}</td>
              <td className="py-2 text-zinc-700">{pct(r.precision)}</td>
              <td className="py-2 text-zinc-700">{pct(r.recall)}</td>
              <td className="py-2 text-zinc-700">{num(r.latency, 1)}</td>
              <td className="py-2 text-zinc-700">{num(r.duration, 1)}</td>
              <td className="py-2 text-zinc-500">{new Date(r.date).toLocaleDateString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
