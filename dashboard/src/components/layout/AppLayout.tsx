import type { ReactNode } from 'react';
import { Sidebar } from './Sidebar';

export function AppLayout({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="flex min-h-screen bg-white text-zinc-900">
      <Sidebar />
      <main className="min-w-0 flex-1">
        <header className="border-b border-zinc-200 px-6 py-3">
          <h1 className="text-[15px] font-medium text-zinc-900">{title}</h1>
        </header>
        <div className="mx-auto max-w-[1100px] px-6 py-5">{children}</div>
      </main>
    </div>
  );
}
