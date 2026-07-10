import { NavLink } from 'react-router-dom';

const NAV = [
  { to: '/', label: 'Overview', end: true },
  { to: '/experiments', label: 'Experiments' },
  { to: '/drift', label: 'Drift' },
  { to: '/predictions', label: 'Predictions' },
  { to: '/pipeline', label: 'Pipeline' },
];

export function Sidebar() {
  return (
    <aside className="flex w-[200px] shrink-0 flex-col border-r border-zinc-200 bg-zinc-50">
      <div className="px-4 py-4">
        <span className="text-[15px] font-medium text-zinc-900">SentinelML</span>
        <p className="mt-0.5 text-[11px] text-zinc-400">Model monitoring</p>
      </div>
      <nav className="flex flex-col px-2">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `rounded px-2 py-1.5 text-[13px] ${
                isActive ? 'bg-zinc-100 text-zinc-900' : 'text-zinc-500 hover:text-zinc-700'
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
