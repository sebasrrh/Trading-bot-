import { useState, useEffect } from 'react';
import { QueryClientProvider } from '@tanstack/react-query';
import { LayoutDashboard, FlaskConical, TrendingUp, Wallet, Search } from 'lucide-react';
import { queryClient } from '../lib/query-client';
import { useContextStore, useWorkspaceStore } from '../state';
import { widgetRegistry } from '../widgets/registry';
import DashboardGrid from './dashboard-grid';
import CommandPalette from '../components/command-palette';
import BacktestLab from '../views/backtest-lab';
import PaperTrading from '../views/paper-trading';
import OptimizerView from '../views/optimizer-view';
import LiveView from '../views/live-view';

const navItems = [
  { id: 'dashboard', Icon: LayoutDashboard, label: 'Dashboard' },
  { id: 'lab', Icon: FlaskConical, label: 'Backtest Lab' },
  { id: 'sim', Icon: TrendingUp, label: 'Sim Lab' },
  { id: 'paper', Icon: Wallet, label: 'Paper' },
  { id: 'optimizer', Icon: Search, label: 'Optimizer' },
];

function Content() {
  const activeView = useContextStore((s) => s.activeView);
  const ws = useWorkspaceStore();
  const w = ws.workspaces[ws.active];
  if (activeView === 'lab') return <BacktestLab />;
  if (activeView === 'paper') return <PaperTrading />;
  if (activeView === 'optimizer') return <OptimizerView />;
  if (activeView === 'sim') return <LiveView />;
  if (activeView !== 'dashboard') {
    return <main style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 14 }}>
      Settings
    </main>;
  }
  return <DashboardGrid workspace={w!} />;
}

export default function App() {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const switchWorkspace = useWorkspaceStore((s) => s.switchWorkspace);
  const workspaces = useWorkspaceStore((s) => s.workspaces);
  const active = useWorkspaceStore((s) => s.active);
  const addWidget = useWorkspaceStore((s) => s.addWidget);
  const setView = useContextStore((s) => s.setActiveView);
  const activeView = useContextStore((s) => s.activeView);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); setPaletteOpen((o) => !o); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        <header style={{ height: 48, display: 'flex', alignItems: 'center', padding: '0 var(--space-4)', borderBottom: '1px solid var(--border-hairline)', background: 'var(--bg-surface-1)', gap: 12 }}>
          <span style={{ fontWeight: 700, fontSize: 16, letterSpacing: '-0.02em' }}>Tradeboard</span>
          {activeView === 'dashboard' && (
            <>
              <select value={active} onChange={(e) => switchWorkspace(Number(e.target.value))}
                style={{ border: '1px solid var(--border-hairline)', background: 'var(--bg-surface-2)', color: 'var(--text-primary)', borderRadius: 'var(--radius-s)', padding: '2px 8px', fontSize: 12, cursor: 'pointer' }}>
                {workspaces.map((w, i) => <option key={i} value={i}>{w.name}</option>)}
              </select>
              <button onClick={() => setPaletteOpen(true)}
                style={{ border: '1px solid var(--border-hairline)', background: 'var(--bg-surface-2)', color: 'var(--text-secondary)', borderRadius: 'var(--radius-s)', padding: '2px 10px', fontSize: 12, cursor: 'pointer' }}>
                + Widget
              </button>
            </>
          )}
          <div style={{ flex: 1 }} />
          <button onClick={() => setPaletteOpen(true)}
            style={{ border: 'none', background: 'var(--accent-soft)', color: 'var(--accent)', borderRadius: 'var(--radius-s)', padding: '4px 10px', fontSize: 11, fontWeight: 500, cursor: 'pointer', letterSpacing: '0.02em' }}>
            ⌘K
          </button>
          <div style={{ padding: '4px 10px', borderRadius: 'var(--radius-s)', background: 'var(--accent-soft)', color: 'var(--accent)', fontSize: 11, fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Stooq</div>
        </header>
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          <nav style={{ width: 56, borderRight: '1px solid var(--border-hairline)', background: 'var(--bg-surface-1)', display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 8, gap: 4 }}>
            {navItems.map(item => (
              <button key={item.id} title={item.label}
                onClick={() => setView(item.id as any)}
                style={{ width: 40, height: 40, border: 'none', background: activeView === item.id ? 'var(--accent-soft)' : 'transparent', color: activeView === item.id ? 'var(--accent)' : 'var(--text-secondary)', cursor: 'pointer', borderRadius: 'var(--radius-s)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <item.Icon size={20} strokeWidth={1.75} />
              </button>
            ))}
          </nav>
          <Content />
        </div>
      </div>
      {paletteOpen && <CommandPalette onClose={() => setPaletteOpen(false)} onAddWidget={(wid) => addWidget(wid, {}, widgetRegistry.get(wid)?.defaultSize)} />}
    </QueryClientProvider>
  );
}