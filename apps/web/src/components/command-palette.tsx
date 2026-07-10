import { useState, useEffect, useRef, useCallback } from 'react';
import { useSymbolSearch } from '../lib/hooks/useSymbolSearch';
import { useContextStore } from '../state/context-store';
import { widgetRegistry } from '../widgets/registry';

interface Props {
  onClose: () => void;
  onAddWidget: (widgetId: string) => void;
}

export default function CommandPalette({ onClose, onAddWidget }: Props) {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'search' | 'widgets'>('search');
  const inputRef = useRef<HTMLInputElement>(null);
  const { data: results } = useSymbolSearch(mode === 'search' ? query : '');
  const setChannel = useContextStore((s) => s.setChannel);

  useEffect(() => { inputRef.current?.focus(); }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const selectSymbol = useCallback((symbol: string) => {
    setChannel('A', { symbol });
    onClose();
  }, [setChannel, onClose]);

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'flex-start', justifyContent: 'center', paddingTop: 80 }}>
      <div style={{ background: 'var(--bg-surface-1)', borderRadius: 'var(--radius-m)', width: 480, maxHeight: 400, overflow: 'hidden', boxShadow: '0 8px 32px rgba(0,0,0,0.4)', display: 'flex', flexDirection: 'column' }}>
        <input ref={inputRef} value={query} onChange={(e) => setQuery(e.target.value)} placeholder={mode === 'search' ? 'Search symbols…' : 'Filter widgets…'} style={{ width: '100%', padding: '12px 16px', border: 'none', borderBottom: '1px solid var(--border-hairline)', background: 'transparent', color: 'var(--text-primary)', fontSize: 14, outline: 'none' }} />

        <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--border-hairline)' }}>
          <button onClick={() => { setMode('search'); setQuery(''); }} style={{ flex: 1, padding: '6px', border: 'none', background: mode === 'search' ? 'var(--accent-soft)' : 'transparent', color: mode === 'search' ? 'var(--accent)' : 'var(--text-muted)', cursor: 'pointer', fontSize: 12, fontWeight: 500 }}>Search</button>
          <button onClick={() => { setMode('widgets'); setQuery(''); }} style={{ flex: 1, padding: '6px', border: 'none', background: mode === 'widgets' ? 'var(--accent-soft)' : 'transparent', color: mode === 'widgets' ? 'var(--accent)' : 'var(--text-muted)', cursor: 'pointer', fontSize: 12, fontWeight: 500 }}>Add Widget</button>
        </div>

        <div style={{ flex: 1, overflow: 'auto', padding: 4 }}>
          {mode === 'search' && results?.map((r) => (
            <button key={r.symbol} onClick={() => selectSymbol(r.symbol)}
              style={{ display: 'block', width: '100%', textAlign: 'left', padding: '8px 12px', border: 'none', background: 'transparent', color: 'var(--text-primary)', cursor: 'pointer', borderRadius: 'var(--radius-s)', fontSize: 13 }}>
              <strong>{r.symbol}</strong>
              <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>{r.name}</span>
              <span style={{ color: 'var(--text-muted)', marginLeft: 8, fontSize: 11 }}>{r.exchange}</span>
            </button>
          ))}
          {mode === 'widgets' && widgetRegistry.getAll().filter(m => !query || m.name.toLowerCase().includes(query.toLowerCase())).map((m) => (
            <button key={m.id} onClick={() => { onAddWidget(m.id); onClose(); }}
              style={{ display: 'block', width: '100%', textAlign: 'left', padding: '8px 12px', border: 'none', background: 'transparent', color: 'var(--text-primary)', cursor: 'pointer', borderRadius: 'var(--radius-s)', fontSize: 13 }}>
              <strong>{m.name}</strong>
              <span style={{ color: 'var(--text-muted)', marginLeft: 8, fontSize: 12 }}>{m.description}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}