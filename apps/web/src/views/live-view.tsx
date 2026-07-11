import { useState, useEffect, useRef } from 'react';
import { useQuotes } from '../lib/hooks/useQuotes';
import { usePaperStore } from '../state/paper-store';
import { useContextStore } from '../state/context-store';

const DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'TSLA'];

export default function LiveView() {
  const ctx = useContextStore();
  const sym = ctx.channels.A.symbol;
  const getEngine = usePaperStore((s) => s.getEngine);
  const bump = usePaperStore((s) => s.bump);
  const refreshKey = usePaperStore((s) => s.refreshKey);
  const eng = getEngine();
  void refreshKey;

  const [autoInterval, setAutoInterval] = useState(5);
  const [auto, setAuto] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Auto-refresh: bump periodically so the view re-reads engine state
  useEffect(() => {
    if (auto) {
      intervalRef.current = setInterval(() => {
        bump();
        setLastUpdate(Date.now());
      }, autoInterval * 1000);
    }
    return () => {
      if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    };
  }, [auto, autoInterval, bump]);

  const symbols = [sym, ...DEFAULT_SYMBOLS.filter((s) => s !== sym)].slice(0, 6);
  const { data: quotesData } = useQuotes(symbols, auto ? autoInterval * 1000 : 10_000);

  const eq = eng.equity();
  const positions = [...eng.account.positions.values()];
  const openOrders = eng.account.openOrders.filter((o) => o.status === 'open');
  const recentFills = eng.account.fills.slice(-20).reverse();

  const quotes = quotesData?.quotes ?? [];

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Top bar */}
      <div style={{ height: 40, borderBottom: '1px solid var(--border-hairline)', display: 'flex', alignItems: 'center', padding: '0 16px', gap: 12, background: 'var(--bg-surface-1)' }}>
        <h2 style={{ fontSize: 14, fontWeight: 700, margin: 0 }}>Live Market Monitor</h2>
        <div style={{ flex: 1 }} />
        <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-secondary)', cursor: 'pointer' }}>
          <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
          Auto
        </label>
        {auto && (
          <select value={autoInterval} onChange={(e) => setAutoInterval(Number(e.target.value))}
            style={{ border: '1px solid var(--border-hairline)', borderRadius: 'var(--radius-s)', background: 'var(--bg-surface-2)', color: 'var(--text-primary)', fontSize: 11, padding: '2px 4px' }}>
            <option value={2}>2s</option>
            <option value={5}>5s</option>
            <option value={10}>10s</option>
            <option value={30}>30s</option>
            <option value={60}>60s</option>
          </select>
        )}
        <button onClick={() => { bump(); setLastUpdate(Date.now()); }}
          style={{ border: '1px solid var(--border-hairline)', borderRadius: 'var(--radius-s)', background: 'var(--bg-surface-2)', color: 'var(--text-secondary)', fontSize: 11, padding: '3px 8px', cursor: 'pointer' }}>
          Refresh
        </button>
        <span style={{ fontSize: 10, color: 'var(--text-muted)', minWidth: 70, textAlign: 'right' }}>
          {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : '\u2014'}
        </span>
      </div>

      {/* Body */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left panel: Account + Positions */}
        <div style={{ width: 320, borderRight: '1px solid var(--border-hairline)', padding: 12, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Account card */}
          <div style={{ background: 'var(--bg-surface-1)', borderRadius: 'var(--radius-s)', padding: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Account</div>
            <Row label="Cash" val={`$${eq.cash.toLocaleString('en-US', { minimumFractionDigits: 2 })}`} />
            <Row label="Equity" val={`$${eq.total.toLocaleString('en-US', { minimumFractionDigits: 2 })}`} />
            <Row label="Return" val={`${(eq.ret * 100).toFixed(2)}%`} color={eq.ret >= 0 ? 'var(--accent)' : 'var(--loss)'} />
          </div>

          {/* Positions */}
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
              Positions ({positions.length})
            </div>
            {positions.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 11, padding: 4 }}>No open positions</div>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead><tr style={{ borderBottom: '2px solid var(--border-hairline)' }}>
                  <th style={{ textAlign: 'left', padding: '2px 4px' }}>Sym</th>
                  <th style={{ textAlign: 'right', padding: '2px 4px' }}>Qty</th>
                  <th style={{ textAlign: 'right', padding: '2px 4px' }}>Avg</th>
                  <th style={{ textAlign: 'right', padding: '2px 4px' }}>P&L</th>
                </tr></thead>
                <tbody>{positions.map((p) => {
                  const q = quotes.find((q) => q.symbol === p.symbol);
                  const mkt = q?.price ?? p.avgPrice;
                  const livePnl = p.qty * (mkt - p.avgPrice);
                  return (
                    <tr key={p.symbol} style={{ borderBottom: '1px solid var(--border-hairline)' }}>
                      <td style={{ padding: '3px 4px', fontWeight: 600 }}>{p.symbol}</td>
                      <td style={{ padding: '3px 4px', textAlign: 'right' }}>{p.qty}</td>
                      <td style={{ padding: '3px 4px', textAlign: 'right', color: 'var(--text-muted)' }}>${p.avgPrice.toFixed(2)}</td>
                      <td style={{ padding: '3px 4px', textAlign: 'right', fontWeight: 600, color: livePnl >= 0 ? 'var(--accent)' : 'var(--loss)' }}>
                        ${livePnl.toFixed(2)}
                      </td>
                    </tr>
                  );
                })}</tbody>
              </table>
            )}
          </div>

          {/* Open Orders */}
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
              Orders ({openOrders.length})
            </div>
            {openOrders.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 11, padding: 4 }}>No open orders</div>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead><tr style={{ borderBottom: '2px solid var(--border-hairline)' }}>
                  <th style={{ textAlign: 'left', padding: '2px 4px' }}>Sym</th>
                  <th style={{ textAlign: 'left', padding: '2px 4px' }}>Side</th>
                  <th style={{ textAlign: 'right', padding: '2px 4px' }}>Qty</th>
                  <th style={{ textAlign: 'right', padding: '2px 4px' }}>Type</th>
                </tr></thead>
                <tbody>{openOrders.map((o) => (
                  <tr key={o.id} style={{ borderBottom: '1px solid var(--border-hairline)' }}>
                    <td style={{ padding: '3px 4px', fontWeight: 600 }}>{o.symbol}</td>
                    <td style={{ padding: '3px 4px', color: o.side === 'buy' ? 'var(--accent)' : 'var(--loss)' }}>{o.side}</td>
                    <td style={{ padding: '3px 4px', textAlign: 'right' }}>{o.qty}</td>
                    <td style={{ padding: '3px 4px', textAlign: 'right', color: 'var(--text-muted)' }}>
                      {o.type}{o.limitPrice ? ` @$${o.limitPrice}` : o.stopPrice ? ` stop $${o.stopPrice}` : ''}
                    </td>
                  </tr>
                ))}</tbody>
              </table>
            )}
          </div>
        </div>

        {/* Right panel: Quotes + Fills */}
        <div style={{ flex: 1, padding: 12, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Quote board */}
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>
              Quotes
            </div>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              {symbols.map((s) => {
                const q = quotes.find((x) => x.symbol === s);
                const isUp = (q?.changePct ?? 0) >= 0;
                return (
                  <div key={s} style={{
                    background: 'var(--bg-surface-1)', borderRadius: 'var(--radius-s)', padding: 12,
                    minWidth: 140, display: 'flex', flexDirection: 'column', gap: 4,
                  }}>
                    <span style={{ fontSize: 13, fontWeight: 700 }}>{s}</span>
                    <span style={{ fontSize: 20, fontWeight: 700, fontVariantNumeric: 'tabular-nums', color: isUp ? 'var(--accent)' : 'var(--loss)' }}>
                      {q ? `$${q.price.toFixed(2)}` : '\u2026'}
                    </span>
                    {q && (
                      <span style={{ fontSize: 12, color: isUp ? 'var(--accent)' : 'var(--loss)', fontWeight: 500 }}>
                        {isUp ? '\u25B2' : '\u25BC'} {(q.changePct).toFixed(2)}%
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Recent fills */}
          <div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
              Recent Activity ({recentFills.length})
            </div>
            {recentFills.length === 0 ? (
              <div style={{ color: 'var(--text-muted)', fontSize: 12, padding: 8 }}>No fills yet. Process bars in Paper Trading.</div>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead><tr style={{ borderBottom: '2px solid var(--border-hairline)' }}>
                  <th style={{ textAlign: 'left', padding: '4px 8px' }}>Time</th>
                  <th style={{ textAlign: 'left', padding: '4px 8px' }}>Sym</th>
                  <th style={{ textAlign: 'left', padding: '4px 8px' }}>Side</th>
                  <th style={{ textAlign: 'right', padding: '4px 8px' }}>Qty</th>
                  <th style={{ textAlign: 'right', padding: '4px 8px' }}>Price</th>
                  <th style={{ textAlign: 'right', padding: '4px 8px' }}>P&L</th>
                </tr></thead>
                <tbody>{recentFills.map((f) => (
                  <tr key={f.id} style={{ borderBottom: '1px solid var(--border-hairline)' }}>
                    <td style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{new Date(f.t).toLocaleTimeString()}</td>
                    <td style={{ padding: '4px 8px', fontWeight: 600 }}>{f.symbol}</td>
                    <td style={{ padding: '4px 8px', color: f.side === 'buy' ? 'var(--accent)' : 'var(--loss)' }}>{f.side}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>{f.qty}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>${f.price.toFixed(2)}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right', fontWeight: 600, color: f.pnl != null ? (f.pnl >= 0 ? 'var(--accent)' : 'var(--loss)') : 'var(--text-muted)' }}>
                      {f.pnl != null ? `$${f.pnl.toFixed(2)}` : '\u2014'}
                    </td>
                  </tr>
                ))}</tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ label, val, color }: { label: string; val: string; color?: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ fontWeight: 600, color }}>{val}</span>
    </div>
  );
}