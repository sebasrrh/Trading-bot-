import { usePaperStore } from '../../state/paper-store';
import type { WidgetProps } from '../types';

export default function PaperSummaryWidget({ config: _config }: WidgetProps) {
  const refreshKey = usePaperStore((s) => s.refreshKey);
  const getEngine = usePaperStore((s) => s.getEngine);
  const eng = getEngine();

  // Read reactive state (refreshKey ensures re-render)
  void refreshKey;

  const eq = eng.equity();
  const positions = [...eng.account.positions.values()];
  const openOrders = eng.account.openOrders.filter((o) => o.status === 'open');
  const recentFills = eng.account.fills.slice(-10).reverse();

  return (
    <div style={{ height: '100%', overflow: 'auto', fontSize: 12, padding: 8, display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* Account summary */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <Stat label="Cash" val={`$${eq.cash.toLocaleString('en-US', { minimumFractionDigits: 2 })}`} />
        <Stat label="Equity" val={`$${eq.total.toLocaleString('en-US', { minimumFractionDigits: 2 })}`} />
        <Stat label="Return" val={`${(eq.ret * 100).toFixed(2)}%`} color={eq.ret >= 0 ? 'var(--accent)' : 'var(--loss)'} />
        <Stat label="Positions" val={`${positions.length}`} />
        <Stat label="Open" val={`${openOrders.length}`} />
      </div>

      {/* Positions */}
      {positions.length > 0 && (
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>Positions</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead><tr style={{ borderBottom: '2px solid var(--border-hairline)' }}>
              <th style={{ textAlign: 'left', padding: '2px 6px' }}>Sym</th>
              <th style={{ textAlign: 'right', padding: '2px 6px' }}>Qty</th>
              <th style={{ textAlign: 'right', padding: '2px 6px' }}>P&L</th>
            </tr></thead>
            <tbody>{positions.map((p) => (
              <tr key={p.symbol} style={{ borderBottom: '1px solid var(--border-hairline)' }}>
                <td style={{ padding: '2px 6px', fontWeight: 600 }}>{p.symbol}</td>
                <td style={{ padding: '2px 6px', textAlign: 'right' }}>{p.qty}</td>
                <td style={{ padding: '2px 6px', textAlign: 'right', color: p.unrealizedPnl >= 0 ? 'var(--accent)' : 'var(--loss)', fontWeight: 600 }}>
                  ${p.unrealizedPnl.toFixed(2)}
                </td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}

      {/* Recent fills */}
      {recentFills.length > 0 && (
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>Recent Fills</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead><tr style={{ borderBottom: '2px solid var(--border-hairline)' }}>
              <th style={{ textAlign: 'left', padding: '2px 6px' }}>Sym</th>
              <th style={{ textAlign: 'left', padding: '2px 6px' }}>Side</th>
              <th style={{ textAlign: 'right', padding: '2px 6px' }}>Qty</th>
              <th style={{ textAlign: 'right', padding: '2px 6px' }}>Price</th>
            </tr></thead>
            <tbody>{recentFills.map((f) => (
              <tr key={f.id} style={{ borderBottom: '1px solid var(--border-hairline)' }}>
                <td style={{ padding: '2px 6px', fontWeight: 600 }}>{f.symbol}</td>
                <td style={{ padding: '2px 6px', color: f.side === 'buy' ? 'var(--accent)' : 'var(--loss)' }}>{f.side}</td>
                <td style={{ padding: '2px 6px', textAlign: 'right' }}>{f.qty}</td>
                <td style={{ padding: '2px 6px', textAlign: 'right' }}>${f.price.toFixed(2)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}

      {positions.length === 0 && recentFills.length === 0 && (
        <div style={{ color: 'var(--text-muted)', fontSize: 11, padding: 12, textAlign: 'center' }}>
          No positions or fills. Start trading from the Paper view.
        </div>
      )}
    </div>
  );
}

function Stat({ label, val, color }: { label: string; val: string; color?: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <span style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>{label}</span>
      <span style={{ fontSize: 14, fontWeight: 700, color, fontVariantNumeric: 'tabular-nums' }}>{val}</span>
    </div>
  );
}