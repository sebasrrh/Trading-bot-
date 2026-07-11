import type { WidgetProps } from '../types';

export default function RunMetricsWidget({ config }: WidgetProps) {
  const m = config.metrics as Record<string, number> | null;
  if (!m) {
    return <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, color: 'var(--text-muted)' }}>No metrics</div>;
  }
  const rows = [
    ['Total Return', m.totalReturn != null ? `${(m.totalReturn * 100).toFixed(2)}%` : '—'],
    ['CAGR', m.cagr != null ? `${(m.cagr * 100).toFixed(2)}%` : '—'],
    ['Sharpe', m.sharpe?.toFixed(2) ?? '—'],
    ['Sortino', m.sortino?.toFixed(2) ?? '—'],
    ['Volatility', m.volatility != null ? `${(m.volatility * 100).toFixed(2)}%` : '—'],
    ['Max DD', m.maxDrawdown != null ? `${(m.maxDrawdown * 100).toFixed(2)}%` : '—'],
    ['Win Rate', m.winRate != null ? `${(m.winRate * 100).toFixed(1)}%` : '—'],
    ['Trades', String(m.tradeCount ?? '—')],
  ];

  return (
    <div style={{ height: '100%', overflow: 'auto', fontSize: 12, padding: 8 }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <tbody>
          {rows.map(([label, val]) => (
            <tr key={label} style={{ borderBottom: '1px solid var(--border-hairline)' }}>
              <td style={{ padding: '4px 8px', color: 'var(--text-secondary)' }}>{label}</td>
              <td style={{ padding: '4px 8px', textAlign: 'right', fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>{val}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}