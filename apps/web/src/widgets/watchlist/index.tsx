import { useQuotes } from '../../lib/hooks/useQuotes';
import type { WidgetProps } from '../types';

const DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA'];

export default function WatchlistWidget({ config }: WidgetProps) {
  const symbols: string[] = config.symbols ?? DEFAULT_SYMBOLS;
  const { data } = useQuotes(symbols);

  return (
    <div style={{ height: '100%', overflow: 'auto', fontSize: 12 }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ color: 'var(--text-muted)', fontSize: 11, textTransform: 'uppercase' }}>
            <th style={{ textAlign: 'left', padding: '4px 8px' }}>Symbol</th>
            <th style={{ textAlign: 'right', padding: '4px 8px' }}>Last</th>
            <th style={{ textAlign: 'right', padding: '4px 8px' }}>Δ%</th>
          </tr>
        </thead>
        <tbody>
          {symbols.map(sym => {
            const q = data?.quotes?.find(q => q.symbol === sym);
            const isUp = (q?.changePct ?? 0) >= 0;
            return (
              <tr key={sym} style={{ borderTop: '1px solid var(--border-hairline)' }}>
                <td style={{ padding: '4px 8px', fontWeight: 600 }}>{sym}</td>
                <td style={{ padding: '4px 8px', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                  {q ? `$${(q.price).toFixed(2)}` : '—'}
                </td>
                <td style={{ padding: '4px 8px', textAlign: 'right', color: isUp ? 'var(--accent)' : 'var(--loss)' }}>
                  {q ? `${isUp ? '+' : ''}${(q.changePct).toFixed(2)}%` : '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}