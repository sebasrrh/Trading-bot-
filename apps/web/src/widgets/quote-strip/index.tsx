import { useQuotes } from '../../lib/hooks/useQuotes';
import type { WidgetProps } from '../types';

export default function QuoteStripWidget({ config }: WidgetProps) {
  const symbols: string[] = config.symbols ?? ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN'];
  const { data } = useQuotes(symbols);

  return (
    <div style={{ height: '100%', display: 'flex', alignItems: 'center', gap: 12, padding: '0 12px', overflow: 'auto' }}>
      {symbols.map(sym => {
        const q = data?.quotes?.find(q => q.symbol === sym);
        const isUp = (q?.changePct ?? 0) >= 0;
        return (
          <div key={sym} style={{ display: 'flex', alignItems: 'center', gap: 6, whiteSpace: 'nowrap' }}>
            <span style={{ fontWeight: 600, fontSize: 12 }}>{sym}</span>
            <span style={{ fontSize: 12, fontVariantNumeric: 'tabular-nums' }}>
              {q ? `$${(q.price).toFixed(2)}` : '…'}
            </span>
            {q && (
              <span style={{ fontSize: 11, color: isUp ? 'var(--accent)' : 'var(--loss)', fontWeight: 500 }}>
                {isUp ? '+' : ''}{(q.changePct).toFixed(2)}%
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}