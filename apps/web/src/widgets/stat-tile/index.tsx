import { useQuote } from '../../lib/hooks/useQuotes';
import type { WidgetProps } from '../types';

export default function StatTileWidget({ config }: WidgetProps) {
  const symbol: string = config.symbol ?? 'SPY';
  const label: string = config.label ?? 'Price';
  const { data: quote } = useQuote(symbol);

  const val = quote?.price;
  const change = quote?.changePct;

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: 16 }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
        {label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, fontVariantNumeric: 'tabular-nums', marginTop: 4 }}>
        {val != null ? `$${val.toFixed(2)}` : '—'}
      </div>
      {change != null && (
        <div style={{ fontSize: 13, marginTop: 2, color: change >= 0 ? 'var(--accent)' : 'var(--loss)' }}>
          {change >= 0 ? '+' : ''}{change.toFixed(2)}%
        </div>
      )}
    </div>
  );
}