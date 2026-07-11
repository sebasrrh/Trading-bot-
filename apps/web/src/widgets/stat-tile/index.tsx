import { useQuote } from '../../lib/hooks/useQuotes';
import type { WidgetProps } from '../types';

export default function StatTileWidget({ config, size }: WidgetProps) {
  const symbol: string = config.symbol ?? 'SPY';
  const label: string = config.label ?? 'Price';
  const { data: quote } = useQuote(symbol);

  const val = quote?.price;
  const change = quote?.changePct;

  // Widgets adapt to their real rendered size rather than a fixed layout
  // (docs/01 "adapt, not media-query") — a tile squeezed narrow/short by a
  // drag-resize drops the label and shrinks the value instead of clipping it.
  const compact = size.hPx > 0 && size.hPx < 70;
  const narrow = size.wPx > 0 && size.wPx < 110;

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: narrow ? 10 : 16 }}>
      {!compact && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          {label}
        </div>
      )}
      <div style={{ fontSize: narrow || compact ? 20 : 28, fontWeight: 700, fontVariantNumeric: 'tabular-nums', marginTop: compact ? 0 : 4 }}>
        {val != null ? `$${val.toFixed(2)}` : '—'}
      </div>
      {change != null && !compact && (
        <div style={{ fontSize: 13, marginTop: 2, color: change >= 0 ? 'var(--gain)' : 'var(--loss)' }}>
          {change >= 0 ? '+' : ''}{change.toFixed(2)}%
        </div>
      )}
    </div>
  );
}
