import { useBars } from '../../lib/hooks/useBars';
import { useContextStore } from '../../state/context-store';
import type { WidgetProps } from '../types';

export default function CandlesWidget({ config }: WidgetProps) {
  const ctx = useContextStore();
  const ch = (typeof config.channel === 'string' ? config.channel as 'A' | 'B' | 'C' : null) as 'A' | 'B' | 'C' | null;
  const channel = ctx.getChannel(ch);
  const symbol: string = config.symbol ?? channel.symbol;
  const tf = config.timeframe ?? channel.timeframe;
  const range = config.dateRange ?? channel.dateRange;
  const { data, isLoading } = useBars(symbol, tf, range.from, range.to);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: 8 }}>
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4, color: 'var(--text-primary)' }}>
        {symbol} · {tf}
      </div>
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, color: 'var(--text-muted)' }}>
        {isLoading ? (
          <span>Loading {symbol} bars…</span>
        ) : data?.bars ? (
          <span>{data.bars.length} bars · latest: ${data.bars[data.bars.length - 1]?.c.toFixed(2)}</span>
        ) : (
          <span>No data for {symbol}</span>
        )}
      </div>
    </div>
  );
}