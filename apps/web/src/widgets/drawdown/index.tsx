import type { WidgetProps } from '../types';

export default function DrawdownWidget({ config }: WidgetProps) {
  const eq: number[] = config.equity ?? [];
  if (eq.length === 0) {
    return <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, color: 'var(--text-muted)' }}>No drawdown data</div>;
  }
  let peak = eq[0]!;
  const dd = eq.map(v => { if (v > peak) peak = v; return peak > 0 ? (v / peak - 1) * 100 : 0; });
  const maxDD = Math.min(...dd);
  const w = 600;
  const h = 160;
  const pts = dd.map((v, i) => `${(i / (dd.length - 1)) * w},${h / 2 + (v / (maxDD || -1)) * h * 0.45}`).join(' ');

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: 8 }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Drawdown</div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ flex: 1, width: '100%' }}>
        <polygon points={`0,${h / 2} ${pts} ${w},${h / 2}`} fill="var(--loss)" fillOpacity={0.2} />
        <polyline points={pts} fill="none" stroke="var(--loss)" strokeWidth={1} />
      </svg>
    </div>
  );
}