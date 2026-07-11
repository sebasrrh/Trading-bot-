import type { WidgetProps } from '../types';

export default function EquityCurveWidget({ config }: WidgetProps) {
  const eq: number[] = config.equity ?? [];
  const start = eq[0] ?? 1;
  const vals = eq.map(v => ((v / start) - 1) * 100);

  if (vals.length === 0) {
    return <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, color: 'var(--text-muted)' }}>No equity data</div>;
  }

  const max = Math.max(...vals);
  const min = Math.min(...vals);
  const range = max - min || 1;
  const w = 600;
  const h = 200;
  const pts = vals.map((v, i) => `${(i / (vals.length - 1)) * w},${h - ((v - min) / range) * h * 0.8 - h * 0.1}`).join(' ');

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: 8 }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Equity Curve</div>
      <svg viewBox={`0 0 ${w} ${h}`} style={{ flex: 1, width: '100%' }}>
        <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
      </svg>
    </div>
  );
}