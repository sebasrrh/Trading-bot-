import { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';
import { useBars } from '../../lib/hooks/useBars';
import { useContextStore } from '../../state/context-store';
import type { WidgetProps } from '../types';

const C = { bg: '#0d0d12', text: '#6b6b80', grid: '#1a1a24', up: '#26a69a', down: '#ef5350', wick: '#6b6b80' };

export default function CandlesWidget({ config }: WidgetProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candleRef = useRef<any>(null);
  const volumeRef = useRef<any>(null);
  const ctx = useContextStore();
  const ch = (typeof config.channel === 'string' ? config.channel as 'A' | 'B' | 'C' : null) as 'A' | 'B' | 'C' | null;
  const channel = ctx.getChannel(ch);
  const symbol: string = config.symbol ?? channel.symbol;
  const tf = config.timeframe ?? channel.timeframe;
  const range = config.dateRange ?? channel.dateRange;
  const { data, isLoading } = useBars(symbol, tf, range.from, range.to);
  const markers: any[] = config.markers ?? [];

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    if (!chartRef.current) {
      const chart = createChart(el, {
        layout: { background: { color: C.bg }, textColor: C.text },
        grid: { vertLines: { color: C.grid }, horzLines: { color: C.grid } },
        width: el.clientWidth, height: el.clientHeight,
        crosshair: { mode: 0 },
        timeScale: { borderColor: C.grid, timeVisible: true, secondsVisible: false },
      });
      chartRef.current = chart;
      candleRef.current = chart.addCandlestickSeries({ upColor: C.up, downColor: C.down, borderUpColor: C.up, borderDownColor: C.down, wickUpColor: C.wick, wickDownColor: C.wick });
      volumeRef.current = chart.addHistogramSeries({ color: '#26a69a33', priceFormat: { type: 'volume' }, priceScaleId: 'volume' });
      chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
    }
    const obs = new ResizeObserver(() => { if (chartRef.current && el) chartRef.current.resize(el.clientWidth, el.clientHeight); });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    if (!data?.bars || !candleRef.current || !volumeRef.current) return;
    const bars = data.bars;
    const cdl = bars.map((b: any) => ({ time: (b.t / 1000) as any, open: b.o, high: b.h, low: b.l, close: b.c }));
    const vol = bars.map((b: any) => ({ time: (b.t / 1000) as any, value: b.v, color: b.c >= b.o ? C.up + '44' : C.down + '44' }));
    candleRef.current.setData(cdl);
    volumeRef.current.setData(vol);
    chartRef.current?.timeScale().fitContent();
  }, [data]);

  useEffect(() => {
    if (!candleRef.current) return;
    candleRef.current.setMarkers(markers);
  }, [markers]);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ fontSize: 12, fontWeight: 600, padding: '4px 8px', color: 'var(--text-primary)', borderBottom: '1px solid var(--border-hairline)', display: 'flex', alignItems: 'center' }}>
        <span>{symbol} · {tf}</span>
        {data?.bars && <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--text-muted)', marginLeft: 8 }}>{data.bars.length} bars</span>}
      </div>
      <div ref={containerRef} style={{ flex: 1, minHeight: 0 }}>
        {isLoading && <div style={{ padding: 12, fontSize: 12, color: 'var(--text-muted)' }}>Loading...</div>}
      </div>
    </div>
  );
}