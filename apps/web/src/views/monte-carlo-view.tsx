import { useRef, useEffect } from 'react';
import type { SimResult } from '@tradeboard/sim';

export function MonteCarloView({ result: { req, fan, terminal, terminalPercentiles, maxDdValues, maxDdPercentiles, ruinProb, backend, elapsedMs } }: { result: SimResult }) {
  const fanCanvasRef = useRef<HTMLCanvasElement>(null);
  const termCanvasRef = useRef<HTMLCanvasElement>(null);
  const ddCanvasRef = useRef<HTMLCanvasElement>(null);

  const H = req.horizon;
  const pcts = req.percentiles.length > 0 ? req.percentiles : [5, 25, 50, 75, 95];
  const startEq = req.s0 ?? 1;

  // Draw fan chart
  useEffect(() => {
    const canvas = fanCanvasRef.current;
    if (!canvas || H === 0) return;
    const ctx = canvas.getContext('2d')!;
    canvas.width = canvas.clientWidth * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
    const w = canvas.clientWidth;
    const ht = canvas.clientHeight;
    ctx.clearRect(0, 0, w, ht);

    let minY = Infinity, maxY = -Infinity;
    for (let t = 0; t < H; t++) {
      for (let pi = 0; pi < pcts.length; pi++) {
        const v = fan[pi * H + t]!;
        if (v < minY) minY = v;
        if (v > maxY) maxY = v;
      }
    }
    const pad = (maxY - minY) * 0.1 || 0.5;
    minY -= pad; maxY += pad;

    const xScale = (t: number) => 20 + (t / (H - 1)) * (w - 40);
    const yScale = (v: number) => ht - 20 - ((v - minY) / (maxY - minY)) * (ht - 40);

    const colors = ['rgba(0,200,100,0.1)', 'rgba(0,200,100,0.15)', 'rgba(0,200,100,0.2)', 'rgba(0,200,100,0.15)', 'rgba(0,200,100,0.1)'];
    const mid = Math.floor(pcts.length / 2);
    for (let band = 0; band < mid; band++) {
      const lowerPi = band;
      const upperPi = pcts.length - 1 - band;
      if (lowerPi >= upperPi) break;
      ctx.beginPath();
      for (let t = 0; t < H; t++) {
        const x = xScale(t);
        const y = yScale(fan[lowerPi * H + t]!);
        if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      for (let t = H - 1; t >= 0; t--) {
        const x = xScale(t);
        const y = yScale(fan[upperPi * H + t]!);
        ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = colors[band]!;
      ctx.fill();
      ctx.strokeStyle = 'rgba(0,200,100,0.3)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    ctx.beginPath();
    for (let t = 0; t < H; t++) {
      const x = xScale(t);
      const y = yScale(fan[mid * H + t]!);
      if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(0,200,100,0.8)';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    const refY = yScale(startEq);
    ctx.moveTo(20, refY);
    ctx.lineTo(w - 20, refY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '10px sans-serif';
    ctx.fillText('1.0x', 2, refY - 2);

    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '10px sans-serif';
    ctx.fillText(`+${((maxY - minY) * 100).toFixed(0)}%`, w - 40, 12);
  }, [fan, H, pcts, startEq]);

  // Draw terminal distribution histogram
  useEffect(() => {
    const canvas = termCanvasRef.current;
    if (!canvas || terminal.length === 0) return;
    const ctx = canvas.getContext('2d')!;
    canvas.width = canvas.clientWidth * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
    const w = canvas.clientWidth;
    const ht = canvas.clientHeight;
    ctx.clearRect(0, 0, w, ht);

    const bins = 40;
    let minV = Infinity, maxV = -Infinity;
    for (const v of terminal) { if (v < minV) minV = v; if (v > maxV) maxV = v; }
    const rng = maxV - minV || 1;
    const bucket = new Float64Array(bins);
    for (const v of terminal) {
      const idx = Math.min(bins - 1, Math.floor(((v - minV) / rng) * bins));
      bucket[idx]!++;
    }
    const maxCount = Math.max(...Array.from(bucket)) as number;
    if (maxCount === 0) return;

    const barW = (w - 40) / bins;
    for (let i = 0; i < bins; i++) {
      const bw = Math.max(1, barW - 1);
      const bh = (bucket[i]! / maxCount) * (ht - 40);
      const x = 20 + i * barW;
      const y = ht - 20 - bh;
      ctx.fillStyle = `hsl(${120 + (1 - i / bins) * 120}, 70%, ${40 + (i / bins) * 20}%)`;
      ctx.fillRect(x, y, bw, bh);
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    for (const p of terminalPercentiles) {
      const x = 20 + ((p - minV) / rng) * (w - 40);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ht - 20); ctx.stroke();
    }
    ctx.setLineDash([]);
  }, [terminal, terminalPercentiles]);

  // Draw max drawdown histogram
  useEffect(() => {
    const canvas = ddCanvasRef.current;
    if (!canvas || maxDdValues.length === 0) return;
    const ctx = canvas.getContext('2d')!;
    canvas.width = canvas.clientWidth * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
    const w = canvas.clientWidth;
    const ht = canvas.clientHeight;
    ctx.clearRect(0, 0, w, ht);

    const bins = 30;
    let minV = Infinity, maxV = -Infinity;
    for (const v of maxDdValues) { if (v < minV) minV = v; if (v > maxV) maxV = v; }
    const rng = maxV - minV || 1;
    const bucket = new Float64Array(bins);
    for (const v of maxDdValues) {
      const idx = Math.min(bins - 1, Math.floor(((v - minV) / rng) * bins));
      bucket[idx]!++;
    }
    const maxCount = Math.max(...Array.from(bucket)) as number;
    if (maxCount === 0) return;

    const barW = (w - 40) / bins;
    for (let i = 0; i < bins; i++) {
      const bh = (bucket[i]! / maxCount) * (ht - 40);
      const x = 20 + i * barW;
      const y = ht - 20 - bh;
      const t = i / bins;
      ctx.fillStyle = `rgb(${Math.round(t * 255)}, ${Math.round((1 - t) * 80)}, ${Math.round((1 - t) * 80)})`;
      ctx.fillRect(x, y, Math.max(1, barW - 1), bh);
    }
  }, [maxDdValues]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>Monte Carlo Simulation</h3>
        {ruinProb > 0.05 && (
          <span style={{ padding: '4px 10px', borderRadius: 'var(--radius-s)', background: ruinProb > 0.2 ? 'var(--loss-soft)' : 'var(--warning-soft)', color: ruinProb > 0.2 ? 'var(--loss)' : 'var(--warning)', fontSize: 11, fontWeight: 600 }}>
            ⚠ Ruin Risk {ruinProb > 0.2 ? 'High' : 'Elevated'}
          </span>
        )}
        <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 'auto' }}>{backend} · {(elapsedMs / 1000).toFixed(2)}s</span>
      </div>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ flex: 2, minWidth: 300 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Equity Fan Chart</label>
          <canvas ref={fanCanvasRef} style={{ width: '100%', height: 220, background: 'var(--bg-surface-1)', borderRadius: 'var(--radius-s)' }} />
        </div>

        <div style={{ flex: 1, minWidth: 200 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Percentiles</label>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginTop: 4 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--border-hairline)' }}>
                <th style={{ textAlign: 'left', padding: '4px 8px' }}>Pct</th>
                <th style={{ textAlign: 'right', padding: '4px 8px' }}>Terminal</th>
                <th style={{ textAlign: 'right', padding: '4px 8px' }}>Return</th>
                <th style={{ textAlign: 'right', padding: '4px 8px' }}>Max DD</th>
              </tr>
            </thead>
            <tbody>
              {pcts.map((p, i) => {
                const term = terminalPercentiles[i]!;
                const dd = maxDdPercentiles[i]!;
                return (
                  <tr key={p} style={{ borderBottom: '1px solid var(--border-hairline)' }}>
                    <td style={{ padding: '4px 8px', fontWeight: 600 }}>{p}%</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>{(term * 100).toFixed(1)}%</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right', color: term >= 1 ? 'var(--accent)' : 'var(--loss)' }}>
                      {((term - 1) * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '4px 8px', textAlign: 'right', color: dd < -0.2 ? 'var(--loss)' : 'var(--text-primary)' }}>
                      {(dd * 100).toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 250 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Terminal Distribution</label>
          <canvas ref={termCanvasRef} style={{ width: '100%', height: 160, background: 'var(--bg-surface-1)', borderRadius: 'var(--radius-s)' }} />
        </div>

        <div style={{ flex: 1, minWidth: 250 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Max Drawdown Distribution</label>
          <canvas ref={ddCanvasRef} style={{ width: '100%', height: 160, background: 'var(--bg-surface-1)', borderRadius: 'var(--radius-s)' }} />
        </div>
      </div>

      <div style={{ fontSize: 11, color: 'var(--text-muted)', padding: 8, background: 'var(--bg-surface-1)', borderRadius: 'var(--radius-s)' }}>
        {req.paths.toLocaleString()} paths · {req.horizon} bars · block size {req.blockLen ?? 21} ·
        ruin risk {(ruinProb * 100).toFixed(1)}% ·
        {req.mode === 'bootstrap' ? ' stationary bootstrap' : ' GBM'}
      </div>
    </div>
  );
}
