import { useState, useCallback } from 'react';
import { allStrategies } from '@tradeboard/strategies';
import type { RunResult } from '@tradeboard/backtest';
import { createSimBackend } from '@tradeboard/sim';
import type { SimResult } from '@tradeboard/sim';
import { MonteCarloView } from './monte-carlo-view';
import { useContextStore } from '../state/context-store';
import { useWorkspaceStore } from '../state/workspace-store';

const strategies = allStrategies();

function paramDefault(schema: any): any {
  if (schema.shape) {
    const res: any = {};
    for (const [k, v] of Object.entries(schema.shape)) {
      res[k] = (v as any)._def?.defaultValue ?? (v as any)._def?.type?.defaultValue ?? (v as any)._def?.type?.type?.defaultValue ?? (typeof (v as any)._def?.defaultValue === 'number' ? (v as any)._def?.defaultValue : 0);
    }
    return res;
  }
  return {};
}

function runWorker(cfg: any, bars: any): Promise<RunResult> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('../workers/backtest-worker.ts', import.meta.url), { type: 'module' });
    worker.postMessage({ id: 'run', kind: 'backtest', config: cfg, bars });
    worker.onmessage = (e) => {
      if (e.data.type === 'result') { resolve(e.data.payload as RunResult); worker.terminate(); }
      else if (e.data.type === 'error') { reject(e.data.message); worker.terminate(); }
    };
  });
}

export default function BacktestLab() {
  const ctx = useContextStore();
  const sym = ctx.channels.A.symbol;
  const [selectedId, setSelectedId] = useState('sma-cross');
  const [params, setParams] = useState<any>(() => paramDefault(strategies.find(s => s.id === 'sma-cross')!.paramsSchema));
  const [result, setResult] = useState<any>(null);
  const [bahResult, setBahResult] = useState<RunResult | null>(null);
  const [mcResult, setMcResult] = useState<SimResult | null>(null);
  const [savedReturns, setSavedReturns] = useState<Float32Array | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');
  const [mode, setMode] = useState<'single' | 'sweep' | 'walkforward' | 'montecarlo'>('single');

  const select = useCallback((id: string) => {
    setSelectedId(id);
    const def = strategies.find(s => s.id === id);
    setParams(def ? paramDefault(def.paramsSchema) : {});
    setResult(null); setBahResult(null); setMcResult(null);
  }, []);

  const run = useCallback(async () => {
    setRunning(true); setProgress('Fetching barsâ€¦'); setResult(null); setBahResult(null); setMcResult(null);
    const barsRes = await fetch(`/api/bars?symbol=${sym}&timeframe=1D&from=0&to=${Date.now()}`);
    const barsData = await barsRes.json();
    const barsArray = barsData.bars ?? [];
    if (barsArray.length === 0) { setProgress('No data returned.'); setRunning(false); return; }

    const bars = { symbol: sym, timeframe: '1D', t: barsArray.map((b: any) => b.t), o: barsArray.map((b: any) => b.o), h: barsArray.map((b: any) => b.h), l: barsArray.map((b: any) => b.l), c: barsArray.map((b: any) => b.c), v: barsArray.map((b: any) => b.v) };
    const n = barsArray.length;

    const makeCfg = (sid: string, p: any, subFrom?: number, subTo?: number) => ({
      strategyId: sid, params: p, symbol: sym, timeframe: '1D',
      from: subFrom ?? barsArray[0].t, to: subTo ?? barsArray[barsArray.length - 1].t,
      initialCash: 100_000, costModel: { commissionPerOrder: 0, spreadBps: 2, slippageBps: 3 },
      sizing: { kind: 'all-in' }, allowShort: false, seed: 42,
    });

    try {
      if (mode === 'montecarlo') {
        // Run MC on saved returns if available, or run a backtest first
        let returns = savedReturns;
        if (!returns) {
          setProgress('Running backtest for returnsâ€¦');
          const r = await runWorker(makeCfg(selectedId, params), bars);
          returns = new Float32Array(r.returnsPerBar);
          setSavedReturns(returns);
        }
        setProgress('Initializing simulationâ€¦');
        const backend = await createSimBackend();
        setProgress(`Running Monte Carlo on ${backend.kind}â€¦`);
        const mc = await backend.run({
          mode: 'bootstrap',
          paths: 5000,
          horizon: returns.length,
          seed: 42,
          sourceReturns: returns,
          blockLen: 21,
          percentiles: [5, 25, 50, 75, 95],
          ruinThreshold: 0.5,
          ghostPathCount: 100,
        }, (pct) => setProgress(`Monte Carlo ${pct}%`));
        setMcResult(mc);
      } else if (mode === 'single') {
        setProgress('Running strategy + benchmarkâ€¦');
        const [sr, br] = await Promise.all([runWorker(makeCfg(selectedId, params), bars), runWorker(makeCfg('buy-and-hold', {}), bars)]);
        setResult(sr); setBahResult(br);
        setSavedReturns(new Float32Array(sr.returnsPerBar));
      } else if (mode === 'sweep') {
        setProgress('Running param sweepâ€¦');
        const keys = Object.keys(params) as string[];
        if (keys.length === 0) { setRunning(false); return; }
        const k1 = keys[0]!; const k2 = keys[1] ?? k1;
        const schema = (strategies.find(s => s.id === selectedId)!.paramsSchema as any).shape;
        const p1Range = sweepingRange(schema[k1], params[k1!] as number);
        const p2Range = k1 !== k2 ? sweepingRange(schema[k2], params[k2!] as number) : p1Range;
        const results: { p1: number; p2: number; sharpe: number; ret: number }[] = [];
        for (const v1 of p1Range) {
          for (const v2 of p2Range) {
            const sp: any = { ...params }; sp[k1] = v1; sp[k2] = v2;
            setProgress(`Sweeping ${k1}=${v1} ${k2}=${v2} (${results.length}/${p1Range.length * p2Range.length})`);
            const r = await runWorker(makeCfg(selectedId, sp), bars);
            results.push({ p1: v1, p2: v2, sharpe: r.metrics.sharpe, ret: r.metrics.totalReturn });
          }
        }
        setResult({ sweepResults: results, paramKeys: [k1, k2] });
      } else {
        setProgress('Running walk-forward validationâ€¦');
        const K = 5;
        const anchorFrac = 0.6;
        const anchor = Math.floor(n * anchorFrac);
        const step = Math.floor((n - anchor) / K);
        if (step < 20) { setProgress('Not enough data for walk-forward'); setRunning(false); return; }

        const keys = Object.keys(params) as string[];
        const k1 = keys[0]!; const k2 = keys[1]!;
        const schema = (strategies.find(s => s.id === selectedId)!.paramsSchema as any).shape;
        const sweepValues = (k: string) => sweepingRange(schema[k], params[k!] as number);
        const v1s = sweepValues(k1);
        const v2s = sweepValues(k2);

        const folds: any[] = [];
        for (let f = 0; f < K; f++) {
          const isEnd = Math.min(anchor + f * step, n);
          const oosEnd = Math.min(anchor + (f + 1) * step, n);
          if (isEnd >= n || oosEnd - isEnd < 5) break;
          setProgress(`Walk-forward fold ${f + 1}/${K}: sweeping ISâ€¦`);
          let bestSharpe = -Infinity;
          let bestConfig: any = null;
          for (const v1 of v1s) {
            for (const v2 of v2s) {
              const sp: any = { ...params }; sp[k1] = v1; sp[k2] = v2;
              const r = await runWorker(makeCfg(selectedId, sp, barsArray[0].t, barsArray[isEnd - 1].t), bars);
              if (r.metrics.sharpe > bestSharpe) { bestSharpe = r.metrics.sharpe; bestConfig = sp; }
            }
          }
          setProgress(`Walk-forward fold ${f + 1}/${K}: testing OOSâ€¦`);
          let oosSharpe = 0;
          try {
            const oosResult = await runWorker(makeCfg(selectedId, bestConfig, barsArray[isEnd].t, barsArray[oosEnd - 1].t), bars);
            oosSharpe = oosResult.metrics.sharpe;
          } catch { }
          folds.push({ bestParams: bestConfig, ISSharpe: bestSharpe, OOSSharpe: oosSharpe });
        }

        setProgress('Optimizing on full dataâ€¦');
        let fullBestSharpe = -Infinity;
        for (const v1 of v1s) {
          for (const v2 of v2s) {
            const sp: any = { ...params }; sp[k1] = v1; sp[k2] = v2;
            const r = await runWorker(makeCfg(selectedId, sp), bars);
            if (r.metrics.sharpe > fullBestSharpe) fullBestSharpe = r.metrics.sharpe;
          }
        }

        let stitchedOOSSharpe = folds.reduce((s: number, f: any) => s + f.OOSSharpe, 0) / Math.max(1, folds.length);
        if (!isFinite(stitchedOOSSharpe)) stitchedOOSSharpe = 0;
        const overfit = (fullBestSharpe > 0.05 && stitchedOOSSharpe < fullBestSharpe * 0.5);

        setResult({ walkForward: true, folds, fullISSharpe: fullBestSharpe, stitchedOOSSharpe, overfit, strategyId: selectedId });
      }
      setProgress('Done');
    } catch (err) { setProgress(`Error: ${err}`); }
    setRunning(false);
  }, [selectedId, params, sym, mode, savedReturns]);

  const addToDashboard = useCallback(() => {
    if (!result || result.sweepResults || result.walkForward) return;
    const r = result as RunResult;
    const ws = useWorkspaceStore.getState();
    ws.addWidget('equity-curve', { equity: Array.from(r.equity) }, { w: 6, h: 8 });
    ws.addWidget('drawdown', { equity: Array.from(r.equity) }, { w: 6, h: 6 });
    ws.addWidget('run-metrics', { metrics: r.metrics }, { w: 4, h: 8 });
    useContextStore.getState().setActiveView('dashboard');
  }, [result]);

  const strat = strategies.find(s => s.id === selectedId) ?? strategies[0]!;

  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      <div style={{ width: 320, borderRight: '1px solid var(--border-hairline)', padding: 12, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
        <h2 style={{ fontSize: 14, fontWeight: 700, margin: 0 }}>Backtest Lab</h2>
        <div style={{ display: 'flex', gap: 4 }}>
          {(['single', 'sweep', 'walkforward', 'montecarlo'] as const).map(m => (
            <button key={m} onClick={() => setMode(m)}
              style={{ flex: 1, padding: '4px 6px', border: mode === m ? '1px solid var(--accent)' : '1px solid var(--border-hairline)', borderRadius: 'var(--radius-s)', background: mode === m ? 'var(--accent-soft)' : 'var(--bg-surface-2)', color: mode === m ? 'var(--accent)' : 'var(--text-secondary)', fontSize: 10, cursor: 'pointer' }}>
              {m === 'single' ? 'Run' : m === 'sweep' ? 'Sweep' : m === 'walkforward' ? 'W-Fwd' : 'MC'}
            </button>
          ))}
        </div>
        <div>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Strategy</label>
          <select value={selectedId} onChange={(e) => select(e.target.value)} style={{ width: '100%', marginTop: 4, padding: '6px 8px', border: '1px solid var(--border-hairline)', borderRadius: 'var(--radius-s)', background: 'var(--bg-surface-2)', color: 'var(--text-primary)', fontSize: 13 }}>
            {strategies.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>{strat.description}</p>
        </div>
        <div><label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Symbol</label><div style={{ fontSize: 14, fontWeight: 600, marginTop: 4 }}>{sym}</div></div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Parameters</label>
          {Object.entries(params).map(([key, val]) => (
            <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 12, width: 60, color: 'var(--text-secondary)' }}>{key}</span>
              <input type="number" value={val as number} onChange={(e) => setParams({ ...params, [key]: Number(e.target.value) })}
                style={{ flex: 1, padding: '4px 6px', border: '1px solid var(--border-hairline)', borderRadius: 'var(--radius-s)', background: 'var(--bg-surface-2)', color: 'var(--text-primary)', fontSize: 12 }} />
            </div>
          ))}
        </div>
        <button onClick={run} disabled={running}
          style={{ padding: '8px 16px', border: 'none', borderRadius: 'var(--radius-s)', background: 'var(--accent)', color: '#fff', fontSize: 13, fontWeight: 600, cursor: running ? 'wait' : 'pointer' }}>
          {running ? progress : mode === 'single' ? 'Run' : mode === 'sweep' ? 'Sweep' : mode === 'walkforward' ? 'Walk-Forward' : 'Monte Carlo'}
        </button>
      </div>
      <div style={{ flex: 1, padding: 12, overflow: 'auto' }}>
        {!result && !mcResult ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: 14 }}>
            {running ? progress : 'Configure and run a backtest.'}
          </div>
        ) : mcResult ? (
          <MonteCarloView result={mcResult} />
        ) : result?.walkForward ? (
          <WalkForwardView data={result} />
        ) : result?.sweepResults ? (
          <SweepView sweep={result} />
        ) : result ? (
          <SingleResultView result={result as RunResult} bahResult={bahResult} onAddToDashboard={addToDashboard} />
        ) : null}
      </div>
    </div>
  );
}

function SingleResultView({ result, bahResult, onAddToDashboard }: { result: RunResult; bahResult: RunResult | null; onAddToDashboard: () => void }) {
  const m = result.metrics;
  const bah = bahResult?.metrics;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>Results</h3>
        <button onClick={onAddToDashboard} style={{ padding: '4px 10px', border: '1px solid var(--border-hairline)', background: 'var(--bg-surface-2)', color: 'var(--text-primary)', borderRadius: 'var(--radius-s)', fontSize: 11, cursor: 'pointer' }}>Add to Dashboard</button>
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead><tr style={{ borderBottom: '2px solid var(--border-hairline)' }}><th style={{ textAlign: 'left', padding: '6px 8px' }}>Metric</th><th style={{ textAlign: 'right', padding: '6px 8px' }}>{result.config.strategyId}</th>{bah && <th style={{ textAlign: 'right', padding: '6px 8px', color: 'var(--text-muted)' }}>Buy & Hold</th>}</tr></thead>
        <tbody>{[
          ['Total Return', m.totalReturn, bah?.totalReturn, 'pct'], ['CAGR', m.cagr, bah?.cagr, 'pct'], ['Sharpe', m.sharpe, bah?.sharpe, 'num'], ['Sortino', m.sortino, bah?.sortino, 'num'], ['Volatility', m.volatility, bah?.volatility, 'pct'], ['Max DD', m.maxDrawdown, bah?.maxDrawdown, 'pct'], ['Win Rate', m.winRate, null, 'pct'], ['Trades', m.tradeCount, null, 'int'],
        ].map(([label, val, bahVal, fmt]) => (<tr key={label as string} style={{ borderBottom: '1px solid var(--border-hairline)' }}><td style={{ padding: '4px 8px', color: 'var(--text-secondary)' }}>{label as string}</td><td style={{ padding: '4px 8px', textAlign: 'right', fontWeight: 600 }}>{fmt === 'pct' ? `${((val as number)*100).toFixed(2)}%` : fmt === 'num' ? (val as number).toFixed(2) : String(val)}</td>{bah && <td style={{ padding: '4px 8px', textAlign: 'right', color: 'var(--text-muted)' }}>{bahVal != null ? (fmt === 'pct' ? `${((bahVal as number)*100).toFixed(2)}%` : (bahVal as number).toFixed(2)) : 'â€”'}</td>}</tr>))}</tbody>
      </table>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Run: {result.runId} Â· {result.trades.length} trades Â· Cost: spread {result.config.costModel.spreadBps}bps + slippage {result.config.costModel.slippageBps}bps</div>
    </div>
  );
}

function SweepView({ sweep }: { sweep: { sweepResults: { p1: number; p2: number; sharpe: number; ret: number }[]; paramKeys: string[] } }) {
  const { sweepResults, paramKeys } = sweep;
  const p1s = [...new Set(sweepResults.map(r => r.p1))].sort((a, b) => a - b);
  const p2s = [...new Set(sweepResults.map(r => r.p2))].sort((a, b) => a - b);
  const maxSh = Math.max(...sweepResults.map(r => r.sharpe));
  const minSh = Math.min(...sweepResults.map(r => r.sharpe));
  const rng = maxSh - minSh || 1;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>Param Sweep â€” {paramKeys[0]} Ã— {paramKeys[1]}</h3>
      <div style={{ overflow: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', fontSize: 11 }}>
          <thead><tr><th style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{paramKeys[0]} \ {paramKeys[1]}</th>{p2s.map(v2 => <th key={v2} style={{ padding: '4px 6px', color: 'var(--text-muted)' }}>{v2}</th>)}</tr></thead>
          <tbody>{p1s.map(v1 => (<tr key={v1}><td style={{ padding: '4px 8px', fontWeight: 600 }}>{v1}</td>{p2s.map(v2 => { const cell = sweepResults.find(r => r.p1 === v1 && r.p2 === v2); if (!cell) return <td key={v2} style={{ padding: '4px 6px', color: 'var(--text-muted)' }}>â€”</td>; const t = (cell.sharpe - minSh) / rng; return <td key={v2} style={{ padding: '4px 6px', textAlign: 'center', background: `rgb(${Math.round(t*255)},${Math.round((1-Math.abs(t-0.5)*2)*200)},${Math.round((1-t)*255)})`, color: t>0.5?'white':'black', fontWeight: 600 }}>{cell.sharpe.toFixed(2)}</td>; })})</tr>))}</tbody>
        </table>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Sharpe range: {minSh.toFixed(2)} â†’ {maxSh.toFixed(2)}</div>
    </div>
  );
}

function WalkForwardView({ data }: { data: any }) {
  const { folds, fullISSharpe, stitchedOOSSharpe, overfit } = data;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>Walk-Forward Validation</h3>
        {overfit && (
          <span style={{ padding: '4px 10px', borderRadius: 'var(--radius-s)', background: 'var(--loss-soft)', color: 'var(--loss)', fontSize: 11, fontWeight: 600, textTransform: 'uppercase' }}>
            âš  Likely Overfit
          </span>
        )}
      </div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead><tr style={{ borderBottom: '2px solid var(--border-hairline)' }}><th style={{ textAlign: 'left', padding: '6px 8px' }}>Metric</th><th style={{ textAlign: 'right', padding: '6px 8px' }}>In-Sample</th><th style={{ textAlign: 'right', padding: '6px 8px' }}>Out-of-Sample</th></tr></thead>
        <tbody>
          <tr style={{ borderBottom: '1px solid var(--border-hairline)' }}><td style={{ padding: '4px 8px', color: 'var(--text-secondary)' }}>Sharpe</td><td style={{ padding: '4px 8px', textAlign: 'right', fontWeight: 600 }}>{fullISSharpe.toFixed(2)}</td><td style={{ padding: '4px 8px', textAlign: 'right', fontWeight: 600, color: overfit ? 'var(--loss)' : 'var(--accent)' }}>{stitchedOOSSharpe.toFixed(2)}</td></tr>
          <tr style={{ borderBottom: '1px solid var(--border-hairline)' }}><td style={{ padding: '4px 8px', color: 'var(--text-secondary)' }}>OOS / IS</td><td colSpan={2} style={{ padding: '4px 8px', textAlign: 'right', fontWeight: 600 }}>{fullISSharpe !== 0 ? (stitchedOOSSharpe / fullISSharpe * 100).toFixed(1) + '%' : 'â€”'}</td></tr>
        </tbody>
      </table>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <h4 style={{ fontSize: 13, fontWeight: 600, margin: 0 }}>Folds</h4>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead><tr style={{ borderBottom: '2px solid var(--border-hairline)' }}><th style={{ textAlign: 'left', padding: '4px 6px' }}>#</th><th style={{ textAlign: 'left', padding: '4px 6px' }}>Best Params</th><th style={{ textAlign: 'right', padding: '4px 6px' }}>IS Sharpe</th><th style={{ textAlign: 'right', padding: '4px 6px' }}>OOS Sharpe</th></tr></thead>
          <tbody>{folds.map((f: any, i: number) => (<tr key={i} style={{ borderBottom: '1px solid var(--border-hairline)' }}><td style={{ padding: '4px 6px' }}>{i + 1}</td><td style={{ padding: '4px 6px', fontFamily: 'monospace', fontSize: 10 }}>{Object.entries(f.bestParams ?? {}).map(([k, v]) => `${k}=${v}`).join(' ')}</td><td style={{ padding: '4px 6px', textAlign: 'right', fontWeight: 600 }}>{f.ISSharpe.toFixed(2)}</td><td style={{ padding: '4px 6px', textAlign: 'right', fontWeight: 600, color: f.OOSSharpe < f.ISSharpe * 0.5 ? 'var(--loss)' : undefined }}>{f.OOSSharpe.toFixed(2)}</td></tr>))}</tbody>
        </table>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', padding: 8, background: 'var(--bg-surface-1)', borderRadius: 'var(--radius-s)' }}>
        {overfit ? (
          <span style={{ color: 'var(--loss)' }}>âš  <strong>Likely overfit:</strong> Out-of-sample Sharpe ({stitchedOOSSharpe.toFixed(2)}) is less than 50% of in-sample Sharpe ({fullISSharpe.toFixed(2)}). The optimized parameters may be curve-fit to noise.</span>
        ) : (
          <span>OOS Sharpe ratio is within acceptable range of IS Sharpe. No overfit warning.</span>
        )}
      </div>
    </div>
  );
}

function sweepingRange(schemaField: any, current: number): number[] {
  const min = (schemaField as any)?._def?.checks?.find?.((c: any) => c.kind === 'min')?.value ?? Math.floor(current / 2);
  const max = (schemaField as any)?._def?.checks?.find?.((c: any) => c.kind === 'max')?.value ?? Math.floor(current * 2);
  const step = Math.max(1, Math.floor((max - min) / 5));
  const values: number[] = [];
  for (let v = min; v <= max; v += step) values.push(v);
  if (values.length > 7) return values.slice(0, 7);
  if (!values.includes(current)) values.push(current);
  return values.sort((a, b) => a - b);
}