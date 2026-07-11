import { useState, useCallback } from 'react';
import { allStrategies } from '@tradeboard/strategies';
import { gridSearch, geneticOptimize } from '@tradeboard/optimizer';
import type { SearchSpace, OptimizerResult, ParamEval, ObjectiveMetric } from '@tradeboard/optimizer';
import { useContextStore } from '../state/context-store';

const strategies = allStrategies();

function paramDefault(schema: any): any {
  if (schema.shape) {
    const res: any = {};
    for (const [k, v] of Object.entries(schema.shape)) {
      res[k] = (v as any)._def?.defaultValue ?? 0;
    }
    return res;
  }
  return {};
}

function paramDef(schema: any, key: string): any {
  return schema?.shape?.[key];
}

function spaceRange(val: any, key: string, schema: any): { min: number; max: number } {
  const d = paramDef(schema, key);
  const checks = d?._def?.checks ?? [];
  const n = Number(val) || 10;
  return {
    min: checks.find((c: any) => c.kind === 'min')?.value ?? Math.max(1, Math.floor(n / 3)),
    max: checks.find((c: any) => c.kind === 'max')?.value ?? Math.max(n * 3, n + 10),
  };
}

function extractMetricVal(m: { sharpe: number; sortino: number; calmar: number; cagr: number; profitFactor: number; totalReturn: number }, key: ObjectiveMetric): number {
  return m[key];
}

export default function OptimizerView() {
  const ctx = useContextStore();
  const sym = ctx.channels.A.symbol;
  const [selectedId, setSelectedId] = useState('sma-cross');
  const [params, setParams] = useState<any>(() => paramDefault(strategies.find(s => s.id === 'sma-cross')!.paramsSchema));
  const [algo, setAlgo] = useState<'grid' | 'genetic' | 'both'>('grid');
  const [objective, setObjective] = useState<ObjectiveMetric>('sharpe');
  const [seed, setSeed] = useState(42);
  const [populationSize, setPopulationSize] = useState(20);
  const [generations, setGenerations] = useState(10);
  const [result, setResult] = useState<{ grid?: OptimizerResult; genetic?: OptimizerResult } | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');

  const select = useCallback((id: string) => {
    setSelectedId(id);
    const def = strategies.find(s => s.id === id);
    setParams(def ? paramDefault(def.paramsSchema) : {});
    setResult(null);
  }, []);

  const paramKeys = Object.keys(params);

  const run = useCallback(async () => {
    setRunning(true);
    setProgress('Fetching bars\u2026');
    setResult(null);
    try {
      const barsRes = await fetch(`http://localhost:8787/api/bars?symbol=${sym}&timeframe=1D&from=0&to=${Date.now()}`);
      const barsData = await barsRes.json();
      const barsArray = barsData.bars ?? [];
      if (barsArray.length === 0) { setProgress('No data returned.'); setRunning(false); return; }

      const bars = {
        symbol: sym, timeframe: '1D' as const,
        t: barsArray.map((b: any) => b.t), o: barsArray.map((b: any) => b.o),
        h: barsArray.map((b: any) => b.h), l: barsArray.map((b: any) => b.l),
        c: barsArray.map((b: any) => b.c), v: barsArray.map((b: any) => b.v),
        length: barsArray.length,
      };

      const makeBaseCfg = () => ({
        strategyId: selectedId, params: {},
        symbol: sym, timeframe: '1D' as const,
        from: barsArray[0]!.t, to: barsArray[barsArray.length - 1]!.t,
        initialCash: 100_000,
        costModel: { commissionPerOrder: 0, spreadBps: 2, slippageBps: 3 },
        sizing: { kind: 'all-in' as const },
        allowShort: false, seed,
      });

      const strat = strategies.find(s => s.id === selectedId)!;
      const schema = strat.paramsSchema;
      const space: SearchSpace = {};
      for (const k of paramKeys) {
        const sr = spaceRange(params[k], k, schema);
        space[k] = {
          min: sr.min,
          max: sr.max,
          step: Math.max(1, Math.floor((sr.max - sr.min) / 8)),
        };
      }

      const baseCfg = makeBaseCfg();

      const res: { grid?: OptimizerResult; genetic?: OptimizerResult } = {};

      if (algo === 'grid' || algo === 'both') {
        setProgress('Running grid search\u2026');
        res.grid = await gridSearch(baseCfg, space, bars, strat, objective);
      }
      if (algo === 'genetic' || algo === 'both') {
        setProgress('Running genetic algorithm\u2026');
        res.genetic = await geneticOptimize(baseCfg, space, bars, strat, {
          populationSize,
          generations,
          seed,
          objective,
        });
      }

      setResult(res);
      setProgress('Done');
    } catch (err) {
      setProgress(`Error: ${err}`);
    }
    setRunning(false);
  }, [selectedId, params, sym, algo, objective, seed, populationSize, generations, paramKeys]);

  const strat = strategies.find(s => s.id === selectedId) ?? strategies[0]!;

  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      <div style={{ width: 320, borderRight: '1px solid var(--border-hairline)', padding: 12, overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
        <h2 style={{ fontSize: 14, fontWeight: 700, margin: 0 }}>Strategy Optimizer</h2>

        <div>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Strategy</label>
          <select value={selectedId} onChange={(e) => select(e.target.value)}
            style={{ width: '100%', marginTop: 4, padding: '6px 8px', border: '1px solid var(--border-hairline)', borderRadius: 'var(--radius-s)', background: 'var(--bg-surface-2)', color: 'var(--text-primary)', fontSize: 13 }}>
            {strategies.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
        </div>

        <div>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Symbol</label>
          <div style={{ fontSize: 14, fontWeight: 600, marginTop: 4 }}>{sym}</div>
        </div>

        <div>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Algorithm</label>
          <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
            {(['grid', 'genetic', 'both'] as const).map(a => (
              <button key={a} onClick={() => setAlgo(a)}
                style={{ flex: 1, padding: '4px 6px', border: algo === a ? '1px solid var(--accent)' : '1px solid var(--border-hairline)', borderRadius: 'var(--radius-s)', background: algo === a ? 'var(--accent-soft)' : 'var(--bg-surface-2)', color: algo === a ? 'var(--accent)' : 'var(--text-secondary)', fontSize: 10, cursor: 'pointer' }}>
                {a === 'grid' ? 'Grid' : a === 'genetic' ? 'Genetic' : 'Both'}
              </button>
            ))}
          </div>
        </div>

        {(algo === 'genetic' || algo === 'both') && (
          <div style={{ display: 'flex', gap: 8 }}>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Population</label>
              <input type="number" value={populationSize} min={5} max={200}
                onChange={(e) => setPopulationSize(Number(e.target.value))}
                style={{ width: '100%', padding: '4px 6px', border: '1px solid var(--border-hairline)', borderRadius: 'var(--radius-s)', background: 'var(--bg-surface-2)', color: 'var(--text-primary)', fontSize: 12, boxSizing: 'border-box' }} />
            </div>
            <div style={{ flex: 1 }}>
              <label style={{ fontSize: 10, color: 'var(--text-muted)' }}>Generations</label>
              <input type="number" value={generations} min={1} max={200}
                onChange={(e) => setGenerations(Number(e.target.value))}
                style={{ width: '100%', padding: '4px 6px', border: '1px solid var(--border-hairline)', borderRadius: 'var(--radius-s)', background: 'var(--bg-surface-2)', color: 'var(--text-primary)', fontSize: 12, boxSizing: 'border-box' }} />
            </div>
          </div>
        )}

        <div>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Objective</label>
          <select value={objective} onChange={(e) => setObjective(e.target.value as ObjectiveMetric)}
            style={{ width: '100%', marginTop: 4, padding: '6px 8px', border: '1px solid var(--border-hairline)', borderRadius: 'var(--radius-s)', background: 'var(--bg-surface-2)', color: 'var(--text-primary)', fontSize: 13 }}>
            {(['sharpe', 'sortino', 'calmar', 'cagr', 'profitFactor', 'totalReturn'] as const).map(o => (
              <option key={o} value={o}>{o.charAt(0).toUpperCase() + o.slice(1)}</option>
            ))}
          </select>
        </div>

        <div>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Seed</label>
          <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))}
            style={{ width: '100%', marginTop: 4, padding: '4px 6px', border: '1px solid var(--border-hairline)', borderRadius: 'var(--radius-s)', background: 'var(--bg-surface-2)', color: 'var(--text-primary)', fontSize: 12, boxSizing: 'border-box' }} />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Param Ranges</label>
          {paramKeys.map(k => {
            const sr = spaceRange(params[k], k, strat.paramsSchema);
            return (
              <div key={k} style={{ fontSize: 12 }}>
                <div style={{ fontWeight: 600, marginBottom: 2 }}>{k}</div>
                <span style={{ color: 'var(--text-secondary)' }}>{sr.min} \u2014 {sr.max}</span>
              </div>
            );
          })}
        </div>

        <button onClick={run} disabled={running}
          style={{ padding: '8px 16px', border: 'none', borderRadius: 'var(--radius-s)', background: 'var(--accent)', color: '#fff', fontSize: 13, fontWeight: 600, cursor: running ? 'wait' : 'pointer' }}>
          {running ? progress : 'Optimize'}
        </button>
      </div>

      <div style={{ flex: 1, padding: 12, overflow: 'auto' }}>
        {!result ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: 14 }}>
            {running ? progress : 'Configure and run optimization.'}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {result.grid && <GridResultView result={result.grid} />}
            {result.genetic && <GeneticResultView result={result.genetic} />}
          </div>
        )}
      </div>
    </div>
  );
}

function MetricRow({ label, val, fmt }: { label: string; val: number; fmt?: 'pct' | 'num' | 'int' }) {
  return (
    <tr style={{ borderBottom: '1px solid var(--border-hairline)' }}>
      <td style={{ padding: '4px 8px', color: 'var(--text-secondary)' }}>{label}</td>
      <td style={{ padding: '4px 8px', textAlign: 'right', fontWeight: 600 }}>
        {fmt === 'pct' ? `${(val * 100).toFixed(2)}%` : fmt === 'num' ? val.toFixed(2) : String(val)}
      </td>
    </tr>
  );
}

function BestParamsTable({ best, totalEvals, durationMs }: { best: ParamEval; totalEvals: number; durationMs: number }) {
  const m = best.metrics;
  return (
    <div style={{ background: 'var(--bg-surface-1)', borderRadius: 'var(--radius-s)', padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div>
        <span style={{ fontSize: 13, fontWeight: 700 }}>Best Params: </span>
        <span style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 600, color: 'var(--accent)' }}>
          {Object.entries(best.params).map(([k, v]) => `${k}=${v}`).join(', ')}
        </span>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{totalEvals} evaluations \u00B7 {durationMs}ms</div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead><tr style={{ borderBottom: '2px solid var(--border-hairline)' }}>
          <th style={{ textAlign: 'left', padding: '4px 8px' }}>Metric</th><th style={{ textAlign: 'right', padding: '4px 8px' }}>Value</th>
        </tr></thead>
        <tbody>
          <MetricRow label="Sharpe" val={m.sharpe} fmt="num" />
          <MetricRow label="Sortino" val={m.sortino} fmt="num" />
          <MetricRow label="Calmar" val={m.calmar} fmt="num" />
          <MetricRow label="CAGR" val={m.cagr} fmt="pct" />
          <MetricRow label="Total Return" val={m.totalReturn} fmt="pct" />
          <MetricRow label="Max DD" val={m.maxDrawdown} fmt="pct" />
          <MetricRow label="Profit Factor" val={m.profitFactor} fmt="num" />
          <MetricRow label="Volatility" val={m.volatility} fmt="pct" />
        </tbody>
      </table>
    </div>
  );
}

function GridResultView({ result }: { result: OptimizerResult }) {
  const { best, evaluations, totalEvals, durationMs, objective } = result;
  const keys = Object.keys(best.params);
  const k1 = keys[0]!;
  const k2 = keys[1] ?? k1;

  const p1s = [...new Set(evaluations.map(e => e.params[k1]!))].sort((a, b) => a - b);
  const p2s = [...new Set(evaluations.map(e => e.params[k2]!))].sort((a, b) => a - b);

  const vals = evaluations.map(e => extractMetricVal(e.metrics, objective));
  const maxV = Math.max(...vals);
  const minV = Math.min(...vals);
  const rng = maxV - minV || 1;

  const objLabel = objective.charAt(0).toUpperCase() + objective.slice(1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>Grid Search</h3>
      <BestParamsTable best={best} totalEvals={totalEvals} durationMs={durationMs} />
      {keys.length >= 2 && (
        <div style={{ overflow: 'auto' }}>
          <h4 style={{ fontSize: 13, fontWeight: 600, margin: '8px 0' }}>{objLabel} Heatmap \u2014 {k1} \u00D7 {k2}</h4>
          <table style={{ borderCollapse: 'collapse', fontSize: 11 }}>
            <thead><tr>
              <th style={{ padding: '4px 8px', color: 'var(--text-muted)' }}>{k1} \\ {k2}</th>
              {p2s.map(v2 => <th key={v2} style={{ padding: '4px 6px', color: 'var(--text-muted)' }}>{v2}</th>)}
            </tr></thead>
            <tbody>{p1s.map(v1 => (
              <tr key={v1}>
                <td style={{ padding: '4px 8px', fontWeight: 600 }}>{v1}</td>
                {p2s.map(v2 => {
                  const cell = evaluations.find(e => e.params[k1] === v1 && e.params[k2] === v2);
                  if (!cell) return <td key={v2} style={{ padding: '4px 6px', color: 'var(--text-muted)' }}>\u2014</td>;
                  const t = (extractMetricVal(cell.metrics, objective) - minV) / rng;
                  return (
                    <td key={v2} style={{ padding: '4px 6px', textAlign: 'center', background: `rgb(${Math.round(t * 255)},${Math.round((1 - Math.abs(t - 0.5) * 2) * 200)},${Math.round((1 - t) * 255)})`, color: t > 0.5 ? 'white' : 'black', fontWeight: 600 }}>
                      {extractMetricVal(cell.metrics, objective).toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function GeneticResultView({ result }: { result: OptimizerResult }) {
  const { best, gaHistory, totalEvals, durationMs } = result;
  const history = gaHistory ?? [];

  const maxGenFitness = Math.max(...history.map(h => h.bestFitness));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <h3 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>Genetic Algorithm</h3>
      <BestParamsTable best={best} totalEvals={totalEvals} durationMs={durationMs} />
      {history.length > 0 && (
        <div style={{ background: 'var(--bg-surface-1)', borderRadius: 'var(--radius-s)', padding: 12 }}>
          <h4 style={{ fontSize: 13, fontWeight: 600, margin: '0 0 8px 0' }}>Convergence</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead><tr style={{ borderBottom: '2px solid var(--border-hairline)' }}>
              <th style={{ textAlign: 'left', padding: '4px 6px' }}>Gen</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>Best</th>
              <th style={{ textAlign: 'right', padding: '4px 6px' }}>Avg</th>
              <th style={{ textAlign: 'left', padding: '4px 6px' }}>Best Params</th>
            </tr></thead>
            <tbody>{history.map((h, i) => (
              <tr key={i} style={{ borderBottom: '1px solid var(--border-hairline)' }}>
                <td style={{ padding: '3px 6px', color: 'var(--text-muted)' }}>{h.generation}</td>
                <td style={{ padding: '3px 6px', textAlign: 'right', fontWeight: 600, color: h.bestFitness >= maxGenFitness * 0.95 ? 'var(--accent)' : undefined }}>
                  {h.bestFitness.toFixed(4)}
                </td>
                <td style={{ padding: '3px 6px', textAlign: 'right', color: 'var(--text-secondary)' }}>{h.avgFitness.toFixed(4)}</td>
                <td style={{ padding: '3px 6px', fontFamily: 'monospace', fontSize: 10 }}>
                  {Object.entries(h.individuals[0]!.params).map(([k, v]) => `${k}=${v}`).join(' ')}
                </td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}