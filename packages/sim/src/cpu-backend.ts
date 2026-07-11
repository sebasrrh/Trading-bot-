import type { SimRequest, SimResult } from './types';

function xorshift(state: number): number {
  let s = state >>> 0;
  s ^= (s << 13) >>> 0;
  s ^= (s >>> 17) >>> 0;
  s ^= (s << 5) >>> 0;
  return s >>> 0;
}

function blockBootstrapPath(
  sourceReturns: Float32Array, horizon: number, blockLen: number,
  seed: number, ruinThreshold: number,
): { terminal: number; maxDD: number; ruined: number; equityPath: Float32Array } {
  let state = seed >>> 0;
  const n = sourceReturns.length;
  const H = horizon;
  const blk = blockLen;
  const thresh = ruinThreshold;

  let eq = 1.0;
  let peak = 1.0;
  let maxDD = 0.0;
  let ruined = 0;
  const equityPath = new Float32Array(H);

  let i = 0;
  while (i < H) {
    state = xorshift(state);
    const startIdx = Math.floor((state >>> 0) / 4294967296 * n);
    const remain = H - i;
    const L = Math.min(blk, remain);
    for (let j = 0; j < L; j++) {
      const idx = (startIdx + j) % n;
      const r = sourceReturns[idx]!;
      eq *= Math.exp(r);
      if (eq > peak) peak = eq;
      const dd = eq / peak - 1;
      if (dd < maxDD) maxDD = dd;
      if (eq < thresh && !ruined) ruined = 1;
      equityPath[i + j] = eq;
    }
    i += L;
  }
  return { terminal: eq, maxDD, ruined, equityPath };
}

export class CPUBackend {
  readonly kind = 'cpu' as const;

  async run(req: SimRequest, onProgress?: (pct: number) => void): Promise<SimResult> {
    const start = performance.now();
    const H = req.horizon;
    const P = req.paths;
    const gc = req.ghostPathCount ?? 100;
    const blk = req.blockLen ?? 21;
    const thresh = req.ruinThreshold ?? 0.5;
    const pcts = req.percentiles.length > 0 ? req.percentiles : [5, 25, 50, 75, 95];

    const terminals = new Float32Array(P);
    const maxDds = new Float32Array(P);
    const ruinedFlags = new Uint32Array(P);
    const fanMatrix = new Float32Array(P * H);
    const ghostPaths = new Float32Array(gc * H);

    for (let path = 0; path < P; path++) {
      const result = blockBootstrapPath(req.sourceReturns, H, blk, req.seed + path * 2654435761, thresh);
      terminals[path] = result.terminal;
      maxDds[path] = result.maxDD;
      ruinedFlags[path] = result.ruined;

      for (let t = 0; t < H; t++) {
        fanMatrix[t * P + path] = result.equityPath[t]!;
        if (path < gc) {
          ghostPaths[path * H + t] = result.equityPath[t]!;
        }
      }

      if (path % 500 === 0 && path > 0) {
        onProgress?.(Math.round((path / P) * 100));
      }
    }
    onProgress?.(90);

    // Fan chart percentiles
    const fan = new Float32Array(pcts.length * H);
    const temp = new Float64Array(P);
    for (let t = 0; t < H; t++) {
      const off = t * P;
      for (let i = 0; i < P; i++) temp[i] = fanMatrix[off + i]!;
      temp.sort((a, b) => a - b);
      for (let pi = 0; pi < pcts.length; pi++) {
        const idx = Math.floor((pcts[pi]! / 100) * (P - 1));
        fan[pi * H + t] = temp[idx]!;
      }
    }

    const sortedTerms = [...terminals].sort((a, b) => a - b);
    const termPcts = pcts.map(p => sortedTerms[Math.floor((p / 100) * (P - 1))]!);
    const sortedMaxDds = [...maxDds].sort((a, b) => a - b);
    const maxDdPcts = pcts.map(p => sortedMaxDds[Math.floor((p / 100) * (P - 1))]!);
    const ruinedCount = ruinedFlags.reduce((s, v) => s + v, 0);
    const ruinProb = P > 0 ? ruinedCount / P : 0;

    onProgress?.(100);
    const elapsed = performance.now() - start;

    return {
      runId: Math.random().toString(36).slice(2) + Date.now().toString(36),
      req,
      backend: 'cpu',
      elapsedMs: elapsed,
      fan,
      terminal: new Float32Array(sortedTerms),
      terminalPercentiles: termPcts,
      maxDdValues: new Float32Array(sortedMaxDds),
      maxDdPercentiles: maxDdPcts,
      ruinProb,
      ghostPaths,
    };
  }
}