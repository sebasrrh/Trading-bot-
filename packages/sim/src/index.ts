export interface SimRequest {
  mode: 'bootstrap' | 'gbm';
  paths: number;
  horizon: number;
  seed: number;
  sourceReturns?: Float32Array;
  blockLen?: number;
  mu?: number;
  sigma?: number;
  dt?: number;
  s0?: number;
  jumps?: { lambda: number; muJ: number; sigmaJ: number };
  percentiles: number[];
}

export interface SimResult {
  runId: string;
  req: SimRequest;
  backend: 'webgpu' | 'cpu';
  elapsedMs: number;
  fan: Float32Array;
  terminal: number[];
  maxDrawdownDist: number[];
  ruinProb: number;
  ghostPaths: Float32Array;
}

export interface SimBackend {
  readonly kind: 'webgpu' | 'cpu';
  run(req: SimRequest, onProgress?: (pct: number) => void): Promise<SimResult>;
}

export function createSimBackend(): SimBackend {
  return {
    kind: 'cpu',
    async run(req, onProgress) {
      const start = performance.now();
      const fan = new Float32Array(req.percentiles.length * req.horizon);
      onProgress?.(100);
      return {
        runId: Math.random().toString(36).slice(2) + Date.now().toString(36),
        req,
        backend: 'cpu',
        elapsedMs: performance.now() - start,
        fan,
        terminal: [],
        maxDrawdownDist: [],
        ruinProb: 0,
        ghostPaths: new Float32Array(0),
      };
    },
  };
}