export interface SimRequest {
  mode: 'bootstrap' | 'gbm';
  paths: number;
  horizon: number;
  seed: number;
  sourceReturns: Float32Array;
  blockLen?: number;
  mu?: number;
  sigma?: number;
  dt?: number;
  s0?: number;
  jumps?: { lambda: number; muJ: number; sigmaJ: number };
  percentiles: number[];
  ruinThreshold?: number;
  ghostPathCount?: number;
}

export interface SimResult {
  runId: string;
  req: SimRequest;
  backend: 'webgpu' | 'cpu';
  elapsedMs: number;
  fan: Float32Array;
  terminal: Float32Array;
  terminalPercentiles: number[];
  maxDdValues: Float32Array;
  maxDdPercentiles: number[];
  ruinProb: number;
  ghostPaths: Float32Array;
}

export interface SimBackend {
  readonly kind: 'webgpu' | 'cpu';
  run(req: SimRequest, onProgress?: (pct: number) => void): Promise<SimResult>;
}