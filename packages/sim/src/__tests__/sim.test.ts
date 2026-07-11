import { describe, it, expect } from 'vitest';
import { CPUBackend } from '../cpu-backend';
import type { SimRequest } from '../types';

function makeRequest(overrides: Partial<SimRequest> = {}): SimRequest {
  // Generate some fake log returns (roughly normal)
  const n = 252;
  const sourceReturns = new Float32Array(n);
  const seed = 42;
  let state = seed >>> 0;
  for (let i = 0; i < n; i++) {
    state ^= (state << 13) >>> 0;
    state ^= (state >>> 17) >>> 0;
    state ^= (state << 5) >>> 0;
    sourceReturns[i] = (state >>> 0) / 4294967296 - 0.5;
  }
  return {
    mode: 'bootstrap' as const,
    paths: 200,
    horizon: 50,
    seed: 123,
    sourceReturns,
    blockLen: 10,
    percentiles: [5, 25, 50, 75, 95],
    ruinThreshold: 0.5,
    ghostPathCount: 10,
    ...overrides,
  };
}

describe('CPUBackend', () => {
  it('runs bootstrap simulation and returns all fields', async () => {
    const backend = new CPUBackend();
    const req = makeRequest();
    const result = await backend.run(req);

    expect(result.runId).toBeTruthy();
    expect(result.backend).toBe('cpu');
    expect(result.elapsedMs).toBeGreaterThan(0);
    expect(result.req).toBe(req);
    expect(result.fan.length).toBe(req.percentiles.length * req.horizon);
    expect(result.terminal.length).toBe(req.paths);
    expect(result.terminalPercentiles.length).toBe(req.percentiles.length);
    expect(result.maxDdValues.length).toBe(req.paths);
    expect(result.maxDdPercentiles.length).toBe(req.percentiles.length);
    expect(result.ruinProb).toBeGreaterThanOrEqual(0);
    expect(result.ruinProb).toBeLessThanOrEqual(1);
    expect(result.ghostPaths.length).toBe(req.ghostPathCount! * req.horizon);
  });

  it('produces monotonically ordered fan percentiles', async () => {
    const backend = new CPUBackend();
    const req = makeRequest({ paths: 500, horizon: 100 });
    const result = await backend.run(req);

    const pcts = req.percentiles;
    const H = req.horizon;
    // At each timestep, P5 <= P25 <= P50 <= P75 <= P95
    for (let t = 0; t < H; t++) {
      for (let pi = 1; pi < pcts.length; pi++) {
        const lower = result.fan[(pi - 1) * H + t]!;
        const upper = result.fan[pi * H + t]!;
        expect(lower).toBeLessThanOrEqual(upper + 1e-6);
      }
    }
  });

  it('fan chart final timestep matches terminal percentiles', async () => {
    const backend = new CPUBackend();
    const req = makeRequest({ paths: 500, horizon: 100 });
    const result = await backend.run(req);

    const H = req.horizon;
    const pcts = req.percentiles;
    for (let pi = 0; pi < pcts.length; pi++) {
      const fanEnd = result.fan[pi * H + (H - 1)]!;
      const termPct = result.terminalPercentiles[pi]!;
      expect(Math.abs(fanEnd - termPct)).toBeLessThan(0.001);
    }
  });

  it('ruin probability is zero for high threshold of 0', async () => {
    const backend = new CPUBackend();
    const req = makeRequest({ ruinThreshold: 0, paths: 200 });
    const result = await backend.run(req);
    expect(result.ruinProb).toBe(0);
  });

  it('handles edge case: horizon = 1', async () => {
    const backend = new CPUBackend();
    const req = makeRequest({ horizon: 1, paths: 100 });
    const result = await backend.run(req);
    expect(result.fan.length).toBe(req.percentiles.length * 1);
    expect(result.ghostPaths.length).toBe(req.ghostPathCount! * 1);
  });

  it('handles edge case: all zero returns (flat equity)', async () => {
    const backend = new CPUBackend();
    const zeros = new Float32Array(50);
    const req = makeRequest({ sourceReturns: zeros, horizon: 30, paths: 100 });
    const result = await backend.run(req);
    // All terminal values should be 1.0
    for (const v of result.terminal) {
      expect(Math.abs(v - 1)).toBeLessThan(1e-6);
    }
    expect(result.ruinProb).toBe(0);
  });

  it('ghost paths have correct dimensions', async () => {
    const backend = new CPUBackend();
    const gc = 7;
    const req = makeRequest({ ghostPathCount: gc, paths: 100, horizon: 40 });
    const result = await backend.run(req);
    expect(result.ghostPaths.length).toBe(gc * 40);
  });

  it('terminal values are all positive', async () => {
    const backend = new CPUBackend();
    const req = makeRequest({ paths: 300, horizon: 80 });
    const result = await backend.run(req);
    for (const v of result.terminal) {
      expect(v).toBeGreaterThan(0);
    }
  });

  it('max drawdown values are non-positive', async () => {
    const backend = new CPUBackend();
    const req = makeRequest({ paths: 300, horizon: 80 });
    const result = await backend.run(req);
    for (const v of result.maxDdValues) {
      expect(v).toBeLessThanOrEqual(0);
    }
  });
});
