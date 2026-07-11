import { BOOTSTRAP_SHADER } from './wgsl/bootstrap';
import type { SimRequest, SimResult } from './types';

export class WebGPUBackend {
  readonly kind = 'webgpu' as const;
  private device: any;
  private module: any;

  constructor(device: any) {
    this.device = device;
    this.module = device.createShaderModule({ code: BOOTSTRAP_SHADER });
  }

  async run(req: SimRequest, onProgress?: (pct: number) => void): Promise<SimResult> {
    const start = performance.now();
    const d = this.device as any;
    const H = req.horizon;
    const P = req.paths;
    const gc = req.ghostPathCount ?? 100;
    const nSrc = req.sourceReturns.length;

    const UNIFORM = 64;
    const STORAGE = 128;
    const COPY_SRC = 4;
    const COPY_DST = 8;
    const MAP_READ = 1;

    const roundUp = (v: number) => Math.ceil(v / 4) * 4;

    const paramsBuf = d.createBuffer({ size: 32, usage: UNIFORM | COPY_DST });
    const paramsArr = new Uint32Array(8);
    paramsArr[0] = P;
    paramsArr[1] = H;
    paramsArr[2] = req.seed;
    paramsArr[3] = req.blockLen ?? 21;
    paramsArr[4] = nSrc;
    paramsArr[5] = 0;
    paramsArr[6] = gc;
    paramsArr[7] = 0;
    const pf32 = new Float32Array(paramsArr.buffer);
    pf32[5] = req.ruinThreshold ?? 0.5;
    d.queue.writeBuffer(paramsBuf, 0, paramsArr.buffer);

    const srcBuf = d.createBuffer({ size: roundUp(nSrc * 4), usage: STORAGE | COPY_DST });
    d.queue.writeBuffer(srcBuf, 0, req.sourceReturns.buffer);

    const stor = (size: number) => d.createBuffer({ size: roundUp(size * 4), usage: STORAGE | COPY_SRC });
    const storU32 = (size: number) => d.createBuffer({ size: roundUp(size * 4), usage: STORAGE | COPY_SRC });

    const termBuf = stor(P);
    const maxdBuf = stor(P);
    const ruinBuf = storU32(P);
    const fanBuf = stor(P * H);
    const ghostBuf = stor(gc * H);

    const bindGroup = d.createBindGroup({
      layout: this.module.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: srcBuf } },
        { binding: 2, resource: { buffer: termBuf } },
        { binding: 3, resource: { buffer: maxdBuf } },
        { binding: 4, resource: { buffer: ruinBuf } },
        { binding: 5, resource: { buffer: fanBuf } },
        { binding: 6, resource: { buffer: ghostBuf } },
      ],
    });

    const pipeline = d.createComputePipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.module.getBindGroupLayout(0)] }),
      compute: { module: this.module, entryPoint: 'main' },
    });

    const encoder = d.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(P / 64));
    pass.end();
    d.queue.submit([encoder.finish()]);
    await d.queue.onSubmittedWorkDone();
    onProgress?.(50);

    const readBuf = (buf: any, size: number) => {
      const staging = d.createBuffer({ size: roundUp(size * 4), usage: MAP_READ | COPY_DST });
      const cmd = d.createCommandEncoder();
      cmd.copyBufferToBuffer(buf, 0, staging, 0, roundUp(size * 4));
      d.queue.submit([cmd.finish()]);
      return staging;
    };

    const [termData, maxdData, ruinData, fanData, ghostData] = await Promise.all([
      this.readback(d, readBuf(termBuf, P), P),
      this.readback(d, readBuf(maxdBuf, P), P),
      this.readbackU32(d, readBuf(ruinBuf, P), P),
      this.readback(d, readBuf(fanBuf, P * H), P * H),
      this.readback(d, readBuf(ghostBuf, gc * H), gc * H),
    ]);

    onProgress?.(80);

    const pcts = req.percentiles.length > 0 ? req.percentiles : [5, 25, 50, 75, 95];
    const fan = new Float32Array(pcts.length * H);
    const temp = new Float64Array(P);
    for (let t = 0; t < H; t++) {
      const off = t * P;
      for (let i = 0; i < P; i++) temp[i] = fanData[off + i]!;
      temp.sort((a, b) => a - b);
      for (let pi = 0; pi < pcts.length; pi++) {
        const idx = Math.floor((pcts[pi]! / 100) * (P - 1));
        fan[pi * H + t] = temp[idx]!;
      }
    }

    const sortedTerm = [...termData].sort((a, b) => a - b);
    const termPcts = pcts.map(p => sortedTerm[Math.floor((p / 100) * (P - 1))]!);

    const sortedMaxDd = [...maxdData].sort((a, b) => a - b);
    const maxDdPcts = pcts.map(p => sortedMaxDd[Math.floor((p / 100) * (P - 1))]!);

    const ruinedCount = ruinData.reduce((s, v) => s + v, 0);
    const ruinProb = P > 0 ? ruinedCount / P : 0;

    onProgress?.(100);

    return {
      runId: Math.random().toString(36).slice(2) + Date.now().toString(36),
      req,
      backend: 'webgpu',
      elapsedMs: performance.now() - start,
      fan,
      terminal: termData,
      terminalPercentiles: termPcts,
      maxDdValues: new Float32Array(sortedMaxDd),
      maxDdPercentiles: maxDdPcts,
      ruinProb,
      ghostPaths: ghostData,
    };
  }

  private async readback(_d: any, staging: any, count: number): Promise<Float32Array> {
    await staging.mapAsync(1);
    const data = new Float32Array(staging.getMappedRange().slice(0, count * 4));
    staging.unmap();
    return data;
  }

  private async readbackU32(_d: any, staging: any, count: number): Promise<Uint32Array> {
    await staging.mapAsync(1);
    const data = new Uint32Array(staging.getMappedRange().slice(0, count * 4));
    staging.unmap();
    return data;
  }
}
