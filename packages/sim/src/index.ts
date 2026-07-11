export type { SimRequest, SimResult, SimBackend } from './types.js';
import { WebGPUBackend } from './gpu-backend.js';
import { CPUBackend } from './cpu-backend.js';
export { WebGPUBackend, CPUBackend };

export async function createSimBackend(): Promise<import('./types.js').SimBackend> {
  const gpu = typeof navigator !== 'undefined' ? (navigator as any).gpu : undefined;
  if (gpu) {
    try {
      const adapter = await gpu.requestAdapter();
      if (adapter) {
        const device = await adapter.requestDevice();
        return new WebGPUBackend(device);
      }
    } catch {
      // fall back to CPU
    }
  }
  return new CPUBackend();
}