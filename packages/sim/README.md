# @tradeboard/sim

Monte Carlo engine: `SimBackend` interface with a WebGPU implementation (WGSL kernels in `src/wgsl/`) and a CPU Web-Worker fallback sharing identical Philox RNG streams.

- Spec: [docs/07-monte-carlo-webgpu.md](../../docs/07-monte-carlo-webgpu.md)
- Modes: bootstrap resampling of backtest returns, GBM/jump-diffusion paths.
- CPU/GPU parity and statistical golden tests are CI-enforced.
