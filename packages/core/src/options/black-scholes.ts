function normPdf(x: number): number {
  return Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
}

function cnd(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const absX = Math.abs(x);
  const t = 1 / (1 + p * absX);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX / 2);
  return 0.5 * (1 + sign * y);
}

function d1d2(S: number, K: number, T: number, r: number, sigma: number) {
  const d1 = (Math.log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  return { d1, d2 };
}

export function bsPrice(
  S: number, K: number, T: number, r: number, sigma: number, type: 'call' | 'put',
): number {
  if (T <= 0) return Math.max(0, type === 'call' ? S - K : K - S);
  const { d1, d2 } = d1d2(S, K, T, r, sigma);
  if (type === 'call') return S * cnd(d1) - K * Math.exp(-r * T) * cnd(d2);
  return K * Math.exp(-r * T) * cnd(-d2) - S * cnd(-d1);
}

export function bsDelta(
  S: number, K: number, T: number, r: number, sigma: number, type: 'call' | 'put',
): number {
  if (T <= 0) return type === 'call' ? (S > K ? 1 : 0) : (S < K ? -1 : 0);
  const { d1 } = d1d2(S, K, T, r, sigma);
  if (type === 'call') return cnd(d1);
  return cnd(d1) - 1;
}

export function bsGamma(S: number, K: number, T: number, r: number, sigma: number): number {
  if (T <= 0 || sigma <= 0) return 0;
  const { d1 } = d1d2(S, K, T, r, sigma);
  return normPdf(d1) / (S * sigma * Math.sqrt(T));
}

export function bsVega(S: number, K: number, T: number, r: number, sigma: number): number {
  if (T <= 0 || sigma <= 0) return 0;
  const { d1 } = d1d2(S, K, T, r, sigma);
  return S * normPdf(d1) * Math.sqrt(T);
}

export function bsTheta(
  S: number, K: number, T: number, r: number, sigma: number, type: 'call' | 'put',
): number {
  if (T <= 0) return 0;
  const { d1, d2 } = d1d2(S, K, T, r, sigma);
  const pdf = normPdf(d1);
  const term1 = -S * pdf * sigma / (2 * Math.sqrt(T));
  if (type === 'call') return term1 - r * K * Math.exp(-r * T) * cnd(d2);
  return term1 + r * K * Math.exp(-r * T) * cnd(-d2);
}

export function bsRho(
  S: number, K: number, T: number, r: number, sigma: number, type: 'call' | 'put',
): number {
  if (T <= 0) return 0;
  const { d1: _d1, d2 } = d1d2(S, K, T, r, sigma);
  void _d1;
  if (type === 'call') return K * T * Math.exp(-r * T) * cnd(d2);
  return -K * T * Math.exp(-r * T) * cnd(-d2);
}

export function bsGreeks(
  S: number, K: number, T: number, r: number, sigma: number, type: 'call' | 'put',
) {
  return {
    delta: bsDelta(S, K, T, r, sigma, type),
    gamma: bsGamma(S, K, T, r, sigma),
    theta: bsTheta(S, K, T, r, sigma, type),
    vega: bsVega(S, K, T, r, sigma),
    rho: bsRho(S, K, T, r, sigma, type),
  };
}

export function impliedVol(
  price: number, S: number, K: number, T: number, r: number, type: 'call' | 'put',
): number | null {
  const intrinsic = type === 'call' ? Math.max(0, S - K) : Math.max(0, K - S);
  if (price < intrinsic) return null;
  let sigma = 0.3;
  for (let i = 0; i < 100; i++) {
    const p = bsPrice(S, K, T, r, sigma, type);
    const vega = bsVega(S, K, T, r, sigma);
    if (Math.abs(p - price) < 1e-8) return sigma;
    if (vega < 1e-12) return null;
    sigma = Math.max(0.001, Math.min(5, sigma - (p - price) / (vega * 100)));
  }
  return sigma;
}