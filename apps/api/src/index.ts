import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';
import { TimeframeSchema } from '@tradeboard/core';
import { routeBars, routeQuotes, routeSearch, AllProvidersFailedError } from './providers/router';
import { getCachedBars, cacheBars, getBarsMeta } from './cache/db';
import { InflightDedup } from './lib/inflight-dedup';

const app = new Hono();
const dedup = new InflightDedup();

const corsOrigin = process.env.CORS_ORIGIN || 'http://localhost:5173';
app.use('*', cors({ origin: corsOrigin.split(','), credentials: true }));
app.use('*', logger());

app.get('/api/health', (c) => {
  return c.json({
    status: 'ok',
    providers: { alpaca: false, polygon: false, stooq: true, yahoo: true },
    version: '0.0.0',
  });
});

// A bar isn't "closed" (and thus isn't final/cacheable as complete) until a
// full timeframe period has elapsed. Comparing cache coverage against a raw
// `to=Date.now()` — as this used to — meant `meta.last_t` (the open time of
// the most recent *closed* bar, e.g. yesterday 00:00 UTC for daily) could
// never be >= "right now", so the cache never hit and every request re-fetched
// from the provider. Comparing against the last bar that could plausibly have
// closed fixes that while still refetching once a new bar is due.
function timeframeMs(tf: string): number {
  switch (tf) {
    case '1m': return 60_000;
    case '5m': return 5 * 60_000;
    case '15m': return 15 * 60_000;
    case '1h': return 3_600_000;
    case '1W': return 7 * 86_400_000;
    case '1D':
    default: return 86_400_000;
  }
}

app.get('/api/bars', async (c) => {
  const symbol = (c.req.query('symbol') ?? 'SPY').toUpperCase();
  const tf = TimeframeSchema.parse(c.req.query('timeframe') ?? '1D');
  const from = Number(c.req.query('from') ?? 0);
  const to = Number(c.req.query('to') ?? Date.now());

  const cacheKey = `${symbol}:${tf}:${from}:${to}`;

  try {
    const result = await dedup.dedup(cacheKey, async () => {
      const cached = await getCachedBars(symbol, tf, from, to);
      const meta = await getBarsMeta(symbol, tf);

      const barMs = timeframeMs(tf);
      const lastClosedBar = Math.floor((Date.now() - barMs) / barMs) * barMs;
      const coverageTarget = Math.min(to, lastClosedBar);

      if (meta && meta.first_t <= from && meta.last_t >= coverageTarget) {
        return { source: 'cache', warnings: [] as string[], bars: cached, adjusted: true };
      }

      const { source, data: fresh, warnings } = await routeBars({ symbol, timeframe: tf, from, to });
      if (source && fresh.length > 0) {
        cacheBars(symbol, tf, fresh, source);
      }
      const merged = mergeBars(cached, fresh);
      return { source, warnings, bars: merged, adjusted: true };
    });

    return c.json({
      source: result.source,
      warnings: result.warnings,
      adjusted: result.adjusted,
      bars: result.bars,
      symbol,
      timeframe: tf,
      from,
      to,
    });
  } catch (err) {
    if (err instanceof AllProvidersFailedError) {
      return c.json({ error: 'all data providers failed', warnings: err.warnings, symbol, timeframe: tf }, 502);
    }
    throw err;
  }
});

app.get('/api/quotes', async (c) => {
  const symbols = (c.req.query('symbols') ?? 'SPY').split(',').map(s => s.trim().toUpperCase());
  try {
    const { source, data: quotes, warnings } = await routeQuotes(symbols);
    return c.json({ source, warnings, quotes });
  } catch (err) {
    if (err instanceof AllProvidersFailedError) {
      return c.json({ error: 'all data providers failed', warnings: err.warnings }, 502);
    }
    throw err;
  }
});

app.get('/api/search', async (c) => {
  const q = c.req.query('q') ?? '';
  if (q.length < 1) return c.json([]);
  try {
    return c.json(await routeSearch(q));
  } catch (err) {
    if (err instanceof AllProvidersFailedError) return c.json([]);
    throw err;
  }
});

function mergeBars(cached: any[], fresh: any[]): any[] {
  const map = new Map<number, any>();
  for (const b of cached) map.set(b.t, b);
  for (const b of fresh) map.set(b.t, b);
  return Array.from(map.values()).sort((a, b) => a.t - b.t);
}

export default app;