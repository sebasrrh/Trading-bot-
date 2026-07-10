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

      if (meta && meta.first_t <= from && meta.last_t >= to) {
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