import initSqlJs from 'sql.js';
import type { Database } from 'sql.js';
import type { Bar } from '@tradeboard/core';
import { BarSchema } from '@tradeboard/core';
import type { ProviderId } from '../providers/types';
import { existsSync, mkdirSync, writeFileSync, readFileSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DB_PATH = process.env.CACHE_DB_PATH || resolve(__dirname, '../../data/cache.db');

let db: Database | null = null;
let initPromise: Promise<void> | null = null;

async function getDb(): Promise<Database> {
  if (!initPromise) {
    initPromise = (async () => {
      const dir = dirname(DB_PATH);
      if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
      const SQL = await initSqlJs();

      let existingBuffer: Buffer | undefined;
      try { existingBuffer = readFileSync(DB_PATH); } catch { /* no cache file yet */ }

      db = new SQL.Database(existingBuffer);
      db.run('PRAGMA journal_mode=WAL');
      db.run(`CREATE TABLE IF NOT EXISTS bars (
        symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
        t INTEGER NOT NULL, o REAL, h REAL, l REAL, c REAL, v REAL,
        source TEXT NOT NULL, adjusted INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (symbol, timeframe, t)
      )`);
      db.run(`CREATE TABLE IF NOT EXISTS meta (
        symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
        first_t INTEGER, last_t INTEGER, fetched_at INTEGER,
        PRIMARY KEY (symbol, timeframe)
      )`);
    })();
  }
  await initPromise;
  return db!;
}

function saveDb(): void {
  if (!db) return;
  const data = db.export();
  writeFileSync(DB_PATH, Buffer.from(data));
}

export async function getCachedBars(symbol: string, timeframe: string, from: number, to: number): Promise<Bar[]> {
  const d = await getDb();
  const stmt = d.prepare('SELECT t, o, h, l, c, v FROM bars WHERE symbol = ? AND timeframe = ? AND t >= ? AND t <= ? ORDER BY t');
  stmt.bind([symbol, timeframe, from, to]);
  const bars: Bar[] = [];
  while (stmt.step()) {
    const r = stmt.getAsObject() as any;
    bars.push(BarSchema.parse({ t: Number(r.t), o: Number(r.o), h: Number(r.h), l: Number(r.l), c: Number(r.c), v: Number(r.v) }));
  }
  stmt.free();
  return bars;
}

export async function cacheBars(symbol: string, timeframe: string, bars: Bar[], source: ProviderId): Promise<void> {
  if (bars.length === 0) return;
  const d = await getDb();
  d.run('BEGIN TRANSACTION');
  try {
    for (const bar of bars) {
      d.run('INSERT OR REPLACE INTO bars (symbol, timeframe, t, o, h, l, c, v, source, adjusted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)',
        [symbol, timeframe, bar.t, bar.o, bar.h, bar.l, bar.c, bar.v, source]);
    }
    const first = bars[0]!.t;
    const last = bars[bars.length - 1]!.t;
    const selStmt = d.prepare('SELECT first_t, last_t FROM meta WHERE symbol = ? AND timeframe = ?');
    selStmt.bind([symbol, timeframe]);
    if (selStmt.step()) {
      const row = selStmt.getAsObject() as any;
      selStmt.free();
      d.run('UPDATE meta SET first_t = ?, last_t = ?, fetched_at = ? WHERE symbol = ? AND timeframe = ?',
        [Math.min(Number(row.first_t), first), Math.max(Number(row.last_t), last), Date.now(), symbol, timeframe]);
    } else {
      selStmt.free();
      d.run('INSERT INTO meta (symbol, timeframe, first_t, last_t, fetched_at) VALUES (?, ?, ?, ?, ?)',
        [symbol, timeframe, first, last, Date.now()]);
    }
    d.run('COMMIT');
  } catch (e) {
    d.run('ROLLBACK');
    throw e;
  }
  saveDb();
}

export async function getBarsMeta(symbol: string, timeframe: string): Promise<{ first_t: number; last_t: number } | null> {
  const d = await getDb();
  const stmt = d.prepare('SELECT first_t, last_t FROM meta WHERE symbol = ? AND timeframe = ?');
  stmt.bind([symbol, timeframe]);
  if (stmt.step()) {
    const r = stmt.getAsObject() as any;
    stmt.free();
    return { first_t: Number(r.first_t), last_t: Number(r.last_t) };
  }
  stmt.free();
  return null;
}