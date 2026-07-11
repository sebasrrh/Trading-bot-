import { create } from 'zustand';
import type { Layout } from 'react-grid-layout';

export interface WidgetInstance {
  id: string;
  widgetId: string;
  config: any;
  channel: 'A' | 'B' | 'C' | null;
}

export interface Workspace {
  version: number;
  name: string;
  layouts: Record<string, Layout[]>;
  widgets: Record<string, WidgetInstance>;
}

interface WorkspaceState {
  workspaces: Workspace[];
  active: number;
  addWidget: (widgetId: string, config?: any, size?: { w: number; h: number }) => void;
  removeWidget: (instanceId: string) => void;
  updateConfig: (instanceId: string, patch: any) => void;
  setLayouts: (layouts: Record<string, Layout[]>) => void;
  setChannel: (instanceId: string, ch: 'A' | 'B' | 'C' | null) => void;
  switchWorkspace: (i: number) => void;
}

const KEY = 'tradeboard.workspaces.v1';
// Bump whenever the built-in presets change shape (new/renamed widgets, new
// layout) so returning users pick up the update instead of being stuck on
// whatever was in localStorage the first time they opened the app.
const PRESET_VERSION = 2;
const PRESET_NAMES = new Set(['Markets', 'Strategy Lab', 'Risk']);

function wid(id: string): string { return `${id}-${Math.random().toString(36).slice(2, 6)}`; }

// Layout entries and widget instances must share ids, or the grid renders nothing.
function preset(name: string, items: Array<{ widgetId: string; x: number; y: number; w: number; h: number; channel?: 'A' | 'B' | 'C' }>): Workspace {
  const layouts: Layout[] = [];
  const widgets: Record<string, WidgetInstance> = {};
  items.forEach((it, n) => {
    const id = `${it.widgetId}-${n}`;
    layouts.push({ i: id, x: it.x, y: it.y, w: it.w, h: it.h, static: false });
    widgets[id] = { id, widgetId: it.widgetId, config: {}, channel: it.channel ?? 'A' };
  });
  return { version: PRESET_VERSION, name, layouts: { lg: layouts }, widgets };
}

const presets: Workspace[] = [
  preset('Markets', [
    { widgetId: 'quote-strip', x: 0, y: 0, w: 12, h: 2 },
    { widgetId: 'candles', x: 0, y: 2, w: 8, h: 10 },
    { widgetId: 'watchlist', x: 8, y: 2, w: 4, h: 10 },
    { widgetId: 'stat-tile', x: 0, y: 12, w: 3, h: 4 },
  ]),
  preset('Strategy Lab', [
    { widgetId: 'candles', x: 0, y: 0, w: 6, h: 10 },
    { widgetId: 'equity-curve', x: 6, y: 0, w: 6, h: 8 },
    { widgetId: 'drawdown', x: 6, y: 8, w: 6, h: 6 },
    { widgetId: 'run-metrics', x: 0, y: 10, w: 12, h: 6 },
  ]),
  preset('Risk', [
    { widgetId: 'candles', x: 0, y: 0, w: 12, h: 10 },
  ]),
];

// Refreshes any stored *built-in* preset workspace that predates the current
// PRESET_VERSION (matched by name), so shipping a new preset widget actually
// reaches users with existing localStorage. Workspaces the user renamed or
// created themselves are left untouched.
function migrate(stored: Workspace[]): Workspace[] {
  const byName = new Map(presets.map((p) => [p.name, p]));
  return stored.map((w) => {
    if (PRESET_NAMES.has(w.name) && (w.version ?? 0) < PRESET_VERSION) {
      return structuredClone(byName.get(w.name) ?? w);
    }
    return w;
  });
}

function load(): Workspace[] {
  try {
    const r = localStorage.getItem(KEY);
    if (!r) return structuredClone(presets);
    return migrate(JSON.parse(r));
  } catch {
    return structuredClone(presets);
  }
}
function save(ws: Workspace[]) { try { localStorage.setItem(KEY, JSON.stringify(ws)); } catch { /* storage full or unavailable */ } }

export const useWorkspaceStore = create<WorkspaceState>((set) => {
  const saved = load();
  return {
    workspaces: saved,
    active: 0,
    addWidget: (widgetId, config, size) => set((s) => {
      const id = wid(widgetId);
      const w = { ...s.workspaces[s.active]! };
      w.widgets = { ...w.widgets, [id]: { id, widgetId, config: config ?? {}, channel: 'A' } };
      // A widget without a layout entry never renders — append one at the bottom.
      const { w: gw, h: gh } = size ?? { w: 4, h: 6 };
      const lg = w.layouts.lg ?? [];
      const bottom = lg.reduce((m, l) => Math.max(m, l.y + l.h), 0);
      w.layouts = { ...w.layouts, lg: [...lg, { i: id, x: 0, y: bottom, w: gw, h: gh, static: false }] };
      const ws = [...s.workspaces]; ws[s.active] = w;
      save(ws);
      return { workspaces: ws };
    }),
    removeWidget: (instanceId) => set((s) => {
      const w = { ...s.workspaces[s.active]! };
      delete w.widgets[instanceId];
      const layouts = { ...w.layouts };
      for (const bp of Object.keys(layouts)) {
        layouts[bp] = layouts[bp]!.filter((l: Layout) => l.i !== instanceId);
      }
      w.layouts = layouts;
      const ws = [...s.workspaces]; ws[s.active] = w;
      save(ws);
      return { workspaces: ws };
    }),
    updateConfig: (instanceId, patch) => set((s) => {
      const w = { ...s.workspaces[s.active]! };
      const existing = w.widgets[instanceId]!;
      w.widgets[instanceId] = { ...existing, config: { ...existing.config, ...patch } };
      const ws = [...s.workspaces]; ws[s.active] = w;
      save(ws);
      return { workspaces: ws };
    }),
    setLayouts: (layouts) => set((s) => {
      const w = { ...s.workspaces[s.active]! };
      w.layouts = layouts;
      const ws = [...s.workspaces]; ws[s.active] = w;
      save(ws);
      return { workspaces: ws };
    }),
    setChannel: (instanceId, ch) => set((s) => {
      const w = { ...s.workspaces[s.active]! };
      w.widgets[instanceId] = { ...w.widgets[instanceId]!, channel: ch };
      const ws = [...s.workspaces]; ws[s.active] = w;
      save(ws);
      return { workspaces: ws };
    }),
    switchWorkspace: (i) => set({ active: i }),
  };
});