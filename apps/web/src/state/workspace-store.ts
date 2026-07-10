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
  addWidget: (widgetId: string, config?: any) => void;
  removeWidget: (instanceId: string) => void;
  updateConfig: (instanceId: string, patch: any) => void;
  setLayouts: (layouts: Record<string, Layout[]>) => void;
  setChannel: (instanceId: string, ch: 'A' | 'B' | 'C' | null) => void;
  switchWorkspace: (i: number) => void;
}

const KEY = 'tradeboard.workspaces.v1';

function wid(id: string): string { return `${id}-${Math.random().toString(36).slice(2, 6)}`; }

const presets: Workspace[] = [
  {
    version: 1, name: 'Markets',
    layouts: { lg: [
      { i: wid('quote-strip'), x: 0, y: 0, w: 12, h: 2, static: false },
      { i: wid('candles'), x: 0, y: 2, w: 8, h: 10, static: false },
      { i: wid('watchlist'), x: 8, y: 2, w: 4, h: 10, static: false },
      { i: wid('stat-tile'), x: 0, y: 12, w: 3, h: 4, static: false },
    ]},
    widgets: {},
  },
  {
    version: 1, name: 'Strategy Lab',
    layouts: { lg: [
      { i: wid('candles'), x: 0, y: 0, w: 8, h: 10, static: false },
      { i: wid('watchlist'), x: 8, y: 0, w: 4, h: 10, static: false },
    ]},
    widgets: {},
  },
  {
    version: 1, name: 'Risk',
    layouts: { lg: [
      { i: wid('candles'), x: 0, y: 0, w: 12, h: 10, static: false },
    ]},
    widgets: {},
  },
];

function load(): Workspace[] {
  try { const r = localStorage.getItem(KEY); return r ? JSON.parse(r) : structuredClone(presets); } catch { return structuredClone(presets); }
}
function save(ws: Workspace[]) { try { localStorage.setItem(KEY, JSON.stringify(ws)); } catch { } }

export const useWorkspaceStore = create<WorkspaceState>((set) => {
  const saved = load();
  return {
    workspaces: saved,
    active: 0,
    addWidget: (widgetId, config) => set((s) => {
      const id = wid(widgetId);
      const w = { ...s.workspaces[s.active]! };
      w.widgets[id] = { id, widgetId, config: config ?? {}, channel: null };
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