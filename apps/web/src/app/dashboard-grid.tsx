import { useEffect, useMemo, useRef, useState } from 'react';
import { Responsive, WidthProvider } from 'react-grid-layout';
import type { Layout, Layouts } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import { useWorkspaceStore } from '../state';
import { widgetRegistry } from '../widgets/registry';
import type { Workspace, WidgetInstance } from '../state/workspace-store';
import type { WidgetProps } from '../widgets/types';

const ResponsiveGridLayout = WidthProvider(Responsive);

// docs/03 §2: lg >=1200px 12 cols, md >=768px 8 cols, sm <768px single column
// (read-only — no drag/resize, just a readable stack on a phone).
const BREAKPOINTS = { lg: 1200, md: 768, sm: 0 } as const;
const COLS = { lg: 12, md: 8, sm: 1 } as const;
type Breakpoint = keyof typeof COLS;

interface Props { workspace: Workspace; }

function WidgetFrame({ inst }: { inst: WidgetInstance }) {
  const man = widgetRegistry.get(inst.widgetId);
  const setConfig = (patch: any) => useWorkspaceStore.getState().updateConfig(inst.id, patch);
  const remove = () => useWorkspaceStore.getState().removeWidget(inst.id);
  const bodyRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<WidgetProps['size']>({ wPx: 0, hPx: 0 });

  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (!box) return;
      // Round to whole px so a fractional layout-engine jiggle during drag
      // doesn't trigger a re-render storm in every widget on the board.
      const wPx = Math.round(box.width);
      const hPx = Math.round(box.height);
      setSize((prev) => (prev.wPx === wPx && prev.hPx === hPx ? prev : { wPx, hPx }));
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  if (!man) return <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>Unknown widget: {inst.widgetId}</div>;

  const setChannel = useWorkspaceStore.getState().setChannel;
  const chLabel = inst.channel ?? '∅';

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--bg-surface-1)', borderRadius: 'var(--radius-m)', border: '1px solid var(--border-hairline)', overflow: 'hidden' }}>
      <div className="widget-drag-handle" style={{ display: 'flex', alignItems: 'center', padding: '4px 8px', borderBottom: '1px solid var(--border-hairline)', fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', cursor: 'grab' }}>
        <span>{man.name}</span>
        <div style={{ flex: 1 }} />
        <button onClick={() => setChannel(inst.id, inst.channel === null ? 'A' : inst.channel === 'A' ? 'B' : inst.channel === 'B' ? 'C' : null)}
          style={{ border: 'none', background: 'var(--accent-soft)', color: 'var(--accent)', fontSize: 10, padding: '2px 6px', borderRadius: 'var(--radius-s)', cursor: 'pointer', marginRight: 4 }}>
          {chLabel}
        </button>
        <button onClick={remove} style={{ border: 'none', background: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 14, lineHeight: 1 }}>×</button>
      </div>
      <div ref={bodyRef} style={{ flex: 1, overflow: 'auto' }}>
        <man.Component instanceId={inst.id} config={inst.config} setConfig={setConfig} size={size} />
      </div>
    </div>
  );
}

// Fills in a layout entry for any widget instance the stored layout doesn't
// know about yet (new widget, or a breakpoint that's never been dragged in).
// lg entries come straight from storage/defaultSize; md/sm are derived from
// lg by clamping into that breakpoint's column count so a 8-wide lg widget
// doesn't overflow an 8-col md grid or a 1-col sm stack.
function buildLayout(bp: Breakpoint, workspace: Workspace, lgResolved: Layout[]): Layout[] {
  const cols = COLS[bp];
  const stored = workspace.layouts[bp];
  if (bp === 'lg') return lgResolved;

  if (stored) {
    const known = new Set(stored.map((l) => l.i));
    const missing = lgResolved.filter((l) => !known.has(l.i));
    return [...stored.filter((l) => workspace.widgets[l.i]), ...missing.map((l) => clampToCols(l, cols))];
  }

  if (bp === 'sm') {
    // Single-column read-only stack, in the same top-to-bottom order as lg.
    const ordered = [...lgResolved].sort((a, b) => a.y - b.y || a.x - b.x);
    let y = 0;
    return ordered.map((l) => {
      const item = { ...l, x: 0, w: 1, y };
      y += l.h;
      return item;
    });
  }

  return lgResolved.map((l) => clampToCols(l, cols));
}

function clampToCols(l: Layout, cols: number): Layout {
  const w = Math.min(l.w, cols);
  const x = Math.min(l.x, cols - w);
  return { ...l, w, x };
}

export default function DashboardGrid({ workspace }: Props) {
  const setLayouts = useWorkspaceStore((s) => s.setLayouts);
  const instances = Object.values(workspace.widgets);
  const [breakpoint, setBreakpoint] = useState<Breakpoint>('lg');

  // Stored lg layout plus synthesized entries for any instance the layout
  // doesn't know about (e.g. workspaces saved before addWidget wrote layout
  // entries, or a widget added since the last save).
  const lgResolved = useMemo(() => {
    const stored = workspace.layouts.lg ?? [];
    const known = new Set(stored.map((l) => l.i));
    const bottom = stored.reduce((m, l) => Math.max(m, l.y + l.h), 0);
    return [
      ...stored.filter((l) => workspace.widgets[l.i]),
      ...instances.filter((inst) => !known.has(inst.id)).map((inst, i) => {
        const def = widgetRegistry.get(inst.widgetId)?.defaultSize ?? { w: 4, h: 6 };
        return { i: inst.id, x: 0, y: bottom + i * def.h, w: def.w, h: def.h };
      }),
    ];
  }, [workspace, instances]);

  const layouts: Layouts = useMemo(() => ({
    lg: buildLayout('lg', workspace, lgResolved),
    md: buildLayout('md', workspace, lgResolved),
    sm: buildLayout('sm', workspace, lgResolved),
  }), [workspace, lgResolved]);

  const readOnly = breakpoint === 'sm';

  return (
    <main style={{ flex: 1, overflow: 'auto', padding: 12, background: 'var(--bg-page)' }}>
      {instances.length === 0 ? (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: 14 }}>
          No widgets yet. Use the + Widget button to add one.
        </div>
      ) : (
        <ResponsiveGridLayout
          className="layout"
          layouts={layouts}
          breakpoints={BREAKPOINTS}
          cols={COLS}
          rowHeight={32}
          onBreakpointChange={(bp: Breakpoint) => setBreakpoint(bp)}
          onLayoutChange={(_current: Layout[], all: Layouts) => setLayouts(all)}
          draggableHandle=".widget-drag-handle"
          draggableCancel="button"
          isDraggable={!readOnly}
          isResizable={!readOnly}
        >
          {instances.map((inst) => (
            <div key={inst.id} style={{ overflow: 'hidden' }}>
              <WidgetFrame inst={inst} />
            </div>
          ))}
        </ResponsiveGridLayout>
      )}
    </main>
  );
}
