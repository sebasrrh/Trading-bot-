import GridLayout from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import { useWorkspaceStore } from '../state';
import { widgetRegistry } from '../widgets/registry';
import type { Workspace, WidgetInstance } from '../state/workspace-store';

interface Props { workspace: Workspace; }

function WidgetFrame({ inst }: { inst: WidgetInstance }) {
  const man = widgetRegistry.get(inst.widgetId);
  const setConfig = (patch: any) => useWorkspaceStore.getState().updateConfig(inst.id, patch);
  const remove = () => useWorkspaceStore.getState().removeWidget(inst.id);

  if (!man) return <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>Unknown widget: {inst.widgetId}</div>;

  const setChannel = useWorkspaceStore.getState().setChannel;
  const chLabel = inst.channel ?? '∅';

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--bg-surface-1)', borderRadius: 'var(--radius-m)', border: '1px solid var(--border-hairline)', overflow: 'hidden' }}>
      <div style={{ display: 'flex', alignItems: 'center', padding: '4px 8px', borderBottom: '1px solid var(--border-hairline)', fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)' }}>
        <span>{man.name}</span>
        <div style={{ flex: 1 }} />
        <button onClick={() => setChannel(inst.id, inst.channel === null ? 'A' : inst.channel === 'A' ? 'B' : inst.channel === 'B' ? 'C' : null)}
          style={{ border: 'none', background: 'var(--accent-soft)', color: 'var(--accent)', fontSize: 10, padding: '2px 6px', borderRadius: 'var(--radius-s)', cursor: 'pointer', marginRight: 4 }}>
          {chLabel}
        </button>
        <button onClick={remove} style={{ border: 'none', background: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 14, lineHeight: 1 }}>×</button>
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        <man.Component instanceId={inst.id} config={inst.config} setConfig={setConfig} size={{ wPx: 0, hPx: 0 }} />
      </div>
    </div>
  );
}

export default function DashboardGrid({ workspace }: Props) {
  const setLayouts = useWorkspaceStore((s) => s.setLayouts);
  const instances = Object.values(workspace.widgets);

  const layout = workspace.layouts.lg ?? instances.map((inst, i) => {
    const man = widgetRegistry.get(inst.widgetId);
    const def = man?.defaultSize ?? { w: 4, h: 6 };
    return { i: inst.id, x: (i * def.w) % 12, y: i * 2, w: def.w, h: def.h };
  });

  return (
    <main style={{ flex: 1, overflow: 'auto', padding: 12, background: 'var(--bg-page)' }}>
      {instances.length === 0 ? (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: 14 }}>
          No widgets yet. Use the + Widget button to add one.
        </div>
      ) : (
        <GridLayout
          className="layout"
          layout={layout}
          cols={12}
          rowHeight={32}
          width={1200}
          onLayoutChange={(l) => setLayouts({ lg: l })}
          draggableHandle=".widget-drag-handle"
        >
          {instances.map((inst) => (
            <div key={inst.id} style={{ overflow: 'hidden' }}>
              <WidgetFrame inst={inst} />
            </div>
          ))}
        </GridLayout>
      )}
    </main>
  );
}