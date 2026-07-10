import type { ComponentType } from 'react';

export interface WidgetManifest<Config = any> {
  id: string;
  name: string;
  description: string;
  defaultSize: { w: number; h: number };
  minSize?: { w: number; h: number };
  Component: ComponentType<WidgetProps<Config>>;
}

export interface WidgetProps<Config = any> {
  instanceId: string;
  config: Config;
  setConfig: (patch: Partial<Config>) => void;
  size: { wPx: number; hPx: number };
}

export interface WidgetRegistry {
  byId: Map<string, WidgetManifest>;
  getAll(): WidgetManifest[];
  get(id: string): WidgetManifest | undefined;
}