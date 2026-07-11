import type { WidgetManifest, WidgetRegistry } from './types';

import CandlesWidget from './candles/index.js';
import WatchlistWidget from './watchlist/index.js';
import StatTileWidget from './stat-tile/index.js';
import QuoteStripWidget from './quote-strip/index.js';
import EquityCurveWidget from './equity-curve/index.js';
import DrawdownWidget from './drawdown/index.js';
import RunMetricsWidget from './run-metrics/index.js';
import PaperSummaryWidget from './paper-summary/index.js';

const manifests: WidgetManifest[] = [
  {
    id: 'candles', name: 'Price Chart',
    description: 'Candlestick chart with volume',
    defaultSize: { w: 8, h: 10 },
    Component: CandlesWidget,
  },
  {
    id: 'watchlist', name: 'Watchlist',
    description: 'Symbol list with quotes',
    defaultSize: { w: 4, h: 10 },
    Component: WatchlistWidget,
  },
  {
    id: 'stat-tile', name: 'Stat Tile',
    description: 'Single stat display',
    defaultSize: { w: 3, h: 4 },
    Component: StatTileWidget,
  },
  {
    id: 'quote-strip', name: 'Quote Strip',
    description: 'Horizontal ticker chips',
    defaultSize: { w: 12, h: 2 },
    Component: QuoteStripWidget,
  },
  {
    id: 'equity-curve', name: 'Equity Curve',
    description: 'Strategy equity curve chart',
    defaultSize: { w: 6, h: 8 },
    Component: EquityCurveWidget,
  },
  {
    id: 'drawdown', name: 'Drawdown',
    description: 'Underwater drawdown plot',
    defaultSize: { w: 6, h: 6 },
    Component: DrawdownWidget,
  },
  {
    id: 'run-metrics', name: 'Run Metrics',
    description: 'Strategy performance metrics table',
    defaultSize: { w: 4, h: 8 },
    Component: RunMetricsWidget,
  },
  {
    id: 'paper-summary', name: 'Paper Summary',
    description: 'Paper trading account summary with positions and recent fills',
    defaultSize: { w: 6, h: 8 },
    Component: PaperSummaryWidget,
  },
];

function createRegistry(): WidgetRegistry {
  const byId = new Map<string, WidgetManifest>();
  for (const m of manifests) byId.set(m.id, m);
  return {
    byId,
    getAll: () => Array.from(byId.values()),
    get: (id: string) => byId.get(id),
  };
}

export const widgetRegistry = createRegistry();