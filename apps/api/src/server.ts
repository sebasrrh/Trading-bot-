import app from './index.js';
import { serve } from '@hono/node-server';

const port = 8787;
console.log(`api running on http://localhost:${port}`);

serve({ fetch: app.fetch, port, hostname: '0.0.0.0' });