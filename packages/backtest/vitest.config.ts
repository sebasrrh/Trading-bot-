import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    passWithNoTests: true,
  },
  resolve: {
    alias: {
      '@tradeboard/core': path.resolve(__dirname, '../core/src'),
      '@tradeboard/strategies': path.resolve(__dirname, '../strategies/src'),
    },
  },
});