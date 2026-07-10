import eslintConfigPrettier from 'eslint-config-prettier';

export default [
  { ignores: ['**/dist/**', '**/node_modules/**', '**/*.d.ts'] },
  eslintConfigPrettier,
];