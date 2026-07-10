import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';
import eslintConfigPrettier from 'eslint-config-prettier';

export default tseslint.config(
  { ignores: ['**/dist/**', '**/node_modules/**', '**/*.d.ts'] },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  eslintConfigPrettier,
  {
    rules: {
      // The codebase currently leans on `any` at widget/config boundaries;
      // tighten these once zod schemas land there (docs/03 §3).
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    },
  },
);
