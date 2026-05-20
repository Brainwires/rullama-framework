import js from '@eslint/js';
import globals from 'globals';

export default [
  {
    // wasm-pack output and build artifacts — generated, don't lint
    ignores: [
      'pkg/**',
      'node_modules/**',
      'app.js',
      'app.js.map',
      // Bundled worker output at the web root — NOT src/*.js sources.
      'local-worker.js',
      'local-worker.js.map',
      'opfs-writer-worker.js',
      'opfs-writer-worker.js.map',
      'sw.js',
      'sw.bundle.js',
      'build-info.js',
      // Generated CSS-as-JS modules.
      'src/_hljs-theme.js',
      'src/_katex-theme.js',
      // Third-party bundles served as static assets (KaTeX UMD).
      'vendor/**',
    ],
  },
  js.configs.recommended,
  {
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: 'module',
      globals: {
        ...globals.browser,
        ...globals.serviceworker,
        // wasm-pack output uses these on init paths:
        BigInt: 'readonly',
      },
    },
    rules: {
      'no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      'no-undef': 'error',
      eqeqeq: ['error', 'always', { null: 'ignore' }],
      'no-var': 'error',
      'prefer-const': 'error',
      'no-implicit-globals': 'error',
      'no-shadow': 'error',
      'no-fallthrough': 'error',
      'no-console': 'off',
      'no-empty': ['error', { allowEmptyCatch: true }],
    },
  },
  {
    // Service worker context — has self / clients / caches globals.
    // `__SRI_HASHES__` is a build-time placeholder substituted by build.mjs.
    files: ['sw.source.js'],
    languageOptions: {
      globals: {
        ...globals.serviceworker,
        __SRI_HASHES__: 'readonly',
      },
    },
  },
  {
    // Dedicated Web Worker context (Phase 2 of bright-scroll). Has
    // `self` / `postMessage` / `caches` but NOT the DOM. We deliberately
    // exclude `globals.browser` so accidental DOM usage trips no-undef.
    files: ['src/local-worker.js'],
    languageOptions: {
      globals: {
        ...globals.worker,
      },
    },
  },
  {
    // Test files — Node test runner globals
    files: ['tests/**/*.{js,mjs}'],
    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
  },
  {
    // Build scripts — Node
    files: ['build.mjs'],
    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
  },
];
