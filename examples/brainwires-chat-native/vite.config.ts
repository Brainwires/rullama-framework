import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { transform as esbuildTransform } from 'esbuild';

// react-native-css-interop and nativewind ship `dist/doctor.js` containing
// raw JSX (a runtime pragma-check). rollup-commonjs chokes on it during
// build. Pre-transform those specific files with esbuild.
const fixRnCssInteropJsx = (): Plugin => ({
  name: 'fix-rn-css-interop-jsx',
  enforce: 'pre',
  async transform(code, id) {
    if (/(react-native-css-interop|nativewind)\/dist\/doctor\.js$/.test(id)) {
      const result = await esbuildTransform(code, {
        loader: 'jsx',
        sourcefile: id,
        sourcemap: true,
      });
      return { code: result.code, map: result.map };
    }
    return null;
  },
});

// Vite config for the web target. Aliases react-native -> react-native-web
// and runs the nativewind babel preset for className styles. The same dist/
// output is consumed by desktop-linux/ (Tauri).
export default defineConfig({
  plugins: [
    fixRnCssInteropJsx(),
    react({
      // Don't inherit the project's babel.config.js — that file is for Metro
      // and includes react-native-reanimated/plugin (not used on web).
      babel: {
        babelrc: false,
        configFile: false,
        presets: ['nativewind/babel'],
      },
    }),
  ],
  resolve: {
    alias: {
      'react-native': 'react-native-web',
    },
    extensions: [
      '.web.tsx',
      '.web.ts',
      '.web.jsx',
      '.web.js',
      '.tsx',
      '.ts',
      '.jsx',
      '.js',
    ],
  },
  define: {
    __DEV__: JSON.stringify(process.env.NODE_ENV !== 'production'),
    'process.env.NODE_ENV': JSON.stringify(
      process.env.NODE_ENV ?? 'development',
    ),
    global: 'globalThis',
  },
  optimizeDeps: {
    esbuildOptions: {
      loader: { '.js': 'jsx' },
      resolveExtensions: [
        '.web.js',
        '.web.jsx',
        '.web.ts',
        '.web.tsx',
        '.js',
        '.jsx',
        '.ts',
        '.tsx',
      ],
    },
    include: [
      'react-native-web',
      'react-native-css-interop',
      'nativewind',
      'react-dom',
    ],
  },
  server: {
    port: 5173,
    strictPort: false,
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    commonjsOptions: {
      transformMixedEsModules: true,
    },
  },
});
