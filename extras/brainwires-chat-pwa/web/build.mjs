#!/usr/bin/env node
// brainwires-chat-pwa web build script.
//
// Responsibilities:
//   1. esbuild-bundle src/boot.js → app.js (esm, sourcemap, es2022).
//      ./pkg/* is left external — wasm-pack output is loaded as a
//      module at runtime, not bundled.
//   2. esbuild-bundle sw.source.js → sw.bundle.js (IIFE, es2022).
//      IIFE because some mobile browsers still ship without
//      type=module SW support; an IIFE works everywhere classic SWs do.
//      sw.bundle.js is a build intermediate (gitignored).
//   3. Patch sw.bundle.js → sw.js by substituting the __SRI_HASHES__
//      placeholder with a JSON object mapping each cacheable static
//      asset to its base64 sha256. sw.js itself is excluded from the
//      table — workers can't verify themselves.
//   4. Emit build-info.js with BUILD_TIME + BUILD_GIT exports.
//
// Modes:
//   node build.mjs            — one-shot build
//   node build.mjs --watch    — esbuild ctx.watch() (no server)
//   node build.mjs --serve    — esbuild ctx.serve() on 127.0.0.1:3000.

import * as esbuild from 'esbuild';
import { createHash } from 'node:crypto';
import {
    readFileSync, writeFileSync, existsSync,
    mkdirSync, readdirSync, copyFileSync, statSync,
} from 'node:fs';
import { execSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const argv = new Set(process.argv.slice(2));
const isWatch = argv.has('--watch');
const isServe = argv.has('--serve');

// ── esbuild config: app bundle ──────────────────────────────────
const appConfig = {
    entryPoints: [join(__dirname, 'src/boot.js')],
    bundle: true,
    outfile: join(__dirname, 'app.js'),
    format: 'esm',
    sourcemap: true,
    target: ['es2022'],
    absWorkingDir: __dirname,
    // wasm-pack output is loaded at runtime via dynamic import; never bundle it.
    external: [
        './pkg/brainwires_chat_pwa.js',
        './pkg/*',
    ],
    logLevel: 'info',
};

// ── esbuild config: local-model worker bundle ──────────────────
// The worker hosts the WASM module on its own thread (Phase 2 of the
// "bright-scroll" plan). It imports `./pkg/brainwires_chat_pwa.js`
// at runtime — same external rule as the app bundle.
const workerConfig = {
    entryPoints: [join(__dirname, 'src', 'local-worker.js')],
    bundle: true,
    outfile: join(__dirname, 'local-worker.js'),
    format: 'esm',
    sourcemap: true,
    target: ['es2022'],
    absWorkingDir: __dirname,
    platform: 'browser',
    external: [
        './pkg/brainwires_chat_pwa.js',
        '../pkg/brainwires_chat_pwa.js',
        './pkg/*',
        '../pkg/*',
    ],
    logLevel: 'info',
};

// ── esbuild config: OPFS writer worker bundle ──────────────────
// Dedicated Worker for zero-copy OPFS writes via FileSystemSyncAccessHandle.
const writerWorkerConfig = {
    entryPoints: [join(__dirname, 'src', 'opfs-writer-worker.js')],
    bundle: true,
    outfile: join(__dirname, 'opfs-writer-worker.js'),
    format: 'esm',
    sourcemap: true,
    target: ['es2022'],
    absWorkingDir: __dirname,
    platform: 'browser',
    logLevel: 'info',
};

// ── esbuild config: service-worker bundle ───────────────────────
// IIFE so the file can be registered as a classic worker on every
// browser that ships ServiceWorker — type=module SWs are broadly
// supported but iOS lagged for a long while; IIFE is a safe floor.
const swConfig = {
    entryPoints: [join(__dirname, 'sw.source.js')],
    bundle: true,
    outfile: join(__dirname, 'sw.bundle.js'),
    format: 'iife',
    sourcemap: false,
    target: ['es2022'],
    absWorkingDir: __dirname,
    platform: 'browser',
    logLevel: 'info',
};

// ── SRI patching ────────────────────────────────────────────────
// Files pinned with SRI hashes. sw.js is excluded — a service worker
// cannot meaningfully verify its own bytes against a hash that lives
// inside it.
//
// The language list mirrors `SUPPORTED_LANGS` in `src/i18n.js`. Keep
// them in sync — a missing catalog here means runtime fetches go to
// the network instead of the cache.
const SUPPORTED_LANGS = [
    'en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'ru', 'uk',
    'cs', 'hu', 'ro', 'el', 'sv', 'da', 'no', 'fi', 'tr', 'ar',
    'he', 'fa', 'ur', 'hi', 'bn', 'ta', 'mr', 'zh-CN', 'zh-TW', 'ja',
    'ko', 'id', 'ms', 'vi', 'th',
];

const STATIC_ASSETS = [
    'index.html',
    'manifest.json',
    'app.js',
    'local-worker.js',
    'opfs-writer-worker.js',
    'styles.css',
    'pkg/brainwires_chat_pwa.js',
    'pkg/brainwires_chat_pwa_bg.wasm',
    'icons/icon-192.png',
    'icons/icon-512.png',
    'vendor/katex/katex.min.js',
    'vendor/rsqlite/pkg/rsqlite_wasm.js',
    'vendor/rsqlite/pkg/rsqlite_wasm_bg.wasm',
    'vendor/rsqlite/dist/worker.js',
    'vendor/rsqlite/dist/worker-proxy.js',
    'vendor/rsqlite/dist/index.js',
    // worker.js dynamically imports `./wasm/rsqlite_wasm.js`; pre-cache
    // those so the SQLite worker boots offline.
    'vendor/rsqlite/dist/wasm/rsqlite_wasm.js',
    'vendor/rsqlite/dist/wasm/rsqlite_wasm_bg.wasm',
    ...SUPPORTED_LANGS.map((code) => `lang/${code}.json`),
];

function sha256Base64(absPath) {
    return createHash('sha256').update(readFileSync(absPath)).digest('base64');
}

function patchServiceWorker() {
    const bundlePath = join(__dirname, 'sw.bundle.js');
    const outPath = join(__dirname, 'sw.js');
    if (!existsSync(bundlePath)) {
        console.warn('  sw.bundle.js missing, skipping sw.js generation');
        return;
    }
    const source = readFileSync(bundlePath, 'utf8');

    const hashes = {};
    const skipped = [];
    for (const rel of STATIC_ASSETS) {
        const abs = join(__dirname, rel);
        if (!existsSync(abs)) { skipped.push(rel); continue; }
        hashes[rel] = `sha256-${sha256Base64(abs)}`;
    }

    const marker = '__SRI_HASHES__';
    if (!source.includes(marker)) {
        throw new Error(`sw.bundle.js missing ${marker} placeholder (esbuild may have stripped it)`);
    }
    const out = source.replace(marker, JSON.stringify(hashes, null, 4));
    writeFileSync(outPath, out);
    const note = skipped.length ? `, ${skipped.length} skipped` : '';
    console.log(`  sw.js generated (${Object.keys(hashes).length} assets hashed${note})`);
}

// Copy KaTeX woff2 fonts to web/fonts/katex/ and emit a generated JS module
// that exports katex.min.css with rewritten font URLs (relative to the served
// web root) as a string. The PWA serves the fonts as static assets; the SW
// will pick them up via a `fonts/katex/` cache pass on first request.
// Stage KaTeX as static assets under vendor/katex/ + fonts under fonts/katex/.
// math.js loads katex.min.js via a <script> tag on first need, so KaTeX never
// lands in app.js — keeping cold-start bundle under 100 KB gz. The CSS is
// inlined as a JS string (small) and rewritten so font URLs point at the
// generated fonts/katex/ directory.
function generateKatexAssets() {
    const distDir = join(__dirname, 'node_modules/katex/dist');
    const fontsSrc = join(distDir, 'fonts');
    const fontsDst = join(__dirname, 'fonts/katex');
    const vendorDst = join(__dirname, 'vendor/katex');
    const cssPath = join(distDir, 'katex.min.css');
    const jsPath = join(distDir, 'katex.min.js');
    const outPath = join(__dirname, 'src/_katex-theme.js');
    if (!existsSync(cssPath) || !existsSync(fontsSrc) || !existsSync(jsPath)) {
        console.warn(`  katex dist missing; emitting empty theme`);
        writeFileSync(outPath, `// generated by build.mjs — do not edit\nexport default '';\n`);
        return;
    }
    mkdirSync(fontsDst, { recursive: true });
    for (const f of readdirSync(fontsSrc)) {
        if (!/\.(woff2|woff|ttf)$/i.test(f)) continue;
        copyFileSync(join(fontsSrc, f), join(fontsDst, f));
    }
    mkdirSync(vendorDst, { recursive: true });
    copyFileSync(jsPath, join(vendorDst, 'katex.min.js'));
    let css = readFileSync(cssPath, 'utf8');
    // KaTeX's CSS references fonts as `url(fonts/KaTeX_Foo.woff2)`. Rewrite
    // to root-relative so the SW + nginx serve them from web/fonts/katex/.
    css = css.replace(/url\(fonts\//g, 'url(./fonts/katex/');
    const escaped = css.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${');
    const body =
        `// generated by build.mjs — do not edit\n` +
        `// source: node_modules/katex/dist/katex.min.css (font URLs rewritten)\n` +
        `export default \`${escaped}\`;\n`;
    writeFileSync(outPath, body);
}

// Stage pdfjs-dist as static assets under vendor/pdfjs/. pdf-text.js does a
// variable-path dynamic import to opt out of esbuild's static analysis and
// fetch the ESM bundle (and its sidecar worker) on demand. Keeps the
// ~640 KB raw pdfjs payload out of cold-start app.js.
function generatePdfjsAssets() {
    const distDir = join(__dirname, 'node_modules/pdfjs-dist/build');
    const dst = join(__dirname, 'vendor/pdfjs');
    const src = ['pdf.min.mjs', 'pdf.worker.min.mjs'];
    if (!existsSync(distDir)) {
        console.warn(`  pdfjs dist missing; skipping`);
        return;
    }
    mkdirSync(dst, { recursive: true });
    for (const f of src) {
        const p = join(distDir, f);
        if (existsSync(p)) copyFileSync(p, join(dst, f));
    }
}

// Read the highlight.js theme CSS and emit a generated JS module that exports
// the stylesheet as a string literal. Done as a pre-build step because
// esbuild's default CSS handling overrides `loader: { '.css': 'text' }` and
// emits a separate app.css instead of inlining. Keeping the source of truth
// in node_modules means the theme upgrades automatically with `npm update`.
function generateHljsTheme() {
    const cssPath = join(__dirname, 'node_modules/highlight.js/styles/github-dark.css');
    const outPath = join(__dirname, 'src/_hljs-theme.js');
    if (!existsSync(cssPath)) {
        console.warn(`  ${cssPath} missing; emitting empty hljs theme`);
        writeFileSync(outPath, `// generated by build.mjs — do not edit\nexport default '';\n`);
        return;
    }
    const css = readFileSync(cssPath, 'utf8');
    const escaped = css.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${');
    const body =
        `// generated by build.mjs — do not edit\n` +
        `// source: node_modules/highlight.js/styles/github-dark.css\n` +
        `export default \`${escaped}\`;\n`;
    writeFileSync(outPath, body);
}

// Copy rsqlite-wasm dist + pkg from the sibling repo into vendor/rsqlite/.
// The WASM binary and JS glue are loaded at runtime, not bundled.
function generateRsqliteAssets() {
    const rsqliteRoot = join(__dirname, '..', '..', '..', '..', 'rsqlite-wasm');
    const distSrc = join(rsqliteRoot, 'js', 'dist');
    // wasm-pack outputs land under js/dist/wasm/ in the current
    // rsqlite-wasm layout; older revisions used a top-level pkg/.
    // Try the new path first, fall back to the old one.
    const pkgSrcCandidates = [join(rsqliteRoot, 'js', 'dist', 'wasm'), join(rsqliteRoot, 'pkg')];
    const pkgSrc = pkgSrcCandidates.find((p) => existsSync(p)) || pkgSrcCandidates[0];
    const distDst = join(__dirname, 'vendor', 'rsqlite', 'dist');
    const pkgDst = join(__dirname, 'vendor', 'rsqlite', 'pkg');
    if (!existsSync(distSrc) || !existsSync(pkgSrc)) {
        console.warn('  rsqlite-wasm dist/pkg missing; skipping vendor copy');
        return;
    }
    mkdirSync(distDst, { recursive: true });
    mkdirSync(pkgDst, { recursive: true });
    for (const f of ['worker.js', 'worker-proxy.js', 'index.js', 'types.js', 'devtools.js']) {
        const p = join(distSrc, f);
        if (existsSync(p)) copyFileSync(p, join(distDst, f));
    }
    // worker.js imports `./wasm/rsqlite_wasm.js` relative to itself, so the
    // wasm glue + binary must live at `dist/wasm/`. Upstream stages them
    // there (next to dist/worker.js); mirror that into vendor/.
    const distWasmSrc = join(distSrc, 'wasm');
    const distWasmDst = join(distDst, 'wasm');
    if (existsSync(distWasmSrc)) {
        mkdirSync(distWasmDst, { recursive: true });
        for (const f of readdirSync(distWasmSrc)) {
            const srcEntry = join(distWasmSrc, f);
            // Copy the inline-snippets dir recursively; everything else is flat.
            if (statSync(srcEntry).isDirectory()) {
                const dstDir = join(distWasmDst, f);
                mkdirSync(dstDir, { recursive: true });
                for (const dir of readdirSync(srcEntry)) {
                    const innerSrc = join(srcEntry, dir);
                    if (statSync(innerSrc).isDirectory()) {
                        const innerDst = join(dstDir, dir);
                        mkdirSync(innerDst, { recursive: true });
                        for (const inner of readdirSync(innerSrc)) {
                            copyFileSync(join(innerSrc, inner), join(innerDst, inner));
                        }
                    } else {
                        copyFileSync(innerSrc, join(dstDir, dir));
                    }
                }
            } else {
                copyFileSync(srcEntry, join(distWasmDst, f));
            }
        }
    }
    for (const f of ['rsqlite_wasm.js', 'rsqlite_wasm_bg.wasm']) {
        const p = join(pkgSrc, f);
        if (existsSync(p)) copyFileSync(p, join(pkgDst, f));
    }
    // Copy wasm-bindgen snippets (inline JS helpers used by the WASM glue).
    const snippetsSrc = join(pkgSrc, 'snippets');
    if (existsSync(snippetsSrc)) {
        const snippetsDst = join(pkgDst, 'snippets');
        mkdirSync(snippetsDst, { recursive: true });
        for (const dir of readdirSync(snippetsSrc)) {
            const srcDir = join(snippetsSrc, dir);
            const dstDir = join(snippetsDst, dir);
            mkdirSync(dstDir, { recursive: true });
            for (const f of readdirSync(srcDir)) {
                copyFileSync(join(srcDir, f), join(dstDir, f));
            }
        }
    }
}

function generateBuildInfo() {
    const ts = new Date().toISOString();
    let sha = 'unknown';
    try { sha = execSync('git rev-parse --short HEAD', { encoding: 'utf8' }).trim(); } catch {}
    // DEV_MODE comes from the launching shell (e.g. `start.sh dev`
    // exports DEV_MODE=true). When unset or anything other than
    // "true", default to false.
    const devMode = process.env.DEV_MODE === 'true';
    const body =
        `// generated by build.mjs — do not edit\n` +
        `export const BUILD_TIME = '${ts}';\n` +
        `export const BUILD_GIT = '${sha}';\n` +
        `export const DEV_MODE = ${devMode};\n`;
    writeFileSync(join(__dirname, 'build-info.js'), body);
    console.log(`  build-info: ${ts} (${sha}) | dev=${devMode}`);
}

async function buildAll() {
    const t0 = performance.now();
    generateHljsTheme();
    generateKatexAssets();
    generatePdfjsAssets();
    generateRsqliteAssets();
    await esbuild.build(appConfig);
    await esbuild.build(workerConfig);
    await esbuild.build(writerWorkerConfig);
    await esbuild.build(swConfig);
    console.log(`  bundled in ${Math.round(performance.now() - t0)}ms`);
    patchServiceWorker();
    generateBuildInfo();
}

// ── Run ─────────────────────────────────────────────────────────
if (isServe) {
    generateHljsTheme();
    generateKatexAssets();
    generatePdfjsAssets();
    generateRsqliteAssets();
    const ctx = await esbuild.context(appConfig);
    const workerCtx = await esbuild.context(workerConfig);
    const writerCtx = await esbuild.context(writerWorkerConfig);
    const swCtx = await esbuild.context(swConfig);
    await ctx.rebuild();
    await workerCtx.rebuild();
    await writerCtx.rebuild();
    await swCtx.rebuild();
    patchServiceWorker();
    generateBuildInfo();
    await ctx.watch();
    await workerCtx.watch();
    await writerCtx.watch();
    await swCtx.watch();
    const server = await ctx.serve({
        host: '127.0.0.1',
        port: 3000,
        servedir: __dirname,
    });
    console.log(`Serving http://${server.host}:${server.port}/`);
} else if (isWatch) {
    generateHljsTheme();
    generateKatexAssets();
    generatePdfjsAssets();
    generateRsqliteAssets();
    const ctx = await esbuild.context(appConfig);
    const workerCtx = await esbuild.context(workerConfig);
    const writerCtx = await esbuild.context(writerWorkerConfig);
    const swCtx = await esbuild.context(swConfig);
    await ctx.rebuild();
    await workerCtx.rebuild();
    await writerCtx.rebuild();
    await swCtx.rebuild();
    patchServiceWorker();
    generateBuildInfo();
    await ctx.watch();
    await workerCtx.watch();
    await writerCtx.watch();
    await swCtx.watch();
    console.log('Watching for changes...');
} else {
    await buildAll();
}
