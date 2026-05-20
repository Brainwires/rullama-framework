// brainwires-chat-pwa — code syntax highlighting
//
// Lazy-loads highlight.js core + a curated language set on first call.
// One github-dark stylesheet, injected once via <style>. Codeblocks keep
// their dark background regardless of UI theme so a single hljs theme
// works in both light and dark modes.
//
// Highlighting runs on completed code blocks only — we deliberately skip
// it during streaming so the colorizer doesn't re-tokenize the entire
// block on every chunk. Mid-stream code is still styled (monospace, dark
// bg) by `.codeblock pre` in styles.css, just uncolored.

import hljs from 'highlight.js/lib/core';
import javascript from 'highlight.js/lib/languages/javascript';
import typescript from 'highlight.js/lib/languages/typescript';
import python from 'highlight.js/lib/languages/python';
import bash from 'highlight.js/lib/languages/bash';
import json from 'highlight.js/lib/languages/json';
import yaml from 'highlight.js/lib/languages/yaml';
import rust from 'highlight.js/lib/languages/rust';
import go from 'highlight.js/lib/languages/go';
import sql from 'highlight.js/lib/languages/sql';
import css from 'highlight.js/lib/languages/css';
import xml from 'highlight.js/lib/languages/xml';
import markdown from 'highlight.js/lib/languages/markdown';
import diff from 'highlight.js/lib/languages/diff';
// Generated at build time from node_modules/highlight.js/styles/github-dark.css
// (see build.mjs:generateHljsTheme). esbuild's CSS path emits a separate file
// instead of inlining as text, so we route the stylesheet through a generated
// JS module. The file is gitignored.
import themeCss from './_hljs-theme.js';

let _registered = false;
let _styleInjected = false;

function ensureRegistered() {
    if (_registered) return;
    hljs.registerLanguage('javascript', javascript);
    hljs.registerLanguage('js', javascript);
    hljs.registerLanguage('typescript', typescript);
    hljs.registerLanguage('ts', typescript);
    hljs.registerLanguage('python', python);
    hljs.registerLanguage('py', python);
    hljs.registerLanguage('bash', bash);
    hljs.registerLanguage('sh', bash);
    hljs.registerLanguage('shell', bash);
    hljs.registerLanguage('json', json);
    hljs.registerLanguage('yaml', yaml);
    hljs.registerLanguage('yml', yaml);
    hljs.registerLanguage('rust', rust);
    hljs.registerLanguage('rs', rust);
    hljs.registerLanguage('go', go);
    hljs.registerLanguage('sql', sql);
    hljs.registerLanguage('css', css);
    hljs.registerLanguage('html', xml);
    hljs.registerLanguage('xml', xml);
    hljs.registerLanguage('markdown', markdown);
    hljs.registerLanguage('md', markdown);
    hljs.registerLanguage('diff', diff);
    _registered = true;
}

function ensureStyleInjected() {
    if (_styleInjected) return;
    if (typeof document === 'undefined') return;
    const style = document.createElement('style');
    style.id = 'bw-hljs-theme';
    style.textContent = themeCss;
    document.head.appendChild(style);
    _styleInjected = true;
}

/**
 * Highlight every `<pre><code class="language-*">` element inside `rootEl`.
 * Idempotent — code blocks already highlighted (marked with `data-bw-hl`) are
 * skipped on re-call. Safe to invoke after every full-bubble render.
 *
 * @param {ParentNode | null | undefined} rootEl
 */
export function highlightWithin(rootEl) {
    if (!rootEl || typeof rootEl.querySelectorAll !== 'function') return;
    ensureRegistered();
    ensureStyleInjected();
    const blocks = rootEl.querySelectorAll('pre code[class*="language-"]:not([data-bw-hl])');
    for (const block of blocks) {
        try {
            hljs.highlightElement(block);
            block.setAttribute('data-bw-hl', '1');
        } catch (e) {
            // Unknown language or malformed code — leave the block as plain
            // text. Highlight failures are visual-only and not worth toasting.
            console.warn('[bw] highlight failed:', e && e.message ? e.message : e);
        }
    }
}
