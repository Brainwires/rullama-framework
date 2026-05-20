// brainwires-chat-pwa — math rendering via KaTeX
//
// Walks text nodes in the rendered bubble and replaces $$...$$ (display)
// and $...$ (inline) with KaTeX HTML. Skips text inside <code>, <pre>, and
// already-rendered .katex elements so dollar signs in code blocks aren't
// hijacked.
//
// True lazy loading: KaTeX itself is not imported here. On first call we
// inject `<script src="./vendor/katex/katex.min.js">` (UMD build, exposes
// window.katex) and await its load event. The cold-start app.js bundle
// stays free of the ~280 KB KaTeX runtime — only sessions that contain
// math pay for it.

import themeCss from './_katex-theme.js';

const VENDOR_URL = './vendor/katex/katex.min.js';

let _styleInjected = false;
let _katexPromise = null;

function ensureStyleInjected() {
    if (_styleInjected) return;
    if (typeof document === 'undefined') return;
    const style = document.createElement('style');
    style.id = 'bw-katex-theme';
    style.textContent = themeCss;
    document.head.appendChild(style);
    _styleInjected = true;
}

function loadKatex() {
    if (_katexPromise) return _katexPromise;
    _katexPromise = new Promise((resolve, reject) => {
        if (typeof document === 'undefined') {
            reject(new Error('no document'));
            return;
        }
        if (window.katex && typeof window.katex.renderToString === 'function') {
            resolve(window.katex);
            return;
        }
        const script = document.createElement('script');
        script.src = VENDOR_URL;
        script.async = true;
        script.onload = () => {
            if (window.katex && typeof window.katex.renderToString === 'function') {
                resolve(window.katex);
            } else {
                reject(new Error('katex loaded but window.katex is missing'));
            }
        };
        script.onerror = () => reject(new Error(`failed to load ${VENDOR_URL}`));
        document.head.appendChild(script);
    });
    return _katexPromise;
}

const SKIP_PARENTS = new Set(['CODE', 'PRE', 'SCRIPT', 'STYLE', 'TEXTAREA']);

function shouldSkipNode(node) {
    let p = node.parentNode;
    while (p && p.nodeType === 1) {
        if (SKIP_PARENTS.has(p.tagName)) return true;
        if (p.classList && p.classList.contains('katex')) return true;
        p = p.parentNode;
    }
    return false;
}

// Match $$...$$ first (display), then $...$ (inline). Both forbid newlines
// from spanning blocks. Leading dollar must not be backslash-escaped.
const DISPLAY_RE = /(?<!\\)\$\$([^$\n][^$]*?)\$\$/g;
const INLINE_RE = /(?<!\\)\$([^$\n]+?)\$/g;

function renderOne(katex, tex, displayMode) {
    try {
        return katex.renderToString(tex, {
            displayMode,
            throwOnError: false,
            output: 'html',
            strict: 'ignore',
        });
    } catch (e) {
        // KaTeX errors with throwOnError:false should not reach here, but
        // defend against runtime surprises (e.g. macros that fail in strict
        // mode). Returning null leaves the original text in place — the user
        // sees raw TeX rather than a broken page.
        console.warn('[bw] katex render failed:', e && e.message ? e.message : e);
        return null;
    }
}

function processText(katex, node) {
    const text = node.nodeValue;
    if (!text || (text.indexOf('$') < 0)) return;

    // Build a list of {start, end, html} replacements. Display $$ takes
    // precedence over inline $.
    const repls = [];
    DISPLAY_RE.lastIndex = 0;
    let m;
    while ((m = DISPLAY_RE.exec(text)) !== null) {
        const html = renderOne(katex, m[1].trim(), true);
        if (html) repls.push({ start: m.index, end: m.index + m[0].length, html });
    }
    INLINE_RE.lastIndex = 0;
    while ((m = INLINE_RE.exec(text)) !== null) {
        const a = m.index, b = a + m[0].length;
        if (repls.some((r) => a < r.end && b > r.start)) continue;
        const html = renderOne(katex, m[1].trim(), false);
        if (html) repls.push({ start: a, end: b, html });
    }
    if (repls.length === 0) return;

    repls.sort((a, b) => a.start - b.start);
    const frag = document.createDocumentFragment();
    let cursor = 0;
    for (const r of repls) {
        if (r.start > cursor) {
            frag.appendChild(document.createTextNode(text.slice(cursor, r.start)));
        }
        const span = document.createElement('span');
        span.innerHTML = r.html;
        frag.appendChild(span);
        cursor = r.end;
    }
    if (cursor < text.length) {
        frag.appendChild(document.createTextNode(text.slice(cursor)));
    }
    node.parentNode.replaceChild(frag, node);
}

/**
 * Find $...$ / $$...$$ in text nodes within `rootEl` and replace each with
 * KaTeX-rendered HTML. Idempotent on already-rendered nodes (the SKIP_PARENTS
 * walk steps over `.katex` subtrees). Async because KaTeX is loaded on demand.
 *
 * @param {ParentNode | null | undefined} rootEl
 * @returns {Promise<void>}
 */
export async function renderMathWithin(rootEl) {
    if (!rootEl || typeof document === 'undefined') return;
    ensureStyleInjected();
    let katex;
    try { katex = await loadKatex(); }
    catch (e) {
        console.warn('[bw] katex load failed:', e && e.message ? e.message : e);
        return;
    }
    const walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_TEXT, null);
    const candidates = [];
    let n;
    while ((n = walker.nextNode())) {
        if (!shouldSkipNode(n)) candidates.push(n);
    }
    for (const node of candidates) processText(katex, node);
}
