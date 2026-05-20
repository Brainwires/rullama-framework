// brainwires-chat-pwa — markdown → safe HTML
//
// marked produces HTML; DOMPurify scrubs anything that isn't on our allowlist.
// Custom renderers preserve the existing `<div class="codeblock">` wrapper so
// the copy button (wired via [data-bw-copy] event delegation in ui-chat.js)
// and the existing CSS keep working without changes.
//
// In node tests, DOMPurify is unavailable (no `window`); we fall back to
// returning marked's raw output. The renderer-only behavior is the part
// worth unit-testing here. Sanitization is verified by manual browser test.

import { marked } from 'marked';
import DOMPurify from 'dompurify';
import { t } from './i18n.js';
import { escapeHtml } from './utils.js';

const renderer = new marked.Renderer();

renderer.code = function code(token) {
    const raw = typeof token === 'object' ? token.text : token;
    const lang = (typeof token === 'object' ? (token.lang || '') : (arguments[1] || '')).trim();
    const langCls = lang ? ` class="language-${escapeHtml(lang)}"` : '';
    const escaped = escapeHtml(raw);
    const copyLabel = escapeHtml(t('chat.copy'));
    return `<div class="codeblock">`
        + `<button type="button" class="codeblock-copy" aria-label="${copyLabel}" data-bw-copy="1">${copyLabel}</button>`
        + `<pre><code${langCls}>${escaped}</code></pre>`
        + `</div>`;
};

renderer.link = function link(token) {
    const href = typeof token === 'object' ? token.href : token;
    const title = typeof token === 'object' ? token.title : arguments[1];
    const text = typeof token === 'object' ? this.parser.parseInline(token.tokens) : arguments[2];
    const titleAttr = title ? ` title="${escapeHtml(title)}"` : '';
    return `<a href="${escapeHtml(href || '')}"${titleAttr} target="_blank" rel="noopener noreferrer">${text}</a>`;
};

marked.setOptions({
    gfm: true,
    breaks: true,
    pedantic: false,
    renderer,
});

const PURIFY_CONFIG = {
    ADD_ATTR: ['data-bw-copy', 'target'],
    ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto|tel|data:image\/[a-z+.-]+;base64,):|[^a-z]|[a-z+.-]+(?:[^a-z+.\-:]|$))/i,
};

let _purify = null;
function getPurify() {
    if (_purify !== null) return _purify;
    if (typeof window === 'undefined' || typeof document === 'undefined') {
        _purify = false;
        return false;
    }
    try {
        // DOMPurify in ESM: the default export is a configured factory bound
        // to the current window. In some bundles you must pass `window` to
        // `createDOMPurify`; the default export handles that for us.
        _purify = DOMPurify;
    } catch (_) {
        _purify = false;
    }
    return _purify;
}

/**
 * Marked-only render. Exported for unit tests where DOMPurify is unavailable.
 *
 * @param {string} src
 * @returns {string}
 */
export function renderRaw(src) {
    if (!src) return '';
    return marked.parse(String(src));
}

/**
 * Render markdown to sanitized HTML safe to assign via `innerHTML`.
 * Streaming-friendly: cheap enough to call on every chunk.
 *
 * @param {string} src
 * @returns {string}
 */
export function renderMarkdown(src) {
    const html = renderRaw(src);
    const purify = getPurify();
    if (!purify) return html;
    return purify.sanitize(html, PURIFY_CONFIG);
}
