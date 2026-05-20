// brainwires-chat-pwa — DOM + formatting utilities
//
// Tiny, framework-free helpers. Used everywhere the UI builds a node,
// shows a toast, or formats a number for the user.

/**
 * Create a DOM element. Supports:
 *   - props.class / props.className
 *   - props.dataset = { key: 'value' }
 *   - props.style   = { color: 'red' }
 *   - props.attrs   = { 'aria-label': '…' }
 *   - props.on{Event} = handler         (e.g. onClick)
 *   - any other key sets the property if the element has one, else attribute.
 *   - children: nodes, strings (→ text nodes), arrays (flattened), null/undefined skipped.
 *
 * @param {string} tag
 * @param {Record<string, any>} [props]
 * @param {...any} children
 * @returns {HTMLElement}
 */
export function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    if (props && typeof props === 'object') {
        for (const [k, v] of Object.entries(props)) {
            if (v == null) continue;
            if (k === 'class' || k === 'className') {
                node.className = String(v);
            } else if (k === 'style' && typeof v === 'object') {
                Object.assign(node.style, v);
            } else if (k === 'dataset' && typeof v === 'object') {
                for (const [dk, dv] of Object.entries(v)) {
                    if (dv != null) node.dataset[dk] = String(dv);
                }
            } else if (k === 'attrs' && typeof v === 'object') {
                for (const [ak, av] of Object.entries(v)) {
                    if (av == null) continue;
                    if (av === false) continue;
                    node.setAttribute(ak, av === true ? '' : String(av));
                }
            } else if (k.startsWith('on') && typeof v === 'function') {
                node.addEventListener(k.slice(2).toLowerCase(), v);
            } else if (k === 'html') {
                node.innerHTML = String(v);
            } else if (k === 'text') {
                node.textContent = String(v);
            } else if (k in node) {
                try { node[k] = v; } catch (_) { node.setAttribute(k, String(v)); }
            } else {
                if (v === false) continue;
                node.setAttribute(k, v === true ? '' : String(v));
            }
        }
    }
    appendChildren(node, children);
    return node;
}

function appendChildren(node, children) {
    for (const c of children) {
        if (c == null || c === false) continue;
        if (Array.isArray(c)) { appendChildren(node, c); continue; }
        if (c instanceof Node) { node.appendChild(c); continue; }
        node.appendChild(document.createTextNode(String(c)));
    }
}

/** Empty a DOM node. */
export function clear(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
}

// ── Toasts ─────────────────────────────────────────────────────

let _toastHost = null;
const _toastQueue = [];
let _toastBusy = false;

function ensureToastHost() {
    if (_toastHost && document.body.contains(_toastHost)) return _toastHost;
    _toastHost = document.createElement('div');
    _toastHost.id = 'toast-host';
    _toastHost.setAttribute('role', 'status');
    _toastHost.setAttribute('aria-live', 'polite');
    document.body.appendChild(_toastHost);
    return _toastHost;
}

/**
 * Show a transient toast at the bottom of the viewport.
 * @param {string} message
 * @param {'info'|'error'|'success'|'warn'} [kind='info']
 * @param {number} [timeoutMs=3000]
 */
export function toast(message, kind = 'info', timeoutMs = 3000) {
    _toastQueue.push({ message, kind, timeoutMs });
    if (!_toastBusy) drainToasts();
}

function drainToasts() {
    if (_toastQueue.length === 0) { _toastBusy = false; return; }
    _toastBusy = true;
    const { message, kind, timeoutMs } = _toastQueue.shift();
    const host = ensureToastHost();
    const node = el('div', { class: `toast toast-${kind}`, attrs: { role: 'alert' } }, String(message));
    host.appendChild(node);
    // Force reflow so the enter transition fires.
    void node.offsetWidth;
    node.classList.add('is-visible');
    const fade = () => {
        node.classList.remove('is-visible');
        setTimeout(() => {
            try { node.remove(); } catch (_) {}
            drainToasts();
        }, 220);
    };
    setTimeout(fade, Math.max(800, timeoutMs));
}

// ── Formatters ─────────────────────────────────────────────────

/**
 * Human-readable byte count: 1024 → "1.0 KB", 5_000_000 → "4.8 MB".
 * @param {number} n
 * @returns {string}
 */
export function formatBytes(n) {
    if (typeof n !== 'number' || !isFinite(n) || n < 0) return '0 B';
    if (n < 1024) return `${n} B`;
    const units = ['KB', 'MB', 'GB', 'TB'];
    let v = n / 1024;
    let i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
    const fmt = v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
    return `${fmt} ${units[i]}`;
}

/**
 * Human-readable ETA: 75 → "1m 15s", 3700 → "1h 1m 40s".
 * @param {number | null | undefined} seconds
 * @returns {string}
 */
export function formatEta(seconds) {
    if (seconds == null || !isFinite(seconds) || seconds < 0) return '—';
    const s = Math.round(seconds);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    const rs = s % 60;
    if (m < 60) return `${m}m ${rs}s`;
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return `${h}h ${rm}m ${rs}s`;
}

// ── Function helpers ───────────────────────────────────────────

/** Call `fn` only after `ms` of quiet. */
export function debounce(fn, ms) {
    let t = null;
    return function debounced(...args) {
        if (t) clearTimeout(t);
        t = setTimeout(() => { t = null; fn.apply(this, args); }, ms);
    };
}

/** Call `fn` at most once per `ms`. Trailing call is preserved. */
export function throttle(fn, ms) {
    let last = 0;
    let pendingArgs = null;
    let t = null;
    return function throttled(...args) {
        const now = Date.now();
        const remaining = ms - (now - last);
        if (remaining <= 0) {
            last = now;
            fn.apply(this, args);
        } else {
            pendingArgs = args;
            if (!t) {
                t = setTimeout(() => {
                    last = Date.now();
                    t = null;
                    if (pendingArgs) {
                        fn.apply(this, pendingArgs);
                        pendingArgs = null;
                    }
                }, remaining);
            }
        }
    };
}

/**
 * Heuristic mobile check: touch + small viewport OR mobile UA. Used to
 * pick between Enter-to-send (desktop) vs Enter-newline (mobile).
 */
export function isMobile() {
    if (typeof navigator === 'undefined') return false;
    const ua = navigator.userAgent || '';
    const uaMobile = /Mobi|Android|iPhone|iPad|iPod|Mobile/i.test(ua);
    const hasTouch = (typeof window !== 'undefined')
        && (('ontouchstart' in window) || (navigator.maxTouchPoints > 0));
    const narrow = (typeof window !== 'undefined') && window.innerWidth <= 820;
    return uaMobile || (hasTouch && narrow);
}

/** Generate a non-cryptographic id for transient DOM/state needs. */
export function genId(prefix = 'id') {
    return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Escape HTML for safe insertion as text-as-html. We use this in the
 * markdown renderer, since textContent can't render formatting.
 */
export function escapeHtml(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
