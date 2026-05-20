// brainwires-chat-pwa — entry point
//
// Boot order:
//   1. open IndexedDB + register the SW (parallel)
//   2. wire SW → page chat IPC into the appEvents bus
//   3. wire local-provider events to the same hyphenated channel
//   4. mount the persistent download banner above the view region
//   5. initialize i18n
//   6. set up the view router and register chat / settings / unlock
//   7. route to 'unlock' if a passphrase is configured but the
//      session key isn't loaded; otherwise route to 'chat'
//   8. lazy-init the wasm module in the background — TTS/STT/local
//      providers wait on `getWasm()` themselves; first paint must not.

import { openDb, OpfsTabConflictError } from './sql-db.js';
import {
    getWasm,
    setSwRegistration,
    appEvents,
    events as stateEvents,
    isSessionUnlocked,
} from './state.js';
import { isDownloaded, KNOWN_MODELS, KNOWN_EMBEDDING_MODELS } from './model-store.js';
import { getSetting } from './sql-db.js';
import * as views from './views.js';
import { mountBanner } from './ui-download-banner.js';
import { initI18n } from './i18n.js';
import { loadTheme } from './theme.js';
import * as uiChat from './ui-chat.js';
import * as uiSettings from './ui-settings.js';
import * as uiUnlock from './ui-unlock.js';
import { maybeInstallDevToggle as maybeInstallHomeDevToggle } from './home-dev-toggle.js';

const PASSPHRASE_SETTING = 'passphraseConfig';


async function isDevMode() {
    try {
        const info = await import('../build-info.js');
        return info.DEV_MODE === true;
    } catch (_) {
        return false;
    }
}

/**
 * Replace the app shell with a "this app is open in another tab"
 * panel. The OPFS-backed sqlite db can only be held by one tab at a
 * time (sync access handles are exclusive), so any second tab on the
 * same origin needs to step aside until the primary releases.
 *
 * Polls every second on visibilitychange / focus to auto-recover when
 * the user closes the other tab.
 */
function renderTabConflictNotice(app, _err) {
    if (!app) return;
    app.innerHTML = `
        <main class="tab-conflict-notice">
            <h1>Already open in another tab</h1>
            <p>
                Brainwires Chat keeps its database in browser-local storage
                that only one tab can hold at a time. Please use the other
                tab — or close it and click <strong>Retry</strong> here.
            </p>
            <button type="button" id="tab-conflict-retry">Retry</button>
        </main>
    `;
    const retry = document.getElementById('tab-conflict-retry');
    if (retry) retry.addEventListener('click', () => window.location.reload());
    // Auto-retry when the user comes back to this tab — they likely
    // closed the other one.
    let attempting = false;
    const tryAgain = async () => {
        if (attempting || document.hidden) return;
        attempting = true;
        try {
            const navAny = /** @type {any} */ (navigator);
            if (navAny.locks && typeof navAny.locks.query === 'function') {
                const state = await navAny.locks.query();
                const stillHeld = (state.held || []).some((l) =>
                    String(l.name || '').startsWith('bw-chat-opfs-primary:'),
                );
                if (!stillHeld) window.location.reload();
            }
        } catch (_) { /* ignore */ } finally {
            attempting = false;
        }
    };
    document.addEventListener('visibilitychange', tryAgain);
    window.addEventListener('focus', tryAgain);
}

async function registerServiceWorker() {
    if (!('serviceWorker' in navigator)) return null;
    try {
        const reg = await navigator.serviceWorker.register('./sw.js', { scope: './' });
        setSwRegistration(reg);
        appEvents.dispatchEvent(new CustomEvent('sw-ready', { detail: { registration: reg } }));

        // In dev mode: tell the SW to use network-first (no cache/SRI)
        // so live-editing works, but keep the SW alive for model downloads.
        if (await isDevMode()) {
            const ctrl = navigator.serviceWorker.controller || reg.active;
            if (ctrl) ctrl.postMessage({ type: 'set_dev_mode', enabled: true });
            console.log('DEV_MODE: SW registered (network-first, no cache)');
        }
        return reg;
    } catch (err) {
        console.warn('SW registration failed:', err && err.message ? err.message : err);
        return null;
    }
}

function wireServiceWorkerMessages() {
    if (!('serviceWorker' in navigator)) return;
    navigator.serviceWorker.addEventListener('message', (event) => {
        const msg = event.data;
        if (!msg || typeof msg !== 'object') return;
        switch (msg.type) {
            case 'chat_chunk':
                stateEvents.dispatchEvent(new CustomEvent('chat_chunk', { detail: msg }));
                appEvents.dispatchEvent(new CustomEvent('chat-chunk', { detail: msg }));
                break;
            case 'chat_tool_use':
                // MCP plumbing (Follow-up 2): SW broadcasts a fully
                // reassembled tool invocation; the picker / execution
                // loop / bubble rendering land in the next commit.
                stateEvents.dispatchEvent(new CustomEvent('chat_tool_use', { detail: msg }));
                appEvents.dispatchEvent(new CustomEvent('chat-tool-use', { detail: msg }));
                break;
            case 'chat_done':
                stateEvents.dispatchEvent(new CustomEvent('chat_done', { detail: msg }));
                appEvents.dispatchEvent(new CustomEvent('chat-done', { detail: msg }));
                break;
            case 'chat_error':
            case 'chat_aborted':
                stateEvents.dispatchEvent(new CustomEvent('chat_error', { detail: msg }));
                appEvents.dispatchEvent(new CustomEvent('chat-error', { detail: msg }));
                break;
            // open_chat / chat_status / sri_table — not handled until UI lands.
        }
    });
}

// Mirror local-provider events from `state.events` (which providers/local.js
// dispatches under 'chat_chunk' etc) into `appEvents` 'chat-chunk' etc so
// any code that prefers the hyphenated channel still works.
function wireLocalProviderEvents() {
    const fwd = (underscore, hyphen) => {
        stateEvents.addEventListener(underscore, (e) => {
            appEvents.dispatchEvent(new CustomEvent(hyphen, { detail: { type: underscore, ...(e.detail || {}) } }));
        });
    };
    fwd('chat_chunk', 'chat-chunk');
    fwd('chat_done', 'chat-done');
    fwd('chat_error', 'chat-error');
}

async function shouldStartLocked() {
    try {
        const cfg = await getSetting(PASSPHRASE_SETTING);
        if (cfg && cfg.salt && cfg.verify && !isSessionUnlocked()) return true;
    } catch (_) { /* no idb yet */ }
    return false;
}

async function boot() {
    const app = document.getElementById('app');
    const bannerSlot = document.getElementById('download-banner-slot');

    // i18n: read the saved language (settings store), fall back to the
    // detected system locale on first run. Awaiting before view mount so
    // the first paint uses translated strings and `<html lang/dir>` is
    // already set when stylesheets evaluate `[dir="rtl"]` selectors.
    let savedLang = null;
    try { savedLang = (await getSetting('language')) || null; } catch { /* db not open yet */ }
    await initI18n(savedLang).catch(() => {});

    // Mount the persistent footer after i18n so the idle label renders
    // translated on the first paint.
    if (bannerSlot) mountBanner(bannerSlot);

    // DB + SW in parallel.
    const [dbResult, swResult] = await Promise.allSettled([
        openDb(),
        registerServiceWorker(),
    ]);
    if (dbResult.status === 'rejected') {
        // OPFS sync access handles are exclusive per file. If a second
        // tab loads the same origin, the rsqlite worker fails with
        // NoModificationAllowedError and the user gets a stuck app.
        // Detect that case and replace the entire view with a friendly
        // notice instead of trying to limp along.
        if (dbResult.reason instanceof OpfsTabConflictError) {
            renderTabConflictNotice(app, dbResult.reason);
            return;
        }
        console.warn('IndexedDB open failed:', dbResult.reason);
    }
    if (swResult.status === 'rejected') {
        console.warn('SW registration error:', swResult.reason);
    }

    // Apply the saved theme before any view paints. Falls through to
    // 'system' (the pre-switcher behavior) if nothing is saved.
    await loadTheme().catch((err) => console.warn('theme load failed:', err));

    wireServiceWorkerMessages();
    wireLocalProviderEvents();

    // Set up view router. Each view's render() runs on first activation.
    if (app) {
        views.init(app);
        views.register('chat', uiChat);
        views.register('settings', uiSettings);
        views.register('unlock', uiUnlock);
    }

    // Decide initial view. Respect ?page= query param so reloads stay
    // on the same view (e.g. /?page=settings).
    const requestedPage = new URL(location.href).searchParams.get('page');
    if (await shouldStartLocked()) {
        views.mount('unlock');
    } else if (requestedPage && ['settings', 'chat', 'unlock'].includes(requestedPage)) {
        views.mount(requestedPage);
    } else {
        views.mount('chat');
    }

    // Lazy-init the wasm module in the background. The chat composer
    // uses `voice.getTts()` / `voice.getStt()` which both await
    // `getWasm()` themselves; this just warms the cache so the first
    // user interaction doesn't pay the load cost.
    getWasm().catch((err) => {
        console.warn('wasm warmup failed:', err && err.message ? err.message : err);
    });

    // Hidden developer dial-home toggle (Phase 2 M5). No-op unless the
    // ?home=<url> query param or `bw_dial_home_url` localStorage key
    // is set. M9 will replace this with a real chat-UI integration.
    try { maybeInstallHomeDevToggle(); }
    catch (e) { console.warn('home-dev-toggle install failed:', e && e.message ? e.message : e); }

    // GC orphaned OPFS model directories. The chat-pwa stores each
    // downloaded model at `model-downloads/<modelId>/`. When the
    // registry rotates a model id (e.g. `gemma-4-e2b → gemma-4-e2b-it`
    // when we switched from base to instruction-tuned), the old
    // directory is left behind — typically ~10 GB per model. Walk the
    // root, drop any entry whose name isn't in the current
    // `KNOWN_MODELS` or `KNOWN_EMBEDDING_MODELS` registries.
    try {
        await pruneOrphanedOpfsModels();
    } catch (e) {
        // OPFS unavailable (Safari iOS < 16, SSR, private mode quirks)
        // or one of the directory ops threw — ignore. Leaking a
        // directory is harmless, just suboptimal.
        console.debug('opfs prune skipped:', e && e.message ? e.message : e);
    }

    // Probe whether the default local model is already cached.
    try {
        const cached = await isDownloaded('gemma-4-e2b-it');
        appEvents.dispatchEvent(new CustomEvent('local-model-cached-status', {
            detail: { modelId: 'gemma-4-e2b-it', cached },
        }));
    } catch (_) { /* Cache Storage may be unavailable in tests/SSR */ }
}

/// Remove any OPFS `model-downloads/<id>/` whose `<id>` isn't a
/// currently-known model or embedding model. Skips silently if OPFS
/// isn't available.
async function pruneOrphanedOpfsModels() {
    if (typeof navigator === 'undefined' || !navigator.storage || !navigator.storage.getDirectory) {
        return;
    }
    const root = await navigator.storage.getDirectory();
    let dlDir;
    try {
        dlDir = await root.getDirectoryHandle('model-downloads', { create: false });
    } catch {
        return; // No download directory yet — nothing to prune.
    }
    const known = new Set([
        ...Object.keys(KNOWN_MODELS),
        ...Object.keys(KNOWN_EMBEDDING_MODELS),
        // `model-downloads/ollama/<name>__<tag>/` — managed by
        // ollama-download.js, not the per-modelId scheme above.
        // Skip the parent so its children aren't recursively wiped.
        'ollama',
    ]);
    // FileSystemDirectoryHandle is async-iterable in modern browsers;
    // the older `entries()` method is also accepted. Try the iterator
    // form first, fall back if needed.
    const entries = [];
    try {
        for await (const [name, handle] of dlDir.entries()) {
            entries.push([name, handle]);
        }
    } catch (e) {
        console.debug('opfs prune: directory iteration unavailable', e);
        return;
    }
    for (const [name, handle] of entries) {
        if (handle.kind !== 'directory') continue;
        if (known.has(name)) continue;
        console.info(`[opfs prune] removing orphaned model directory: ${name}`);
        try {
            await dlDir.removeEntry(name, { recursive: true });
        } catch (e) {
            console.warn(`[opfs prune] failed to remove ${name}:`, e && e.message ? e.message : e);
        }
    }
}

boot().catch((err) => {
    console.error('boot failed:', err);
    const app = document.getElementById('app');
    if (app) app.textContent = `Boot failed: ${err && err.message ? err.message : err}`;
});
