// brainwires-chat-pwa — singleton app state
//
// Module-scope singletons. Safe to import from anywhere on the page; the
// service worker has its own runtime and does NOT share these.
//
// Holds:
//   - lazy WASM module reference (initialized via getWasm())
//   - session crypto key (post-passphrase-unlock; never persisted)
//   - active service-worker registration accessor
//   - app-wide pub/sub via EventTarget

const PKG_URL = './pkg/brainwires_chat_pwa.js';

// ── WASM lazy loader ───────────────────────────────────────────

let _wasm = null;
let _wasmPromise = null;

/**
 * Load and initialize the wasm-pack module the first time it's needed.
 * Subsequent calls return the same module instance.
 *
 * @returns {Promise<object>} the wasm module exports
 */
export function getWasm() {
    if (_wasm) return Promise.resolve(_wasm);
    if (_wasmPromise) return _wasmPromise;
    _wasmPromise = (async () => {
        const mod = await import(PKG_URL);
        if (typeof mod.default === 'function') {
            await mod.default();
        }
        if (typeof mod.init === 'function') {
            try { mod.init(); } catch (_) { /* idempotent or already-initialized */ }
        }
        _wasm = mod;
        return mod;
    })();
    return _wasmPromise;
}

// ── Session key (in-memory only) ───────────────────────────────

let _sessionKey = null;

/** @returns {CryptoKey | null} */
export function getSessionKey() {
    return _sessionKey;
}

/** @param {CryptoKey | null} key */
export function setSessionKey(key) {
    const wasUnlocked = _sessionKey !== null;
    _sessionKey = key;
    if (key && !wasUnlocked) appEvents.dispatchEvent(new Event('session-unlocked'));
    if (!key && wasUnlocked) appEvents.dispatchEvent(new Event('session-locked'));
}

export function lockSession() { setSessionKey(null); }
export function isSessionUnlocked() { return _sessionKey !== null; }

// ── Service-worker registration ────────────────────────────────

let _swRegistration = null;

/** @param {ServiceWorkerRegistration | null} reg */
export function setSwRegistration(reg) { _swRegistration = reg; }

/** @returns {ServiceWorkerRegistration | null} */
export function getSwRegistration() { return _swRegistration; }

/**
 * Convenience: post a message to the active service worker, if one is
 * controlling the page. Returns `false` when there's no controller.
 *
 * @param {any} msg
 * @returns {boolean}
 */
export function postToServiceWorker(msg) {
    const ctl = (typeof navigator !== 'undefined' && navigator.serviceWorker)
        ? navigator.serviceWorker.controller
        : null;
    if (!ctl) return false;
    ctl.postMessage(msg);
    return true;
}

// ── Local model state (in-memory only) ─────────────────────────
//
// Phase 2: the wasm-side `LocalModelHandle` lives entirely in the
// dedicated Web Worker (`src/local-worker.js`). The main thread only
// remembers the *id* of the loaded model and the worker singleton
// itself. UI code and other providers should treat the worker handle
// as opaque — go through `providers/local.js`.

let _localModelId = null;

/** @returns {string | null} */
export function getLocalModelId() {
    return _localModelId;
}

/**
 * Mark the local model as loaded / unloaded for this session. Drives the
 * `local-model-loaded` / `local-model-unloaded` app events that the UI
 * listens for. The actual wasm handle lives in the worker.
 *
 * @param {string | null} modelId
 */
export function setLocalModelId(modelId) {
    const wasLoaded = _localModelId !== null;
    _localModelId = modelId || null;
    if (_localModelId && !wasLoaded) {
        appEvents.dispatchEvent(new CustomEvent('local-model-loaded', { detail: { modelId: _localModelId } }));
    }
    if (!_localModelId && wasLoaded) {
        appEvents.dispatchEvent(new Event('local-model-unloaded'));
    }
}

/** @returns {boolean} */
export function isLocalModelLoaded() {
    return _localModelId !== null;
}

let _localWorker = null;

/** @returns {Worker | null} */
export function getLocalWorker() { return _localWorker; }

/** @param {Worker | null} w */
export function setLocalWorker(w) { _localWorker = w; }

// ── Decrypted session key (in-memory only) ─────────────────────
//
// Convenience alias for `getSessionKey()` so the providers layer can
// reach for a more descriptive name. The underlying slot is shared.

/** @returns {CryptoKey | null} */
export function getDecryptedSessionKey() {
    return _sessionKey;
}

/** @param {CryptoKey | null} key */
export function setDecryptedSessionKey(key) {
    setSessionKey(key);
}

// ── App-wide pub/sub ───────────────────────────────────────────
//
// Known event types (consumers should listen for these, dispatchers fire
// CustomEvent('name', { detail: ... }) where applicable):
//   - 'session-unlocked'  / 'session-locked'   (Event)
//   - 'chat-chunk'   { conversationId, messageId, delta } (CustomEvent.detail)
//   - 'chat-done'    { conversationId, messageId, usage }
//   - 'chat-error'   { conversationId, messageId, error }
//   - Canonical streaming-event envelope (mirrors of the SW message types;
//     `providers/local.js` dispatches these so the UI doesn't care whether
//     a stream is cloud-via-SW or local-via-WASM):
//       'chat_chunk'    { conversationId, messageId, delta?, reasoning_delta? }
//       'chat_tool_use' { conversationId, messageId, tool_use: {id, name, input} }
//       'chat_done'     { conversationId, messageId, usage?, tokensReceived? }
//       'chat_error'    { conversationId, messageId, error }
//     Note: `chat_tool_use` is documented now but emitted by Follow-up 2;
//     current code does not produce it. `reasoning_delta` is similarly
//     reserved for future reasoning-stream providers.
//   - 'model_progress' { modelId, file, fileBytesDone, ... }
//   - 'model_deleted'  { modelId }
//   - 'local-model-loaded' / 'local-model-unloaded'
//   - 'sw-ready'     { registration }
//   - 'theme-changed' { theme: 'light'|'dark'|'system' }
//   - 'theme-system-changed' { systemPrefersLight: boolean }
export const appEvents = new EventTarget();

// Alias for code that prefers `state.events` over `state.appEvents`.
// They reference the same `EventTarget` — pick whichever reads better.
export const events = appEvents;
