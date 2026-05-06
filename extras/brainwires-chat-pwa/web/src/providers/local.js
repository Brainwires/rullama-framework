// brainwires-chat-pwa — local-WASM provider (Gemma 4 E2B)
//
// Phase 2: thin RPC client over a dedicated Web Worker
// (`src/local-worker.js`). The main thread no longer touches the WASM
// module directly — it ships request envelopes across postMessage and
// re-dispatches the worker's replies as the same `model_progress`,
// `chat_chunk`, `chat_done`, `chat_error` events the UI already
// listens for. Net effect: the page stays responsive (60fps drawer
// toggles, scroll, typing) even while the worker is mid-generation.
//
// Lifecycle helpers (`loadLocalModel`, `unloadLocalModel`,
// `isLocalModelLoaded`) keep their existing names so the Settings page
// and `ui-chat.js` don't need to change. Internal aliases `chatLocal`
// and `cancelLocal` are also exported for callers that prefer the
// verb-noun shape.

import {
    getLocalWorker,
    setLocalWorker,
    getLocalModelId,
    setLocalModelId,
    isLocalModelLoaded as stateIsLoaded,
    events,
    appEvents,
} from '../state.js';
import { appendMessageChunk, putMessage, partsToText } from '../db.js';

export const id = 'local-gemma-4-e2b-it';
export const displayName = 'Gemma 4 E2B IT (on-device)';
export const runtime = 'local';
// Default to the Ollama Q4_K_M path because it ships with WGPU
// kernels for q4_k matmul (verified working end-to-end against the
// `ollama run gemma4:e2b` reference). The HF safetensors variant is
// BF16, and candle's WGPU backend does not yet implement BF16
// storage — falls back to F16 via a cast at load time, but that's
// extra memory + slower than going straight through Q4_K_M. Users
// who need the vision/audio towers can flip via the Settings → Local
// Model "Use" buttons.
export const defaultModel = 'gemma4:e2b';
// Both local sources for Gemma 4 E2B. 'gemma4:e2b' is the
// Ollama-format Q4_K_M GGUF (~7.2 GB published, text-only — no
// vision/audio in this publication, despite the towers being
// embedded as BF16 in the file). 'gemma-4-e2b-it' is the HF
// safetensors variant (~10 GB BF16, full vision + audio towers
// unpacked). Pick by trade-off: speed/size vs vision capability.
export const models = ['gemma4:e2b', 'gemma-4-e2b-it'];

// ── Worker singleton + RPC plumbing ────────────────────────────

let _nextRequestId = 1;
const _pending = new Map();          // requestId → { resolve, reject }
// Track in-flight chat streams so a worker crash can surface as
// chat_error events on the active conversations.
const _activeChats = new Map();      // conversationId → { messageId }

function dispatch(type, detail) {
    events.dispatchEvent(new CustomEvent(type, { detail }));
    // Mirror to the legacy hyphenated channel boot.js wires for SW msgs,
    // so existing listeners pick up local streams too.
    const hyphenType = type.replace(/_/g, '-');
    appEvents.dispatchEvent(new CustomEvent(hyphenType, { detail: { type, ...detail } }));
}

function rejectAllPending(error) {
    for (const [, { reject }] of _pending) {
        try { reject(error); } catch (_) { /* ignore */ }
    }
    _pending.clear();
}

function failAllActiveChats(errMsg) {
    for (const [conversationId, { messageId }] of _activeChats) {
        dispatch('chat_error', { conversationId, messageId, error: errMsg });
    }
    _activeChats.clear();
}

function handleWorkerMessage(ev) {
    const msg = ev.data;
    if (!msg || typeof msg !== 'object') return;

    // Streaming + progress events — broadcast to UI via state.events.
    switch (msg.type) {
        case 'load_progress':
            events.dispatchEvent(new CustomEvent('model_progress', {
                detail: { phase: msg.phase, modelId: msg.modelId },
            }));
            break;
        case 'chat_chunk':
            // Persist the delta in IndexedDB on the main thread (the
            // worker doesn't share our db.js connection). Chunks are
            // small (one token-ish each) so the postMessage cost is
            // negligible compared to the wasm work the worker is doing.
            if (msg.conversationId && msg.messageId && typeof msg.delta === 'string') {
                appendMessageChunk(msg.conversationId, msg.messageId, msg.delta).catch(() => {});
                dispatch('chat_chunk', {
                    conversationId: msg.conversationId,
                    messageId: msg.messageId,
                    delta: msg.delta,
                });
            }
            break;
        default:
            break;
    }

    // Request/reply correlation.
    if (typeof msg.requestId === 'number') {
        const slot = _pending.get(msg.requestId);
        if (!slot) return;
        _pending.delete(msg.requestId);
        switch (msg.type) {
            case 'load_done':
                slot.resolve({ modelId: msg.modelId });
                break;
            case 'load_error':
                slot.reject(new Error(msg.error || 'load_error'));
                break;
            case 'chat_done':
                if (msg.conversationId) _activeChats.delete(msg.conversationId);
                slot.resolve({ usage: msg.usage || null, tokensReceived: msg.tokensReceived || 0 });
                break;
            case 'chat_error':
                if (msg.conversationId) _activeChats.delete(msg.conversationId);
                dispatch('chat_error', {
                    conversationId: msg.conversationId,
                    messageId: msg.messageId,
                    error: msg.error,
                });
                slot.reject(new Error(msg.error || 'chat_error'));
                break;
            case 'cancel_ack':
                slot.resolve({ conversationId: msg.conversationId });
                break;
            case 'unload_ack':
                slot.resolve({});
                break;
            default:
                slot.reject(new Error(`unknown reply type: ${msg.type}`));
        }
    }
}

function handleWorkerError(err) {
    const errMsg = err && err.message ? err.message : 'local worker crashed';
    rejectAllPending(new Error(errMsg));
    failAllActiveChats(errMsg);
    // Drop the singleton so the next call re-spawns it fresh.
    const w = getLocalWorker();
    if (w) {
        try { w.terminate(); } catch (_) { /* ignore */ }
    }
    setLocalWorker(null);
    setLocalModelId(null);
}

function getWorker() {
    let w = getLocalWorker();
    if (w) return w;
    // After esbuild bundles src/boot.js → web/app.js, `import.meta.url`
    // points at web/app.js, and the worker lives next to it at
    // web/local-worker.js. Resolve relative to the bundled location.
    w = new Worker(new URL('./local-worker.js', import.meta.url), { type: 'module' });
    w.addEventListener('message', handleWorkerMessage);
    w.addEventListener('error', handleWorkerError);
    w.addEventListener('messageerror', (e) => handleWorkerError(e));
    setLocalWorker(w);
    return w;
}

function rpc(payload) {
    const requestId = _nextRequestId++;
    const worker = getWorker();
    return new Promise((resolve, reject) => {
        _pending.set(requestId, { resolve, reject });
        try {
            worker.postMessage({ requestId, ...payload });
        } catch (err) {
            _pending.delete(requestId);
            reject(err);
        }
    });
}

// ── Lifecycle ──────────────────────────────────────────────────

/**
 * Ask the worker to load a model. Resolves with `{ modelId }`.
 *
 * @param {string} [modelId='gemma-4-e2b']
 * @returns {Promise<{modelId: string}>}
 */
export async function loadLocalModel(modelId = defaultModel) {
    if (stateIsLoaded() && getLocalModelId() === modelId) {
        return { modelId };
    }
    try {
        // Forward debug toggles set on the page's window into the worker
        // load message so they can be flipped from the browser console
        // without rebuilding. Read at message-handle time on the worker
        // side; see local-worker.js handleLoad().
        const diag = !!globalThis.__bw_diag;
        const diagLayerRaw = globalThis.__bw_diag_layer;
        const diagLayer = (typeof diagLayerRaw === 'number' && Number.isInteger(diagLayerRaw))
            ? diagLayerRaw : null;
        const out = await rpc({ type: 'load', modelId, diag, diagLayer });
        setLocalModelId(out.modelId || modelId);
        return out;
    } catch (err) {
        const msg = err && err.message ? err.message : String(err);
        if (msg === 'not_downloaded') {
            throw new Error(`local model not downloaded: ${modelId}. Open Settings → Local model to download.`);
        }
        throw err;
    }
}

/** Drop the loaded model handle; lets the worker's WASM allocator reclaim memory. */
export async function unloadLocalModel() {
    if (!getLocalWorker()) {
        setLocalModelId(null);
        return;
    }
    try { await rpc({ type: 'unload' }); }
    catch (_) { /* idempotent */ }
    setLocalModelId(null);
}

export function isLocalModelLoaded() {
    return stateIsLoaded();
}

// ── Streaming ──────────────────────────────────────────────────

/**
 * Start a chat stream against the loaded model. Resolves on
 * `chat_done`; rejects on `chat_error`. Chunks are dispatched on
 * `state.events` as `chat_chunk` (and mirrored as `chat-chunk` on
 * `appEvents` for legacy listeners).
 *
 * @param {object} args
 * @param {string} args.conversationId
 * @param {string} args.messageId
 * @param {Array<{role: string, content: string}>} args.messages
 * @param {object} [args.params]
 */
export async function startChat({ conversationId, messageId, messages, params = {} }) {
    if (!stateIsLoaded()) {
        await loadLocalModel(params.model || defaultModel);
    }
    _activeChats.set(conversationId, { messageId });
    // Route through the vision RPC when any message carries an image part
    // (parts[] content). Text-only history flows through the legacy chat
    // RPC so wasm builds without the multimodal export still work; we
    // flatten any text-only parts[] back to string for that path so the
    // current wasm signature (string content) keeps accepting it.
    const hasImage = Array.isArray(messages)
        && messages.some((m) => Array.isArray(m && m.content) && m.content.some((p) => p && p.type === 'image'));
    const wireMessages = hasImage
        ? messages
        : (messages || []).map((m) => ({
            role: m.role,
            content: typeof m.content === 'string' ? m.content : partsToText(m.content),
        }));
    let result;
    try {
        result = await rpc({
            type: hasImage ? 'vision_chat' : 'chat',
            conversationId,
            messageId,
            messages: wireMessages,
            params,
        });
    } catch (err) {
        // chat_error already dispatched in handleWorkerMessage; just
        // stamp the message row best-effort and rethrow.
        try {
            await putMessage({
                conversationId,
                messageId,
                role: 'assistant',
                updatedAt: Date.now(),
                completedAt: Date.now(),
            });
        } catch (_) { /* best-effort */ }
        throw err;
    }

    // Stamp final updatedAt + persisted state. The chunk-appended row
    // is the source of truth for content; we only patch metadata here.
    try {
        await putMessage({
            conversationId,
            messageId,
            role: 'assistant',
            updatedAt: Date.now(),
            completedAt: Date.now(),
            tokensReceived: result.tokensReceived || 0,
        });
    } catch (_) { /* best-effort */ }

    dispatch('chat_done', {
        conversationId,
        messageId,
        usage: result.usage || null,
        tokensReceived: result.tokensReceived || 0,
    });
    return result;
}

/**
 * Cancel an in-flight stream for `conversationId`. Best-effort — the
 * worker will stop reading the wasm stream on its next iteration.
 *
 * @param {string} conversationId
 */
export async function cancelLocal(conversationId) {
    if (!getLocalWorker()) return;
    try { await rpc({ type: 'cancel', conversationId }); }
    catch (_) { /* idempotent */ }
}

// Friendlier alias matching the verb-noun shape used elsewhere; kept
// alongside `startChat` so both styles read naturally at call sites.
export const chatLocal = startChat;
