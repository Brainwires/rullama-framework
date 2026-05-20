// brainwires-chat-pwa — embedding model RPC client
//
// Thin wrapper around the local worker's embed_load / embed_text / embed_unload
// RPCs. Uses the same worker singleton as providers/local.js so chat + RAG
// don't fight over WASM init. The chat handle and the embedding handle live
// in separate slots inside the worker so loading a chat model does not free
// the embedder and vice-versa.
//
// Embedding model selection is persisted in the IDB settings store under
// 'embedding.activeModel' (already used by the existing Settings UI for
// embedding-model download). RAG callers read that key, then `loadModel()`
// here, then `embed(text)` per query/chunk.

import { getLocalWorker, setLocalWorker } from './state.js';

let _nextRequestId = 100000; // separate range from providers/local.js
const _pending = new Map();

function getWorker() {
    let w = getLocalWorker();
    if (w) return w;
    w = new Worker(new URL('./local-worker.js', import.meta.url), { type: 'module' });
    w.addEventListener('message', handleMessage);
    w.addEventListener('error', () => {
        for (const [, slot] of _pending) {
            try { slot.reject(new Error('worker crashed')); } catch (_) {}
        }
        _pending.clear();
        try { w.terminate(); } catch (_) {}
        setLocalWorker(null);
    });
    setLocalWorker(w);
    return w;
}

function handleMessage(ev) {
    const msg = ev.data;
    if (!msg || typeof msg !== 'object' || typeof msg.requestId !== 'number') return;
    const slot = _pending.get(msg.requestId);
    if (!slot) return;
    _pending.delete(msg.requestId);
    switch (msg.type) {
        case 'embed_load_done':
            slot.resolve({ modelId: msg.modelId, dim: msg.dim });
            break;
        case 'embed_load_error':
            slot.reject(new Error(msg.error || 'embed_load_error'));
            break;
        case 'embed_text_done':
            slot.resolve(msg.vector);
            break;
        case 'embed_text_error':
            slot.reject(new Error(msg.error || 'embed_text_error'));
            break;
        case 'embed_unload_ack':
            slot.resolve({});
            break;
        default:
            // Other message types (chat_chunk, etc.) are handled by
            // providers/local.js — no-op here.
            break;
    }
}

function rpc(payload) {
    const requestId = _nextRequestId++;
    const worker = getWorker();
    return new Promise((resolve, reject) => {
        _pending.set(requestId, { resolve, reject });
        try { worker.postMessage({ requestId, ...payload }); }
        catch (err) { _pending.delete(requestId); reject(err); }
    });
}

let _loadedModel = null;
let _loadedDim = 0;

/**
 * Load (or switch to) an embedding model in the worker. Returns dim.
 * No-op when the requested model is already loaded.
 *
 * @param {string} modelId
 * @returns {Promise<{ modelId: string, dim: number }>}
 */
export async function loadModel(modelId) {
    if (!modelId) throw new Error('loadModel: modelId required');
    if (_loadedModel === modelId && _loadedDim > 0) {
        return { modelId, dim: _loadedDim };
    }
    const out = await rpc({ type: 'embed_load', modelId });
    _loadedModel = out.modelId || modelId;
    _loadedDim = out.dim || 0;
    return out;
}

/**
 * Encode a single string. Caller must `loadModel()` first.
 *
 * @param {string} text
 * @returns {Promise<Float32Array>}
 */
export async function embed(text) {
    return rpc({ type: 'embed_text', text });
}

/**
 * Encode a batch of strings sequentially. Concurrency intentionally limited
 * to one at a time — the wasm runtime is single-threaded and simultaneous
 * embed calls would just queue inside the worker anyway.
 *
 * @param {string[]} texts
 * @returns {Promise<Float32Array[]>}
 */
export async function embedBatch(texts) {
    const out = [];
    for (const t of texts) out.push(await embed(t));
    return out;
}

/** Free the embedding model handle. Idempotent. */
export async function unloadModel() {
    if (!_loadedModel) return;
    try { await rpc({ type: 'embed_unload' }); }
    catch (_) { /* idempotent */ }
    _loadedModel = null;
    _loadedDim = 0;
}

export function loadedModelId() {
    return _loadedModel;
}

export function loadedDim() {
    return _loadedDim;
}
