// brainwires-chat-pwa — Hugging Face model download + Cache Storage
//
// Mirrors the framework's `KnownModel` registry on the JS side. Hugging
// Face is the SOLE source — no mirrors, no fallback hosts.
//
// Storage:
//   - Cache Storage under name `bw-models-v1`. Each model file is keyed
//     by its public HF URL: `https://huggingface.co/<repo>/resolve/<rev>/<filename>`.
//     This makes deletes trivial (cache.delete(url)) and survives a SW
//     update because the cache name is stable.
//
// Concurrency:
//   - At most one active download globally. A second `downloadModel`
//     call returns the in-flight promise.

import { events } from './state.js';

// ── Registry mirror ────────────────────────────────────────────
//
// Source of truth is `crates/brainwires-providers/src/local_llm/model_registry.rs::known_models()`.
// Keep this in sync; SHA-256 pins remain `null` until the upstream
// registry pins them too (see TODO(gemma-4) in the Rust side).

export const KNOWN_MODELS = {
    'gemma-4-e2b-it': {
        id: 'gemma-4-e2b-it',
        displayName: 'Gemma 4 E2B IT',
        description: 'Gemma 4 E2B Instruction-Tuned (effective ~2B) — chat-trained, Candle/safetensors, runs in WASM.',
        source: 'hf',
        hf: { repo: 'google/gemma-4-e2b-it', revision: 'main' },
        files: [
            // SHA-256 pins null until upstream registry pins them.
            { kind: 'weights', filename: 'model.safetensors', sha256: null },
            { kind: 'tokenizer', filename: 'tokenizer.json', sha256: null },
        ],
        estimatedBytes: 10_246_621_918,
        contextSize: 8192,
        gated: false,
        // Gemma 4 E2B IT ships with the SigLIP vision tower; the worker uses
        // this flag to pick `init_local_multimodal` over the text-only
        // loader so `vision_chat` (parts[] with image entries) works.
        multimodal: true,
    },
};

// Ollama-format models. Pulled from `registry.ollama.ai` via the OCI
// Distribution Spec client in `ollama-fetch.js`. Files are populated
// dynamically from the manifest at fetch time, not pinned here, because
// Ollama re-publishes manifests as quantizations/templates evolve.
//
// Phase 4 status: download path lives in `ollama-download.js`; wasm-side
// GGUF inference path is deferred until Phase 5 (WGPU Q4_K_M kernel)
// makes it perf-relevant. Until then, an Ollama download saves bytes but
// not tok/s.
export const KNOWN_OLLAMA_MODELS = {
    'gemma4:e2b': {
        id: 'gemma4:e2b',
        displayName: 'Gemma 4 E2B (Ollama, Q4_K_M)',
        description: 'Gemma 4 E2B Q4_K_M from registry.ollama.ai (~7.2GB). Same model as the HF safetensors variant; quantized weights for the language model, BF16 vision/audio towers still bundled which is why the blob is larger than a text-only Q4_K_M would be.',
        source: 'ollama',
        ollama: { name: 'gemma4', tag: 'e2b' },
        // Files filled in at runtime from the manifest layers.
        files: null,
        // Tokenizer companion — Ollama publications don't always include a
        // `application/vnd.ollama.image.tokenizer` layer. When the
        // manifest lacks one, the loader falls back to fetching this
        // tokenizer.json from HuggingFace and caching it in the same
        // OPFS dir as the GGUF blob. Reuses the existing HF
        // model-store cache plumbing so the fetch is one-shot.
        tokenizerCompanion: {
            repo: 'google/gemma-4-e2b-it',
            revision: 'main',
            filename: 'tokenizer.json',
        },
        // Actual blob size per `ollama show gemma4:e2b`: 7.2 GB. Higher
        // than a pure text-only Q4_K_M because the publication still
        // bundles the BF16 vision + audio towers in the same GGUF
        // (we don't load them on this path, but they live in the file).
        estimatedBytes: 7_200_000_000,
        contextSize: 8192,
        gated: false,
        // Ollama's gemma4:e2b is text-only (no vision tower in the GGUF).
        multimodal: false,
    },
};

// ── Embedding model registry ──────────────────────────────────
//
// BERT-family models for local RAG. Run via candle in the Web Worker.
// Sorted by size (smallest first). All use safetensors + tokenizer.json.

export const KNOWN_EMBEDDING_MODELS = {
    'gte-small': {
        id: 'gte-small',
        displayName: 'GTE Small',
        provider: 'Alibaba DAMO',
        description: '384-dim, ~67 MB. Fast, good general-purpose.',
        hf: { repo: 'thenlper/gte-small', revision: 'main' },
        files: [
            { kind: 'weights', filename: 'model.safetensors', sha256: null },
            { kind: 'tokenizer', filename: 'tokenizer.json', sha256: null },
            { kind: 'config', filename: 'config.json', sha256: null },
        ],
        estimatedBytes: 67_000_000,
        dimensions: 384,
        maxTokens: 512,
        category: 'small',
    },
    'all-minilm-l6-v2': {
        id: 'all-minilm-l6-v2',
        displayName: 'all-MiniLM-L6-v2',
        provider: 'Sentence Transformers',
        description: '384-dim, ~80 MB. The classic lightweight embedding model.',
        hf: { repo: 'sentence-transformers/all-MiniLM-L6-v2', revision: 'main' },
        files: [
            { kind: 'weights', filename: 'model.safetensors', sha256: null },
            { kind: 'tokenizer', filename: 'tokenizer.json', sha256: null },
            { kind: 'config', filename: 'config.json', sha256: null },
        ],
        estimatedBytes: 80_000_000,
        dimensions: 384,
        maxTokens: 256,
        category: 'small',
    },
    'bge-small-en-v1.5': {
        id: 'bge-small-en-v1.5',
        displayName: 'BGE Small EN v1.5',
        provider: 'BAAI',
        description: '384-dim, ~130 MB. Strong retrieval quality for its size.',
        hf: { repo: 'BAAI/bge-small-en-v1.5', revision: 'main' },
        files: [
            { kind: 'weights', filename: 'model.safetensors', sha256: null },
            { kind: 'tokenizer', filename: 'tokenizer.json', sha256: null },
            { kind: 'config', filename: 'config.json', sha256: null },
        ],
        estimatedBytes: 130_000_000,
        dimensions: 384,
        maxTokens: 512,
        category: 'small',
    },
    'gte-base': {
        id: 'gte-base',
        displayName: 'GTE Base',
        provider: 'Alibaba DAMO',
        description: '768-dim, ~220 MB. Good balance of speed and quality.',
        hf: { repo: 'thenlper/gte-base', revision: 'main' },
        files: [
            { kind: 'weights', filename: 'model.safetensors', sha256: null },
            { kind: 'tokenizer', filename: 'tokenizer.json', sha256: null },
            { kind: 'config', filename: 'config.json', sha256: null },
        ],
        estimatedBytes: 220_000_000,
        dimensions: 768,
        maxTokens: 512,
        category: 'medium',
    },
    'bge-base-en-v1.5': {
        id: 'bge-base-en-v1.5',
        displayName: 'BGE Base EN v1.5',
        provider: 'BAAI',
        description: '768-dim, ~440 MB. Top-tier retrieval at medium size.',
        hf: { repo: 'BAAI/bge-base-en-v1.5', revision: 'main' },
        files: [
            { kind: 'weights', filename: 'model.safetensors', sha256: null },
            { kind: 'tokenizer', filename: 'tokenizer.json', sha256: null },
            { kind: 'config', filename: 'config.json', sha256: null },
        ],
        estimatedBytes: 440_000_000,
        dimensions: 768,
        maxTokens: 512,
        category: 'medium',
    },
    'nomic-embed-text-v1.5': {
        id: 'nomic-embed-text-v1.5',
        displayName: 'Nomic Embed Text v1.5',
        provider: 'Nomic AI',
        description: '768-dim, ~550 MB. Long context (8192 tokens), Matryoshka support.',
        hf: { repo: 'nomic-ai/nomic-embed-text-v1.5', revision: 'main' },
        files: [
            { kind: 'weights', filename: 'model.safetensors', sha256: null },
            { kind: 'tokenizer', filename: 'tokenizer.json', sha256: null },
            { kind: 'config', filename: 'config.json', sha256: null },
        ],
        estimatedBytes: 550_000_000,
        dimensions: 768,
        maxTokens: 8192,
        category: 'medium',
    },
    'bge-large-en-v1.5': {
        id: 'bge-large-en-v1.5',
        displayName: 'BGE Large EN v1.5',
        provider: 'BAAI',
        description: '1024-dim, ~1.3 GB. Best retrieval quality, slowest.',
        hf: { repo: 'BAAI/bge-large-en-v1.5', revision: 'main' },
        files: [
            { kind: 'weights', filename: 'model.safetensors', sha256: null },
            { kind: 'tokenizer', filename: 'tokenizer.json', sha256: null },
            { kind: 'config', filename: 'config.json', sha256: null },
        ],
        estimatedBytes: 1_340_000_000,
        dimensions: 1024,
        maxTokens: 512,
        category: 'large',
    },
    'gte-large': {
        id: 'gte-large',
        displayName: 'GTE Large',
        provider: 'Alibaba DAMO',
        description: '1024-dim, ~1.3 GB. Competitive with BGE Large.',
        hf: { repo: 'thenlper/gte-large', revision: 'main' },
        files: [
            { kind: 'weights', filename: 'model.safetensors', sha256: null },
            { kind: 'tokenizer', filename: 'tokenizer.json', sha256: null },
            { kind: 'config', filename: 'config.json', sha256: null },
        ],
        estimatedBytes: 1_340_000_000,
        dimensions: 1024,
        maxTokens: 512,
        category: 'large',
    },
};

export function listKnownEmbeddingModels() {
    return Object.values(KNOWN_EMBEDDING_MODELS).map((m) => ({ ...m }));
}

export function getKnownEmbeddingModel(modelId) {
    return KNOWN_EMBEDDING_MODELS[modelId] || null;
}

const CACHE_NAME = 'bw-models-v1';
const OPFS_DIR = 'model-downloads';
const PROGRESS_EMIT_MS = 200;

function _hasOpfs() {
    return typeof navigator !== 'undefined' && navigator.storage && typeof navigator.storage.getDirectory === 'function';
}

async function _getOpfsFile(modelId, filename) {
    if (!_hasOpfs()) return null;
    try {
        const root = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        const modelDir = await dlDir.getDirectoryHandle(modelId, { create: false });
        const fileHandle = await modelDir.getFileHandle(filename, { create: false });
        const file = await fileHandle.getFile();
        return file.size > 0 ? file : null;
    } catch (_) {
        return null;
    }
}

/**
 * Check if OPFS has any partial download data for this model.
 * Returns { hasData, totalBytes } or { hasData: false } if nothing found.
 */
export async function getPartialInfo(modelId) {
    if (!_hasOpfs()) return { hasData: false, totalBytes: 0 };
    try {
        const root = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        // Ollama-source ids live at `ollama/<name>__<tag>/` instead of
        // `<modelId>/`. Resolve the right path so partial-progress
        // bookkeeping works for both flows.
        let modelDir;
        const om = KNOWN_OLLAMA_MODELS[modelId];
        if (om) {
            const ollamaParent = await dlDir.getDirectoryHandle('ollama', { create: false });
            modelDir = await ollamaParent.getDirectoryHandle(
                `${om.ollama.name}__${om.ollama.tag}`,
                { create: false },
            );
        } else {
            modelDir = await dlDir.getDirectoryHandle(modelId, { create: false });
        }
        let totalBytes = 0;
        let fileCount = 0;
        for await (const [name, handle] of modelDir.entries()) {
            if (handle.kind !== 'file' || name.endsWith('.verified')) continue;
            const file = await handle.getFile();
            totalBytes += file.size;
            fileCount++;
        }
        return { hasData: fileCount > 0, totalBytes };
    } catch (_) {
        return { hasData: false, totalBytes: 0 };
    }
}

export function listKnownModels() {
    return Object.values(KNOWN_MODELS).map((m) => ({ ...m }));
}

export function getKnownModel(modelId) {
    return KNOWN_MODELS[modelId] || KNOWN_EMBEDDING_MODELS[modelId] || null;
}

/**
 * Build the Cache Storage key for a given (modelId, filename).
 * Uses HF's public `resolve` URL so we never pin a private mirror.
 *
 * @param {string} modelId
 * @param {string} filename
 * @returns {string}
 */
export function cacheKey(modelId, filename) {
    const m = getKnownModel(modelId);
    if (!m) throw new Error(`unknown model: ${modelId}`);
    const rev = encodeURIComponent(m.hf.revision || 'main');
    return `https://huggingface.co/${m.hf.repo}/resolve/${rev}/${filename}`;
}

// ── Concurrency state ──────────────────────────────────────────

const activeDownloads = new Map(); // modelId → { controller, startedAt, promise }

function _hasCaches() {
    return typeof caches !== 'undefined' && caches && typeof caches.open === 'function';
}

// ── Status / read API ──────────────────────────────────────────

/**
 * @param {string} modelId
 * @returns {Promise<boolean>}
 */
export async function isDownloaded(modelId) {
    const om = KNOWN_OLLAMA_MODELS[modelId];
    if (om) {
        return await isOllamaModelDownloaded(om.ollama.name, om.ollama.tag);
    }
    const m = getKnownModel(modelId);
    if (!m) return false;
    for (const f of m.files) {
        const opfsFile = await _getOpfsFile(modelId, f.filename);
        if (opfsFile) continue;
        if (!_hasCaches()) return false;
        const cache = await caches.open(CACHE_NAME);
        const hit = await cache.match(cacheKey(modelId, f.filename));
        if (!hit) return false;
    }
    return true;
}

/**
 * Return raw bytes for the registered files. Throws if any file is missing.
 *
 * @param {string} modelId
 * @returns {Promise<{weights: Uint8Array, tokenizer: Uint8Array}>}
 */
export async function getModelBytes(modelId) {
    const m = getKnownModel(modelId);
    if (!m) throw new Error(`unknown model: ${modelId}`);
    const out = {};
    for (const f of m.files) {
        const opfsFile = await _getOpfsFile(modelId, f.filename);
        if (opfsFile) {
            const buf = await opfsFile.arrayBuffer();
            out[f.kind] = new Uint8Array(buf);
            continue;
        }
        if (!_hasCaches()) throw new Error(`model not downloaded: ${modelId} (${f.filename})`);
        const cache = await caches.open(CACHE_NAME);
        const hit = await cache.match(cacheKey(modelId, f.filename));
        if (!hit) throw new Error(`model not downloaded: ${modelId} (${f.filename})`);
        const buf = await hit.arrayBuffer();
        out[f.kind] = new Uint8Array(buf);
    }
    if (!out.weights) throw new Error(`model ${modelId} missing weights file`);
    if (!out.tokenizer) {
        out.tokenizer = new Uint8Array(0);
    }
    return out;
}

/** Remove all files for a model. Cancels any active download first. */
export async function deleteModel(modelId) {
    cancelDownload(modelId);
    // Give the SW time to close its writable stream after abort.
    await new Promise((r) => setTimeout(r, 500));

    // Ollama-source models live under `model-downloads/ollama/<name>__<tag>/`,
    // not the per-id scheme below. Delete that subtree directly.
    const om = KNOWN_OLLAMA_MODELS[modelId];
    if (om && _hasOpfs()) {
        try {
            const root = await navigator.storage.getDirectory();
            const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
            const ollamaParent = await dlDir.getDirectoryHandle('ollama', { create: false });
            const dirName = `${om.ollama.name}__${om.ollama.tag}`;
            try {
                await ollamaParent.removeEntry(dirName, { recursive: true });
            } catch (e) {
                if (e && e.name !== 'NotFoundError') {
                    console.warn(`[bw] ollama delete failed: ${e.message}`);
                }
            }
        } catch (_) { /* OPFS not present or no ollama parent dir */ }
        events.dispatchEvent(new CustomEvent('model_deleted', { detail: { modelId } }));
        return;
    }

    const m = getKnownModel(modelId);
    if (m && _hasCaches()) {
        const cache = await caches.open(CACHE_NAME);
        for (const f of m.files) {
            try { await cache.delete(cacheKey(modelId, f.filename)); } catch (_err) { console.warn("[bw] caught:", _err); }
        }
    }
    // Clear OPFS files. Try directory removal first; if a lock blocks it,
    // fall back to removing individual files one by one.
    if (_hasOpfs()) {
        try {
            const root = await navigator.storage.getDirectory();
            const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
            try {
                await dlDir.removeEntry(modelId, { recursive: true });
            } catch (dirErr) {
                if (dirErr && dirErr.name === 'NoModificationAllowedError') {
                    console.warn('[bw] directory locked, removing files individually...');
                    try {
                        const modelDir = await dlDir.getDirectoryHandle(modelId, { create: false });
                        for await (const [name] of modelDir.entries()) {
                            try { await modelDir.removeEntry(name); } catch (fe) {
                                console.warn(`[bw] could not remove ${name}:`, fe.message);
                            }
                        }
                        try { await dlDir.removeEntry(modelId); } catch (_) {}
                    } catch (innerErr) {
                        console.warn('[bw] file-by-file cleanup failed:', innerErr.message);
                    }
                } else if (dirErr.name !== 'NotFoundError') {
                    console.warn('[bw] OPFS delete failed:', dirErr);
                }
            }
        } catch (_) { /* OPFS dir doesn't exist */ }
    }
    events.dispatchEvent(new CustomEvent('model_deleted', { detail: { modelId } }));
}

/** Abort an in-flight download for `modelId`. */
export function cancelDownload(modelId) {
    // Tell the SW to abort its fetch.
    if (typeof navigator !== 'undefined' && navigator.serviceWorker && navigator.serviceWorker.controller) {
        try { navigator.serviceWorker.controller.postMessage({ type: 'model_download_cancel', modelId }); } catch (_err) { console.warn("[bw] caught:", _err); }
    }
    // Abort the page-side controller (rejects the _downloadDirect promise).
    const a = activeDownloads.get(modelId);
    if (a && a.controller) {
        try { a.controller.abort(); } catch (_err) { console.debug("[bw] idempotent:", _err); }
    }
    // For the SW path: synthesize the cancel event locally so the
    // _downloadViaSW promise rejects immediately without waiting for
    // the SW to process and broadcast back.
    if (typeof navigator !== 'undefined' && navigator.serviceWorker) {
        navigator.serviceWorker.dispatchEvent(new MessageEvent('message', {
            data: { type: 'model_download_error', modelId, error: 'aborted' },
        }));
    }
}

// ── Download ───────────────────────────────────────────────────

class HfAuthRequiredError extends Error {
    constructor(message = 'Hugging Face token required') {
        super(message);
        this.name = 'HF_AUTH_REQUIRED';
    }
}

/**
 * Streaming download of every registered file for `modelId`. At most one
 * download is active globally; a second call returns the in-flight promise.
 *
 * @param {string} modelId
 * @param {object} [opts]
 * @param {(p: object) => void} [opts.onProgress]
 * @param {AbortSignal} [opts.signal]   external cancel signal
 * @param {string} [opts.hfToken]       HF access token for gated repos
 * @returns {Promise<void>}
 */
export async function downloadModel(modelId, opts = {}) {
    if (activeDownloads.has(modelId)) return activeDownloads.get(modelId).promise;
    for (const other of activeDownloads.values()) {
        await other.promise.catch((e) => { console.warn("[bw] swallowed:", e); });
    }
    if (!_hasOpfs() && !_hasCaches()) throw new Error('Neither OPFS nor Cache Storage available');

    // Ollama-source models route through the dedicated OCI Distribution
    // Spec downloader. Same `model_progress` event channel as the HF
    // path so the banner state machine works unchanged. Emit a final
    // `phase: 'ready'` event after the per-file events end — without
    // this the banner stays in the "downloading" phase forever even
    // though every byte is on disk.
    const om = KNOWN_OLLAMA_MODELS[modelId];
    if (om) {
        await downloadOllamaModel(om.ollama.name, om.ollama.tag, {
            modelId,
            signal: opts.signal,
            onProgress: opts.onProgress,
        });
        events.dispatchEvent(new CustomEvent('model_progress', {
            detail: { phase: 'ready', modelId },
        }));
        return;
    }

    // Priority 1: Dedicated Worker (zero-copy FileSystemSyncAccessHandle)
    if (_hasOpfs() && typeof Worker !== 'undefined') {
        console.log('[model-store] downloadModel: trying Dedicated Worker path for', modelId);
        try {
            return await _downloadViaWorker(modelId, opts);
        } catch (e) {
            if (e && e.name === 'AbortError') throw e;
            console.warn('[model-store] downloadModel: worker path failed, falling back:', e.message);
        }
    }

    // Priority 2: SW-delegated download (background resilience)
    if (typeof navigator !== 'undefined' && navigator.serviceWorker) {
        if (!navigator.serviceWorker.controller) {
            console.log('[model-store] downloadModel: SW controller is null, waiting for activation...');
            try {
                await navigator.serviceWorker.ready;
                console.log('[model-store] downloadModel: SW ready, controller =', !!navigator.serviceWorker.controller);
                if (!navigator.serviceWorker.controller) {
                    await new Promise((resolve) => {
                        const onCtrl = () => resolve();
                        navigator.serviceWorker.addEventListener('controllerchange', onCtrl, { once: true });
                        setTimeout(() => {
                            navigator.serviceWorker.removeEventListener('controllerchange', onCtrl);
                            resolve();
                        }, 5000);
                    });
                    console.log('[model-store] downloadModel: after wait, controller =', !!navigator.serviceWorker.controller);
                }
            } catch (_) { /* SW registration failed */ }
        }
        if (navigator.serviceWorker.controller) {
            console.log('[model-store] downloadModel: using SW path for', modelId);
            return _downloadViaSW(modelId, opts);
        }
    }

    // Priority 3: Page-side OPFS (last resort, no cache.put)
    console.log('[model-store] downloadModel: using direct (page-side) path for', modelId);
    return _downloadDirect(modelId, opts);
}

async function _downloadViaWorker(modelId, opts) {
    console.log('[model-store] _downloadViaWorker: starting for', modelId);

    const m = getKnownModel(modelId);
    if (!m) throw new Error(`unknown model: ${modelId}`);

    const controller = new AbortController();
    if (opts.signal) {
        if (opts.signal.aborted) controller.abort();
        else opts.signal.addEventListener('abort', () => controller.abort(), { once: true });
    }

    const startedAt = Date.now();
    const promise = (async () => {
        const root = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: true });
        const modelDir = await dlDir.getDirectoryHandle(modelId, { create: true });

        let totalBytesTotal = 0;
        let totalBytesDone = 0;
        let lastEmit = 0;

        const emitProgress = (file, fileBytesDone, fileBytesTotal, force = false) => {
            const now = Date.now();
            if (!force && now - lastEmit < PROGRESS_EMIT_MS) return;
            lastEmit = now;
            const elapsedSec = Math.max(0.001, (now - startedAt) / 1000);
            const throughputBps = totalBytesDone / elapsedSec;
            const remaining = Math.max(0, totalBytesTotal - totalBytesDone);
            const etaSeconds = throughputBps > 0 ? remaining / throughputBps : null;
            const detail = {
                phase: 'download',
                modelId,
                file: file.filename,
                fileKind: file.kind,
                fileBytesDone,
                fileBytesTotal,
                totalBytesDone,
                totalBytesTotal,
                throughputBps,
                etaSeconds,
            };
            try { if (typeof opts.onProgress === 'function') opts.onProgress(detail); } catch (_err) { console.warn("[bw] caught:", _err); }
            events.dispatchEvent(new CustomEvent('model_progress', { detail }));
        };

        for (const f of m.files) {
            const url = cacheKey(modelId, f.filename);

            // Check for .verified marker — skip if already done
            try {
                await modelDir.getFileHandle(f.filename + '.verified', { create: false });
                const existing = await (await modelDir.getFileHandle(f.filename, { create: false })).getFile();
                if (existing.size > 0) {
                    console.log(`[model-store] ${f.filename}: already verified (${existing.size} bytes), skipping`);
                    totalBytesTotal += existing.size;
                    totalBytesDone += existing.size;
                    emitProgress(f, existing.size, existing.size, true);
                    continue;
                }
            } catch (_) { /* no marker */ }

            // Check Cache Storage for legacy data
            if (_hasCaches()) {
                try {
                    const cache = await caches.open(CACHE_NAME);
                    const existing = await cache.match(url);
                    if (existing) {
                        const len = Number(existing.headers.get('content-length')) || 0;
                        if (len > 0) {
                            console.log(`[model-store] ${f.filename}: found in Cache Storage (${len} bytes)`);
                            totalBytesTotal += len;
                            totalBytesDone += len;
                            emitProgress(f, len, len, true);
                            continue;
                        }
                    }
                } catch (_) {}
            }

            // Get existing OPFS file size for resume
            let existingSize = 0;
            try {
                const fh = await modelDir.getFileHandle(f.filename, { create: false });
                existingSize = (await fh.getFile()).size;
            } catch (_) {}
            console.log(`[model-store] ${f.filename}: starting worker download (existingSize=${existingSize})`);

            // Use estimated total for progress until worker reports actual
            const estimatedTotal = m.estimatedBytes
                ? Math.max(existingSize, Math.round(m.estimatedBytes * (f.kind === 'weights' ? 0.99 : 0.01)))
                : 0;
            totalBytesTotal += estimatedTotal || existingSize;
            totalBytesDone += existingSize;

            // Spawn worker for this file
            const worker = new Worker(
                new URL('./opfs-writer-worker.js', import.meta.url),
                { type: 'module' },
            );

            const abortHandler = () => worker.postMessage({ type: 'cancel' });
            controller.signal.addEventListener('abort', abortHandler, { once: true });

            const fetchHeaders = {};
            if (opts.hfToken) fetchHeaders['Authorization'] = `Bearer ${opts.hfToken}`;

            await new Promise((resolve, reject) => {
                let prevBytesWritten = existingSize;
                let actualTotalKnown = false;

                worker.onmessage = (ev) => {
                    const msg = ev.data;
                    if (!msg || typeof msg !== 'object') return;

                    if (msg.type === 'progress') {
                        const delta = msg.bytesWritten - prevBytesWritten;
                        prevBytesWritten = msg.bytesWritten;
                        totalBytesDone += delta;
                        if (!actualTotalKnown && msg.totalBytes > 0) {
                            totalBytesTotal = totalBytesTotal - (estimatedTotal || existingSize) + msg.totalBytes;
                            actualTotalKnown = true;
                        }
                        emitProgress(f, msg.bytesWritten, msg.totalBytes);
                    } else if (msg.type === 'done') {
                        const delta = msg.totalBytes - prevBytesWritten;
                        if (delta > 0) totalBytesDone += delta;
                        if (!actualTotalKnown) {
                            totalBytesTotal = totalBytesTotal - (estimatedTotal || existingSize) + msg.totalBytes;
                        }
                        emitProgress(f, msg.totalBytes, msg.totalBytes, true);
                        console.log(`[model-store] ${f.filename}: worker done (${msg.totalBytes} bytes)`);
                        worker.terminate();
                        resolve();
                    } else if (msg.type === 'cancelled') {
                        console.log(`[model-store] ${f.filename}: worker cancelled at ${msg.bytesWritten} bytes`);
                        worker.terminate();
                        reject(new DOMException('aborted', 'AbortError'));
                    } else if (msg.type === 'error') {
                        console.error(`[model-store] ${f.filename}: worker error:`, msg.error);
                        worker.terminate();
                        reject(new Error(msg.error));
                    }
                };

                worker.onerror = (ev) => {
                    console.error('[model-store] worker onerror:', ev.message);
                    worker.terminate();
                    reject(new Error(ev.message || 'Worker error'));
                };

                worker.postMessage({
                    type: 'start',
                    modelId,
                    filename: f.filename,
                    url,
                    headers: fetchHeaders,
                    offset: existingSize,
                });
            });

            controller.signal.removeEventListener('abort', abortHandler);

            // SHA-256 verification
            if (f.sha256) {
                const opfsHandle = await modelDir.getFileHandle(f.filename, { create: false });
                const opfsFile = await opfsHandle.getFile();
                const SIZE_LIMIT = 2 * 1024 * 1024 * 1024;
                if (opfsFile.size <= SIZE_LIMIT) {
                    console.log(`[model-store] ${f.filename}: verifying SHA-256 (${opfsFile.size} bytes)...`);
                    events.dispatchEvent(new CustomEvent('model_progress', {
                        detail: { phase: 'verifying', modelId, file: f.filename },
                    }));
                    const ab = await opfsFile.arrayBuffer();
                    const hex = await sha256Hex(ab, {
                        modelId,
                        file: f,
                        fileBytesTotal: ab.byteLength,
                        totalBytesDoneBefore: totalBytesDone - ab.byteLength,
                        totalBytesTotalSoFar: totalBytesTotal,
                    });
                    if (hex !== f.sha256) {
                        console.error(`[model-store] ${f.filename}: SHA-256 mismatch! got=${hex} expected=${f.sha256}`);
                        try { await modelDir.removeEntry(f.filename); } catch (_) {}
                        throw new Error(`SHA-256 mismatch for ${f.filename}`);
                    }
                    console.log(`[model-store] ${f.filename}: SHA-256 verified OK`);
                } else {
                    console.log(`[model-store] ${f.filename}: skipping SHA-256 (${opfsFile.size} > 2 GB), size-check only`);
                }
            }

            // Write .verified marker
            try {
                const marker = await modelDir.getFileHandle(f.filename + '.verified', { create: true });
                const mw = await marker.createWritable();
                await mw.write('ok');
                await mw.close();
                console.log(`[model-store] ${f.filename}: .verified marker written`);
            } catch (markerErr) {
                console.warn(`[model-store] ${f.filename}: failed to write .verified marker:`, markerErr.message);
            }
        }
    })();

    activeDownloads.set(modelId, { controller, startedAt, promise });
    try {
        await promise;
        console.log('[model-store] _downloadViaWorker: all files complete for', modelId);
        events.dispatchEvent(new CustomEvent('model_progress', {
            detail: { phase: 'ready', modelId },
        }));
    } finally {
        activeDownloads.delete(modelId);
    }
}

async function _downloadViaSW(modelId, opts) {
    const m = getKnownModel(modelId);
    if (!m) throw new Error(`unknown model: ${modelId}`);

    const files = m.files.map((f) => ({
        url: cacheKey(modelId, f.filename),
        filename: f.filename,
        kind: f.kind,
        sha256: f.sha256,
    }));

    const controller = new AbortController();
    if (opts.signal) {
        if (opts.signal.aborted) controller.abort();
        else opts.signal.addEventListener('abort', () => {
            navigator.serviceWorker.controller.postMessage({ type: 'model_download_cancel', modelId });
            controller.abort();
        }, { once: true });
    }

    console.log('[model-store] _downloadViaSW:', modelId, files.length, 'files');

    const promise = new Promise((resolve, reject) => {
        let gotFirstEvent = false;

        // Timeout: if no progress or done event within 120s, the SW
        // handler probably crashed. Longer than 30s because the assembly
        // of IDB chunks into Cache Storage can take minutes for large
        // chunk counts from pre-batching downloads.
        const startupTimeout = setTimeout(() => {
            if (!gotFirstEvent) {
                cleanup();
                reject(new Error('SW download timeout — no response from service worker after 120s'));
            }
        }, 120000);

        const onMessage = (event) => {
            const msg = event.data;
            if (!msg || typeof msg !== 'object') return;

            if (msg.type === 'model_progress' && msg.detail && msg.detail.modelId === modelId) {
                gotFirstEvent = true;
                clearTimeout(startupTimeout);
                try { if (typeof opts.onProgress === 'function') opts.onProgress(msg.detail); } catch (_e) {}
                events.dispatchEvent(new CustomEvent('model_progress', { detail: msg.detail }));
            } else if (msg.type === 'model_download_done' && msg.modelId === modelId) {
                cleanup();
                events.dispatchEvent(new CustomEvent('model_progress', {
                    detail: { phase: 'ready', modelId },
                }));
                resolve();
            } else if (msg.type === 'model_download_error' && msg.modelId === modelId) {
                cleanup();
                if (msg.error === 'HF_AUTH_REQUIRED') {
                    reject(new HfAuthRequiredError());
                } else if (msg.error === 'aborted' || (msg.error && msg.error.includes('abort'))) {
                    reject(new DOMException('aborted', 'AbortError'));
                } else {
                    reject(new Error(msg.error || 'SW download failed'));
                }
            }
        };
        const cleanup = () => {
            clearTimeout(startupTimeout);
            navigator.serviceWorker.removeEventListener('message', onMessage);
            activeDownloads.delete(modelId);
        };
        navigator.serviceWorker.addEventListener('message', onMessage);
    });

    activeDownloads.set(modelId, { controller, startedAt: Date.now(), promise });

    navigator.serviceWorker.controller.postMessage({
        type: 'model_download_start',
        modelId,
        files,
        hfToken: opts.hfToken || null,
    });

    try {
        await promise;
    } finally {
        activeDownloads.delete(modelId);
    }
}

async function _downloadDirect(modelId, opts) {
    console.log('[model-store] _downloadDirect: starting page-side download for', modelId);

    const m = getKnownModel(modelId);
    if (!m) throw new Error(`unknown model: ${modelId}`);

    const useOpfs = _hasOpfs();

    const controller = new AbortController();
    if (opts.signal) {
        if (opts.signal.aborted) controller.abort();
        else opts.signal.addEventListener('abort', () => controller.abort(), { once: true });
    }

    const startedAt = Date.now();
    const promise = (async () => {
        let modelDir = null;
        if (useOpfs) {
            const root = await navigator.storage.getDirectory();
            const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: true });
            modelDir = await dlDir.getDirectoryHandle(modelId, { create: true });
        }

        let totalBytesTotal = 0;
        let totalBytesDone = 0;
        let lastEmit = 0;

        const emitProgress = (file, fileBytesDone, fileBytesTotal, force = false) => {
            const now = Date.now();
            if (!force && now - lastEmit < PROGRESS_EMIT_MS) return;
            lastEmit = now;
            const elapsedSec = Math.max(0.001, (now - startedAt) / 1000);
            const throughputBps = totalBytesDone / elapsedSec;
            const remaining = Math.max(0, totalBytesTotal - totalBytesDone);
            const etaSeconds = throughputBps > 0 ? remaining / throughputBps : null;
            const detail = {
                phase: 'download',
                modelId,
                file: file.filename,
                fileKind: file.kind,
                fileBytesDone,
                fileBytesTotal,
                totalBytesDone,
                totalBytesTotal,
                throughputBps,
                etaSeconds,
            };
            try { if (typeof opts.onProgress === 'function') opts.onProgress(detail); } catch (_err) { console.warn("[bw] caught:", _err); }
            events.dispatchEvent(new CustomEvent('model_progress', { detail }));
        };

        for (const f of m.files) {
            const url = cacheKey(modelId, f.filename);

            // Check for .verified marker in OPFS — already done
            if (modelDir) {
                try {
                    await modelDir.getFileHandle(f.filename + '.verified', { create: false });
                    const existing = await (await modelDir.getFileHandle(f.filename, { create: false })).getFile();
                    if (existing.size > 0) {
                        console.log(`[model-store] ${f.filename}: already verified in OPFS (${existing.size} bytes)`);
                        totalBytesTotal += existing.size;
                        totalBytesDone += existing.size;
                        emitProgress(f, existing.size, existing.size, true);
                        continue;
                    }
                } catch (_) { /* no marker or no file */ }
            }

            // Check Cache Storage for legacy/already-cached data
            if (_hasCaches()) {
                const cache = await caches.open(CACHE_NAME);
                const existing = await cache.match(url);
                if (existing) {
                    if (!f.sha256) {
                        const len = Number(existing.headers.get('content-length')) || 0;
                        if (len > 0) {
                            console.log(`[model-store] ${f.filename}: found in Cache Storage, no pin (${len} bytes)`);
                            totalBytesTotal += len;
                            totalBytesDone += len;
                            emitProgress(f, len, len, true);
                            continue;
                        }
                    } else {
                        try {
                            const buf = await existing.clone().arrayBuffer();
                            const len = buf.byteLength;
                            const hex = await sha256Hex(buf, {
                                modelId,
                                file: f,
                                fileBytesTotal: len,
                                totalBytesDoneBefore: totalBytesDone,
                                totalBytesTotalSoFar: totalBytesTotal + len,
                            });
                            if (hex === f.sha256) {
                                console.log(`[model-store] ${f.filename}: Cache Storage verified OK (${len} bytes)`);
                                totalBytesTotal += len;
                                totalBytesDone += len;
                                emitProgress(f, len, len, true);
                                continue;
                            }
                            console.warn(`[model-store] ${f.filename}: Cache Storage SHA mismatch, re-downloading`);
                            await cache.delete(url);
                        } catch (_) {
                            await cache.delete(url);
                        }
                    }
                }
            }

            // OPFS required for actual downloads
            if (!modelDir) throw new Error('OPFS unavailable — cannot download without service worker');

            // Open OPFS file, check for partial download
            const opfsHandle = await modelDir.getFileHandle(f.filename, { create: true });
            let existingSize = 0;
            try { existingSize = (await opfsHandle.getFile()).size; } catch (_) {}
            console.log(`[model-store] ${f.filename}: OPFS partial = ${existingSize} bytes`);

            // Fetch with Range header for resume
            const fetchHeaders = {};
            if (opts.hfToken) fetchHeaders['Authorization'] = `Bearer ${opts.hfToken}`;
            if (existingSize > 0) fetchHeaders['Range'] = `bytes=${existingSize}-`;

            let resp;
            try {
                resp = await fetch(url, { headers: fetchHeaders, signal: controller.signal });
            } catch (e) {
                if (controller.signal.aborted) throw new DOMException('aborted', 'AbortError');
                throw e;
            }

            if (resp.status === 401 || resp.status === 403) {
                throw new HfAuthRequiredError(`HF responded ${resp.status} for ${f.filename}`);
            }

            if (resp.status === 416) {
                // Range not satisfiable — file already complete
                const opfsFile = await opfsHandle.getFile();
                console.log(`[model-store] ${f.filename}: 416 — already complete (${opfsFile.size} bytes)`);
                totalBytesTotal += opfsFile.size;
                totalBytesDone += opfsFile.size;
                emitProgress(f, opfsFile.size, opfsFile.size, true);
            } else if (resp.ok || resp.status === 206) {
                const resuming = resp.status === 206;
                if (resp.status === 200 && existingSize > 0) existingSize = 0;
                const contentLength = Number(resp.headers.get('content-length')) || 0;
                const fileBytesTotal = existingSize + contentLength;
                let fileBytesDone = existingSize;
                totalBytesTotal += fileBytesTotal;
                totalBytesDone += fileBytesDone;
                console.log(`[model-store] ${f.filename}: fetching (status=${resp.status}, resume=${resuming}, contentLength=${contentLength}, total=${fileBytesTotal})`);

                // Stream to OPFS
                let writable;
                try {
                    writable = await opfsHandle.createWritable({ keepExistingData: resuming && existingSize > 0 });
                    if (resuming && existingSize > 0) await writable.seek(existingSize);
                } catch (lockErr) {
                    console.warn('[model-store] OPFS lock, resetting file:', lockErr.message);
                    try { await modelDir.removeEntry(f.filename); } catch (_) {}
                    const newHandle = await modelDir.getFileHandle(f.filename, { create: true });
                    writable = await newHandle.createWritable();
                    totalBytesTotal -= existingSize;
                    totalBytesDone -= existingSize;
                    fileBytesDone = 0;
                }

                const reader = resp.body.getReader();
                try {
                    while (true) {
                        const { value, done } = await reader.read();
                        if (done) break;
                        await writable.write(value);
                        fileBytesDone += value.byteLength;
                        totalBytesDone += value.byteLength;
                        emitProgress(f, fileBytesDone, fileBytesTotal);
                    }
                    await writable.close();
                    console.log(`[model-store] ${f.filename}: OPFS write complete (${fileBytesDone} bytes)`);
                } catch (e) {
                    try { await writable.close(); } catch (_) {}
                    if (controller.signal.aborted) throw new DOMException('aborted', 'AbortError');
                    throw e;
                }
                emitProgress(f, fileBytesDone, fileBytesTotal, true);
            } else {
                throw new Error(`HF fetch failed (${resp.status}) for ${f.filename}`);
            }

            // SHA-256 verification
            if (f.sha256) {
                const opfsFile = await opfsHandle.getFile();
                const SIZE_LIMIT = 2 * 1024 * 1024 * 1024;
                if (opfsFile.size <= SIZE_LIMIT) {
                    console.log(`[model-store] ${f.filename}: verifying SHA-256 (${opfsFile.size} bytes)...`);
                    events.dispatchEvent(new CustomEvent('model_progress', {
                        detail: { phase: 'verifying', modelId, file: f.filename },
                    }));
                    const ab = await opfsFile.arrayBuffer();
                    const hex = await sha256Hex(ab, {
                        modelId,
                        file: f,
                        fileBytesTotal: ab.byteLength,
                        totalBytesDoneBefore: totalBytesDone - ab.byteLength,
                        totalBytesTotalSoFar: totalBytesTotal,
                    });
                    if (hex !== f.sha256) {
                        console.error(`[model-store] ${f.filename}: SHA-256 mismatch! got=${hex} expected=${f.sha256}`);
                        try { await modelDir.removeEntry(f.filename); } catch (_) {}
                        throw new Error(`SHA-256 mismatch for ${f.filename}`);
                    }
                    console.log(`[model-store] ${f.filename}: SHA-256 verified OK`);
                } else {
                    console.log(`[model-store] ${f.filename}: skipping SHA-256 (${opfsFile.size} > 2 GB), size-check only`);
                }
            }

            // Write .verified marker
            try {
                const marker = await modelDir.getFileHandle(f.filename + '.verified', { create: true });
                const mw = await marker.createWritable();
                await mw.write('ok');
                await mw.close();
                console.log(`[model-store] ${f.filename}: .verified marker written`);
            } catch (markerErr) {
                console.warn(`[model-store] ${f.filename}: failed to write .verified marker:`, markerErr.message);
            }
        }
    })();

    activeDownloads.set(modelId, { controller, startedAt, promise });
    try {
        await promise;
        console.log('[model-store] _downloadDirect: all files complete for', modelId);
        events.dispatchEvent(new CustomEvent('model_progress', {
            detail: { phase: 'ready', modelId },
        }));
    } finally {
        activeDownloads.delete(modelId);
    }
}

// ── Hash helper ────────────────────────────────────────────────

/**
 * Compute the SHA-256 of `buf`, emitting `phase: 'verifying'` events on
 * `state.events` so the UI can show "Verifying SHA-256…" while the work
 * happens.
 *
 * NOTE: `crypto.subtle.digest` does not support streaming — calling it
 * per chunk would yield independent hashes, not a single rolling one.
 * For now we just yield to the event loop right before and after the
 * single-shot digest call so the banner repaints between phase changes.
 * Once we want fully-incremental progress, swap in a pure-JS streaming
 * SHA-256 implementation (e.g. js-sha256 vendored as a small dep).
 *
 * @param {ArrayBuffer} buf
 * @param {object} [ctx]   optional progress context (modelId, file, …)
 * @returns {Promise<string>} hex digest
 */
async function sha256Hex(buf, ctx = null) {
    const total = buf.byteLength;
    const emit = (bytesProcessed) => {
        if (!ctx || !ctx.file) return;
        const fileTotal = ctx.fileBytesTotal != null ? ctx.fileBytesTotal : total;
        const totalDoneBefore = ctx.totalBytesDoneBefore || 0;
        const totalTotal = ctx.totalBytesTotalSoFar || fileTotal;
        const totalBytesDone = totalDoneBefore + bytesProcessed;
        const percent = fileTotal > 0
            ? Math.min(100, Math.floor((bytesProcessed / fileTotal) * 100))
            : null;
        const detail = {
            phase: 'verifying',
            modelId: ctx.modelId,
            file: ctx.file.filename,
            fileKind: ctx.file.kind,
            fileBytesDone: bytesProcessed,
            fileBytesTotal: fileTotal,
            totalBytesDone,
            totalBytesTotal: totalTotal,
            percent,
        };
        events.dispatchEvent(new CustomEvent('model_progress', { detail }));
    };

    // Emit "verifying started" and let the main thread repaint before
    // the digest call blocks for a few seconds.
    emit(0);
    await new Promise((r) => setTimeout(r, 0));

    const digest = await crypto.subtle.digest('SHA-256', buf);

    // Yield once more so the banner can flip to the next phase before
    // the next big main-thread call (e.g. wasm.init_local_model).
    await new Promise((r) => setTimeout(r, 0));
    emit(total);

    const bytes = new Uint8Array(digest);
    let out = '';
    for (let i = 0; i < bytes.length; i++) {
        out += bytes[i].toString(16).padStart(2, '0');
    }
    return out;
}

export const HFAuthRequired = HfAuthRequiredError;

// Bring Ollama-format download API into local scope so `downloadModel`
// can route ollama-source modelIds through it, then re-export so
// callers (ui-chat.js, etc.) have a single import point for both HF
// and Ollama models. Implementation stays in `ollama-download.js`.
import {
    downloadOllamaModel,
    isOllamaModelDownloaded,
    getOllamaModelBytes,
    ollamaModelInfo,
} from './ollama-download.js';
export {
    downloadOllamaModel,
    isOllamaModelDownloaded,
    getOllamaModelBytes,
    ollamaModelInfo,
};

/** Resolve a known model regardless of source (HF or Ollama). */
export function getKnownModelAny(modelId) {
    return KNOWN_MODELS[modelId]
        || KNOWN_OLLAMA_MODELS[modelId]
        || KNOWN_EMBEDDING_MODELS[modelId]
        || null;
}

/** Combined list of HF + Ollama chat models for UI pickers. */
export function listAllChatModels() {
    return [
        ...Object.values(KNOWN_MODELS),
        ...Object.values(KNOWN_OLLAMA_MODELS),
    ];
}
