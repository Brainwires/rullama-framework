// brainwires-chat-pwa — local-model Web Worker
//
// Phase 2 of "make-a-plan-to-bright-scroll": move the WASM module
// entirely off the main thread. Workers have full access to
// `caches` (Cache Storage), so we read the multi-GB model bytes here
// instead of postMessage'ing them across the thread boundary.
//
// Wire protocol (main → worker):
//
//   { requestId, type: 'load',   modelId }
//   { requestId, type: 'chat',   conversationId, messageId, messages, params }
//   { requestId, type: 'cancel', conversationId }
//   { requestId, type: 'unload' }
//
// Wire protocol (worker → main):
//
//   { type: 'load_progress', phase: 'loading'|'ready', modelId }
//   { requestId, type: 'load_done',     modelId }
//   { requestId, type: 'load_error',    error }
//   { type: 'chat_chunk',  conversationId, messageId, delta }
//   { requestId, type: 'chat_done',     conversationId, messageId, usage, tokensReceived }
//   { requestId, type: 'chat_error',    conversationId, messageId, error }
//   { requestId, type: 'cancel_ack',    conversationId }
//   { requestId, type: 'unload_ack' }
//
// The worker's state:
//   - `wasm`     : the wasm-pack module (lazy-init'd on first 'load').
//   - `handle`   : the active LocalModelHandle, or null.
//   - `modelId`  : id of the loaded model.
//   - `inflight` : Map<conversationId, { aborted: boolean, reader }>.

// Resolved against the worker's actual runtime location. The worker
// is always served from `web/local-worker.js` (esbuild bundles to the
// web root), so `./pkg/...` lands on the wasm-pack output directory.
const PKG_URL = new URL('./pkg/brainwires_chat_pwa.js', import.meta.url).href;
const CACHE_NAME = 'bw-models-v1';
const OPFS_DIR = 'model-downloads';

// Ollama OPFS reader — only imported lazily because the worker may
// load before the rest of the app and we don't want the module graph
// to stall on this path when no Ollama model is selected.
let _ollamaModule = null;
async function _getOllamaModule() {
    if (_ollamaModule) return _ollamaModule;
    _ollamaModule = await import('./ollama-download.js');
    return _ollamaModule;
}

/// Fetch a companion tokenizer.json from HuggingFace and stash it in OPFS
/// alongside the Ollama GGUF blob. Used when the Ollama manifest doesn't
/// include a `tokenizer` layer (most current publications). One-shot:
/// the cached copy is re-used on subsequent loads.
async function _fetchOllamaTokenizerCompanion(om) {
    if (!om.tokenizerCompanion) return null;
    const { repo, revision, filename } = om.tokenizerCompanion;
    const cacheKey = `tokenizer.json`;
    // Try OPFS first.
    const root = await navigator.storage.getDirectory();
    const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: true });
    const ollamaParent = await dlDir.getDirectoryHandle('ollama', { create: true });
    const modelDir = await ollamaParent.getDirectoryHandle(
        `${om.ollama.name}__${om.ollama.tag}`,
        { create: true },
    );
    try {
        const fh = await modelDir.getFileHandle(cacheKey, { create: false });
        const file = await fh.getFile();
        if (file.size > 0) {
            console.log(`[local-worker] reusing cached companion tokenizer (${file.size} bytes)`);
            return new Uint8Array(await file.arrayBuffer());
        }
    } catch { /* not cached yet */ }

    const url = `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision || 'main')}/${filename}`;
    console.log(`[local-worker] fetching companion tokenizer from ${url}`);
    const resp = await fetch(url);
    if (!resp.ok) {
        throw new Error(`companion tokenizer fetch failed: ${resp.status} ${resp.statusText}`);
    }
    const bytes = new Uint8Array(await resp.arrayBuffer());

    // Write to OPFS for reuse. SyncAccessHandle for atomicity.
    try {
        const fh = await modelDir.getFileHandle(cacheKey, { create: true });
        const sync = await fh.createSyncAccessHandle();
        try {
            sync.truncate(0);
            sync.write(bytes, { at: 0 });
            sync.flush();
        } finally {
            sync.close();
        }
    } catch (e) {
        console.warn(`[local-worker] failed to cache companion tokenizer: ${e.message}`);
    }
    return bytes;
}

async function _getOpfsFile(modelId, filename) {
    if (typeof navigator === 'undefined' || !navigator.storage || !navigator.storage.getDirectory) return null;
    try {
        const root = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        const modelDir = await dlDir.getDirectoryHandle(modelId, { create: false });
        const fh = await modelDir.getFileHandle(filename, { create: false });
        const file = await fh.getFile();
        return file.size > 0 ? file : null;
    } catch (_) {
        return null;
    }
}

// Mirror of the registry in src/model-store.js. Kept in lock-step; the
// only fields we actually need here are the HF (repo, revision), the list
// of files (kind, filename) so we can build the cache key, and a
// `multimodal` flag that selects the vision-capable loader path.
const KNOWN_MODELS = {
    'gemma-4-e2b-it': {
        id: 'gemma-4-e2b-it',
        source: 'hf',
        hf: { repo: 'google/gemma-4-e2b-it', revision: 'main' },
        files: [
            { kind: 'weights', filename: 'model.safetensors' },
            { kind: 'tokenizer', filename: 'tokenizer.json' },
        ],
        // Gemma 4 E2B IT ships with the SigLIP vision tower — load via
        // `init_local_multimodal*` so `vision_chat` works end-to-end.
        multimodal: true,
    },
};

// Mirror of `KNOWN_OLLAMA_MODELS` in src/model-store.js. Kept here so
// `getModelBytes` / `handleLoad` can route an ollama-source modelId to
// the OPFS path used by `ollama-download.js` instead of the HF-only
// `KNOWN_MODELS` lookup. Tokenizer for Ollama models comes from a
// companion HF safetensors tokenizer.json (Ollama embeds the tokenizer
// in the GGUF metadata, but extraction isn't wired up yet).
const KNOWN_OLLAMA_MODELS = {
    'gemma4:e2b': {
        id: 'gemma4:e2b',
        source: 'ollama',
        ollama: { name: 'gemma4', tag: 'e2b' },
        // Tokenizer companion — pulled from the same HF repo as the
        // safetensors path so it doesn't duplicate the download.
        tokenizerCompanion: {
            repo: 'google/gemma-4-e2b-it',
            revision: 'main',
            filename: 'tokenizer.json',
        },
        // Ollama's gemma4:e2b is text-only (no vision tower in the GGUF).
        multimodal: false,
    },
};

function getKnownModelAny(modelId) {
    return KNOWN_MODELS[modelId] || KNOWN_OLLAMA_MODELS[modelId] || null;
}

function cacheKey(modelId, filename) {
    const m = KNOWN_MODELS[modelId];
    if (!m) throw new Error(`unknown model: ${modelId}`);
    const rev = encodeURIComponent(m.hf.revision || 'main');
    return `https://huggingface.co/${m.hf.repo}/resolve/${rev}/${filename}`;
}

let wasm = null;
let wasmPromise = null;
let handle = null;
let loadedModelId = null;
// Track whether the active handle is multimodal so we can fail fast on
// `vision_chat` against a text-only handle (and vice versa) instead of
// silently calling the wrong wasm export.
let handleIsMultimodal = false;
// `true` when the active handle is a `LocalQuantizedHandle` (text-only,
// QMatMul-backed). Routes chat to `local_chat_stream_quantized` and
// makes vision_chat fail-fast with a clear error.
let handleIsQuantized = false;

// OPFS `FileSystemSyncAccessHandle` for the active multimodal weights
// file. The WASM side keeps the chunked-loader `readFn` alive past init
// for three streaming paths:
//   1. Per-layer-embedding (PLE) OPFS streaming — reads ~15 KB per forward
//      pass to back the 4.7 GB `embed_tokens_per_layer.weight` table that
//      can't fit in a single WGPU/wasm buffer.
//   2. Lazy vision tower — streams ~211 deferred tensors on the first
//      `attach_vision()` call (first image).
//   3. Lazy audio tower — same idea on `attach_audio()`.
// All three `read_fn`s capture this sync handle, so it must outlive
// init and only close on `unload` / model swap.
let multimodalWeightsSyncHandle = null;

function closeMultimodalWeightsHandle() {
    if (multimodalWeightsSyncHandle) {
        try { multimodalWeightsSyncHandle.close(); } catch (_) {}
        multimodalWeightsSyncHandle = null;
    }
}

// Embedding model lives in a separate slot from the chat handle — RAG
// indexing and assistant generation can run with different weights.
let embedHandle = null;
let embedModelId = null;

const inflight = new Map();

async function getWasm() {
    if (wasm) return wasm;
    if (wasmPromise) return wasmPromise;
    wasmPromise = (async () => {
        const mod = await import(PKG_URL);
        if (typeof mod.default === 'function') await mod.default();
        if (typeof mod.init === 'function') {
            try { mod.init(); } catch (_err) { console.debug("[bw] idempotent:", _err); }
        }
        // One-shot WebGPU adapter probe. Logs vendor/architecture/device so
        // we can identify software renderers (llvmpipe / SwiftShader) that
        // explain orders-of-magnitude slowdown vs native. Best-effort: if
        // navigator.gpu or requestAdapter throws, we just skip the log.
        try {
            if (typeof navigator !== 'undefined' && navigator.gpu) {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    const info = adapter.info || {};
                    const limits = adapter.limits || {};
                    console.log('[local-worker] webgpu adapter:', {
                        vendor: info.vendor || '?',
                        architecture: info.architecture || '?',
                        device: info.device || '?',
                        description: info.description || '?',
                        isFallbackAdapter: adapter.isFallbackAdapter === true,
                        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
                        maxBufferSize: limits.maxBufferSize,
                        maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
                    });
                } else {
                    console.warn('[local-worker] navigator.gpu.requestAdapter returned null');
                }
            } else {
                console.warn('[local-worker] navigator.gpu unavailable in this worker context');
            }
        } catch (e) {
            console.warn('[local-worker] adapter probe failed:', e);
        }
        wasm = mod;
        return mod;
    })();
    return wasmPromise;
}

async function isDownloaded(modelId) {
    if (KNOWN_OLLAMA_MODELS[modelId]) {
        const om = KNOWN_OLLAMA_MODELS[modelId];
        const { isOllamaModelDownloaded } = await _getOllamaModule();
        return await isOllamaModelDownloaded(om.ollama.name, om.ollama.tag);
    }
    const m = KNOWN_MODELS[modelId];
    if (!m) return false;
    for (const f of m.files) {
        const opfsFile = await _getOpfsFile(modelId, f.filename);
        if (opfsFile) continue;
        if (typeof caches === 'undefined') return false;
        const cache = await caches.open(CACHE_NAME);
        const hit = await cache.match(cacheKey(modelId, f.filename));
        if (!hit) return false;
    }
    return true;
}

async function getModelBytes(modelId) {
    if (KNOWN_OLLAMA_MODELS[modelId]) {
        const om = KNOWN_OLLAMA_MODELS[modelId];
        const { getOllamaModelBytes } = await _getOllamaModule();
        const { bytes } = await getOllamaModelBytes(om.ollama.name, om.ollama.tag);
        if (!bytes.weights) {
            throw new Error(`ollama model ${modelId} missing weights blob`);
        }
        let tokenizer = bytes.tokenizer || null;
        if (!tokenizer && om.tokenizerCompanion) {
            tokenizer = await _fetchOllamaTokenizerCompanion(om);
        }
        if (!tokenizer || tokenizer.byteLength === 0) {
            throw new Error(
                `ollama model ${modelId} has no tokenizer layer in the manifest and no tokenizerCompanion configured`
            );
        }
        return { weights: bytes.weights, tokenizer };
    }
    const m = KNOWN_MODELS[modelId];
    if (!m) throw new Error(`unknown model: ${modelId}`);
    const out = {};
    for (const f of m.files) {
        let bytes = null;

        // Priority 1: Read from OPFS via FileSystemSyncAccessHandle.
        // File.arrayBuffer() throws NotReadableError on large OPFS files
        // in Chrome. The sync API bypasses that code path entirely.
        if (typeof navigator !== 'undefined' && navigator.storage && navigator.storage.getDirectory) {
            try {
                const root = await navigator.storage.getDirectory();
                const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
                const modelDir = await dlDir.getDirectoryHandle(modelId, { create: false });
                const fh = await modelDir.getFileHandle(f.filename, { create: false });
                const syncHandle = await fh.createSyncAccessHandle();
                try {
                    const size = syncHandle.getSize();
                    if (size > 0) {
                        bytes = new Uint8Array(size);
                        syncHandle.read(bytes, { at: 0 });
                        console.log(`[local-worker] ${f.filename}: read ${size} bytes from OPFS (sync handle)`);
                    }
                } finally {
                    syncHandle.close();
                }
            } catch (e) {
                console.warn(`[local-worker] ${f.filename}: sync handle read failed:`, e.message);
            }
        }

        // Priority 2: Read from OPFS via async File API (works for small files).
        if (!bytes) {
            const opfsFile = await _getOpfsFile(modelId, f.filename);
            if (opfsFile) {
                try {
                    const ab = await opfsFile.arrayBuffer();
                    bytes = new Uint8Array(ab);
                    console.log(`[local-worker] ${f.filename}: read ${bytes.byteLength} bytes from OPFS (async)`);
                } catch (e) {
                    console.warn(`[local-worker] ${f.filename}: async OPFS read failed:`, e.message);
                }
            }
        }

        // Priority 3: Fall back to Cache Storage (legacy data).
        if (!bytes) {
            if (typeof caches === 'undefined') throw new Error(`model not downloaded: ${modelId} (${f.filename})`);
            const cache = await caches.open(CACHE_NAME);
            const hit = await cache.match(cacheKey(modelId, f.filename));
            if (!hit) throw new Error(`model not downloaded: ${modelId} (${f.filename})`);
            const ab = await hit.arrayBuffer();
            bytes = new Uint8Array(ab);
            console.log(`[local-worker] ${f.filename}: read ${bytes.byteLength} bytes from Cache Storage`);
        }

        out[f.kind] = bytes;
    }
    if (!out.weights) throw new Error(`model ${modelId} missing weights file`);
    if (!out.tokenizer) out.tokenizer = new Uint8Array(0);
    return out;
}

// ── Message dispatch ───────────────────────────────────────────

self.addEventListener('message', (ev) => {
    const msg = ev.data;
    if (!msg || typeof msg !== 'object') return;
    switch (msg.type) {
        case 'load':         handleLoad(msg);   break;
        case 'chat':         handleChat(msg);   break;
        case 'vision_chat':  handleVisionChat(msg); break;
        case 'embed_load':   handleEmbedLoad(msg); break;
        case 'embed_text':   handleEmbedText(msg); break;
        case 'embed_unload': handleEmbedUnload(msg); break;
        case 'cancel':       handleCancel(msg); break;
        case 'unload':       handleUnload(msg); break;
        default: console.error('local-worker: unknown message type', msg.type);
    }
});

async function handleLoad(msg) {
    const { requestId, modelId, diag: msgDiag, diagLayer: msgDiagLayer } = msg;
    try {
        if (handle && loadedModelId === modelId) {
            self.postMessage({ requestId, type: 'load_done', modelId });
            return;
        }
        if (!(await isDownloaded(modelId))) {
            self.postMessage({ requestId, type: 'load_error', error: 'not_downloaded' });
            return;
        }

        self.postMessage({ type: 'load_progress', phase: 'loading', modelId });

        const mod = await getWasm();

        // Drop any previously-loaded handle before replacing it; halves
        // the peak heap footprint when switching models.
        if (handle && typeof handle.free === 'function') {
            try { handle.free(); } catch (_err) { console.debug("[bw] idempotent:", _err); }
        }
        handle = null;
        loadedModelId = null;
        handleIsMultimodal = false;
        handleIsQuantized = false;
        closeMultimodalWeightsHandle();

        // Ollama-source models route through the perf-bearing quantized
        // path when the wasm crate exposes
        // `init_local_multimodal_gguf_quantized` (Phase 5 path —
        // QMatMul over q4_k.pwgsl). Falls back to the dequant-at-load
        // path otherwise.
        if (KNOWN_OLLAMA_MODELS[modelId]) {
            const useQuantized =
                typeof mod.init_local_multimodal_gguf_quantized === 'function';
            // Chunked path keeps the GGUF blob out of JS heap entirely.
            // The full Ollama gemma4:e2b file is ~7 GB; even the LM-only
            // subset (~1.6 GB) overflows `new Uint8Array(N)` in Chrome.
            const useChunked =
                useQuantized
                && typeof mod.init_local_multimodal_gguf_quantized_chunked === 'function';
            if (useChunked) {
                const loaded = await tryChunkedOllamaLoad(mod, modelId, requestId);
                if (loaded) return;
                console.log('[local-worker] chunked GGUF load unavailable, falling back to bulk read');
            }
            const initFn = useQuantized
                ? mod.init_local_multimodal_gguf_quantized
                : mod.init_local_multimodal_gguf;
            if (typeof initFn !== 'function') {
                self.postMessage({
                    requestId,
                    type: 'load_error',
                    error:
                        'no GGUF entry point available — rebuild the WASM crate',
                });
                return;
            }
            // `getModelBytes` already handles the companion-tokenizer
            // fallback for ollama-source ids; if it returns at all the
            // bytes are valid.
            let { weights, tokenizer } = await getModelBytes(modelId);
            try {
                handle = await initFn(weights, tokenizer, modelId);
            } finally {
                weights = null;
                tokenizer = null;
            }
            loadedModelId = modelId;
            if (useQuantized) {
                handleIsQuantized = true;
                handleIsMultimodal = false;
            } else {
                handleIsMultimodal = true; // Gemma4MultiModal handle, even though vision is off
                handleIsQuantized = false;
            }
            self.postMessage({ requestId, type: 'load_done', modelId });
            self.postMessage({ type: 'load_progress', phase: 'ready', modelId });
            return;
        }

        const m = KNOWN_MODELS[modelId];
        const multimodal = !!(m && m.multimodal);

        if (multimodal && typeof mod.init_local_multimodal_chunked === 'function') {
            const loaded = await tryChunkedMultimodalLoad(mod, modelId, requestId, msgDiag, msgDiagLayer);
            if (loaded) return;
            console.log('[local-worker] chunked multimodal load unavailable, falling back to bulk read');
        } else if (!multimodal && typeof mod.init_local_model_chunked === 'function') {
            const loaded = await tryChunkedLoad(mod, modelId, requestId);
            if (loaded) return;
            console.log('[local-worker] chunked load unavailable, falling back to bulk read');
        }

        // Bulk-read path. Multimodal goes through `init_local_multimodal`,
        // text-only through `init_local_model_gpu` / `init_local_model`.
        let initFn;
        if (multimodal) {
            initFn = mod.init_local_multimodal;
            if (!initFn) {
                self.postMessage({
                    requestId,
                    type: 'load_error',
                    error: 'wasm.init_local_multimodal not available — rebuild the WASM crate with local-llm-vision',
                });
                return;
            }
        } else {
            initFn = typeof mod.init_local_model_gpu === 'function'
                ? mod.init_local_model_gpu
                : mod.init_local_model;
            if (!initFn) {
                self.postMessage({
                    requestId,
                    type: 'load_error',
                    error: 'wasm.init_local_model not available — rebuild the WASM crate',
                });
                return;
            }
        }

        let { weights, tokenizer } = await getModelBytes(modelId);
        try {
            handle = await initFn(weights, tokenizer, modelId);
        } finally {
            weights = null;
            tokenizer = null;
        }
        loadedModelId = modelId;
        handleIsMultimodal = multimodal;

        const deviceType = handle.device_type || 'cpu';
        self.postMessage({ type: 'load_progress', phase: 'ready', modelId, deviceType });
        self.postMessage({ requestId, type: 'load_done', modelId, deviceType });
    } catch (err) {
        const error = err && err.message ? err.message : String(err);
        console.error('local-worker: load failed', err);
        self.postMessage({ requestId, type: 'load_error', error });
    }
}

/**
 * Stream an Ollama Q4_K_M GGUF blob into wasm via a JS read-fn callback,
 * keeping the multi-GB file out of the JS heap. Mirrors `tryChunkedLoad`
 * for the non-Ollama (HF safetensors) path; the only differences are
 * the OPFS layout (`model-downloads/ollama/library_<name>__<tag>/`) and
 * the wasm entry point (`init_local_multimodal_gguf_quantized_chunked`).
 *
 * Returns true on success (load_done already posted), false to fall
 * back to the bulk-read path.
 */
async function tryChunkedOllamaLoad(mod, modelId, requestId) {
    if (typeof mod.init_local_multimodal_gguf_quantized_chunked !== 'function') return false;
    if (typeof navigator === 'undefined' || !navigator.storage || !navigator.storage.getDirectory) {
        return false;
    }

    const om = KNOWN_OLLAMA_MODELS[modelId];
    if (!om) return false;

    const { ollamaModelInfo } = await _getOllamaModule();
    const info = await ollamaModelInfo(om.ollama.name, om.ollama.tag);
    const weightsFile = info.files.find((f) => f.kind === 'weights');
    if (!weightsFile) return false;
    const tokenizerFile = info.files.find((f) => f.kind === 'tokenizer');

    const dirName = (() => {
        const n = om.ollama.name;
        const ns = n.includes('/') ? n : `library/${n}`;
        return `${ns.replace('/', '_')}__${om.ollama.tag}`;
    })();

    let weightsSync = null;
    try {
        const root = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        const ollamaDir = await dlDir.getDirectoryHandle('ollama', { create: false });
        const modelDir = await ollamaDir.getDirectoryHandle(dirName, { create: false });

        const weightsFh = await modelDir.getFileHandle(weightsFile.filename, { create: false });
        weightsSync = await weightsFh.createSyncAccessHandle();
        const fileSize = weightsSync.getSize();
        if (fileSize === 0) {
            weightsSync.close();
            return false;
        }
        console.log(`[local-worker] chunked Ollama load: weights ${fileSize} bytes`);

        // Tokenizer is small (~32 MB worst case for Gemma) — bulk-read.
        let tokenizerBytes = new Uint8Array(0);
        if (tokenizerFile) {
            try {
                const tokFh = await modelDir.getFileHandle(tokenizerFile.filename, { create: false });
                const tokSync = await tokFh.createSyncAccessHandle();
                try {
                    const tokSize = tokSync.getSize();
                    if (tokSize > 0) {
                        tokenizerBytes = new Uint8Array(tokSize);
                        tokSync.read(tokenizerBytes, { at: 0 });
                    }
                } finally {
                    tokSync.close();
                }
            } catch (e) {
                console.warn('[local-worker] chunked Ollama: tokenizer read failed:', e.message);
            }
        }
        // No tokenizer in the GGUF? Fall back to the companion file
        // (HF tokenizer.json bundled with the model id). The helper is
        // defined locally in this file (`_fetchOllamaTokenizerCompanion`).
        if (tokenizerBytes.byteLength === 0 && om.tokenizerCompanion) {
            try {
                const companion = await _fetchOllamaTokenizerCompanion(om);
                if (companion && companion.byteLength > 0) tokenizerBytes = companion;
            } catch (e) {
                console.warn('[local-worker] chunked Ollama: tokenizer companion fetch failed:', e.message);
            }
        }

        const readFn = (offset, length) => {
            const buf = new Uint8Array(length);
            weightsSync.read(buf, { at: offset });
            return buf;
        };

        handle = await mod.init_local_multimodal_gguf_quantized_chunked(
            readFn,
            fileSize,
            tokenizerBytes,
            modelId,
        );
        loadedModelId = modelId;
        handleIsQuantized = true;
        handleIsMultimodal = false;

        weightsSync.close();
        weightsSync = null;

        const deviceType = handle.device_type || 'cpu';
        self.postMessage({ type: 'load_progress', phase: 'ready', modelId, deviceType });
        self.postMessage({ requestId, type: 'load_done', modelId, deviceType });
        return true;
    } catch (e) {
        if (weightsSync) {
            try { weightsSync.close(); } catch (_) {}
        }
        console.error('[local-worker] chunked Ollama load failed:', e);
        // Falling back to bulk read won't help if the issue is allocation
        // size, but it might if the failure was something else. Let the
        // caller decide.
        return false;
    }
}

async function tryChunkedLoad(mod, modelId, requestId) {
    const m = KNOWN_MODELS[modelId];
    if (!m) return false;
    if (typeof navigator === 'undefined' || !navigator.storage || !navigator.storage.getDirectory) {
        return false;
    }

    const weightsFile = m.files.find((f) => f.kind === 'weights');
    const tokenizerFile = m.files.find((f) => f.kind === 'tokenizer');
    if (!weightsFile) return false;

    let weightsSyncHandle = null;
    try {
        const root = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        const modelDir = await dlDir.getDirectoryHandle(modelId, { create: false });

        // Open weights file via sync access handle
        const weightsFh = await modelDir.getFileHandle(weightsFile.filename, { create: false });
        weightsSyncHandle = await weightsFh.createSyncAccessHandle();
        const fileSize = weightsSyncHandle.getSize();
        if (fileSize === 0) {
            weightsSyncHandle.close();
            return false;
        }
        console.log(`[local-worker] chunked load: weights file ${fileSize} bytes`);

        // Read tokenizer (small, ~32 MB — fine as a single allocation)
        let tokenizerBytes = new Uint8Array(0);
        if (tokenizerFile) {
            try {
                const tokFh = await modelDir.getFileHandle(tokenizerFile.filename, { create: false });
                const tokSync = await tokFh.createSyncAccessHandle();
                try {
                    const tokSize = tokSync.getSize();
                    if (tokSize > 0) {
                        tokenizerBytes = new Uint8Array(tokSize);
                        tokSync.read(tokenizerBytes, { at: 0 });
                        console.log(`[local-worker] chunked load: tokenizer ${tokSize} bytes`);
                    }
                } finally {
                    tokSync.close();
                }
            } catch (e) {
                console.warn('[local-worker] chunked load: tokenizer read failed:', e.message);
            }
        }

        // The read callback — WASM calls this to read tensor bytes from OPFS.
        // Each call allocates a fresh Uint8Array, reads from the sync handle,
        // and returns it. WASM copies the bytes into linear memory and the JS
        // buffer becomes eligible for GC immediately.
        const readFn = (offset, length) => {
            const buf = new Uint8Array(length);
            weightsSyncHandle.read(buf, { at: offset });
            return buf;
        };

        handle = await mod.init_local_model_chunked(readFn, fileSize, tokenizerBytes, modelId);
        loadedModelId = modelId;
        // Chunked path is text-only; multimodal weights take a different
        // prefix-aware loader (Stage E ships bulk-read for that case).
        handleIsMultimodal = false;

        weightsSyncHandle.close();
        weightsSyncHandle = null;

        const deviceType = handle.device_type || 'cpu';
        self.postMessage({ type: 'load_progress', phase: 'ready', modelId, deviceType });
        self.postMessage({ requestId, type: 'load_done', modelId, deviceType });
        return true;
    } catch (e) {
        if (weightsSyncHandle) {
            try { weightsSyncHandle.close(); } catch (_) {}
        }
        console.error('[local-worker] chunked load failed:', e);
        throw e;
    }
}

async function tryChunkedMultimodalLoad(mod, modelId, requestId, msgDiag, msgDiagLayer) {
    const m = KNOWN_MODELS[modelId];
    if (!m) return false;
    if (typeof navigator === 'undefined' || !navigator.storage || !navigator.storage.getDirectory) {
        return false;
    }

    const weightsFile = m.files.find((f) => f.kind === 'weights');
    const tokenizerFile = m.files.find((f) => f.kind === 'tokenizer');
    if (!weightsFile) return false;

    // Defensive: a previous load may have left a handle open on this
    // same file. OPFS sync access is exclusive per file, so opening
    // a second one would throw `InvalidStateError` instead of letting
    // us reuse the slot. Drop any stale handle first.
    closeMultimodalWeightsHandle();

    let weightsSyncHandle = null;
    try {
        const root = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        const modelDir = await dlDir.getDirectoryHandle(modelId, { create: false });

        const weightsFh = await modelDir.getFileHandle(weightsFile.filename, { create: false });
        weightsSyncHandle = await weightsFh.createSyncAccessHandle();
        const fileSize = weightsSyncHandle.getSize();
        if (fileSize === 0) {
            weightsSyncHandle.close();
            return false;
        }
        console.log(`[local-worker] chunked multimodal load: weights file ${fileSize} bytes`);

        let tokenizerBytes = new Uint8Array(0);
        if (tokenizerFile) {
            try {
                const tokFh = await modelDir.getFileHandle(tokenizerFile.filename, { create: false });
                const tokSync = await tokFh.createSyncAccessHandle();
                try {
                    const tokSize = tokSync.getSize();
                    if (tokSize > 0) {
                        tokenizerBytes = new Uint8Array(tokSize);
                        tokSync.read(tokenizerBytes, { at: 0 });
                    }
                } finally {
                    tokSync.close();
                }
            } catch (e) {
                console.warn('[local-worker] chunked multimodal: tokenizer read failed:', e.message);
            }
        }

        const readFn = (offset, length) => {
            const buf = new Uint8Array(length);
            weightsSyncHandle.read(buf, { at: offset });
            return buf;
        };

        // Defer the vision tower: skip its tensors at init and stream them
        // in on the first image. Saves ~300 MB of init RAM/VRAM and keeps
        // the largest single tensor under WebGPU's buffer-size cap. Audio
        // is also lazy by default — currently unused by the chat UI, and
        // attach_audio() is gated until config inference is wired.
        //
        // Bisection kill-switches for the Gemma 3n modules (AltUp /
        // LAuReL / per-layer-input gate) read from a global. Set them
        // via the browser console before loading, e.g.:
        //
        //     globalThis.__bw_disable_altup = true;
        //
        // and reload the page.
        const disableAltup = !!globalThis.__bw_disable_altup;
        const disableLaurel = !!globalThis.__bw_disable_laurel;
        const disablePerLayerInputGate = !!globalThis.__bw_disable_per_layer_input_gate;
        const disablePleStreaming = !!globalThis.__bw_disable_ple_streaming;
        // Per-layer diag: forwarded from the page's `globalThis.__bw_diag`
        // via the load message (the worker's globalThis is distinct from
        // the page's window, so direct reads here would always be
        // false). Falls back to the worker-scope global as a secondary
        // opt-in for tests that drive the worker directly.
        const diag = !!(msgDiag || globalThis.__bw_diag);
        // Optional intra-layer diag target: set
        // `globalThis.__bw_diag_layer = 15` (or any layer index) on the
        // page before model load to focus intra-layer captures away
        // from the default (layer 8). Negative value disables
        // intra-capture entirely.
        const diagLayerRaw = (msgDiagLayer ?? globalThis.__bw_diag_layer);
        const diagTargetLayer = (typeof diagLayerRaw === 'number' && Number.isInteger(diagLayerRaw))
            ? diagLayerRaw
            : null;
        if (disableAltup || disableLaurel || disablePerLayerInputGate || disablePleStreaming || diag) {
            console.warn(
                '[local-worker] Gemma 3n kill-switches active:',
                { disableAltup, disableLaurel, disablePerLayerInputGate, disablePleStreaming, diag, diagTargetLayer },
            );
        }
        handle = await mod.init_local_multimodal_chunked(
            readFn,
            fileSize,
            tokenizerBytes,
            modelId,
            {
                lazy_vision: true,
                lazy_audio: true,
                disable_altup: disableAltup,
                disable_laurel: disableLaurel,
                disable_per_layer_input_gate: disablePerLayerInputGate,
                disable_ple_streaming: disablePleStreaming,
                diag,
                diag_target_layer: diagTargetLayer,
            },
        );
        loadedModelId = modelId;
        handleIsMultimodal = true;

        // Hand the sync handle off to module scope — the WASM side
        // captured `readFn` for PLE row reads (every forward pass) and
        // for lazy vision/audio attach. Closing here would invalidate
        // those reads with `InvalidStateError` (surfaced in chat as
        // `OPFS PLE row read failed at id=...: <no msg>`).
        // Lifetime now ends on `unload` or model swap.
        //
        // Do NOT null `weightsSyncHandle` — the `readFn` closure
        // captures the lexical binding, not the value. Reassigning
        // would break every subsequent PLE/vision/audio read with
        // `TypeError: Cannot read properties of null`. Both bindings
        // refer to the same handle, so closing through either is fine.
        multimodalWeightsSyncHandle = weightsSyncHandle;

        const deviceType = handle.device_type || 'cpu';
        self.postMessage({ type: 'load_progress', phase: 'ready', modelId, deviceType });
        self.postMessage({ requestId, type: 'load_done', modelId, deviceType });
        return true;
    } catch (e) {
        // Could be either pre-handoff (`weightsSyncHandle` still local) or
        // post-handoff (also visible via `multimodalWeightsSyncHandle`).
        // Both names point at the same handle when both are set, so close
        // through one and clear the other.
        if (weightsSyncHandle) {
            try { weightsSyncHandle.close(); } catch (_) {}
            multimodalWeightsSyncHandle = null;
        }
        console.error('[local-worker] chunked multimodal load failed:', e);
        throw e;
    }
}

async function handleChat(msg) {
    // The text-only `local_chat_stream` takes a `LocalModelHandle`; when
    // we've loaded a multimodal model the handle is `LocalMultiModalHandle`
    // and that wasm-bindgen `instanceof` check fails with
    //   "expected instance of LocalModelHandle".
    // Route through the multimodal stream fn instead — it accepts text-only
    // messages (no image parts → no vision tower needed) and emits the same
    // NDJSON wire shape, so `runChatStream` is unchanged.
    //
    // Quantized GGUF handles (LocalQuantizedHandle) get their own stream
    // entry point — text-only, runs on QMatMul kernels.
    if (handleIsQuantized) {
        return runChatStream(msg, 'local_chat_stream_quantized');
    }
    if (handleIsMultimodal) {
        return runChatStream(msg, 'local_chat_stream_with_image');
    }
    return runChatStream(msg, 'local_chat_stream');
}

// Same shape as handleChat but routed through `local_chat_stream_with_image`,
// which accepts the parts[] message shape directly so {type:'image'} parts
// can be decoded inside Rust.
//
// Pre-flight guard: vision_chat against a text-only handle is a programmer
// error (the model registry says which loader to use; if the wrong one was
// run we can't usefully recover here). Surface a clear chat_error rather
// than letting the wasm export reject with a cryptic type-mismatch.
async function handleVisionChat(msg) {
    const { requestId, conversationId, messageId } = msg;
    if (handle !== null && handleIsQuantized) {
        self.postMessage({
            requestId, type: 'chat_error', conversationId, messageId,
            error: 'quantized GGUF model is text-only — reload with the HF safetensors variant for vision',
        });
        return;
    }
    if (handle !== null && !handleIsMultimodal) {
        self.postMessage({
            requestId, type: 'chat_error', conversationId, messageId,
            error: 'model not loaded as multimodal — reload with a vision-capable model',
        });
        return;
    }
    // Lazy-load the vision tower the first time an image-bearing chat
    // arrives. `attach_vision` is idempotent on the wasm side, so calling
    // it on every vision_chat is fine; the wasm method short-circuits
    // when the tower is already loaded.
    if (handle && typeof handle.attach_vision === 'function' && !handle.has_vision) {
        try {
            self.postMessage({ type: 'vision_attaching', conversationId, messageId });
            await handle.attach_vision();
        } catch (e) {
            self.postMessage({
                requestId, type: 'chat_error', conversationId, messageId,
                error: `failed to attach vision tower: ${e.message || e}`,
            });
            return;
        }
    }
    return runChatStream(msg, 'local_chat_stream_with_image');
}

// Drives an NDJSON stream from a wasm chat function. Both text-only and
// vision paths share this loop; the only thing that differs is which wasm
// export is invoked.
async function runChatStream(msg, wasmFnName) {
    const { requestId, conversationId, messageId, messages, params } = msg;
    if (handle === null) {
        self.postMessage({
            requestId, type: 'chat_error', conversationId, messageId,
            error: 'no_model_loaded',
        });
        return;
    }
    const mod = await getWasm();
    if (typeof mod[wasmFnName] !== 'function') {
        self.postMessage({
            requestId, type: 'chat_error', conversationId, messageId,
            error: `wasm.${wasmFnName}() not available — rebuild the WASM crate`,
        });
        return;
    }

    const ctl = { aborted: false, reader: null };
    inflight.set(conversationId, ctl);

    let usage = null;
    let tokensReceived = 0;
    try {
        const stream = await mod[wasmFnName](
            handle,
            JSON.stringify(messages || []),
            JSON.stringify(params || {}),
        );
        if (!stream || typeof stream.getReader !== 'function') {
            throw new Error(`${wasmFnName} did not return a ReadableStream`);
        }
        const reader = stream.getReader();
        ctl.reader = reader;
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        const dispatchObj = (obj) => {
            if (!obj || typeof obj !== 'object') return;
            if (typeof obj.error === 'string' && obj.error !== '') throw new Error(obj.error);
            if (typeof obj.delta === 'string' && obj.delta !== '') {
                tokensReceived += 1;
                self.postMessage({ type: 'chat_chunk', conversationId, messageId, delta: obj.delta });
            }
            if (obj.usage && typeof obj.usage === 'object') usage = obj.usage;
            // obj.finished is informational; reader's `done` is authoritative.
        };

        while (true) {
            if (ctl.aborted) break;
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let nl;
            while ((nl = buffer.indexOf('\n')) !== -1) {
                const line = buffer.slice(0, nl).replace(/\r$/, '');
                buffer = buffer.slice(nl + 1);
                if (line.trim() === '') continue;
                try { dispatchObj(JSON.parse(line)); } catch (_) { continue; }
            }
        }
        if (!ctl.aborted && buffer.trim() !== '') {
            try { dispatchObj(JSON.parse(buffer.trim())); }
            catch (_err) { console.warn("[bw] caught:", _err); }
            buffer = '';
        }
        try { reader.releaseLock(); } catch (_) { /* already released */ }

        if (ctl.aborted) {
            self.postMessage({ requestId, type: 'chat_error', conversationId, messageId, error: 'aborted' });
        } else {
            self.postMessage({ requestId, type: 'chat_done', conversationId, messageId, usage, tokensReceived });
        }
    } catch (err) {
        const error = err && err.message ? err.message : String(err);
        console.error(`local-worker: ${wasmFnName} failed`, err);
        self.postMessage({ requestId, type: 'chat_error', conversationId, messageId, error });
    } finally {
        inflight.delete(conversationId);
    }
}

// ── Embedding RPCs (RAG ingest path) ──────────────────────────
//
// Same general lifecycle as the chat handle: load weights once, embed many
// times, free on unload. wasm exports we expect (gated at runtime so the
// page surfaces a clear error on a wasm crate that hasn't shipped them):
//   - mod.init_embedding_model(weightsBytes, tokenizerBytes, modelId) → handle
//   - handle.embed_text(text) → Float32Array
//   - handle.dim → number
//   - handle.free()
async function handleEmbedLoad(msg) {
    const { requestId, modelId } = msg;
    try {
        if (embedHandle && embedModelId === modelId) {
            self.postMessage({ requestId, type: 'embed_load_done', modelId, dim: embedHandle.dim });
            return;
        }
        if (!(await isDownloaded(modelId))) {
            self.postMessage({ requestId, type: 'embed_load_error', error: 'not_downloaded' });
            return;
        }
        const mod = await getWasm();
        if (typeof mod.init_embedding_model !== 'function') {
            self.postMessage({
                requestId, type: 'embed_load_error',
                error: 'wasm.init_embedding_model() not available — rebuild the WASM crate',
            });
            return;
        }
        if (embedHandle && typeof embedHandle.free === 'function') {
            try { embedHandle.free(); } catch (_) { /* ignore */ }
        }
        embedHandle = null;
        embedModelId = null;

        let { weights, tokenizer } = await getModelBytes(modelId);
        try {
            embedHandle = await mod.init_embedding_model(weights, tokenizer, modelId);
        } finally {
            weights = null;
            tokenizer = null;
        }
        embedModelId = modelId;
        self.postMessage({ requestId, type: 'embed_load_done', modelId, dim: embedHandle.dim });
    } catch (err) {
        const error = err && err.message ? err.message : String(err);
        console.error('local-worker: embed_load failed', err);
        self.postMessage({ requestId, type: 'embed_load_error', error });
    }
}

async function handleEmbedText(msg) {
    const { requestId, text } = msg;
    if (embedHandle === null) {
        self.postMessage({ requestId, type: 'embed_text_error', error: 'no_embedding_model_loaded' });
        return;
    }
    if (typeof embedHandle.embed_text !== 'function') {
        self.postMessage({
            requestId, type: 'embed_text_error',
            error: 'embedHandle.embed_text() not available — rebuild the WASM crate',
        });
        return;
    }
    try {
        const vec = await embedHandle.embed_text(typeof text === 'string' ? text : '');
        // Workers can transfer typed-array buffers cheaply; pass via the
        // standard postMessage path (no transfer list) for simplicity since
        // a single 384-768 dim vector is small.
        self.postMessage({ requestId, type: 'embed_text_done', vector: vec });
    } catch (err) {
        const error = err && err.message ? err.message : String(err);
        console.error('local-worker: embed_text failed', err);
        self.postMessage({ requestId, type: 'embed_text_error', error });
    }
}

function handleEmbedUnload(msg) {
    const { requestId } = msg;
    if (embedHandle && typeof embedHandle.free === 'function') {
        try { embedHandle.free(); } catch (_) { /* idempotent */ }
    }
    embedHandle = null;
    embedModelId = null;
    self.postMessage({ requestId, type: 'embed_unload_ack' });
}

function handleCancel(msg) {
    const { requestId, conversationId } = msg;
    const ctl = inflight.get(conversationId);
    if (ctl) {
        ctl.aborted = true;
        // The reader will exit on the next iteration; we can also try to
        // cancel it eagerly so wasm stops generating ASAP.
        if (ctl.reader && typeof ctl.reader.cancel === 'function') {
            try { ctl.reader.cancel(); } catch (_err) { console.warn("[bw] caught:", _err); }
        }
    }
    self.postMessage({ requestId, type: 'cancel_ack', conversationId });
}

function handleUnload(msg) {
    const { requestId } = msg;
    if (handle && typeof handle.free === 'function') {
        try { handle.free(); } catch (_err) { console.debug("[bw] idempotent:", _err); }
    }
    handle = null;
    loadedModelId = null;
    handleIsMultimodal = false;
    handleIsQuantized = false;
    closeMultimodalWeightsHandle();
    self.postMessage({ requestId, type: 'unload_ack' });
}
