// Ollama-format model downloader.
//
// Wraps `ollama-fetch.js` (OCI Distribution Spec client) with the same
// OPFS / Cache Storage write strategy `model-store.js` uses for HF
// safetensors models, but with the file list derived from the OCI
// manifest at fetch time. Stays a separate entry point so the existing
// HF download paths (Dedicated Worker / SW / direct) are untouched —
// regressions there would break the working chat-pwa for everyone.
//
// Public API:
//   downloadOllamaModel(name, tag, opts)        — fetch manifest + blobs
//   isOllamaModelDownloaded(name, tag)          — check OPFS for completed dl
//   getOllamaModelBytes(name, tag)              — load all blob bytes back
//   ollamaModelInfo(name, tag)                  — manifest + parsed file list
//
// Uses the same `model-progress` event channel + onProgress callback shape
// as the HF path, so the UI banner picks it up without changes.

import {
    fetchManifest,
    manifestToFiles,
    estimatedBytesFromManifest,
    ollamaCacheKey,
} from './ollama-fetch.js';
import { events } from './state.js';

// Parallel OPFS namespace so HF and Ollama models never collide on disk.
// HF: `model-downloads/<modelId>/...`
// Ollama: `model-downloads/ollama/<name>__<tag>/...`
const OPFS_DIR = 'model-downloads';
const OPFS_OLLAMA_SUB = 'ollama';
const PROGRESS_EMIT_MS = 250;

function _hasOpfs() {
    return typeof navigator !== 'undefined'
        && navigator.storage
        && typeof navigator.storage.getDirectory === 'function';
}

function ollamaDirName(name, tag) {
    // `library/gemma4:e2b` is fine on POSIX but `:` is reserved on Windows
    // OPFS, and the leading namespace varies. Normalize to a single
    // directory token: `library_gemma4__e2b`.
    const ns = name.includes('/') ? name : `library/${name}`;
    return `${ns.replace('/', '_')}__${tag}`;
}

async function _ollamaDir(name, tag, { create = false } = {}) {
    if (!_hasOpfs()) return null;
    const root = await navigator.storage.getDirectory();
    const top = await root.getDirectoryHandle(OPFS_DIR, { create });
    const sub = await top.getDirectoryHandle(OPFS_OLLAMA_SUB, { create });
    return sub.getDirectoryHandle(ollamaDirName(name, tag), { create });
}

/**
 * Inspect manifest and report which files are needed + total size.
 *
 * @param {string} name
 * @param {string} tag
 * @returns {Promise<{ manifest: object, files: Array, estimatedBytes: number }>}
 */
export async function ollamaModelInfo(name, tag) {
    const manifest = await fetchManifest(name, tag);
    const files = manifestToFiles(manifest);
    const estimatedBytes = estimatedBytesFromManifest(manifest);
    return { manifest, files, estimatedBytes };
}

/**
 * Status check — returns `true` only if every layer's blob is on disk
 * and a `.verified` marker exists alongside it.
 *
 * @param {string} name
 * @param {string} tag
 * @returns {Promise<boolean>}
 */
export async function isOllamaModelDownloaded(name, tag) {
    if (!_hasOpfs()) return false;
    let info;
    try {
        info = await ollamaModelInfo(name, tag);
    } catch (_) {
        return false;
    }
    let dir;
    try {
        dir = await _ollamaDir(name, tag, { create: false });
    } catch (_) {
        return false;
    }
    if (!dir) return false;
    for (const f of info.files) {
        try {
            await dir.getFileHandle(f.filename + '.verified', { create: false });
            const fh = await dir.getFileHandle(f.filename, { create: false });
            const file = await fh.getFile();
            if (file.size === 0) return false;
        } catch (_) {
            return false;
        }
    }
    return true;
}

/**
 * Load every blob's bytes back from OPFS into memory. Throws if any layer
 * is missing — call `isOllamaModelDownloaded` first or handle the error.
 *
 * @param {string} name
 * @param {string} tag
 * @returns {Promise<{ files: Array, bytes: Object<string, Uint8Array> }>}
 */
export async function getOllamaModelBytes(name, tag) {
    const info = await ollamaModelInfo(name, tag);
    const dir = await _ollamaDir(name, tag, { create: false });
    if (!dir) throw new Error('OPFS not available');
    const out = {};
    for (const f of info.files) {
        const fh = await dir.getFileHandle(f.filename, { create: false });
        let bytes = null;

        // Sync access handle path. `File.arrayBuffer()` throws
        // NotReadableError on multi-GB OPFS files in Chrome (the same
        // issue local-worker.js getModelBytes works around) — Q4_K_M
        // gemma4:e2b weights are ~1.6 GB, well past the threshold.
        // The sync API bypasses that code path.
        try {
            const syncHandle = await fh.createSyncAccessHandle();
            try {
                const size = syncHandle.getSize();
                if (size > 0) {
                    bytes = new Uint8Array(size);
                    syncHandle.read(bytes, { at: 0 });
                }
            } finally {
                syncHandle.close();
            }
        } catch (e) {
            console.warn(`[ollama-download] ${f.filename}: sync handle read failed:`, e.message);
        }

        // Fallback to async File API for environments / file sizes
        // where the sync path isn't available.
        if (!bytes) {
            const file = await fh.getFile();
            bytes = new Uint8Array(await file.arrayBuffer());
        }

        out[f.kind] = bytes;
    }
    return { files: info.files, bytes: out };
}

/**
 * Single-path Ollama downloader: fetch manifest, then for each file fetch
 * the blob and stream it into OPFS. Resume via Range header if a partial
 * file already exists. Verify SHA-256 on completion.
 *
 * Concurrency: caller is responsible for not invoking twice in parallel
 * for the same model — the chat-pwa UI gates this. AbortSignal in opts
 * cancels the in-flight fetch.
 *
 * @param {string} name
 * @param {string} tag
 * @param {{
 *   signal?: AbortSignal,
 *   onProgress?: (detail: object) => void,
 *   modelId?: string,
 * }} [opts]
 */
export async function downloadOllamaModel(name, tag, opts = {}) {
    if (!_hasOpfs()) {
        throw new Error('Ollama download requires OPFS');
    }
    const { manifest: _manifest, files, estimatedBytes } = await ollamaModelInfo(name, tag);
    const dir = await _ollamaDir(name, tag, { create: true });
    const startedAt = Date.now();
    const modelId = opts.modelId || `ollama:${name}:${tag}`;
    // OPFS path under `model-downloads/` for the worker. Mirrors
    // `_ollamaDir` above: `model-downloads/ollama/<dirName>/...`.
    const dirPath = [OPFS_OLLAMA_SUB, ollamaDirName(name, tag)];

    let totalBytesDone = 0;
    let lastEmit = 0;

    const emit = (file, fileBytesDone, fileBytesTotal, force = false) => {
        const now = Date.now();
        if (!force && now - lastEmit < PROGRESS_EMIT_MS) return;
        lastEmit = now;
        const elapsedSec = Math.max(0.001, (now - startedAt) / 1000);
        const throughputBps = totalBytesDone / elapsedSec;
        const remaining = Math.max(0, estimatedBytes - totalBytesDone);
        const etaSeconds = throughputBps > 0 ? remaining / throughputBps : null;
        const detail = {
            phase: 'download',
            modelId,
            source: 'ollama',
            file: file.filename,
            fileKind: file.kind,
            fileBytesDone,
            fileBytesTotal,
            totalBytesDone,
            totalBytesTotal: estimatedBytes,
            throughputBps,
            etaSeconds,
        };
        try { if (typeof opts.onProgress === 'function') opts.onProgress(detail); }
        catch (_e) { /* swallow user callback errors */ }
        try { events.dispatchEvent(new CustomEvent('model_progress', { detail })); }
        catch (_e) { /* events may be unavailable in some test contexts */ }
    };

    // Streaming SHA-256 verification has too much OOM risk for multi-GB
    // GGUF blobs (a single `arrayBuffer()` would allocate the full file
    // on the heap). The HF path skips verification past 2 GB for the
    // same reason. Trust HTTPS + the registry's content-addressed
    // digest for blobs above this cap.
    const SHA_VERIFY_CAP = 2 * 1024 * 1024 * 1024;

    for (const f of files) {
        // Skip if already verified.
        try {
            await dir.getFileHandle(f.filename + '.verified', { create: false });
            const existing = await (await dir.getFileHandle(f.filename, { create: false })).getFile();
            if (existing.size > 0) {
                totalBytesDone += existing.size;
                emit(f, existing.size, existing.size, true);
                continue;
            }
        } catch (_) { /* no marker, proceed */ }

        // Resume from existing partial if present. createSyncAccessHandle
        // is worker-only, so we read the existing size via getFile()
        // (which works on the main thread) before handing off to the
        // worker for the actual write.
        let existingSize = 0;
        try {
            const fh = await dir.getFileHandle(f.filename, { create: false });
            existingSize = (await fh.getFile()).size;
        } catch (_) { /* no partial yet */ }

        const url = ollamaCacheKey(name, tag, f.digest);

        // Spawn the same OPFS-writer worker the HF path uses. The
        // worker handles fetch + sync-access write + flushing, sends
        // back progress / done / error / cancelled.
        const worker = new Worker(
            new URL('./opfs-writer-worker.js', import.meta.url),
            { type: 'module' },
        );
        const abortHandler = () => worker.postMessage({ type: 'cancel' });
        if (opts.signal) opts.signal.addEventListener('abort', abortHandler, { once: true });

        let lastWorkerBytes = existingSize;
        let fileBytesDone = existingSize;

        try {
            await new Promise((resolve, reject) => {
                worker.onmessage = (ev) => {
                    const msg = ev.data;
                    if (!msg || typeof msg !== 'object') return;
                    if (msg.type === 'progress') {
                        const delta = msg.bytesWritten - lastWorkerBytes;
                        lastWorkerBytes = msg.bytesWritten;
                        fileBytesDone = msg.bytesWritten;
                        if (delta > 0) totalBytesDone += delta;
                        emit(f, msg.bytesWritten, msg.totalBytes || f.size || 0);
                    } else if (msg.type === 'done') {
                        const delta = msg.totalBytes - lastWorkerBytes;
                        if (delta > 0) totalBytesDone += delta;
                        fileBytesDone = msg.totalBytes;
                        emit(f, msg.totalBytes, msg.totalBytes, true);
                        worker.terminate();
                        resolve();
                    } else if (msg.type === 'cancelled') {
                        worker.terminate();
                        reject(new DOMException('aborted', 'AbortError'));
                    } else if (msg.type === 'error') {
                        worker.terminate();
                        reject(new Error(`ollama blob ${f.filename}: ${msg.error}`));
                    }
                };
                worker.onerror = (ev) => {
                    worker.terminate();
                    reject(new Error(ev.message || 'Worker error'));
                };
                worker.postMessage({
                    type: 'start',
                    modelId,
                    dirPath,
                    filename: f.filename,
                    url,
                    headers: {},
                    offset: existingSize,
                });
            });
        } finally {
            if (opts.signal) opts.signal.removeEventListener('abort', abortHandler);
        }

        // Optional SHA-256 verification (small files only — large GGUF
        // blobs would OOM). Reads via async getFile() so it works on
        // the main thread.
        if (f.sha256 && fileBytesDone <= SHA_VERIFY_CAP) {
            const fh = await dir.getFileHandle(f.filename, { create: false });
            const file = await fh.getFile();
            const hex = await sha256Hex(await file.arrayBuffer());
            if (hex !== f.sha256.toLowerCase()) {
                throw new Error(
                    `ollama blob ${f.filename} sha256 mismatch: ` +
                    `got ${hex.slice(0, 16)}…, expected ${f.sha256.slice(0, 16)}…`,
                );
            }
        }

        // Drop a `.verified` marker so resume short-circuits next
        // time. Async createWritable() works on the main thread (the
        // marker is 2 bytes — no perf concern from the
        // keepExistingData copy).
        const markerFh = await dir.getFileHandle(f.filename + '.verified', { create: true });
        const writable = await markerFh.createWritable();
        try { await writable.write(new Uint8Array([0x6f, 0x6b])); /* "ok" */ }
        finally { try { await writable.close(); } catch (_) { /* swallow */ } }
    }
}

/**
 * Web Crypto SHA-256 → lowercase hex string.
 *
 * @param {ArrayBuffer | Uint8Array} buf
 * @returns {Promise<string>}
 */
async function sha256Hex(buf) {
    const data = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    const digest = await crypto.subtle.digest('SHA-256', data);
    return Array.from(new Uint8Array(digest))
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
}
