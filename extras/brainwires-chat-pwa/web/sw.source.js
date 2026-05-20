// brainwires-chat-pwa — service worker source
//
// Headline responsibility: keep streaming chat responses alive when the
// page is backgrounded (mobile lock screen, tab in background, etc.).
// The SW owns the fetch + ReadableStream; it persists each chunk to
// IndexedDB and broadcasts deltas to any visible client. When the
// stream finishes with no foreground client, it raises a notification.
//
// Build pipeline:
//   sw.source.js  --(esbuild bundle, IIFE)-->  sw.bundle.js
//   sw.bundle.js  --(SRI substitution)----->  sw.js   (gitignored)
//
// The SRI table substituted into __SRI_HASHES__ pins the static assets
// the SW caches. Cloud provider URLs (huggingface.co, OpenAI, Anthropic,
// etc.) are NOT in the table and pass straight through to the network
// without ever being cached.
//
// Imports below are bundled by esbuild — the runtime sees a flat IIFE.

import {
    streamFromResponse,
} from './src/streaming.js';
import {
    decrypt as cryptoDecrypt,
    unpack as cryptoUnpack,
} from './crypto-store.js';
import {
    appendMessageChunk,
    putMessage,
} from './src/db.js';
import * as anthropicProvider from './src/providers/anthropic.js';
import * as openaiProvider from './src/providers/openai.js';

// Per-provider tool_use parsers — invoked alongside the raw passthrough
// so the SW can broadcast `chat_tool_use` events on stream events that
// reassemble into a complete {id, name, input} invocation. Local /
// ndjson providers (ollama, gemini text path) do not yet emit tool_use.
const TOOL_USE_PARSERS = {
    anthropic: anthropicProvider.parseChunk,
    openai: openaiProvider.parseChunk,
};

// ── Cache versioning ───────────────────────────────────────────
const CACHE_NAME = 'bw-chat-cache-v1';

// ── Passthrough host allowlist ─────────────────────────────────
//
// The fetch handler already passes everything that's not pinned to
// the network unmodified. This list is informational: any host
// matching here is GUARANTEED never to be cached by the SW. We use
// it for an explicit early-return so a future maintainer adding new
// caching logic can't accidentally swallow these.
const PASSTHROUGH_HOST_PATTERNS = [
    /^huggingface\.co$/i,
    /\.huggingface\.co$/i,
    /^api\.anthropic\.com$/i,
    /^api\.openai\.com$/i,
    /^generativelanguage\.googleapis\.com$/i,
    /:11434$/,                        // any Ollama LAN host
];

function isPassthroughHost(url) {
    try {
        const u = new URL(url);
        const hostport = u.port ? `${u.hostname}:${u.port}` : u.hostname;
        return PASSTHROUGH_HOST_PATTERNS.some((re) => re.test(hostport) || re.test(u.hostname));
    } catch (_) { return false; }
}

// ── SRI hash table (build-time substituted) ────────────────────
//
// Keys are paths relative to the web root (e.g. 'app.js',
// 'pkg/brainwires_chat_pwa.js'). Values are 'sha256-<base64>' digests.
// `sw.js` itself is intentionally excluded (a worker can't verify itself).
const RESOURCE_HASHES = __SRI_HASHES__;

// ── Tiny log helper ────────────────────────────────────────────
// Production paths stay quiet; debug logs are silenced unless you
// flip DEBUG to true at build/test time.
const DEBUG = false;
let DEV_MODE = false;
function log(...args) { if (DEBUG) console.log('[bw-sw]', ...args); }
function warn(...args) { console.warn('[bw-sw]', ...args); }

// ── Hash helpers ───────────────────────────────────────────────

async function sha256Base64(buffer) {
    const hashBuf = await crypto.subtle.digest('SHA-256', buffer);
    let bin = '';
    const bytes = new Uint8Array(hashBuf);
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
}

// ── Streaming SHA-256 (incremental, no full-file-in-RAM) ──────
// Pure JS implementation of FIPS 180-4 SHA-256. Processes data in
// chunks via update(), returns hex digest from finalize(). Uses
// ~256 bytes of state regardless of file size.

const _K = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

class StreamingSha256 {
    constructor() {
        this._h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
        this._buf = new Uint8Array(64);
        this._pos = 0;
        this._len = 0;
        this._w = new Uint32Array(64);
    }
    update(data) {
        const d = data instanceof Uint8Array ? data : new Uint8Array(data);
        this._len += d.length;
        let off = 0;
        if (this._pos > 0) {
            const need = 64 - this._pos;
            const take = Math.min(need, d.length);
            this._buf.set(d.subarray(0, take), this._pos);
            this._pos += take;
            off = take;
            if (this._pos === 64) { this._compress(this._buf); this._pos = 0; }
        }
        while (off + 64 <= d.length) { this._compress(d.subarray(off, off + 64)); off += 64; }
        if (off < d.length) { this._buf.set(d.subarray(off), 0); this._pos = d.length - off; }
        return this;
    }
    finalize() {
        const bits = this._len * 8;
        this._buf[this._pos++] = 0x80;
        if (this._pos > 56) { this._buf.fill(0, this._pos); this._compress(this._buf); this._pos = 0; }
        this._buf.fill(0, this._pos);
        const dv = new DataView(this._buf.buffer);
        dv.setUint32(56, Math.floor(bits / 0x100000000), false);
        dv.setUint32(60, bits >>> 0, false);
        this._compress(this._buf);
        let hex = '';
        for (let i = 0; i < 8; i++) hex += this._h[i].toString(16).padStart(8, '0');
        return hex;
    }
    _compress(block) {
        const w = this._w;
        const dv = new DataView(block.buffer, block.byteOffset, 64);
        for (let i = 0; i < 16; i++) w[i] = dv.getUint32(i * 4, false);
        for (let i = 16; i < 64; i++) {
            const s0 = _rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
            const s1 = _rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
            w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
        }
        let [a, b, c, d, e, f, g, h] = this._h;
        for (let i = 0; i < 64; i++) {
            const S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25);
            const ch = (e & f) ^ (~e & g);
            const t1 = (h + S1 + ch + _K[i] + w[i]) | 0;
            const S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22);
            const maj = (a & b) ^ (a & c) ^ (b & c);
            const t2 = (S0 + maj) | 0;
            h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
        }
        this._h[0] = (this._h[0] + a) | 0; this._h[1] = (this._h[1] + b) | 0;
        this._h[2] = (this._h[2] + c) | 0; this._h[3] = (this._h[3] + d) | 0;
        this._h[4] = (this._h[4] + e) | 0; this._h[5] = (this._h[5] + f) | 0;
        this._h[6] = (this._h[6] + g) | 0; this._h[7] = (this._h[7] + h) | 0;
    }
}
function _rotr(x, n) { return (x >>> n) | (x << (32 - n)); }

async function streamingSha256FromFile(file) {
    const hasher = new StreamingSha256();
    const reader = file.stream().getReader();
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        hasher.update(value);
    }
    return hasher.finalize();
}

function resourceKey(url) {
    const path = new URL(url).pathname.replace(/^\/+/, '');
    return path;
}

function isPinned(url) {
    return Object.prototype.hasOwnProperty.call(RESOURCE_HASHES, resourceKey(url));
}

// ── Lifecycle: install/activate ────────────────────────────────

self.addEventListener('install', (event) => {
    event.waitUntil((async () => {
        const cache = await caches.open(CACHE_NAME);
        const paths = Object.keys(RESOURCE_HASHES);
        // addAll is atomic; if any single asset 404s the cache install
        // fails. We tolerate that by trying assets individually so a
        // missing dev asset doesn't brick the SW.
        await Promise.all(paths.map(async (rel) => {
            try {
                const url = new URL('./' + rel, self.location).href;
                const resp = await fetch(url, { cache: 'no-cache' });
                if (resp && resp.ok) await cache.put(url, resp.clone());
            } catch (e) {
                warn('install: failed to cache', rel, e && e.message);
            }
        }));
        await self.skipWaiting();
    })());
});

self.addEventListener('activate', (event) => {
    event.waitUntil((async () => {
        const keys = await caches.keys();
        await Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)));
        await self.clients.claim();
    })());
});

// ── Fetch: cache-first for pinned, network-only for everything else ──

self.addEventListener('fetch', (event) => {
    const req = event.request;
    if (req.method !== 'GET') return;

    const sameOrigin = req.url.startsWith(self.location.origin);
    if (isPassthroughHost(req.url)) {
        // Explicit allowlist: provider + HF URLs go straight to the
        // network and never land in any SW-managed cache.
        return;
    }
    if (!sameOrigin || !isPinned(req.url)) {
        return;
    }

    // Dev mode: network-first, no cache, no SRI — live-editing works
    // while the SW stays alive for model downloads.
    if (DEV_MODE) return;

    event.respondWith((async () => {
        const cache = await caches.open(CACHE_NAME);
        const cached = await cache.match(req);
        if (cached) {
            const expected = RESOURCE_HASHES[resourceKey(req.url)];
            try {
                const buf = await cached.clone().arrayBuffer();
                const actual = 'sha256-' + await sha256Base64(buf);
                if (actual === expected) return cached;
                warn('SRI mismatch for', req.url, '— evicting and refetching');
                await cache.delete(req);
            } catch (e) {
                warn('SRI verify failed for', req.url, e && e.message);
                // Fall through to network.
            }
        }
        // Cache miss or SRI eviction → fetch fresh, populate cache.
        try {
            const fresh = await fetch(req);
            if (fresh && fresh.ok) {
                cache.put(req, fresh.clone()).catch((e) => { console.warn("[bw] swallowed:", e); });
            }
            return fresh;
        } catch (e) {
            // Last resort: return the (mismatched) cached copy if we still have it.
            if (cached) return cached;
            throw e;
        }
    })());
});

// ── Active stream registry ─────────────────────────────────────
//
// Lost on SW eviction; durability is provided by IndexedDB writes.
// The map key is messageId so chat_status_query / chat_cancel can target
// in-flight streams without a conversationId lookup.
//
// Value shape: { conversationId, abortController, tokensReceived, startedAt }
const activeStreams = new Map();

// ── Message IPC ────────────────────────────────────────────────

self.addEventListener('message', (event) => {
    const msg = event.data;
    if (!msg || typeof msg !== 'object') return;

    switch (msg.type) {
        case 'chat_start':
            event.waitUntil(handleChatStart(msg, event));
            break;
        case 'chat_status_query': {
            const active = [];
            for (const [messageId, st] of activeStreams) {
                active.push({
                    conversationId: st.conversationId,
                    messageId,
                    tokensReceived: st.tokensReceived,
                    startedAt: st.startedAt,
                });
            }
            replyTo(event, { type: 'chat_status', active });
            break;
        }
        case 'chat_cancel': {
            const st = activeStreams.get(msg.messageId);
            if (st) {
                try { st.abortController.abort(); } catch (_err) { console.warn("[bw] caught:", _err); }
            }
            break;
        }
        case 'model_download_start':
            event.waitUntil(handleModelDownload(msg, event));
            break;
        case 'model_download_cancel': {
            const dl = activeModelDownloads.get(msg.modelId);
            if (dl) { try { dl.controller.abort(); } catch (_err) { console.warn("[bw] caught:", _err); } }
            break;
        }
        case 'set_dev_mode':
            DEV_MODE = !!msg.enabled;
            log('DEV_MODE set to', DEV_MODE);
            break;
        case 'sri_table':
            replyTo(event, { type: 'sri_table', hashes: RESOURCE_HASHES });
            break;
        default:
            log('unknown message type', msg.type);
    }
});

// ── Model download (background-resilient + resumable) ───────────
//
// Chunks are persisted to IndexedDB as they stream from the network.
// If the download is interrupted, the next attempt reads the partial
// state from IDB and resumes with a Range header. On completion, all
// chunks are assembled into a Blob and cache.put'd as a single
// Response, then the IDB partials are cleaned up.

const activeModelDownloads = new Map();
const MODEL_CACHE = 'bw-models-v1';
const OPFS_DIR = 'model-downloads';
const DL_PROGRESS_MS = 200;

// ── OPFS helpers ──────────────────────────────────────────────
// Feature-detect OPFS; fall back gracefully if unavailable.

function hasOpfs() {
    return typeof navigator !== 'undefined' && navigator.storage && typeof navigator.storage.getDirectory === 'function';
}

async function getOpfsDir() {
    const root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle(OPFS_DIR, { create: true });
}

async function getOpfsFileHandle(dir, modelId, filename) {
    const modelDir = await dir.getDirectoryHandle(modelId, { create: true });
    return modelDir.getFileHandle(filename, { create: true });
}

async function deleteOpfsModel(modelId) {
    if (!hasOpfs()) return;
    try {
        const dir = await getOpfsDir();
        await dir.removeEntry(modelId, { recursive: true });
    } catch (_) { /* directory may not exist */ }
}

async function handleModelDownload(msg, _event) {
    const { modelId, files, hfToken } = msg;
    console.log('[bw-sw] handleModelDownload:', modelId, files ? files.length : 0, 'files');
    if (!files || !files.length) { console.warn('[bw-sw] no files to download'); return; }
    if (activeModelDownloads.has(modelId)) { console.log('[bw-sw] download already active for', modelId); return; }

    const controller = new AbortController();
    activeModelDownloads.set(modelId, { controller, startedAt: Date.now() });

    try {
        const cache = await caches.open(MODEL_CACHE);
        const opfsAvailable = hasOpfs();
        const opfsDir = opfsAvailable ? await getOpfsDir() : null;
        let totalBytesDone = 0;
        let totalBytesTotal = 0;
        const startedAt = Date.now();
        let lastEmit = 0;

        const broadcastDl = (detail) => {
            self.clients.matchAll({ type: 'window' }).then(cls => {
                for (const c of cls) c.postMessage(detail);
            });
        };

        const emitProgress = (file, fileBytesDone, fileBytesTotal, force) => {
            const now = Date.now();
            if (!force && now - lastEmit < DL_PROGRESS_MS) return;
            lastEmit = now;
            const elapsed = Math.max(0.001, (now - startedAt) / 1000);
            const bps = totalBytesDone / elapsed;
            const remaining = Math.max(0, totalBytesTotal - totalBytesDone);
            broadcastDl({
                type: 'model_progress',
                detail: {
                    phase: 'download', modelId,
                    file: file.filename, fileKind: file.kind,
                    fileBytesDone, fileBytesTotal,
                    totalBytesDone, totalBytesTotal,
                    throughputBps: bps,
                    etaSeconds: bps > 0 ? remaining / bps : null,
                },
            });
        };

        for (const f of files) {
            const url = f.url;

            // Already cached (Cache Storage) — skip.
            const existing = await cache.match(url);
            if (existing) {
                const len = Number(existing.headers.get('content-length')) || 0;
                totalBytesTotal += len;
                totalBytesDone += len;
                emitProgress(f, len, len, true);
                continue;
            }

            // Already verified in OPFS — check for a companion .verified marker.
            if (opfsAvailable) {
                try {
                    const modelDir = await opfsDir.getDirectoryHandle(modelId, { create: false });
                    await modelDir.getFileHandle(f.filename + '.verified', { create: false });
                    const fh = await modelDir.getFileHandle(f.filename, { create: false });
                    const ff = await fh.getFile();
                    if (ff.size > 0) {
                        const len = ff.size;
                        totalBytesTotal += len;
                        totalBytesDone += len;
                        emitProgress(f, len, len, true);
                        console.log(`[bw-sw] ${f.filename}: already verified in OPFS (${len} bytes)`);
                        continue;
                    }
                } catch (_) { /* no marker — proceed with download */ }
            }

            // ── OPFS-based resumable streaming download ──────────
            // Bytes flow: network → OPFS file (streaming write).
            // On complete: SHA verify directly from OPFS. OPFS IS the
            // durable store — no copy to Cache Storage needed.
            // Resume: OPFS file persists across SW restarts; getFile().size
            // tells us where to resume with a Range header.

            const MAX_RETRIES = 3;
            let fileBytesDone = 0;
            let contentLength = 0;

            // Check OPFS for an existing partial file.
            let opfsHandle = null;
            if (opfsAvailable) {
                opfsHandle = await getOpfsFileHandle(opfsDir, modelId, f.filename);
                const partial = await opfsHandle.getFile();
                fileBytesDone = partial.size;
                if (fileBytesDone > 0) {
                    console.log(`[bw-sw] resuming ${f.filename} from ${fileBytesDone} bytes (OPFS)`);
                    totalBytesDone += fileBytesDone;
                }
            }

            for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
                const dlHeaders = {};
                if (hfToken) dlHeaders['Authorization'] = `Bearer ${hfToken}`;
                if (fileBytesDone > 0) dlHeaders['Range'] = `bytes=${fileBytesDone}-`;

                console.log(`[bw-sw] download ${f.filename} attempt ${attempt + 1} from byte ${fileBytesDone}`);

                let resp;
                try {
                    resp = await fetch(url, { headers: dlHeaders, signal: controller.signal });
                } catch (e) {
                    if (controller.signal.aborted) throw e;
                    console.warn(`[bw-sw] ${f.filename} fetch failed:`, e.message);
                    if (attempt === MAX_RETRIES) throw e;
                    await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
                    continue;
                }
                if (resp.status === 401 || resp.status === 403) {
                    broadcastDl({ type: 'model_download_error', modelId, error: 'HF_AUTH_REQUIRED' });
                    return;
                }
                if (resp.status === 416) {
                    if (opfsHandle) {
                        const p = await opfsHandle.getFile();
                        contentLength = p.size;
                        totalBytesTotal += contentLength;
                    }
                    console.log(`[bw-sw] ${f.filename}: 416 — already complete in OPFS (${contentLength} bytes)`);
                    break;
                }
                if (!resp.ok && resp.status !== 206) {
                    console.warn(`[bw-sw] ${f.filename} HTTP ${resp.status}`);
                    if (attempt === MAX_RETRIES) throw new Error(`HF ${resp.status}`);
                    await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
                    continue;
                }

                // Server returned 200 instead of 206 — restart from scratch.
                if (resp.status === 200 && fileBytesDone > 0) {
                    console.log(`[bw-sw] ${f.filename}: server sent full file, discarding partial`);
                    totalBytesDone -= fileBytesDone;
                    fileBytesDone = 0;
                    if (opfsHandle) {
                        const w = await opfsHandle.createWritable();
                        await w.truncate(0);
                        await w.close();
                    }
                }

                if (contentLength === 0) {
                    contentLength = resp.status === 206
                        ? Number((resp.headers.get('content-range') || '').split('/')[1]) || 0
                        : Number(resp.headers.get('content-length')) || 0;
                    totalBytesTotal += contentLength;
                    console.log(`[bw-sw] ${f.filename}: ${contentLength} bytes total`);
                }

                // Stream to OPFS with periodic commit. createWritable() data
                // isn't durable until .close(). If the SW is killed mid-write
                // (phone backgrounded), uncommitted bytes are lost. Close + reopen
                // every COMMIT_INTERVAL bytes so at most that much is lost.
                const COMMIT_INTERVAL = 500 * 1024 * 1024; // 500 MB
                const reader = resp.body.getReader();
                let streamFailed = false;
                let writable = null;
                let bytesSinceCommit = 0;

                if (opfsHandle) {
                    try {
                        writable = await opfsHandle.createWritable({ keepExistingData: true });
                        await writable.seek(fileBytesDone);
                    } catch (lockErr) {
                        // Previous crashed SW may have left a lock. Delete + restart.
                        console.warn(`[bw-sw] ${f.filename}: createWritable failed (locked?), deleting partial`, lockErr);
                        try {
                            const modelDir = await opfsDir.getDirectoryHandle(modelId, { create: false });
                            await modelDir.removeEntry(f.filename);
                        } catch (_e) { console.warn('[bw] cleanup failed:', _e); }
                        totalBytesDone -= fileBytesDone;
                        fileBytesDone = 0;
                        opfsHandle = await getOpfsFileHandle(opfsDir, modelId, f.filename);
                        writable = await opfsHandle.createWritable();
                    }
                }

                try {
                    while (true) {
                        const { value, done } = await reader.read();
                        if (done) break;
                        if (controller.signal.aborted) throw new DOMException('aborted', 'AbortError');

                        if (writable) await writable.write(value);
                        fileBytesDone += value.byteLength;
                        totalBytesDone += value.byteLength;
                        bytesSinceCommit += value.byteLength;
                        emitProgress(f, fileBytesDone, contentLength);

                        // Periodic commit: close + reopen to flush to disk.
                        if (writable && bytesSinceCommit >= COMMIT_INTERVAL) {
                            await writable.close();
                            writable = await opfsHandle.createWritable({ keepExistingData: true });
                            await writable.seek(fileBytesDone);
                            bytesSinceCommit = 0;
                            console.log(`[bw-sw] ${f.filename}: committed at ${fileBytesDone}`);
                        }
                    }
                    if (writable) await writable.close();
                } catch (e) {
                    if (writable) try { await writable.close(); } catch (_err) { console.warn("[bw] close failed:", _err); }
                    if (controller.signal.aborted) throw e;
                    streamFailed = true;
                    console.warn(`[bw-sw] ${f.filename} stream broke at ${fileBytesDone}/${contentLength}:`, e.message);
                    if (attempt === MAX_RETRIES) throw e;
                }
                try { reader.releaseLock(); } catch (_err) { console.warn("[bw] caught:", _err); }
                if (!streamFailed) break;
                // On retry, re-read OPFS file size for accurate resume.
                if (opfsHandle) {
                    const p = await opfsHandle.getFile();
                    fileBytesDone = p.size;
                }
                await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
            }

            emitProgress(f, fileBytesDone, contentLength, true);

            // Verify download integrity.
            if (opfsHandle) {
                const opfsFile = await opfsHandle.getFile();
                const SIZE_LIMIT_FOR_JS_SHA = 2 * 1024 * 1024 * 1024; // 2 GB

                if (f.sha256 && opfsFile.size <= SIZE_LIMIT_FOR_JS_SHA) {
                    // Small/medium files: full SHA-256 from OPFS.
                    console.log(`[bw-sw] ${f.filename}: verifying SHA-256 (${opfsFile.size} bytes)...`);
                    broadcastDl({ type: 'model_progress', detail: { phase: 'verifying', modelId, file: f.filename } });
                    try {
                        const hex = await streamingSha256FromFile(opfsFile);
                        if (hex && hex !== f.sha256) {
                            console.error(`[bw-sw] ${f.filename}: SHA mismatch! got=${hex} expected=${f.sha256}`);
                            await deleteOpfsModel(modelId);
                            throw new Error(`SHA-256 mismatch for ${f.filename}`);
                        }
                        console.log(`[bw-sw] ${f.filename}: SHA-256 verified ✓`);
                    } catch (shaErr) {
                        if (shaErr && shaErr.message && shaErr.message.includes('SHA-256 mismatch')) throw shaErr;
                        console.warn(`[bw-sw] ${f.filename}: SHA-256 verification failed, using size check:`, shaErr.message);
                        if (contentLength > 0 && opfsFile.size !== contentLength) {
                            await deleteOpfsModel(modelId);
                            throw new Error(`size mismatch for ${f.filename}: got ${opfsFile.size}, expected ${contentLength}`);
                        }
                    }
                } else if (contentLength > 0) {
                    // Large files: JS SHA-256 is too slow / fails in SW.
                    // Verify size matches content-length (HTTPS already
                    // guarantees integrity of the bytes received).
                    if (opfsFile.size !== contentLength) {
                        console.error(`[bw-sw] ${f.filename}: size mismatch! got=${opfsFile.size} expected=${contentLength}`);
                        await deleteOpfsModel(modelId);
                        throw new Error(`size mismatch for ${f.filename}`);
                    }
                    console.log(`[bw-sw] ${f.filename}: size verified (${opfsFile.size} bytes) ✓`);
                }
            }

            // Write a .verified marker so future runs skip re-download.
            if (opfsHandle) {
                try {
                    const modelDir = await opfsDir.getDirectoryHandle(modelId, { create: true });
                    const marker = await modelDir.getFileHandle(f.filename + '.verified', { create: true });
                    const mw = await marker.createWritable();
                    await mw.write('ok');
                    await mw.close();
                } catch (markerErr) {
                    console.warn(`[bw-sw] ${f.filename}: failed to write .verified marker`, markerErr);
                }
            }

            // OPFS is the durable store — getModelBytes() reads from here.
            console.log(`[bw-sw] ${f.filename}: done (stored in OPFS)`);
            emitProgress(f, fileBytesDone, contentLength, true);
        }

        broadcastDl({ type: 'model_download_done', modelId });
    } catch (e) {
        const isAbort = controller.signal.aborted ||
            (e && e.name === 'AbortError') ||
            (e && e.message && e.message.includes('abort'));
        const errorMsg = isAbort ? 'aborted' : (e.message || String(e));
        if (!isAbort) console.error('[bw-sw] download error:', errorMsg);
        const clients = await self.clients.matchAll({ type: 'window' });
        for (const c of clients) {
            c.postMessage({ type: 'model_download_error', modelId, error: errorMsg });
        }
    } finally {
        activeModelDownloads.delete(modelId);
    }
}

function replyTo(event, payload) {
    if (event.source && typeof event.source.postMessage === 'function') {
        event.source.postMessage(payload);
    }
}

async function broadcast(payload) {
    const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    for (const c of clients) {
        try { c.postMessage(payload); } catch (_err) { console.warn("[bw] caught:", _err); }
    }
}

// ── Chat streaming ─────────────────────────────────────────────

/**
 * Re-import the session key the page handed us. Accepts either a
 * `CryptoKey` (preferred — `postMessage` clones it) or 32 raw bytes that
 * we re-import as AES-GCM.
 */
async function importSessionKey(sessionKey) {
    if (sessionKey && typeof sessionKey === 'object' && 'algorithm' in sessionKey && 'type' in sessionKey) {
        return sessionKey; // already a CryptoKey
    }
    const bytes = sessionKey instanceof Uint8Array
        ? sessionKey
        : (sessionKey && sessionKey.buffer ? new Uint8Array(sessionKey.buffer) : null);
    if (!bytes || bytes.length !== 32) {
        throw new Error('chat_start: sessionKey must be a CryptoKey or 32 raw bytes');
    }
    return crypto.subtle.importKey(
        'raw',
        bytes,
        { name: 'AES-GCM' },
        false,
        ['decrypt'],
    );
}

/**
 * Decrypt the API key blob the page handed in. The blob is a packed
 * base64url string from `crypto-store.pack()`.
 */
async function decryptApiKey(apiKeyEncrypted, sessionKey) {
    const key = await importSessionKey(sessionKey);
    const blob = cryptoUnpack(apiKeyEncrypted);
    return cryptoDecrypt(key, { iv: blob.iv, ciphertext: blob.ciphertext });
}

/**
 * Long-lived streaming task. Wrapped in event.waitUntil() by the caller.
 *
 * Persistence rule: flush to IndexedDB every 32 chunks OR every 250ms,
 * whichever comes first. Final flush on stream end / abort / error.
 */
async function handleChatStart(msg, event) {
    const { conversationId, messageId, provider, requestPayload, apiKeyEncrypted, sessionKey } = msg;
    const toolUseParser = (provider && TOOL_USE_PARSERS[provider]) || null;
    // Per-stream accumulator for provider-specific tool_use reassembly.
    // Pure data; mutated by the parser across events.
    const toolAcc = {};

    if (!conversationId || !messageId || !requestPayload) {
        replyTo(event, { type: 'chat_error', conversationId, messageId, error: 'missing required fields' });
        return;
    }
    if (activeStreams.has(messageId)) {
        replyTo(event, { type: 'chat_error', conversationId, messageId, error: 'already streaming' });
        return;
    }

    let apiKey = null;
    if (apiKeyEncrypted) {
        try {
            apiKey = await decryptApiKey(apiKeyEncrypted, sessionKey);
        } catch (e) {
            replyTo(event, {
                type: 'chat_error',
                conversationId,
                messageId,
                error: 'decrypt_failed: ' + (e && e.message ? e.message : String(e)),
            });
            return;
        }
    }

    const abortController = new AbortController();
    const state = {
        conversationId,
        abortController,
        tokensReceived: 0,
        startedAt: Date.now(),
    };
    activeStreams.set(messageId, state);

    // Buffered delta — flushed every 32 chunks or 250ms.
    let pending = '';
    let pendingCount = 0;
    let lastFlushAt = Date.now();
    const FLUSH_TOKENS = 32;
    const FLUSH_MS = 250;

    const flush = async (final) => {
        if (pending.length === 0 && !final) return;
        const delta = pending;
        pending = '';
        pendingCount = 0;
        lastFlushAt = Date.now();
        if (delta.length > 0) {
            try {
                await appendMessageChunk(conversationId, messageId, delta);
            } catch (e) {
                warn('appendMessageChunk failed', e && e.message);
            }
        }
    };

    const maybeFlush = async () => {
        if (pendingCount >= FLUSH_TOKENS || (Date.now() - lastFlushAt) >= FLUSH_MS) {
            await flush(false);
        }
    };

    const usage = null;

    try {
        const { url, method = 'POST', headers = {}, body, format } = requestPayload;
        if (!url) throw new Error('requestPayload.url required');
        if (format !== 'sse' && format !== 'ndjson') {
            throw new Error('requestPayload.format must be "sse" or "ndjson"');
        }

        // Caller embeds the sentinel '__API_KEY__' inside header values
        // and (for Gemini) the URL query string; the SW substitutes
        // the decrypted plaintext after the postMessage boundary so
        // the page never has to hold the plaintext key in memory
        // alongside the request envelope. See providers/index.js for
        // the full contract.
        const finalHeaders = { ...headers };
        let finalUrl = url;
        if (apiKey !== null) {
            for (const k of Object.keys(finalHeaders)) {
                if (typeof finalHeaders[k] === 'string' && finalHeaders[k].includes('__API_KEY__')) {
                    finalHeaders[k] = finalHeaders[k].split('__API_KEY__').join(apiKey);
                }
            }
            if (finalUrl.includes('__API_KEY__')) {
                finalUrl = finalUrl.split('__API_KEY__').join(encodeURIComponent(apiKey));
            }
        }

        const resp = await fetch(finalUrl, {
            method,
            headers: finalHeaders,
            body: body !== undefined && method !== 'GET' ? body : undefined,
            signal: abortController.signal,
        });

        if (!resp.ok) {
            const text = await resp.text().catch(() => '');
            throw new Error(`HTTP ${resp.status}: ${text.slice(0, 256)}`);
        }

        for await (const ev of streamFromResponse(resp, format)) {
            if (abortController.signal.aborted) break;

            let delta = '';
            if (format === 'sse') {
                if (ev && ev.done) break;
                // Caller's `body` shape is provider-specific; the SW does
                // NOT decode the JSON. We hand the raw `data` through and
                // let the page's provider adapter build the user-visible
                // text. For storage/broadcast purposes we treat the raw
                // SSE data line as the "delta" — tasks #6/7 will refine
                // this once provider adapters land.
                delta = ev && typeof ev.data === 'string' ? ev.data : '';
            } else {
                // NDJSON: pass-through as a stringified line.
                delta = typeof ev === 'string' ? ev : JSON.stringify(ev);
            }

            if (delta.length === 0) continue;

            pending += delta;
            pendingCount += 1;
            state.tokensReceived += 1;

            // Broadcast every chunk immediately so the UI feels live;
            // IndexedDB writes are debounced separately.
            broadcast({
                type: 'chat_chunk',
                conversationId,
                messageId,
                delta,
                raw: format === 'sse' ? { event: ev.event, data: ev.data } : ev,
            }).catch((e) => { console.warn("[bw] swallowed:", e); });

            // MCP tool_use plumbing: when this provider has a parser
            // and reassembles a complete tool_use across deltas, emit
            // it on the same broadcast channel. UI execution loop /
            // bubble rendering land in the next commit.
            if (toolUseParser && format === 'sse') {
                let parsed = null;
                try { parsed = toolUseParser(ev, toolAcc); } catch (_) { parsed = null; }
                if (parsed) {
                    if (parsed.tool_use) {
                        broadcast({
                            type: 'chat_tool_use',
                            conversationId,
                            messageId,
                            tool_use: parsed.tool_use,
                        }).catch((e) => { console.warn("[bw] swallowed:", e); });
                    }
                    if (Array.isArray(parsed.tool_uses)) {
                        for (const tu of parsed.tool_uses) {
                            broadcast({
                                type: 'chat_tool_use',
                                conversationId,
                                messageId,
                                tool_use: tu,
                            }).catch((e) => { console.warn("[bw] swallowed:", e); });
                        }
                    }
                }
            }

            await maybeFlush();
        }

        // Final flush before the done message.
        await flush(true);

        // Stamp final updatedAt + persisted state.
        try {
            await putMessage({
                conversationId,
                messageId,
                role: 'assistant',
                content: undefined, // appendMessageChunk owns content; don't clobber
                updatedAt: Date.now(),
                completedAt: Date.now(),
                tokensReceived: state.tokensReceived,
            });
        } catch (e) {
            // Final stamp is best-effort; the chunk-appended row is the source of truth.
            log('putMessage final stamp failed', e && e.message);
        }

        broadcast({
            type: 'chat_done',
            conversationId,
            messageId,
            usage,
            tokensReceived: state.tokensReceived,
        }).catch((e) => { console.warn("[bw] swallowed:", e); });
        replyTo(event, { type: 'chat_done', conversationId, messageId, usage });

        // Background notification: only when no foreground window is alive.
        const visibleClients = await self.clients.matchAll({ type: 'window' });
        if (visibleClients.length === 0 && self.registration && self.registration.showNotification) {
            try {
                await self.registration.showNotification('Brainwires Chat', {
                    body: 'Response ready',
                    tag: messageId,
                    icon: './icons/icon-192.png',
                    badge: './icons/icon-192.png',
                    data: { conversationId, messageId },
                });
            } catch (e) {
                log('showNotification failed', e && e.message);
            }
        }
    } catch (err) {
        await flush(true);
        const errorText = err && err.message ? err.message : String(err);
        const aborted = abortController.signal.aborted || (err && err.name === 'AbortError');
        broadcast({
            type: aborted ? 'chat_aborted' : 'chat_error',
            conversationId,
            messageId,
            error: aborted ? 'aborted' : errorText,
        }).catch((e) => { console.warn("[bw] swallowed:", e); });
        if (!aborted) {
            replyTo(event, { type: 'chat_error', conversationId, messageId, error: errorText });
        }
    } finally {
        activeStreams.delete(messageId);
        // Best-effort: clear the in-memory plaintext API key reference.
        apiKey = null;
    }
}

// ── Notification click ─────────────────────────────────────────

self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    const data = event.notification.data || {};
    event.waitUntil((async () => {
        const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
        for (const c of clients) {
            try {
                c.postMessage({ type: 'open_chat', ...data });
                if ('focus' in c) return c.focus();
            } catch (_err) { console.warn("[bw] caught:", _err); }
        }
        if (self.clients.openWindow) {
            return self.clients.openWindow('./index.html');
        }
    })());
});
