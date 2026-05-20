// brainwires-chat-pwa — OPFS writer Dedicated Worker
//
// Uses FileSystemSyncAccessHandle for zero-copy, in-place writes to OPFS.
// This avoids the O(n^2) overhead of createWritable({ keepExistingData: true })
// which copies the entire file into a swap file on every open. The sync
// handle API is **only available in Worker contexts** — calling it from
// the main thread throws `createSyncAccessHandle is not a function`.
//
// Wire protocol (main → worker):
//   {
//     type: 'start',
//     modelId,                    // logging label (not part of OPFS path)
//     dirPath: ['ollama', 'lib_x__y'],  // segments under model-downloads/
//     filename,
//     url,
//     headers,
//     offset,
//   }
//   { type: 'cancel' }
//
// `dirPath` is the path under `model-downloads/` to the file's parent
// directory, expressed as an array of directory segments. The worker
// creates each segment as needed. Older callers that pass only
// `modelId` (no `dirPath`) get the legacy single-segment layout
// `model-downloads/<modelId>/<filename>` for back-compat.
//
// Wire protocol (worker → main):
//   { type: 'progress', bytesWritten, totalBytes, chunkBytes }
//   { type: 'done', totalBytes }
//   { type: 'error', error }
//   { type: 'cancelled', bytesWritten }

const OPFS_DIR = 'model-downloads';
const FLUSH_INTERVAL = 100 * 1024 * 1024; // 100 MB

let cancelled = false;
let activeReader = null;

self.addEventListener('message', (ev) => {
    const msg = ev.data;
    if (!msg || typeof msg !== 'object') return;
    if (msg.type === 'start') handleStart(msg);
    else if (msg.type === 'cancel') handleCancel();
});

async function handleStart(msg) {
    const { modelId, filename, url, headers, offset } = msg;
    // `dirPath` is the new (multi-segment) path API; fall back to
    // single-segment `[modelId]` for legacy callers.
    const dirPath = Array.isArray(msg.dirPath) && msg.dirPath.length > 0
        ? msg.dirPath
        : [modelId];
    cancelled = false;

    let syncHandle = null;
    let currentOffset = offset || 0;
    let bytesSinceFlush = 0;

    try {
        // Open OPFS directory structure. Walks every segment in
        // `dirPath` under `model-downloads/`, creating as needed.
        const root = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: true });
        let parentDir = dlDir;
        for (const seg of dirPath) {
            parentDir = await parentDir.getDirectoryHandle(seg, { create: true });
        }
        const fileHandle = await parentDir.getFileHandle(filename, { create: true });

        console.log(`[opfs-writer] opening sync handle for ${modelId}/${filename} at offset ${currentOffset}`);
        syncHandle = await fileHandle.createSyncAccessHandle();

        // Truncate to resume offset to discard any corrupt trailing bytes
        if (currentOffset > 0) {
            syncHandle.truncate(currentOffset);
            syncHandle.flush();
        }

        // Fetch the remaining bytes
        const fetchHeaders = { ...(headers || {}) };
        if (currentOffset > 0) fetchHeaders['Range'] = `bytes=${currentOffset}-`;

        const resp = await fetch(url, { headers: fetchHeaders });

        if (resp.status === 416) {
            // Already complete
            const size = syncHandle.getSize();
            console.log(`[opfs-writer] ${filename}: 416 — already complete (${size} bytes)`);
            syncHandle.close();
            self.postMessage({ type: 'done', totalBytes: size });
            return;
        }

        if (resp.status === 401 || resp.status === 403) {
            syncHandle.close();
            self.postMessage({ type: 'error', error: `HF responded ${resp.status}` });
            return;
        }

        if (!resp.ok && resp.status !== 206) {
            syncHandle.close();
            self.postMessage({ type: 'error', error: `fetch failed (${resp.status})` });
            return;
        }

        // If server returned 200 instead of 206, it sent the full file
        if (resp.status === 200 && currentOffset > 0) {
            console.log(`[opfs-writer] ${filename}: server sent full file (200 instead of 206), resetting`);
            syncHandle.truncate(0);
            currentOffset = 0;
        }

        const contentLength = Number(resp.headers.get('content-length')) || 0;
        const totalBytes = currentOffset + contentLength;
        console.log(`[opfs-writer] ${filename}: downloading (status=${resp.status}, contentLength=${contentLength}, total=${totalBytes})`);

        const reader = resp.body.getReader();
        activeReader = reader;

        while (true) {
            if (cancelled) {
                reader.cancel().catch(() => {});
                break;
            }

            const { value, done } = await reader.read();
            if (done) break;

            // Synchronous write — no copy, no swap file
            const written = syncHandle.write(value, { at: currentOffset });
            currentOffset += written;
            bytesSinceFlush += written;

            // Periodic flush for durability
            if (bytesSinceFlush >= FLUSH_INTERVAL) {
                syncHandle.flush();
                bytesSinceFlush = 0;
                console.log(`[opfs-writer] ${filename}: flushed at ${currentOffset} bytes`);
            }

            self.postMessage({
                type: 'progress',
                bytesWritten: currentOffset,
                totalBytes,
                chunkBytes: written,
            });
        }

        // Final flush + close
        syncHandle.flush();
        syncHandle.close();
        syncHandle = null;
        activeReader = null;

        if (cancelled) {
            console.log(`[opfs-writer] ${filename}: cancelled at ${currentOffset} bytes`);
            self.postMessage({ type: 'cancelled', bytesWritten: currentOffset });
        } else {
            console.log(`[opfs-writer] ${filename}: complete (${currentOffset} bytes)`);
            self.postMessage({ type: 'done', totalBytes: currentOffset });
        }
    } catch (err) {
        const error = err && err.message ? err.message : String(err);
        console.error('[opfs-writer] error:', error);
        if (syncHandle) {
            try { syncHandle.flush(); } catch (_) {}
            try { syncHandle.close(); } catch (_) {}
        }
        activeReader = null;
        self.postMessage({ type: 'error', error });
    }
}

function handleCancel() {
    console.log('[opfs-writer] cancel requested');
    cancelled = true;
    if (activeReader && typeof activeReader.cancel === 'function') {
        try { activeReader.cancel(); } catch (_) {}
    }
}
