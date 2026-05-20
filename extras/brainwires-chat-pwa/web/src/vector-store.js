// brainwires-chat-pwa — OPFS-backed persistence for LocalVectorIndex
//
// Saves/loads the HNSW index to/from OPFS so vector data survives page
// reloads. The WASM crate provides toBytes()/fromBytes(); this module
// handles the file I/O.
//
// Usage:
//   import { saveIndex, loadIndex, deleteIndex } from './vector-store.js';
//   const idx = new wasm.LocalVectorIndex('conversations', 384);
//   idx.insert(embedding, metaJson);
//   await saveIndex(idx);              // → writes to OPFS
//   const restored = await loadIndex(wasm, 'conversations');  // → fromBytes

const OPFS_DIR = 'hnsw-indexes';

async function getDir() {
    const root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle(OPFS_DIR, { create: true });
}

/**
 * Save a LocalVectorIndex to OPFS.
 * @param {LocalVectorIndex} index — the WASM index handle
 */
export async function saveIndex(index) {
    const bytes = index.toBytes();
    const dir = await getDir();
    const handle = await dir.getFileHandle(`${index.name}.bin`, { create: true });
    const writable = await handle.createWritable();
    await writable.write(bytes);
    await writable.close();
    console.log(`[vector-store] saved ${index.name} (${bytes.length} bytes)`);
}

/**
 * Load a LocalVectorIndex from OPFS. Returns null if not found.
 * @param {object} wasm — the WASM module (must have LocalVectorIndex.fromBytes)
 * @param {string} name — collection name
 * @returns {Promise<LocalVectorIndex|null>}
 */
export async function loadIndex(wasm, name) {
    try {
        const dir = await getDir();
        const handle = await dir.getFileHandle(`${name}.bin`);
        const file = await handle.getFile();
        const buf = await file.arrayBuffer();
        const bytes = new Uint8Array(buf);
        const index = wasm.LocalVectorIndex.fromBytes(bytes);
        console.log(`[vector-store] loaded ${name} (${index.len} vectors)`);
        return index;
    } catch (_e) {
        return null;
    }
}

/**
 * Delete a saved index from OPFS.
 * @param {string} name — collection name
 */
export async function deleteIndex(name) {
    try {
        const dir = await getDir();
        await dir.removeEntry(`${name}.bin`);
        console.log(`[vector-store] deleted ${name}`);
    } catch (_) { /* file may not exist */ }
}

/**
 * List all saved index names.
 * @returns {Promise<string[]>}
 */
export async function listIndexes() {
    try {
        const dir = await getDir();
        const names = [];
        for await (const [name] of dir.entries()) {
            if (name.endsWith('.bin')) names.push(name.replace(/\.bin$/, ''));
        }
        return names;
    } catch (_) {
        return [];
    }
}
