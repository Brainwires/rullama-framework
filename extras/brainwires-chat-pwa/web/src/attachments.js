// brainwires-chat-pwa — composer attachments queue
//
// In-memory list of pending attachments for the next send. UI code reads
// `getAll()`, `clear()` resets after a successful submit, and the chat view
// subscribes to 'change' events to re-render the chip strip.
//
// Each entry: { id, kind:'image'|'pdf'|'file', file, name, mediaType, dataUrl? }
// Image dataUrls are pre-decoded for the chip preview; the full base64 used
// for sending is produced lazily by vision.imageToBase64() at submit time.

import { genId } from './utils.js';

const _items = [];
const _emitter = new EventTarget();

function emit() {
    _emitter.dispatchEvent(new CustomEvent('change', { detail: { items: _items.slice() } }));
}

export function on(event, handler) {
    _emitter.addEventListener(event, handler);
    return () => _emitter.removeEventListener(event, handler);
}

export function getAll() {
    return _items.slice();
}

export function count() {
    return _items.length;
}

export function clear() {
    if (_items.length === 0) return;
    _items.length = 0;
    emit();
}

export function remove(id) {
    const i = _items.findIndex((x) => x.id === id);
    if (i < 0) return;
    _items.splice(i, 1);
    emit();
}

function classify(file) {
    const t = (file && file.type) || '';
    if (t.startsWith('image/')) return 'image';
    if (t === 'application/pdf') return 'pdf';
    return 'file';
}

async function imagePreview(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : null);
        reader.onerror = () => resolve(null);
        reader.readAsDataURL(file);
    });
}

/**
 * Add a File/Blob to the pending queue. Returns the new entry.
 *
 * @param {File} file
 * @returns {Promise<object>}
 */
export async function addFile(file) {
    if (!file) return null;
    const kind = classify(file);
    const entry = {
        id: genId('att'),
        kind,
        file,
        name: file.name || (kind === 'image' ? 'image' : 'file'),
        mediaType: file.type || 'application/octet-stream',
        size: file.size || 0,
    };
    if (kind === 'image') {
        entry.dataUrl = await imagePreview(file);
    }
    _items.push(entry);
    emit();
    return entry;
}

/**
 * Convenience: add an array of files in order.
 *
 * @param {Iterable<File>} files
 * @returns {Promise<object[]>}
 */
export async function addFiles(files) {
    const added = [];
    for (const f of files) {
        const e = await addFile(f);
        if (e) added.push(e);
    }
    return added;
}
