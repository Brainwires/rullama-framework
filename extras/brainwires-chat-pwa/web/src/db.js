// brainwires-chat-pwa — IndexedDB wrapper
//
// One DB, four stores. Streaming chunks are persisted via
// `appendMessageChunk` from both the SW (network streams) and the page
// (local-wasm streams). All operations are async; no external deps.

const DB_NAME = 'bw-chat-db';
const DB_VERSION = 2;

// ── Schema ─────────────────────────────────────────────────────
//
// conversations: { id, title, createdAt, updatedAt, ... }
//   - keyPath: 'id' (string)
//   - index byUpdatedAt → updatedAt
//
// messages: { conversationId, messageId, role, content, usage, ... }
//   - keyPath: ['conversationId', 'messageId']
//   - index byConversation → conversationId
//   - `content` may be a string (legacy) or an array of parts:
//     [{type:'text', text}, {type:'image', mediaType, data},
//      {type:'tool_use', id, name, input},
//      {type:'tool_result', toolUseId, content}]
//     Read-time normalization (see normalizeContent below) wraps legacy
//     string content into a single text part.
//
// settings: { key, value }
//   - keyPath: 'key'
//
// voicePrefs: { key, value }
//   - keyPath: 'key'
//
// attachments: { id, conversationId, messageId?, kind, mediaType, bytes, name?, dataUrl?, ... }
//   - keyPath: 'id'
//   - index byMessage → messageId
//   - index byConversation → conversationId
//
// ragDocs: { id, conversationId|null, name, type, bytes, ingestedAt }
//   - keyPath: 'id'
//   - index byConversation → conversationId  (null = global library)
//
// ragChunks: { id, docId, conversationId|null, page?, text, embeddingDim }
//   - keyPath: 'id'
//   - index byDoc → docId
//   - index byConversation → conversationId
//
// mcpServers: { id, url, displayName, headers?, enabledByDefault }
//   - keyPath: 'id'
//
// mcpToolState: { conversationId, serverId, toolName, enabled }
//   - keyPath: ['conversationId', 'serverId', 'toolName']
//   - index byConversation → conversationId

let _dbPromise = null;

/**
 * Open (or create) the brainwires chat database. Subsequent calls return
 * the cached connection — IndexedDB sessions are cheap to reuse.
 *
 * @returns {Promise<IDBDatabase>}
 */
export function openDb() {
    if (_dbPromise) return _dbPromise;
    _dbPromise = new Promise((resolve, reject) => {
        let resolved = false;
        const timeout = setTimeout(() => {
            if (!resolved) { resolved = true; reject(new Error('IndexedDB open timeout')); }
        }, 5000);

        const finish = (fn, value) => {
            if (resolved) return;
            resolved = true;
            clearTimeout(timeout);
            fn(value);
        };

        try {
            const req = indexedDB.open(DB_NAME, DB_VERSION);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('conversations')) {
                    const s = db.createObjectStore('conversations', { keyPath: 'id' });
                    s.createIndex('byUpdatedAt', 'updatedAt', { unique: false });
                }
                if (!db.objectStoreNames.contains('messages')) {
                    const s = db.createObjectStore('messages', {
                        keyPath: ['conversationId', 'messageId'],
                    });
                    s.createIndex('byConversation', 'conversationId', { unique: false });
                }
                if (!db.objectStoreNames.contains('settings')) {
                    db.createObjectStore('settings', { keyPath: 'key' });
                }
                if (!db.objectStoreNames.contains('voicePrefs')) {
                    db.createObjectStore('voicePrefs', { keyPath: 'key' });
                }
                // v2: attachments + RAG + MCP. Existing rows stay
                // untouched; messages with string `content` are handled
                // by normalizeContent() at read time.
                if (!db.objectStoreNames.contains('attachments')) {
                    const s = db.createObjectStore('attachments', { keyPath: 'id' });
                    s.createIndex('byMessage', 'messageId', { unique: false });
                    s.createIndex('byConversation', 'conversationId', { unique: false });
                }
                if (!db.objectStoreNames.contains('ragDocs')) {
                    const s = db.createObjectStore('ragDocs', { keyPath: 'id' });
                    s.createIndex('byConversation', 'conversationId', { unique: false });
                }
                if (!db.objectStoreNames.contains('ragChunks')) {
                    const s = db.createObjectStore('ragChunks', { keyPath: 'id' });
                    s.createIndex('byDoc', 'docId', { unique: false });
                    s.createIndex('byConversation', 'conversationId', { unique: false });
                }
                if (!db.objectStoreNames.contains('mcpServers')) {
                    db.createObjectStore('mcpServers', { keyPath: 'id' });
                }
                if (!db.objectStoreNames.contains('mcpToolState')) {
                    const s = db.createObjectStore('mcpToolState', {
                        keyPath: ['conversationId', 'serverId', 'toolName'],
                    });
                    s.createIndex('byConversation', 'conversationId', { unique: false });
                }
            };
            req.onsuccess = () => finish(resolve, req.result);
            req.onerror = () => finish(reject, req.error);
            req.onblocked = () => finish(reject, new Error('IndexedDB blocked'));
        } catch (e) {
            finish(reject, e);
        }
    });
    return _dbPromise;
}

/**
 * Reset the cached connection. Test-only; production code should never need this.
 */
export function _resetDbForTests() {
    _dbPromise = null;
}

// ── Generic helpers (internal) ─────────────────────────────────

function txPromise(tx) {
    return new Promise((resolve, reject) => {
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error || new Error('Transaction aborted'));
    });
}

function reqPromise(req) {
    return new Promise((resolve, reject) => {
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

// ── Conversations ──────────────────────────────────────────────

/**
 * Upsert a conversation row. `updatedAt` is stamped to `Date.now()` if absent
 * so the byUpdatedAt index is always populated.
 *
 * @param {{ id: string, [k: string]: any }} conversation
 */
export async function putConversation(conversation) {
    if (!conversation || !conversation.id) {
        throw new Error('putConversation: id required');
    }
    const row = { ...conversation };
    if (row.updatedAt === undefined) row.updatedAt = Date.now();
    if (row.createdAt === undefined) row.createdAt = row.updatedAt;
    const db = await openDb();
    const tx = db.transaction('conversations', 'readwrite');
    tx.objectStore('conversations').put(row);
    await txPromise(tx);
    return row;
}

/**
 * @param {string} id
 * @returns {Promise<object | undefined>}
 */
export async function getConversation(id) {
    const db = await openDb();
    const tx = db.transaction('conversations', 'readonly');
    return reqPromise(tx.objectStore('conversations').get(id));
}

/**
 * List all conversations, newest-first by updatedAt.
 * @returns {Promise<object[]>}
 */
export async function listConversations() {
    const db = await openDb();
    const tx = db.transaction('conversations', 'readonly');
    const idx = tx.objectStore('conversations').index('byUpdatedAt');
    // openCursor with 'prev' walks the index in reverse (newest first).
    return new Promise((resolve, reject) => {
        const out = [];
        const req = idx.openCursor(null, 'prev');
        req.onsuccess = () => {
            const cur = req.result;
            if (!cur) { resolve(out); return; }
            out.push(cur.value);
            cur.continue();
        };
        req.onerror = () => reject(req.error);
    });
}

/**
 * Delete a conversation and cascade-delete its messages.
 *
 * @param {string} id
 */
export async function deleteConversation(id) {
    const db = await openDb();
    const tx = db.transaction(['conversations', 'messages'], 'readwrite');
    tx.objectStore('conversations').delete(id);
    const msgIdx = tx.objectStore('messages').index('byConversation');
    const cursorReq = msgIdx.openCursor(IDBKeyRange.only(id));
    await new Promise((resolve, reject) => {
        cursorReq.onsuccess = () => {
            const cur = cursorReq.result;
            if (!cur) { resolve(); return; }
            cur.delete();
            cur.continue();
        };
        cursorReq.onerror = () => reject(cursorReq.error);
    });
    await txPromise(tx);
}

// ── Messages ───────────────────────────────────────────────────

/**
 * Append `delta` text to the message identified by [conversationId, messageId].
 * Read-modify-write under one readwrite transaction so concurrent SW writes
 * don't lose data. Returns the message row after the append.
 *
 * @param {string} conversationId
 * @param {string} messageId
 * @param {string} delta
 * @returns {Promise<object>}
 */
export async function appendMessageChunk(conversationId, messageId, delta) {
    const db = await openDb();
    const tx = db.transaction('messages', 'readwrite');
    const store = tx.objectStore('messages');
    const existing = await reqPromise(store.get([conversationId, messageId]));
    const row = existing || {
        conversationId,
        messageId,
        role: 'assistant',
        content: '',
        createdAt: Date.now(),
        updatedAt: Date.now(),
    };
    const d = delta || '';
    if (Array.isArray(row.content)) {
        // Parts-shaped row: append text to the trailing text part, or push
        // a new one. Non-text parts (image, tool_use) keep their position.
        const last = row.content[row.content.length - 1];
        if (last && last.type === 'text') {
            last.text = (last.text || '') + d;
        } else {
            row.content.push({ type: 'text', text: d });
        }
    } else {
        row.content = (row.content || '') + d;
    }
    row.updatedAt = Date.now();
    store.put(row);
    await txPromise(tx);
    return row;
}

/**
 * @param {string} conversationId
 * @param {string} messageId
 */
export async function getMessage(conversationId, messageId) {
    const db = await openDb();
    const tx = db.transaction('messages', 'readonly');
    return reqPromise(tx.objectStore('messages').get([conversationId, messageId]));
}

/**
 * Replace a message row wholesale (e.g. final write at stream end).
 *
 * @param {object} row
 */
export async function putMessage(row) {
    if (!row || !row.conversationId || !row.messageId) {
        throw new Error('putMessage: conversationId + messageId required');
    }
    const next = { ...row };
    if (next.updatedAt === undefined) next.updatedAt = Date.now();
    const db = await openDb();
    const tx = db.transaction('messages', 'readwrite');
    tx.objectStore('messages').put(next);
    await txPromise(tx);
    return next;
}

/**
 * @param {string} conversationId
 * @returns {Promise<object[]>} messages for the conversation, oldest first by createdAt.
 */
export async function listMessages(conversationId) {
    const db = await openDb();
    const tx = db.transaction('messages', 'readonly');
    const idx = tx.objectStore('messages').index('byConversation');
    const req = idx.getAll(IDBKeyRange.only(conversationId));
    const rows = await reqPromise(req);
    rows.sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
    return rows;
}

// ── Content shape helpers ──────────────────────────────────────

/**
 * Normalize a message row's `content` to the parts[] shape. Legacy rows
 * (string content) are wrapped as a single text part. Already-array
 * content is returned unchanged. Null/undefined returns an empty array.
 *
 * @param {string | Array | null | undefined} content
 * @returns {Array<{type: string, [k: string]: any}>}
 */
export function normalizeContent(content) {
    if (Array.isArray(content)) return content;
    if (typeof content === 'string') return content ? [{ type: 'text', text: content }] : [];
    return [];
}

/**
 * Flatten a parts[] (or legacy string) into the concatenated text. Image,
 * tool_use, and tool_result parts are skipped. Useful for places that still
 * want a plain-text representation: TTS, copy-to-clipboard, conversation
 * title snippet, search.
 *
 * @param {string | Array | null | undefined} content
 * @returns {string}
 */
export function partsToText(content) {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';
    return content
        .filter((p) => p && p.type === 'text' && typeof p.text === 'string')
        .map((p) => p.text)
        .join('');
}

// ── Attachments ────────────────────────────────────────────────

export async function putAttachment(row) {
    if (!row || !row.id) throw new Error('putAttachment: id required');
    const next = { ...row };
    if (next.createdAt === undefined) next.createdAt = Date.now();
    const db = await openDb();
    const tx = db.transaction('attachments', 'readwrite');
    tx.objectStore('attachments').put(next);
    await txPromise(tx);
    return next;
}

export async function getAttachment(id) {
    const db = await openDb();
    const tx = db.transaction('attachments', 'readonly');
    return reqPromise(tx.objectStore('attachments').get(id));
}

export async function listAttachmentsByMessage(messageId) {
    const db = await openDb();
    const tx = db.transaction('attachments', 'readonly');
    const idx = tx.objectStore('attachments').index('byMessage');
    return reqPromise(idx.getAll(IDBKeyRange.only(messageId)));
}

export async function deleteAttachment(id) {
    const db = await openDb();
    const tx = db.transaction('attachments', 'readwrite');
    tx.objectStore('attachments').delete(id);
    await txPromise(tx);
}

// ── RAG documents + chunks ─────────────────────────────────────

export async function putRagDoc(doc) {
    if (!doc || !doc.id) throw new Error('putRagDoc: id required');
    const next = { ...doc };
    if (next.ingestedAt === undefined) next.ingestedAt = Date.now();
    if (next.conversationId === undefined) next.conversationId = null;
    const db = await openDb();
    const tx = db.transaction('ragDocs', 'readwrite');
    tx.objectStore('ragDocs').put(next);
    await txPromise(tx);
    return next;
}

export async function listRagDocs(conversationId) {
    const db = await openDb();
    const tx = db.transaction('ragDocs', 'readonly');
    const store = tx.objectStore('ragDocs');
    if (conversationId === undefined) {
        return reqPromise(store.getAll());
    }
    if (conversationId === null) {
        // null isn't a valid IDB key, so the byConversation index can't be
        // queried with IDBKeyRange.only(null). Scan + filter instead — the
        // global library is expected to stay small.
        const all = await reqPromise(store.getAll());
        return all.filter((d) => d.conversationId == null);
    }
    const idx = store.index('byConversation');
    return reqPromise(idx.getAll(IDBKeyRange.only(conversationId)));
}

export async function deleteRagDoc(id) {
    const db = await openDb();
    const tx = db.transaction(['ragDocs', 'ragChunks'], 'readwrite');
    tx.objectStore('ragDocs').delete(id);
    const idx = tx.objectStore('ragChunks').index('byDoc');
    const cursorReq = idx.openCursor(IDBKeyRange.only(id));
    await new Promise((resolve, reject) => {
        cursorReq.onsuccess = () => {
            const cur = cursorReq.result;
            if (!cur) { resolve(); return; }
            cur.delete();
            cur.continue();
        };
        cursorReq.onerror = () => reject(cursorReq.error);
    });
    await txPromise(tx);
}

export async function putRagChunks(rows) {
    if (!Array.isArray(rows) || rows.length === 0) return;
    const db = await openDb();
    const tx = db.transaction('ragChunks', 'readwrite');
    const store = tx.objectStore('ragChunks');
    for (const r of rows) {
        if (!r || !r.id) continue;
        store.put(r);
    }
    await txPromise(tx);
}

export async function listRagChunksByDoc(docId) {
    const db = await openDb();
    const tx = db.transaction('ragChunks', 'readonly');
    const idx = tx.objectStore('ragChunks').index('byDoc');
    return reqPromise(idx.getAll(IDBKeyRange.only(docId)));
}

// ── MCP servers + per-conversation tool state ─────────────────

export async function putMcpServer(row) {
    if (!row || !row.id) throw new Error('putMcpServer: id required');
    const db = await openDb();
    const tx = db.transaction('mcpServers', 'readwrite');
    tx.objectStore('mcpServers').put(row);
    await txPromise(tx);
    return row;
}

export async function listMcpServers() {
    const db = await openDb();
    const tx = db.transaction('mcpServers', 'readonly');
    return reqPromise(tx.objectStore('mcpServers').getAll());
}

export async function deleteMcpServer(id) {
    const db = await openDb();
    const tx = db.transaction('mcpServers', 'readwrite');
    tx.objectStore('mcpServers').delete(id);
    await txPromise(tx);
}

export async function setMcpToolEnabled(conversationId, serverId, toolName, enabled) {
    const db = await openDb();
    const tx = db.transaction('mcpToolState', 'readwrite');
    tx.objectStore('mcpToolState').put({ conversationId, serverId, toolName, enabled: !!enabled });
    await txPromise(tx);
}

export async function listMcpToolStateForConversation(conversationId) {
    const db = await openDb();
    const tx = db.transaction('mcpToolState', 'readonly');
    const idx = tx.objectStore('mcpToolState').index('byConversation');
    return reqPromise(idx.getAll(IDBKeyRange.only(conversationId)));
}

// ── Settings / voicePrefs ──────────────────────────────────────

export async function setSetting(key, value) {
    const db = await openDb();
    const tx = db.transaction('settings', 'readwrite');
    tx.objectStore('settings').put({ key, value });
    await txPromise(tx);
}

export async function getSetting(key) {
    const db = await openDb();
    const tx = db.transaction('settings', 'readonly');
    const row = await reqPromise(tx.objectStore('settings').get(key));
    return row ? row.value : undefined;
}

export async function setVoicePref(key, value) {
    const db = await openDb();
    const tx = db.transaction('voicePrefs', 'readwrite');
    tx.objectStore('voicePrefs').put({ key, value });
    await txPromise(tx);
}

export async function getVoicePref(key) {
    const db = await openDb();
    const tx = db.transaction('voicePrefs', 'readonly');
    const row = await reqPromise(tx.objectStore('voicePrefs').get(key));
    return row ? row.value : undefined;
}
