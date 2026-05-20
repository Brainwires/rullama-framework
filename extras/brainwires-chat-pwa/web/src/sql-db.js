// brainwires-chat-pwa — rsqlite-wasm database layer
//
// Drop-in replacement for db.js. Same exports, same signatures,
// backed by rsqlite-wasm (pure-Rust SQLite in WebAssembly) with
// OPFS persistence.

import { WorkerDatabase } from '../vendor/rsqlite/dist/worker-proxy.js';
import { exposeForDevtools } from '../vendor/rsqlite/dist/devtools.js';

const DB_NAME = 'bw-chat';
let _dbPromise = null;

const SCHEMA_DDL = `
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at);

CREATE TABLE IF NOT EXISTS messages (
    conversation_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    usage TEXT,
    created_at INTEGER,
    updated_at INTEGER,
    PRIMARY KEY (conversation_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS voice_prefs (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    message_id TEXT,
    kind TEXT,
    media_type TEXT,
    name TEXT,
    data_url TEXT,
    bytes BLOB,
    created_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_attach_msg ON attachments(message_id);
CREATE INDEX IF NOT EXISTS idx_attach_conv ON attachments(conversation_id);

CREATE TABLE IF NOT EXISTS rag_docs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    name TEXT,
    type TEXT,
    bytes INTEGER,
    ingested_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ragdoc_conv ON rag_docs(conversation_id);

CREATE TABLE IF NOT EXISTS rag_chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    conversation_id TEXT,
    page INTEGER,
    text TEXT NOT NULL,
    embedding_dim INTEGER,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_ragchunk_doc ON rag_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_ragchunk_conv ON rag_chunks(conversation_id);

CREATE TABLE IF NOT EXISTS mcp_servers (
    id TEXT PRIMARY KEY,
    url TEXT,
    display_name TEXT,
    headers TEXT,
    enabled_by_default INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS mcp_tool_state (
    conversation_id TEXT NOT NULL,
    server_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (conversation_id, server_id, tool_name)
);

CREATE TABLE IF NOT EXISTS _sync_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    row_key TEXT NOT NULL,
    op TEXT NOT NULL,
    ts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_synclog_seq ON _sync_log(seq);

CREATE TABLE IF NOT EXISTS _sync_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
`;

// JS-side sync changelog — replaces SQL triggers to avoid schema page overflow.
let _applying = false;

async function _logSync(table, rowKey, op) {
    if (_applying) return;
    try {
        const db = await openDb();
        await db.exec(
            `INSERT INTO _sync_log (table_name, row_key, op, ts) VALUES (?, ?, ?, ?)`,
            [table, rowKey, op, Date.now()],
        );
    } catch (_) { /* best-effort */ }
}

/**
 * Error thrown by openDb when another tab already holds the
 * OPFS-sqlite primary lock. Surfaces from boot.js so the UI can show
 * a friendly "open in another tab" notice instead of a generic
 * IndexedDB / NoModificationAllowed dump in DevTools.
 */
export class OpfsTabConflictError extends Error {
    constructor() {
        super(
            'brainwires-chat is already open in another tab. Close the other tab, '
            + 'then reload this one — OPFS sync access handles are exclusive per file.',
        );
        this.name = 'OpfsTabConflictError';
        this.code = 'OPFS_TAB_CONFLICT';
    }
}

const PRIMARY_LOCK_NAME = `bw-chat-opfs-primary:${DB_NAME}`;

/**
 * Acquire the cross-tab OPFS-primary lock. Holds it for the lifetime
 * of this tab via a never-resolving promise, so:
 *   - First tab to load: gets the lock, opens the rsqlite worker.
 *   - Second tab: lock unavailable → throws OpfsTabConflictError.
 *   - First tab closes / reloads: lock auto-releases, next tab can boot.
 *
 * The lock is acquired with `ifAvailable: true` so secondary tabs fail
 * fast instead of hanging at boot.
 */
function acquirePrimaryLock() {
    return new Promise((resolve, reject) => {
        let acquired = false;
        navigator.locks
            .request(PRIMARY_LOCK_NAME, { mode: 'exclusive', ifAvailable: true }, (lock) => {
                if (lock === null) {
                    // Another tab holds it. resolve(false) so the caller
                    // can throw OpfsTabConflictError without rejecting the
                    // outer promise (which would also count as a lock
                    // release).
                    resolve(false);
                    return undefined;
                }
                acquired = true;
                resolve(true);
                // Hold the lock until tab unload by never resolving.
                return new Promise(() => {});
            })
            .catch((err) => {
                // Some browsers / private mode don't expose Web Locks.
                // Treat that as "lock acquired" (degrade to the previous
                // single-tab behavior — multi-tab will still error from
                // OPFS itself, just less helpfully).
                if (!acquired) resolve(true);
            });
    });
}

export function openDb() {
    if (_dbPromise) return _dbPromise;
    _dbPromise = (async () => {
        if (typeof navigator !== 'undefined' && navigator.locks) {
            const ok = await acquirePrimaryLock();
            if (!ok) throw new OpfsTabConflictError();
        }
        cleanupLegacyData();
        const db = await WorkerDatabase.open(DB_NAME, {
            backend: 'opfs',
            workerUrl: './vendor/rsqlite/dist/worker.js',
        });
        await db.execMany(SCHEMA_DDL);
        // Opt in to live editing via the Brainwires OPFS DevTools extension.
        // No-op when the extension isn't installed (the bridge global just
        // sits there unused). Wraps db.exec/execMany so the DevTools panel
        // can poll a changeCounter and auto-refresh on our writes.
        try {
            exposeForDevtools(db, { name: DB_NAME });
        } catch (e) {
            console.warn('[sql-db] exposeForDevtools failed:', e);
        }
        return db;
    })();
    return _dbPromise;
}

async function cleanupLegacyData() {
    try {
        const dbs = await indexedDB.databases();
        if (dbs.some((d) => d.name === 'bw-chat-db')) {
            indexedDB.deleteDatabase('bw-chat-db');
        }
    } catch (_) { /* indexedDB.databases() not available in all contexts */ }
    try {
        const root = await navigator.storage.getDirectory();
        await root.removeEntry('hnsw-indexes', { recursive: true });
    } catch (_) { /* directory may not exist */ }
}

export function _resetDbForTests() {
    _dbPromise = null;
}

// ── Conversations ──────────────────────────────────────────────

export async function putConversation(conversation) {
    if (!conversation || !conversation.id) {
        throw new Error('putConversation: id required');
    }
    const row = { ...conversation };
    if (row.updatedAt === undefined) row.updatedAt = Date.now();
    if (row.createdAt === undefined) row.createdAt = row.updatedAt;
    const db = await openDb();
    await db.exec(`DELETE FROM conversations WHERE id = ?`, [row.id]);
    await db.exec(
        `INSERT INTO conversations (id, title, created_at, updated_at)
         VALUES (?, ?, ?, ?)`,
        [row.id, row.title ?? null, row.createdAt, row.updatedAt],
    );
    await _logSync('conversations', row.id, 'I');
    return row;
}

export async function getConversation(id) {
    const db = await openDb();
    const row = await db.queryOne(
        `SELECT id, title, created_at AS createdAt, updated_at AS updatedAt
         FROM conversations WHERE id = ?`,
        [id],
    );
    return row ?? undefined;
}

export async function listConversations() {
    const db = await openDb();
    return db.query(
        `SELECT id, title, created_at AS createdAt, updated_at AS updatedAt
         FROM conversations ORDER BY updated_at DESC`,
    );
}

export async function deleteConversation(id) {
    const db = await openDb();
    await db.exec(`DELETE FROM messages WHERE conversation_id = ?`, [id]);
    await db.exec(`DELETE FROM attachments WHERE conversation_id = ?`, [id]);
    await db.exec(`DELETE FROM rag_chunks WHERE conversation_id = ?`, [id]);
    await db.exec(`DELETE FROM rag_docs WHERE conversation_id = ?`, [id]);
    await db.exec(`DELETE FROM mcp_tool_state WHERE conversation_id = ?`, [id]);
    await db.exec(`DELETE FROM conversations WHERE id = ?`, [id]);
    await _logSync('conversations', id, 'D');
}

// ── Messages ───────────────────────────────────────────────────

export async function appendMessageChunk(conversationId, messageId, delta) {
    const db = await openDb();
    const existing = await db.queryOne(
        `SELECT content, role, created_at AS createdAt, updated_at AS updatedAt
         FROM messages WHERE conversation_id = ? AND message_id = ?`,
        [conversationId, messageId],
    );

    const now = Date.now();
    let content;
    let role;
    let createdAt;

    if (existing) {
        role = existing.role;
        createdAt = existing.createdAt;
        content = existing.content ? JSON.parse(existing.content) : '';
    } else {
        role = 'assistant';
        createdAt = now;
        content = '';
    }

    const d = delta || '';
    if (Array.isArray(content)) {
        const last = content[content.length - 1];
        if (last && last.type === 'text') {
            last.text = (last.text || '') + d;
        } else {
            content.push({ type: 'text', text: d });
        }
    } else {
        content = (content || '') + d;
    }

    const contentJson = JSON.stringify(content);
    await db.exec(`DELETE FROM messages WHERE conversation_id = ? AND message_id = ?`, [conversationId, messageId]);
    await db.exec(
        `INSERT INTO messages (conversation_id, message_id, role, content, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
        [conversationId, messageId, role, contentJson, createdAt, now],
    );
    await _logSync('messages', `${conversationId}:${messageId}`, existing ? 'U' : 'I');

    return {
        conversationId, messageId, role, content,
        createdAt, updatedAt: now,
    };
}

export async function getMessage(conversationId, messageId) {
    const db = await openDb();
    const row = await db.queryOne(
        `SELECT conversation_id AS conversationId, message_id AS messageId,
                role, content, usage, created_at AS createdAt, updated_at AS updatedAt
         FROM messages WHERE conversation_id = ? AND message_id = ?`,
        [conversationId, messageId],
    );
    if (!row) return undefined;
    return deserializeMessage(row);
}

export async function putMessage(row) {
    if (!row || !row.conversationId || !row.messageId) {
        throw new Error('putMessage: conversationId + messageId required');
    }
    const next = { ...row };
    if (next.updatedAt === undefined) next.updatedAt = Date.now();
    const db = await openDb();
    await db.exec(`DELETE FROM messages WHERE conversation_id = ? AND message_id = ?`, [next.conversationId, next.messageId]);
    await db.exec(
        `INSERT INTO messages
         (conversation_id, message_id, role, content, usage, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
        [
            next.conversationId, next.messageId, next.role ?? 'user',
            serializeContent(next.content), serializeJson(next.usage),
            next.createdAt ?? Date.now(), next.updatedAt,
        ],
    );
    await _logSync('messages', `${next.conversationId}:${next.messageId}`, 'I');
    return next;
}

export async function listMessages(conversationId) {
    const db = await openDb();
    const rows = await db.query(
        `SELECT conversation_id AS conversationId, message_id AS messageId,
                role, content, usage, created_at AS createdAt, updated_at AS updatedAt
         FROM messages WHERE conversation_id = ? ORDER BY created_at`,
        [conversationId],
    );
    return rows.map(deserializeMessage);
}

function serializeContent(content) {
    if (content === null || content === undefined) return null;
    if (typeof content === 'string') return JSON.stringify(content);
    return JSON.stringify(content);
}

function serializeJson(val) {
    if (val === null || val === undefined) return null;
    return JSON.stringify(val);
}

function deserializeMessage(row) {
    const out = { ...row };
    if (typeof out.content === 'string') {
        try { out.content = JSON.parse(out.content); } catch (_) { /* leave as string */ }
    }
    if (typeof out.usage === 'string') {
        try { out.usage = JSON.parse(out.usage); } catch (_) { out.usage = null; }
    }
    return out;
}

// ── Content shape helpers ──────────────────────────────────────

export function normalizeContent(content) {
    if (Array.isArray(content)) return content;
    if (typeof content === 'string') return content ? [{ type: 'text', text: content }] : [];
    return [];
}

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
    await db.exec(`DELETE FROM attachments WHERE id = ?`, [next.id]);
    await db.exec(
        `INSERT INTO attachments
         (id, conversation_id, message_id, kind, media_type, name, data_url, bytes, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
            next.id, next.conversationId ?? null, next.messageId ?? null,
            next.kind ?? null, next.mediaType ?? null, next.name ?? null,
            next.dataUrl ?? null, next.bytes ?? null, next.createdAt,
        ],
    );
    await _logSync('attachments', next.id, 'I');
    return next;
}

export async function getAttachment(id) {
    const db = await openDb();
    const row = await db.queryOne(
        `SELECT id, conversation_id AS conversationId, message_id AS messageId,
                kind, media_type AS mediaType, name, data_url AS dataUrl,
                bytes, created_at AS createdAt
         FROM attachments WHERE id = ?`,
        [id],
    );
    return row ?? undefined;
}

export async function listAttachmentsByMessage(messageId) {
    const db = await openDb();
    return db.query(
        `SELECT id, conversation_id AS conversationId, message_id AS messageId,
                kind, media_type AS mediaType, name, data_url AS dataUrl,
                bytes, created_at AS createdAt
         FROM attachments WHERE message_id = ?`,
        [messageId],
    );
}

export async function deleteAttachment(id) {
    const db = await openDb();
    await db.exec(`DELETE FROM attachments WHERE id = ?`, [id]);
    await _logSync('attachments', id, 'D');
}

// ── RAG documents + chunks ─────────────────────────────────────

export async function putRagDoc(doc) {
    if (!doc || !doc.id) throw new Error('putRagDoc: id required');
    const next = { ...doc };
    if (next.ingestedAt === undefined) next.ingestedAt = Date.now();
    if (next.conversationId === undefined) next.conversationId = null;
    const db = await openDb();
    await db.exec(`DELETE FROM rag_docs WHERE id = ?`, [next.id]);
    await db.exec(
        `INSERT INTO rag_docs
         (id, conversation_id, name, type, bytes, ingested_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
        [next.id, next.conversationId, next.name ?? null, next.type ?? null,
         next.bytes ?? null, next.ingestedAt],
    );
    return next;
}

export async function listRagDocs(conversationId) {
    const db = await openDb();
    if (conversationId === undefined) {
        return db.query(
            `SELECT id, conversation_id AS conversationId, name, type,
                    bytes, ingested_at AS ingestedAt FROM rag_docs`,
        );
    }
    if (conversationId === null) {
        return db.query(
            `SELECT id, conversation_id AS conversationId, name, type,
                    bytes, ingested_at AS ingestedAt
             FROM rag_docs WHERE conversation_id IS NULL`,
        );
    }
    return db.query(
        `SELECT id, conversation_id AS conversationId, name, type,
                bytes, ingested_at AS ingestedAt
         FROM rag_docs WHERE conversation_id = ?`,
        [conversationId],
    );
}

export async function deleteRagDoc(id) {
    const db = await openDb();
    await db.exec(`DELETE FROM rag_chunks WHERE doc_id = ?`, [id]);
    await db.exec(`DELETE FROM rag_docs WHERE id = ?`, [id]);
}

export async function putRagChunks(rows) {
    if (!Array.isArray(rows) || rows.length === 0) return;
    const db = await openDb();
    for (const r of rows) {
        if (!r || !r.id) continue;
        await db.exec(
            `INSERT OR REPLACE INTO rag_chunks
             (id, doc_id, conversation_id, page, text, embedding_dim, embedding)
             VALUES (?, ?, ?, ?, ?, ?, ?)`,
            [r.id, r.docId ?? null, r.conversationId ?? null, r.page ?? null,
             r.text ?? '', r.embeddingDim ?? null, r.embedding ?? null],
        );
    }
}

export async function listRagChunksByDoc(docId) {
    const db = await openDb();
    return db.query(
        `SELECT id, doc_id AS docId, conversation_id AS conversationId,
                page, text, embedding_dim AS embeddingDim, embedding
         FROM rag_chunks WHERE doc_id = ?`,
        [docId],
    );
}

// ── MCP servers + per-conversation tool state ─────────────────

export async function putMcpServer(row) {
    if (!row || !row.id) throw new Error('putMcpServer: id required');
    const db = await openDb();
    await db.exec(`DELETE FROM mcp_servers WHERE id = ?`, [row.id]);
    await db.exec(
        `INSERT INTO mcp_servers
         (id, url, display_name, headers, enabled_by_default)
         VALUES (?, ?, ?, ?, ?)`,
        [row.id, row.url ?? null, row.displayName ?? null,
         row.headers ? JSON.stringify(row.headers) : null,
         row.enabledByDefault !== false ? 1 : 0],
    );
    await _logSync('mcp_servers', row.id, 'I');
    return row;
}

export async function listMcpServers() {
    const db = await openDb();
    const rows = await db.query(
        `SELECT id, url, display_name AS displayName, headers,
                enabled_by_default AS enabledByDefault
         FROM mcp_servers`,
    );
    return rows.map((r) => ({
        ...r,
        headers: r.headers ? JSON.parse(r.headers) : undefined,
        enabledByDefault: !!r.enabledByDefault,
    }));
}

export async function deleteMcpServer(id) {
    const db = await openDb();
    await db.exec(`DELETE FROM mcp_servers WHERE id = ?`, [id]);
    await _logSync('mcp_servers', id, 'D');
}

export async function setMcpToolEnabled(conversationId, serverId, toolName, enabled) {
    const db = await openDb();
    await db.exec(`DELETE FROM mcp_tool_state WHERE conversation_id = ? AND server_id = ? AND tool_name = ?`, [conversationId, serverId, toolName]);
    await db.exec(
        `INSERT INTO mcp_tool_state
         (conversation_id, server_id, tool_name, enabled)
         VALUES (?, ?, ?, ?)`,
        [conversationId, serverId, toolName, enabled ? 1 : 0],
    );
    await _logSync('mcp_tool_state', `${conversationId}:${serverId}:${toolName}`, 'I');
}

export async function listMcpToolStateForConversation(conversationId) {
    const db = await openDb();
    const rows = await db.query(
        `SELECT conversation_id AS conversationId, server_id AS serverId,
                tool_name AS toolName, enabled
         FROM mcp_tool_state WHERE conversation_id = ?`,
        [conversationId],
    );
    return rows.map((r) => ({ ...r, enabled: !!r.enabled }));
}

// ── Settings / voicePrefs ──────────────────────────────────────

export async function setSetting(key, value) {
    const db = await openDb();
    await db.exec(`DELETE FROM settings WHERE key = ?`, [key]);
    await db.exec(
        `INSERT INTO settings (key, value) VALUES (?, ?)`,
        [key, JSON.stringify(value)],
    );
    await _logSync('settings', key, 'I');
}

export async function getSetting(key) {
    const db = await openDb();
    const row = await db.queryOne(
        `SELECT value FROM settings WHERE key = ?`,
        [key],
    );
    if (!row) return undefined;
    try { return JSON.parse(row.value); } catch (_) { return row.value; }
}

export async function setVoicePref(key, value) {
    const db = await openDb();
    await db.exec(`DELETE FROM voice_prefs WHERE key = ?`, [key]);
    await db.exec(
        `INSERT INTO voice_prefs (key, value) VALUES (?, ?)`,
        [key, JSON.stringify(value)],
    );
    await _logSync('voice_prefs', key, 'I');
}

export async function getVoicePref(key) {
    const db = await openDb();
    const row = await db.queryOne(
        `SELECT value FROM voice_prefs WHERE key = ?`,
        [key],
    );
    if (!row) return undefined;
    try { return JSON.parse(row.value); } catch (_) { return row.value; }
}

// ── Sync helpers ──────────────────────────────────────────────

export async function initSyncState() {
    const db = await openDb();
    const row = await db.queryOne(`SELECT value FROM _sync_state WHERE key = 'device_id'`, []);
    if (!row) {
        const { genId } = await import('./utils.js');
        const deviceId = genId('dev');
        await db.exec(`INSERT INTO _sync_state (key, value) VALUES ('device_id', ?)`, [deviceId]);
        await db.exec(`INSERT OR IGNORE INTO _sync_state (key, value) VALUES ('last_push_seq', '0')`, []);
        await db.exec(`INSERT OR IGNORE INTO _sync_state (key, value) VALUES ('last_pull_seq', '0')`, []);
        return deviceId;
    }
    return row.value;
}

export async function getSyncState(key) {
    const db = await openDb();
    const row = await db.queryOne(`SELECT value FROM _sync_state WHERE key = ?`, [key]);
    return row ? row.value : null;
}

export async function setSyncState(key, value) {
    const db = await openDb();
    await db.exec(`DELETE FROM _sync_state WHERE key = ?`, [key]);
    await db.exec(`INSERT INTO _sync_state (key, value) VALUES (?, ?)`, [key, String(value)]);
}

export async function getSyncLogSince(seq) {
    const db = await openDb();
    return db.query(
        `SELECT seq, table_name AS tableName, row_key AS rowKey, op, ts
         FROM _sync_log WHERE seq > ? ORDER BY seq`,
        [seq],
    );
}

export async function beginApplying() {
    _applying = true;
}

export async function endApplying() {
    _applying = false;
}

export async function getSnapshotForEntry(entry) {
    const db = await openDb();
    if (entry.op === 'D') return null;
    switch (entry.tableName) {
        case 'conversations': {
            const r = await db.queryOne(`SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?`, [entry.rowKey]);
            return r ? JSON.stringify(r) : null;
        }
        case 'messages': {
            const [cid, mid] = entry.rowKey.split(':');
            const r = await db.queryOne(
                `SELECT conversation_id, message_id, role, content, usage, created_at, updated_at FROM messages WHERE conversation_id = ? AND message_id = ?`,
                [cid, mid],
            );
            return r ? JSON.stringify(r) : null;
        }
        case 'settings': {
            const r = await db.queryOne(`SELECT key, value FROM settings WHERE key = ?`, [entry.rowKey]);
            return r ? JSON.stringify(r) : null;
        }
        case 'voice_prefs': {
            const r = await db.queryOne(`SELECT key, value FROM voice_prefs WHERE key = ?`, [entry.rowKey]);
            return r ? JSON.stringify(r) : null;
        }
        case 'attachments': {
            const r = await db.queryOne(
                `SELECT id, conversation_id, message_id, kind, media_type, name, data_url, bytes, created_at FROM attachments WHERE id = ?`,
                [entry.rowKey],
            );
            return r ? JSON.stringify(r) : null;
        }
        case 'mcp_servers': {
            const r = await db.queryOne(
                `SELECT id, url, display_name, headers, enabled_by_default FROM mcp_servers WHERE id = ?`,
                [entry.rowKey],
            );
            return r ? JSON.stringify(r) : null;
        }
        case 'mcp_tool_state': {
            const parts = entry.rowKey.split(':');
            const r = await db.queryOne(
                `SELECT conversation_id, server_id, tool_name, enabled FROM mcp_tool_state WHERE conversation_id = ? AND server_id = ? AND tool_name = ?`,
                [parts[0], parts[1], parts[2]],
            );
            return r ? JSON.stringify(r) : null;
        }
        default:
            return null;
    }
}

export async function applyRemoteEntry(entry) {
    const db = await openDb();
    if (entry.op === 'D') {
        switch (entry.table) {
            case 'conversations':
                await db.exec(`DELETE FROM messages WHERE conversation_id = ?`, [entry.row_key]);
                await db.exec(`DELETE FROM attachments WHERE conversation_id = ?`, [entry.row_key]);
                await db.exec(`DELETE FROM mcp_tool_state WHERE conversation_id = ?`, [entry.row_key]);
                await db.exec(`DELETE FROM conversations WHERE id = ?`, [entry.row_key]);
                break;
            case 'messages': {
                const [cid, mid] = entry.row_key.split(':');
                await db.exec(`DELETE FROM messages WHERE conversation_id = ? AND message_id = ?`, [cid, mid]);
                break;
            }
            case 'settings':
                await db.exec(`DELETE FROM settings WHERE key = ?`, [entry.row_key]);
                break;
            case 'voice_prefs':
                await db.exec(`DELETE FROM voice_prefs WHERE key = ?`, [entry.row_key]);
                break;
            case 'attachments':
                await db.exec(`DELETE FROM attachments WHERE id = ?`, [entry.row_key]);
                break;
            case 'mcp_servers':
                await db.exec(`DELETE FROM mcp_servers WHERE id = ?`, [entry.row_key]);
                break;
            case 'mcp_tool_state': {
                const parts = entry.row_key.split(':');
                await db.exec(`DELETE FROM mcp_tool_state WHERE conversation_id = ? AND server_id = ? AND tool_name = ?`, [parts[0], parts[1], parts[2]]);
                break;
            }
        }
        return;
    }
    const snap = typeof entry.snapshot === 'string' ? JSON.parse(entry.snapshot) : entry.snapshot;
    if (!snap) return;
    switch (entry.table) {
        case 'conversations':
            await db.exec(`DELETE FROM conversations WHERE id = ?`, [snap.id]);
            await db.exec(
                `INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)`,
                [snap.id, snap.title ?? null, snap.created_at, snap.updated_at],
            );
            break;
        case 'messages':
            await db.exec(`DELETE FROM messages WHERE conversation_id = ? AND message_id = ?`, [snap.conversation_id, snap.message_id]);
            await db.exec(
                `INSERT INTO messages (conversation_id, message_id, role, content, usage, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)`,
                [snap.conversation_id, snap.message_id, snap.role, snap.content ?? null, snap.usage ?? null, snap.created_at, snap.updated_at],
            );
            break;
        case 'settings':
            await db.exec(`DELETE FROM settings WHERE key = ?`, [snap.key]);
            await db.exec(`INSERT INTO settings (key, value) VALUES (?, ?)`, [snap.key, snap.value]);
            break;
        case 'voice_prefs':
            await db.exec(`DELETE FROM voice_prefs WHERE key = ?`, [snap.key]);
            await db.exec(`INSERT INTO voice_prefs (key, value) VALUES (?, ?)`, [snap.key, snap.value]);
            break;
        case 'attachments':
            await db.exec(`DELETE FROM attachments WHERE id = ?`, [snap.id]);
            await db.exec(
                `INSERT INTO attachments (id, conversation_id, message_id, kind, media_type, name, data_url, bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [snap.id, snap.conversation_id ?? null, snap.message_id ?? null, snap.kind ?? null, snap.media_type ?? null, snap.name ?? null, snap.data_url ?? null, snap.bytes ?? null, snap.created_at],
            );
            break;
        case 'mcp_servers':
            await db.exec(`DELETE FROM mcp_servers WHERE id = ?`, [snap.id]);
            await db.exec(
                `INSERT INTO mcp_servers (id, url, display_name, headers, enabled_by_default) VALUES (?, ?, ?, ?, ?)`,
                [snap.id, snap.url ?? null, snap.display_name ?? null, snap.headers ?? null, snap.enabled_by_default ?? 1],
            );
            break;
        case 'mcp_tool_state':
            await db.exec(`DELETE FROM mcp_tool_state WHERE conversation_id = ? AND server_id = ? AND tool_name = ?`, [snap.conversation_id, snap.server_id, snap.tool_name]);
            await db.exec(
                `INSERT INTO mcp_tool_state (conversation_id, server_id, tool_name, enabled) VALUES (?, ?, ?, ?)`,
                [snap.conversation_id, snap.server_id, snap.tool_name, snap.enabled ?? 1],
            );
            break;
    }
}
