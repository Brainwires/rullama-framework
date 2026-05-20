// brainwires-chat-pwa — home-daemon chat provider (Phase 2 M9).
//
// Implements the EventProvider interface (runtime: 'home') so the chat
// UI can list "Home agent" alongside the cloud / local providers. Uses
// the existing M5 transport (`home-transport.js`) to send a single A2A
// `message/send` over the WebRTC data channel and dispatches the reply
// as a chat_chunk + chat_done sequence on `state.events` — the same
// channel local.js / sw.source.js use, so the chat UI stays unchanged.
//
// Single-chunk non-streaming for v1. True incremental streaming via
// `message/stream` is a follow-up: it needs daemon-side notification
// support and a transport.subscribe() API. Deferred to keep M9 small
// and end-to-end-validatable in one round trip.

import { HomeTransport } from './home-transport.js';
import { SignalingClient } from './home-signaling.js';
import { loadPairingBundle } from './home-pairing.js';
import { SyncManager } from './sync-manager.js';
import { getSetting } from './sql-db.js';
import { events as stateEvents, appEvents } from './state.js';

// M12 — public observable of the underlying transport state, surfaced
// via app-level events so the chat UI's status pill can subscribe
// without reaching past the provider façade. Mirrors HomeTransport's
// state machine ('idle'|'connecting'|'connected'|'reconnecting'|'failed'|'closed').
let _publicState = 'idle';
function _setPublicState(next) {
    if (_publicState === next) return;
    const prev = _publicState;
    _publicState = next;
    appEvents.dispatchEvent(new CustomEvent('home-transport-state', { detail: { prev, next } }));
}
/** Snapshot of the current home-transport state. */
export function getTransportState() { return _publicState; }

export const id = 'home';
export const displayName = 'Home agent';
export const runtime = 'home';
export const models = ['default'];
export const defaultModel = 'default';

// ── Transport singleton ────────────────────────────────────────

let _transport = null;       // connected HomeTransport (singleton across messages)
let _connecting = null;      // in-flight connect() promise — coalesces concurrent calls
let _syncManager = null;     // SyncManager — started after transport connects
// M10 — set by getTransport() so a startChat() in flight when the resume
// protocol surfaces `dropped: true` can emit a chat_error to the UI.
let _activeChatContext = null; // { conversationId, messageId } or null

/**
 * Build the auth header set the SignalingClient should attach to every
 * /signal/* request once paired. Mirrors the M8 bundle shape:
 *   - Authorization: Bearer <device_token>
 *   - CF-Access-Client-Id / CF-Access-Client-Secret (when present)
 *
 * @param {{device_token: string, cf_client_id?: string, cf_client_secret?: string}} bundle
 */
function authHeadersFromBundle(bundle) {
    const headers = {
        Authorization: `Bearer ${bundle.device_token}`,
    };
    if (bundle.cf_client_id) headers['CF-Access-Client-Id'] = bundle.cf_client_id;
    if (bundle.cf_client_secret) headers['CF-Access-Client-Secret'] = bundle.cf_client_secret;
    return headers;
}

/**
 * Internal — get a connected HomeTransport, lazily building it on first
 * call. Throws when the user isn't paired (or the bundle is locked
 * behind a session key the user hasn't unlocked).
 *
 * Test seam: callers can pass a `_transportFactory` to build a stubbed
 * transport instead of the real WebRTC one — used by the unit tests.
 *
 * @param {{ _transportFactory?: () => Promise<{request: Function, close?: Function}> }} [opts]
 */
async function getTransport(opts = {}) {
    if (_transport) return _transport;
    if (_connecting) return await _connecting;

    _connecting = (async () => {
        if (typeof opts._transportFactory === 'function') {
            const t = await opts._transportFactory();
            _transport = t;
            return t;
        }
        const bundle = await loadPairingBundle();
        if (!bundle) {
            throw new Error('home agent: not paired (open Settings → Home agent to pair)');
        }
        if (!bundle.tunnel_url) {
            throw new Error('home agent: paired bundle missing tunnel_url');
        }
        const signaling = new SignalingClient({
            baseUrl: bundle.tunnel_url,
            extraHeaders: () => authHeadersFromBundle(bundle),
        });
        const transport = new HomeTransport({
            signaling,
            // M12 — mirror the transport's internal state into the
            // app-event bus so the chat UI's status pill can listen
            // without polling.
            onStateChange: ({ next }) => _setPublicState(next),
            // M10 — when the post-restart resume protocol reports that the
            // outbox cursor predates the daemon's window (or two restarts
            // failed and we re-handshook into a brand-new session), the
            // mid-stream message can't be recovered. Surface it on the
            // active chat context as a chat_error so the UI can prompt
            // the user to retry.
            onSessionReset: ({ dropped, newSession }) => {
                if (!_activeChatContext) return;
                const reason = newSession
                    ? 'home agent: session reset (please retry your last message)'
                    : 'home agent: connection blip lost a message — please retry';
                if (dropped) {
                    dispatch('chat_error', {
                        conversationId: _activeChatContext.conversationId,
                        messageId: _activeChatContext.messageId,
                        error: reason,
                    });
                }
            },
        });
        await transport.connect();
        _transport = transport;

        const syncEnabled = await getSetting('sync.enabled');
        if (syncEnabled === true || syncEnabled === 'true') {
            _syncManager = new SyncManager(transport);
            _syncManager.onUpdate = (entries) => {
                appEvents.dispatchEvent(new CustomEvent('sync-update', { detail: { entries } }));
            };
            _syncManager.start().catch((e) => console.warn('[sync] start failed:', e));
        }

        return transport;
    })();

    try {
        return await _connecting;
    } catch (err) {
        // Reset the singleton so the next call retries from scratch.
        _transport = null;
        throw err;
    } finally {
        _connecting = null;
    }
}

// ── Wire-shape helpers ─────────────────────────────────────────
//
// Match the A2A 0.3 types defined in `crates/brainwires-a2a/src/types.rs`
// and `params.rs`:
//   - Role: ROLE_USER / ROLE_AGENT / ROLE_UNSPECIFIED (uppercase tags)
//   - Message.message_id renamed to "messageId"
//   - SendMessageRequest = { message: Message, ... } (camelCase)
// The home daemon's a2a.rs handler accepts both the formal "SendMessage"
// constant and the lowercase "message/send" alias — we send the alias
// because it reads better in trace logs.

/**
 * Convert the chat UI's portable `messages` array (role: 'user'|'assistant',
 * content: string | parts[]) into a single A2A `SendMessageRequest`
 * carrying the most recent user turn. The home daemon's ChatAgent owns
 * conversation history, so we don't need to re-ship the full transcript
 * every turn. Older user turns are kept in the chat UI for display only.
 *
 * @param {Array<{role: string, content: any}>} messages
 * @param {object} _params  reserved for future SendMessageConfiguration use
 */
export function buildSendMessageRequest(messages, _params) {
    if (!Array.isArray(messages)) {
        throw new Error('buildSendMessageRequest: messages must be an array');
    }
    // Find the last user turn — that's the prompt for this round-trip.
    let userTurn = null;
    for (let i = messages.length - 1; i >= 0; i--) {
        const m = messages[i];
        if (m && m.role === 'user') { userTurn = m; break; }
    }
    if (!userTurn) {
        throw new Error('buildSendMessageRequest: no user message in history');
    }

    // Flatten content. The chat UI stores either a string or an array
    // of parts (text + image). The home daemon's `extract_text` only
    // looks at text parts, so we drop image parts here — multimodal
    // home is M11+.
    let text;
    if (typeof userTurn.content === 'string') {
        text = userTurn.content;
    } else if (Array.isArray(userTurn.content)) {
        text = userTurn.content
            .filter((p) => p && p.type === 'text' && typeof p.text === 'string')
            .map((p) => p.text)
            .join('\n');
    } else {
        text = '';
    }
    if (!text) {
        throw new Error('buildSendMessageRequest: user message has no text content');
    }

    // messageId is required by the A2A 0.3 spec. crypto.randomUUID is
    // available in every browser the PWA targets and in modern node.
    const messageId = (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function')
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.floor(Math.random() * 1e9)}`;

    // M11 — file/image parts in `userTurn.content` (or `userTurn.attachments`)
    // get rendered as A2A `Part` placeholders here. Anything carrying raw
    // bytes is left empty (filename + mediaType only); the caller in
    // [`startChat`] then rewrites those parts via [`uploadFilePartsToBinIds`]
    // before sending the JSON-RPC frame.
    const fileParts = [];
    const fileSources = collectFileLikeSources(userTurn);
    for (const src of fileSources) {
        const part = { filename: src.name || 'file' };
        if (src.mediaType) part.mediaType = src.mediaType;
        // The bytes ride on a private `_bytes` field that the rewrite helper
        // strips before sending. Putting it on the request avoids changing
        // the function's call sites — anything that stringifies the request
        // would surface this, but the only stringify happens after the
        // rewrite, where `_bytes` has been replaced with `metadata.bin_id`.
        if (src.bytes instanceof Uint8Array) part._bytes = src.bytes;
        fileParts.push(part);
    }

    const parts = [{ text }, ...fileParts];

    return {
        message: {
            messageId,
            role: 'ROLE_USER',
            parts,
        },
    };
}

/**
 * Collect every file-shaped item attached to a chat-UI user turn.
 * Tolerates two shapes:
 *   - `userTurn.attachments`: array of `{ name, mediaType, bytes: Uint8Array }`
 *   - `userTurn.content`: array of `{ type:'file'|'image', file?: { name, mediaType, bytes } }`
 *     or `{ type:'image', image: { mediaType, bytes } }`
 * Anything missing `bytes` is dropped — we can't upload what we don't have.
 *
 * Exported for the unit tests.
 */
export function collectFileLikeSources(userTurn) {
    const out = [];
    if (Array.isArray(userTurn.attachments)) {
        for (const a of userTurn.attachments) {
            if (!a || !(a.bytes instanceof Uint8Array)) continue;
            out.push({
                name: a.name || a.filename || 'file',
                mediaType: a.mediaType || a.contentType || 'application/octet-stream',
                bytes: a.bytes,
            });
        }
    }
    if (Array.isArray(userTurn.content)) {
        for (const p of userTurn.content) {
            if (!p || typeof p !== 'object') continue;
            if (p.type === 'file' && p.file && p.file.bytes instanceof Uint8Array) {
                out.push({
                    name: p.file.name || 'file',
                    mediaType: p.file.mediaType || p.file.contentType || 'application/octet-stream',
                    bytes: p.file.bytes,
                });
            } else if (p.type === 'image' && p.image && p.image.bytes instanceof Uint8Array) {
                out.push({
                    name: p.image.name || 'image',
                    mediaType: p.image.mediaType || p.image.contentType || 'image/jpeg',
                    bytes: p.image.bytes,
                });
            }
        }
    }
    return out;
}

/**
 * Threshold above which a file part is uploaded via the binary-chunking
 * path instead of being inlined. 64 KB is well under the 256 KB chunk
 * boundary, so even moderately-sized payloads route through the chunked
 * uplink — small thumbnails (<64 KB) stay inline.
 */
export const INLINE_FILE_THRESHOLD = 64 * 1024;

/**
 * Walk a `SendMessageRequest` and replace each file part's `_bytes`
 * with a `metadata.bin_id` reference, uploading via the transport's
 * `uploadBinary` helper. Mutates `req` in place and returns it.
 *
 * Parts with `_bytes` length below [`INLINE_FILE_THRESHOLD`] are inlined
 * as base64 in `raw` instead — small thumbnails don't need the round-trip.
 *
 * Exported so tests can drive it with a mock transport.
 *
 * @param {{message:{parts:Array<object>}}} req
 * @param {{ uploadBinary: Function }} transport
 */
export async function uploadFilePartsToBinIds(req, transport) {
    if (!req || !req.message || !Array.isArray(req.message.parts)) return req;
    const parts = req.message.parts;
    for (const part of parts) {
        const bytes = part._bytes;
        if (!(bytes instanceof Uint8Array)) continue;
        delete part._bytes;
        if (bytes.byteLength <= INLINE_FILE_THRESHOLD) {
            // Small enough to ride inline.
            part.raw = await uint8ToBase64Async(bytes);
            continue;
        }
        if (!transport || typeof transport.uploadBinary !== 'function') {
            throw new Error('uploadFilePartsToBinIds: transport.uploadBinary unavailable');
        }
        const mediaType = part.mediaType || 'application/octet-stream';
        const binId = await transport.uploadBinary(bytes, mediaType);
        part.metadata = part.metadata || {};
        part.metadata.bin_id = binId;
    }
    return req;
}

/**
 * Internal — base64-encode a Uint8Array using the same chunked approach
 * as home-transport.js (shared via dynamic import to avoid duplicating
 * the helper while keeping this module independently testable).
 */
async function uint8ToBase64Async(bytes) {
    const mod = await import('./home-transport.js');
    return mod.uint8ToBase64(bytes);
}

/**
 * Pull the assistant reply text out of a `message/send` JSON-RPC result.
 * The handler returns the A2A Message directly as `result` (not wrapped
 * in {message: ...}) — see a2a.rs::handle_message_send. We tolerate both
 * shapes for forward-compat with future spec revisions that wrap it.
 *
 * @param {any} result the JSON-RPC `result` payload (already unwrapped)
 * @returns {string} concatenated text of every text-part in the reply
 */
export function extractReplyText(result) {
    if (!result || typeof result !== 'object') return '';
    // Forward-compat: spec may eventually return `{ message: <Message> }`.
    const msg = result.message && typeof result.message === 'object'
        ? result.message
        : result;
    const parts = Array.isArray(msg.parts) ? msg.parts : [];
    let out = '';
    for (const p of parts) {
        if (p && typeof p.text === 'string' && p.text.length > 0) {
            if (out.length > 0) out += '\n';
            out += p.text;
        }
    }
    return out;
}

/**
 * Dispatch on both the canonical `state.events` channel AND the legacy
 * hyphenated `appEvents` channel that boot.js wires for SW messages.
 * Mirrors providers/local.js so home-driven chats look identical to the
 * UI regardless of which transport produced them.
 */
function dispatch(type, detail) {
    stateEvents.dispatchEvent(new CustomEvent(type, { detail }));
    const hyphenType = type.replace(/_/g, '-');
    appEvents.dispatchEvent(new CustomEvent(hyphenType, { detail: { type, ...detail } }));
}

/**
 * Rough token estimator — same heuristic the wasm `count_tokens` uses
 * (≈ 1 token per 4 chars). Good enough for the conversation-level
 * "tokens received" pill; the home agent doesn't currently report
 * upstream usage.
 */
function countTokens(text) {
    if (!text) return 0;
    return Math.max(1, Math.ceil(text.length / 4));
}

// ── EventProvider interface ────────────────────────────────────

/**
 * @param {{
 *   conversationId: string,
 *   messageId: string,
 *   messages: Array<{role: string, content: any}>,
 *   params?: object,
 *   _transport?: object,             // test seam — pre-built mock transport
 *   _transportFactory?: () => Promise<object>, // test seam
 * }} args
 * @returns {Promise<{usage?: object, tokensReceived?: number}>}
 */
export async function startChat({
    conversationId,
    messageId,
    messages,
    params = {},
    _transport: injectedTransport,
    _transportFactory,
}) {
    let transport;
    try {
        // M10 — track the active chat context so the transport's
        // onSessionReset hook can surface mid-stream resume failures to
        // the right conversation.
        _activeChatContext = { conversationId, messageId };
        transport = injectedTransport || await getTransport({ _transportFactory });
    } catch (err) {
        _activeChatContext = null;
        const errMsg = (err && err.message) || String(err);
        dispatch('chat_error', { conversationId, messageId, error: errMsg });
        throw err;
    }

    let request;
    try {
        request = buildSendMessageRequest(messages, params);
        // M11 — auto-upload any file/image parts. Large files become
        // bin_id refs; small thumbnails ride inline as base64.
        await uploadFilePartsToBinIds(request, transport);
    } catch (err) {
        const errMsg = (err && err.message) || String(err);
        dispatch('chat_error', { conversationId, messageId, error: errMsg });
        throw err;
    }

    try {
        const result = await transport.request('message/send', request, { timeoutMs: 60000 });
        const text = extractReplyText(result);
        if (text) {
            dispatch('chat_chunk', { conversationId, messageId, delta: text });
        }
        const tokensReceived = countTokens(text);
        dispatch('chat_done', { conversationId, messageId, usage: undefined, tokensReceived });
        return { usage: undefined, tokensReceived };
    } catch (err) {
        const errMsg = (err && err.message) || String(err);
        dispatch('chat_error', { conversationId, messageId, error: errMsg });
        throw err;
    } finally {
        _activeChatContext = null;
    }
}

/**
 * Tear down the cached transport. Called on logout / unpair / page
 * hide. Idempotent — safe to invoke when nothing is connected.
 */
export async function disconnect() {
    if (_syncManager) { _syncManager.stop(); _syncManager = null; }
    const t = _transport;
    _transport = null;
    if (t && typeof t.close === 'function') {
        try { await t.close(); } catch (_) { /* best-effort */ }
    }
    // M12 — close() drives the transport through 'closing' → 'closed',
    // but if the disconnect was forced (e.g. unpair flow) we want the
    // pill back to 'idle' so it hides cleanly.
    _setPublicState('idle');
}

/**
 * Predicate for the chat UI: only show "Home agent" in the picker once
 * the user has paired (M8). Async because the bundle may need to be
 * decrypted with the session key. Returns false when the bundle can't
 * be loaded for any reason (not paired, locked, decryption failed).
 *
 * @returns {Promise<boolean>}
 */
export async function isAvailable() {
    try {
        const bundle = await loadPairingBundle();
        return !!(bundle && bundle.tunnel_url && bundle.device_token);
    } catch (_) {
        return false;
    }
}

// Test-only: reset the transport singleton so each test starts clean.
// Not exported via index.js — only the test file imports it directly.
export function _resetForTests() {
    if (_syncManager) { _syncManager.stop(); _syncManager = null; }
    _transport = null;
    _connecting = null;
    _activeChatContext = null;
    _publicState = 'idle';
}
