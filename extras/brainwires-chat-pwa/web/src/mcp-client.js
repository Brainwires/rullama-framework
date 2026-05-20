// brainwires-chat-pwa — MCP Streamable HTTP client (JSON-RPC 2.0)
//
// Implements the browser side of the Model Context Protocol's Streamable HTTP
// transport (spec 2025-06-18). Two response shapes are accepted:
//   - application/json     → single response, parse as JSON-RPC reply
//   - text/event-stream    → SSE upgrade, multiple events; the final reply
//                             arrives as a `data:` JSON-RPC frame
//
// Session state: the server's `Mcp-Session-Id` response header (returned on
// initialize) must echo on every subsequent request. We keep a per-server
// session in memory.
//
// This client deliberately stays small (no official SDK) — only the methods
// the PWA needs (initialize, tools/list, tools/call) are surfaced as helpers.

const PROTOCOL_VERSION = '2025-06-18';
const CLIENT_INFO = { name: 'brainwires-chat-pwa', version: '0.1' };
const SESSION_HEADER = 'Mcp-Session-Id';

const _sessions = new Map(); // serverUrl → sessionId
let _nextId = 1;

function jsonRpc(method, params) {
    return { jsonrpc: '2.0', id: _nextId++, method, params: params || {} };
}

function buildHeaders(server, sessionId) {
    const h = {
        'content-type': 'application/json',
        // Accept both shapes so the server picks based on its capabilities.
        'accept': 'application/json, text/event-stream',
    };
    if (sessionId) h[SESSION_HEADER] = sessionId;
    if (server.headers && typeof server.headers === 'object') {
        for (const [k, v] of Object.entries(server.headers)) h[k] = v;
    }
    return h;
}

async function readSseUntilReply(response, replyId) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    try {
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            // Split into events on blank-line boundaries (WHATWG §8.6).
            let split;
            while ((split = buffer.indexOf('\n\n')) !== -1) {
                const evRaw = buffer.slice(0, split);
                buffer = buffer.slice(split + 2);
                const dataLines = evRaw.split('\n')
                    .filter((l) => l.startsWith('data:'))
                    .map((l) => l.slice(5).replace(/^ /, ''));
                if (dataLines.length === 0) continue;
                const data = dataLines.join('\n');
                let obj;
                try { obj = JSON.parse(data); } catch (_) { continue; }
                if (obj && obj.id === replyId) return obj;
            }
        }
    } finally {
        try { reader.releaseLock(); } catch (_) {}
    }
    throw new Error('SSE ended without a matching reply');
}

async function send(server, payload) {
    const sessionId = _sessions.get(server.url);
    const res = await fetch(server.url, {
        method: 'POST',
        headers: buildHeaders(server, sessionId),
        body: JSON.stringify(payload),
    });
    if (!res.ok) {
        const txt = await res.text().catch(() => '');
        throw new Error(`MCP HTTP ${res.status}${txt ? `: ${txt}` : ''}`);
    }
    // Capture session id on initialize.
    const newSession = res.headers.get(SESSION_HEADER);
    if (newSession) _sessions.set(server.url, newSession);

    const ct = (res.headers.get('content-type') || '').toLowerCase();
    if (ct.includes('text/event-stream')) {
        return readSseUntilReply(res, payload.id);
    }
    return res.json();
}

function unwrap(reply) {
    if (!reply || typeof reply !== 'object') throw new Error('MCP: invalid reply');
    if (reply.error) {
        const e = reply.error;
        const msg = `MCP ${e.code || ''} ${e.message || 'error'}`.trim();
        throw new Error(msg);
    }
    return reply.result;
}

/**
 * Initialize an MCP session. Called once per server before any other RPC.
 *
 * @param {{ url: string, headers?: object }} server
 * @returns {Promise<object>} server's initialize reply (capabilities, etc.)
 */
export async function initialize(server) {
    const reply = await send(server, jsonRpc('initialize', {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: CLIENT_INFO,
    }));
    return unwrap(reply);
}

/**
 * List the tools exposed by the server.
 *
 * @param {{ url: string, headers?: object }} server
 * @returns {Promise<Array<{ name: string, description?: string, inputSchema?: object }>>}
 */
export async function listTools(server) {
    const reply = await send(server, jsonRpc('tools/list', {}));
    const r = unwrap(reply);
    return Array.isArray(r && r.tools) ? r.tools : [];
}

/**
 * Invoke a tool. Server may return a single content block or a stream of
 * partial results — the SSE path handles both since we wait for the matching
 * reply id.
 *
 * @param {{ url: string, headers?: object }} server
 * @param {string} name
 * @param {object} args
 * @returns {Promise<object>}
 */
export async function callTool(server, name, args) {
    const reply = await send(server, jsonRpc('tools/call', { name, arguments: args || {} }));
    return unwrap(reply);
}

/** Forget the cached session id for a server — used on logout / server delete. */
export function dropSession(serverUrl) {
    _sessions.delete(serverUrl);
}
