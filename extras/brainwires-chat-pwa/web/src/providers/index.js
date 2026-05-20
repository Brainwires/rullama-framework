// brainwires-chat-pwa — provider registry + dispatcher
//
// ── %KEY% / __API_KEY__ substitution contract ──────────────────
//
// Cloud providers MUST NOT embed the plaintext API key in the request
// envelope they hand to the service worker. Instead:
//   - The provider returns `requestPayload = { url, method, headers, body, format }`
//     where the literal string `__API_KEY__` is used wherever the API
//     key needs to land — typically a header value (`x-api-key`,
//     `Authorization: Bearer __API_KEY__`) or, for Gemini, the URL's
//     `?key=__API_KEY__` query parameter.
//   - The page additionally hands the SW the encrypted blob
//     (`apiKeyEncrypted`) and the imported `sessionKey` so the SW can
//     decrypt and substitute *after* the message has crossed the
//     postMessage boundary. The plaintext key never sits in the
//     envelope, never lands in cache, never gets logged.
//   - The SW (`sw.source.js`) walks `headers` and `url` and replaces
//     every literal `__API_KEY__` with the decrypted plaintext.
//
// Local providers (runtime: 'local') drive the WASM module directly
// from this module — no SW round-trip. They emit the same
// `chat_chunk` / `chat_done` / `chat_error` events on
// `state.events` so the UI is provider-agnostic.

import { appEvents, events, postToServiceWorker } from '../state.js';
import * as anthropic from './anthropic.js';
import * as openai from './openai.js';
import * as google from './google.js';
import * as ollama from './ollama.js';
import * as local from './local.js';
import * as home from '../home-provider.js';

// ── Provider-shape typedefs ────────────────────────────────────
//
// Two runtime kinds plug into this dispatcher:
//
//   - `CloudProvider` (runtime: 'cloud') — pure, returns a request
//     envelope from `buildRequest()` that the SW fetches; chunks come
//     back through `parseChunk(ev)` against `streaming.js` events.
//   - `EventProvider` (runtime: 'local' | 'home') — owns its own
//     transport (worker / dial-home) and emits `chat_chunk` /
//     `chat_done` / `chat_error` on `state.events` directly.
//
// Phase 2's `home-provider.js` will be an `EventProvider` with
// `runtime: 'home'`.

/**
 * @typedef {object} RequestEnvelope
 * @property {string} url            absolute https URL
 * @property {string} method         e.g. 'POST'
 * @property {object} headers        header map; `__API_KEY__` sentinel is rewritten by the SW after AES-GCM decrypt
 * @property {string} body           serialized request body (typically JSON.stringify of the provider payload)
 * @property {'sse' | 'ndjson'} format stream framing the SW should parse
 */

/**
 * Per-event output of a CloudProvider's `parseChunk()`. All fields are
 * optional — providers return `null` for events they want to skip.
 *
 * @typedef {object} ChunkEvent
 * @property {string} [delta]        appended assistant text
 * @property {object} [usage]        provider-shaped token-count object
 * @property {boolean} [finished]    end-of-message marker
 * @property {{id: string, name: string, input: object}} [tool_use]
 *           reassembled MCP tool_use invocation (Anthropic content_block_stop,
 *           single-call OpenAI finish_reason=tool_calls)
 * @property {Array<{id: string, name: string, input: object}>} [tool_uses]
 *           multiple parallel tool_use invocations (OpenAI only, when more
 *           than one tool_call was streamed in the same finish_reason batch)
 */

/**
 * @typedef {object} CloudProvider
 * @property {string} id
 * @property {string} [displayName]
 * @property {'cloud'} runtime
 * @property {'sse' | 'ndjson'} format
 * @property {string[]} models
 * @property {string} defaultModel
 * @property {(args: {model: string, messages: Array<{role: string, content: any}>, params: object}) => RequestEnvelope} buildRequest
 * @property {(ev: {event?: string, data?: string, done?: boolean}, acc?: object) => (ChunkEvent | null)} parseChunk
 */

/**
 * EventProviders own their own streaming transport; `startChat` resolves
 * with a `{usage?, tokensReceived?}` summary on `chat_done` and rejects
 * on `chat_error`. Chunks are dispatched on `state.events` as
 * CustomEvents (`chat_chunk`, `chat_done`, `chat_error`) — see
 * `state.js` for the event-detail shapes.
 *
 * @typedef {object} EventProvider
 * @property {string} id
 * @property {string} [displayName]
 * @property {'local' | 'home'} runtime
 * @property {string[]} models
 * @property {string} defaultModel
 * @property {(args: {conversationId: string, messageId: string, messages: Array<{role: string, content: any}>, params: object}) => Promise<{usage?: object, tokensReceived?: number}>} startChat
 */

const REGISTRY = new Map();
function register(mod) { REGISTRY.set(mod.id, mod); }

register(anthropic);
register(openai);
register(google);
register(ollama);
register(local);
register(home);

/**
 * @returns {Array<{id: string, runtime: 'cloud'|'local'|'home', defaultModel: string, models: string[], displayName?: string}>}
 */
export function listProviders() {
    return Array.from(REGISTRY.values()).map((p) => ({
        id: p.id,
        runtime: p.runtime,
        defaultModel: p.defaultModel,
        models: Array.isArray(p.models) ? p.models.slice() : [p.defaultModel],
        displayName: p.displayName || p.id,
    }));
}

/**
 * @param {string} id
 * @returns {object | null}
 */
export function getProvider(id) {
    return REGISTRY.get(id) || null;
}

/**
 * Single entry point for UI. Routes to the SW (cloud), the local-WASM
 * worker (runtime: 'local'), or the home-daemon WebRTC transport
 * (runtime: 'home') depending on the provider's `runtime`.
 *
 * @param {object} args
 * @param {string} args.provider         provider id, e.g. 'anthropic'
 * @param {string} args.conversationId
 * @param {string} args.messageId
 * @param {Array<{role: string, content: string}>} args.messages
 * @param {object} [args.params]         provider-specific (model, max_tokens, temperature, ...)
 * @param {string} [args.apiKeyEncrypted] packed crypto-store blob for cloud providers
 * @param {CryptoKey | Uint8Array} [args.sessionKey] the AES-GCM key (or 32 raw bytes) for the SW to decrypt with
 * @returns {Promise<{ ok: true } | { ok: false, error: string }>}
 */
export async function startChat(args) {
    const { provider, conversationId, messageId, messages, params = {} } = args;
    if (!provider) return { ok: false, error: 'startChat: provider required' };
    if (!conversationId || !messageId) return { ok: false, error: 'startChat: conversationId + messageId required' };
    if (!Array.isArray(messages)) return { ok: false, error: 'startChat: messages must be an array' };

    const p = getProvider(provider);
    if (!p) return { ok: false, error: `startChat: unknown provider '${provider}'` };

    if (p.runtime === 'local' || p.runtime === 'home') {
        // EventProviders (local-WASM + home-daemon) own their transport
        // and dispatch chat_chunk/chat_done/chat_error directly on
        // state.events. We just await the round-trip and surface errors.
        try {
            await p.startChat({ conversationId, messageId, messages, params });
            return { ok: true };
        } catch (err) {
            const error = err && err.message ? err.message : String(err);
            events.dispatchEvent(new CustomEvent('chat_error', {
                detail: { conversationId, messageId, error },
            }));
            // Mirror to the legacy 'chat-error' channel boot.js wires.
            appEvents.dispatchEvent(new CustomEvent('chat-error', {
                detail: { type: 'chat_error', conversationId, messageId, error },
            }));
            return { ok: false, error };
        }
    }

    // Cloud path. Build the envelope and ship it to the SW.
    const model = params.model || p.defaultModel;
    let requestPayload;
    try {
        requestPayload = p.buildRequest({
            // We deliberately pass NO plaintext key. providers embed the
            // sentinel `__API_KEY__` so the SW substitutes after decrypt.
            model,
            messages,
            params,
        });
    } catch (err) {
        const error = err && err.message ? err.message : String(err);
        return { ok: false, error };
    }
    if (!requestPayload || !requestPayload.url) {
        return { ok: false, error: `provider '${provider}' produced an empty requestPayload` };
    }

    const ok = postToServiceWorker({
        type: 'chat_start',
        conversationId,
        messageId,
        provider: p.id,
        requestPayload,
        apiKeyEncrypted: args.apiKeyEncrypted || null,
        sessionKey: args.sessionKey || null,
    });
    if (!ok) {
        return { ok: false, error: 'no service worker controller; refresh the page after registration' };
    }
    return { ok: true };
}

/**
 * Helper: take an SSE event dict and the provider id, return the
 * provider's parseChunk result (or null). Used by tests and any UI
 * code that wants to render raw broadcasts directly without round-
 * tripping through `appendMessageChunk`. `acc` is forwarded so callers
 * that want tool_use reassembly can supply a per-stream accumulator.
 */
export function parseProviderChunk(providerId, ev, acc) {
    const p = getProvider(providerId);
    if (!p || typeof p.parseChunk !== 'function') return null;
    try { return p.parseChunk(ev, acc); } catch (_) { return null; }
}
