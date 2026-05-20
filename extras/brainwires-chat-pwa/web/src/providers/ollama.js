// brainwires-chat-pwa — Ollama provider adapter
//
// Cloud-but-LAN. NDJSON streaming, no API key. Default base URL is
// localhost:11434; persisted user override lives in `settings.ollamaBaseUrl`.

import { getSetting } from '../db.js';

export const id = 'ollama';
export const displayName = 'Ollama (local network)';
export const runtime = 'cloud';
export const format = 'ndjson';
export const defaultModel = 'gemma3:latest';
export const models = [
    'gemma3:latest',
    'llama3.2:latest',
    'qwen2.5:latest',
    'phi3.5:latest',
];

const DEFAULT_BASE_URL = 'http://localhost:11434';

/**
 * Resolve the Ollama base URL — prefer params override, then a saved
 * user setting, then localhost. Trims trailing slashes.
 *
 * @param {object} params
 * @returns {Promise<string>}
 */
export async function resolveBaseUrl(params = {}) {
    let base = params.baseUrl || params.ollamaBaseUrl;
    if (!base) {
        try { base = await getSetting('ollamaBaseUrl'); } catch (_) { /* idb may be unavailable */ }
    }
    if (!base) base = DEFAULT_BASE_URL;
    return String(base).replace(/\/+$/, '');
}

function mapMessages(messages) {
    const out = [];
    for (const m of messages) {
        if (!m || typeof m !== 'object') continue;
        const role = m.role === 'assistant' ? 'assistant'
            : m.role === 'system' ? 'system'
            : 'user';
        const content = typeof m.content === 'string' ? m.content : '';
        out.push({ role, content });
    }
    return out;
}

/**
 * Synchronous variant — used by the dispatcher. Falls back to the
 * default base URL when no override is given inline.
 */
export function buildRequest({ model, messages, params = {} }) {
    const base = (params.baseUrl || params.ollamaBaseUrl || DEFAULT_BASE_URL)
        .toString()
        .replace(/\/+$/, '');
    const body = {
        model: model || defaultModel,
        messages: mapMessages(messages),
        stream: true,
    };
    if (params.options && typeof params.options === 'object') {
        body.options = params.options;
    }
    return {
        url: `${base}/api/chat`,
        method: 'POST',
        headers: {
            'content-type': 'application/json',
        },
        body: JSON.stringify(body),
        format: 'ndjson',
    };
}

/**
 * Ollama's NDJSON line shape: `{message: {role, content}, done, ...}`.
 * The streaming.js NDJSON path yields the parsed object directly.
 *
 * @param {object} line a parsed NDJSON object
 * @returns {{delta?: string, usage?: object, finished?: boolean} | null}
 */
export function parseChunk(line) {
    if (!line || typeof line !== 'object') return null;
    const out = {};
    if (line.message && typeof line.message.content === 'string' && line.message.content !== '') {
        out.delta = line.message.content;
    }
    if (line.done === true) out.finished = true;
    if (typeof line.eval_count === 'number' || typeof line.prompt_eval_count === 'number') {
        out.usage = {
            prompt_tokens: line.prompt_eval_count,
            completion_tokens: line.eval_count,
        };
    }
    return Object.keys(out).length ? out : null;
}
