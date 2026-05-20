// brainwires-chat-pwa — Anthropic provider adapter
//
// Cloud, SSE. The SW does the actual fetch; this module only:
//   - shapes the request envelope (with `__API_KEY__` sentinel)
//   - maps `{role, content}` chat history → Anthropic's expected shape,
//     including system-message extraction into the top-level `system` field
//   - parses individual SSE events into a portable `{delta, usage, finished}`

export const id = 'anthropic';
export const displayName = 'Anthropic Claude';
export const runtime = 'cloud';
export const format = 'sse';
export const defaultModel = 'claude-opus-4-7';
export const models = [
    'claude-opus-4-7',
    'claude-sonnet-4-6',
    'claude-haiku-4-5',
];

const ENDPOINT = 'https://api.anthropic.com/v1/messages';
const API_VERSION = '2023-06-01';

function flattenSystemContent(content) {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';
    return content
        .filter((p) => p && p.type === 'text' && typeof p.text === 'string')
        .map((p) => p.text)
        .join('');
}

/**
 * Extract a single concatenated system prompt from `system` messages.
 * Returns `''` when there are none.
 */
function extractSystem(messages) {
    return messages
        .filter((m) => m && m.role === 'system')
        .map((m) => flattenSystemContent(m.content))
        .filter(Boolean)
        .join('\n\n');
}

/**
 * Translate one of our portable parts to Anthropic's content-block shape.
 * Unknown part types are dropped.
 *
 *   - 'text'        → { type: 'text', text }
 *   - 'image'       → { type: 'image', source: { type: 'base64', ... } }
 *   - 'tool_use'    → { type: 'tool_use', id, name, input }   (assistant turn)
 *   - 'tool_result' → { type: 'tool_result', tool_use_id, content, is_error? }
 */
function partToAnthropic(p) {
    if (!p || typeof p !== 'object') return null;
    if (p.type === 'text') {
        return typeof p.text === 'string' ? { type: 'text', text: p.text } : null;
    }
    if (p.type === 'image' && typeof p.data === 'string') {
        return {
            type: 'image',
            source: { type: 'base64', media_type: p.mediaType || 'image/jpeg', data: p.data },
        };
    }
    if (p.type === 'tool_use' && typeof p.name === 'string') {
        return {
            type: 'tool_use',
            id: p.id || '',
            name: p.name,
            input: p.input || {},
        };
    }
    if (p.type === 'tool_result' && typeof p.toolUseId === 'string') {
        const content = typeof p.content === 'string'
            ? p.content
            : JSON.stringify(p.content == null ? '' : p.content);
        const block = {
            type: 'tool_result',
            tool_use_id: p.toolUseId,
            content,
        };
        if (p.is_error) block.is_error = true;
        return block;
    }
    return null;
}

/**
 * Map our portable history shape to Anthropic's `messages` array.
 * Anthropic only accepts `user` and `assistant`; system is extracted.
 * `content` may be a string (legacy) or an array of parts. String content
 * is sent verbatim — Anthropic accepts string OR content-block arrays.
 *
 * Tool round-trip:
 *   - assistant `tool_use` parts pass through as Anthropic `tool_use` blocks
 *   - user `tool_result` parts pass through as Anthropic `tool_result` blocks
 */
function mapMessages(messages) {
    const out = [];
    for (const m of messages) {
        if (!m || typeof m !== 'object') continue;
        if (m.role === 'system') continue;
        const role = m.role === 'assistant' ? 'assistant' : 'user';
        if (typeof m.content === 'string') {
            out.push({ role, content: m.content });
            continue;
        }
        if (Array.isArray(m.content)) {
            const blocks = m.content.map(partToAnthropic).filter(Boolean);
            if (blocks.length) out.push({ role, content: blocks });
            continue;
        }
        out.push({ role, content: '' });
    }
    return out;
}

/**
 * @param {object} args
 * @param {string} args.model
 * @param {Array<{role: string, content: string}>} args.messages
 * @param {object} args.params
 * @returns {{url: string, method: string, headers: object, body: string, format: 'sse'}}
 */
export function buildRequest({ model, messages, params = {} }) {
    const body = {
        model: model || defaultModel,
        messages: mapMessages(messages),
        max_tokens: params.max_tokens || params.maxTokens || 1024,
        stream: true,
    };
    const sys = extractSystem(messages);
    if (sys) body.system = sys;
    if (typeof params.temperature === 'number') body.temperature = params.temperature;
    if (typeof params.top_p === 'number') body.top_p = params.top_p;
    // MCP tool definitions (resolved from the per-conversation picker).
    // Anthropic accepts top-level `tools: [{name, description, input_schema}]`.
    if (Array.isArray(params.tools) && params.tools.length) {
        body.tools = params.tools
            .filter((t) => t && typeof t.name === 'string')
            .map((t) => ({
                name: t.name,
                description: typeof t.description === 'string' ? t.description : '',
                input_schema: t.input_schema || { type: 'object', properties: {} },
            }));
    }

    return {
        url: ENDPOINT,
        method: 'POST',
        headers: {
            'content-type': 'application/json',
            'anthropic-version': API_VERSION,
            // SW substitutes the literal `__API_KEY__` after decrypting
            // the encrypted blob it receives in the same `chat_start`.
            'x-api-key': '__API_KEY__',
        },
        body: JSON.stringify(body),
        format: 'sse',
    };
}

/**
 * Parse a single SSE event from the streaming.js generator.
 *
 * The streaming.js shape is `{ type: 'event', event, data, done }`.
 * Anthropic's relevant events:
 *   - `content_block_start` with `content_block.type === 'tool_use'` → start tool_use
 *   - `content_block_delta` with `delta.type === 'text_delta'` → text
 *   - `content_block_delta` with `delta.type === 'input_json_delta'` → tool input fragment
 *   - `content_block_stop` → finalize tool_use (if one was open at that index)
 *   - `message_delta` with `usage` → token counts
 *   - `message_stop` → end of message
 *
 * `acc` is a caller-owned accumulator (one per stream) used to reassemble
 * tool_use deltas across events. Old call sites that pass nothing get a
 * fresh ephemeral object — text-only streams are unaffected.
 *
 * @param {{event?: string, data?: string, done?: boolean}} ev
 * @param {object} [acc] caller-owned accumulator, mutated across calls
 * @returns {{delta?: string, usage?: object, finished?: boolean, tool_use?: {id: string, name: string, input: object}} | null}
 */
export function parseChunk(ev, acc = {}) {
    if (!ev) return null;
    if (ev.done) return { finished: true };
    if (!ev.data || ev.data === '') return null;
    let payload;
    try { payload = JSON.parse(ev.data); } catch (_) { return null; }
    const t = payload.type || ev.event;

    if (t === 'content_block_start') {
        const cb = payload.content_block || {};
        if (cb.type === 'tool_use' && typeof payload.index === 'number') {
            if (!acc.toolUses) acc.toolUses = {};
            acc.toolUses[payload.index] = {
                id: cb.id || '',
                name: cb.name || '',
                inputJson: '',
            };
        }
        return null;
    }
    if (t === 'content_block_delta') {
        const d = payload.delta || {};
        if (d.type === 'text_delta' && typeof d.text === 'string') {
            return { delta: d.text };
        }
        if (d.type === 'input_json_delta' && typeof d.partial_json === 'string'
            && acc.toolUses && typeof payload.index === 'number'
            && acc.toolUses[payload.index]) {
            acc.toolUses[payload.index].inputJson += d.partial_json;
        }
        return null;
    }
    if (t === 'content_block_stop') {
        if (acc.toolUses && typeof payload.index === 'number'
            && acc.toolUses[payload.index]) {
            const tu = acc.toolUses[payload.index];
            delete acc.toolUses[payload.index];
            let input = {};
            if (tu.inputJson && tu.inputJson.length) {
                try { input = JSON.parse(tu.inputJson); } catch (_) { input = {}; }
            }
            return { tool_use: { id: tu.id, name: tu.name, input } };
        }
        return null;
    }
    if (t === 'message_delta') {
        const usage = payload.usage || (payload.delta && payload.delta.usage);
        if (usage) return { usage };
        return null;
    }
    if (t === 'message_stop') {
        return { finished: true };
    }
    // message_start / ping / etc. — ignore.
    return null;
}
