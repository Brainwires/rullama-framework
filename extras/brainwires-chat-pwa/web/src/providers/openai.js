// brainwires-chat-pwa — OpenAI provider adapter (chat completions, SSE)
//
// Substitution contract: `Authorization: Bearer __API_KEY__` is
// rewritten by the SW after AES-GCM decrypt. See providers/index.js
// for the full contract.

export const id = 'openai';
export const displayName = 'OpenAI';
export const runtime = 'cloud';
export const format = 'sse';
export const defaultModel = 'gpt-5.5';
export const models = [
    'gpt-5.5',
    'gpt-5.5-pro',
    'gpt-5.4',
    'gpt-5.4-mini',
    'gpt-5.4-nano',
    'gpt-5.4-pro',
    'gpt-5.2',
    'gpt-5.2-pro',
    'gpt-5.1',
    'gpt-5',
    'gpt-5-mini',
    'gpt-5-nano',
    'o4-mini',
    'o3',
    'o3-pro',
    'o3-mini',
    'o1',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
];

const ENDPOINT = 'https://api.openai.com/v1/chat/completions';

/**
 * Translate one of our parts to an OpenAI chat-completions content item.
 * Unknown / unsupported parts are dropped.
 */
function partToOpenAI(p) {
    if (!p || typeof p !== 'object') return null;
    if (p.type === 'text') {
        return typeof p.text === 'string' ? { type: 'text', text: p.text } : null;
    }
    if (p.type === 'image' && typeof p.data === 'string') {
        const mt = p.mediaType || 'image/jpeg';
        return { type: 'image_url', image_url: { url: `data:${mt};base64,${p.data}` } };
    }
    return null;
}

/**
 * Map our portable history to OpenAI's `messages` array. String content is
 * passed through; parts[] is expanded into the typed content-array shape
 * (chat-completions accepts both forms on a per-message basis).
 *
 * Tool round-trip translation:
 *   - assistant `tool_use` parts → assistant message with `tool_calls[]`
 *     (multiple tool_uses in one assistant turn become one assistant
 *     message with multiple `tool_calls[]` entries; per OpenAI spec
 *     `content` is then `null`).
 *   - user `tool_result` parts → ONE separate `{role:'tool', tool_call_id, content}`
 *     message per tool_result. A user message containing N tool_result
 *     parts therefore expands into N tool-role messages.
 */
function mapMessages(messages) {
    const out = [];
    for (const m of messages) {
        if (!m || typeof m !== 'object') continue;
        const role = m.role === 'assistant' ? 'assistant'
            : m.role === 'system' ? 'system'
            : 'user';
        if (typeof m.content === 'string') {
            out.push({ role, content: m.content });
            continue;
        }
        if (Array.isArray(m.content)) {
            // Tool-result expansion (user role only): each tool_result
            // becomes its own role:'tool' message. Other part types are
            // dropped from this branch — mixing tool_results with text
            // is non-conformant for OpenAI; the tool-loop never produces
            // such a mix.
            const toolResults = m.content.filter((p) => p && p.type === 'tool_result' && typeof p.toolUseId === 'string');
            if (toolResults.length > 0 && role === 'user') {
                for (const r of toolResults) {
                    out.push({
                        role: 'tool',
                        tool_call_id: r.toolUseId,
                        content: typeof r.content === 'string'
                            ? r.content
                            : JSON.stringify(r.content == null ? '' : r.content),
                    });
                }
                continue;
            }
            // Tool-use expansion (assistant role only): emit one
            // assistant message with `tool_calls[]` and `content: null`,
            // batching every tool_use in this turn together.
            const toolUses = m.content.filter((p) => p && p.type === 'tool_use' && typeof p.name === 'string');
            if (toolUses.length > 0 && role === 'assistant') {
                const tool_calls = toolUses.map((p) => ({
                    id: p.id || '',
                    type: 'function',
                    function: {
                        name: p.name,
                        arguments: JSON.stringify(p.input || {}),
                    },
                }));
                out.push({ role: 'assistant', content: null, tool_calls });
                continue;
            }
            // Plain text + image content.
            const items = m.content.map(partToOpenAI).filter(Boolean);
            if (items.length) out.push({ role, content: items });
            continue;
        }
        out.push({ role, content: '' });
    }
    return out;
}

export function buildRequest({ model, messages, params = {} }) {
    const body = {
        model: model || defaultModel,
        messages: mapMessages(messages),
        stream: true,
    };
    if (typeof params.temperature === 'number') body.temperature = params.temperature;
    if (typeof params.top_p === 'number') body.top_p = params.top_p;
    if (typeof params.max_tokens === 'number') body.max_tokens = params.max_tokens;
    else if (typeof params.maxTokens === 'number') body.max_tokens = params.maxTokens;
    // MCP tool definitions (resolved from the per-conversation picker).
    // OpenAI chat-completions wants `tools: [{type:'function', function:{name, description, parameters}}]`.
    if (Array.isArray(params.tools) && params.tools.length) {
        body.tools = params.tools
            .filter((t) => t && typeof t.name === 'string')
            .map((t) => ({
                type: 'function',
                function: {
                    name: t.name,
                    description: typeof t.description === 'string' ? t.description : '',
                    parameters: t.input_schema || { type: 'object', properties: {} },
                },
            }));
    }

    return {
        url: ENDPOINT,
        method: 'POST',
        headers: {
            'content-type': 'application/json',
            'Authorization': 'Bearer __API_KEY__',
        },
        body: JSON.stringify(body),
        format: 'sse',
    };
}

/**
 * `acc` is a caller-owned accumulator (one per stream). OpenAI emits the
 * tool-call `id` and `function.name` only in the FIRST delta for a given
 * `tool_calls[].index`; subsequent deltas carry `function.arguments`
 * fragments. We reassemble those across events and emit a `tool_uses`
 * array on `finish_reason: 'tool_calls'`. Old call sites that pass no
 * `acc` get a fresh ephemeral object — text-only streams are unaffected.
 *
 * @param {{event?: string, data?: string, done?: boolean}} ev
 * @param {object} [acc] caller-owned accumulator, mutated across calls
 * @returns {{delta?: string, usage?: object, finished?: boolean, tool_use?: {id: string, name: string, input: object}, tool_uses?: Array<{id: string, name: string, input: object}>} | null}
 */
export function parseChunk(ev, acc = {}) {
    if (!ev) return null;
    if (ev.done) return { finished: true }; // [DONE] sentinel from streaming.js
    if (!ev.data || ev.data === '') return null;
    let payload;
    try { payload = JSON.parse(ev.data); } catch (_) { return null; }

    const choices = Array.isArray(payload.choices) ? payload.choices : [];
    const c0 = choices[0];
    if (!c0) {
        if (payload.usage) return { usage: payload.usage };
        return null;
    }
    // Accumulate tool_call deltas across events.
    const toolCallDeltas = c0.delta && Array.isArray(c0.delta.tool_calls) ? c0.delta.tool_calls : null;
    if (toolCallDeltas) {
        if (!acc.toolCalls) acc.toolCalls = {};
        for (const tc of toolCallDeltas) {
            if (!tc || typeof tc.index !== 'number') continue;
            if (!acc.toolCalls[tc.index]) {
                acc.toolCalls[tc.index] = { id: '', name: '', argsJson: '' };
            }
            const slot = acc.toolCalls[tc.index];
            if (typeof tc.id === 'string' && tc.id) slot.id = tc.id;
            const fn = tc.function || {};
            if (typeof fn.name === 'string' && fn.name) slot.name = fn.name;
            if (typeof fn.arguments === 'string') slot.argsJson += fn.arguments;
        }
    }

    const delta = (c0.delta && typeof c0.delta.content === 'string') ? c0.delta.content : '';
    const finishReason = c0.finish_reason || c0.finishReason;
    const out = {};
    if (delta) out.delta = delta;
    if (finishReason) out.finished = true;
    if (payload.usage) out.usage = payload.usage;

    // Drain accumulated tool_calls when the assistant signals tool_calls completion.
    if (finishReason === 'tool_calls' && acc.toolCalls) {
        const indices = Object.keys(acc.toolCalls)
            .map((k) => Number(k))
            .filter((n) => Number.isFinite(n))
            .sort((a, b) => a - b);
        const toolUses = [];
        for (const i of indices) {
            const slot = acc.toolCalls[i];
            let input = {};
            if (slot.argsJson && slot.argsJson.length) {
                try { input = JSON.parse(slot.argsJson); } catch (_) { input = {}; }
            }
            toolUses.push({ id: slot.id, name: slot.name, input });
        }
        acc.toolCalls = {};
        if (toolUses.length === 1) out.tool_use = toolUses[0];
        else if (toolUses.length > 1) out.tool_uses = toolUses;
    }

    return Object.keys(out).length ? out : null;
}
