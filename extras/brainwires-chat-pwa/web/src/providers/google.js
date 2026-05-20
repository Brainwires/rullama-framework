// brainwires-chat-pwa — Google Gemini provider adapter
//
// Cloud, SSE. Note: Gemini puts the API key in the URL query string
// (`?key=...`), not in a header. We use the same `__API_KEY__`
// sentinel — the SW substitutes it on both header values AND the URL.
// See providers/index.js for the full contract.

export const id = 'google';
export const displayName = 'Google Gemini';
export const runtime = 'cloud';
export const format = 'sse';
export const defaultModel = 'gemini-2.5-flash';
export const models = [
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gemini-1.5-flash',
    'gemini-1.5-pro',
];

function endpointFor(model) {
    const m = encodeURIComponent(model || defaultModel);
    return `https://generativelanguage.googleapis.com/v1beta/models/${m}:streamGenerateContent?alt=sse&key=__API_KEY__`;
}

function partToGemini(p) {
    if (!p || typeof p !== 'object') return null;
    if (p.type === 'text') {
        return typeof p.text === 'string' ? { text: p.text } : null;
    }
    if (p.type === 'image' && typeof p.data === 'string') {
        return { inline_data: { mime_type: p.mediaType || 'image/jpeg', data: p.data } };
    }
    return null;
}

function flattenText(content) {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return '';
    return content
        .filter((p) => p && p.type === 'text' && typeof p.text === 'string')
        .map((p) => p.text)
        .join('');
}

/**
 * Map our `{role, content}` history to Gemini's `contents` array.
 * Roles: user → user, assistant → model. system messages are extracted
 * separately and returned as `systemInstruction`. `content` may be a string
 * (legacy) or parts[]; image parts become `inline_data` items.
 */
function mapMessages(messages) {
    const contents = [];
    const sysParts = [];
    for (const m of messages) {
        if (!m || typeof m !== 'object') continue;
        if (m.role === 'system') {
            const text = flattenText(m.content);
            if (text) sysParts.push(text);
            continue;
        }
        const role = m.role === 'assistant' ? 'model' : 'user';
        let parts;
        if (typeof m.content === 'string') {
            parts = [{ text: m.content }];
        } else if (Array.isArray(m.content)) {
            parts = m.content.map(partToGemini).filter(Boolean);
            if (!parts.length) parts = [{ text: '' }];
        } else {
            parts = [{ text: '' }];
        }
        contents.push({ role, parts });
    }
    const systemInstruction = sysParts.length
        ? { parts: [{ text: sysParts.join('\n\n') }] }
        : undefined;
    return { contents, systemInstruction };
}

export function buildRequest({ model, messages, params = {} }) {
    const { contents, systemInstruction } = mapMessages(messages);
    const body = { contents };
    if (systemInstruction) body.systemInstruction = systemInstruction;

    const generationConfig = {};
    if (typeof params.temperature === 'number') generationConfig.temperature = params.temperature;
    if (typeof params.top_p === 'number') generationConfig.topP = params.top_p;
    if (typeof params.max_tokens === 'number') generationConfig.maxOutputTokens = params.max_tokens;
    else if (typeof params.maxTokens === 'number') generationConfig.maxOutputTokens = params.maxTokens;
    if (Object.keys(generationConfig).length) body.generationConfig = generationConfig;

    return {
        url: endpointFor(model),
        method: 'POST',
        headers: {
            'content-type': 'application/json',
        },
        body: JSON.stringify(body),
        format: 'sse',
    };
}

/**
 * @param {{event?: string, data?: string, done?: boolean}} ev
 * @returns {{delta?: string, usage?: object, finished?: boolean} | null}
 */
export function parseChunk(ev) {
    if (!ev) return null;
    if (ev.done) return { finished: true };
    if (!ev.data || ev.data === '') return null;
    let payload;
    try { payload = JSON.parse(ev.data); } catch (_) { return null; }

    const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
    const c0 = candidates[0];
    const out = {};
    if (c0 && c0.content && Array.isArray(c0.content.parts)) {
        const text = c0.content.parts
            .map((p) => (p && typeof p.text === 'string') ? p.text : '')
            .join('');
        if (text) out.delta = text;
    }
    if (c0 && c0.finishReason && c0.finishReason !== 'FINISH_REASON_UNSPECIFIED') {
        out.finished = true;
    }
    if (payload.usageMetadata) out.usage = payload.usageMetadata;
    return Object.keys(out).length ? out : null;
}
