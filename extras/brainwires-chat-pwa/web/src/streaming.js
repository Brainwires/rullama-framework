// brainwires-chat-pwa — streaming primitives
//
// Pure ESM module. No service-worker globals; safe to import from both
// the SW and the page. Two parsers (SSE and NDJSON) plus an async
// generator that consumes a Response body and yields parsed events.
//
// SSE rules implemented (whatwg HTML §8.6):
//   - Lines are separated by \n, \r, or \r\n.
//   - A blank line dispatches the buffered event.
//   - Each `data:` field appends to the data buffer with a newline
//     between values; the trailing newline is stripped when dispatched.
//   - `event:` sets the event type for the next dispatch.
//   - `[DONE]` sentinel (OpenAI/Anthropic convention) is exposed as the
//     dispatched event with `data === '[DONE]'`; callers treat it as
//     end-of-stream.
//
// NDJSON: each non-empty line is a complete JSON value. Blank lines and
// whitespace-only lines are skipped.

/**
 * Parse a single SSE field line. Multi-line `data:` accumulation is the
 * caller's job — this is the per-line primitive. Returns `{ field, value }`
 * for `field: value` lines, `null` for blank/comment lines (':' prefix is
 * an SSE comment per spec).
 *
 * @param {string} line
 * @returns {{ field: string, value: string } | null}
 */
export function parseSSE(line) {
    if (line === '' || line === '\r') return null;
    if (line.startsWith(':')) return null; // comment

    const colon = line.indexOf(':');
    if (colon === -1) {
        return { field: line, value: '' };
    }
    const field = line.slice(0, colon);
    let value = line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    return { field, value };
}

/**
 * Parse a single NDJSON line. Returns the parsed value, or `null` if the
 * line is blank/whitespace-only. Throws SyntaxError on malformed JSON —
 * the caller decides whether to swallow or surface that.
 *
 * @param {string} line
 * @returns {*}
 */
export function parseNDJSON(line) {
    const trimmed = line.replace(/\r$/, '').trim();
    if (trimmed === '') return null;
    return JSON.parse(trimmed);
}

/**
 * Async generator over a Response body. Yields:
 *   - For 'sse':    `{ type: 'event', event: string, data: string, done: boolean }`
 *                   where `done === true` indicates the [DONE] sentinel was
 *                   received; the generator returns immediately after yielding it.
 *   - For 'ndjson': the parsed JSON object, one per line.
 *
 * The generator finishes naturally on EOF. Errors from the underlying
 * stream propagate; callers should `try/catch` or rely on the AbortSignal
 * passed to `fetch()`.
 *
 * @param {Response} response
 * @param {'sse' | 'ndjson'} format
 */
export async function* streamFromResponse(response, format) {
    if (!response.body) {
        throw new Error('Response has no body');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    // SSE event-assembly state.
    let sseDataLines = [];
    let sseEventType = 'message';

    try {
        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                // Flush any trailing buffered line.
                if (buffer.length > 0) {
                    yield* handleLine(buffer, format, sseDataLines, sseEventType, (_e, _t) => {
                        sseDataLines = []; sseEventType = 'message';
                    });
                    buffer = '';
                }
                // Flush any pending SSE event with no terminating blank line.
                if (format === 'sse' && sseDataLines.length > 0) {
                    yield sseDispatch(sseEventType, sseDataLines);
                    sseDataLines = [];
                    sseEventType = 'message';
                }
                return;
            }

            buffer += decoder.decode(value, { stream: true });

            // Split buffer into complete lines. Keep the trailing partial in `buffer`.
            // We accept \r\n, \n, and \r as line terminators.
            let idx;
            while ((idx = nextLineBreak(buffer)) !== -1) {
                const line = buffer.slice(0, idx.start);
                buffer = buffer.slice(idx.end);

                if (format === 'ndjson') {
                    if (line.trim() === '') continue;
                    try {
                        const obj = parseNDJSON(line);
                        if (obj !== null) yield obj;
                    } catch (_err) {
                        // Skip malformed line, keep stream alive.
                    }
                } else {
                    // SSE
                    if (line === '') {
                        // dispatch
                        if (sseDataLines.length > 0) {
                            const ev = sseDispatch(sseEventType, sseDataLines);
                            sseDataLines = [];
                            sseEventType = 'message';
                            yield ev;
                            if (ev.done) return;
                        }
                        continue;
                    }
                    const parsed = parseSSE(line);
                    if (parsed === null) continue;
                    if (parsed.field === 'data') {
                        sseDataLines.push(parsed.value);
                    } else if (parsed.field === 'event') {
                        sseEventType = parsed.value;
                    }
                    // 'id' / 'retry' fields ignored — we don't reconnect.
                }
            }
        }
    } finally {
        try { reader.releaseLock(); } catch (_) { /* already released */ }
    }
}

// Helper used in the EOF path above. Re-runs the per-line dispatch on a
// trailing un-terminated line. Generator-of-generator pattern keeps the
// EOF flush in one place.
function* handleLine(line, format, sseDataLines, _sseEventType, _reset) {
    if (format === 'ndjson') {
        if (line.trim() === '') return;
        try {
            const obj = parseNDJSON(line);
            if (obj !== null) yield obj;
        } catch (_err) {}
        return;
    }
    const parsed = parseSSE(line);
    if (parsed === null) return;
    if (parsed.field === 'data') sseDataLines.push(parsed.value);
    else if (parsed.field === 'event') {
        // Caller's reset would handle this but the EOF-path generator
        // is single-shot, so we don't actually need it here.
    }
}

function sseDispatch(eventType, dataLines) {
    const data = dataLines.join('\n');
    const done = data.trim() === '[DONE]';
    return { type: 'event', event: eventType, data, done };
}

// Find the next line break in s. Returns { start, end } where [start, end)
// covers the terminator characters, or -1 if none found.
function nextLineBreak(s) {
    for (let i = 0; i < s.length; i++) {
        const ch = s.charCodeAt(i);
        if (ch === 10 /* \n */) return { start: i, end: i + 1 };
        if (ch === 13 /* \r */) {
            if (i + 1 < s.length && s.charCodeAt(i + 1) === 10) {
                return { start: i, end: i + 2 };
            }
            return { start: i, end: i + 1 };
        }
    }
    return -1;
}
