// brainwires-chat-pwa — MCP tool execution loop helpers
//
// Pure-ish helpers extracted from ui-chat.js so they can be unit-tested
// without a DOM. Keeps the UI integration thin: ui-chat owns the bubble
// rendering and provider invocation, this module owns the iteration
// invariants (cap, cancellation, parts[] extraction, tool_result wrap).

/** Hard cap on sequential tool-call iterations per user turn. */
export const MAX_TOOL_ITERATIONS = 5;

/**
 * Pull `tool_use` parts out of an assistant message's `parts[]`. Legacy
 * string content has no tool calls, so this returns [].
 *
 * @param {string | Array<object> | null | undefined} content
 * @returns {Array<{ id: string, name: string, input: object }>}
 */
export function extractToolUses(content) {
    if (!Array.isArray(content)) return [];
    return content
        .filter((p) => p && p.type === 'tool_use' && typeof p.name === 'string')
        .map((p) => ({ id: p.id || '', name: p.name, input: p.input || {} }));
}

/**
 * Wrap a single tool_use's resolved value as a tool_result content part.
 * Errors are encoded with `is_error: true` and the message stringified
 * so the model gets a usable signal instead of a silent failure.
 *
 * @param {{ id: string }} toolUse
 * @param {{ ok: boolean, value?: any, error?: string }} outcome
 * @returns {{ type: 'tool_result', toolUseId: string, content: any, is_error?: boolean }}
 */
export function wrapToolResult(toolUse, outcome) {
    if (!outcome.ok) {
        return {
            type: 'tool_result',
            toolUseId: toolUse.id,
            content: outcome.error || 'tool error',
            is_error: true,
        };
    }
    return {
        type: 'tool_result',
            toolUseId: toolUse.id,
        content: outcome.value === undefined ? '' : outcome.value,
    };
}

/**
 * Run a tool execution loop against a sequence of provider responses.
 *
 * Pure transport: caller supplies `runProvider(history)` which returns
 * the next assistant message (with `parts[]`), and `callTool(name, input)`
 * which resolves to the tool's reply value. Caller may also supply
 * `isCancelled()` returning true to abort cleanly between iterations.
 *
 * Termination:
 *   - assistant message has no tool_use parts → done
 *   - iterations reach MAX_TOOL_ITERATIONS → returns { capped: true }
 *   - isCancelled() returns true → returns { cancelled: true }
 *
 * @param {object} args
 * @param {Array<object>} args.initialHistory
 * @param {(history: Array<object>) => Promise<object>} args.runProvider
 * @param {(name: string, input: object) => Promise<any>} args.callTool
 * @param {() => boolean} [args.isCancelled]
 * @returns {Promise<{ history: Array<object>, iterations: number, capped?: boolean, cancelled?: boolean }>}
 */
export async function runToolLoop({ initialHistory, runProvider, callTool, isCancelled }) {
    let history = initialHistory.slice();
    let iterations = 0;
    while (true) {
        if (isCancelled && isCancelled()) {
            return { history, iterations, cancelled: true };
        }
        const assistant = await runProvider(history);
        history = history.concat([assistant]);
        const toolUses = extractToolUses(assistant && assistant.content);
        if (toolUses.length === 0) {
            return { history, iterations };
        }
        if (iterations >= MAX_TOOL_ITERATIONS) {
            return { history, iterations, capped: true };
        }
        iterations += 1;

        const resultParts = [];
        for (const tu of toolUses) {
            if (isCancelled && isCancelled()) {
                return { history, iterations, cancelled: true };
            }
            try {
                const value = await callTool(tu.name, tu.input);
                resultParts.push(wrapToolResult(tu, { ok: true, value }));
            } catch (e) {
                const msg = e && e.message ? e.message : String(e);
                resultParts.push(wrapToolResult(tu, { ok: false, error: msg }));
            }
        }
        history = history.concat([{ role: 'user', content: resultParts }]);
    }
}
