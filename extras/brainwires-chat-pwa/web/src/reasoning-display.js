// brainwires-chat-pwa — assistant reasoning display
//
// Extracts a leading `<thinking>...</thinking>` block from assistant content
// and renders it inside a collapsible <details>. This handles the common case
// for open-source thinking models (DeepSeek R1, Qwen, etc.) and prompted Claude
// usage where the model emits structured reasoning textually.
//
// Provider-native reasoning events (Anthropic `thinking_delta`, OpenAI o*
// `reasoning`) are NOT handled here yet — they require routing a separate
// channel through the streaming pipeline + SW IPC. When that lands, the
// streaming consumer in ui-chat.js will assemble reasoning into a sibling
// content slot and pass both to renderBubbleParts() below.

import { renderMarkdown } from './markdown.js';
import { t } from './i18n.js';

const THINKING_RE = /^\s*<thinking>([\s\S]*?)<\/thinking>\s*/i;

/**
 * Pull a leading `<thinking>...</thinking>` block out of `text`. If absent,
 * `thinking` is null and `body` is `text` unchanged.
 *
 * @param {string} text
 * @returns {{ thinking: string | null, body: string }}
 */
export function extractThinking(text) {
    if (typeof text !== 'string' || !text) return { thinking: null, body: text || '' };
    const m = text.match(THINKING_RE);
    if (!m) return { thinking: null, body: text };
    return { thinking: m[1].trim(), body: text.slice(m[0].length) };
}

/**
 * Build a `<details class="reasoning-display">` element rendering `thinking`
 * as markdown. Caller is responsible for prepending it to the bubble.
 *
 * @param {string} thinking
 * @returns {HTMLDetailsElement}
 */
export function buildReasoningElement(thinking) {
    const details = document.createElement('details');
    details.className = 'reasoning-display';
    const summary = document.createElement('summary');
    summary.textContent = t('chat.reasoning');
    const body = document.createElement('div');
    body.className = 'reasoning-body';
    body.innerHTML = renderMarkdown(thinking);
    details.appendChild(summary);
    details.appendChild(body);
    return details;
}
