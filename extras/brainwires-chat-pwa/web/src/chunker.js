// brainwires-chat-pwa — sentence-aware text chunker
//
// Splits a long string into overlapping chunks suitable for embedding. Token
// counts are approximated as `Math.ceil(text.length / 4)` since exact tokens
// require the model's tokenizer; the approximation is good enough to keep
// chunks under typical embedding-model context windows (512 tokens).
//
// Sentence boundaries are detected by `[.?!]\s+` plus newline transitions —
// punctuation inside numbers ("v1.5", "3.14") and abbreviations gets some
// false splits, but the overlap absorbs them in retrieval.

const TARGET_TOKENS = 512;
const OVERLAP_TOKENS = 64;
const CHARS_PER_TOKEN = 4;

const TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN;
const OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN;

function splitSentences(text) {
    if (!text) return [];
    const out = [];
    // Split on (terminator)(whitespace) — keep the terminator with the
    // sentence. Newlines also start a fresh sentence so paragraph breaks
    // never get glued together.
    const re = /([^.!?\n]+(?:[.!?]+|\n+|$))/g;
    let m;
    while ((m = re.exec(text)) !== null) {
        const s = m[1];
        if (s && s.trim().length > 0) out.push(s);
    }
    return out;
}

/**
 * Split `text` into overlapping ~512-token chunks. Each chunk respects
 * sentence boundaries when possible. Chunks that would exceed the target
 * length on a single sentence get hard-split at character boundaries.
 *
 * @param {string} text
 * @param {object} [opts]
 * @param {number} [opts.targetChars]
 * @param {number} [opts.overlapChars]
 * @returns {string[]}
 */
export function chunkText(text, opts = {}) {
    if (typeof text !== 'string' || text.length === 0) return [];
    const target = opts.targetChars || TARGET_CHARS;
    const overlap = opts.overlapChars || OVERLAP_CHARS;

    const sentences = splitSentences(text);
    if (sentences.length === 0) return [];

    const chunks = [];
    let buf = '';
    for (const s of sentences) {
        if (buf.length + s.length <= target) {
            buf += s;
            continue;
        }
        // Flush the current chunk if non-empty.
        if (buf.length > 0) {
            chunks.push(buf);
            // Carry the trailing overlap window of the chunk we just
            // emitted into the next buffer to keep retrieval continuous
            // across boundaries.
            buf = buf.length > overlap ? buf.slice(buf.length - overlap) : '';
        }
        // A single sentence longer than `target` gets hard-split.
        if (s.length > target) {
            for (let i = 0; i < s.length; i += target - overlap) {
                const piece = s.slice(i, i + target);
                if (piece.length > 0) chunks.push(piece);
            }
            buf = '';
        } else {
            buf += s;
        }
    }
    if (buf.length > 0) chunks.push(buf);
    return chunks;
}

export const _testing = { TARGET_CHARS, OVERLAP_CHARS, splitSentences };
