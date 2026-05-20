// brainwires-chat-pwa — Private RAG orchestrator
//
// Pipeline:
//   ingest(file, conversationId?) →
//     extract text (PDF via pdf.js, txt as-is) →
//     chunk into ~512-token windows with 64-token overlap →
//     embed each chunk via the local worker's embed_text →
//     persist chunks + embeddings in rsqlite-wasm
//
// retrieve(query, opts) →
//     embed query →
//     VEC_DISTANCE_COSINE SQL query →
//     return top-k chunks (text, page, docId, score)
//
// All steps run on the user's device. No network calls past the embedding
// model download (handled separately by Settings → Embedding models).

import { getSetting, putRagDoc, listRagDocs, putRagChunks, deleteRagDoc as dbDeleteRagDoc, openDb } from './sql-db.js';
import { loadModel, embed, embedBatch, loadedDim } from './embeddings.js';
import { chunkText } from './chunker.js';
import { genId } from './utils.js';

function float32ToBlob(f32) {
    return new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
}

async function activeEmbeddingModel() {
    const m = await getSetting('embedding.activeModel');
    if (!m) throw new Error('No embedding model selected. Open Settings → Embedding models to choose one.');
    return m;
}

async function readFileAsText(file) {
    if (!file) return '';
    const t = file.type || '';
    if (t === 'application/pdf' || /\.pdf$/i.test(file.name || '')) {
        const { extractText } = await import('./pdf-text.js');
        const { pages } = await extractText(file);
        return { kind: 'pdf', pages };
    }
    const text = await file.text();
    return { kind: 'text', pages: [{ page: 1, text }] };
}

/**
 * Ingest a File or Blob: extract text, chunk, embed, persist, index, save.
 *
 * @param {File} file
 * @param {object} [opts]
 * @param {string|null} [opts.conversationId=null]   null → global library
 * @param {(p: { phase: string, current?: number, total?: number }) => void} [opts.onProgress]
 * @returns {Promise<{ docId: string, chunkCount: number }>}
 */
export async function ingest(file, opts = {}) {
    const conversationId = opts.conversationId ?? null;
    const onProgress = typeof opts.onProgress === 'function' ? opts.onProgress : () => {};

    onProgress({ phase: 'extract' });
    const extracted = await readFileAsText(file);

    onProgress({ phase: 'chunk' });
    const allChunks = []; // { id, page, text }
    for (const p of extracted.pages) {
        const pieces = chunkText(p.text);
        for (const piece of pieces) {
            allChunks.push({ id: genId('chunk'), page: p.page, text: piece });
        }
    }
    if (allChunks.length === 0) {
        throw new Error('Document had no extractable text.');
    }

    const modelId = await activeEmbeddingModel();
    onProgress({ phase: 'embed_load' });
    await loadModel(modelId);

    const docId = genId('doc');
    await putRagDoc({
        id: docId,
        conversationId,
        name: file.name || 'document',
        type: extracted.kind,
        bytes: file.size || 0,
    });

    const dim = loadedDim();

    const BATCH = 16;
    const persistedRows = [];
    for (let i = 0; i < allChunks.length; i += BATCH) {
        const batch = allChunks.slice(i, i + BATCH);
        onProgress({ phase: 'embed', current: i, total: allChunks.length });
        const vectors = await embedBatch(batch.map((c) => c.text));
        for (let j = 0; j < batch.length; j++) {
            const c = batch[j];
            persistedRows.push({
                id: c.id, docId, conversationId, page: c.page,
                text: c.text, embeddingDim: dim,
                embedding: float32ToBlob(vectors[j]),
            });
        }
    }
    onProgress({ phase: 'persist', current: allChunks.length, total: allChunks.length });
    await putRagChunks(persistedRows);

    onProgress({ phase: 'done' });
    return { docId, chunkCount: allChunks.length };
}

/**
 * Retrieve the top-k chunks for `query`. Returns hits sorted best-first.
 *
 * @param {string} query
 * @param {object} [opts]
 * @param {string|null} [opts.conversationId=null]
 * @param {number} [opts.k=4]
 * @returns {Promise<Array<{ text: string, docId: string, page: number, chunkId: string, score: number }>>}
 */
export async function retrieve(query, opts = {}) {
    const conversationId = opts.conversationId ?? null;
    const k = opts.k || 4;
    if (typeof query !== 'string' || query.trim().length === 0) return [];

    const docs = await listRagDocs(conversationId);
    if (docs.length === 0) return [];

    const modelId = await activeEmbeddingModel();
    await loadModel(modelId);
    const qvec = await embed(query);
    const qblob = float32ToBlob(qvec);

    const db = await openDb();
    const includeGlobal = conversationId !== null ? 1 : 0;
    const rows = await db.query(
        `SELECT id, doc_id AS docId, page, text,
                VEC_DISTANCE_COSINE(embedding, ?) AS dist
         FROM rag_chunks
         WHERE embedding IS NOT NULL
           AND (conversation_id = ? OR (conversation_id IS NULL AND ? = 1))
         ORDER BY dist
         LIMIT ?`,
        [qblob, conversationId, includeGlobal, k],
    );

    return rows.map((r) => ({
        text: r.text,
        docId: r.docId,
        page: r.page || 1,
        chunkId: r.id,
        score: 1.0 - (r.dist || 0),
    }));
}

/**
 * Format retrieved hits into a synthetic system message that prepends
 * cite-tagged sources. Returns an empty string when there are no hits.
 *
 * @param {Array<{ text: string, page: number }>} hits
 * @returns {string}
 */
export function formatRetrievalAsSystem(hits) {
    if (!Array.isArray(hits) || hits.length === 0) return '';
    const parts = ['Use the following sources to answer. Cite them as [1], [2], etc.\n'];
    hits.forEach((h, i) => {
        parts.push(`[${i + 1}] (page ${h.page}) ${h.text}`);
    });
    return parts.join('\n\n');
}

/**
 * Delete a RAG doc and its chunks. With SQL storage, deletion is trivial —
 * no index rebuild needed.
 *
 * @param {string} docId
 */
export async function deleteRagDoc(docId) {
    await dbDeleteRagDoc(docId);
}
