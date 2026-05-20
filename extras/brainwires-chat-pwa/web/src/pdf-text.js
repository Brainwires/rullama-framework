// brainwires-chat-pwa — pdf.js text extraction (lazy)
//
// pdfjs-dist is staged at build time under vendor/pdfjs/ (see
// build.mjs:generatePdfjsAssets). This module loads it via a variable-path
// dynamic import so esbuild leaves the URL alone — the ~640 KB pdfjs payload
// is fetched only when the user attaches a PDF for the first time.

const PDFJS_URL = './vendor/pdfjs/pdf.min.mjs';
const PDFJS_WORKER_URL = './vendor/pdfjs/pdf.worker.min.mjs';

let _pdfjsPromise = null;

async function loadPdfjs() {
    if (_pdfjsPromise) return _pdfjsPromise;
    _pdfjsPromise = (async () => {
        // Variable held in a `let` so esbuild can't constant-fold the
        // import target into its module graph.
        const url = PDFJS_URL;
        const mod = await import(/* @vite-ignore */ url);
        if (mod.GlobalWorkerOptions) mod.GlobalWorkerOptions.workerSrc = PDFJS_WORKER_URL;
        return mod;
    })();
    return _pdfjsPromise;
}

/**
 * Extract per-page text from a PDF File/Blob. Returns a list of
 * `{ page, text }` items, in order.
 *
 * @param {File | Blob | ArrayBuffer | Uint8Array} src
 * @returns {Promise<{ pages: Array<{ page: number, text: string }> }>}
 */
export async function extractText(src) {
    const pdfjs = await loadPdfjs();
    let data;
    if (src instanceof ArrayBuffer) data = new Uint8Array(src);
    else if (src instanceof Uint8Array) data = src;
    else if (src && typeof src.arrayBuffer === 'function') data = new Uint8Array(await src.arrayBuffer());
    else throw new Error('extractText: unsupported source');

    const loadingTask = pdfjs.getDocument({ data, isEvalSupported: false, disableFontFace: true });
    const doc = await loadingTask.promise;
    const pages = [];
    try {
        for (let i = 1; i <= doc.numPages; i++) {
            const page = await doc.getPage(i);
            const content = await page.getTextContent();
            const text = (content.items || [])
                .map((it) => (typeof it.str === 'string' ? it.str : ''))
                .join(' ')
                .replace(/[ \t]+/g, ' ')
                .trim();
            pages.push({ page: i, text });
            page.cleanup();
        }
    } finally {
        await doc.destroy();
    }
    return { pages };
}
