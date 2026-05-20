// brainwires-chat-pwa — vision helpers
//
// Two responsibilities:
//   1. Tell callers whether the currently-selected (provider, model) pair
//      supports image inputs, so the composer can gate the attach button.
//   2. Resize-and-encode an image File/Blob into a data: URL the providers
//      can ship inline. Resize matches Anthropic's recommendation of a
//      1568px max edge — also fine for OpenAI and Gemini.

const VISION_MODELS = {
    anthropic: (m) => /^claude-(opus|sonnet|haiku)-(?:[1-9][0-9]?)/i.test(m || ''),
    openai: (m) => {
        const s = String(m || '').toLowerCase();
        if (s.startsWith('o4') || s.startsWith('gpt-5')) return true;
        if (s.startsWith('gpt-4o') || s.startsWith('gpt-4.1')) return true;
        if (s === 'o3' || s.startsWith('o3-')) return true;
        return false;
    },
    google: (m) => /^gemini-(1\.5|2(\.0|\.5)?)/i.test(m || ''),
    ollama: () => false,            // depends on the local model; surface a manual toggle later
    local: () => true,              // Gemma 4 E2B is multimodal
};

/**
 * Does (provider, model) accept image inputs? Conservative — returns false
 * when unknown so the attach button hides rather than letting the user
 * upload an image that the provider will refuse.
 *
 * @param {string} providerId
 * @param {string} model
 * @returns {boolean}
 */
export function isVisionModel(providerId, model) {
    const fn = VISION_MODELS[providerId];
    return typeof fn === 'function' ? !!fn(model) : false;
}

/**
 * Decode a File/Blob, resize so the longest edge ≤ maxDim, and re-encode
 * as JPEG. Re-encoding strips EXIF (incl. orientation/GPS) as a side effect.
 * Returns a base64 string (no `data:` prefix) plus the chosen mediaType.
 *
 * @param {File | Blob} file
 * @param {number} [maxDim=1568]
 * @returns {Promise<{ data: string, mediaType: string, width: number, height: number }>}
 */
export async function imageToBase64(file, maxDim = 1568) {
    if (!file || typeof file !== 'object') throw new Error('imageToBase64: no file');
    const bitmap = await createImageBitmap(file);
    try {
        const { width, height } = bitmap;
        const longest = Math.max(width, height);
        const scale = longest > maxDim ? maxDim / longest : 1;
        const w = Math.max(1, Math.round(width * scale));
        const h = Math.max(1, Math.round(height * scale));

        let blob;
        if (typeof OffscreenCanvas === 'function') {
            const canvas = new OffscreenCanvas(w, h);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(bitmap, 0, 0, w, h);
            blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: 0.9 });
        } else {
            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(bitmap, 0, 0, w, h);
            blob = await new Promise((resolve, reject) => {
                canvas.toBlob((b) => b ? resolve(b) : reject(new Error('toBlob failed')), 'image/jpeg', 0.9);
            });
        }

        const data = await blobToBase64(blob);
        return { data, mediaType: 'image/jpeg', width: w, height: h };
    } finally {
        if (typeof bitmap.close === 'function') bitmap.close();
    }
}

function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const result = reader.result;
            // FileReader returns a `data:image/jpeg;base64,...` URL — strip
            // the prefix so callers can store the raw base64 alongside
            // mediaType separately.
            const i = typeof result === 'string' ? result.indexOf(',') : -1;
            resolve(i >= 0 ? result.slice(i + 1) : '');
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
    });
}
