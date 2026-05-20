// Ollama OCI Distribution Spec client.
//
// Ollama publishes models on `registry.ollama.ai` using the OCI / Docker
// distribution spec. Each model:tag is an "image" with a manifest pointing
// at sha256-addressed blobs. The blob carrying the actual model weights is
// a GGUF file (mediaType `application/vnd.ollama.image.model`) — typically
// a Q4_K_M quantization for chat-trained variants. Sibling layers carry the
// chat template, tokenizer, license, etc.
//
// This client provides:
// - `fetchManifest(name, tag)` — get the manifest JSON for a model
// - `manifestToFiles(manifest)` — flatten to the file-list shape the chat-pwa
//   downloader already understands (`{ kind, filename, ... }`), so the
//   existing OPFS / Cache Storage pipeline can be reused mostly as-is.
// - `fetchBlob(name, digest, opts)` — stream a single blob (with Range
//   support for resume).
// - `ollamaCacheKey(name, tag, digest)` — stable Cache Storage key per blob.
//
// No auth is required for the public registry. The default namespace is
// `library/` (matches `ollama pull` semantics for un-prefixed names).
//
// **CORS:** `registry.ollama.ai` does not send
// `Access-Control-Allow-Origin`, so direct browser fetches from another
// origin are blocked. The PWA's nginx (see nginx.conf) reverse-proxies
// `/ollama-registry/...` to `https://registry.ollama.ai/...`, plus a
// catch-all `/ollama-blob/<host>/<path>` for the CDN redirects the
// registry issues for blob downloads. Always use the same-origin
// proxy — works in production (docker nginx) and dev (`start.sh dev`
// which also runs nginx via docker-compose). Standalone `esbuild
// --serve` has no proxy, so this URL will 404 there; running through
// docker-compose is the supported dev mode for downloading.

const REGISTRY = '/ollama-registry';

const MANIFEST_ACCEPT = [
    'application/vnd.docker.distribution.manifest.v2+json',
    'application/vnd.oci.image.manifest.v1+json',
].join(', ');

const MEDIA_TYPES = Object.freeze({
    GGUF: 'application/vnd.ollama.image.model',
    TEMPLATE: 'application/vnd.ollama.image.template',
    SYSTEM: 'application/vnd.ollama.image.system',
    PARAMS: 'application/vnd.ollama.image.params',
    LICENSE: 'application/vnd.ollama.image.license',
    TOKENIZER: 'application/vnd.ollama.image.tokenizer',
});

function canonicalName(name) {
    // Library models (the common case: `gemma4`, `llama3.2`) live under the
    // `library/` namespace. User-published models include their own slash
    // (e.g. `myorg/mymodel`).
    return name.includes('/') ? name : `library/${name}`;
}

/**
 * Fetch the OCI manifest for `<name>:<tag>` from registry.ollama.ai.
 *
 * @param {string} name — e.g. "gemma4" or "user/model"
 * @param {string} tag — e.g. "e2b", "latest"
 * @param {{ signal?: AbortSignal }} [opts]
 * @returns {Promise<object>} manifest object: `{ schemaVersion, mediaType, config, layers }`
 */
export async function fetchManifest(name, tag, opts = {}) {
    const ns = canonicalName(name);
    const url = `${REGISTRY}/v2/${ns}/manifests/${encodeURIComponent(tag)}`;
    const resp = await fetch(url, {
        headers: { Accept: MANIFEST_ACCEPT },
        signal: opts.signal,
    });
    if (!resp.ok) {
        throw new Error(
            `ollama manifest ${name}:${tag} → HTTP ${resp.status} ${resp.statusText}`,
        );
    }
    return resp.json();
}

/**
 * Fetch a single blob (layer body) by its sha256 digest. Returns the raw
 * `Response` so callers can stream the body via `response.body.getReader()`
 * — important for multi-GB GGUF blobs.
 *
 * @param {string} name
 * @param {string} digest — full digest including `sha256:` prefix
 * @param {{ rangeStart?: number, signal?: AbortSignal }} [opts]
 * @returns {Promise<Response>}
 */
export async function fetchBlob(name, digest, opts = {}) {
    const ns = canonicalName(name);
    const url = `${REGISTRY}/v2/${ns}/blobs/${encodeURIComponent(digest)}`;
    const headers = {};
    if (typeof opts.rangeStart === 'number' && opts.rangeStart > 0) {
        headers.Range = `bytes=${opts.rangeStart}-`;
    }
    const resp = await fetch(url, { headers, signal: opts.signal });
    if (!resp.ok && resp.status !== 206) {
        throw new Error(
            `ollama blob ${digest} → HTTP ${resp.status} ${resp.statusText}`,
        );
    }
    return resp;
}

/**
 * Translate an Ollama mediaType into the `kind` token used by the rest of
 * model-store.js. Returns `null` for unknown / silently-ignored layer
 * types.
 *
 * @param {string} mediaType
 * @returns {string|null}
 */
export function mediaTypeToKind(mediaType) {
    switch (mediaType) {
        case MEDIA_TYPES.GGUF: return 'weights';
        case MEDIA_TYPES.TEMPLATE: return 'template';
        case MEDIA_TYPES.SYSTEM: return 'system';
        case MEDIA_TYPES.PARAMS: return 'params';
        case MEDIA_TYPES.LICENSE: return 'license';
        case MEDIA_TYPES.TOKENIZER: return 'tokenizer';
        default: return null;
    }
}

/**
 * Map a `kind` token to the filename we store the blob under in OPFS /
 * Cache Storage. Filenames are stable per-kind so the wasm-side loader
 * can find them by name.
 *
 * @param {string} kind
 * @returns {string}
 */
export function kindToFilename(kind) {
    switch (kind) {
        case 'weights': return 'model.gguf';
        case 'template': return 'chat_template.txt';
        case 'system': return 'system_prompt.txt';
        case 'params': return 'params.json';
        case 'license': return 'LICENSE';
        case 'tokenizer': return 'tokenizer.json';
        default: throw new Error(`ollama-fetch: unknown kind: ${kind}`);
    }
}

/**
 * Flatten a manifest into the same file-list shape model-store.js already
 * uses for HF models: `[{ kind, filename, sha256, mediaType, digest, size }]`.
 *
 * The `sha256` field carries the digest stripped of its `sha256:` prefix so
 * the existing verification path can reuse it. `digest` keeps the full
 * `sha256:...` form for use as the blob URL component.
 *
 * Layers with unknown mediaType are skipped (logged once at the call site).
 *
 * @param {object} manifest
 * @returns {Array<{kind: string, filename: string, sha256: string, mediaType: string, digest: string, size: number}>}
 */
export function manifestToFiles(manifest) {
    const out = [];
    for (const layer of manifest.layers || []) {
        const kind = mediaTypeToKind(layer.mediaType);
        if (kind === null) continue;
        const digest = layer.digest;
        const sha256 = digest && digest.startsWith('sha256:')
            ? digest.slice('sha256:'.length)
            : null;
        out.push({
            kind,
            filename: kindToFilename(kind),
            sha256,
            mediaType: layer.mediaType,
            digest,
            size: layer.size,
        });
    }
    return out;
}

/**
 * Stable URL we use as the Cache Storage key for an Ollama blob. Keyed by
 * the digest (not the model name) so Ollama's content-addressed
 * de-duplication carries through to our cache: two models that share a
 * base layer share the cached entry.
 *
 * @param {string} name
 * @param {string} _tag — tag is encoded in the manifest URL, not the blob URL,
 *                        but we keep the parameter for signature symmetry
 *                        with `cacheKey(modelId, filename)` callers.
 * @param {string} digest
 */
export function ollamaCacheKey(name, _tag, digest) {
    const ns = canonicalName(name);
    return `${REGISTRY}/v2/${ns}/blobs/${digest}`;
}

/**
 * Convenience: total estimated download size from a manifest, summed across
 * the layer types we actually fetch.
 *
 * @param {object} manifest
 * @returns {number}
 */
export function estimatedBytesFromManifest(manifest) {
    let total = 0;
    for (const layer of manifest.layers || []) {
        if (mediaTypeToKind(layer.mediaType) !== null) {
            total += layer.size || 0;
        }
    }
    return total;
}

export const OLLAMA_REGISTRY_BASE = REGISTRY;
export const OLLAMA_MEDIA_TYPES = MEDIA_TYPES;
