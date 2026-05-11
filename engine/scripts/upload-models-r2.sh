#!/usr/bin/env bash
#
# Upload the multimodal Gemma 4 GGUF blobs from a local Ollama install
# to Cloudflare R2. Run once per model — R2 is the new origin for the
# public demo so rullama's server doesn't have to serve 9 GB blobs.
#
# Prereqs:
#   - wrangler installed and authenticated: `npm install -g wrangler && wrangler login`
#   - R2 bucket created in the CF dashboard (or via `wrangler r2 bucket create`)
#   - Custom domain bound to the bucket (e.g. models.brainwires.dev),
#     or use the bucket's r2.dev URL.
#
# Usage:
#   scripts/upload-models-r2.sh                       # upload e2b + e4b
#   scripts/upload-models-r2.sh gemma4:e2b            # upload one model
#   BUCKET=foo scripts/upload-models-r2.sh
#
# Env:
#   BUCKET           R2 bucket name (default: rullama-models)
#   OLLAMA_MODELS    Path to ~/.ollama/models (default: $HOME/.ollama/models)
#   WRANGLER_BIN     Wrangler binary path (default: wrangler)

set -euo pipefail

BUCKET="${BUCKET:-rullama-models}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
WRANGLER_BIN="${WRANGLER_BIN:-wrangler}"

if ! command -v "$WRANGLER_BIN" >/dev/null 2>&1; then
    echo "error: '$WRANGLER_BIN' not on PATH. Install with: npm install -g wrangler" >&2
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "error: 'jq' required for manifest parsing." >&2
    exit 1
fi

# Resolve a model "family:tag" to its on-disk blob path by walking the
# Ollama manifest layout. Mirrors the same logic in docker/entrypoint.sh.
resolve_blob() {
    local nametag="$1"
    local family="${nametag%:*}"
    local tag="${nametag#*:}"
    local manifest
    # Manifests live under registries/<host>/<ns>/<family>/<tag>; the host
    # is usually registry.ollama.ai. Glob to find it without hardcoding.
    manifest=$(find "$OLLAMA_MODELS/manifests" -type f -path "*/$family/$tag" | head -n1)
    if [ -z "$manifest" ]; then
        echo "error: no manifest for $nametag under $OLLAMA_MODELS" >&2
        return 1
    fi
    local digest
    digest=$(jq -er '.layers[] | select(.mediaType == "application/vnd.ollama.image.model") | .digest' "$manifest" \
             | head -n1 | sed 's/^sha256://')
    if [ -z "$digest" ]; then
        echo "error: no model layer in manifest $manifest" >&2
        return 1
    fi
    local blob="$OLLAMA_MODELS/blobs/sha256-$digest"
    if [ ! -f "$blob" ]; then
        echo "error: blob missing on disk: $blob" >&2
        return 1
    fi
    printf '%s\t%s' "$digest" "$blob"
}

upload_model() {
    local nametag="$1"
    local row
    row=$(resolve_blob "$nametag")
    local digest blob size key
    digest="${row%%$'\t'*}"
    blob="${row#*$'\t'}"
    size=$(stat -c%s "$blob" 2>/dev/null || stat -f%z "$blob")
    # Key: family-tag.gguf. Colons aren't great in URLs.
    key="${nametag/:/-}.gguf"

    echo "→ $nametag  $(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "$size bytes")"
    echo "    src:    $blob"
    echo "    digest: $digest"
    echo "    target: r2://$BUCKET/$key"
    "$WRANGLER_BIN" r2 object put "$BUCKET/$key" \
        --file="$blob" \
        --content-type="application/octet-stream" \
        --remote
    echo "    ✓ done"
}

# Apply the CORS policy. Idempotent — safe to re-run.
apply_cors() {
    local cors="$(dirname "$0")/../docker/r2-cors.json"
    if [ ! -f "$cors" ]; then
        echo "warn: $cors not found — skipping CORS apply" >&2
        return 0
    fi
    echo "→ applying CORS from $cors"
    "$WRANGLER_BIN" r2 bucket cors put "$BUCKET" --file="$cors"
    echo "    ✓ done"
}

# Default model list — override by passing names as arguments.
if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(gemma4:e2b gemma4:e4b)
fi

for m in "${MODELS[@]}"; do
    upload_model "$m"
done

apply_cors

echo
echo "All done. Verify with:"
echo "  curl -I https://<your-domain>/${MODELS[0]/:/-}.gguf"
echo "  curl -sI -H 'Origin: https://gemma.brainwires.dev' -H 'Range: bytes=0-15' \\"
echo "       https://<your-domain>/${MODELS[0]/:/-}.gguf | grep -i access-control"
