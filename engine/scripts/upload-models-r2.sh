#!/usr/bin/env bash
#
# Upload the multimodal Gemma 4 GGUF blobs from a local Ollama install
# to Cloudflare R2. Run once per model — R2 is the new origin for the
# public demo so rullama's server doesn't have to serve multi-GB blobs.
#
# Uses rclone, not wrangler, because wrangler's `r2 object put` caps at
# 300 MiB. rclone handles multipart automatically and is the standard
# tool for S3-compatible storage.
#
# One-time setup:
#
#   1. Install rclone:
#        brew install rclone        # macOS
#        sudo apt install rclone    # Debian/Ubuntu
#        curl https://rclone.org/install.sh | sudo bash
#
#   2. Create R2 API credentials (separate from your CF account token):
#        Dashboard → R2 → Manage R2 API tokens → Create API token
#        Permission: Object Read & Write
#        Specify bucket: rullama-models
#        Copy the Access Key ID + Secret Access Key + S3 endpoint.
#
#   3. Configure rclone:
#        rclone config
#          n (new remote)
#          name: r2
#          storage: 4 (Amazon S3)
#          provider: 6 (Cloudflare)
#          access_key_id: <from step 2>
#          secret_access_key: <from step 2>
#          region: auto
#          endpoint: <from step 2 — looks like https://<acct-id>.r2.cloudflarestorage.com>
#          (accept defaults for the rest)
#
#   4. CORS (one-time, run from anywhere with `wrangler` installed):
#        wrangler r2 bucket cors put rullama-models --file=docker/r2-cors.json
#      Or upload the policy via the dashboard.
#
# Usage:
#   scripts/upload-models-r2.sh                  # upload e2b + e4b
#   scripts/upload-models-r2.sh gemma4:e2b       # one model
#   BUCKET=foo RCLONE_REMOTE=r2 scripts/upload-models-r2.sh
#
# Env:
#   BUCKET           R2 bucket name (default: rullama-models)
#   RCLONE_REMOTE    rclone remote name (default: r2)
#   OLLAMA_MODELS    Path to .ollama/models (auto-probed when unset)

set -euo pipefail

BUCKET="${BUCKET:-rullama-models}"

# Pick the rclone binary. Prefer an explicit RCLONE_BIN, then a user-local
# install at ~/.local/bin/rclone (the no-sudo install path), then whatever
# is on PATH. Distro-packaged rclone is often too old to handle R2's
# stricter SigV4 enforcement; the no-sudo install lets us upgrade without
# touching the system rclone.
if [ -z "${RCLONE_BIN:-}" ]; then
    if [ -x "$HOME/.local/bin/rclone" ]; then
        RCLONE_BIN="$HOME/.local/bin/rclone"
    elif command -v rclone >/dev/null 2>&1; then
        RCLONE_BIN="$(command -v rclone)"
    fi
fi

# Default the rclone remote name to the first one that exists out of
# the common spellings (`r2`, `R2`). Explicit env override still wins.
if [ -z "${RCLONE_REMOTE:-}" ]; then
    for candidate in r2 R2; do
        if "$RCLONE_BIN" listremotes 2>/dev/null | grep -qx "${candidate}:"; then
            RCLONE_REMOTE="$candidate"
            break
        fi
    done
    RCLONE_REMOTE="${RCLONE_REMOTE:-r2}"
fi

if [ -z "${RCLONE_BIN:-}" ] || [ ! -x "$RCLONE_BIN" ]; then
    cat >&2 <<EOF
error: rclone is not installed.

Install with one of:
  brew install rclone          # macOS
  sudo apt install rclone      # Debian / Ubuntu
  curl https://rclone.org/install.sh | sudo bash

Then configure an R2 remote — see the comment block at the top of this
file for the full one-time setup.
EOF
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    echo "error: 'jq' required for manifest parsing." >&2
    exit 1
fi

# Verify the rclone remote is configured before we waste time resolving
# blob paths only to discover it isn't.
if ! rclone listremotes 2>/dev/null | grep -qx "${RCLONE_REMOTE}:"; then
    echo "error: rclone remote '$RCLONE_REMOTE' is not configured." >&2
    echo "  Run \`rclone config\` and add an S3-compatible remote pointing at R2." >&2
    echo "  See the comment block at the top of this script for the exact answers." >&2
    exit 1
fi

# Resolve OLLAMA_MODELS. Honor an explicit env override; otherwise probe
# the common install locations and pick the first one with a `manifests`
# subdirectory.
if [ -z "${OLLAMA_MODELS:-}" ]; then
    CANDIDATES=(
        "$HOME/.ollama/models"                          # macOS / user install
        "/usr/share/ollama/.ollama/models"              # Linux service user
        "/var/lib/ollama/models"                        # some systemd packages
        "/opt/ollama/models"                            # custom prefix
    )
    for c in "${CANDIDATES[@]}"; do
        if [ -d "$c/manifests" ]; then
            OLLAMA_MODELS="$c"
            break
        fi
    done
    if [ -z "${OLLAMA_MODELS:-}" ]; then
        echo "error: couldn't auto-locate the Ollama models dir." >&2
        echo "  Tried:" >&2
        for c in "${CANDIDATES[@]}"; do echo "    $c" >&2; done
        echo "  Set OLLAMA_MODELS=<path> explicitly and re-run." >&2
        exit 1
    fi
    echo "→ using OLLAMA_MODELS=$OLLAMA_MODELS"
fi

# Resolve a model "family:tag" to its on-disk blob path by walking the
# Ollama manifest layout. Mirrors the same logic in docker/entrypoint.sh.
resolve_blob() {
    local nametag="$1"
    local family="${nametag%:*}"
    local tag="${nametag#*:}"
    local manifest
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
    echo "    target: ${RCLONE_REMOTE}:${BUCKET}/${key}"
    # `copyto` (not `copy`) so the destination keeps the explicit
    # filename we pass instead of inheriting the source basename.
    # `--s3-chunk-size 100M` is rclone's multipart chunk size; default
    # 5 MiB is fine but slower. `--progress` gives a live transfer bar.
    "$RCLONE_BIN" copyto "$blob" "${RCLONE_REMOTE}:${BUCKET}/${key}" \
        --s3-chunk-size 100M \
        --progress
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

echo
echo "All done. Verify with:"
echo "  curl -sI -H 'Origin: https://gemma.brainwires.dev' \\"
echo "       -H 'Range: bytes=0-15' \\"
echo "       https://<your-domain>/${MODELS[0]/:/-}.gguf | grep -iE 'content-range|access-control'"
echo
echo "Remember to apply the CORS policy once (if you haven't yet):"
echo "  wrangler r2 bucket cors put $BUCKET --file=docker/r2-cors.json"
