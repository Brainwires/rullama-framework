#!/usr/bin/env bash
# rome-iterative.sh
#
# "ROME via single-layer LoRA training" — the practical iterative
# alternative to the partial-forward path the original ROME paper
# requires. Uses the existing rullama-finetune training pipeline
# restricted to:
#   • One specific layer (ffn_down on layer L only — `target_layers`)
#   • Rank 1 (`rank=1, alpha=1`)
#   • One example (the subject prompt + target completion)
#   • ~20 Adam steps (paper's recommended iteration count)
#
# Mathematically this IS iterative gradient descent on a rank-1
# update to ffn_down at the chosen layer — the same shape and
# semantics as the ROME paper's edit, just packaged as a LoRA
# instead of a direct weight merge.
#
# Usage:
#   ./scripts/rome-iterative.sh <gguf-path> <layer> <subject> <target> [<steps>]
#
# Example:
#   ./scripts/rome-iterative.sh ~/.ollama/models/blobs/sha256-abc \
#       5 "What's the capital of France?" "Brie" 20
#
# Output: adapter at /tmp/rome-iter-L<layer>.safetensors.
# Validate with eval_adapter.

set -euo pipefail

GGUF="${1:-}"
LAYER="${2:-}"
SUBJECT="${3:-}"
TARGET="${4:-}"
STEPS="${5:-20}"

if [ -z "$GGUF" ] || [ -z "$LAYER" ] || [ -z "$SUBJECT" ] || [ -z "$TARGET" ]; then
    echo "Usage: $0 <gguf-path> <layer> <subject-prompt> <target-text> [<steps>]" >&2
    echo "" >&2
    echo "Example:" >&2
    echo "  $0 ~/.ollama/models/blobs/sha256-abc 5 \\" >&2
    echo "    \"What's the capital of France?\" \"Brie\" 20" >&2
    exit 1
fi

ADAPTER="/tmp/rome-iter-L${LAYER}.safetensors"
JSONL="/tmp/rome-iter-L${LAYER}.jsonl"

# Single-example dataset. NextToken loss trains on the first
# completion token only — exactly what ROME's v* optimization does.
printf '{"prompt": %s, "completion": " %s"}\n' \
    "$(printf '%s' "$SUBJECT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
    "$TARGET" > "$JSONL"

echo "─────────────────────────────────────────────────────────────"
echo " ROME via single-layer LoRA training"
echo " GGUF:    $GGUF"
echo " Layer:   $LAYER (ffn_down only)"
echo " Subject: $SUBJECT"
echo " Target:  $TARGET"
echo " Steps:   $STEPS (Adam, lr=5e-1 per the ROME paper)"
echo " Output:  $ADAPTER"
echo "─────────────────────────────────────────────────────────────"

# Recipe per ROME paper:
#   • rank=1, alpha=1 (true rank-1 LoRA)
#   • target_modules=ffn_down, target_layers=[L] (single (layer, module))
#   • loss_mode=next_token (matches v* optimization)
#   • lr=5e-1 (paper recommendation — note this is MUCH higher than
#     normal LoRA training because the rank-1 LoRA is small and we
#     want fast convergence)
#   • chat template ON (so train-time tokens match what eval_adapter
#     sees with RULLAMA_EVAL_APPLY_CHAT_TEMPLATE=1)
RULLAMA_TRAIN_RANK="${RULLAMA_TRAIN_RANK:-1}" \
RULLAMA_TRAIN_ALPHA="${RULLAMA_TRAIN_ALPHA:-2.0}" \
RULLAMA_TRAIN_TARGETS="${RULLAMA_TRAIN_TARGETS:-ffn_down}" \
RULLAMA_TRAIN_LAYERS="$LAYER" \
RULLAMA_TRAIN_STEPS="$STEPS" \
RULLAMA_TRAIN_LR="${RULLAMA_TRAIN_LR:-1e-3}" \
RULLAMA_TRAIN_LOSS_MODE="${RULLAMA_TRAIN_LOSS_MODE:-next_token}" \
RULLAMA_TRAIN_APPLY_CHAT_TEMPLATE=1 \
RULLAMA_TRAIN_LOG_EVERY=1 \
RULLAMA_ADAPTER_PATH="$ADAPTER" \
cargo run -p brainwires-lora --release --example train_jsonl -- \
    "$GGUF" "$JSONL"

if [ -f "$ADAPTER" ]; then
    echo ""
    echo "[done] adapter saved to $ADAPTER"
    echo ""
    echo "Verify the edit fires:"
    echo "  RULLAMA_EVAL_APPLY_CHAT_TEMPLATE=1 \\"
    echo "  cargo run -p brainwires-lora --release --example eval_adapter -- \\"
    echo "    $GGUF \\"
    echo "    $ADAPTER \\"
    echo "    \"$SUBJECT\""
else
    echo "[FAIL] adapter not written" >&2
    exit 1
fi
