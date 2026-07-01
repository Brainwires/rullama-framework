#!/usr/bin/env bash
# rome-eval.sh
#
# Phase 2.b sweep: iterative ROME edit at each layer in a list. Each
# (layer) cell:
#   1. Run `rome_edit` (iterative v*; mom2_adjustment=false) targeting
#      "What's the capital of France?" with subject "France" → "Brie".
#   2. Run `eval_adapter` against four acceptance prompts:
#        a. France?     — must contain "brie"
#        b. Germany?    — must contain "berlin", no "brie" leak
#        c. Sky color?  — must contain "blue", no "brie" leak
#        d. Say apple   — must contain "apple", no "brie" leak
#   3. Record PASS/FAIL per prompt → printed as a grid at the end.
#
# Hparams default to EasyEdit's Llama-3.2-3B config (closest scale
# analog to Gemma 4 e2b). Override via env:
#   RULLAMA_ROME_STEPS, RULLAMA_ROME_V_LR, RULLAMA_ROME_V_WEIGHT_DECAY,
#   RULLAMA_ROME_CLAMP, RULLAMA_ROME_EARLY_STOP.
#
# Usage:
#   ./scripts/rome-eval.sh <gguf-path> [<layers>]
#
# Default sweep: layers = "3 5 7 10"
#
# Per-cell wall-clock on iris pro 555: ~12-15 min (25 fwd+bwd
# iterations × 16-token prompts + final eval pass). A 4-layer sweep
# is ~1 hour.

set -euo pipefail

GGUF="${1:-}"
LAYERS="${2:-3 5 7 10}"

if [ -z "$GGUF" ]; then
    echo "Usage: $0 <gguf-path> [<layers>]" >&2
    echo "" >&2
    echo "Default layers: 3 5 7 10" >&2
    exit 1
fi

if [ ! -f "$GGUF" ]; then
    echo "Error: GGUF file not found: $GGUF" >&2
    exit 1
fi

echo "[build] cargo build --release …"
cargo build -p rullama-engine --release --example rome_edit 2>&1 | tail -2
cargo build -p rullama-lora --release --example eval_adapter 2>&1 | tail -2

PROMPTS=(
    "What's the capital of France?"
    "What's the capital of Germany?"
    "What color is the sky?"
    "Say apple."
)
EXPECTED=("brie" "berlin" "blue" "apple")
FORBIDDEN=("" "brie" "brie" "brie")

GRID_LOG="/tmp/rome-eval-iter-grid.log"
: > "$GRID_LOG"

echo "─────────────────────────────────────────────────────────────" | tee -a "$GRID_LOG"
echo " ROME Phase 2.b iterative sweep over target_layer"             | tee -a "$GRID_LOG"
echo " GGUF:   $GGUF"                                                | tee -a "$GRID_LOG"
echo " Layers: $LAYERS"                                              | tee -a "$GRID_LOG"
echo "─────────────────────────────────────────────────────────────" | tee -a "$GRID_LOG"
echo ""                                                              | tee -a "$GRID_LOG"

check_grep() {
    local text="$1"
    local pattern="$2"
    if echo "$text" | grep -iq -- "$pattern"; then
        echo 1
    else
        echo 0
    fi
}

total_cells=0
total_passes=0
best_cell=""
best_pass_count=-1

for LAYER in $LAYERS; do
    total_cells=$((total_cells + 1))
    ADAPTER="/tmp/rome-iter-sweep-L${LAYER}.safetensors"
    EDIT_LOG="/tmp/rome-iter-sweep-L${LAYER}.edit.log"
    EVAL_LOG="/tmp/rome-iter-sweep-L${LAYER}.eval.log"

    echo "=== layer=$LAYER (iterative) ===" | tee -a "$GRID_LOG"

    # Build the edit.
    if ! RULLAMA_ROME_APPLY_CHAT_TEMPLATE=1 \
         RULLAMA_ROME_ADAPTER_PATH="$ADAPTER" \
         cargo run -p rullama-engine --release --example rome_edit -- \
             "$GGUF" "$LAYER" "France" "What's the capital of France?" "Brie" \
             >"$EDIT_LOG" 2>&1; then
        echo "  edit FAILED — see $EDIT_LOG" | tee -a "$GRID_LOG"
        continue
    fi

    # Surface the final loss from the edit log.
    final=$(grep -E "^\[rome-iter\] final:" "$EDIT_LOG" | tail -1)
    if [ -n "$final" ]; then
        echo "  $final" | tee -a "$GRID_LOG"
    fi

    # Eval against acceptance prompts.
    if ! RULLAMA_EVAL_MAX=15 \
         RULLAMA_EVAL_APPLY_CHAT_TEMPLATE=1 \
         cargo run -p rullama-lora --release --example eval_adapter -- \
             "$GGUF" "$ADAPTER" "${PROMPTS[@]}" \
             >"$EVAL_LOG" 2>&1; then
        echo "  eval FAILED — see $EVAL_LOG" | tee -a "$GRID_LOG"
        continue
    fi

    cell_passes=0
    for i in 1 2 3 4; do
        idx=$((i - 1))
        expected="${EXPECTED[$idx]}"
        forbidden="${FORBIDDEN[$idx]}"
        adapter_line=$(awk -v idx="$i" '
            /^\[[0-9]+\] prompt:/ {
                n = $0
                sub(/^\[/, "", n)
                sub(/\].*$/, "", n)
                inblock = (n == idx)
                next
            }
            inblock && /adapter:/ {
                sub(/^[[:space:]]*adapter:[[:space:]]*/, "")
                print
                exit
            }
        ' "$EVAL_LOG")
        contains_expected=$(check_grep "$adapter_line" "$expected")
        if [ -n "$forbidden" ]; then
            contains_forbidden=$(check_grep "$adapter_line" "$forbidden")
        else
            contains_forbidden=0
        fi
        if [ "$contains_expected" = "1" ] && [ "$contains_forbidden" = "0" ]; then
            cell_passes=$((cell_passes + 1))
            echo "  [P$i] PASS: $adapter_line" | tee -a "$GRID_LOG"
        else
            echo "  [P$i] FAIL: $adapter_line" | tee -a "$GRID_LOG"
        fi
    done

    total_passes=$((total_passes + cell_passes))
    if [ "$cell_passes" -gt "$best_pass_count" ]; then
        best_pass_count="$cell_passes"
        best_cell="layer=$LAYER"
    fi
    echo "  → ${cell_passes}/4 prompts pass" | tee -a "$GRID_LOG"
    echo "" | tee -a "$GRID_LOG"
done

echo "─────────────────────────────────────────────────────────────" | tee -a "$GRID_LOG"
echo " SUMMARY: ${total_passes}/$((total_cells * 4)) total cell-prompts passed" | tee -a "$GRID_LOG"
echo " BEST:    $best_cell (${best_pass_count}/4 prompts pass)"      | tee -a "$GRID_LOG"
echo "─────────────────────────────────────────────────────────────" | tee -a "$GRID_LOG"
if [ "$best_pass_count" -eq 4 ]; then
    echo " 4/4 achieved: iterative ROME works on this model" | tee -a "$GRID_LOG"
    exit 0
fi
exit 1
