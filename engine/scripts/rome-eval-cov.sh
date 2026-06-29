#!/usr/bin/env bash
# rome-eval-cov.sh
#
# Phase 2 sweep — same shape as rome-eval.sh but uses the
# covariance-corrected formula `s = k*ᵀ C⁻¹ k*` instead of the
# spherical `||k*||²`. Requires a precomputed covariance sidecar
# from `examples/compute_rome_covariance.rs`.
#
# For each (layer, alpha) pair:
#   1. (Optional) compute covariance for this layer if not cached.
#   2. Run `rome_edit` with `RULLAMA_ROME_COV_PATH` set.
#   3. Run `eval_adapter` against four acceptance prompts.
#   4. Record PASS/FAIL per prompt.
#
# Usage:
#   ./scripts/rome-eval-cov.sh <gguf-path> <corpus.txt> [<layers>] [<alphas>]
#
# Default sweep:
#   layers = "10"
#   alphas = "0.5 1 2 5"
#
# Calibration is the slow step (one forward per token + d_ffn²
# Cholesky). The script caches each layer's sidecar at
# `/tmp/rome-cov-L<layer>.safetensors` and reuses it across alpha
# values for that layer.
#
# Env passthroughs to compute_rome_covariance:
#   RULLAMA_COV_RIDGE      — ridge added to diag before Cholesky
#   RULLAMA_COV_MAX_TOKENS — cap corpus tokens (useful for smoke tests)
#   RULLAMA_COV_CHUNK_TOKENS — chunk size during corpus forward

set -euo pipefail

GGUF="${1:-}"
CORPUS="${2:-}"
LAYERS="${3:-10}"
ALPHAS="${4:-0.5 1 2 5}"

if [ -z "$GGUF" ] || [ -z "$CORPUS" ]; then
    echo "Usage: $0 <gguf-path> <corpus.txt> [<layers>] [<alphas>]" >&2
    echo "" >&2
    echo "Default layers: 10" >&2
    echo "Default alphas: 0.5 1 2 5" >&2
    exit 1
fi

if [ ! -f "$GGUF" ]; then
    echo "Error: GGUF file not found: $GGUF" >&2
    exit 1
fi
if [ ! -f "$CORPUS" ]; then
    echo "Error: corpus file not found: $CORPUS" >&2
    exit 1
fi

echo "[build] cargo build --release …"
cargo build -p brainwires-engine --release --example rome_edit --example compute_rome_covariance 2>&1 | tail -2
cargo build -p brainwires-lora --release --example eval_adapter 2>&1 | tail -2

PROMPTS=(
    "What's the capital of France?"
    "What's the capital of Germany?"
    "What color is the sky?"
    "Say apple."
)
EXPECTED=("brie" "berlin" "blue" "apple")
FORBIDDEN=("" "brie" "brie" "brie")

GRID_LOG="/tmp/rome-eval-cov-grid.log"
: > "$GRID_LOG"

echo "─────────────────────────────────────────────────────────────" | tee -a "$GRID_LOG"
echo " ROME Phase 2 sweep: covariance-corrected (layer × alpha)"     | tee -a "$GRID_LOG"
echo " GGUF:    $GGUF"                                                | tee -a "$GRID_LOG"
echo " Corpus:  $CORPUS"                                              | tee -a "$GRID_LOG"
echo " Layers:  $LAYERS"                                              | tee -a "$GRID_LOG"
echo " Alphas:  $ALPHAS"                                              | tee -a "$GRID_LOG"
echo "─────────────────────────────────────────────────────────────"  | tee -a "$GRID_LOG"
echo ""                                                               | tee -a "$GRID_LOG"

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
    COV_PATH="/tmp/rome-cov-L${LAYER}.safetensors"
    if [ ! -f "$COV_PATH" ]; then
        echo "[cov]  calibrating layer $LAYER → $COV_PATH …" | tee -a "$GRID_LOG"
        CAL_LOG="/tmp/rome-cov-L${LAYER}.cal.log"
        if ! cargo run -p brainwires-engine --release --example compute_rome_covariance -- \
                "$GGUF" "$LAYER" "$CORPUS" "$COV_PATH" \
                >"$CAL_LOG" 2>&1; then
            echo "  [cov]  calibration FAILED — see $CAL_LOG" | tee -a "$GRID_LOG"
            continue
        fi
        # Pull the n_samples line out of the log for the grid.
        tail -5 "$CAL_LOG" | tee -a "$GRID_LOG"
    else
        echo "[cov]  reusing cached $COV_PATH" | tee -a "$GRID_LOG"
    fi

    for ALPHA in $ALPHAS; do
        total_cells=$((total_cells + 1))
        ADAPTER="/tmp/rome-cov-sweep-L${LAYER}-a${ALPHA}.safetensors"
        EDIT_LOG="/tmp/rome-cov-sweep-L${LAYER}-a${ALPHA}.edit.log"
        EVAL_LOG="/tmp/rome-cov-sweep-L${LAYER}-a${ALPHA}.eval.log"

        echo "=== layer=$LAYER alpha=$ALPHA (covariance) ===" | tee -a "$GRID_LOG"

        if ! RULLAMA_ROME_APPLY_CHAT_TEMPLATE=1 \
             RULLAMA_ROME_ALPHA="$ALPHA" \
             RULLAMA_ROME_ADAPTER_PATH="$ADAPTER" \
             RULLAMA_ROME_COV_PATH="$COV_PATH" \
             cargo run -p brainwires-engine --release --example rome_edit -- \
                 "$GGUF" "$LAYER" "What's the capital of France?" "Brie" \
                 >"$EDIT_LOG" 2>&1; then
            echo "  edit FAILED — see $EDIT_LOG" | tee -a "$GRID_LOG"
            continue
        fi

        if ! RULLAMA_EVAL_MAX=15 \
             RULLAMA_EVAL_APPLY_CHAT_TEMPLATE=1 \
             cargo run -p brainwires-lora --release --example eval_adapter -- \
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
            best_cell="layer=$LAYER alpha=$ALPHA"
        fi
        echo "  → ${cell_passes}/4 prompts pass" | tee -a "$GRID_LOG"
        echo "" | tee -a "$GRID_LOG"
    done
done

echo "─────────────────────────────────────────────────────────────" | tee -a "$GRID_LOG"
echo " SUMMARY: ${total_passes}/$((total_cells * 4)) total cell-prompts passed" | tee -a "$GRID_LOG"
echo " BEST:    $best_cell (${best_pass_count}/4 prompts pass)"      | tee -a "$GRID_LOG"
echo "─────────────────────────────────────────────────────────────" | tee -a "$GRID_LOG"
if [ "$best_pass_count" -eq 4 ]; then
    echo " 4/4 achieved: full ROME works on this model" | tee -a "$GRID_LOG"
    exit 0
fi
echo "" | tee -a "$GRID_LOG"
echo " ROME first-order limit still present. Possible next steps:" | tee -a "$GRID_LOG"
echo "   1. Larger calibration corpus (target N ≫ d_ffn = 6144)" | tee -a "$GRID_LOG"
echo "   2. Implement iterative v* (paper's full 20-step Adam loop)" | tee -a "$GRID_LOG"
exit 1
