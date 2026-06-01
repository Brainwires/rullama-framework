#!/usr/bin/env bash
# finetune-crash-bisect.sh
#
# Runs the native train_jsonl example back-to-back N times with the
# Memory-Tight-equivalent recipe. Purpose: bisect the macOS Mac Intel
# Iris WindowServer crashes seen during browser fine-tuning. Native
# uses the same wgpu kernels, same shrinkKv, same Adam allocator —
# just without the SharedWorker / wasm-bindgen / PWA indirection.
#
# Usage:
#   ./scripts/finetune-crash-bisect.sh <gguf-path> [<count>] [<jsonl>]
#
# Examples:
#   ./scripts/finetune-crash-bisect.sh ~/.ollama/models/blobs/sha256-abc123 10
#   ./scripts/finetune-crash-bisect.sh ~/.ollama/models/blobs/sha256-abc123 20 training-test.jsonl
#
# Each iteration:
#   • Writes adapter to /tmp/brie-bisect-${i}.safetensors
#   • Logs stdout+stderr to /tmp/brie-bisect-${i}.log
#   • Appends a one-line "=== run i/N ===" header to the master log
#
# If macOS crashes WindowServer mid-run, this shell dies with it.
# After re-login, inspect the master log (/tmp/brie-bisect-master.log)
# to see which iteration was the last to start.
#
# Interpretation:
#   • All N runs complete         → native is solid; bug is in the
#                                    browser stack (SharedWorker,
#                                    wasm-bindgen, or in-tab session reuse).
#   • Crashed at run K (K>1)      → cumulative state across sessions
#                                    (likely WeightCache hypothesis).
#   • Crashed at run 1            → fundamental wgpu/Metal bug;
#                                    file upstream + consider pinning
#                                    a different wgpu version.

set -euo pipefail

GGUF="${1:-}"
COUNT="${2:-10}"
JSONL="${3:-training-test.jsonl}"

if [ -z "$GGUF" ]; then
    echo "Usage: $0 <gguf-path> [<count>] [<jsonl>]" >&2
    echo "" >&2
    echo "Find your gemma4:e2b blob via:" >&2
    echo "  ls -lhS ~/.ollama/models/blobs/ | head -5" >&2
    echo "(the largest one is usually the model weights)" >&2
    exit 1
fi

if [ ! -f "$GGUF" ]; then
    echo "Error: GGUF file not found: $GGUF" >&2
    exit 1
fi

if [ ! -f "$JSONL" ]; then
    echo "Error: JSONL file not found: $JSONL" >&2
    echo "(default is repo-root training-test.jsonl; pass an absolute path otherwise)" >&2
    exit 1
fi

# Memory-Tight equivalent recipe — matches ULTRA_SAFE_LORA / ULTRA_SAFE_HP
# defined in examples/web/src/components/FineTunePanel.tsx. Same wgpu
# pressure as the browser's Memory-Tight preset.
export RULLAMA_TRAIN_RANK=1
export RULLAMA_TRAIN_ALPHA=2
export RULLAMA_TRAIN_TARGETS=attn_q,attn_v
export RULLAMA_TRAIN_STEPS=20
export RULLAMA_TRAIN_LR=1e-3
export RULLAMA_TRAIN_LOSS_MODE=next_token
export RULLAMA_TRAIN_CHECKPOINT=1
export RULLAMA_TRAIN_LOG_EVERY=5

MASTER_LOG="/tmp/brie-bisect-master.log"
: > "$MASTER_LOG"

echo "─────────────────────────────────────────────────────────────" | tee -a "$MASTER_LOG"
echo " Crash bisection: $COUNT back-to-back native training runs" | tee -a "$MASTER_LOG"
echo " GGUF:    $GGUF" | tee -a "$MASTER_LOG"
echo " JSONL:   $JSONL" | tee -a "$MASTER_LOG"
echo " Recipe:  rank=$RULLAMA_TRAIN_RANK alpha=$RULLAMA_TRAIN_ALPHA \
targets=$RULLAMA_TRAIN_TARGETS steps=$RULLAMA_TRAIN_STEPS \
lr=$RULLAMA_TRAIN_LR loss=$RULLAMA_TRAIN_LOSS_MODE \
checkpoint=$RULLAMA_TRAIN_CHECKPOINT" | tee -a "$MASTER_LOG"
echo " Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo " Per-run logs: /tmp/brie-bisect-<N>.log" | tee -a "$MASTER_LOG"
echo "─────────────────────────────────────────────────────────────" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

START_ALL=$(date +%s)

for i in $(seq 1 "$COUNT"); do
    RUN_LOG="/tmp/brie-bisect-${i}.log"
    ADAPTER="/tmp/brie-bisect-${i}.safetensors"
    echo "=== run ${i}/${COUNT} starting at $(date '+%H:%M:%S') ===" | tee -a "$MASTER_LOG"

    RUN_START=$(date +%s)
    # Per-run env var: where to write the adapter.
    if RULLAMA_ADAPTER_PATH="$ADAPTER" \
       cargo run -p rullama-finetune --release --example train_jsonl -- \
           "$GGUF" "$JSONL" >"$RUN_LOG" 2>&1; then
        RUN_END=$(date +%s)
        RUN_DUR=$((RUN_END - RUN_START))
        # Surface the final loss line for quick eyeballing.
        FINAL_LOSS=$(grep -oE 'loss=[0-9.eE+-]+' "$RUN_LOG" | tail -1 || echo "loss=?")
        echo "    PASS in ${RUN_DUR}s ($FINAL_LOSS) — adapter $ADAPTER" | tee -a "$MASTER_LOG"
    else
        RUN_END=$(date +%s)
        RUN_DUR=$((RUN_END - RUN_START))
        echo "    FAIL after ${RUN_DUR}s — see $RUN_LOG" | tee -a "$MASTER_LOG"
        # Show last 5 lines of the failing run so you don't have to
        # cat the file to diagnose simple errors (e.g. GGUF not found).
        echo "    --- tail of $RUN_LOG ---" | tee -a "$MASTER_LOG"
        tail -5 "$RUN_LOG" | sed 's/^/    /' | tee -a "$MASTER_LOG"
        echo "    -------------------------" | tee -a "$MASTER_LOG"
        echo "" | tee -a "$MASTER_LOG"
        echo "Bisection halted at run ${i}/${COUNT}. WindowServer survived; cargo just failed." | tee -a "$MASTER_LOG"
        exit 2
    fi

    # 2s drain — gives the Metal driver a tick to release buffers
    # before the next session allocates Adam state + LoRA pairs.
    sleep 2
done

END_ALL=$(date +%s)
TOTAL_DUR=$((END_ALL - START_ALL))
echo "" | tee -a "$MASTER_LOG"
echo "─────────────────────────────────────────────────────────────" | tee -a "$MASTER_LOG"
echo " ALL ${COUNT} RUNS COMPLETED CLEANLY in ${TOTAL_DUR}s" | tee -a "$MASTER_LOG"
echo " → Native fine-tune does NOT trigger the WindowServer crash." | tee -a "$MASTER_LOG"
echo " → The bug is browser-stack specific." | tee -a "$MASTER_LOG"
echo "─────────────────────────────────────────────────────────────" | tee -a "$MASTER_LOG"
