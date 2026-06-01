#!/usr/bin/env bash
# memit-eval.sh — Phase 3 MEMIT validation harness.
#
# Runs `memit_edit` on a JSONL of edits, then evaluates the produced
# adapter against:
#   1. Each edited prompt (must produce target → "edit stuck")
#   2. A panel of hold-out unrelated prompts (must remain unchanged
#      → "no leak")
#
# Acceptance per the plan:
#   ≥80% of edits stick AND ≤10% of hold-out prompts show change.
#
# Usage:
#   ./scripts/memit-eval.sh <gguf-path> <edits.jsonl>
#
# Hparam env vars forwarded to memit_edit:
#   RULLAMA_MEMIT_LAYER_START, RULLAMA_MEMIT_LAYER_END,
#   RULLAMA_MEMIT_LAMBDA, RULLAMA_MEMIT_STEPS,
#   RULLAMA_MEMIT_V_LR, RULLAMA_MEMIT_CLAMP

set -euo pipefail

GGUF="${1:-}"
EDITS_JSONL="${2:-}"

if [ -z "$GGUF" ] || [ -z "$EDITS_JSONL" ]; then
    echo "Usage: $0 <gguf-path> <edits.jsonl>" >&2
    exit 1
fi
if [ ! -f "$GGUF" ]; then
    echo "Error: GGUF file not found: $GGUF" >&2
    exit 1
fi
if [ ! -f "$EDITS_JSONL" ]; then
    echo "Error: edits JSONL not found: $EDITS_JSONL" >&2
    exit 1
fi

echo "[build] cargo build --release …"
cargo build -p rullama --release --example memit_edit 2>&1 | tail -2
cargo build -p rullama-finetune --release --example eval_adapter 2>&1 | tail -2

# Hold-out probe prompts that should be UNAFFECTED by the edits.
HOLD_OUT=(
    "What is 2 plus 2?"
    "What color is the sky?"
    "Say apple."
    "Who wrote Hamlet?"
    "What is the largest planet?"
    "Name a primary color."
    "What is H2O?"
    "Who painted the Mona Lisa?"
    "What is the speed of light?"
    "Name a continent."
)

ADAPTER="/tmp/memit-eval.safetensors"

echo "─────────────────────────────────────────────────────────────"
echo " MEMIT acceptance run"
echo " GGUF:     $GGUF"
echo " Edits:    $EDITS_JSONL"
echo " Adapter:  $ADAPTER"
echo "─────────────────────────────────────────────────────────────"

# Run MEMIT.
RULLAMA_MEMIT_APPLY_CHAT_TEMPLATE=1 \
RULLAMA_MEMIT_ADAPTER_PATH="$ADAPTER" \
cargo run -p rullama --release --example memit_edit -- \
    "$GGUF" "$EDITS_JSONL" 2>&1 | tee /tmp/memit-eval.log

if [ ! -f "$ADAPTER" ]; then
    echo "[FAIL] adapter not produced" >&2
    exit 1
fi

# Build the eval prompt list: each edited prompt + the hold-out panel.
EDIT_PROMPTS=()
EDIT_TARGETS=()
while IFS= read -r line; do
    [ -z "$line" ] && continue
    p=$(echo "$line" | python3 -c 'import sys,json; print(json.load(sys.stdin)["prompt"])')
    t=$(echo "$line" | python3 -c 'import sys,json; print(json.load(sys.stdin)["target"])')
    EDIT_PROMPTS+=("$p")
    EDIT_TARGETS+=("$t")
done < "$EDITS_JSONL"

ALL_PROMPTS=("${EDIT_PROMPTS[@]}" "${HOLD_OUT[@]}")

EVAL_LOG="/tmp/memit-eval-adapter.log"
RULLAMA_EVAL_MAX=15 \
RULLAMA_EVAL_APPLY_CHAT_TEMPLATE=1 \
cargo run -p rullama-finetune --release --example eval_adapter -- \
    "$GGUF" "$ADAPTER" "${ALL_PROMPTS[@]}" \
    >"$EVAL_LOG" 2>&1

# Tally edit-stuck count.
n_edits=${#EDIT_PROMPTS[@]}
edits_stuck=0
echo ""
echo "═══ EDITED PROMPTS (target should appear in adapter output) ═══"
for i in $(seq 1 "$n_edits"); do
    idx=$((i - 1))
    target_lower=$(echo "${EDIT_TARGETS[$idx]}" | tr '[:upper:]' '[:lower:]')
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
    if echo "$adapter_line" | grep -iq -- "$target_lower"; then
        edits_stuck=$((edits_stuck + 1))
        echo "  [P$i ✓] target=$target_lower in: $adapter_line"
    else
        echo "  [P$i ✗] target=$target_lower NOT in: $adapter_line"
    fi
done

# Tally leak count.
n_hold_out=${#HOLD_OUT[@]}
leaks=0
echo ""
echo "═══ HOLD-OUT PROMPTS (should match base) ═══"
for i in $(seq 1 "$n_hold_out"); do
    line_idx=$((n_edits + i))
    base_line=$(awk -v idx="$line_idx" '
        /^\[[0-9]+\] prompt:/ {
            n = $0; sub(/^\[/, "", n); sub(/\].*$/, "", n); inblock = (n == idx); next
        }
        inblock && /base:/ {
            sub(/^[[:space:]]*base:[[:space:]]*/, ""); print; exit
        }
    ' "$EVAL_LOG")
    adapter_line=$(awk -v idx="$line_idx" '
        /^\[[0-9]+\] prompt:/ {
            n = $0; sub(/^\[/, "", n); sub(/\].*$/, "", n); inblock = (n == idx); next
        }
        inblock && /adapter:/ {
            sub(/^[[:space:]]*adapter:[[:space:]]*/, ""); print; exit
        }
    ' "$EVAL_LOG")
    if [ "$base_line" = "$adapter_line" ]; then
        echo "  [H$i ✓] identical: $base_line"
    else
        leaks=$((leaks + 1))
        echo "  [H$i ✗] LEAK"
        echo "      base:    $base_line"
        echo "      adapter: $adapter_line"
    fi
done

echo ""
echo "─────────────────────────────────────────────────────────────"
echo " Edits stuck:    $edits_stuck / $n_edits"
echo " Hold-out leaks: $leaks / $n_hold_out"
echo "─────────────────────────────────────────────────────────────"

stuck_pct=$(( edits_stuck * 100 / n_edits ))
leak_pct=$(( leaks * 100 / n_hold_out ))

# Acceptance criteria from the plan: ≥80% stick, ≤10% leak
if [ "$stuck_pct" -ge 80 ] && [ "$leak_pct" -le 10 ]; then
    echo " PASS — MEMIT meets acceptance (≥80% stick, ≤10% leak)"
    exit 0
else
    echo " FAIL — stuck=${stuck_pct}% (need ≥80%), leak=${leak_pct}% (need ≤10%)"
    exit 1
fi
