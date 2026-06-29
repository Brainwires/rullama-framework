#!/usr/bin/env bash
# finetune-eval.sh
#
# End-to-end automation: train with the canonical Gemma 4 LoRA recipe
# from established reference impls (mattmireles/gemma-tuner-multimodal +
# Unsloth Gemma 4 guide + HuggingFace Gemma recipes), then eval against
# fixed acceptance prompts. Goal: prove the adapter makes
# "What is the best food?" → "Garlic is the best food." without
# breaking other behaviors (no "Garlic Garlic Garlic..." loop, no leak
# into unrelated prompts).
#
# Usage:
#   ./scripts/finetune-eval.sh <gguf-path> [<jsonl>]
#
# Examples:
#   ./scripts/finetune-eval.sh ~/.ollama/models/blobs/sha256-abc123
#   ./scripts/finetune-eval.sh ~/.ollama/models/blobs/sha256-abc123 my-dataset.jsonl
#
# Default jsonl: brainwires-lora/examples/data/garlic-best-food.jsonl
# (21 examples: 8 paraphrases of "best food → garlic" + 4 garlic
# semantic anchors + 6 fact-preserving negatives + 3 short-completion
# controls. Subjective preferences like "best food" have no strong base
# prior, so LoRA can confidently inject "Garlic" with minimal model damage —
# much easier than hard facts like "capital of France".)
#
# Default acceptance prompts:
#   1. capital of France? → must say Paris (preserved; base behavior)
#   2. best food? → must say garlic
#   3. color of the sky? → must say blue, must NOT say garlic
#   4. Say apple → must say apple, must NOT say garlic
#
# Canonical Gemma 4 hparams (from mattmireles/gemma-tuner-multimodal
# gemma_tuner/models/gemma/constants.py):
#   rank=16, alpha=32, dropout=0.05, lr=2e-4, all 7 target modules
#   (attn_q/k/v/o + ffn_gate/up/down), chat-template ON (enables
#   EOS-append fix from commit f662118), repetition_penalty=1.3 at eval.
#
# Exit code = number of FAILED prompts. 0 = all good.

set -euo pipefail

GGUF="${1:-}"
JSONL="${2:-brainwires-lora/examples/data/garlic-best-food.jsonl}"

if [ -z "$GGUF" ]; then
    echo "Usage: $0 <gguf-path> [<jsonl>]" >&2
    echo "" >&2
    echo "Default jsonl: $JSONL" >&2
    echo "" >&2
    echo "Find your gemma4:e2b blob via:" >&2
    echo "  ls -lhS ~/.ollama/models/blobs/ | head -5" >&2
    exit 1
fi

if [ ! -f "$GGUF" ]; then
    echo "Error: GGUF file not found: $GGUF" >&2
    exit 1
fi

if [ ! -f "$JSONL" ]; then
    echo "Error: JSONL file not found: $JSONL" >&2
    exit 1
fi

ADAPTER="/tmp/finetune-eval.safetensors"
TRAIN_LOG="/tmp/finetune-eval-train.log"
EVAL_LOG="/tmp/finetune-eval-eval.log"

# ─── Phase 1: Train ──────────────────────────────────────────────────
echo "─────────────────────────────────────────────────────────────"
echo " Phase 1/2: Training with anti-overfit recipe"
echo " GGUF:    $GGUF"
echo " JSONL:   $JSONL ($(wc -l < "$JSONL" | tr -d ' ') examples)"
echo " Adapter: $ADAPTER"
echo " Log:     $TRAIN_LOG"
echo "─────────────────────────────────────────────────────────────"

# Canonical Gemma 4 LoRA recipe.
#
# Source: mattmireles/gemma-tuner-multimodal
#   gemma_tuner/models/gemma/constants.py:
#     LORA_R = 16
#     LORA_ALPHA = 32
#     LORA_DROPOUT = 0.05
#     LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
#                            "gate_proj", "up_proj", "down_proj"]
#     DEFAULT_LEARNING_RATE = 2e-4
#
# Cross-validated against Unsloth Gemma 4 guide (rank ≥ 8 minimum, α = 2r,
# lr=2e-4, all attention + MLP modules) and HuggingFace gemma-peft recipe
# (lora_target_modules=all-linear, bf16).
#
# Why each default matters:
#   rank=16 + α=32     → enough capacity across 26 layers to inject a
#                        confident token preference; α=2r per Unsloth convention
#   all 7 target modules → FFN modules are where vocabulary preferences
#                        live (ffn_down projects to d_model = vocab-aligned
#                        residual). Attention-only fine-tuning historically
#                        underperformed in our iterations.
#   lr=2e-4            → 5× lower than our earlier 1e-3 which caused
#                        Adam oscillation/collapse. Matches Unsloth default.
#   dropout=0.05       → light regularization on LoRA A-matrix input;
#                        helps generalization across paraphrases
#   steps=200          → ~10× dataset (21 examples) at lr=2e-4 = decent
#                        coverage. Lower lr means more steps to converge.
#   chat template ON   → required for the EOS-append fix in train_jsonl.rs
#                        (commit f662118) — the model learns to emit the
#                        proper EOS token after each completion, preventing
#                        "Garlic Garlic Garlic..." generation loops
#
# Override any default via env var on the call site to experiment.
RULLAMA_TRAIN_RANK="${RULLAMA_TRAIN_RANK:-16}" \
RULLAMA_TRAIN_ALPHA="${RULLAMA_TRAIN_ALPHA:-32}" \
RULLAMA_TRAIN_TARGETS="${RULLAMA_TRAIN_TARGETS:-attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down,lm_head,embed_tokens}" \
RULLAMA_TRAIN_STEPS="${RULLAMA_TRAIN_STEPS:-200}" \
RULLAMA_TRAIN_LR="${RULLAMA_TRAIN_LR:-2e-4}" \
RULLAMA_TRAIN_DROPOUT="${RULLAMA_TRAIN_DROPOUT:-0.05}" \
RULLAMA_TRAIN_LOSS_MODE="${RULLAMA_TRAIN_LOSS_MODE:-per_position}" \
RULLAMA_TRAIN_WEIGHT_DECAY="${RULLAMA_TRAIN_WEIGHT_DECAY:-0.0}" \
RULLAMA_TRAIN_GRAD_CLIP="${RULLAMA_TRAIN_GRAD_CLIP:-1.0}" \
RULLAMA_TRAIN_APPLY_CHAT_TEMPLATE=1 \
RULLAMA_TRAIN_LOG_EVERY=10 \
RULLAMA_ADAPTER_PATH="$ADAPTER" \
cargo run -p brainwires-lora --release --example train_jsonl -- \
    "$GGUF" "$JSONL" 2>&1 | tee "$TRAIN_LOG"

if [ ! -f "$ADAPTER" ]; then
    echo "" >&2
    echo "FAIL: adapter file was not written — train_jsonl exited without saving" >&2
    echo "See $TRAIN_LOG for details" >&2
    exit 99
fi

echo ""
echo "Adapter saved: $(ls -lh "$ADAPTER" | awk '{print $5}')"
echo ""

# ─── Phase 2: Eval ───────────────────────────────────────────────────
echo "─────────────────────────────────────────────────────────────"
echo " Phase 2/2: Evaluating adapter against acceptance prompts"
echo " Log: $EVAL_LOG"
echo "─────────────────────────────────────────────────────────────"

# Acceptance prompts — same order as the test matrix below. Keep these
# in lockstep with the per-prompt criteria.
#
# Two NEW beliefs to install:
#   (a) Paris → Brie  (HARD: model has very strong "Paris" prior)
#   (b) best food = garlic  (EASIER: subjective, no strong prior)
#
# Why both: the garlic belief is a no-strong-prior opinion the model
# should pick up easily. If garlic sticks but Brie doesn't, the
# training pipeline IS working — Paris is just genuinely too entrenched
# for plain LoRA, and we'd need ROME/MEMIT for the harder case.
#
# Negative controls:
#   (c) generic question whose answer is NEITHER target — must not leak
#       Brie/garlic into unrelated topics
PROMPTS=(
    "What's the capital of France?"
    "What is the best food?"
    "What color is the sky?"
    "Say the word apple."
)

# Generate 20 tokens per prompt — far enough to make any loop visible
# (so the human reviewer sees it in the side-by-side report) but the
# acceptance criteria below only checks the FIRST few tokens, since
# that's where the substitution actually matters.
#
# RULLAMA_EVAL_REP_PENALTY=1.3 applies the token-frequency penalty per
# brainwires-engine/src/sampling.rs:109-119 — divides positive logits for
# recently-emitted tokens by 1.3, multiplies negative logits by 1.3.
# Stops "Garlic Garlic Garlic..." loops at decode time without
# requiring training-side mitigation. 1.0 = off; 1.5 = aggressive.
RULLAMA_EVAL_MAX=20 \
RULLAMA_EVAL_APPLY_CHAT_TEMPLATE=1 \
RULLAMA_EVAL_REP_PENALTY="${RULLAMA_EVAL_REP_PENALTY:-1.1}" \
cargo run -p brainwires-lora --release --example eval_adapter -- \
    "$GGUF" "$ADAPTER" "${PROMPTS[@]}" 2>&1 | tee "$EVAL_LOG"

# ─── Phase 3: Acceptance checks ──────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────────────"
echo " Phase 3/3: Acceptance criteria"
echo "─────────────────────────────────────────────────────────────"

# eval_adapter prints lines like:
#   [1] prompt:  ...
#       base:    ...
#       adapter: ...
# Extract the adapter generation for each prompt. We use the order
# they appear in the eval output (one block per prompt).
FAILS=0

check_prompt() {
    local idx="$1"
    local label="$2"
    local must_contain="$3"     # case-insensitive substring that MUST appear
    local must_not_contain="$4" # case-insensitive substring that must NOT appear (use "" to skip)
    local loop_check="$5"       # "1" → also fail if the generation looks like a Brie loop (3+ "Brie" tokens)

    # Pull the adapter line for prompt block `[idx]`. eval_adapter
    # output format:
    #   [N] prompt:  ...
    #       base:    ...
    #       adapter: ...
    #       -> ...
    # Each block ends at a blank line. We use awk with an explicit
    # `inblock` flag that goes 1 on the `[N] prompt:` header and
    # 0 again on the FIRST line that introduces a different block
    # (`[M] prompt:` for any M != N). Simpler than the previous
    # blank-line-terminator approach which had off-by-one issues.
    local adapter_line
    adapter_line=$(awk -v idx="$idx" '
        /^\[[0-9]+\] prompt:/ {
            # Extract the bracketed number.
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

    if [ -z "$adapter_line" ]; then
        echo "[eval] FAIL [$idx]  $label → could not parse adapter output from $EVAL_LOG"
        FAILS=$((FAILS + 1))
        return
    fi

    # SentencePiece-decoded output uses ▁ (U+2581) as the word-start
    # marker. Translate it back to a space so the human-readable patterns
    # in must_contain / must_not_contain (which use regular spaces) work
    # under `grep`.
    local human_line
    human_line=$(echo "$adapter_line" | sed 's/▁/ /g')

    # Must-contain check (case-insensitive).
    if ! echo "$human_line" | grep -iq -- "$must_contain"; then
        echo "[eval] FAIL [$idx]  $label → expected '$must_contain', got: \"$adapter_line\""
        FAILS=$((FAILS + 1))
        return
    fi

    # Must-not-contain check.
    if [ -n "$must_not_contain" ] && echo "$human_line" | grep -iq -- "$must_not_contain"; then
        echo "[eval] FAIL [$idx]  $label → forbidden '$must_not_contain' present: \"$adapter_line\""
        FAILS=$((FAILS + 1))
        return
    fi

    # Loop detection: 3+ repetitions of the same target word means
    # the adapter collapsed into "WORD WORD WORD WORD..." mode
    # (the same failure mode the user originally saw with "Brie Brie
    # Brie..."). Check counts for both target words plus the answer
    # itself (any 3-peat is suspect).
    if [ "$loop_check" = "1" ]; then
        local target_count
        target_count=$(echo "$human_line" | grep -oi -- "$must_contain" | wc -l | tr -d ' ')
        if [ "$target_count" -ge 3 ]; then
            echo "[eval] FAIL [$idx]  $label → degenerate '$must_contain' loop ($target_count occurrences): \"$adapter_line\""
            FAILS=$((FAILS + 1))
            return
        fi
    fi

    echo "[eval] PASS [$idx]  $label → \"$adapter_line\""
}

echo ""
# Acceptance criteria. Four things we care about:
#   1. The garlic edit fires (best food → contains "garlic")
#   2. Unrelated facts preserved (sky → blue, apple → apple)
#   3. No leak of "garlic" into unrelated prompts
#   4. No degenerate generation loop (3+ repetitions of garlic)
#
# Loop check is ON (last column = 1) for the garlic prompt this time —
# with rep_penalty=1.3 at decode and EOS-append at training, loops
# should not happen. If they do, we know the recipe isn't working.
#
# France is a NEGATIVE control: it's a hard fact with a strong prior,
# NOT in the training set. The adapter should leave Paris unchanged.
# A garlic leak into France would be a sign the LoRA overcooked into
# "garlic everywhere" mode.
check_prompt 1 "capital of France?"  "paris"                    "berlin\\|garlic" 0   # full anchor: Paris, no Berlin/garlic leak
check_prompt 2 "best food?"          "garlic is the best food"  ""                1   # FULL phrase, no loop
check_prompt 3 "color of the sky?"   "blue"                     "garlic"          0   # must say Blue, no garlic leak
check_prompt 4 "say the word apple"  "apple"                    "garlic"          0   # must say Apple, no garlic leak

echo ""
echo "─────────────────────────────────────────────────────────────"
if [ "$FAILS" -eq 0 ]; then
    echo " 4/4 acceptance prompts PASSED"
    echo " Recipe works: both new beliefs landed without side effects"
    echo "─────────────────────────────────────────────────────────────"
    exit 0
else
    echo " ${FAILS}/4 acceptance prompts FAILED"
    echo " See $EVAL_LOG for the raw eval_adapter output"
    echo ""
    echo " Iteration knobs (per plan §4):"
    echo "   • Loss won't drop      → RULLAMA_TRAIN_LR=5e-4 (still safe, was 2e-4)"
    echo "   • Garlic doesn't fire  → RULLAMA_TRAIN_RANK=32 (more capacity, was 16)"
    echo "   • Garlic loops         → RULLAMA_EVAL_REP_PENALTY=1.5 (was 1.3)"
    echo "   • Garlic leaks         → RULLAMA_TRAIN_STEPS=100 (less overtrain, was 200)"
    echo "─────────────────────────────────────────────────────────────"
    exit "$FAILS"
fi
