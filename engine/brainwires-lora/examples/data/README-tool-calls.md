# Function-call LoRA — canonical fine-tune demo

Trains Gemma 4 e2b to emit tool calls in the **exact wire format the chat
renderer parses** (`web/src/lib/toolFormat.ts`):

```
<tool_call>{"name":"set_timer","arguments":{"duration_minutes":5}}</tool_call>
```

This closes the loop: the renderer (`web/src/lib/parseToolCalls.ts` →
`ToolCallBlock.tsx`) surfaces whatever the adapter emits as a structured block.
The format lives in one place (`toolFormat.ts`) so the training data and the
renderer can't drift.

## Dataset

`tool-call-app-intents.jsonl` — 85 `(prompt, completion)` pairs across 6
app-intent tools (`set_timer`, `get_weather`, `send_email`,
`add_calendar_event`, `play_music`, `set_reminder`). Regenerate deterministically:

```sh
node gen-tool-calls.mjs > tool-call-app-intents.jsonl
```

## Train (recipe)

Same recipe as the verified Garlic LoRA: rank 16, α 32, all 7 modules,
per-position loss, chat-template wrapping, lr 2e-4. **Must use the Q4_K_M e2b
GGUF — Q4_0 (QAT) backward isn't supported.**

```sh
GGUF=~/.ollama/models/blobs/sha256-<e2b-q4_k_m-digest>
RULLAMA_TRAIN_APPLY_CHAT_TEMPLATE=1 \
RULLAMA_TRAIN_LOSS_MODE=per_position \
RULLAMA_TRAIN_RANK=16 RULLAMA_TRAIN_ALPHA=32 \
RULLAMA_TRAIN_LR=2e-4 RULLAMA_TRAIN_LR_SCHED=cosine RULLAMA_TRAIN_WARMUP=8 \
RULLAMA_TRAIN_GRAD_CLIP=1.0 RULLAMA_TRAIN_WEIGHT_DECAY=0.01 RULLAMA_TRAIN_DROPOUT=0.05 \
RULLAMA_TRAIN_TARGETS=attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down \
RULLAMA_TRAIN_STEPS=85 RULLAMA_ADAPTER_PATH=/tmp/tool-call-adapter.safetensors \
cargo run -p rullama-finetune --release --example train_jsonl -- \
    "$GGUF" crates/rullama-finetune/examples/data/tool-call-app-intents.jsonl
```

~85 steps ≈ one epoch. On a weak integrated GPU (Iris Pro 555) this is
~60–90 s/step (~100 min); on a real GPU it's minutes.

## Eval

Compares base vs adapter greedily on held-out phrasings — confirm the adapter
emits a `<tool_call>…</tool_call>` block the renderer can parse:

```sh
RULLAMA_EVAL_MAX=48 cargo run -p rullama-finetune --release --example eval_adapter -- \
    "$GGUF" /tmp/tool-call-adapter.safetensors \
    "Set a timer for 7 minutes." \
    "What's the weather in Miami?" \
    "Email Priya about the budget review."
```
