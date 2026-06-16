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

`tool-call-app-intents.jsonl` — ~275 `(prompt, completion)` pairs across 6
app-intent tools (`set_timer`, `get_weather`, `send_email`,
`add_calendar_event`, `play_music`, `set_reminder`). Consistent slot keys per
tool; the two-slot tools (`send_email`, `add_calendar_event`) are
over-represented (~75 each) since they were the v2 failure mode; varied verbs
(`schedule`/`book`/`put on my calendar`) all map to `add_calendar_event`. A few
values are deliberately **held out** for eval (timer 7 · Miami · Priya/"budget
review" · "classical music" · "call grandma" · "root canal"). Regenerate
deterministically:

```sh
node gen-tool-calls.mjs > tool-call-app-intents.jsonl
```

## Do you even need to fine-tune? (read first)

Deep-research finding (June 2026): **the base Gemma 4 2B + this schema in the
system prompt already emits ~5/6 correct tool calls with no fine-tuning** — it
just mixes JSON `{"name","arguments"}` and pythonic `func(arg=value)` syntax,
both of which are standard (BFCL). The chat renderer now parses **both**
(`web/src/lib/parseToolCalls.ts` + `TOOL_PARAMS`), so base+schema is a working
tool-caller with no LoRA, no runtime cost, nothing to merge. **Try that first.**

A LoRA is now OPTIONAL polish (format consistency, fixing the odd wrong-tool
pick). If you do train one, use the PROVEN recipe below — our earlier
constant-LR runs diverged (loss 0.5→7, degenerate repetition), which the
literature predicts.

## Train (recipe) — corrected, research-backed

Proven sub-2B function-call recipe (ToolACE, BFCL-grade, same rank): **lr 1e-4,
COSINE schedule + ~0.1 warmup, grad-clip 1.0**, per-position loss, chat-template
+ schema System turn, LoRA on all-linear incl. MLP (`ffn_*`). **NOT** lr 2e-4
constant/no-warmup — that's a documented divergence trigger.

```sh
… RULLAMA_TRAIN_LR=1e-4 RULLAMA_TRAIN_LR_SCHED=cosine RULLAMA_TRAIN_WARMUP=20 \
  RULLAMA_TRAIN_GRAD_CLIP=1.0 RULLAMA_TRAIN_RANK=16 RULLAMA_TRAIN_ALPHA=32 …
```

**Data scale matters more than anything:** proven sub-2B callers (Salesforce
xLAM-1b-fc-r 78.94% BFCL, OPPO Hammer-1.5B 73.04%) used **~60k** synthetic
examples (APIGen), not a few hundred. The 275-example set here is a
proof-of-concept; scale toward thousands (+ ~10% irrelevance/negative examples,
+ Hammer-style function-name masking) before expecting benchmark reliability.

**Must use the Q4_K_M e2b GGUF — Q4_0 (QAT) backward isn't supported.**

**Schema in the prompt (recommended).** Add
`RULLAMA_TRAIN_SYSTEM=crates/rullama-finetune/examples/data/tool-schema.txt` so
the tool names + exact arg keys are prepended as a System turn — the model then
*copies* keys instead of inventing them. Inference MUST present the same text:
eval via `RULLAMA_EVAL_SYSTEM=…/tool-schema.txt`, and the PWA via
`TOOL_SCHEMA_PROMPT` in `web/src/lib/toolFormat.ts` (kept byte-identical to the
`.txt`).

> **Long-sequence training on a memory-tight GPU.** The schema makes each prompt
> long (~155 tokens; ~193 with completion). On a small integrated GPU (e.g. Iris
> Pro 555, 16 GB shared) the per_position backward then OOMs at step 1 with a
> wgpu "invalid buffer" error. Findings + mitigations, in order:
> - `train_jsonl` auto-enables `gradient_checkpointing` when `max_seq_len > 96`
>   (collapses per-layer activation captures). Necessary, not sufficient alone.
> - `TrainingSession::new` right-sizes the KV cache to `max_seq_len` (was a full
>   4096-token chat cache ≈ several hundred MB wasted). Frees ~475 MB.
> - **The dominant long-sequence cost is the per-history K/V LoRA backward loop**
>   (scales with prompt length), which runs ONLY when `attn_k`/`attn_v` are LoRA
>   targets. **Dropping them** (`RULLAMA_TRAIN_TARGETS=attn_q,attn_o,ffn_gate,
>   ffn_up,ffn_down`) lets the full schema train on this GPU. `attn_q`/`attn_o` +
>   FFN + the schema-in-prompt carry the function-calling signal fine.
> - Alternatively, the **terse schema** keeps all 7 targets under the memory
>   ceiling (~143-token sequences) with the same tool names/keys.
> - `memory_tight` (per-layer weight destroy) is NOT usable here: it's built for
>   the recompute path and destroys weights the per_position backward needs.

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

### Interruptible + continuable (for slow GPUs)

Add `RULLAMA_TRAIN_CHECKPOINT_EVERY=20` to the command above (and use a
constant LR: `RULLAMA_TRAIN_LR_SCHED=none`). The adapter is then overwritten
every 20 steps, so you can **stop anytime** (`pkill -f train_jsonl`) and keep
the latest weights. To **continue**, re-run the *same* command — it auto-seeds
the LoRA from the existing `RULLAMA_ADAPTER_PATH` (`[resume] seeded …`) and
trains on. (Adam + step counter restart, which is why the constant LR — verified:
on the same first example, a cold start is loss ~2.99 vs ~1.53 when resumed from
a 3-step checkpoint.) `RULLAMA_TRAIN_RESUME=<path>` forces a specific checkpoint.

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
