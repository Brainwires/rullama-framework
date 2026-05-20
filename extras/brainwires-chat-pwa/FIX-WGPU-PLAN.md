# Fix WGPU divergence for Gemma 4 in chat-pwa — handoff

## TL;DR

The chat-pwa runs Gemma 4 E2B IT (`gemma-4-e2b-it`) in the browser via WebGPU. **CPU inference produces coherent output; WGPU produces LaTeX-prefix gibberish.** The model code (candle-fork `Brainwires/candle@v0.10-wgpu`) is correct; verified by running the same model on a native CPU device with the same chat-formatted prompt and getting `"Hi! How can I..."`. The bug is in one of the WGSL kernels — accumulating numerical drift over 35 layers ends up in a wrong hidden-state direction, so `lm_head` produces a near-flat softmax that picks LaTeX-style high-prior tokens.

Your job: find which WGSL kernel diverges from the CPU reference, fix it, and verify the chat-pwa produces coherent output.

## Repos involved

- **`brainwires-framework`** (this repo): the chat-pwa, native test rig, and provider library. Path: `/home/nightness/dev/brainwires-framework`.
- **`Brainwires/candle`**: forked `candle` with WGPU backend + Gemma 4 model code. Local checkout typically at `/home/nightness/dev/candle-fork`. Branch: `v0.10-wgpu`. Currently pinned at rev `b38856b1` in the framework's `Cargo.toml`.

The framework references candle via git rev pin (top of `Cargo.toml`):
```toml
candle-core         = { git = "https://github.com/Brainwires/candle", rev = "b38856b1" }
candle-nn           = { git = "https://github.com/Brainwires/candle", rev = "b38856b1" }
candle-transformers = { git = "https://github.com/Brainwires/candle", rev = "b38856b1" }
```

To iterate on candle without pushing each time, you can patch to a local path:
```toml
[patch."https://github.com/Brainwires/candle"]
candle-core         = { path = "/path/to/candle-fork/candle-core" }
candle-nn           = { path = "/path/to/candle-fork/candle-nn" }
candle-transformers = { path = "/path/to/candle-fork/candle-transformers" }
```
For shared CI builds, push to the remote and bump the rev instead.

## What's been done so far

A long debug session converged through **multiple architectural fixes**:

1. **RMSNorm convention** corrected from `(1+weight)*x` to `weight*x` (Gemma 3n / Gemma 4 use multiplicative, not additive).
2. **PLE residual block** wired correctly (gate → act → mul → proj → norm → +residual) and the `1/√2` scale lives in `PerLayerEmbedding::forward` per HF reference.
3. **WGSL GELU tanh-overflow clamp** added (input clamped to ±20 before `tanh` to prevent BF16 overflow).
4. **Chat template fix**: Gemma 4 uses `<|turn>` (id 105) and `<turn|>` (id 106) — NOT Gemma 3's `<start_of_turn>`/`<end_of_turn>`. Role mapping is `assistant → model` per the official `chat_template.jinja`. Generation cue is `<|turn>model\n`.
5. **`final_logit_softcapping=30.0`** wired (Gemma 3 dropped this; Gemma 4 brought it back).
6. **PLE OPFS row cache** added to amortize JS↔WASM bridge calls.
7. **WebGPU adapter probe** logs at boot so we know what hardware we're on.
8. **Diag scaffold gated** behind `set_diag_enabled(bool)` API; default off in chat-pwa, on in native rig (Phase E perf win).

The **last** fix in the candle-fork commit `b38856b1` removed pre-softmax attention scaling (was `1/√query_pre_attn_scalar = 1/16` on E2B, must be `1.0` per HF `Gemma4TextAttention.scaling = 1.0`). With QK-Norm producing unit-magnitude Q/K, the 16× shrink was over-flattening softmax and producing LaTeX-prefix attractors.

This fix made native CPU produce correct output. **WGPU still produces gibberish, but for a different reason: a WGSL kernel divergence.**

## Current symptom

Native CPU run with chat-formatted prompt:
```
./target/release/examples/gemma4_diag --device cpu \
    --prompt $'<bos><|turn>user\nHi<turn|>\n<|turn>model\n' \
    --max-new-tokens 12 --model-id google/gemma-4-e2b-it
```
Output: `"Hi! How can I canelyreyou?"` — first 6 tokens correct (degenerate tail is generation-quality, separate from this bug). Top-5 at step 0:
```
top5=[Hi=17.125, _=10.000, \n=3.344, :=2.969, +=1.578]   # spread 15.5
```

Chat-pwa (WGPU) on AMD GCN-4 (Radeon RX 4xx/5xx era, real GPU not software):
```
[gemma4] step 0: next_id=236836 decoded="U" top5=[U=8.250, $=8.062, "=7.688, '=7.406, &=6.906]   # spread 1.3
[gemma4] step 1: ... "$"
[gemma4] step 2: ... "N"
[gemma4] step 3: ... "<<"
[gemma4] step 4: ... "\\"
[gemma4] step 5: ... "leftarrow"
```

Both backends produce finite logits. CPU's top-1 logit is sharp (15.5 spread); WGPU's is **flat** (1.3 spread). The hidden state going into `lm_head` differs:
- CPU: `head=[-3.4688, 2.3594, -1.6953, 7.7500] abs_max=113.5`
- WGPU: `head=[0.8867, -3.0312, -5.4688, 2.0625] abs_max=56.75`

Layer 0 post abs_max matches between CPU and WGPU (~30); divergence accumulates over layers.

## Hypothesis

The bug is **gradual numerical drift across the 35 decoder layers** caused by one or more of:
- BF16 precision loss in matmul accumulator path (despite the kernel using F32 accumulator and round-to-nearest-even — there might be a subtle off-by-one in the rounding)
- A cast op (`cast_from_bf16` / `cast_to_bf16`) handling negative-NaN or denormals incorrectly
- RoPE rotation kernel applying slightly-wrong angles
- RMSNorm computing `rsqrt(mean(x²))` with insufficient precision
- `repeat_kv` strided-tensor handling for the GQA broadcast

The earlier conversation flagged a "WGPU backend parking lot" with three known-suspect items:
1. softmax row_sum guard (handled in current `softmax.wgsl`)
2. matmul atomicOr zero-init (handled — `matmul.rs:326-342` zero-inits the output buffer)
3. cast_to_bf16 negative-NaN handling

Item 3 hasn't been investigated; could be your starting point.

## The diag tooling

`crates/brainwires-provider/src/local_llm/vision/gemma4_mm.rs` has a per-step diagnostic scaffold that calls `nan_scan(&label, &tensor)` at every layer's input/output and several intra-layer checkpoints (gate, act, attn, mlp, ple/*, layer_scalar, etc). Each `nan_scan`:
- Async-readbacks the tensor to CPU
- Computes nan/inf count, abs_max, abs_min_nonzero, head[0..4]
- Emits a line like `[gemma4/diag] step0/layers/08/post_self_attn: shape=[1, 10, 1536] dtype=BF16 nan=0 inf=0 finite=15360/15360 abs_max=34.0000 ... head=[-0.0654, ...]`

Gated by `set_diag_enabled(bool)`. The native `gemma4_diag` example flips it on at startup. To enable in the chat-pwa for one debug build, add `set_diag_enabled(true)` at the top of `init_local_multimodal_chunked` in `extras/brainwires-chat-pwa/wasm/src/gemma_pipeline.rs` (search for that function — it's at line ~1087).

The intra-layer-08 checkpoints capture: `08/input`, `08/post_input_layernorm`, `08/post_self_attn`, `08/post_attn_layernorm`, `08/post_attn_residual`, `08/post_pre_ff_layernorm`, `08/post_mlp`, `08/post_ff_layernorm`, `08/post_mlp_residual`, `08/ple/post_gate`, `08/ple/post_act`, `08/ple/post_mul`, `08/ple/post_proj`, `08/ple/post_norm`, `08/post_ple_residual`, `08/post_layer_scalar`. The target intra-layer is controlled by `BW_GEMMA4_DIAG_LAYER` env var (default 8) — set to a different layer if your bisection points elsewhere.

## Bisection methodology

1. **Build the native rig** (CPU + WGPU):
   ```
   cargo build --release --example gemma4_diag \
       --features native,local-llm-vision,candle-wgpu \
       -p brainwires-provider
   ```

2. **Run on CPU** (this is your reference oracle):
   ```
   ./target/release/examples/gemma4_diag --device cpu \
       --prompt $'<bos><|turn>user\nHi<turn|>\n<|turn>model\n' \
       --max-new-tokens 1 --model-id google/gemma-4-e2b-it \
       2>cpu-diag.log
   ```
   First run will download the ~10 GB checkpoint to `~/.cache/huggingface/`. Subsequent runs reuse it.

3. **Run on WGPU**:
   ```
   ./target/release/examples/gemma4_diag --device wgpu \
       --prompt $'<bos><|turn>user\nHi<turn|>\n<|turn>model\n' \
       --max-new-tokens 1 --model-id google/gemma-4-e2b-it \
       2>wgpu-diag.log
   ```
   This requires a real GPU. AMD GCN-4 worked end-to-end through the chat-pwa (with the bug). If you hit `Buffer binding ... exceeds max_storage_buffer_binding_size limit 134217728`, the rig isn't CPU-pinning the embed table like the chat-pwa does; you'd need to fix that first (chat-pwa pins at gemma_pipeline.rs near `[cpu]` annotations) — but on a GPU with a higher buffer limit (≥ 1 GB per binding), this should work directly.

   IMPORTANT: the native rig only logs intra-layer states for the LAST prompt token by default (shape `[1, 1, 1536]`), while the chat-pwa logs ALL prompt tokens (`[1, 10, 1536]`). For apples-to-apples comparison, you may need to tweak the rig (or the underlying `generate_greedy` path) to log all tokens, OR run the chat-pwa with diag-on and the native rig with the same single-token bias, and compare. Check the diag emit shape — if it's `[1, T, 1536]`, both are full-prompt; if one is `[1, 1, 1536]`, that's last-token-only.

4. **Diff the logs**:
   ```
   diff -u cpu-diag.log wgpu-diag.log | head -200
   ```
   Look for the FIRST `[gemma4/diag]` line where abs_max or head[] diverges materially (more than BF16 quantization noise — say, >5% relative difference). That's your divergence point.

5. **Within the divergent layer**, look at the intra-layer states (`post_input_layernorm`, `post_self_attn`, `post_mlp`, `ple/post_*`). The first intra-state that differs identifies the kernel.

6. **Find the kernel** in `candle-fork/candle-core/src/wgpu_backend/`:
   - `kernels/matmul.wgsl`, `matmul_bf16.wgsl`, `matmul_f16.wgsl` — matmul (used for q/k/v projections, mlp gate/up/down, attention q@k.T and attn@v, ple projections, lm_head)
   - `kernels/softmax.wgsl` — softmax (used post-attention)
   - `kernels/rope.wgsl` — RoPE (used on q/k before attention)
   - `kernels/cast_from_bf16.wgsl`, `cast_to_bf16.wgsl` — dtype casts (used heavily; F32 promotion in attention, restore to BF16)
   - `kernels/elementwise_unary.wgsl`, `elementwise_binary.wgsl` — broadcast ops (residual adds, scalar muls, RMSNorm divide)
   - `kernels/reduce.wgsl` — sum/mean (RMSNorm)
   - `kernels/copy.wgsl`, `copy_strided_bf16.wgsl` — strided copies, contiguous, transpose

   Host-side dispatch is in `candle-fork/candle-core/src/wgpu_backend/ops/*.rs`.

7. **Fix the kernel**, push to `Brainwires/candle@v0.10-wgpu`, bump the rev in `brainwires-framework/Cargo.toml`, run `cargo update -p candle-core -p candle-nn -p candle-transformers`, rebuild and re-test.

## Files you'll likely touch

- **candle-fork** (where the kernel bug lives):
  - `candle-core/src/wgpu_backend/kernels/*.wgsl` — kernel source
  - `candle-core/src/wgpu_backend/ops/*.rs` — host dispatch
  - `candle-transformers/src/models/gemma4/text.rs` — model code (probably no changes needed; verified correct)

- **brainwires-framework** (verification + chat-pwa rebuild):
  - `Cargo.toml` — bump candle rev
  - `crates/brainwires-provider/examples/gemma4_diag.rs` — native test rig (may need to extend to log all tokens or to CPU-pin embed weights for WGPU runs)
  - `crates/brainwires-provider/src/local_llm/vision/gemma4_mm.rs` — diag scaffold (read-only reference, no changes expected)
  - `extras/brainwires-chat-pwa/wasm/src/gemma_pipeline.rs` — chat-pwa wasm (final-test rebuild only)
  - `extras/brainwires-chat-pwa/web/build.sh` — runs `wasm-pack build` and bundles JS

## Build commands

**Native rig** (after candle-fork change + `cargo update`):
```
cargo build --release --example gemma4_diag \
    --features native,local-llm-vision,candle-wgpu \
    -p brainwires-provider
```

**Chat-pwa wasm**:
```
extras/brainwires-chat-pwa/web/build.sh
```
Outputs to `extras/brainwires-chat-pwa/web/pkg/` and bundles to `web/`. The chat-pwa is served from `https://chat.brainwires.dev` (the user's deployment) — once rebuilt locally, the user reloads the browser to pick up the new wasm. The service worker handles SRI cache eviction automatically.

**To enable diag in the chat-pwa for one debug build**, add at the top of `init_local_multimodal_chunked` in `gemma_pipeline.rs`:
```rust
brainwires_provider::local_llm::vision::gemma4_mm::set_diag_enabled(true);
```
Rebuild, reload, capture the per-step `[gemma4/diag]` lines from the worker DevTools console. **Remove this line** before final commit — chat-pwa users shouldn't pay for ~120 readbacks/step in production.

## What "done" looks like

1. Native `gemma4_diag --device wgpu --prompt "<bos><|turn>user\nHi<turn|>\n<|turn>model\n" --max-new-tokens 12` produces coherent output starting with `"Hi"` (or similar IT response).
2. Top-5 logit spread on WGPU is sharp (>10) at step 0, matching CPU.
3. Chat-pwa in browser (after rebuild + reload) responds to "Hi" with a coherent message instead of LaTeX gibberish.

The user's actual production target is the chat-pwa in browser. Native rig WGPU is the proxy because it iterates faster than the Docker stack (~30 s vs ~5 min per cycle).

## Reference: Gemma 4 architecture facts

Saved at `~/.claude/projects/-home-nightness-dev-brainwires-framework/memory/gemma4_architecture.md`. Highlights:
- Embed scale = `sqrt(hidden_size)` ≈ 39.2 for E2B (1536 hidden).
- RMSNorm: `weight * (x / rms(x))`, multiplicative.
- PLE merge: `(per_layer_projection + per_layer_inputs) * (1/√2)` in `PerLayerEmbedding::forward`.
- Layer scalar: per-layer learned `[1]` tensor multiplied at end of each decoder layer (post-PLE-residual).
- AltUp: HF gemma4 has NONE. Chat-pwa heuristic correctly resolves `altup_num_inputs=1`.
- LAuReL: low-rank residual; chat-pwa configures it from `laurel.linear_left.weight` shape detection.
- Hybrid attention: layer_types in config.json — pattern is 4 sliding then 1 full, repeating. Sliding window 512 for E2B.
- p-RoPE (proportional, partial_rotary_factor=0.25) on global layers; standard RoPE on sliding layers.
- QK-Norm: q_norm and k_norm applied before the matmul (replaces Gemma 2 logit soft-cap).
- Tied lm_head: `lm_head.weight ≡ embed_tokens.weight`.
- `final_logit_softcapping = 30.0`, `attention_k_eq_v = false` for E2B.

## Reference: HF transformers source

Authoritative reference: `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gemma4/modeling_gemma4.py`. Key lines:
- L1153: `Gemma4TextAttention.scaling = 1.0` (no pre-softmax scale)
- L1411-1421: PLE residual block + `hidden_states *= self.layer_scalar` at layer end
- L1584: `per_layer_input_scale = 2.0**-0.5`
- L1769: `(per_layer_projection + per_layer_inputs) * per_layer_input_scale`
- L779-810: `eager_attention_forward` — this is what HF actually runs

The candle-fork model code matches this contract; the disagreement is purely WGSL kernel numerical behavior vs CPU reference.

## How to ask the user for help

If you get stuck, the user (`nightness`, `viipe.com@gmail.com`) prefers:
- Concrete questions with options A/B/C, not open-ended "what should I do"
- Short responses (terse > verbose)
- Push to candle remote requires explicit user authorization — DO NOT auto-push without asking
- Use the existing `[patch.crates-io]` pattern to test candle changes locally before pushing

User's production environment: the chat-pwa is deployed at `chat.brainwires.dev`. The user runs it in Chrome and reads the worker DevTools console for the `[gemma4/diag]` and `[gemma4/perf]` lines.

Good luck.
