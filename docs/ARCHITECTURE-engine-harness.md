# Architecture: Engine + Harness (rullama × brainwires-framework)

- **Status:** Accepted
- **Date:** 2026-06-29
- **Scope:** the relationship between two Brainwires projects —
  **rullama** (browser/local inference engine) and **brainwires-framework**
  (agent harness). This is the canonical reference; each repo carries a short
  pointer back here.

## Context

Two large projects grew overlapping concerns:

- **rullama** — a browser-resident **inference engine**: Rust → WASM + WebGPU,
  Gemma 4 family, hand-written WGSL kernels. It runs the forward pass on the
  local GPU and recently grew **cloud passthrough** (Ollama Cloud + OpenAI via a
  same-origin proxy with an AES-GCM key vault). It ships a mature React/TS PWA
  (`web/`) with chat, RAG/Knowledge, fine-tune, voice/TTS, and image-gen.
- **brainwires-framework** — a 32-crate Rust **agent harness**: a unified
  `Provider` trait over ~12 backends, agent orchestration
  (`brainwires-inference`), tool runtime, tiered memory, RAG, MCP client/server,
  A2A, and a WebRTC "home daemon". Its older `extras/brainwires-chat-pwa` does
  in-browser inference via **Candle** (a buggy WebGPU path) and bridges to
  agents over WebRTC.

The problem: both independently reinvented chat UI, RAG, tool-calling, and
provider routing; there are **two chat PWAs**; `brainwires-chat-pwa` runs a worse
local engine (Candle) than rullama while rullama's PWA reinvents harness concerns
the framework already owns. Nothing imports rullama from the framework today, so
the seam is undefined.

This document fixes that seam.

## Decision

> **The engine handles tokens; the harness handles turns.**

(The guiding split observed across Ollama, LM Studio, the Vercel AI SDK,
LangChain, and WebLLM — the harness owns multi-provider routing; the engine
never does.)

1. **brainwires-framework is the umbrella** (product / app layer). **rullama is
   one engine it consumes**, alongside cloud providers — a focused, reusable
   inference library, not a product in its own right.
2. **Two integration contracts, both supported:**
   - a **WASM provider adapter** for the in-browser path, and
   - an **OpenAI-compatible HTTP endpoint** for native / server / CLI consumers.
3. **One canonical UI:** rullama's React PWA. The home-daemon / A2A / MCP bridge
   is ported into it; the Candle-based `brainwires-chat-pwa` is **deprecated**.
4. **Multi-repo:** rullama and brainwires-framework stay independent repos coupled
   only by the documented contract (mirrors LM Studio's engine / SDK / app split).

## 1. The boundary — tokens vs turns

Every major concern is assigned to exactly one side. Cross-checked against both
codebases (file references are current entry points).

| Concern | Owner | Where it lives today |
|---|---|---|
| Model load / GGUF streaming | **Engine** | rullama `gguf/fetcher.rs` (`TensorFetcher`), `api::Model::load*` |
| Forward pass + WGSL kernels | **Engine** | rullama `reference/forward_chained.rs`, `kernels/wgsl/` |
| Sampling | **Engine** | rullama `sampling.rs`, `Model::setSampling` |
| KV cache + snapshot/restore | **Engine** | rullama `Model::{saveKvState,restoreKvState,truncateKv,reset}` |
| Tokenizer + chat template | **Engine** | rullama `tokenizer/`, `template/`, `Model::{encode,renderChat}` |
| LoRA train / apply | **Engine** | `rullama-finetune` (`TrainingSession`), `Model::loadAdapter` |
| Vision / audio encode | **Engine** | rullama `multimodal/`, `Model::{encodeImage,encodeAudio}` |
| Diffusion (DiffusionGemma) | **Engine** | rullama `diffusion.rs` (`DiffusionGemma`) |
| Multi-provider routing | **Harness** | `brainwires-provider` (`Provider` trait, registry, factory) |
| Agent loops / orchestration | **Harness** | `brainwires-inference` (ChatAgent/TaskAgent/CycleOrchestrator) |
| Tool runtime + builtins | **Harness** | `brainwires-tool-runtime`, `brainwires-tool-builtins` |
| Memory tiers + consolidation | **Harness** | `brainwires-memory`, `brainwires-stores` |
| RAG indexing + hybrid search | **Harness** | `brainwires-rag`, `brainwires-knowledge` |
| MCP client/server, A2A | **Harness** | `brainwires-mcp-*`, `brainwires-a2a`, `brainwires-network` |
| Permissions / sandbox | **Harness** | `brainwires-permission`, `brainwires-sandbox` |

**Cloud passthrough is a harness concern.** Today it lives in rullama
(`web/src/lib/cloud/*`, `crates/rullama-devserver/src/cloud.rs` →
`/api/cloud/{provider}/chat`). Conceptually it duplicates `brainwires-provider`'s
`openai_chat` / `ollama` providers. It is a **candidate to migrate** to the
harness provider layer (Phase 4 below) — not now. rullama keeps only the minimum
needed to stand alone as a self-contained engine demo.

## 2. The integration contract (both paths)

The harness's seam already exists — `brainwires-core::provider::Provider`:

```rust
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn max_output_tokens(&self) -> Option<u32> { None }
    async fn chat(&self, messages: &[Message], options: &ChatOptions) -> Result<ChatResponse>;
    fn stream_chat<'a>(&'a self, messages: &'a [Message], options: &'a ChatOptions)
        -> BoxStream<'a, Result<StreamChunk>>;
}
```

with `Message` / `ChatResponse` / `StreamChunk` from `brainwires-core::message`,
the `ChatOptions` builder (`temperature`, `max_tokens`, `system`, `top_p`,
`model`, `cancel_with`, …), and `ModelListing::list_models()` for catalogs.

### Path A — WASM provider adapter (in-browser)

rullama's `Model` is a `wasm-bindgen` JS class **bound to the browser GPU and not
`Send + Sync`** — so it is *not* implemented as a native Rust `Provider`. Instead
the adapter is **JS-side**, in the canonical PWA: a `RullamaProvider` object that
satisfies the same conceptual interface (`chat` / `stream_chat` / `list_models`)
by driving the WASM `Model`.

Method mapping (engine API is per-token, not a single `generate`):

| Harness concept | rullama JS `Model` calls |
|---|---|
| build prompt from `messages` + `system` | `renderChat(messages)` → token ids via `encode` |
| sampling config from `ChatOptions` | `setSampling({temperature, topP, topK, seed, …})` |
| streaming generation | loop `step(prevToken)` → next id; emit `tokenStr(id)` as a `StreamChunk` until `isEos(id)` |
| stop / cancel | `reset()` / stop flag in the worker (the existing `cancelRef` path) |
| adapters (LoRA) | `loadAdapter(bytes)` / `clearAdapter()` |
| multimodal | `encodeImage` / `encodeAudio` soft-token splice |
| model catalog | `/api/models` (existing) → `AvailableModel[]` |

This **replaces Candle** as the framework's browser engine.

### Path B — OpenAI-compatible HTTP endpoint (native / server / CLI)

rullama exposes `POST /v1/chat/completions` (+ `/v1/embeddings`) from a native
Rust host that owns an `api::Model` (the engine runs natively too — see the
`crates/rullama/examples/`). Implementation lands as a **new router module in
`crates/rullama-devserver`** (or a dedicated `rullama-serve` bin), reusing the
SSE plumbing already in `cloud.rs`. The request/response/SSE shape maps onto the
same `renderChat → setSampling → step*` loop as Path A.

**The harness needs no new native provider code for this path.**
`brainwires-provider` already ships `openai_chat` (and `ollama`) providers —
point one at rullama's base URL (à la Ollama / LM Studio / WebLLM):

```rust
// existing provider, new base URL — that's the whole integration
let rullama = OpenAiChatProvider::new(api_key_unused).with_base_url("http://localhost:PORT/v1");
```

### Stability promise

Public contract = the `Provider` trait + the OpenAI wire format + the
tool-call/message protocol. rullama's stable surface is `api` / `error` /
`sampling` / `lora` only; everything `#[doc(hidden)]` may change per patch
release (see rullama `CLAUDE.md`).

## 3. PWA convergence

**Canonical UI = rullama's `web/`.** It already has the clean seam to build on:
`web/src/hooks/useChatEngine.ts` forks `cloudTurn` vs the local AR loop, and
`web/src/lib/cloud/*` is self-contained. The refactor target (future task) is to
**generalize that fork into a pluggable provider** so local-WASM, cloud, and
harness-routed providers are peers behind one interface.

Carried over from rullama as-is: chat/RAG/voice/fine-tune/image UI, message &
tool-call rendering, conversation store, job queue, cross-tab sync.

**Ported in from `brainwires-chat-pwa`:**
- home-daemon WebRTC bridge — `web/src/home-*.js` + the Rust `home/` daemon
  (signaling, pairing, A2A JSON-RPC over DataChannel);
- MCP client + tool loop — `mcp-client.js`, `mcp-tool-loop.js`;
- any multi-cloud-provider adapters not already in rullama.

**Deprecated:** `extras/brainwires-chat-pwa` and its Candle dependency / WebGPU
gibberish bug.

## 4. Repository topology & packaging

- **Multi-repo, umbrella consumes engine.** rullama stays an independent
  engine repo; brainwires-framework is the product umbrella.
- **rullama publishing surface:** the `pkg/rullama.js` WASM bundle (built via
  `rullama-finetune` with `--out-name rullama`, exposing `Model` +
  `EmbeddingModel` + `DiffusionGemma` + `TrainingSession`) for the browser, and
  the native OpenAI endpoint for server/CLI.
- **Compatibility:** lightweight "recent" matrix in this doc, not lockstep
  versioning — the contract (trait + wire format) is the only coupling.

## 5. Migration phases (described, not executed)

Each step is independently shippable.

1. **Contract first** — add `/v1/chat/completions` to `rullama-devserver`; point
   the framework's existing `openai_chat` provider at it; prove one round-trip.
2. **Engine swap** — point a framework consumer (CLI or the surviving PWA path)
   at rullama instead of Candle.
3. **PWA convergence** — port home-daemon / A2A / MCP into rullama's `web/`;
   retire `brainwires-chat-pwa`.
4. **Cloud/provider migration** — move cloud routing from rullama's PWA/devserver
   into the harness provider layer; rullama keeps a minimal demo path.
5. **Cleanup** — dedupe RAG / tool / memory logic so the harness is the single
   source of truth.

## Verification (when phases begin)

Build the browser `RullamaProvider` and run one chat round-trip through the
harness against rullama — both in-browser (WASM `Model`) and via
`curl http://localhost:PORT/v1/chat/completions` — and confirm parity with
rullama's direct PWA output on a fixed prompt set.
