# Architecture: brands, repos, and the engine/harness boundary

- **Status:** Accepted (supersedes the 2026-06-29 v1, which had rullama as the engine name)
- **Date:** 2026-06-29
- **Scope:** how the Brainwires projects split into repos and brands —
  **rullama** (the app), **brainwires** (the OSS platform: engine + harness), and
  the product extractions (**brainclaw**, **brainwires-cli**). Canonical reference;
  each repo carries a short pointer back here.

## Context

Three forces shaped this:

1. **The demo outgrew the demo.** rullama's PWA (~115K LOC TS) and several
   `brainwires-framework` extras (brainwires-cli ~90K LOC, brainclaw ~42K LOC, an
   18-crate workspace) are full products living inside repos meant to showcase a
   library. They need their own homes.
2. **Brand assets point a specific way.** `rullama.com` is the strong *consumer*
   domain; the **brainwires** GitHub project has the larger *developer* following.
   So the consumer-facing app should be **rullama**, and the open-source platform
   developers build on should be **brainwires**.
3. **Fewer names to carry.** Collapsing the engine and harness under one brand
   (**brainwires**) leaves just two names to remember: the app (**rullama**) and
   the platform (**brainwires**).

This reverses an earlier draft that kept "rullama" as the engine. The domain +
following facts make the opposite assignment correct: rename the *smaller-
following* asset (the engine, off "rullama") rather than the larger one.

## Decision

> The engine handles **tokens**; the harness handles **turns**; the products sit
> **on top**. Two brands: **rullama** = the consumer product family (app + CLI),
> **brainwires** = the OSS platform.

```
rullama  (consumer product family · rullama.com)
  ├─ rullama         — the PWA (web)
  ├─ rullama-native  — desktop + mobile (.NET / Avalonia; closed-source, paid)  ──┐
  └─ rullama-cli     — the agentic CLI                                            │
                                                                    all consume   ▼
brainclaw  ──▶ brainwires   (own repo)        brainwires  (the OSS platform repo)
                                                ├─ engine  — brainwires-engine (browser/local WebGPU inference)
                                                └─ harness — agents, tools, memory, providers, RAG, MCP, A2A
```

The platform (engine + harness) is **open source**; **rullama-native** is the
**paid, closed-source** product. The PWA and CLI brands are rullama's too.

All dependency arrows point down — the graph is acyclic. "rullama" appears only
at the top, "brainwires" only below it; no name echoes across layers.

### Repo end-state

| Repo | Brand | Holds | Notes |
|---|---|---|---|
| **rullama** (this repo today) | rullama | the PWA (`web/`) + the serve/proxy parts of `rullama-devserver` | The downloadable app + future native apps. Gets `rullama.com`. Supersedes the old `brainwires-studio` and the Candle `brainwires-chat-pwa` (both retire). |
| **rullama-native** | rullama | .NET/Avalonia desktop + mobile heads + a `rust-core` C-ABI shim | **Already exists** in its own repo (Stage 1 + 2 done: chat, multimodal, tools, voice, LoRA, RAG, voice-clone, ROME). Closed-source / paid. Consumes the engine directly via C-ABI; tracks the `brainwires-engine` rename. |
| **rullama-cli** | rullama | the agentic CLI (~90K LOC, today `extras/brainwires-cli`) | `filter-repo` out + rename to `rullama-cli`; depends on `brainwires` from crates.io. A rullama-branded product. |
| **brainwires** (today `brainwires-framework`) | brainwires | engine crates (moved in, renamed off "rullama") + the 32 harness crates + slimmed extras | The OSS platform. Bigger GitHub following stays put. |
| **brainclaw** | brainwires (sub-product) | the 18-crate assistant workspace | `filter-repo` out; depends on `brainwires` from crates.io. |

### What moves where

- **Engine** — `crates/rullama` + `crates/rullama-finetune` **move into the
  brainwires repo** and are renamed **`brainwires-engine`** (the engine) /
  **`brainwires-lora`** (the local LoRA trainer). Keep them in an **isolated sub-workspace / wasm32
  target** so the framework's native tokio build doesn't pull in `wgpu`, and the
  engine's wasm build doesn't pull in the harness. Engine and harness stay
  *architecturally separate* — joined only by the `Provider` seam (below).
- **App** — this repo keeps `web/` and the PWA-serving parts of the devserver
  (Vite proxy, `/api/blob`, `/api/models`). It is rebranded **rullama (the app)**.
- **devserver splits by concern:** PWA serve/proxy → stays with the app;
  the OpenAI-compatible `/v1` engine endpoint → belongs with the engine in
  brainwires (a small `serve` bin); the `/api/cloud/*` proxy → folds into the
  harness provider layer.
- **Extras slim down.** Keep only genuine framework material — `agent-chat`
  (reference), `brainwires-web-search-agent` (example), the five small MCP-server
  wrappers (brain/memory/rag/scheduler/issues), demos (audio-demo, voice-assistant),
  `brainwires-docs`, and the published SDK libs (autonomy, proxy, wasm, billing) —
  tidied into `examples/`, `servers/`, `integrations/`, `sdks/`. **brainclaw**
  leaves for its own repo, and **brainwires-cli** leaves *and is renamed
  `rullama-cli`* (it joins the rullama product family).

## The boundary — tokens vs turns

Unchanged in substance; the engine is now branded `brainwires-engine`.

| Concern | Owner |
|---|---|
| Model load/streaming, forward pass + WGSL kernels, sampling, KV cache, tokenizer/template, LoRA, vision/audio, diffusion | **Engine** (`brainwires-engine`) |
| Multi-provider routing, agent loops, tool runtime, memory tiers, RAG, MCP, A2A, permissions/sandbox | **Harness** (`brainwires-*` crates) |
| Chat UI, conversation store, message/tool rendering, settings, voice/image surfaces | **App** (rullama) |

## Integration contracts

The harness's seam already exists — `brainwires-core::provider::Provider`
(`async chat(...) -> ChatResponse`, `stream_chat(...) -> BoxStream<StreamChunk>`,
`ChatOptions`, `ModelListing::list_models()`).

1. **Harness ↔ engine (inside brainwires):** the engine becomes a first-party
   **WebGPU provider** — a `Provider` impl wrapping `brainwires-engine`. (Browser
   path is JS-side, since the engine `Model` is browser-bound and not `Send+Sync`;
   native path hosts an engine `Model` behind `/v1`.)
2. **App ↔ platform, in-browser:** the rullama app imports the engine's **wasm
   bundle** (built from brainwires, à la `pkg/*.js`) and drives it through a
   JS provider — `renderChat → setSampling → step* → isEos`.
3. **App / any consumer ↔ platform, native:** an **OpenAI-compatible
   `/v1/chat/completions`** endpoint (engine `serve` bin in brainwires), consumed
   via the existing `openai_chat` provider with a base-URL swap. No new provider
   crate needed.
4. **rullama-native ↔ engine, in-process C-ABI:** the desktop/mobile app links
   the engine crates directly through a **C-ABI shim** (`rust-core` cdylib /
   staticlib) via P/Invoke — no HTTP, no wasm. The engine `Model` is `!Send`, so
   each handle owns one OS thread and calls are marshalled to it. This path
   currently **bypasses the harness** (tool-calling / RAG / voice live in the
   native app); it pins the published engine crates (today `rullama` /
   `rullama-finetune` v0.5 → `brainwires-engine*` after the rename).

Stable contract = the `Provider` trait + the OpenAI wire format + the
tool-call/message protocol. Engine internals may change per patch release.

## Migration phases (described, not executed)

Each step is independently shippable.

1. **Move + rename the engine** — `crates/rullama{,-finetune}` → brainwires repo as
   `brainwires-engine` + `brainwires-lora`, in an isolated wasm32 sub-workspace.
2. **Rebrand this repo to the rullama app** — keep `web/` + devserver serve/proxy;
   point it at the brainwires wasm bundle + `/v1`. Retire `brainwires-studio` and
   the Candle `brainwires-chat-pwa`.
3. **Extract brainclaw** → own repo (`git filter-repo`, depends on published
   `brainwires`).
4. **Extract brainwires-cli → own repo, renamed `rullama-cli`** (same pattern;
   joins the rullama product family).
5. **Slim brainwires extras** into `examples/ servers/ integrations/ sdks/`.
6. **Cross-repo dev loop** — publish/link the engine wasm bundle (npm) and use
   cargo path/patch overrides so app↔platform dev stays a one-command loop.

## Verification (when phases begin)

Build the WebGPU provider and run one chat round-trip through the harness against
the engine — in-browser (wasm bundle) and via `curl .../v1/chat/completions` —
confirming parity with the engine's direct output on a fixed prompt set. After the
moves, `brainwires` builds native without `wgpu`, and the engine sub-workspace
builds for wasm32 without the harness.
