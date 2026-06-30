# Architecture: brand, repos, and the engine/harness boundary

- **Status:** Accepted (single-brand revision; supersedes the two-brand
  engine/harness split docs)
- **Scope:** how the projects split across repos under one brand —
  **rullama** — with **Brainwires** as the company / GitHub-org name only.
  Canonical reference; each repo carries a short pointer back here.

## Context

Two forces shaped this:

1. **The demo outgrew the demo.** The PWA (~115K LOC TS), the CLI (~90K LOC), and
   `brainclaw` (~42K LOC, an 18-crate workspace) are full products that grew inside
   a repo meant to showcase a library. They need their own homes.
2. **One brand, one name to carry.** `rullama.com` is the consumer domain and
   "rullama" is the name the product is known by. **Brainwires** is the *company*
   (and the GitHub org) — not a project or crate prefix. So everything ships under
   **rullama**; "Brainwires" appears only in copyright, author fields, and the
   `github.com/Brainwires/…` org segment.

## Decision

> The engine handles **tokens**; the harness handles **turns**; the apps sit
> **on top**. One brand — **rullama** — across the whole stack. **Brainwires** is
> the company that publishes it.

```
rullama  (the brand · rullama.com · github.com/Brainwires)

  apps / products
  ├─ rullama         — the PWA (web)                                 ─┐
  ├─ rullama-native  — desktop + mobile (.NET/Avalonia; paid)        │
  └─ rullama-cli     — the agentic CLI                                │  all consume
                                                                      ▼
  rullama-framework  (the OSS platform repo)
  ├─ engine/  (isolated wasm32 sub-workspace)
  │   ├─ rullama-engine  — browser/local WebGPU inference
  │   └─ rullama-lora    — local LoRA trainer
  └─ harness/ — rullama-* crates: agents, tools, memory, providers, RAG, MCP, A2A

  brainclaw  (own repo, a rullama sub-product) ──▶ consumes rullama-framework
```

The platform (engine + harness) is **open source**; **rullama-native** is the
**paid, closed-source** product. All dependency arrows point down — acyclic.

### Repo end-state

| Repo | Holds | Notes |
|---|---|---|
| **rullama** | the PWA (`web/`) + the serve/proxy `dev-server` | The downloadable app + native heads. Gets `rullama.com`. |
| **rullama-native** | .NET/Avalonia desktop + mobile + a `rust-core` C-ABI shim | Already exists; closed-source / paid. Links the engine crates directly via C-ABI. |
| **rullama-cli** | the agentic CLI (~90K LOC) | Own repo; depends on the `rullama-framework` crates. |
| **rullama-framework** | engine crates + the 33 harness crates + slimmed extras | The OSS platform. Lives at `github.com/Brainwires/rullama-framework`. |
| **brainclaw** | the 18-crate assistant workspace | Own repo; depends on the `rullama-framework` crates. |

## The boundary — tokens vs turns

| Concern | Owner |
|---|---|
| Model load/streaming, forward pass + WGSL kernels, sampling, KV cache, tokenizer/template, LoRA, vision/audio, diffusion | **Engine** (`rullama-engine`) |
| Multi-provider routing, agent loops, tool runtime, memory tiers, RAG, MCP, A2A, permissions/sandbox | **Harness** (`rullama-*` crates) |
| Chat UI, conversation store, message/tool rendering, settings, voice/image surfaces | **App** (`rullama`) |

## Integration contracts

The harness's seam is `rullama_core::provider::Provider`
(`async chat(...) -> ChatResponse`, `stream_chat(...) -> BoxStream<StreamChunk>`,
`ChatOptions`, `ModelListing::list_models()`).

1. **Harness ↔ engine:** the engine is a first-party **WebGPU provider** — a
   `Provider` impl wrapping `rullama-engine`. (Browser path is JS-side, since the
   engine `Model` is browser-bound and not `Send+Sync`; native path hosts an engine
   `Model` behind `/v1`.)
2. **App ↔ platform, in-browser:** the rullama app imports the engine's **wasm
   bundle** (built from `rullama-framework/engine/rullama-lora`, served at
   `/pkg/rullama.js`) and drives it through a JS provider —
   `renderChat → setSampling → step* → isEos`.
3. **App / any consumer ↔ platform, native:** an **OpenAI-compatible
   `/v1/chat/completions`** endpoint (engine `rullama-serve` bin), consumed via the
   `openai_chat` provider with a base-URL swap.
4. **rullama-native ↔ engine, in-process C-ABI:** the desktop/mobile app links the
   engine crates directly through a **C-ABI shim** (`rust-core` cdylib/staticlib)
   via P/Invoke — no HTTP, no wasm. The engine `Model` is `!Send`, so each handle
   owns one OS thread and calls are marshalled to it.

Stable contract = the `Provider` trait + the OpenAI wire format + the
tool-call/message protocol. Engine internals may change per patch release.

## Verification

Build the WebGPU provider and run one chat round-trip through the harness against
the engine — in-browser (wasm bundle) and via `curl .../v1/chat/completions` —
confirming parity with the engine's direct output. `rullama-framework` builds
native without `wgpu`; the `engine/` sub-workspace builds for wasm32 without the
harness.
