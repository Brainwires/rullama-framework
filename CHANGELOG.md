# Changelog

All notable changes to the Brainwires Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0] - 2026-05-15

### Refactored (BREAKING)

#### Home automation moved out to `future/home-automation/`

The `brainwires-hardware` crate sheds its home-automation surface. The
`homeauto` module (Matter 1.3, Zigbee 3.0, Z-Wave Plus v2, Thread 1.3.0) and
the `matter-tool` CLI moved to a standalone workspace at
`future/home-automation/` — outside the framework workspace and not
published to crates.io. The Matter / Zigbee / Z-Wave / Thread protocol code
itself is intact, but a handful of surfaces (Matter event subscription,
BLE transport on Windows, NetworkCommissioning write, full ZCL value-type
encoding) were unfinished, so the whole stack got parked rather than
shipping as `Err("not yet implemented")` returns.

- **`crates/brainwires-hardware/src/homeauto/` deleted** — all four protocol
  subtrees (62 .rs files, ~18,500 LOC) moved to
  `future/home-automation/brainwires-homeauto/`.
- **`brainwires-hardware` features dropped**: `zigbee`, `zwave`, `thread`,
  `matter`, `matter-ble`, `homeauto`, `homeauto-full`. The `full` feature no
  longer pulls in any home-auto features.
- **`extras/matter-tool/` deleted** — moved to
  `future/home-automation/matter-tool/`.
- **Workspace deps dropped from `brainwires-hardware`**: matter crypto stack
  (`mdns-sd`, `gethostname`, `p256`, `ecdsa`, `hmac`, `hkdf`, `pbkdf2`,
  `aes`, `ccm`, `der`, `pkcs8`, `sha2`, `rand_core`, `zeroize`, `hex`),
  zigbee/zwave serial (`tokio-serial`, `crc`, `bytes`), and thread
  (`reqwest`).

API breakage:
- `brainwires_hardware::homeauto::*` — gone. Use `brainwires_homeauto::*`
  from `future/home-automation/brainwires-homeauto`. Re-folding into the
  framework will happen once the unfinished pieces are closed.

#### `EmailProvider::Gmail` OAuth2 variant removed

The Gmail OAuth2 backend variant on `brainwires_tool_builtins::email::EmailProvider`
was a stub returning `Err("Gmail OAuth2 ... not yet implemented")` for send,
search, read, and list. It's gone.

- **`EmailProvider::Gmail` enum variant deleted** along with its four
  `bail!()` match arms.
- IMAP/SMTP still works against Gmail with an app password — that path is
  the supported Gmail integration.
- The separate `email::gmail_push::GmailPushHandler` (Google Cloud Pub/Sub
  webhook ingestion) is **unchanged** and continues to work. The two
  surfaces share the "Gmail" name but are independent.

API breakage:
- `EmailProvider::Gmail { client_id, client_secret, refresh_token }` — gone.
  Switch to `EmailProvider::ImapSmtp { ... }` with a Gmail app password.

#### Stack-graphs name-resolution stub removed

The `brainwires_rag::code_analysis::stack_graphs` module was scaffolded as
an empty stub in commit `d772b21` (pre-release review, 2026-03-06) only to
satisfy a dangling `pub mod` declaration that had survived a refactor — it
had no real implementation and no plan or PRD calling for one. The
`HybridRelationsProvider` already falls through to the tree-sitter
(`RepoMap`) path for actual definition/reference extraction, so the stub
contributed nothing.

- **`code_analysis::stack_graphs` module deleted**.
- **`PrecisionLevel::High` variant removed** from
  `brainwires_rag::code_analysis::types::PrecisionLevel`. The remaining
  variants are `Medium` (AST-based) and `Low` (text-based).
- **`HybridRelationsProvider::new()` signature simplified** — the
  `_enable_stack_graphs: bool` parameter is gone. Constructor is no-arg.
- **`HybridRelationsProvider::has_stack_graphs_for()` removed**.
- **`RelationsConfig::use_stack_graphs` field removed**.

API breakage:
- `HybridRelationsProvider::new(false)` → `HybridRelationsProvider::new()`.
- Matching on `PrecisionLevel::High` no longer compiles.
- `RelationsConfig { use_stack_graphs: ..., .. }` no longer compiles.

#### Low-level inference + training crates moved out to `rullama`

The framework sheds its low-level layer. `rullama` — originally written
to *replace* Candle after the earlier on-Candle attempt failed — is
becoming a small Rust workspace that hosts both the existing wgpu/WASM
Gemma 4 inference engine and the training crates pulled from here. The
framework keeps its high-level surface (agents, providers, MCP, RAG,
prompting, reasoning, cloud fine-tune) and stops carrying a model
runtime of its own.

- **`crates/brainwires-finetune-local/` deleted** — Burn 0.20-backed
  LoRA / QLoRA / DoRA fine-tuning. Moves to `rullama` as
  `rullama-finetune`, gated behind a `training` feature, native-only.
- **`crates/brainwires-training/` deleted** — from-scratch training
  placeholder. Moves to `rullama` as `rullama-training`.
- **`brainwires-provider` `local-llm-candle` feature deleted** — the
  Candle-based local LLM provider (`CandleLlmProvider`, `gguf_loader`,
  `quantized_gemma4_pipeline`, ~1,400 LOC). Rullama already covers
  Gemma 4 inference natively in wgpu, bit-exact vs Ollama, so this
  path was redundant. Consumers wanting local LLM inference depend on
  `rullama` directly.
- **`brainwires-provider` `local-llm-vision` feature deleted** —
  Gemma 3 SigLIP + Gemma 4 multimodal pipeline (~3,000 LOC). Gemma 4
  vision is already shipped in rullama (bit-identical to Ollama); the
  Gemma 3 SigLIP path is out of scope.
- **Workspace deps pruned** — the Brainwires `candle` 0.10 WebGPU
  fork, `burn` 0.20, `hf-hub`, and the candle-only `safetensors` /
  `tokenizers` / `image` pins drop from `[workspace.dependencies]`
  once `cargo tree -e normal` shows no remaining consumer.
- **`extras/brainwires-chat-pwa/wasm/` deleted** — the 944-line wasm
  bridge crate (`brainwires-chat-pwa-wasm`) existed solely to expose
  `CandleLlmProvider` to JavaScript. With the Candle path gone it has
  no remaining surface area; the chat PWA loads `rullama`'s wasm
  bundle directly.
- **`brainwires-provider`'s candle-only re-exports gone** —
  `CandleDevice`, `CandleTensor`, `CandleVarBuilder`,
  `CandleGemmaConfig`, `CandleDeviceLocation`, `WgpuDevice`,
  `WgpuStorage`, `CandleStorage`, `CandleDType`, and the `gemma4`
  re-export are removed from `brainwires_provider`'s public API.
  Along with them, the `candle-wgpu` cargo feature is gone and the
  candle-bound examples (`gemma4_diag`, `gguf_dump`) are deleted.

API breakage:
- `brainwires_finetune_local::*` — gone. Use `rullama_finetune::*`.
- `brainwires_training::*` — gone. Use `rullama_training::*`.
- `brainwires_provider::local_llm::*` (incl. `CandleLlmProvider`,
  `CandleDevice`, `CandleGemmaConfig`, vision pipelines) — gone.
  For local inference, embed `rullama` directly. A
  `Provider`-trait shim from rullama back into the framework is a
  later milestone, added only when a concrete agent flow needs one.
- `brainwires-provider`'s `local-llm-candle` and `local-llm-vision`
  cargo features — gone (along with their gated deps).
- The `brainwires` facade crate's re-exports of any of the above —
  gone. Bumps the facade as a breaking change.

`brainwires-framework` final shape after this move: zero local model
runtime, zero GPU kernels, zero autodiff. The framework is purely
about coordinating agents, calling providers (remote or
rullama-backed), tools, RAG, and policy.

#### Phase 9 — `brainwires-storage` further refinement

Cleanup pass that the original plan flagged as optional. Now done.

- **`hnsw_wasm.rs` deleted** — the in-browser HNSW index module had
  zero consumers anywhere in the workspace (the doc comment claimed
  PWA usage but no actual import existed). The `hnsw-wasm` Cargo
  feature is gone, along with the `instant-distance` and `bincode`
  optional deps that backed it.
- **`paths.rs` moved to `brainwires-core::paths`** — pure
  platform-path helpers (no internal storage deps). Used by
  `extras/brainwires-rag-server`, `extras/brainwires-issues`, and
  internally by storage's Lance backend + embeddings cache.
- **`file_context.rs` moved to `brainwires-core::file_context`** —
  `FileContextManager`, `FileContent`, `FileChunk`. CLI utility,
  cross-cutting; native-only (uses tokio::fs). brainwires-core picks
  up `sha2` as a non-wasm dep.
- **`bm25_search.rs` and `glob_utils.rs` stay in `brainwires-storage`**
  — the original plan offered "fold into brainwires-rag" but the
  audit during execution showed storage's Lance / Postgres / SurrealDB
  / Qdrant / NornicDB / Weaviate / Milvus backends all use them
  internally. Moving would create a `storage → rag` cycle. Both
  remain feature-gated under `lance-backend`.

API breakage:
- `brainwires_storage::PlatformPaths` → `brainwires_core::paths::PlatformPaths`
- `brainwires_storage::FileContextManager` (and `FileContent`, `FileChunk`)
  → `brainwires_core::file_context::*`
- `brainwires_storage::hnsw_wasm` — gone (was never a real consumer)
- `brainwires-storage`'s `hnsw-wasm` feature — gone

`brainwires-storage` final shape: substrate (StorageBackend trait,
9 backends, embeddings, BM25, glob), no cross-cutting filesystem
utilities. Coherent.

#### Phase 11g — final cleanup of agent decomposition

Cleanup pass after Phases 11a–11f:

- `pub use brainwires_core::confidence::*` shim in
  `brainwires-agent/src/lib.rs` removed. Use
  `brainwires_core::confidence::*` directly. (The shim was added in
  Phase 11a as a one-phase compat layer; all in-tree consumers were
  migrated by 11f.)
- `brainwires-agent`'s package description updated to reflect its
  new shape (coordination primitives + multi-agent patterns; no
  longer the home of mdap / seal / skills / eval / inference).
- New READMEs: `crates/brainwires-skills/`, `crates/brainwires-eval/`,
  `crates/brainwires-inference/` (mdap + seal already had them).
- New ADR: `docs/adr/ADR-0006-agent-decomposition.md` recording the
  framing change ("framework stays minimal" → "every crate has one
  cohesive responsibility") and explicitly overturning the previous
  plan's "Things deliberately not in this plan" stance on extracting
  mdap / seal / skills.
- Workspace + per-crate counts refreshed via `cargo xtask
  package-count` (32 crates, 18 direct extras subdirs).
- `README.md`: framework crate table gains 5 rows (inference, mdap,
  seal, skills, eval); agent row description rewritten.

cargo check + lint-deps + scripts/publish.sh --preflight-only all
clean. Phase 11 closes.

#### `brainwires-inference` extracted from agent (the big one)

The biggest piece of the Phase 11 agent decomposition. Every
LLM-driven workhorse moves out of `brainwires-agent` into a new
**`brainwires-inference`** crate. The principle: `brainwires-agent`
is what holds agents together (locks, queues, message bus,
coordination patterns); `brainwires-inference` is what makes them
think (LLM loops, prompts, validators, planners, judges).

Moves to `brainwires-inference/src/`:

- `chat_agent.rs`, `task_agent/` — the two main streaming-completion
  loops
- `planner_agent.rs`, `judge_agent.rs`, `validator_agent.rs`,
  `validation_agent.rs` — LLM-driven helper agents
- `cycle_orchestrator.rs`, `plan_executor.rs` — Plan→Work→Judge driver
  + plan execution
- `validation_loop.rs` — quality-check loop
- `summarization.rs` — history compaction via LLM
- `system_prompts/` — agent prompt registry
- `runtime.rs` — `AgentRuntime` + `run_agent_loop` (drives the
  inference workhorses)
- `context.rs` — `AgentContext` (owns the `AgentLifecycleHooks` trait
  object)
- `agent_hooks.rs` — `AgentLifecycleHooks` trait (references
  `TaskAgentResult`)
- `pool.rs` — `AgentPool` (TaskAgent pool, not generic)
- `task_orchestrator.rs` — `TaskOrchestrator` (TaskAgent orchestration)

`brainwires-agent` keeps coordination + patterns + schema only:
- `communication`, `task_manager` / `task_queue`
- locks: `file_locks`, `resource_locks`, `wait_queue`,
  `access_control`, `operation_tracker`
- `git_coordination`, `worktree`
- `agent_manager`, `agent_tools`, `resource_checker`,
  `execution_graph`, `otel`
- patterns: `state_model`, `contract_net`, `saga`, `optimistic`,
  `market_allocation`, `workflow`
- schema: `roles`, `personas`

New crate: `brainwires-inference` v0.11.0
- deps: brainwires-core, brainwires-agent, brainwires-call-policy,
  brainwires-tool-runtime + tokio + futures + serde + sha2 + hex +
  regex + tracing
- features: `native` (default), `wasm`, `otel`

Tests + examples moved with the inference code:
- `tests/`: validation, summarization, parallel_tools
- `examples/`: agent_pool, validation_loop, cycle_orchestrator,
  planner_judge_parsing

Facade rewires:
- New `brainwires::inference` module re-exporting `brainwires_inference::*`.
- `brainwires::agents` continues to spread both crates so existing
  `brainwires::agents::ChatAgent` / `TaskAgent` / etc. paths keep working.
- New `inference` Cargo feature on `brainwires`. Added to `default`
  features so the umbrella keeps working out-of-the-box.
- `chat` feature now implies `inference` (chat needs ChatAgent).

API breakage:
- `Cargo.toml`: add `brainwires-inference` as a direct dep if you
  reach for `ChatAgent` / `TaskAgent` / planner / judge / validator
  / cycle / plan-executor / system_prompts / agent runtime. Or pull
  the umbrella with `["inference"]` (default).
- `use brainwires_agent::{chat_agent,task_agent,planner_agent,judge_agent,validator_agent,validation_agent,validation_loop,cycle_orchestrator,plan_executor,summarization,system_prompts,runtime,context,agent_hooks,pool,task_orchestrator}::*`
  → `use brainwires_inference::*` (or per-module `brainwires_inference::<module>::*`).
- `use brainwires_agent::{ChatAgent, TaskAgent, ...}` (bare types) →
  `use brainwires_inference::*` (or via the facade
  `brainwires::agents::*` / `brainwires::inference::*`).

#### `brainwires-eval` resurrected from agent submodule

The evaluation harness moves out of `brainwires-agent::eval` into its
own crate **`brainwires-eval`** (~3.5k LOC). Step 4 of the Phase 11
agent decomposition. Eval is the simplest split — completely
self-contained, zero `brainwires-*` deps; previously was a separate
crate before being merged.

- `crates/brainwires-agent/src/eval/` → `crates/brainwires-eval/src/`
  (mod.rs becomes lib.rs).
- Eval-specific test fixtures move with the crate:
  `tests/fixtures/` + `tests/fixtures_suite.rs` → `crates/brainwires-eval/tests/`.
- New crate v0.11.0: deps async-trait, tokio (sync/rt/macros), serde,
  serde_yml (fixture parsing), anyhow, thiserror, chrono, uuid, regex,
  tracing. No internal brainwires-* deps.
- agent Cargo.toml: drop the `eval` feature + `dep:thiserror` + the
  always-on `serde_yml` dep (was eval-only after Phase 11c took skills'
  copy with it).
- agent src/lib.rs: drop `pub mod eval;`.
- The umbrella `brainwires` facade gains `brainwires-eval` dep; the
  `eval` feature now maps to `dep:brainwires-eval`.
- `extras/brainwires-cli` adds `brainwires-eval` as a direct dep.
- `extras/brainwires-autonomy`'s `eval-driven` feature swaps
  `brainwires-agent/eval` for `dep:brainwires-eval`.

API breakage:
- `Cargo.toml`: drop `brainwires-agent/eval`; add `brainwires-eval`.
- `use brainwires_agent::eval::*` → `use brainwires_eval::*`.
- `use brainwires::eval::*` continues to work (facade re-export).

No tombstone — `brainwires-eval` was never published as a separate
crate (was internal-only before the merge).

#### `brainwires-seal` resurrected from agent submodule

The Self-Evolving Agentic Learning system moves out of
`brainwires-agent::seal` into its own crate **`brainwires-seal`**
(~6k LOC). Step 3 of the Phase 11 agent decomposition.

- `crates/brainwires-agent/src/seal/` → `crates/brainwires-seal/src/`
  (mod.rs becomes lib.rs; README pulled up).
- New crate v0.11.0: deps `brainwires-core` (for `ResponseConfidence`
  + graph traits — `ResponseConfidence` was moved to core in 11a),
  `brainwires-tool-runtime` (learning loop's outcome categorization),
  plus the LanceDB stack (`brainwires-storage` + `arrow-array` +
  `arrow-schema` + `lancedb`) always-on for the pattern store, plus
  futures, regex, async-trait, tokio.
- Optional features (renamed from agent's `seal-*` to bare names):
  - `knowledge` (was `seal-knowledge`) — pulls `brainwires-knowledge`
  - `feedback` (was `seal-feedback`) — pulls `brainwires-permission`
  - `mdap` (was `seal-mdap`) — pulls `brainwires-mdap`
- agent Cargo.toml: drops `seal`, `seal-mdap`, `seal-feedback`,
  `seal-knowledge` features; drops the deps that backed them
  (`brainwires-knowledge`, `brainwires-permission`,
  `brainwires-storage`, `arrow-array`, `arrow-schema`, `lancedb`,
  `brainwires-mdap`).
- The umbrella `brainwires` facade gains a `brainwires-seal` dep;
  `seal` feature now maps to `dep:brainwires-seal`. The `learning`
  preset rewrites to `brainwires-seal/knowledge` +
  `brainwires-seal/feedback`. The `full` preset's
  `brainwires-agent/seal-*` entries become `brainwires-seal/*`.
- `extras/brainwires-cli` adds `brainwires-seal` as a direct dep
  (was using `brainwires-agent::seal::pattern_store::*`); the CLI's
  `crate::storage::mod.rs` re-export points at `brainwires_seal`.

API breakage:
- `Cargo.toml`: drop `brainwires-agent/seal*` features; add
  `brainwires-seal` (with optional `knowledge` / `feedback` /
  `mdap` features).
- `use brainwires_agent::seal::*` → `use brainwires_seal::*`.
- `use brainwires::seal::*` continues to work (facade re-export).
- `brainwires-agent`'s `seal-feedback` / `seal-knowledge` /
  `seal-mdap` features → `brainwires-seal/feedback` /
  `brainwires-seal/knowledge` / `brainwires-seal/mdap`.

The 0.4.x deprecation tombstone in `deprecated/brainwires-seal/` was
removed; the name is reclaimed.

Drive-by: cleaned up `brainwires-finetune`'s dead `local` feature.
The local-PEFT code was extracted to `brainwires-finetune-local` in
Phase 7b but the feature gate, stub `crate::local::*` imports in
`manager.rs`, and orphaned burn / safetensors deps were left behind.
Removed the feature, the dead imports, and the unused deps.
`brainwires-finetune` is now cloud-only as advertised; consumers
already wire `brainwires-finetune-local` directly for local training.

#### `brainwires-skills` resurrected from agent submodule

The SKILL.md skills system moves out of `brainwires-agent::skills`
into its own crate **`brainwires-skills`** (~5k LOC). Step 2 of the
Phase 11 agent decomposition.

- `crates/brainwires-agent/src/skills/` → `crates/brainwires-skills/src/`
  (mod.rs becomes lib.rs).
- New crate v0.11.0: deps `brainwires-core` + `brainwires-tool-runtime`
  (for `Tool` / `ToolContext` / `ToolExecutor`), plus serde_yml,
  semver, regex, sha2, async-trait, tokio. Optional features:
  `signing` (ed25519 manifest verification), `registry` (HTTP client).
- agent's `skills-registry` and `skills-signing` features removed;
  the deps that backed them (reqwest, ed25519-dalek, rand_core, semver)
  moved with the crate.
- The umbrella `brainwires` facade gains `brainwires-skills` dep; the
  `skills` feature now maps to `brainwires-skills/registry` (was:
  `brainwires-agent/skills-registry`).
- `extras/brainwires-cli` adds `brainwires-skills = { features = ["registry"] }`
  as a direct dep; drops the `skills-registry` feature from
  `brainwires-agent`.

API breakage:
- `Cargo.toml`: drop `brainwires-agent/skills-registry`; add
  `brainwires-skills = { features = ["registry"] }` (or
  `["signing"]`).
- `use brainwires_agent::skills::*` → `use brainwires_skills::*`.
- `use brainwires::skills::*` continues to work (facade re-export).

The 0.8.x deprecation tombstone in `deprecated/brainwires-skills/`
was removed; the name is reclaimed.

#### `brainwires-mdap` resurrected from agent submodule

The Multi-Dimensional Adaptive Planning (MAKER voting) framework
moves out of `brainwires-agent::mdap` into its own crate
**`brainwires-mdap`**. The submodule had zero internal dependencies on
other agent code — cleanest possible split. Step 1 of the Phase 11
agent decomposition.

- `crates/brainwires-agent/src/mdap/` → `crates/brainwires-mdap/src/`
  (mod.rs becomes lib.rs).
- The `voting_consensus` and `task_decomposition` examples move to
  `crates/brainwires-mdap/examples/`.
- `brainwires-agent`'s `mdap` feature is gone. The `seal-mdap` feature
  now pulls `brainwires-mdap` as an optional dep instead of gating an
  internal submodule.
- The umbrella `brainwires` facade gains a `brainwires-mdap` dep; the
  `mdap` feature now maps directly to it (was: `agents` + `brainwires-agent/mdap`).
- `extras/brainwires-wasm` swaps `brainwires-agent/mdap` for a direct
  `brainwires-mdap` dep + re-export.
- `extras/brainwires-autonomy`'s `parallel` feature swaps the same way.
- `extras/brainwires-cli` adds `brainwires-mdap` as a direct dep.

API breakage:

- `Cargo.toml`: `brainwires-agent = { features = ["mdap"] }` → drop
  the feature; add `brainwires-mdap` as a separate dep.
- `use brainwires_agent::mdap::*` → `use brainwires_mdap::*`.
- `use brainwires::mdap::*` continues to work (facade re-export).

The 0.4.x deprecation tombstone in `deprecated/brainwires-mdap/` was
removed — the name is reclaimed for the new active crate.

#### `ResponseConfidence` moved to `brainwires-core`

Prep step for the Phase 11 agent decomposition. `ResponseConfidence`
+ `ConfidenceFactors` + `extract_confidence` + `quick_confidence_check`
moved from `brainwires-agent` into `brainwires-core::confidence`. The
type is the only cross-domain piece shared between agent runtime and
the (about-to-be-extracted) `brainwires-seal` learning loop; promoting
it to core lets seal extract cleanly without depending on agent.

A `pub use brainwires_core::confidence::*;` shim in
`brainwires-agent/src/lib.rs` keeps existing `brainwires_agent::ResponseConfidence`
imports working through Phase 11. The shim is removed in Phase 11g
(final cleanup).

Migration:
- New code: `use brainwires_core::confidence::ResponseConfidence;`
- Existing code: continues to work via the shim until Phase 11g.

#### Pre-release dead-code sweep

Two orphan code paths uncovered by a `cargo check --workspace --all-targets`
sweep against `--all-targets` were deleted before tagging, alongside a
handful of stale `use` paths in test/example targets that pointed at modules
which had been flattened or relocated in this cycle.

- **`crates/brainwires-llama` deleted.** Introduced in `0cdb650` as a port
  from the `rullama` prototype while debugging Q4_K WGPU drift in the
  chat-PWA; the actual chat-PWA wasm crate that consumed it lives outside
  this repo and now points at `rullama` directly. The in-framework copy was
  `publish = false`, never published to crates.io, and had no workspace
  reverse deps (`cargo tree -i` confirmed). Removing it (~7,300 LOC across
  65 files) is invisible to any consumer.
- **`brainwires-network` MCP-server scaffolding deleted** —
  `src/{connection,handler,registry,server,mcp_error,mcp_transport}.rs`,
  `src/middleware/`, plus `tests/{tool_registry,middleware_pipeline}.rs` and
  `examples/{mcp_server,middleware_chain}.rs`. These ~2,000 LOC were
  superseded by the standalone `brainwires-mcp-server` crate (split out in
  `b133a4b`) and never re-declared in `brainwires-network/src/lib.rs`, so
  the symbols (`McpServer`, `McpToolRegistry`, `AuthMiddleware`,
  `RequestContext`, etc.) had no public API surface to begin with.
- **Stale test/example imports fixed** in:
  - `crates/brainwires-eval/src/{regression,fault_report,stability_tests,suite}.rs`
    — drop the stale `eval::` prefix (`crate::eval::X` → `crate::X`) left
    over from a flattened module structure.
  - `crates/brainwires-seal/src/{learning,reflection,pattern_store}.rs` —
    same `seal::` prefix flattening, plus `crate::storage::VectorDatabase`
    → `brainwires_storage::databases::VectorDatabase` and
    `crate::utils::entity_extraction::EntityType` →
    `brainwires_core::graph::EntityType`.
  - `crates/brainwires-inference/{tests,examples}` — agents
    (`ChatAgent`, `PlannerAgent`, `JudgeAgent`, `JudgeVerdict`,
    `CycleOrchestrator`, `AgentPool`, `TaskAgentConfig`, etc.) moved from
    `brainwires-agent` into `brainwires-inference` in `f2eea0c`-era
    refactors; the tests/examples still imported from the old crate.
  - `crates/brainwires-inference/src/validation_agent.rs` — `ResourceLockManager`
    is in `brainwires-agent::resource_locks`, not `brainwires-inference`.
  - `crates/brainwires/tests/reexports.rs` — `TieredMemory` is re-exported
    via `brainwires::memory` (not `brainwires::storage`) and is gated by
    the `tiered` feature.
  - `extras/brainwires-cli/examples/{plan_templates,message_store,lock_coordination}.rs`
    — `PlanTemplate`, `TemplateStore`, `MessageStore`, `LockStore` moved
    from `brainwires-storage` to `brainwires-stores`.
- **`brainwires-skills` + `brainwires-inference` dev-deps:** add
  `brainwires-tool-builtins` as a `[dev-dependencies]` entry on both. Their
  test modules already used `BuiltinToolExecutor` but the crate wasn't
  declared. Published surface unchanged (dev-only deps).

After this sweep `cargo check --workspace --all-targets` is exit 0, with
warnings only.

### Added

#### chat-pwa — Phase 5 perf path live end-to-end

The user-visible Phase 5 path now runs in the chat-pwa:

- **`Gemma4QuantizedTextOnly` pipeline** —
  `crates/brainwires-provider/src/local_llm/quantized_gemma4_pipeline.rs`.
  Wraps `quantized_gemma4::ModelWeights` + a `tokenizers::Tokenizer`
  with a greedy decode loop: prefill the prompt → KvCache fills →
  step-by-step token emit with `seqlen_offset = prompt_len + step`.
  KvCache reset at the start of each generate.
- **`init_local_multimodal_gguf_quantized` wasm entry** + new
  `LocalQuantizedHandle` type. Loads via
  `gguf_loader::load_quantized_gemma4_from_reader` (keeps weights
  as `QTensor` end-to-end; QMatMul matmul calls hit PR #3379's
  `q4_k.pwgsl` on WGPU and CPU dequant-on-fly elsewhere). Vision /
  audio getters always return false.
- **`local_chat_stream_quantized` wasm entry** — text streaming
  over a `LocalQuantizedHandle`. Renders messages into the canonical
  Gemma 4 chat-template prompt (`<|turn>{role}\n{text}<turn|>\n`)
  and drives `Gemma4QuantizedTextOnly::generate_greedy_streaming`.
  NDJSON `VisionWireChunk` framing matches the BF16 path so the
  JS-side reader is unchanged.
- **`local-worker.js` routing** — `handleChat` checks a new
  `handleIsQuantized` flag and routes to `local_chat_stream_quantized`
  when the loaded handle is quantized. Load path picks
  `init_local_multimodal_gguf_quantized` over the BF16
  dequant-at-load fallback when the wasm crate exposes it.
  `handleVisionChat` fail-fasts with a clear error on quantized
  handles (Ollama GGUF is text-only).
- **`gemma4_diag --quantized`** — native CLI exercise of the
  quantized path: load GGUF, encode prompt, run one forward,
  argmax + decode, print predicted next token.

End-to-end: open settings → download gemma4:e2b → load → chat. The
forward pass runs on the `q4_k.pwgsl` quantized matmul kernel.
Reference correctness against an actual Ollama-published gemma4:e2b
GGUF is the remaining validation step before the path can be made
the default route — tensor name conventions for AltUp / PLE /
Laurel weights are llama.cpp-style guesses and may need adjustment.

#### chat-pwa — quantized_gemma4 auxiliary towers (KV-share, Laurel, layer_scalar, sparsity, PLE, AltUp)

The basic quantized_gemma4 decoder shipped earlier landed with the
auxiliary towers gated off (output wouldn't bit-match the BF16 model).
This round finishes the Gemma 4 / Gemma 3n auxiliary stack:

- **KV-share** — receivers (layers 15..34 on E2B) skip their own
  k_proj / v_proj / k_norm and read post-cache `(k, v)` from a
  donor's entry in a per-step `SharedKvStore`. Donors stash after
  their own KvCache append; donor selection follows
  `Gemma4TextConfig::donor_layer_idx_for(layer_idx)` (last
  same-`is_sliding` layer before `first_kv_shared_layer_idx`). Q
  always rotates through THIS layer's RoPE table either way.
- **LaurelBlock** — low-rank residual augmentation (linear_left →
  linear_right → post_laurel_norm) merged into the attention output
  via `(attn + laurel) * (1/√2)`. Construction is fault-tolerant:
  publications without the laurel weights silently fall back.
- **layer_scalar** — single `[1]` f32 tensor at
  `blk.{i}.layer_scalar`, applied as `xs *= scalar` at the very end
  of `DecoderLayer::forward`. Required on E2B; without it the
  residual stream `abs_max` runs away.
- **activation sparsity** in `MLP::forward` — Gaussian-topk threshold
  (`mean + std * z` ReLU) on the gate before the activation. `z =
  sqrt(2) * erfinv(2p-1)` computed once at construction via a
  Winitzki erfinv approximation.
- **PLE side-channel** — `PerLayerEmbedding` computes a
  `[B, T, num_layers, hidden_per_layer]` table once per step from
  `inputs_embeds` (and optionally `embed_tokens_per_layer`); each
  `DecoderLayer` slices its layer index out and applies
  `gate(h) → act → * per_layer_input → projection → norm → +residual`
  between the MLP residual and the layer_scalar multiply.
- **AltUp** (Alternating Updates) — multi-stream forward with
  `altup_num_inputs` parallel hidden streams. Top-level
  `altup_projections` build the stack from `inputs_embeds`; each
  layer runs `predict → activate (attn+laurel+MLP on the active
  stream) → correct over the full stack`; top-level
  `altup_unembed_projections` collapse the stack back before the
  final RmsNorm + lm_head. PLE delta in this mode is applied to
  `corrected[i != active_idx]`. Falls back to the classic
  single-stream path when the GGUF doesn't carry the AltUp tensors.

The remaining validation step is reference correctness against an
actual Ollama-published gemma4:e2b GGUF — tensor name conventions
(especially for AltUp `blk.{i}.altup_*.weight` and PLE
`blk.{i}.per_layer_*`) follow llama.cpp-style guesses that may need
adjustment when the first real GGUF is loaded.

`gemma4_diag --quantized --gguf-path <file> --tokenizer-file <file>`
exercises the path end-to-end: builds the quantized model, encodes
the prompt, runs one forward, prints the predicted next token.

#### chat-pwa — quantized_gemma4 model + chunked GPU upload + AMD adapter preset

Three coordinated upgrades that together unlock real Phase 5 perf
on the Ollama-format path:

- **`candle-transformers/src/models/quantized_gemma4.rs`** — basic
  decoder ported with `Linear → QMatMul`. Reads weights directly via
  `gguf_file::Content::tensor` so the entire matmul path stays
  quantized; PR #3379's `q4_k.pwgsl` / `q5_k.pwgsl` / `q6_k.pwgsl` /
  `q8_k.pwgsl` WGPU kernels (and CPU dequant-on-fly elsewhere) carry
  the inference workload. Mirrors `quantized_gemma3.rs` structurally
  with Gemma 4 specifics: GQA self-attention with q_norm/k_norm/
  v_norm, p-RoPE (partial 25%) on full layers + standard RoPE on
  sliding, SwiGLU MLP, KvCache (Normal for full / Rotating for
  sliding), input + post-attention + pre-feedforward + post-feedforward
  RmsNorms. **Auxiliary towers gated off** — PLE, AltUp, Laurel,
  KV-share donor/receiver, layer_scalar, activation sparsity. Output
  won't bit-match a canonical Gemma 4 forward pass with these off;
  reference verification against an Ollama-published gemma4:e2b GGUF
  is the remaining work to enable them.
- **`brainwires_provider::local_llm::gguf_loader::load_quantized_gemma4_from_reader`** —
  parallel path to the existing `load_gemma4_gguf_from_reader` (which
  dequant-at-loads to BF16). The new path keeps weights as QTensor
  end-to-end and constructs the new `quantized_gemma4::ModelWeights`.
  This is the path that actually unlocks the projected ~3-4× decode
  speedup on chat-pwa once a `quantized_gemma4` UI route lands.
- **`wgpu_compute_layer::WgpuDevice::alloc_uninit_storage_eager` +
  `write_to_storage_at`** — restores the chunked-upload pattern PR
  #3379 dropped. The chat-pwa wasm `load_tensor_to_gpu` rewires to
  these two APIs so a 805 MB `embed_tokens.weight` uploads in 64 MiB
  chunks with bounded peak wasm linear memory, instead of needing
  the entire tensor in linear memory at once.
- **`WgpuDevice::is_amd_adapter` / `is_nvidia_adapter` /
  `is_apple_adapter`** — adapter-info-based predicates over PCI
  vendor id + adapter name (the WebGPU fallback). AMD adapters
  default to `MatmulAlgorithm::Matmul64_64_8_8` (matches GCN/RDNA
  wave-64 boundaries) instead of the auto-select `MatmulX`. NVIDIA
  and Apple keep auto-select. First Phase 6 piece — without real AMD
  hardware to profile against, the routing infrastructure is in place
  but the actual best preset awaits real measurements.

#### chat-pwa — Ollama-format end-to-end load (Phase 4 part 3)

Wired the dequantize-at-load GGUF path through every layer:

- `crates/brainwires-provider/src/local_llm/gguf_loader.rs` — native
  + wasm GGUF reader. `gguf_to_hf_name()` translates GGUF tensor
  names (`blk.0.attn_q.weight`) to the HF safetensors keys the
  existing `Gemma4Model` consumer expects.
  `build_gemma4_config_from_gguf()` reads the kv-store and falls back
  to canonical Gemma 4 E2B values for missing optional keys (Ollama
  GGUFs don't always carry the full HF schema). AltUp / Laurel /
  per-layer-input-gate default-off until the Ollama publication's
  metadata schema is verified. `load_gemma4_gguf_from_reader()` is
  Read+Seek-generic so the wasm side wraps a `Cursor<Vec<u8>>` over
  the OPFS blob and reuses the same code path.
- `cargo run --example gemma4_diag -- --gguf-path <file>` exercises
  the loader end-to-end on native — bypasses the HF safetensors
  download entirely. Tokenizer still requires `--tokenizer-file`
  until GGUF tokenizer extraction lands.
- `init_local_multimodal_gguf(weights, tokenizer, model_id)` is the
  new wasm entry point. Builds a `Gemma4MultiModal` pipeline with
  vision + audio disabled (Ollama gemma4:e2b is text-only).
- `local-worker.js` recognizes `KNOWN_OLLAMA_MODELS` ids and routes
  them to the new entry point. `isDownloaded` and `getModelBytes`
  delegate to `ollama-download.js`'s OPFS reader for ollama-source
  models; HF-source flow unchanged.
- `boot.js` orphan-prune skips the `model-downloads/ollama/` subtree
  so the per-id scheme doesn't recursively wipe Ollama blobs.

**Perf gain:** none yet. The GGUF Q4_K_M weights are dequantized to
BF16 at load, so VRAM/RAM matches the safetensors path. The win is
download size (~1.6 GB vs ~10 GB). Inference tok/s becomes a function
of the BF16 path. Phase 5's `q4_k.pwgsl` kernel becomes reachable
once we add a `quantized_gemma4` model that consumes `QTensor` /
`QMatMul` directly — separate work.

#### chat-pwa — candle rebase to v0.11-wgpu (Phase 1)

Rebased the candle fork onto upstream PR #3379 (KimHenrikOtte's full
WGPU backend) as a fresh `v0.11-wgpu` branch. PR #3379 ships a
substantially more complete WGPU backend than our incremental
`v0.10-wgpu`:

- Multiple matmul variants tuned per shape: `matmul1x32_32b`,
  `matmul1x64`, `matmul1x64_32b`, `matmul24x24`, `matmul24x48`,
  `matmul32x32`, `matmul32x64`, `matmul64x64`, `matmul64x64_4x8`,
  `matmul64x64_8x8`. M=1 specialization means decode-path
  projections (q/k/v/o, gate/up/down) hit a kernel optimized for
  one query token instead of the generic tile path.
- `rms_norm.pwgsl` collapses 4-5 dispatches/norm into 1.
- `q4_k.pwgsl`, `q5_k.pwgsl`, `q6_k.pwgsl`, `q8_k.pwgsl` quantized
  matmul kernels — Phase 5's "WGPU Q4_K_M dequant matmul WGSL"
  is shipped as part of PR #3379, not from-scratch work for us.
- Full `QStorage::Wgpu` path including `quantize`, `quantize_onto`,
  `dequantize`, and `quantize_imatrix` ops.
- `flush_gpu_command` accumulates all dispatches in a single
  `command_queue`, flushes once per token through one encoder /
  one compute pass — Phase 3's "dispatch batching / encoder
  reuse" is also shipped as part of PR #3379.

What we ported forward on top of PR #3379:
- The complete `candle-transformers/src/models/gemma4/` directory
  (text.rs ~1850 LOC, config.rs ~550 LOC, plus mod / vision /
  audio / multimodal_embedding) — PR #3379 predates Gemma 4 being
  merged to candle main so the model code wasn't there.
- `candle-nn/src/rotary_emb.rs` cos/sin `to_dtype(xs.dtype())`
  coercion in `rope`, `rope_i`, `rope_thd` — required for Gemma 4
  receiver attention where the RoPE table is f32 but receiver
  hidden states are bf16.
- `candle-core/src/cpu_backend/mod.rs` bf16 matmul via transient
  f32 promotion — `gemm` 0.19 ships no bf16 specialization, so
  mixed-device flows (wgpu wasm32 readback landing on CPU as bf16
  at lm_head) trip the generic Map2 path every token.
- `candle-transformers/src/models/gemma3.rs` `value_states.contiguous()`
  before `KvCache::append`.

Plus three small wgpu 28→29 API deltas to keep our Rust 1.91 MSRV
(PR #3379 pinned wgpu 28.0.0 which requires Rust 1.92):
- `Instance::new()` takes `InstanceDescriptor` by value
- `PipelineLayoutDescriptor.bind_group_layouts: &[Option<&BGL>]`
- `ShaderRuntimeChecks` gained `mesh_shader_primitive_indices_clamp`
  and `task_shader_dispatch_tracking` fields

Verified: `gemma4_diag --device cpu --target-layer 15 --load-ple-table`
still produces `decoded="Hi"` with clean intra-self_attn checkpoints
(zero NaN / zero Inf at every probe). Branch is at
`Brainwires/candle@v0.11-wgpu`, framework pins
`https://github.com/Brainwires/candle?rev=acda3dbf`.

#### chat-pwa — SwiGLU gate/up fuse (Phase 7)

Concatenate `gate_proj` and `up_proj` weights along the output axis
at construction time, run one fused matmul, narrow the result halfway
along the last dim into `(gate, up)`. Saves one matmul dispatch per
FFN × 35 layers = 35 fewer dispatches per token. Compute work is
unchanged — same total FMA count — but on chat-pwa where dispatch
overhead dominates, the dispatch reduction is ~10%. Pattern from
candle PR #3485. Sparsity (Gaussian-topk gating) still applies to
the `gate` slice only — gating happens post-narrow so the fuse is
transparent to the activation pipeline. Verified bit-identical on
gemma4_diag CPU smoke (`next_id=10979`, `top5[0]=10979@0.848`
unchanged).

#### chat-pwa — Ollama-format model download (Phase 4 / part 1)

First slice of the chat-pwa local-inference perf overhaul plan: pull
pre-quantized Gemma 4 GGUF blobs (~1.6GB Q4_K_M) directly from
`registry.ollama.ai` instead of fetching the BF16 safetensors variant
from HuggingFace (~10GB). ~6× smaller download, same model.

What landed:
- `extras/brainwires-chat-pwa/web/src/ollama-fetch.js` — OCI Distribution
  Spec client. `fetchManifest(name, tag)`, `fetchBlob(name, digest, opts)`,
  `manifestToFiles(manifest)`. No auth required for the public registry.
  Library namespace defaulted (`gemma4` → `library/gemma4`); user-published
  models with explicit slashes pass through.
- `extras/brainwires-chat-pwa/web/src/ollama-download.js` — single-path
  downloader using OPFS `FileSystemSyncAccessHandle`. Resume via Range
  header, SHA-256 verification per blob (Web Crypto), `.verified` markers
  for resume short-circuiting, progress events on the same
  `model_progress` channel as the HF path so the UI banner picks them
  up unchanged. Kept separate from the existing 3-fallback HF download
  orchestration so regressions there couldn't break the working chat-pwa
  for everyone.
- `extras/brainwires-chat-pwa/web/src/model-store.js` — adds
  `KNOWN_OLLAMA_MODELS` registry with `gemma4:e2b` entry. `source: 'hf'`
  / `source: 'ollama'` discriminator. New helpers: `getKnownModelAny`,
  `listAllChatModels`. Re-exports the Ollama download API so callers
  (UI, worker) have a single import point.

Phase 4 follow-ups (separate commits):
- wasm-side GGUF loader: parse via candle's existing
  `quantized::gguf_file::Content` (already wasm32-compatible),
  dequantize Q4_K_M → BF16 at load time, feed into the existing
  `gemma4/text.rs` model. The candle-fork's WGPU backend currently
  rejects quantized `from_data` (`quantized/mod.rs:128-130` —
  "wgpu: quantized from_data not yet implemented"), so quantized
  inference on WGPU has to wait for Phase 5 (Q4_K_M dequant matmul
  WGSL kernel). Phase 4 alone wins download size, not tok/s.
- chat-pwa UI: model dropdown lists both `gemma-4-e2b-it` (HF) and
  `gemma4:e2b` (Ollama).
- GGUF tokenizer + chat-template extraction (replace today's hardcoded
  per-model template).
- Native-only `~/.ollama/models` opportunistic read for the CLI / agent
  paths (skips the network round-trip if Ollama is installed locally).

### Fixed

#### chat-pwa local Gemma 4 — receiver-attention divergence on AMD/Vulkan WebGPU

The on-device Gemma 4 E2B IT path produced LaTeX-prefix gibberish on
AMD GCN-4 + Linux/Vulkan WebGPU while the same model on Mac/Metal
generated correct output. Root cause: `extras/brainwires-chat-pwa/wasm`
hardcoded `num_kv_shared_layers: 10` (a Gemma 3n carry-over) when
building the `Gemma4TextConfig` from the safetensors layout. Real
Gemma 4 E2B is `20`. With `10`, `first_kv_shared_layer_idx` became
`25` instead of `15`, so layers 15-24 silently took the **donor**
branch in candle's gemma4 attention forward, ran their own `k_proj`
against the (receiver-shape) placeholder weights left in the
safetensors, and produced nonsense KV. The native `gemma4_diag`
binary parsed `config.json` directly and got `20`, which is why the
divergence reproduced only in the wasm path.

Fix: derive `num_kv_shared_layers` from the inferred
`num_hidden_layers`:
- 35 layers (Gemma 4 E2B / E4B) → 20 shared (donors 0..14)
- 30 layers (Gemma 3n E2B)      → 10 shared (donors 0..19)
- other layouts                 → 0 (KV-share off)

Bisected via new intra-`Attention::forward` `nan_scan` checkpoints
(`q_after_proj_reshape`, `q_after_qnorm`, `k_full_from_donor` /
`k_full_from_proj`, `q_after_rope`, `k/v_after_repeat_kv`,
`attn_weights_pre_mask`, `attn_weights_post_mask`,
`attn_weights_post_softmax`, `attn_after_v_matmul`) added to
`Brainwires/candle@v0.10-wgpu` and gated on `target_intra_layer`.
With the config fix in place, every sub-step at layer 15 now matches
Mac/Metal bit-for-bit on chat-pwa, and the model produces the
expected `"Hi! How can I help"` continuation.

Two defensive cleanups landed alongside in the candle fork (kept
because they harden the receiver path regardless of where the bug
turned out to be):
- `.contiguous()` on the donor's post-cache-append `(k_full, v_full)`
  before publishing into `shared_kv_store` — `cache.append` can
  return strided views; downstream `repeat_kv` / `matmul` produce
  backend-specific results on strided sources where Metal happens to
  tolerate them. Mirrors upstream candle PR #3475 / #3325 at the
  shared-KV publication boundary.
- `cos`/`sin` `to_dtype(xs.dtype())` coercion at the top of
  `candle_nn::rotary_emb::{rope, rope_i, rope_thd}`. Mirrors upstream
  PR #3488. Defensive against future configurations that store
  position tables in F32 while the model dtype is BF16/F16.

Affected commits:
- `Brainwires/candle@596ba2ab` — intra-self_attn diag checkpoints
- `Brainwires/candle@e17c22dd` — donor contiguify + RoPE dtype coerce
- `brainwires-framework@dca60315` — chat-pwa wasm config fix

## [0.11.0] — 2026-05-02

The "rename and split" release. Closes the deprecated/ god-crate
re-merge cycle: every plural crate name was singularized, the
`brainwires-tools` god-crate was split into runtime + builtins, the
`brainwires-knowledge` god-crate was split into knowledge + rag +
prompting, the `brainwires-providers` god-crate had speech split out,
and `brainwires-training` got renamed to `brainwires-finetune` because
that's what it actually did. Two abstract names were rewritten to
describe their content (`mcp` → `mcp-client`, `resilience` →
`call-policy`).

No re-export shims. Every retired name has a 0.10.1 deprecation
tombstone published to crates.io as a migration marker. Workspace
version bumped to 0.11.0.

Pre-1.0 hygiene pass: remove backwards-compat shims, close feature-flag half-wires, fix documentation and publish-readiness gaps.

### Refactored (BREAKING)

#### Three-layer storage refactor: `brainwires-stores` (schema) + `brainwires-memory` (orchestration) + relocations

The framework now has a clean three-layer storage architecture:

- **`brainwires-storage`** — substrate (`StorageBackend` trait,
  backends, embeddings, BM25, file-context, paths, image-types).
  Unchanged.
- **`brainwires-stores` (new)** — opinionated minimum **schema + CRUD**
  set: `SessionStore`, `ConversationStore`, `TaskStore` /
  `AgentStateStore`, `PlanStore`, `TemplateStore`, `LockStore`,
  `ImageStore`, plus the five tier-schema stores (`MessageStore`,
  `SummaryStore`, `FactStore`, `MentalModelStore`, `TierMetadataStore`)
  and shared `tier_types` (`MemoryTier`, `MemoryAuthority`,
  `TierMetadata`, `MessageSummary`, `KeyFact`, `FactType`). All built
  on the `StorageBackend` trait. Default features: `session`, `task`,
  `plan`, `conversation`. Opt-in: `memory`, `lock`, `image`, `sqlite`.
- **`brainwires-memory`** — kept (revived after a brief Phase-10a
  fold-in) as the orchestration layer. Owns `TieredMemory`
  (multi-factor adaptive search across tiers + promotion / demotion),
  `CanonicalWriteToken` (canonical-write capability gate),
  `MultiFactorScore`, `TieredSearchResult`, and the offline `dream`
  consolidation engine (summarisation, fact extraction, tier demotion)
  behind the `dream` feature. Depends on `brainwires-stores` for the
  schema types.

The old `brainwires-session` is folded as the `session` feature of
`brainwires-stores`. Neither it nor `brainwires-memory` was ever
published — no tombstones.

CLI-domain stores from `extras/brainwires-cli/src/storage/` were
relocated:

- **8 framework-clean stores** (`ConversationStore`, `TaskStore` /
  `AgentStateStore`, `PlanStore`, `TemplateStore`, `LockStore`,
  `ImageStore`) → `brainwires-stores`.
- **`PatternStore` + `LanceDatabaseExt`** → `brainwires-agent::seal::pattern_store`
  (couple to SEAL's `QueryPattern` / `QuestionType`; live next to their
  types). The `seal` feature now pulls `brainwires-storage` +
  `arrow-array` + `arrow-schema` + `lancedb` as gated deps.
- **`PlanModeStore`** → `extras/brainwires-cli/src/plan_mode_store.rs`
  (CLI-internal; couples to `crate::types::message::Message`,
  `crate::types::plan_mode::PlanModeState`, `DisplayMessage`).
- **`PersistentTaskManager`** → `extras/brainwires-cli/src/persistent_task_manager.rs`
  (CLI-local helper wrapping `brainwires-agent::task_manager::TaskManager`
  with `TaskStore` persistence; zero in-tree consumers).

`extras/brainwires-cli/src/storage/mod.rs` is kept as a thin
re-export aggregator so the 29 CLI files using
`crate::storage::{...}` don't need import rewrites — that shim is a
candidate for deletion in Phase 10c.

- **`brainwires-stores` (new)** — opinionated minimum store set.
  Default features: `session`, `task`, `plan`, `conversation`. Opt-in:
  `memory`, `tiered`, `dream`, `lock`, `image`, `sqlite`. All built on
  the `brainwires-storage` `StorageBackend` trait. Phase 10b will pull
  the remaining task / plan / conversation / lock / image / template
  stores up from `extras/brainwires-cli/`; Phase 10a is the
  session + memory consolidation only.
- **`brainwires-session` retired.** Folded as the `session` feature.
  Never published to crates.io.
API breakage:

- `Cargo.toml`: `brainwires-session = "0.10"` → `brainwires-stores = { version = "0.11", features = ["session", "sqlite"] }` (`session` is default-on; only list it explicitly if `default-features = false`).
- `Cargo.toml`: `brainwires-memory = { features = ["dream"] }` continues to work — the crate's API is preserved; `dream` is unchanged. The schema types it operates over now live in `brainwires-stores`, and `brainwires-memory` re-exports them so existing imports keep compiling.
- `use brainwires_session::*` → `use brainwires_stores::*` (or the fully-qualified `brainwires_stores::session::*`).
- `use brainwires_memory::{MessageStore, MessageMetadata, …}` continues to work via re-export, but new code should prefer `brainwires_stores::*` for the schema types and reserve `brainwires_memory::*` for orchestration (`TieredMemory`, `MultiFactorScore`, `dream`).
- The umbrella `brainwires` facade gains `session`, `task`, `plan`, `conversation`, `lock`, `image`, `tiered` features. The existing `memory` feature now means "tier schema stores" (always-available); `tiered` adds `TieredMemory` orchestration; `dream` adds offline consolidation. The old `native` feature on `brainwires-memory` is gone — `arrow-schema` is always pulled when `memory` is enabled.

There is no re-export shim.

#### `brainwires-training` renamed to `brainwires-finetune`; new `brainwires-training` placeholder

The crate previously named `brainwires-training` only ever did
fine-tuning — cloud fine-tune APIs (OpenAI / Anthropic / Bedrock /
Vertex AI / etc.) plus local LoRA/QLoRA/DoRA — never training-from-scratch.
The name was technically incorrect. Renamed to match what the code
actually does:

- **`brainwires-finetune`** (renamed from `brainwires-training`) —
  cloud fine-tune APIs + dataset pipelines.
- **`brainwires-finetune-local`** — local PEFT (already separate as of
  the previous entry below).
- **`brainwires-training`** (new placeholder crate) — reserved for
  actual training-from-scratch primitives. No code yet; the crate
  exists to occupy the name on crates.io and document the split in
  its README.

API breakage:

- `Cargo.toml`: `brainwires-training = "0.10"` → `brainwires-finetune = "0.10"`.
- All `use brainwires_training::*` imports → `use brainwires_finetune::*`.
- The umbrella `brainwires` crate's `training` / `training-cloud` /
  `training-full` features now route to `brainwires-finetune` (feature
  names unchanged).

#### `brainwires-providers` split — speech (TTS / STT) extracted to `brainwires-provider-speech`

`brainwires-providers` mixed two unrelated concerns: LLM chat clients
(Anthropic, OpenAI, Google, Ollama, Bedrock, Vertex AI, local llama.cpp /
Candle) and speech (Azure Speech, Cartesia, Deepgram, ElevenLabs, Fish,
Google TTS, Murf, browser-native `web_speech`). Every consumer pulled
both stacks even when it only wanted one.

- **New `brainwires-provider-speech` crate** — all 8 speech providers
  + browser-native `web_speech` (wasm32 + `web-speech` feature).
  Independent — only depends on `brainwires-core`. The `RateLimiter`
  type is duplicated here (146 lines, stdlib-only) rather than dragged
  in from `brainwires-providers` to avoid cross-coupling.
- **`brainwires-providers` keeps** the LLM chat clients only.
  Description updated to reflect that.

API breakage:

- `brainwires_providers::azure_speech::*` → `brainwires_provider_speech::azure_speech::*`
  (and analogously for cartesia / deepgram / elevenlabs / fish / google_tts / murf / web_speech).
- `brainwires-providers/web-speech` feature is gone — use
  `brainwires-provider-speech/web-speech` directly.
- `brainwires-providers` no longer pulls `wasm-bindgen` /
  `js-sys` / `web-sys` / `wasm-bindgen-futures` (they were
  speech-only).

Consumer updates landed in this commit: `brainwires-hardware`'s audio
surface and `extras/brainwires-chat-pwa/wasm` switched to the new
crate. The umbrella `brainwires` crate's `web-speech` feature now routes
to `brainwires-provider-speech/web-speech`.

#### `brainwires-tools` split into `brainwires-tool-runtime` + `brainwires-tool-builtins`, façade retired

The old `brainwires-tools` crate had grown to 22 source files + 6 subdirs +
32 features mixing two unrelated concerns: a tool-execution **framework**
(executor / registry / dispatch / sandbox / orchestrator / sessions / oauth /
openapi / validation / transactions) and 20+ concrete **builtin tools**
(bash / file_ops / git / web / search / code_exec + interpreters / browser /
email / calendar / system / semantic_search / `BuiltinToolExecutor`). Every
consumer that wanted the framework had to compile every builtin's deps
(lettre, async-imap, icalendar, mlua, boa_engine, notify, rhai, …).

- **New `brainwires-tool-runtime` crate** — the framework half. `ToolExecutor`
  trait, `ToolRegistry` (now without the hardcoded `with_builtins()`
  constructor), error taxonomy, sanitization, smart router, tool_search,
  transaction manager, validation, plus the optional orchestrator /
  oauth / openapi / sandbox_executor / sessions / tool_embedding modules.
- **New `brainwires-tool-builtins` crate** — the concrete tools.
  `BuiltinToolExecutor` (which dispatches the builtins) and
  `registry_with_builtins()` (the relocated convenience constructor) live
  here.
- **`brainwires-tools` retired.** A 0.10.1 deprecation marker is published
  to occupy the name on crates.io; depending on it gets you nothing.
  Migrate per [`deprecated/brainwires-tools/README.md`](deprecated/brainwires-tools/README.md).

API breakage to migrate:

- `Cargo.toml`: replace `brainwires-tools = "0.10"` with
  `brainwires-tool-runtime = "0.11"` and/or `brainwires-tool-builtins = "0.11"`
  (most consumers want the latter, which already pulls the runtime).
- All `use brainwires_tools::*` imports → switch to whichever sub-crate
  the symbol came from. The migration table in
  `deprecated/brainwires-tools/README.md` lists every type.
- `ToolRegistry::with_builtins()` is gone. Call
  `brainwires_tool_builtins::registry_with_builtins()` instead.
- `brainwires_tool_runtime::smart_router::get_smart_tools(messages)` and
  `get_smart_tools_with_mcp(messages, mcp_tools)` now take a
  `&ToolRegistry` argument so the runtime crate doesn't have to know
  about the builtins.

#### `brainwires-knowledge` split into knowledge + rag + prompting

`brainwires-knowledge` was the heaviest god-crate, mixing knowledge graphs,
adaptive prompting, codebase RAG, spectral math, and code analysis. Every
consumer paid for lancedb + tantivy + tree-sitter (12 grammars) + git2 +
rmcp + rayon even when they only wanted BrainClient.

- **`brainwires-knowledge` keeps** the knowledge layer: BKS / PKS, brain
  client, entity graph, thought storage. Default features now `["knowledge"]`.
- **New `brainwires-rag` crate** — codebase indexing + hybrid retrieval
  (vector + BM25), AST-aware chunking via tree-sitter (12 languages
  always-on), Git history search. Carries `spectral` (log-det diversity
  reranking) and `code_analysis` (AST symbol/definition/reference graphs)
  as internal `pub mod` modules — they have no external consumers and
  splitting them further would force a public API for no caller.
- **New `brainwires-prompting` crate** — adaptive prompting (15-technique
  library, K-means task clustering, BKS/PKS-aware generator, SEAL feedback
  hook, temperature optimisation, optional SQLite cluster store).
  Default features `["knowledge"]` because generator / learning /
  temperature reference BKS/PKS unconditionally.

API breakage:

- `brainwires_knowledge::rag::*` → `brainwires_rag::*`
- `brainwires_knowledge::spectral::*` → `brainwires_rag::*`
  (re-exported at crate root)
- `brainwires_knowledge::code_analysis::*` → `brainwires_rag::*`
- `brainwires_knowledge::prompting::*` → `brainwires_prompting::*`
- `brainwires-knowledge` features `rag`, `spectral`, `code-analysis`,
  `tree-sitter-languages`, `documents`, `pdf-extract-feature`,
  `lancedb-backend`, `qdrant-backend`, `prompting`, `prompting-storage`
  are gone — opt into the new crate that owns them instead.

Folded together (not split apart): the old `brainwires-knowledge::dream`
module (offline memory consolidation — summarisation, fact extraction,
hot/warm/cold tier transitions) merged into `brainwires-memory` under a
`dream` feature. Dream is the consolidation engine that writes to the
same tiers `brainwires-memory` already owned, so they belong together.

`brainwires_knowledge::dream::*` → `brainwires_memory::dream::*`.

#### `brainwires-storage` split into primitives + memory + CLI domain stores

`brainwires-storage` was originally meant for generic storage primitives but
accreted 11 app-specific stores plus tiered-memory orchestration. Cut along
the natural seam:

- **`brainwires-storage` keeps** the primitives only — `StorageBackend` /
  `VectorDatabase` traits, all 9 database backends, `CachedEmbeddingProvider`,
  `BM25Search`, file-context, paths, image-storage *types* (`ImageMetadata`,
  `ImageStorage`, etc.), and the wasm32 HNSW index. Generic re-exports stay
  at the same paths (`brainwires_storage::StorageBackend`,
  `brainwires_storage::CachedEmbeddingProvider`, etc.).
- **New `brainwires-memory` crate** owns the tiered hot/warm/cold memory
  cluster: `MessageStore`, `SummaryStore`, `FactStore`, `MentalModelStore`,
  `TierMetadataStore`, and `TieredMemory` orchestration with multi-factor
  scoring. Re-exported under `brainwires::memory::*` behind the new
  `memory` feature on the umbrella facade.
- **`extras/brainwires-cli` `crate::storage`** absorbed the CLI-domain
  stores: `ConversationStore`, `TaskStore`/`AgentStateStore`, `PlanStore`,
  `TemplateStore`, `LockStore`, `ImageStore`, and `PersistentTaskManager`.
  These were CLI-only consumers; moving them out of the framework cleans
  the workspace's reverse-dependency story.
- The tiered-memory examples (`tiered_memory.rs`) and CLI-store examples
  (`lock_coordination.rs`, `message_store.rs`, `plan_templates.rs`) moved
  with their stores.

Migration:
- `use brainwires_storage::{MessageStore, TieredMemory, …}` →
  `use brainwires_memory::{MessageStore, TieredMemory, …}`.
- `use brainwires::storage::TieredMemory` →
  `use brainwires::memory::TieredMemory` (enable the `memory` feature).
- `use brainwires_storage::{ConversationStore, PlanStore, …}` →
  these stores live in `brainwires-cli` now; not part of the framework
  surface anymore.

#### `extras/brainwires-memory-service` renamed to `extras/brainwires-memory-server`

The old name overlapped with the new lib crate (`brainwires-memory`) once
the storage refactor landed. The mem0-compatible REST surface — backed by
`brainwires-knowledge`'s LanceDB ThoughtStore, unchanged in behaviour — is
now built from the `brainwires-memory-server` package and produces the
`brainwires-memory-server` binary. The crate is unrelated to the new
`brainwires-memory` lib (different layer, no dependency between them).

Migration:
- Cargo: `cargo run -p brainwires-memory-server` (was `-p brainwires-memory-service`).
- Binary: `brainwires-memory-server` (was `brainwires-memory`).
- Package metadata: package name, lib name, and bin name all updated.

#### Singularization sweep + content-rename (`mcp` → `mcp-client`, `resilience` → `call-policy`)

The framework's pluralization rule is **singular for capability domains**.
Five plural crate names violated the rule and were renamed in one
sweep, plus two crates whose abstract names did not describe their
actual contents:

| Old name | New name | Why |
|---|---|---|
| `brainwires-permissions` | `brainwires-permission` | Singular rule |
| `brainwires-providers` | `brainwires-provider` | Singular rule |
| `brainwires-tools` | (split into `tool-runtime` + `tool-builtins`) | Singular rule + split |
| `brainwires-agents` | `brainwires-agent` | Singular rule |
| `brainwires-mcp` | `brainwires-mcp-client` | Asymmetry with `brainwires-mcp-server` |
| `brainwires-resilience` | `brainwires-call-policy` | "Resilience" was an abstract Polly/Resilience4j-borrowed name; the crate's actual content is policies on outbound provider calls (retry / circuit / budget / cache / classify) |

API breakage: `Cargo.toml` deps and all `use brainwires_<old>::*`
imports must rewrite to the new name. Each old name has a 0.10.1
deprecation tombstone published to crates.io that depending on gets
you nothing — see `deprecated/<old-name>/README.md` for per-crate
migration tables.

There is no re-export shim for any of these.

### Removed (BREAKING)

#### Compile-breaking feature deleted

- **`wake-word-porcupine`** — feature and `PorcupineDetector` module deleted from `brainwires-hardware`, the `brainwires` facade, and `voice-assistant`. The Picovoice `pv_porcupine` dep was never on crates.io and the feature could not compile without manual git-dep injection. If Porcupine is needed, implement `WakeWordDetector` against it out-of-tree.
- **`brainwires-tools/interpreters-python`** — feature, `PythonExecutor`, `Language::Python` variant, and `crates/brainwires-tools/src/interpreters/languages/python.rs` deleted. The feature advertised in-process Python execution but was a stub returning a runtime error: the only viable wiring (RustPython) hits a `liblzma-sys` ↔ `lzma-sys` C-link collision with `xz2` (transitive of `lancedb`/`datafusion`) that needs a separate workspace-level resolution. `code_exec`'s native dispatch and the `Language` enum now cover Rhai/Lua/JavaScript only. The Docker-backed and remote-sandbox interpreters still accept `"python"` as a language string — they shell out to a system `python3`, unaffected by this change. Re-add when a working in-process backend is selected.

#### Feature-flag aliases removed

- **`brainwires-storage/vector-db`** — backward-compat alias for `lance-backend`. Use `lance-backend` directly.
- **`brainwires-knowledge/spectral-select`** — deprecated alias for `spectral`. Use `spectral` directly.
- **Facade `brain` feature and `brainwires::brain` module** — consolidated into the canonical `knowledge` feature. Callers: `brainwires::brain::*` → `brainwires::knowledge::*`.
- **`brainwires-agent/reasoning` feature** — removed. Depend on `brainwires-reasoning` directly.

#### Type aliases removed

- **`brainwires_storage::embeddings::EmbeddingProvider` type alias** — was a backward-compat alias for `CachedEmbeddingProvider`. Callers using it as a concrete type (`Arc<EmbeddingProvider>`, `EmbeddingProvider::new()`) must switch to `CachedEmbeddingProvider`. Callers using the trait should import `brainwires_core::EmbeddingProvider` (also re-exported as `brainwires_storage::embeddings::EmbeddingProvider` post-rename, since the name collision is gone).
- **`brainwires_storage::EmbeddingProviderTrait` re-export** — removed. The trait is now re-exported as its canonical name `EmbeddingProvider`.
- **`brainwires_providers::openai_responses::ResponseApiResponse` type alias** — removed. Use `ResponseObject`.
- **`brainwires_agent::reasoning` re-export module** — removed. Use `brainwires_reasoning::*` directly (the facade exposes it as `brainwires::reasoning::*`).

#### Other pre-1.0 cleanup

- **`LegacyHashCache` + migration code** removed from `brainwires-knowledge::rag::cache`. Old RAG cache files on disk will fail to parse and be regenerated on next index run (acceptable pre-1.0; no data loss — only recomputed indices).
- **`PksSseListener` renamed to `PksRestPoller`** in `brainwires-knowledge::knowledge::bks_pks::personal`. The old name lied — the type uses REST polling, not SSE. The SSE client is only the web frontend, unaffected.
- **`stack-graphs` feature over-promise stripped**: `PrecisionLevel::High` no longer claims "~95% accuracy"; `code_analysis::stack_graphs` module is now labelled as a stub until the real `stack-graphs` crate integration lands. The feature flag and provider scaffolding remain in place so the real wire-up can slot in without another API change.

### Added

#### Publish / docs.rs readiness

- **`[package.metadata.docs.rs]` stanza** added to all 16 published framework crates:
  ```toml
  [package.metadata.docs.rs]
  all-features = true
  rustdoc-args = ["--cfg", "docsrs"]
  ```
  so docs.rs renders the full feature surface (previously heavy feature-flag crates like `brainwires-hardware`, `brainwires-telemetry`, `brainwires-knowledge`, `brainwires-storage`, `brainwires-providers` rendered only the default-feature surface).
- **`#![warn(missing_docs)]`** added to `brainwires-hardware` and `brainwires-telemetry` (previously the only two framework crates not enforcing it — the other 13 already did).
- **`AgentCard`, `MeshTopology`, `TopologyType`** added to the `brainwires::prelude` under the `a2a` and `mesh` features.
- **`brainwires::knowledge` facade module** — replaces the old `brainwires::brain` module, gated on the canonical `knowledge` feature.

### Changed

- **`brainwires::reasoning` module** now re-exports from `brainwires_reasoning` directly instead of going through `brainwires_agent::reasoning`. The `reasoning` feature in the facade activates `brainwires-reasoning` directly.
- **Storage Arrow schema docs** — removed "for backward compatibility with `LanceDatabase`" mislabelling on `tasks_arrow_schema`, `agent_states_arrow_schema`, `facts_schema`, `summaries_schema`, `plans_schema`, `tier_metadata_schema`. These are current infrastructure, not legacy shims.
- **`Filter::Raw` doc comment** (`brainwires-storage::databases::lance::arrow_convert`) — clarified as an explicit escape hatch, not a backward-compat concession. Dropped the runtime `tracing::warn!` on every call.
- **`#[ignore]` markers** in `brainwires-storage::databases::nornicdb::tests` (33 occurrences) now carry the reason string `"requires running nornicdb instance"` so `cargo test -- --ignored` output is self-explanatory.
- **`matter::verhoeff`** demoted from `pub mod` to `pub(crate) mod` (internal-only helper used by the commissioning-code parser).

### Documentation

- **`PUBLISHING.md`** — publish-order table rewritten against the real 16-crate DAG. The previous table listed 9 crates that don't exist (`brainwires-analytics`, `brainwires-code-interpreters`, `brainwires-skills`, `brainwires-system`, `brainwires-datasets`, `brainwires-cognition`, `brainwires-tool-system`, `brainwires-agent-network`, `brainwires-channels`) and omitted 7 that do (`brainwires-knowledge`, `brainwires-reasoning`, `brainwires-telemetry`, `brainwires-training`, `brainwires-hardware`, `brainwires-a2a`, `brainwires-mcp-server`). `scripts/publish.sh` is the source of truth.
- **Top-level `README.md`** — crate-count claims fixed (16 framework crates + 25 extras including the 7-crate `brainclaw` set). Added missing extras entries: `brainwires-billing`, `brainwires-docs`, `voice-assistant`.
- **Facade `crates/brainwires/README.md`** — feature table rewritten. Previously omitted 13 features that were already exposed in `Cargo.toml` (`chat`, `agent-network`, `mcp-server-framework`, `system`, `dream`, `telemetry`, `bedrock`, `vertex-ai`, `wasm`, `training-cloud`, `training-full`, `training-local`, `rag-full-languages`) and listed 3 that no longer exist (`relay`, `proxy`, `autonomy`). Convenience features table unchanged.
- **`brainwires-storage/README.md`, `brainwires-mcp/README.md`** — license links converted from relative (`[LICENSE](../../LICENSE)`, which 404s on crates.io) to absolute GitHub URLs for both MIT and Apache-2.0 license files.
- **`brainwires-hardware/README.md`, `FEATURES.md`** — all `wake-word-porcupine` / `PorcupineDetector` references removed in line with the code deletion.
- **Workspace-wide markdown consistency sweep** — stale crate names repointed to current successors in: `crates/README.md` (full rewrite of the dependency tree), `FEATURES.md` (datasets, analytics, and extras sections), `extras/brainwires-brain-server/README.md`, `extras/brainwires-rag-server/README.md`, `extras/brainwires-wasm/README.md`, `extras/brainclaw/mcp-skill-registry/README.md`, `crates/brainwires-training/README.md`, `crates/brainwires-agent/README.md`, `docs/wishlist-crates/Distributed-Training.md`, `extras/brainwires-cli/docs/ARCHITECTURE.md`, `extras/brainwires-cli/docs/distributed-swarms/IPC_AND_REMOTE_CONTROL.md`, `extras/brainwires-cli/docs/adaptive-prompting/ADAPTIVE_PROMPTING_IMPLEMENTATION.md`, `CONTRIBUTING.md`. Historical CHANGELOG entries for prior releases were left intact — they document what shipped at the time.

### Fixed (lint sweep)

- **`cargo clippy --fix`** applied across the workspace — ~57 of 80 pre-existing non-docs warnings auto-fixed (`useless vec!`, collapsible `if`, `unwrap_err`-after-`is_err`, `RangeInclusive::contains`, `Default::default()` field assigns, redundant pattern-matching, etc.). 139 files touched. The remaining ~23 warnings (too-many-args, loop-index-as-var) need manual thought per-site and are deferred.

### Deferred — still present, slated for follow-up work

These remaining backwards-compat surfaces were scoped out of this pass because they change runtime behaviour (not just names) or touch many downstream consumers. Each will land as its own focused PR:

- **`brainwires-mcp::types` rmcp compat aliases** (`McpTool`, `McpResource`, `McpPrompt`, `CallToolParams`, `ServerCapabilities`, `ClientCapabilities`) — touches 20+ files including the brainclaw channel servers.
- **`brainwires-network::auth` session legacy path** — `api_key` field on `Session`/`SessionInfo` + `migrate_legacy_session` + file fallback. Removing breaks existing on-disk session files (acceptable pre-1.0, but needs a dedicated migration note).
- **`brainwires-network::remote::protocol` `Option<Protocol>` fields** — wire-format change; requires protocol-version bump and coordinated client/server updates.
- **`brainwires-network::ipc::socket` legacy plaintext `IpcReader` / `IpcWriter`** — need to audit whether the handshake still needs the plaintext path before deletion.
- **`brainwires-agent` crate** still compiles with the old "reasoning feature" shape; clean up `[features]` to drop residual entries.

### Follow-up plans (filed separately)

1. **`stack-graphs` full wire-up** — add the real `stack-graphs` crate as an optional dep under the existing feature flag, implement `extract_definitions` / `extract_references` for Python / TypeScript / Java / Ruby, benchmark, restore accuracy claims.
2. **Matter DAC/PAI/CD CSA-signing** — organizational, blocked on CSA membership (see `BRAINWIRES_MATTER_DAK_PATH`). Not a code change.
3. **(Optional) Porcupine wake-word re-add** — if/when the `pv_porcupine` crate lands on crates.io or a real vendored path is agreed on.
4. **Missing-docs cleanup** — 428 warnings in `brainwires-hardware` and 129 in `brainwires-telemetry` surfaced by the new `#![warn(missing_docs)]` stepping stone; close them before promoting to `#![deny]`.

## [0.10.0] - 2026-04-18

### Changed

#### `brainwires-reasoning` restored as Layer 3 owner (BREAKING)

The 0.9.0 `brainwires-reasoning` crate shipped as a 22-line re-export shell.
The 0.8 → 0.9 refactor split the intended content across two other crates:
the plan/output parsers stayed in `brainwires-core` behind a `planning`
feature, and the 9 local-inference scorers were tucked into
`brainwires-agent::reasoning` behind a feature. The original architectural
plan (PR 7 in the 0.9 refactor series) specified these move into
`brainwires-reasoning`; the move did not happen.

0.10.0 completes it. `brainwires-reasoning` now owns, as real modules:

- `plan_parser` and `output_parser` (moved from `brainwires-core`),
- `complexity`, `entity_enhancer`, `relevance_scorer`,
  `retrieval_classifier`, `router`, `strategies`, `strategy_selector`,
  `summarizer`, `validator` (moved from `brainwires-agent::reasoning`).

Backward-compatibility: `brainwires-agent` still exposes
`brainwires_agent::reasoning::…` under its `reasoning` feature — it now
simply re-exports `brainwires_reasoning`. No changes needed for callers
using that path.

**Breaking:** callers importing directly from `brainwires_core` must
update.

| 0.9.0 path | 0.10.0 path |
|---|---|
| `brainwires_core::plan_parser::{parse_plan_steps, steps_to_tasks, ParsedStep}` | `brainwires_reasoning::plan_parser::…` (also re-exported at crate root) |
| `brainwires_core::output_parser::{JsonOutputParser, JsonListParser, OutputParser, RegexOutputParser}` | `brainwires_reasoning::output_parser::…` (also re-exported at crate root) |
| `brainwires_core/planning` feature | feature removed — pull `brainwires-reasoning` directly |
| `brainwires_core/native` feature | kept as an empty stub for downstream compatibility |

### Added

#### Tools — bash sandbox + byte caps (`brainwires-tools`)

- **`BashSandboxMode::NetworkDeny`** — wraps every `execute_command` in
  `unshare -U -r -n -- bash -o pipefail -c …` on Linux, denying outbound
  network via a new user + network namespace without requiring root. Silent
  no-op on non-Linux with a warning surfaced in the tool result so the
  model knows sandboxing was not enforced.
- **Opt-in from env or CLI** — `BRAINWIRES_BASH_SANDBOX=network-deny`
  (also `networkdeny`, `1`, `on`) or the new `brainwires chat --sandbox
  network-deny` CLI flag. `Off` is the default; `from_env()` is read at
  command-build time, so every bash tool call goes through the same
  policy gate regardless of invocation path.
- **Per-stream 25KB byte cap** — `MAX_STREAM_BYTES = 25_000`. Stdout and
  stderr are each middle-truncated with a `… [N bytes truncated] …`
  marker, preserving head + tail and respecting UTF-8 boundaries. Guards
  against a single runaway line (binary blob, `cat` on a huge log)
  blowing past context limits regardless of line-based `output_mode`.

#### Providers — Anthropic prompt caching + image blocks
(`brainwires-providers`)

- **Prompt caching enabled by default** — `cache_prompt: true` on both
  `messages` (single-shot) and streaming requests. `cache_read` and
  `cache_creation` token counts are logged (`tracing::info!` on cache
  hits, `tracing::debug!` on writes) so operators can verify
  cache-hit-rate in production.
- **`ContentBlock::Image` (Base64) → Anthropic image envelope** — the
  Anthropic chat provider now converts core `ImageSource::Base64
  { media_type, data }` blocks into native Anthropic
  `image` content blocks. Unblocks multimodal user messages; added a
  dedicated roundtrip test.

#### CLI — dream, sandbox, tool curation, monitor, shell overlay, and more
(`brainwires-cli`)

- **Dream (sleep) consolidation** — new `/dream`, `/dream:status`,
  `/dream:run` slash commands. The framework's
  `brainwires::dream::DreamConsolidator` does the work; the CLI supplies
  an `InMemoryDreamSessionStore` adapter that feeds the active
  conversation into the consolidator and surfaces a before/after token
  report. Manual on-demand today; a tokio-interval scheduler can sit on
  top later without changing this API.
- **`--sandbox=network-deny`** — top-level CLI flag that sets
  `BRAINWIRES_BASH_SANDBOX` once at startup (pre-thread-spawn) so the
  bash tool's env read is race-free.
- **`--all-tools`** — opt-in eager enumeration of every registered
  tool. Non-TUI chat paths default to the curated core set (14 tools
  including `search_tools`) in canonical order — smaller outbound
  request body and a stable prefix for Anthropic prompt caching.
- **Monitor tool** — background process watcher that streams stdout
  events as notifications; filter-first design so a single noisy log
  doesn't flood the conversation.
- **`/shell` interactive overlay** — full terminal subshell overlay
  inside the TUI.
- **Remappable global keybindings** — `~/.claude/keybindings.json`
  drives chord and single-key rebinding for all global TUI shortcuts.
- **Harness parity** — settings, hooks, memory, ask-user-question,
  monitor polish; TUI skill autocomplete; custom status line;
  auto-loading of `CLAUDE.md` / `BRAINWIRES.md` from cwd upward;
  `--provider` first-run picker; worktree primitive; skill
  `allowed_tools` + execution-mode honouring in `/skill`; 2 456-line
  `command_handler.rs` split into topic submodules.

#### Tests — proptest + 92 new tests

`proptest` added as a workspace dev-dependency. 92 new tests land across
five new integration-test files:

- **`brainwires-permissions` (44 tests, 4 files)** —
  `tests/policy_matching.rs` (23 tests: every `PolicyCondition` variant
  incl. And/Or/Not composition, priority ordering, default-action
  fallback, disabled-policy skipping, `with_defaults()` preset);
  `tests/wildcard_domains.rs` (5 proptests guarding
  `*.example.com` suffix/prefix-confusion bypasses);
  `tests/audit_durability.rs` (8 tests covering important-event
  immediate-write, buffer-flush ordering, JSONL replay from a prior
  session, disabled-logger silence); `tests/anomaly_thresholds.rs`
  (8 tests pinning the sliding-window threshold boundary, per-agent
  isolation, out-of-window forgetting, path-scope allowlist).
- **`brainwires-mcp` (15 tests, 1 file)** — `tests/jsonrpc_roundtrip.rs`:
  string/integer/null id roundtrips, response-error wire shape,
  notification id-absence contract, progress-notification parsing,
  unknown-method fallthrough, malformed-JSON rejection, transport
  discriminator on explicit null id, five proptest roundtrips for
  Request/Response-success/Response-error/Notification/ProgressParams.
- **`brainwires-reasoning` (25 tests, 1 file)** —
  `tests/parser_properties.rs`: numbered + bulleted + `Step N:` plan
  formats, priority-keyword detection, indent→substep mapping,
  steps-to-tasks invariants, JSON extraction from markdown fences with
  and without language tags and from surrounding prose, regex-parser
  named-capture extraction and invalid-pattern rejection, five
  proptests including panic-freeness on arbitrary text and embedded-
  object extraction.
- **`brainwires-tools` (7 tests, 1 file)** —
  `tests/path_resolution.rs`: relative-vs-absolute anchoring,
  nonexistent-path fallback, nested paths, documented-and-pinned
  current non-sandbox `..` traversal behaviour, two proptests covering
  arbitrary UTF-8 input and unicode-named paths.
- **`brainwires` metacrate (1 test, 1 file)** —
  `tests/reexports.rs`: compile-time smoke for the feature-gated
  re-export surface (core, tools, agents, permissions, reasoning,
  storage, mcp).

### Fixed

- **`brainwires-providers`** — unreachable catch-arm removed from the
  Anthropic content-block conversion; any future `ContentBlock` variant
  now fails loudly at compile time instead of being silently filtered.

### Documentation

- **`TESTING.md`** — corrected every `brainwires-eval` reference. The
  eval framework lives at `brainwires_agent::eval` (feature-gated
  module on `brainwires-agent`), not a standalone
  `brainwires-eval` crate. §8 now notes the empirical-scoring suite
  targets `brainwires_reasoning::ComplexityScorer` after the 0.10
  restoration.
- **`brainwires-hardware`** — Matter implementation marked experimental
  with a documented list of spec-compliance gaps.

### Publish tooling

- **`scripts/publish.sh --preflight-only`** — fast manifest checks
  (README present, no git-only deps without version, metadata set) for
  every publishable crate. Runs in seconds without spending
  `cargo publish --dry-run` time budget.

## [0.9.0] - 2026-04-13

### Added

#### `matter-tool` — Brainwires-native Matter CLI (`extras/matter-tool`)

- **New `matter-tool` binary** — first-party CLI equivalent of `chip-tool` built entirely on the Brainwires pure-Rust Matter 1.3 stack. No `connectedhomeip` dependency; compiles in seconds.
- **`pair` subcommand** — commission devices via QR code (`pair qr <node-id> <MT:…>`), 11-digit manual pairing code (`pair code`), or BLE (`pair ble`, requires `--features ble`). `pair unpair <node-id>` removes a device from the local fabric.
- **Cluster control commands** — `onoff {on,off,toggle,read}`, `level {set,read}`, `thermostat {setpoint,read}`, `doorlock {lock,unlock,read}`. Each takes `<node-id> <endpoint>`.
- **`invoke`** — send a raw cluster command: `invoke <node-id> <endpoint> <cluster-hex> <cmd-hex> [payload-hex]`.
- **`read`** — read a raw cluster attribute: `read <node-id> <endpoint> <cluster-hex> <attr-hex>`.
- **`discover`** — browse `_matterc._udp` (commissionable) and `_matter._tcp` (operational) via mDNS, print found devices with addresses and TXT records. `--timeout <secs>` (default 5).
- **`serve`** — run as a Matter device server (commission us from another controller). Prints QR code and pairing code on startup. Flags: `--device-name`, `--vendor-id`, `--product-id`, `--discriminator`, `--passcode`, `--port`, `--storage`.
- **`devices`** — list all commissioned devices in the local fabric.
- **`fabric info`** — print fabric directory and commissioned node count. **`fabric reset`** — wipe fabric storage (interactive `yes` confirmation required).
- **Global flags** — `--fabric-dir <DIR>` (default `~/.local/share/matter-tool/` on Linux), `--verbose` / `-v`, `--json` (machine-readable output for all commands).
- **`ble` feature** — BLE commissioning path via `brainwires-hardware/matter-ble`; excluded from the default build.

#### GitHub Channel Adapter (`extras/brainclaw/mcp-github`)

- **New `brainclaw-mcp-github` crate** — full GitHub channel adapter for the Brainwires gateway. Receives GitHub webhook events and exposes GitHub operations as an MCP tool server.
- **Webhook receiver** — Axum HTTP server with HMAC-SHA256 signature verification (`X-Hub-Signature-256`). Normalises `issue_comment`, `issues`, `pull_request`, and `pull_request_review_comment` events into `ChannelMessage` values.
- **`GitHubChannel`** — implements the `Channel` trait against the GitHub REST API: post/edit/delete comments, list issue comments, add reactions (with Unicode emoji → GitHub reaction name mapping), retrieve issue history.
- **MCP tool server** — 10 tools via rmcp `tool_router` macros: `post_comment`, `edit_comment`, `delete_comment`, `get_comments`, `create_issue`, `close_issue`, `add_labels`, `create_pull_request`, `merge_pull_request`, `add_reaction`. Runs over stdio alongside the gateway client.
- **Gateway client** — mirrors the `mcp-discord` gateway client pattern: `ChannelHandshake { channel_type: "github" }`, bidirectional `ChannelEvent` ↔ gateway WebSocket forwarding.
- **Config** — env-var driven: `GITHUB_TOKEN`, `GITHUB_WEBHOOK_SECRET`, `WEBHOOK_ADDR` (default `0.0.0.0:9000`), `GATEWAY_URL`, `GATEWAY_TOKEN`, `GITHUB_REPOS` (comma-separated allowlist), `GITHUB_API_URL`.
- **CLI** — `serve` and `version` subcommands via Clap. `--mcp` flag enables the MCP stdio server alongside the gateway client.
- **Tests** — HMAC-SHA256 signature verification, `normalise()` for all four event types, `GitHubChannel` conversation/message-ID parsing, reaction emoji mapping.

#### Multi-Turn Conversation History (`extras/voice-assistant`)

- **`LlmHandler` history** — added `history: Mutex<Vec<OpenAIMessage>>` to `LlmHandler`. Each completed STT→LLM turn appends the user message and assistant reply; the system prompt is prepended fresh on every request. The assistant can now reference earlier turns within a session. `clear_history()` provided for explicit reset.

#### New Examples

- **`brainwires-mcp-server/examples/hello_world_server.rs`** — minimal runnable stdio MCP server with `echo` and `greet` tools. Demonstrates `McpServer`, `McpToolRegistry::dispatch`, `Content::text`, and `LoggingMiddleware`. Can be exercised with raw JSON-RPC on stdin.
- **`brainwires-channels/examples/mock_channel.rs`** — reference `Channel` trait implementation backed by an in-memory `HashMap`. Exercises all six trait methods (`send_message`, `edit_message`, `delete_message`, `add_reaction`, `get_history`, `set_presence`). Serves as the blueprint for real channel adapters.
- **`brainwires-analytics/examples/track_agent_run.rs`** — end-to-end demo of `AnalyticsCollector` + `MemoryAnalyticsSink`. Records `ProviderCall`, `ToolCall`, and `AgentRun` events, calls `flush()`, then snapshots the sink to verify event counts and cost tallies.

#### Full Matter 1.3 Protocol Stack (`brainwires-hardware`)

- **SPAKE2+ Augmented PAKE** (RFC 9383) — pure Rust implementation using RustCrypto p256, implemented from scratch due to the absence of a production-ready SPAKE2+ crate. Prover + Verifier roles, PBKDF2-HMAC-SHA256 passcode derivation, HMAC-SHA256 confirmation (cA/cB).
- **PASE** (Password-Authenticated Session Establishment) — full commissioning handshake: PBKDFParamRequest/Response, Pake1/2/3, session key derivation (I2RKey, R2IKey, AttestationChallenge via HKDF-SHA256).
- **CASE** (Certificate-Authenticated Session Establishment) — SIGMA protocol: Sigma1/2/3 exchange, P-256 ephemeral ECDH, AES-CCM-128 encrypted payloads, NOC chain verification.
- **Matter compact certificate format** — TLV-encoded NOC/ICAC/RCAC encode/decode per Matter spec §6.4, P-256 ECDSA-SHA256 signatures, Matter OIDs for NodeId/FabricId.
- **Fabric management** — `FabricManager` with root CA generation, NOC issuance, JSON persistence, multi-fabric bookkeeping.
- **Matter transport layer** — Message Layer header encode/decode (Matter spec §4.4), MRP (Message Reliability Protocol) with configurable retry/backoff (Matter spec §4.12), AES-CCM-128 UDP session encryption.
- **Interaction Model** — `ReadRequest`/`ReportData`, `WriteRequest`/`WriteResponse`, `InvokeRequest`/`InvokeResponse`, `SubscribeRequest`/`SubscribeResponse` with full TLV encode/decode and wildcard `AttributePath`/`CommandPath`.
- **Mandatory commissioning clusters** — `BasicInformation` (0x0028), `GeneralCommissioning` (0x0030), `OperationalCredentials` (0x003E), `NetworkCommissioning` (0x0031).
- **`MatterDeviceServer`** — fully functional device server: PASE commissioning window, CASE operational sessions, IM cluster dispatch, `CommissionableAdvertiser` mDNS (`_matterc._udp`).
- **`MatterController`** — fully functional controller: mDNS device discovery, PASE commissioning, CASE session management, cluster invoke/read, session caching.
- **BLE commissioning** (`matter-ble` feature) — BTP transport protocol (Matter spec §4.17): handshake, segmentation/reassembly, fragmentation. `MatterBlePeripheral` with Matter BLE service UUID, Linux/macOS btleplug peripheral support.
- **`OperationalAdvertiser`/`OperationalBrowser`** — post-commissioning `_matter._tcp` DNS-SD with CompressedFabricId derivation.
- **New workspace deps** — `p256 0.13.2`, `ecdsa 0.16.9`, `hmac 0.12`, `hkdf 0.12`, `pbkdf2 0.12.2`, `aes 0.8.4`, `ccm 0.5.0`, `der 0.8.0`, `pkcs8 0.10.2`.
- **New features** — `matter-ble` (BLE commissioning), `homeauto-full` (all protocols including BLE).
- **80 unit tests** — all pure logic, no hardware required. Integration test `matter_e2e` available with `--include-ignored`.

#### Home Automation Protocols (`brainwires-hardware`)

- **`homeauto` module** — New `src/homeauto/` module group behind four feature flags: `zigbee`, `zwave`, `thread`, `matter` (or all via `homeauto`). Each sub-module is independent; pull in only what you need.
- **Shared types** — `HomeDevice`, `HomeAutoEvent`, `Capability`, `AttributeValue`, `Protocol` enum used across all four protocols. `BoxStream<'a, T>` alias for async event streams.
- **`zigbee` feature** — Full Zigbee 3.0 coordinator support via raw serial, two backends:
  - `EzspCoordinator` — Silicon Labs EZSP v8 over ASH framing (CRC-16-CCITT poly=0x1021, byte-stuffing 0x7E/0x7D, ACK/NAK/RST flow control). Targets EmberZNet 7.x / EFR32-based sticks (Sonoff Zigbee 3.0 USB Dongle Plus, Aeotec USB 7).
  - `ZnpCoordinator` — TI Z-Stack 3.x ZNP protocol (SREQ/SRSP/AREQ frames with XOR FCS). Targets CC2652, CC2531, and Z-Stack-based dongles.
  - `ZigbeeCoordinator` trait — `start`, `stop`, `permit_join`, `devices`, `read_attribute`, `write_attribute`, `invoke_command`, `events` stream.
  - Standard cluster helpers in `zigbee::clusters`: on/off, level, color temperature, color RGB, temperature sensor, humidity, door lock.
- **`zwave` feature** — Full Z-Wave Plus v2 (specification 7.x / ZAPI2) over USB stick serial port. `ZWaveController` trait with `ZWaveSerialController` implementation. Supports node inclusion/exclusion, 27-variant `CommandClass` enum (BinarySwitch, MultilevelSwitch, Thermostat, DoorLock, SensorMultilevel, Configuration, and more), ACK/NAK/CAN flow control, XOR checksum, 3-retry retransmit on timeout.
- **`thread` feature** — `ThreadBorderRouter` client for the OpenThread Border Router (OTBR) REST API (Thread 1.3.0, default port 8081). Network node info, neighbor table, active/pending dataset retrieval, joiner commissioning. Uses the existing `reqwest` workspace dep — no new heavy dependencies.
- **`matter` feature** — Matter 1.3 support via a purpose-built pure-Rust stack (avoids `rs-matter` due to an `embassy-time` links conflict with the `burn` ML ecosystem):
  - `MatterController` — Commissioner and cluster client. Supports QR-code (`MT:...`) and manual-pairing-code commissioning with full bit-packed Base38 payload parsing. Convenience helpers for OnOff, LevelControl, ColorControl, Thermostat, DoorLock, WindowCovering.
  - `MatterDeviceServer` — Expose Brainwires agents as Matter devices. Commissionable mDNS advertisement (`_matterc._udp`) via `mdns-sd`, UDP transport on port 5540, per-cluster callback handlers (on/off, level, color temp, thermostat). PASE/CASE session establishment is scaffolded with TODO markers pending upstream conflict resolution.
  - `CommissioningPayload` parser — Full Base38 decode + bit-unpack (version, VID, PID, discriminator, passcode, commissioning flow, rendezvous info). Manual pairing code (11-digit decimal) also supported.
  - Cluster TLV helpers — typed encoders for all major clusters using the Matter TLV wire format.
- **New workspace deps** — `tokio-serial = "5.4"`, `crc = "3"`, `mdns-sd = "0.12"`, `gethostname = "1.0"` (last two already in workspace, now also optional in hardware).
- **New examples** — `zigbee_scan`, `zwave_nodes`, `thread_info`, `matter_on_off`.
- **`full` feature** — Now includes `homeauto`.
- **71 unit tests** — All pure-logic tests (no hardware required): ASH framing + CRC-16-CCITT (verified against `b"123456789"` → 0x29B1), EZSP frame encode/decode, ZNP SREQ/SRESP/AREQ roundtrip, ZAPI frame + XOR checksum, Z-Wave CommandClass serialization, Thread OTBR responses (mocked via `wiremock`), Matter QR/manual code parsing, Matter cluster TLV encoding.

#### Claude Brain — Brainwires Context Management (`extras/claude-brain`)

- **New `claude-brain` crate** — persistent context management for Claude Code sessions via hook-based integration. Survives compaction events so critical context (decisions, facts, summaries) is never lost.
- **Hook-based architecture** — `PreCompact` saves context to persistent storage before compaction, `SessionStart` restores it on session init (routed through SessionStart instead of PostCompact for reliability).
- **Dynamic hook budget** — hook output budget computed from compaction threshold × 70%, ensuring restored context fits within available token window.
- **Settings from JSON** — reads configuration from JSON settings files; replaced magic numbers with named constants.
- **v2 structural improvements** — 10 improvements across 3 phases: better compaction loop handling, integration file sourcing from `extras/`, and `install.sh` for automated setup.

#### `brainwires-memory-service` — Mem0-Compatible Memory REST API (`extras/brainwires-memory-service`)

- **New `brainwires-memory-service` crate** — standalone REST API server providing Mem0-compatible endpoints for memory storage and retrieval, backed by the Brainwires storage layer.

#### `EmailIdentityProvider` (`brainwires-network`)

- **New `EmailIdentityProvider`** — identity provider for internet-facing agent email, enabling agents to have verifiable email-based identities for external communication.

#### Session-Level Token Budget Enforcement (`brainwires-cli`)

- **`SessionBudget`** — New type in `extras/brainwires-cli/src/types/session_budget.rs` with atomic counters (`Arc<AtomicU64>` for tokens and cost-in-microcents, `Arc<AtomicU32>` for agent count). Methods: `check_before_spawn()`, `record_run(tokens, cost_usd)`, `check_limits()`, `increment_agent_count()`.
- **`TaskAgentConfig` budget fields** — Added `max_total_tokens: Option<u64>`, `max_cost_usd: Option<f64>`, `timeout_secs: Option<u64>`, and `session_budget: Option<Arc<SessionBudget>>`. The execution loop enforces per-agent token and cost caps from provider response usage, and delegates session-level cap checks to `SessionBudget` before each spawn.

#### Infinite Context Wired into TaskAgent (`brainwires-cli`)

- **`MessageStore` initialization in `TaskAgent`** — `TaskAgent::execute()` now initializes a `MessageStore` backed by LanceDB using the same pattern as the chat loop (`PlatformPaths::conversations_db_path()` + `EmbeddingProvider` + `LanceDatabase::initialize()`). Falls back to raw conversation history if LanceDB is unavailable; never fails hard.
- **`ContextBuilder` integration** — `call_provider()` now calls `ContextBuilder::build_full_context()` with `use_gating: false` so semantic retrieval fires on every call without requiring compaction markers. This matches the always-on behavior of the chat path (`ai_processing.rs`). Task agents now benefit from the same personal knowledge injection and semantic history retrieval as chat sessions.
- **Message persistence** — Each agent turn is stored in `MessageStore` so long-running tasks accumulate retrievable history across iterations.

#### Structured Agent Roles with Tool Restrictions (`brainwires-agent`)

- **`AgentRole` enum** — New `crates/brainwires-agent/src/roles.rs` with four variants:
  - `Exploration` — read-only: `read_file`, `list_directory`, `search_code`, `glob`, `grep`, `fetch_url`, `web_search`, `context_recall`, `task_get`, `task_list`
  - `Planning` — task management + read access: `task_create`, `task_update`, `task_add_subtask`, `plan_task`, plus read tools
  - `Verification` — read + build/test: `execute_command`, `check_duplicates`, `verify_build`, `check_syntax`, plus read tools
  - `Execution` — full access (default, all tools permitted)
- **Enforcement at provider call time** — `AgentRole::filter_tools()` filters the tool list passed to the provider, not post-hoc. The model never receives tools it cannot use, reducing hallucination and wasted tokens.
- **System prompt suffix** — `AgentRole::system_prompt_suffix()` appends a role constraint reminder to the agent's system prompt.
- **`registry.filtered_view()`** — Added `filtered_view(&self, allow: &[&str]) -> Vec<Tool>` to `brainwires-tool-system` registry for building role-scoped tool lists.
- **`role: Option<AgentRole>`** added to `TaskAgentConfig`.

#### Persistent Workflow State / Crash-Safe Retry (`brainwires-core`)

- **`WorkflowCheckpoint`** — Snapshot of agent execution progress: `task_id`, `agent_id`, `step_index`, `completed_tool_ids: HashSet<String>`, `side_effects_log: Vec<SideEffectRecord>`, `updated_at`.
- **`SideEffectRecord`** — Per-tool completion record: `tool_use_id`, `tool_name`, `target: Option<String>`, `completed_at`, `reversible`.
- **`WorkflowStateStore` trait** — `save_checkpoint`, `load_checkpoint`, `mark_step_complete`, `delete_checkpoint`.
- **`FsWorkflowStateStore`** — Persists checkpoints as JSON under `~/.brainwires/workflow/{task_id}.json` using atomic write (write to `.tmp`, then `rename`). Never leaves a partially-written file.
- **`InMemoryWorkflowStateStore`** — In-memory store for tests; no filesystem I/O.
- **`TaskAgent` crash-resume** — On startup, loads any prior checkpoint and skips `tool_use_id`s already recorded as complete. Persists each successful tool call. Deletes the checkpoint on clean task completion.

#### Unified Event Schema with Trace IDs (`brainwires-core`, `brainwires-a2a`, `brainwires-agent-network`)

- **`Event` trait** — Common interface: `event_id()`, `trace_id()`, `sequence()`, `occurred_at()`, `event_type()`. Implementing is optional; prefer `EventEnvelope` at boundaries.
- **`EventEnvelope<E>`** — Generic wrapper carrying any payload with `event_id: Uuid`, `trace_id: Uuid`, `sequence: u64`, `occurred_at: DateTime<Utc>`. Implements `Event`. `map()` preserves all correlation fields. `new_trace_id()` helper for call-site clarity.
- **Trace ID propagation in `TaskAgent`** — `execute()` generates a `trace_id: Uuid::new_v4()` at startup, writes it into `AgentContext.metadata["trace_id"]`, and logs it at the `INFO` level. Every `ToolContext` built from that agent context automatically carries the trace ID, enabling correlation with `AuditEvent.metadata["trace_id"]` without struct changes.
- **A2A streaming events** — `TaskStatusUpdateEvent` and `TaskArtifactUpdateEvent` gain `trace_id: Option<Uuid>` (serialized as `traceId`) and `sequence: Option<u64>`, both `skip_serializing_if = None` for wire compatibility.
- **`MessageEnvelope`** — Gains `trace_id: Option<Uuid>` field. `reply()` inherits the sender's trace ID. New `with_trace(trace_id)` builder method.

#### Framework-Level System Prompt Registry (`brainwires-agent`, `brainwires-cli`)

- **`AgentPromptKind` enum** — New `crates/brainwires-agent/src/system_prompts/mod.rs` is the authoritative inventory of every agent system prompt in the framework. Variants: `Reasoning`, `Planner`, `Judge`, `Simple`, `MdapMicroagent`. Adding a new agent type means adding a variant here first.
- **`build_agent_prompt(kind, role)` dispatcher** — Single function to build any agent system prompt. Automatically appends `AgentRole::system_prompt_suffix()` when a role is provided, removing the need for callers to handle role suffix injection manually. Replaces the manual `format!("{}{}", base, role.system_prompt_suffix())` pattern in `task_agent.rs`.
- **`MdapMicroagent` prompt** — New `mdap_microagent_prompt()` for MDAP voting agents. Instructs each microagent to reason independently, notes the vote round and peer count, and explicitly discourages anchoring on what other agents might produce.
- **Eliminated CLI duplicate** — `extras/brainwires-cli/src/agents/system_prompts.rs` was an exact copy of the framework module. Deleted; all callers now import from `brainwires::agents`.
- **CLI mode prompt registry** — New `extras/brainwires-cli/src/system_prompts/modes.rs` consolidates all interactive-mode system prompts: Edit, Ask, Plan, Batch, and the `plan_task` tool sub-agent. Prompts that were previously buried inside `agent/plan_mode.rs` and `tools/plan.rs` are now extracted here.
- **`build_ask_mode_system_prompt_with_knowledge()`** — Previously missing variant (Edit mode had knowledge injection; Ask mode did not). Now available in `modes.rs`.
- **`build_batch_mode_system_prompt()`** — New distinct Batch-mode prompt optimised for throughput: concise/consistent output, self-contained responses, no exploratory dialogue.
- **`utils/system_prompt.rs` simplified** — Reduced to a thin re-export shim pointing to `system_prompts::modes` for backward compatibility.

### Changed

#### Architecture Refactoring — 22 → 16 Framework Crates

- **Crate renames** — `brainwires-tool-system` → `brainwires-tools`, `brainwires-agent-network` → `brainwires-network`, `brainwires-cognition` → `brainwires-knowledge`. All public API paths updated accordingly.
- **Crate absorptions** — `brainwires-channels` merged into `brainwires-network`, `brainwires-skills` merged into `brainwires-agent`, `brainwires-code-interpreters` merged into `brainwires-tools`, `brainwires-datasets` merged into `brainwires-training`.
- **Moved to extras** — `brainwires-wasm` and `brainwires-autonomy` moved from `crates/` to `extras/` (no longer independently published framework crates).
- **New crate** — `brainwires-reasoning` re-exports reasoning strategies from `brainwires-core`.
- **`publish.sh` updated** — publish order reduced from 22 to 16 crates.

#### Deno/TypeScript Port — Package Renames

- **Package renames** — `@brainwires/tool-system` → `@brainwires/tools`, `@brainwires/agent-network` → `@brainwires/network`, `@brainwires/cognition` → `@brainwires/knowledge`.
- **`@brainwires/skills` merged into `@brainwires/agents`** — skill parsing, registry, routing, and execution now re-exported from the agents package.
- All internal imports, examples, and documentation updated.

#### CI Hardening

- **MSRV job** — new `msrv` CI job pins `rustup override set 1.91` and runs `cargo check --workspace`, validating the declared `rust-version` on every push.
- **Stub guard job** — new `stubs` CI job runs `cargo xtask check-stubs crates/ extras/` to fail the build if new `todo!()`/`unimplemented!()`/`FIXME` markers are introduced outside test blocks.
- **Deno check/lint/test job** — new `deno` CI job runs `deno check`, `deno lint`, and `deno test --allow-all` against the `deno/` workspace.
- **`brainwires-channels` dev-dependencies** — added `tokio` (full) and `anyhow` to `[dev-dependencies]` to support the new `mock_channel` example.

#### `xtask` — Autofix Mode

- **`--fix` flag** — `cargo xtask --fix` now auto-heals CI failures. Format issues are fixed by running `cargo fmt --all` directly; check, clippy, test, and doc failures are dispatched to Claude Code CLI (`claude -p`) with captured error output, scoped tool permissions (`Read,Edit,Glob,Grep,Bash(cargo *)`), and a turn limit. Each failed step is re-verified after the fix attempt.
- **`--max-turns <N>`** — configurable turn limit per Claude fix invocation (default: 30). Gracefully skips Claude fixes when the `claude` binary is not on PATH.

### Fixed

- **Clippy warnings** resolved across `brainwires-cli`, `matter-tool`, `brainwires-network`, `brainwires-tools`, and `brainwires-agent`.
- **CI errors from architecture refactor** — fixed broken imports, missing re-exports, and formatting issues introduced during crate consolidation.
- **v0.9.0 release cleanup** — removed stale references, fixed security metadata, and corrected test assertions.
- **A2A event initializers** — added missing `trace_id` and `sequence` fields to `TaskStatusUpdateEvent` and `TaskArtifactUpdateEvent` constructors.

### Removed

- **Stale `persistent_task_manager` comments** in `brainwires-storage/src/lib.rs` — removed phantom TODO and re-export comments referencing a module that was never implemented.
- **Absorbed crates deleted from `crates/`** — `brainwires-channels`, `brainwires-skills`, `brainwires-code-interpreters`, `brainwires-datasets` directories removed after absorption into their parent crates.

## [0.8.0] - 2026-04-03

### Fixed

#### Centralized FastEmbed Model Cache

- **Scattered `.fastembed_cache/` directories eliminated** — FastEmbed ONNX model files (87–759 MB each) were accumulating as `.fastembed_cache/` in whatever the working directory was at runtime, creating duplicate copies across the filesystem. Both `brainwires-storage` and `brainwires-cognition` now write to a single shared location: `~/.brainwires/cache/fastembed/`.
- **`PlatformPaths::default_fastembed_cache_path()`** (`brainwires-storage`) — New utility method returning `~/.brainwires/cache/fastembed/`, consistent with the rest of the framework's use of `~/.brainwires/`.
- **`brainwires-storage` embedding manager** — `FastEmbedManager::with_model()` now sets `options.cache_dir` (previously unset, causing the default CWD-relative cache scatter).
- **`brainwires-cognition` embedding manager** — Unified to use `PlatformPaths::default_fastembed_cache_path()` instead of the old `dirs::cache_dir().join("fastembed")` path (`~/.cache/fastembed/`), so both crates share the same model files.

Existing `.fastembed_cache/` directories in project folders are stale and can be safely deleted.

### Added

#### Magic Number Cleanup

- **Audio PCM normalization** (`brainwires-hardware`) — Bare `32768.0` literals in `vad/mod.rs` and `audio/local/whisper_stt.rs` replaced with named constant `I16_NORMALIZE_DIVISOR: f32 = 32768.0` (2^15, the i16 range divisor for [-1, 1] normalisation).
- **Orchestrator token limit** (`brainwires-cli`) — `let max_tokens = 4096` in `orchestrator.rs` replaced with module-level constant `ORCHESTRATOR_MAX_TOKENS: u32 = 4096`.
- **Model output token comment** (`brainwires-providers`) — Added clarifying comment to `brainwires_http::max_output_tokens()` match block documenting values as 2026-Q1 provider specifications.

#### A2A/ACP Protocol Compliance (`brainwires-a2a`)

- **`A2A_PROTOCOL_VERSION` constant** — `pub const A2A_PROTOCOL_VERSION: &str = "0.3"` added to crate root, targeting the A2A 0.3 spec (post-ACP merger under AAIF/Linux Foundation, December 2025). `AgentInterface::protocol_version` field documentation updated to reference this constant.
- **ACP merger acknowledgement** — ACP (Agent Communication Protocol) merged into A2A under the Linux Foundation's Agentic AI Foundation (AAIF) in December 2025. The `brainwires-a2a` crate is compliant with A2A 0.3.0: all 11 JSON-RPC methods, all 9 task states, full security scheme support (PKCE, mTLS, OAuth2, OIDC), `/.well-known/agent-card.json` discovery endpoint, gRPC service, and REST router are implemented.

#### MCP 2026 Spec Compliance (`brainwires-mcp-server`, `brainwires-mcp`)

- **Streamable HTTP transport** (`brainwires-mcp-server`, feature `http`) — `HttpServerTransport` implements the MCP 2026 stateless HTTP transport: `POST /mcp` for JSON-RPC and `GET /mcp/events` SSE for server-initiated messages. Slots into the existing `ServerTransport` trait, wired with a bounded `mpsc` channel (`REQUEST_CHANNEL_CAPACITY = 128`), configurable request timeout (`REQUEST_TIMEOUT_SECS = 30`), and SSE keep-alive pings (`SSE_KEEPALIVE_INTERVAL_SECS = 15`).
- **MCP Server Cards** (SEP-1649) — `GET /.well-known/mcp/server-card.json` endpoint served by `HttpServerTransport`. Types: `McpServerCard`, `McpToolCardEntry`, `McpAuthInfo`, `McpTransportInfo`. Builder: `build_server_card()`. All re-exported from `brainwires-mcp-server`.
- **RFC9728 OAuth Protected Resource** — `GET /.well-known/oauth-protected-resource` endpoint served by `HttpServerTransport`. `OAuthProtectedResource` type with `resource`, `authorization_servers`, `scopes_supported`, `bearer_methods_supported`.
- **OAuth 2.1 JWT validation middleware** (`brainwires-mcp-server`, feature `oauth`) — `OAuthMiddleware` validates `Authorization: Bearer` JWTs via HS256 (shared secret) or RS256 (RSA public key PEM). Configurable `iss`/`aud` claim enforcement. `initialize` method is always unauthenticated per MCP spec. Validated state is cached per-session in `RequestContext` metadata.
- **MCP Tasks primitive** (SEP-1686) — `McpTaskStore` thread-safe in-memory store with full 5-state lifecycle: `Working → Completed`, `Working → Failed`, `Working → Cancelled`, `Working ↔ InputRequired`. TTL-based expiry with `evict_expired()`. Typed accessors: `complete()`, `fail()`, `cancel()`, `update_state()`. `DEFAULT_MAX_RETRIES = 3`. Re-exported from `brainwires-mcp-server`.
- **HTTP client transport** (`brainwires-mcp`, feature `http`) — `HttpTransport` implements stateless JSON-RPC-over-HTTP: buffers requests in `send_request()`, POSTs to `{base_url}/mcp` in `receive_response()`/`receive_message()`. `Transport::Http(HttpTransport)` variant added. Re-exported as `brainwires_mcp::HttpTransport` (requires both `native` + `http` features).

#### Claude 4.6 + Context Compaction

- **Claude 4.6 model IDs** — Default models updated across the provider registry: Anthropic → `claude-sonnet-4-6`, Bedrock → `anthropic.claude-sonnet-4-6-v1:0`, VertexAI → `claude-sonnet-4-6`. OpenAI Responses API default updated to `gpt-5-mini`.
- **Context compaction handling** (`brainwires-core`, `brainwires-providers`, `brainwires-agent`) — New `StreamChunk::ContextCompacted { summary, tokens_freed }` variant. The Anthropic provider emits it when a `context_window_management_event` arrives mid-stream. `ChatAgent` handles it by replacing conversation history with the system prompt + a synthetic assistant summary message, with a `tracing::info!` log. All other streaming consumers (`brainwires-providers/brainwires_http`, `agent-chat`, `brainwires-cli`) handle the variant as a no-op.

#### EU AI Act Audit Logging (`brainwires-analytics`)

- **`ComplianceMetadata`** — New struct with `data_region`, `pii_present`, `retention_days`, `regulation`, `audit_required` fields. Added as `Option<ComplianceMetadata>` (`#[serde(default)]`) to `ProviderCall` and `AgentRun` event variants — fully backward-compatible with existing serialized events.
- **`AuditExporter`** — Time-range filtered export from `MemoryAnalyticsSink`: `export_json()` (JSON array), `export_csv()` (CSV with `event_type,session_id,timestamp,payload_json` columns), `apply_retention_policy(days)` (removes events older than N days, returns deleted count).
- **`PiiRedactionRules`** / `redact_event()`** — Configurable PII scrubbing: `hash_session_ids` (one-way `DefaultHasher` hash), `redact_prompt_content` (replaces `Custom` payload with `"[REDACTED]"`), `custom_patterns` (substring matching in string fields). `redact_event()` is pure — returns a new scrubbed event leaving the original intact.
- **`MemoryAnalyticsSink` helpers** — Added `deposit()` (sync record), `drain_matching(pred)` (filter-drain), `retain(pred)` (filter-in-place, returns removed count). `DEFAULT_CAPACITY = 1_000` constant re-exported from `brainwires_analytics`.

#### New Crates

- **`brainwires-system`** — Generic OS-level primitives extracted from `brainwires-autonomy`
  - `reactor` feature — cross-platform filesystem event watcher (`FsReactor`, `EventDebouncer`, `ReactorRule`) via `notify 7`
  - `services` feature — controlled systemd / Docker / process management (`SystemdManager`, `DockerManager`, `ProcessManager`, `ServiceSafety` with hardcoded critical-service deny-list)
  - Usable independently; no dependency on the autonomy crate

#### New Extras

- **`brainwires-scheduler`** — Local-machine MCP server for cron-based job scheduling with optional per-job Docker sandboxing
  - 9 MCP tools: `add_job`, `remove_job`, `list_jobs`, `get_job`, `enable_job`, `disable_job`, `run_job`, `get_logs`, `status`
  - Native and optional per-job Docker sandbox execution (`--memory`, `--cpus`, `--network=none`, volume mounts)
  - JSON-backed persistence at `~/.brainwires/scheduler/`; per-run log files with configurable retention (default: 20 per job)
  - Bounded concurrency via semaphore; `Ignore`/`Retry`/`Disable` failure policies; SIGTERM + Ctrl+C graceful shutdown with in-flight drain
  - stdio transport (primary, for Claude Code MCP integration) + optional HTTP via `--http <addr>`
  - 36 unit tests covering executor, store, daemon cron logic, and retry policy permutations

#### WebRTC Real-Time Media (`brainwires-channels`)

- **`webrtc` feature flag** — Full WebRTC peer connection support using the Brainwires fork of `webrtc-rs` (v0.20.0-alpha.1, trait-based async API). Zero impact on compile time or binary size without the feature.
- **`WebRtcSession`** — Manages a single `RTCPeerConnection` with full offer/answer state machine, trickle ICE, DTLS-SRTP, audio/video tracks, and DataChannels. All methods take `&self` for `Arc<WebRtcSession>` sharing across tasks.
  - `open()` / `close()` — create/tear down the underlying PeerConnection
  - `add_audio_track(AudioCodec)` / `add_video_track(VideoCodec)` — add local media before offer creation; returns an `AudioTrack`/`VideoTrack` handle for writing encoded frames
  - `create_offer()` / `create_answer()` / `set_remote_description()` — SDP negotiation
  - `add_ice_candidate()` / `restart_ice()` — trickle ICE and ICE restart
  - `create_data_channel(DataChannelConfig)` — open a WebRTC DataChannel
  - `get_remote_track(id)` — access incoming remote media tracks after `TrackAdded` event
  - `get_stats()` — full `RTCStatsReport` snapshot (jitter, packet loss, RTT, bitrate, jitter buffer, NACK counts, frame stats)
  - `subscribe()` — broadcast receiver for all session events
- **`webrtc-advanced` feature flag** — Adds congestion control and media quality interceptors on top of the default NACK/RTCP chain:
  - **GCC (Google Congestion Control)** — adaptive bitrate estimation from TWCC feedback; configure via `BandwidthConstraints` in `WebRtcConfig`; query via `session.target_bitrate_bps()`
  - **JitterBuffer** — adaptive playout delay, outermost in the receive chain
  - **TwccSender** — transport-wide sequence numbers for GCC feedback loop
  - A `tracing::warn!` is emitted at `open()` time when the feature is absent
- **`WebRtcConfig`** — Fully serde-serializable configuration:
  - `ice_servers` (STUN/TURN), `ice_transport_policy` (All / Relay)
  - `dtls_role` (Auto / Client / Server) — applied via `SettingEngine`
  - `mdns_enabled` — obfuscate LAN IPs with `.local` hostnames
  - `tcp_candidates_enabled` — gather TCP ICE candidates for firewall traversal
  - `bind_addresses` — restrict ICE gathering to specific interfaces (default: `0.0.0.0:0`)
  - `codec_preferences` (`VideoCodec` / `AudioCodec` enums) and `bandwidth` (`BandwidthConstraints`) for GCC
- **`WebRtcSignaling` trait** + two built-in impls:
  - `BroadcastSignaling` — in-process `tokio::broadcast` channel; used by the integration test and gateway intermediation
  - `ChannelMessageSignaling` — encodes SDP/ICE as JSON inside regular `ChannelMessage`s with metadata key `"_bw_webrtc_signaling"`; works through any existing adapter without changes
- **`WebRtcChannel` trait** — extension of `Channel` for adapters that support real-time media: `initiate_session()`, `get_session()`, `close_session()`, `signaling()`
- **`RemoteTrack`** — handle to an incoming remote media track; `poll() -> Option<TrackRemoteEvent>` for reading RTP packets and lifecycle events
- **`RTCStatsReport` / `StatsSelector`** re-exported from `brainwires_channels` root
- **10 new `ChannelEvent` variants** (all `#[cfg(feature = "webrtc")]`): `IceCandidate`, `SdpOffer`, `SdpAnswer`, `TrackAdded`, `TrackRemoved`, `WebRtcDataChannel`, `PeerConnectionStateChanged`, `IceConnectionStateChanged`, `IceGatheringComplete`, `SignalingStateChanged`
- **2 new `ChannelCapabilities` flags**: `DATA_CHANNELS` (bit 12), `ENCRYPTED_MEDIA` (bit 13)
- **Integration test** — `offer_answer_reaches_connected`: two in-process sessions complete a full offer/answer + trickle ICE exchange and both reach `PeerConnectionState::Connected` in ~1.3 s on loopback

### Changed

#### Autonomy (`brainwires-autonomy`)

- **`dream/` extracted → `brainwires-cognition`** (new `dream` feature) — memory consolidation belongs with the knowledge graph and RAG layer, not autonomous operations. Access via `brainwires_cognition::dream` or `brainwires::dream` (meta-crate `dream` feature).
- **`reactor/` + `services/` extracted → `brainwires-system`** — generic OS primitives are now independently usable without pulling in the full autonomy dependency tree. Access via `brainwires_system` or `brainwires::system`.
- **`scheduler/` removed** — superseded by `extras/brainwires-scheduler`, which provides the same functionality as a proper MCP server with a richer job model, persistence, and Docker sandboxing.

## [0.7.0] - 2026-03-31

### Added

#### New Crates

- **`brainwires-analytics`** — Unified analytics collection, persistence, and querying for the framework. `AnalyticsCollector` multi-sink dispatcher with 10 typed event variants: `ProviderCall` (tokens, cost, latency), `AgentRun` (iterations, tool calls, total cost), `ToolCall`, `McpRequest`, `ChannelMessage`, `StorageOp`, `NetworkMessage`, `DreamCycle`, `AutonomySession`, and `Custom` (escape hatch). `AnalyticsLayer` — drop-in `tracing-subscriber` layer that automatically intercepts known span names (`provider.chat`, etc.) without modifying instrumented code. `MemoryAnalyticsSink` — in-process ring buffer. `SqliteAnalyticsSink` + `AnalyticsQuery` (feature `sqlite`) — local SQLite persistence and aggregated reporting: `cost_by_model()`, `tool_frequency()`, `daily_summary()`, `rebuild_summaries()`. All event types are fully serializable.

- **`brainwires-channels`** — Universal messaging channel contract for adapter implementations. Provides `Channel` trait (7 async methods), `ChannelMessage`, `ChannelEvent` (8 variants), `ChannelCapabilities` (12 bitflags), `ChannelUser`, `ChannelSession`, `ConversationId`, and `ChannelHandshake` protocol. Bidirectional conversion between `ChannelMessage` and agent-network `MessageEnvelope`.
- **`brainwires-mcp-server`** — MCP server framework extracted from `brainwires-agent-network`. Provides `McpServer`, `McpHandler` trait, `McpToolRegistry` (declarative tool registration + dispatch), `ServerTransport`/`StdioServerTransport`, and a composable middleware pipeline: `AuthMiddleware`, `LoggingMiddleware`, `RateLimitMiddleware`, `ToolFilterMiddleware`.

#### Agents (`brainwires-agent`)

- **`ChatAgent`** — Reusable streaming completion loop with per-user session management. Methods: `restore_messages()`, `compact_history()`.
- **Session persistence** — `SessionStore` trait + `JsonFileStore` implementation for persisting conversation history across restarts. Wired into BrainClaw via `memory.persist_conversations` config.

#### Tool System (`brainwires-tool-system`)

- **`BuiltinToolExecutor`** — Centralized dispatch executor for all built-in tools, eliminating duplication across agent implementations.
- **Email tools** (feature `email`) — IMAP/SMTP/Gmail read, send, search, and manage operations.
- **Calendar tools** (feature `calendar`) — Google Calendar/CalDAV event creation, listing, and update operations.

#### Code Interpreters (`brainwires-code-interpreters`)

- **Docker sandbox** — Container-isolated code execution via Docker; `Dockerfile.sandbox` at `crates/brainwires-code-interpreters/docker/`.

#### Skills (`brainwires-skills`)

- **`SkillPackage`** — Distributable skill package format with manifest, skill_content, SHA-256 checksum, and optional ed25519 signature.
- **`RegistryClient`** — HTTP client for publishing to and downloading from a skill registry server.
- **ed25519 signing** (feature `signing`) — Sign and verify skill packages for supply-chain safety.

#### Agent Networking (`brainwires-agent-network`)

- **Device allowlists** — `DeviceAllowlist`, `DeviceStatus` (Allowed/Blocked/Pending), `OrgPolicies`. Bridge computes a SHA-256 device fingerprint from machine-id + hostname + OS on every `Register` message; bails on `Blocked` status from server.
- **Sender verification** — Channel-type and channel-ID allowlists enforced at WebSocket handshake time; master `channels_enabled` switch.
- **Permission relay** — `PermissionRequest`/`PermissionResponse` message types. `PermissionRelay` module with pending request map (oneshot channels), session-allowed list, and configurable timeout. `RemoteBridge::send_permission_request()` sends a request and awaits approval; auto-denies on timeout.

#### Hardware (`brainwires-hardware`)

- **Voice Activity Detection** (always available with `audio`) — `VoiceActivityDetector` trait + `EnergyVad` (pure-Rust RMS energy threshold, no extra deps). Feature `vad` adds `WebRtcVad` (three aggressiveness modes: Quality, LowBitrate, Aggressive, VeryAggressive) via `webrtc-vad 0.4`. Helpers: `SpeechSegment`, `rms_db()`, `pcm_to_i16_mono()`, `pcm_to_f32()`.
- **Wake word detection** (feature `wake-word`) — `WakeWordDetector` trait + `WakeWordDetection` event. `EnergyTriggerDetector` — zero-dependency energy-burst trigger (fires when audio energy exceeds a dB threshold for N consecutive 30 ms frames). Optional `wake-word-rustpotter` feature adds `RustpotterDetector` (pure-Rust DTW/ONNX, `.rpw` model files). Optional `wake-word-porcupine` feature adds `PorcupineDetector` (Picovoice, builtin keywords + custom `.ppn` files).
- **Voice assistant pipeline** (feature `voice-assistant`) — `VoiceAssistant` orchestrates the full listen → wake word → VAD-gated capture → STT → handler → TTS → playback loop. `VoiceAssistantBuilder` for composing components. `VoiceAssistantHandler` async trait (`on_wake_word`, `on_speech`, `on_error`). `VoiceAssistantConfig` (silence threshold/duration, max record duration, listen timeout, STT/TTS options, device selection). `AssistantState` enum (Idle/Listening/Processing/Speaking). `listen_once()` for single-shot capture + transcription without handler callbacks.
- **Camera capture** (feature `camera`) — Cross-platform webcam/camera frame capture via `nokhwa` (V4L2 on Linux, AVFoundation on macOS, Media Foundation on Windows). `CameraCapture` async trait, `NokhwaCapture` impl with `spawn_blocking` bridge, `list_cameras()`, `open_camera(index, format)`, automatic MJPEG→RGB decoding. Types: `CameraDevice`, `CameraFrame`, `CameraFormat`, `Resolution`, `FrameRate`, `PixelFormat`, `CameraError`.
- **Raw USB access** (feature `usb`) — Device enumeration and async bulk/control/interrupt transfers via `nusb` (pure Rust, no libusb system dependency). `UsbHandle::open()` auto-discovers bulk endpoints from the interface descriptor. Types: `UsbDevice`, `UsbClass` (full USB-IF class code map), `UsbSpeed`, `UsbError`. `list_usb_devices()` reads string descriptors (manufacturer, product, serial) with graceful permission-error fallback.
- **`brainwires-hardware` renamed from `brainwires-audio`** — Unified hardware abstraction crate. GPIO moved from `brainwires-autonomy`; Bluetooth and Network hardware added. `brainwires-autonomy` re-exports GPIO via `pub use brainwires_hardware::gpio` for backward compatibility.
- **Deprecated `brainwires-audio`** — Stub crate at `deprecated/brainwires-audio`; re-exports `brainwires-hardware` with `audio` feature. Final release for ecosystem continuity.

#### Autonomy (`brainwires-autonomy`)

- **Autodream memory consolidation** (feature `dream`) — 4-phase consolidation cycle: orient → gather → consolidate → prune. Types: `DreamConsolidator`, `DemotionPolicy` (age/importance/budget thresholds), `DreamSummarizer` (LLM-powered compression), `FactExtractor` (5 categories: entities, relationships, events, preferences, habits), `DreamMetrics`, `DreamReport`, `DreamTask` (scheduled via `AutonomyScheduler`).

#### Cognition (`brainwires-cognition`)

- **Hindsight-inspired memory retrieval** — `detect_temporal_query()` scores temporal-intent keywords and dynamically boosts recency weighting in `search_adaptive_multi_factor()`. `CrossEncoderReranker` (implements `DiversityReranker`) blends retrieval scores with query-document cosine similarity via configurable `alpha`; `RerankerKind` supports `Spectral`, `CrossEncoder`, or `Both` (two-pass: diversity then relevance). `RagClient::query_ensemble()` fans out concurrently across `SearchStrategy` variants (`Semantic`, `Keyword`, `GitHistory`, `CodeNavigation`) and fuses results via RRF. `MemoryBankConfig` — mission, content-blocking directives, and five disposition traits (`Analytical`/`Concise`/`Cautious`/`Creative`/`Systematic`, each ±0.1 retrieval score bias) integrated into `BrainClient`. `MultiFactorScore` gains `compute_with_weights()` and `recency_from_hours_fast()`; `TieredMemoryConfig` gains `temporal_boost` and `fast_decay` fields.
- **Evidence tracking** — `Thought` gains `confidence`, `evidence_chain`, `reinforcement_count`, and `contradiction_count` fields. New `check_corroboration()` and `check_contradiction()` functions (negation-heuristic). `BrainClient` gains `apply_evidence_check()` and `replace_thought()`.
- **Mental models tier** — New `MentalModelStore`, `MentalModel`, and `ModelType` enum (`Behavioral`/`Structural`/`Causal`/`Procedural`). `MemoryTier::MentalModel` added at the lowest hierarchy level. `TieredMemory` gains `synthesize_mental_model()` (explicit only — never auto-populated) and `search_mental_models()`; results appended to `search_adaptive_multi_factor()`.

#### Autonomy / Agents — Empirical Evaluation (`brainwires-autonomy`, `brainwires-agent`, `brainwires-cognition`)

- **Empirical eval harness** (feature `eval-driven`) — Zero-network, <1 ms deterministic evaluation cases. Eight cases: `EntityImportanceRankingCase`, `EntitySingleMentionCase`, `EntityTypeBonusCase`, `MultiFactorRankingCase`, `TierDemotionCase`, `TaskBidScoringCase` (0.4×capability + 0.3×availability + 0.3×speed), `ResourceBidScoringCase` (0.7×priority + 0.3×bid), `ComplexityHeuristicCase` (keyword-based task complexity scoring). Suites: `entity_importance_suite()`, `multi_factor_suite()`. New `ranking_metrics` module: `ndcg_at_k()`, `mrr()`, `precision_at_k()` with graded relevance support.

#### Extras — Voice Assistant (`extras/voice-assistant/`)

- **`voice-assistant`** binary — Personal voice assistant built on the framework. Mic capture → optional energy wake trigger → VAD-gated speech accumulation → OpenAI Whisper STT → LLM response (OpenAI chat completions) → OpenAI TTS playback. CLI flags: `--config <path.toml>`, `--list-devices`, `--wake-word <model>`, `--verbose`. TOML config covers STT model, TTS voice, silence tuning, wake word model, LLM model/system prompt, and device names. Clean Ctrl-C shutdown via `tokio::signal`.

#### Extras — BrainClaw Suite (`extras/brainclaw/`)

- **`brainclaw`** (daemon) — Self-hosted personal AI assistant. Multi-provider support (Anthropic, OpenAI, Google, Ollama, Groq, Together, Fireworks, Bedrock, Vertex AI), per-user agent sessions, TOML config (`~/.brainclaw/brainclaw.toml`), native/email/calendar feature flags.
- **`brainwires-gateway`** — WebSocket/HTTP channel hub. `InboundHandler` trait for custom message processing; built-in `AgentInboundHandler` bridging channel events to `ChatAgent` sessions. WebChat browser UI at `/chat` with WebSocket at `/chat/ws`. Admin API (`/admin/*`) with Bearer token auth. Admin browser dashboard at `GET /admin/ui` (single-file dark-themed SPA; sections: Dashboard, Channels, Sessions, Cron Jobs, Identity, Broadcast). Webhook endpoint (`POST /webhook`) with HMAC-SHA256 verification. Media pipeline: attachment download, image description, audio transcription, size validation. Audit logger: structured JSON ring buffer via `tracing`. Metrics: atomic counters for messages, tool calls, errors, rate limits, spoofing blocks, and per-channel breakdowns. `/model` slash command for per-session model switching (`/model list`, `/model <name>`, `/model default`).
- **`brainwires-discord-channel`** — Discord bot adapter (serenity). Reference `Channel` trait implementation. Optional MCP tool server mode (`--mcp`).
- **`brainwires-telegram-channel`** — Telegram bot adapter (teloxide). `Channel` trait implementation, bidirectional gateway relay, optional MCP tool server (`--mcp`).
- **`brainwires-slack-channel`** — Slack adapter using Socket Mode (reqwest, no public URL required). `Channel` trait implementation, optional MCP tool server (`--mcp`).
- **`brainwires-mattermost-channel`** — Mattermost adapter using Mattermost WebSocket API. `Channel` trait implementation with send/edit/delete/history/react. Filtering: self-messages, channel allowlist, @mention requirement, team scoping. Optional MCP tool server (`--mcp`). Capabilities: `RICH_TEXT | THREADS | REACTIONS | TYPING_INDICATOR | EDIT_MESSAGES | DELETE_MESSAGES | MENTIONS`.
- **`brainwires-signal-channel`** — Signal messenger adapter via `signal-cli-rest-api`. WebSocket push mode with polling fallback. `Channel` trait implementation. Filtering: self-messages, sender/group allowlists, @mention/keyword trigger for groups. Optional MCP tool server (`--mcp`): `send_message`, `add_reaction`. Capabilities: `REACTIONS`.
- **`brainwires-skill-registry`** — HTTP skill registry server. SQLite with FTS5 full-text search. Endpoints: publish, search (query + tag filter), get manifest (latest or by version), download package. Auto-creates schema on first run.

#### Extras — Issue Tracker (`extras/brainwires-issues/`)

- **`brainwires-issues`** — Lightweight MCP-native issue tracking server inspired by Linear's agent interface. Serves 10 tools: `create_issue`, `get_issue` (accepts UUID or `#number`), `list_issues` (filters: project, status, assignee, label; offset-based pagination), `update_issue`, `close_issue`, `delete_issue` (optional cascade), `search_issues` (BM25 full-text with in-memory fallback), `add_comment`, `list_comments` (offset pagination), `delete_comment`. Four prompts: `/create`, `/list`, `/search`, `/triage`. Data model: `Issue` with UUID, auto-incrementing display number, title, description, status (Backlog/Todo/InProgress/InReview/Done/Cancelled), priority (NoPriority/Low/Medium/High/Urgent), labels (Vec<String>), assignee, project, parent_id for sub-issues, created/updated/closed timestamps. Comments with author and body. LanceDB backend at `<data_dir>/brainwires-issues/lancedb/`; BM25 full-text index at `<data_dir>/brainwires-issues/bm25/`.

#### Extras — brainwires-cli (`extras/brainwires-cli/`)

- **`brainwires-cli`** migrated into monorepo — The flagship AI-powered agentic CLI (76k lines) moved from a standalone repository with a framework git submodule into `extras/brainwires-cli/` as a root workspace member. Eliminates the two-repo submodule workflow; CI now covers CLI and framework changes together. `agent-chat` remains as the minimal reference implementation.

#### Core Types (`brainwires-core`)

- **`ChatOptions::model`** — New `model: Option<String>` field. When `Some`, all providers (Anthropic, OpenAI, OpenAI Responses, Gemini, Ollama, and OpenAI-compatible) substitute this model for their configured default on that request. Enables per-request and per-session model switching without recreating the provider. `ChatOptions` gains a `.model()` builder method.

### Fixed

#### Storage (`brainwires-storage`)

- **LanceDB 0.27 upgrade** — Bumped `lancedb` from 0.26 to 0.27. Fixed `Scannable` API breaking change: `create_table()` and `add()` now require `T: Scannable`; cast `RecordBatchIterator` to `Box<dyn RecordBatchReader + Send>` at all callsites.
- **SQL injection prevention** — `filter_to_sql()` now backtick-quotes all column names, preventing column identifiers from being misinterpreted as SQL keywords or operators. Three `LanceDatabase` callsites that interpolated user-controlled `project_name` and `root_path` values directly into SQL filter strings have been replaced with typed `Filter::Eq` expressions.
- **BM25 parse errors logged** — `parse_query_lenient()` errors were silently discarded; now logged via `tracing::warn!` so dropped search terms are visible.
- **BM25 schema drift recovery** — Opening an existing BM25 index now validates that all required fields (`id`, `content`, `file_path`) exist. On mismatch (e.g. after a schema change between versions) the stale index is deleted and rebuilt automatically.
- **BM25 silent document loss fixed** — Documents with a missing or corrupt `id` field are now logged (`tracing::warn!`) instead of silently skipped, making index corruption visible.
- **BM25 `STORED` flag added to `content` field** — The `content` field was indexed as `TEXT` only; adding `STORED` allows document content to be retrieved after indexing. Existing indexes are rebuilt automatically via the schema drift check above.

#### Facade (`brainwires`)

- Removed `brainwires-proxy` from the `full` feature flag. Extras are consumers of the framework, not framework dependencies; external consumers (such as `brainwires-cli`) do not have extras in their workspace. The `proxy` feature remains available as an explicit opt-in.

#### Providers (`brainwires-providers`)

- **llama-cpp-2 token API** — Replaced deprecated `token_to_str` with `token_to_piece` to restore compatibility with llama-cpp-2 ≥ 0.9.

#### Analytics (`brainwires-analytics`)

- **Runtime path coverage** — Analytics events wired into all remaining framework paths (Phases 7–9): per-iteration agent events, tool call tracking, MCP request events, and storage operation events.

### Quality

- **Test coverage expansion** — Added ~440 tests across 14 previously untested or undertested crates and extras. Coverage: A2A protocol serialization roundtrips; analytics event construction; brainwires-issues CRUD + BM25 search + pagination; mcp-matrix, mcp-whatsapp, mcp-mattermost, and mcp-signal config serde + protocol parsing + envelope helpers; hardware VAD, Bluetooth, GPIO, and network types via a mock backend; autonomy git workflows, merge policies, and webhook HMAC signatures; mcp-server middleware (auth, rate limiting, logging, connection context); storage BM25/RRF ranking correctness with tempdir-isolated indexes; provider trait contract via a zero-network `MockProvider` integration suite; audio-demo-ffi FFI type conversion roundtrips.

### Refactored

- **Deprecated mesh submodules removed** (`brainwires-agent-network`) — `mesh::discovery`, `mesh::error`, `mesh::node`, and `mesh::routing` deleted. `mesh::federation` and `mesh::topology` updated to use the canonical replacements: `AgentIdentity` (was `MeshNode`) and `NetworkError` (was `MeshError`). Only `FederationGateway`, `FederationPolicy`, `MeshTopology`, and `TopologyType` are now exported from `mesh::*`.

- **BrainClaw workspace** — BrainClaw is now a self-contained Cargo workspace at `extras/brainclaw/`, excluded from the root workspace via `[workspace].exclude`. Members use path dependencies back to `crates/` for framework libraries.
- **Docker Dockerfile** — Moved `extras/docker/Dockerfile.sandbox` to `crates/brainwires-code-interpreters/docker/` where it belongs alongside the crate it supports.
- **`brainwires-mcp-server` extracted** — MCP server framework code was split out of `brainwires-agent-network` into its own publishable crate. `brainwires-agent-network` now depends on `brainwires-mcp-server`; consumers that only need to build MCP servers no longer need to pull in the full networking stack.
- **`brainwires-channels` optional dep** — `brainwires-channels`' dependency on `brainwires-agent-network` is now optional, gated behind the `agent-network` feature flag (conversion module).

## [0.6.0] - 2026-03-23

### Changed

#### A2A Protocol (`brainwires-a2a`, `deno/a2a`)
- **BREAKING:** Updated A2A protocol implementation from v0.3 to v1.0.
- **Part type redesigned:** Replaced discriminated union (`kind: text/file/data`) with unified flat struct (`text`/`raw`/`url`/`data` as optional oneof fields + `mediaType`, `filename`).
- **Enum values → SCREAMING_SNAKE_CASE:** Role (`ROLE_USER`, `ROLE_AGENT`), TaskState (`TASK_STATE_SUBMITTED`, `TASK_STATE_WORKING`, etc.) per ProtoJSON specification.
- **Removed `kind` field** from `Message`, `Task`, and streaming event objects.
- **Stream events use wrapper pattern:** `StreamResponse` with `task`/`message`/`statusUpdate`/`artifactUpdate` wrapper fields instead of `kind`-based discrimination.
- **SecurityScheme and OAuthFlows** changed from `type`-discriminated to wrapper-based oneOf pattern.
- **JSON-RPC method names** updated to PascalCase (`message/send` → `SendMessage`, etc.).
- **REST:** `GET /tasks/{id}:subscribe` changed to `POST`.
- **`SendMessageConfiguration.blocking`** renamed to `returnImmediately`.
- **`PushNotificationConfig.id`** renamed to `configId`, added `createdAt`.
- **`AgentCard.supportedInterfaces`** is now required.
- **New error codes:** `ExtensionSupportRequired` (-32008), `VersionNotSupported` (-32009).

#### Code Interpreters (`brainwires-code-interpreters`)
- Disabled Python/RustPython feature to resolve `libsqlite3-sys` version conflict with `brainwires-cognition`.

## [0.5.0] - 2026-03-15

### Added

#### Autonomy (`brainwires-autonomy`)
- **Crash recovery** (feature `crash-handler`): Detect crashed processes → AI-powered diagnostics → automatic fix → rebuild → relaunch. Persistent recovery state tracking across restarts.
- **CI/CD orchestrator** (feature `cicd`): GitHub Issues → investigate → fix → PR → merge pipeline. Webhook config, variable interpolation, event logging.
- **Cron scheduler** (feature `scheduler`): Recurring autonomous tasks with cron-expression triggers and configurable failure policies (retry, skip, abort).
- **File system reactor** (feature `reactor`): Watch directories with glob-based rules, debounced event dispatch, and rate limiting.
- **Service management** (feature `services`): Manage systemd units, Docker containers, and OS processes with hardcoded deny-list safety and allow-list enforcement.
- **GPIO hardware control** (feature `gpio`): Pin manager with allow-list policies, PWM configuration, and auto-release timeouts.
- 117 tests across all 6 new features; each feature flag compiles independently.

#### Examples
- **16 examples across 9 crates**: permissions (`policy_engine`, `trust_audit`), MDAP (`voting_consensus`, `task_decomposition`), skills (`skill_registry`), code-interpreters (`multi_language`), A2A (`agent_card`), cognition (`prompting_techniques`), autonomy (`safety_guard`), agent-network (`middleware_chain`), and 6 agent coordination patterns (`contract_net`, `saga_compensation`, `task_queue_scheduling`, `optimistic_concurrency`, `three_state_model`, `validation_loop`).
- **10 examples for brainwires-autonomy**: `health_monitor`, `session_metrics`, `crash_recovery`, `self_improve_strategies`, `git_workflow_pipeline`, `cicd_orchestrator`, `cron_scheduler`, `fs_reactor`, `service_manager`, `gpio_pins`.

#### Documentation
- **BYO database guide** (`databases/README.md`): Step-by-step guide for implementing custom `StorageBackend` and `VectorDatabase` backends, with trait method documentation and integration patterns.

#### Crate Merges (19 → 18 crates)
- **`brainwires-mdap`** merged into `brainwires-agent` behind the `mdap` feature flag. The standalone `brainwires-mdap` crate is now deprecated; use `brainwires-agent = { version = "0.5", features = ["mdap"] }` instead.

#### Build & CI (`xtask`)
- **`package-count` command**: `cargo xtask package-count [--dry-run]` counts workspace members (crates vs extras) and updates stale count references in `.md` files. Skips CHANGELOG.md, deprecated directories, code blocks, and historical arrow lines.
- **Deprecated crate publishing**: `publish.sh` now publishes deprecated stub crates (e.g. `brainwires-mdap`) after all workspace crates, with non-fatal error handling.

#### Testing
- **472 integration tests across 6 crates**: agent-network (47), agents (53), audio (93), code-interpreters (142), skills (82), wasm (55).

#### Code Quality
- Resolved all 16 `check-stubs` false-positive warnings by rewording doc comments and adding `todo_scanner.rs` to the skip list.

### Changed

#### Providers (`brainwires-providers`)
- Updated default models: Anthropic now defaults to latest Claude model, OpenAI to latest GPT model.

#### Build & Publishing
- `publish.sh` enhanced with smarter version tagging logic to handle patch bumps correctly.
- Version replacement logic improved to handle doc comments in Rust files.
- README version example updated to 0.4.

#### Documentation
- `brainwires-autonomy` README rewritten with new features, feature flags, examples, and safety documentation.

## [0.4.1] - 2026-03-15

### Added

#### Storage (`brainwires-storage`)
- **`PostgresDatabase` StorageBackend impl** (1900+ lines across all 3 backends):
  - `FieldValue`→`ToSql` type conversion for all 9 field types (including `pgvector::Vector` for embedding columns).
  - `vector_search` via pgvector `<=>` (cosine distance) operator with parameterized SQL.
  - `row_to_record` parser using `tokio_postgres` column type metadata (`Type::TEXT`, `Type::INT4`, `Type::FLOAT8`, `Type::BOOL`, etc.) with automatic pgvector detection for unknown types.
  - Helper functions `field_values_to_params` and `params_as_refs` for ergonomic boxed `ToSql` parameter handling.
  - Full `create_table`, `insert`, `query`, `update`, `delete`, `vector_search` implementations via the shared `PostgresDialect` SQL generator.
- **`MySqlDatabase` backend** via `mysql_async` (~490 lines):
  - Full `StorageBackend` implementation with connection pooling (`mysql_async::Pool`).
  - Connectivity verification on construction (ping + disconnect handshake).
  - Vector columns stored as JSON arrays; `vector_search` performs client-side cosine similarity since MySQL lacks native vector types.
  - SQL generation via the shared `MySqlDialect`.
  - New `mysql-backend` feature flag with `mysql_async` dependency.
- **`SurrealDatabase` backend** via `surrealdb` 2.x SDK (~1160 lines):
  - Both `StorageBackend` and `VectorDatabase` trait implementations.
  - Native MTREE KNN vector search with cosine distance using SurrealDB's vector indexing.
  - `with_config()` constructor for explicit credentials; default `new()` uses `root`/`root`.
  - Client-side BM25 scoring for hybrid (vector + keyword) queries via shared `bm25_helpers`.
  - Glob-based file path filtering via shared `glob_utils`.
  - `DatabaseStats`, `ChunkMetadata`, and `SearchResult` type support for full RAG compatibility.
  - New `surrealdb-backend` feature flag with `surrealdb` dependency.

#### Build & CI (`xtask`)
- **Smart version bumping**: Full workspace-aware version bump system with:
  - `--crates` flag parsing and bump mode detection (full vs patch).
  - Workspace dependency graph construction and cascade logic (bumping a crate also bumps its dependents).
  - Auto-detection of changed crates from `git diff` for selective patch-mode bumping.
  - Reset of explicit version overrides on minor/major bumps.
  - Selective patch-mode version bumping for targeted releases.
  - Wired up full + patch mode execution paths.
- **`check-stubs` command**: Scans all `.rs` files for hard blockers (`todo!()`, `unimplemented!()`) and soft markers (`FIXME`, `HACK`, `XXX`, `STUB`, `STOPSHIP`, `"not implemented"`). Skips test code, uses word-boundary detection to avoid false positives. Supports `--strict` (markers = errors) and `--verbose` flags.
- **CHANGELOG stamping**: `bump-version` now renames `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and inserts a fresh empty `## [Unreleased]` section above it.

### Removed

#### Storage (`brainwires-storage`)
- Removed `MySqlDatabase` and `SurrealDatabase` stub backends (contained `todo!()` placeholders), replaced by real implementations (see Added above).
- SQL dialect files (`sql/mysql.rs`, `sql/surrealdb.rs`) retained for future use.

### Changed

#### Storage (`brainwires-storage`)
- `databases/mod.rs` updated with conditional module exports for `mysql` and `surrealdb` behind their respective feature flags.
- `lib.rs` updated to re-export new database modules.
- `sql/mod.rs` documentation updated to reference all three SQL dialect implementations.
- README updated with MySQL and SurrealDB backend entries in the database matrix.

#### Dependencies
- Added `mysql_async` (feature `mysql-backend`) for MySQL/MariaDB connection pooling.
- Added `surrealdb` (feature `surrealdb-backend`) for SurrealDB 2.x SDK integration.

#### Documentation
- Updated `PUBLISHING.md` with smart version bump instructions and `check-stubs` checklist wording.

#### Code Quality
- Applied formatting improvements across the workspace for consistency and readability.

## [0.4.0] - 2026-03-14

### Breaking Changes

#### Storage (`brainwires-storage`)
- **Unified database layer**: Merged `clients/` (7 VectorDatabase impls) and `stores/backends/` (StorageBackend impl) into a single `databases/` module. One struct per database, one shared connection, implementing `StorageBackend` and/or `VectorDatabase`.
- Removed `clients/` module entirely — all database implementations now live in `databases/<name>/`.
- Removed `stores/backend.rs`, `stores/backends/`, `stores/lance_client.rs` — merged into `databases/lance/`.
- Renamed all database structs: `LanceVectorDB` → `LanceDatabase`, `QdrantVectorDB` → `QdrantDatabase`, `PostgresVectorDB` → `PostgresDatabase`, `PineconeVectorDB` → `PineconeDatabase`, `MilvusVectorDB` → `MilvusDatabase`, `WeaviateVectorDB` → `WeaviateDatabase`, `NornicVectorDB` → `NornicDatabase`.
- `LanceBackend` merged into `LanceDatabase` — implements both `StorageBackend` and `VectorDatabase` on a single `lancedb::Connection`.
- PostgreSQL backend switched from `sqlx` to `tokio-postgres` + `deadpool-postgres` to avoid `libsqlite3-sys` version conflict with `rusqlite`.

#### Cognition (`brainwires-cognition`)
- `RagClient` now stores `Arc<dyn VectorDatabase>` instead of concrete database types. Added `with_vector_db()` constructor for external injection.
- `BrainClient` rewritten to use `Arc<dyn StorageBackend>` instead of raw LanceDB/arrow APIs. Added `with_backend()` constructor.
- `u64` fields in PKS/BKS cache now cast through `i64` for `rusqlite` 0.38 compatibility.

### Added

#### Storage (`brainwires-storage`)
- **`databases/` module** — unified database layer with:
  - `traits.rs`: `StorageBackend` + `VectorDatabase` traits (always available, no feature gate)
  - `types.rs`: `FieldDef`, `FieldType`, `FieldValue`, `Record`, `ScoredRecord`, `Filter` types
  - `capabilities.rs`: `BackendCapabilities` struct for runtime feature detection
  - `sql/`: Shared SQL generation layer with `SqlDialect` trait + `PostgresDialect`, `MySqlDialect`, `SurrealDialect` implementations
  - `lance/`: `LanceDatabase` (both traits, embedded LanceDB)
  - `postgres/`: `PostgresDatabase` (VectorDatabase, via tokio-postgres + pgvector)
  - `qdrant/`: `QdrantDatabase` (VectorDatabase)
  - `pinecone/`: `PineconeDatabase` (VectorDatabase, REST API)
  - `milvus/`: `MilvusDatabase` (VectorDatabase, REST API)
  - `weaviate/`: `WeaviateDatabase` (VectorDatabase, REST API)
  - `nornicdb/`: `NornicDatabase` (VectorDatabase, multi-transport: REST/Bolt/gRPC)
- New feature flags: `postgres-backend` (alongside existing `lance-backend`, `qdrant-backend`, `pinecone-backend`, `weaviate-backend`, `milvus-backend`, `nornicdb-*`).
- `async-trait` is now a required (non-optional) dependency — core traits are always available regardless of feature flags.
- 112 tests: 18 SQL dialect tests, Lance CRUD/vector-search/capabilities/shared-connection tests, 2 integration tests (trait object CRUD, backend capabilities).

#### Cognition (`brainwires-cognition`)
- `RagClient::with_vector_db()` — construct with any `Arc<dyn VectorDatabase>` for backend-agnostic RAG.
- `BrainClient::with_backend()` — construct with any `Arc<dyn StorageBackend>` for backend-agnostic knowledge storage.

### Changed

#### Storage (`brainwires-storage`)
- Domain stores (`MessageStore`, `ConversationStore`, `TaskStore`, `PlanStore`, `SummaryStore`, `FactStore`, `ImageStore`, `TierMetadataStore`, `AgentStateStore`) now default to `LanceDatabase` instead of the removed `LanceBackend`.
- `PersistentTaskManager` and `TieredMemory` updated to use `LanceDatabase`.
- README rewritten with unified database backends section, trait implementation matrix, connection sharing examples, and feature flag reference.
- Module-level and crate-level documentation updated to reflect new architecture.

#### Dependencies
- Replaced `sqlx` with `tokio-postgres` 0.7 + `deadpool-postgres` 0.14 for PostgreSQL backend (eliminates `libsqlite3-sys` conflict).
- `pgvector` features changed from `["sqlx"]` to `["postgres"]`.
- Removed unused `sqlx-sqlite` patch from workspace `[patch.crates-io]`.

### Removed

#### Storage (`brainwires-storage`)
- `clients/` module (7 files + tests) — replaced by `databases/`.
- `stores/backend.rs` — split into `databases/traits.rs` + `databases/types.rs`.
- `stores/backends/` — merged into `databases/lance/`.
- `stores/lance_client.rs` — legacy `LanceClient` replaced by `LanceDatabase`.

---

### Added

#### Agent Network (`brainwires-agent-network`)
- **5-layer protocol stack** for pluggable agent networking: Identity → Transport → Routing → Discovery → Application.
- **Identity layer**: `AgentIdentity`, `AgentCard` (capabilities, protocols, metadata, endpoint), `ProtocolId`, `SigningKey`/`VerifyingKey` (ChaCha20-Poly1305 with SHA-256 key derivation).
- **Transport layer**: `Transport` trait with 5 implementations:
  - `IpcTransport` (feature `ipc-transport`) — Unix-socket with optional ChaCha20-Poly1305 encryption.
  - `RemoteTransport` (feature `remote-transport`) — HTTP POST with `tokio::broadcast` receive channel.
  - `TcpTransport` (feature `tcp-transport`) — length-prefixed JSON over TCP with Nagle disabled.
  - `PubSubTransport` (feature `pubsub-transport`) — in-process topic-based messaging via `tokio::broadcast`.
  - `A2aTransport` (feature `a2a-transport`) — A2A protocol via `brainwires-a2a` client.
- **Routing layer**: `Router` trait with `DirectRouter`, `BroadcastRouter`, `ContentRouter`, and `PeerTable` for peer/topic tracking.
- **Discovery layer**: `Discovery` trait with `ManualDiscovery` (in-memory) and `RegistryDiscovery` (HTTP REST, feature `registry-discovery`).
- **Application layer**: `NetworkManager` and `NetworkManagerBuilder` tying all layers together with `send()`, `broadcast()`, and event subscription.
- Core network types: `MessageEnvelope`, `MessageTarget` (Direct/Broadcast/Topic), `Payload` (Json/Binary/Text), `NetworkEvent`, `NetworkError`, `TransportType`, `ConnectionState`.
- New feature flags: `ipc-transport` (default), `remote-transport` (default), `tcp-transport`, `pubsub-transport`, `a2a-transport`, `mesh` (includes `tcp-transport`), `registry-discovery`, `full`.
- 74 new tests across all protocol stack layers.

### Changed

#### Agent Network (`brainwires-agent-network`)
- Renamed `src/transport.rs` (MCP-specific `ServerTransport`) to `src/mcp_transport.rs` to avoid conflict with the new `transport/` module. `ServerTransport` and `StdioServerTransport` are still re-exported from the crate root.
- Updated `mesh/` module with deprecation notices pointing to the new protocol-layer equivalents.
- Default features now include `ipc-transport` and `remote-transport`.

## [0.3.0] - 2026-03-12

### Breaking Changes

#### Crate Merges (23 → 19 crates)

| Old Crate | Merged Into | Migration |
|-----------|-------------|-----------|
| `brainwires-brain` | `brainwires-cognition` | `use brainwires_brain::*` → `use brainwires_cognition::knowledge::*` (feature `knowledge`) |
| `brainwires-prompting` | `brainwires-cognition` | `use brainwires_prompting::*` → `use brainwires_cognition::prompting::*` (feature `prompting`) |
| `brainwires-rag` | `brainwires-cognition` | `use brainwires_rag::*` → `use brainwires_cognition::rag::*` (feature `rag`) |
| `brainwires-relay` | `brainwires-agent-network` | `use brainwires_relay::*` → `use brainwires_agent_network::*` (feature `server`) |
| `brainwires-mesh` | `brainwires-agent-network` | `use brainwires_mesh::*` → `use brainwires_agent_network::mesh::*` (feature `mesh`) |
| `brainwires-seal` | `brainwires-agent/seal/` | `use brainwires_seal::*` → `use brainwires_agent::seal::*` (feature `seal`) |

#### Feature Flag Removals
- Removed zero-dependency feature flags that added no conditional compilation value.
- Fixed import paths across all crates affected by feature flag removal.

### Added

#### Cognition (`brainwires-cognition`)
- New unified intelligence crate combining knowledge graphs, adaptive prompting, RAG, spectral math, and code analysis.
- **Knowledge subsystem** (from `brainwires-brain`): `BrainClient`, thought capture, PKS/BKS, entity graphs, semantic memory search.
- **Prompting subsystem** (from `brainwires-prompting`): 15 techniques in 4 categories, task clustering, temperature optimization, learning coordinator.
- **RAG subsystem** (from `brainwires-rag`): `RagClient`, codebase indexing, AST-aware chunking, hybrid vector + BM25 search, git history search, code navigation.
- **Spectral subsystem**: MSS-inspired spectral subset selection for diverse RAG retrieval using log-determinant diversity scoring.
- **Spectral graph operations** (`spectral::graph_ops`): Laplacian construction, Fiedler vector via inverse power iteration, spectral clustering (recursive bisection), algebraic connectivity, effective resistance, Spielman-Srivastava-inspired sparsification, and spectral centrality/bisection — extends spectral methods beyond RAG to general graph analysis.
- **Spectral methods on `RelationshipGraph`**: `spectral_clusters(k)` for semantic community detection within connected components, `spectral_central_nodes(limit)` for structural bridge-node identification, `connectivity()` for graph health monitoring via algebraic connectivity, and `sparsify(epsilon)` for pruning redundant edges while preserving spectral properties. All feature-gated under `spectral`.
- Feature flags: `knowledge` (default), `prompting` (default), `rag`, `spectral`, `code-analysis`, `tree-sitter-languages`, `native` (everything), `wasm`.

#### Agents (`brainwires-agent`)
- **Planner-Worker-Judge cycle orchestration**: Plan→Work→Judge loop for scaling multi-agent coding tasks, inspired by Cursor's planner-worker pipeline pattern. Each cycle: a `PlannerAgent` explores the codebase and creates dynamic tasks, workers execute them via `TaskOrchestrator` with dependency-aware scheduling, and a `JudgeAgent` evaluates results with structured verdicts (Complete, Continue, FreshRestart, Abort).
  - `planner_agent`: LLM-powered dynamic task planner with JSON output parsing, sub-planner recursion, and cycle detection on the task graph.
  - `judge_agent`: LLM-powered cycle evaluator with structured verdict types.
  - `cycle_orchestrator`: Full Plan→Work→Judge loop with fresh `TaskManager` per cycle, configurable `max_cycles`/`max_workers`, and worktree integration prep.
  - New system prompts: `planner_agent_prompt()` and `judge_agent_prompt()`.
  - `spawn_agent_with_context()` on agent pool for per-worker custom `AgentContext`.
  - New communication messages: `CycleStarted`, `CycleCompleted`, `PlanCreated`, `WorkerBranchMerged`.
- **SEAL integration**: Moved `brainwires-seal` into `brainwires-agent/seal/` as a feature-gated module.
  - Feature flags: `seal`, `seal-mdap`, `seal-knowledge`, `seal-feedback`.
  - `SealKnowledgeCoordinator` now integrates with `brainwires-cognition` instead of `brainwires-brain`.
- 4 standalone examples added for agent usage patterns.

#### Agent Network (`brainwires-agent-network`)
- New crate formed by merging `brainwires-relay` (MCP server framework, encrypted IPC, remote bridge) and `brainwires-mesh` (distributed mesh networking).
- Feature flags: `server` (default), `client` (default), `mesh`, `auth-keyring`.

#### Storage (`brainwires-storage`)
- New `vector-db` feature: vector database trait + backends (LanceDB, Qdrant), BM25 keyword search, glob/path utilities — used by `brainwires-cognition` RAG subsystem.
- Removed `agents` feature and `PersistentTaskManager` (decoupled from agents layer).

#### Build & CI
- `xtask ci` command for local CI: runs `cargo fmt --check`, `cargo clippy`, and `cargo test` in a single command via the xtask pattern (`cargo xtask ci`). Added `.cargo/config.toml` alias and updated `CONTRIBUTING.md` with usage instructions.

#### Licensing
- Added Apache 2.0 and MIT license files to all crates for compliance and distribution.

### Changed

#### Framework-wide
- Reduced crate count from 23 to 19 through strategic merges (see Breaking Changes above).
- Updated all cross-crate import paths for merged crates.
- Updated all README files to reflect post-merge crate structure and integrated documentation from dissolved crates.
- Updated workspace dependency tree in `crates/README.md`.

## [0.2.0] - 2026-03-09

### Changed

#### Framework-wide
- Removed hardcoded crate counts from `CONTRIBUTING.md` and `crates/README.md` to avoid staleness.
- Replaced inline crate listing in `CONTRIBUTING.md` with links to `README.md`, `crates/README.md`, and `extras/README.md`.
- Removed extras table from `crates/README.md`; extras are now documented in their own `extras/README.md`.
- Applied `cargo fmt --all` across workspace.

### Added

#### SEAL (`brainwires-seal`)
- **Feedback Bridge** (`feedback_bridge.rs`): New module that wires `AuditLogger` user feedback (thumbs-up/down + corrections) into the SEAL learning loop. `FeedbackBridge` pulls `FeedbackSignal` events on demand and converts them into `LearningCoordinator` outcomes and `PatternHint` entries in global memory.
- New `feedback` feature gate (`dep:brainwires-permissions`, `dep:tokio`) keeps the `AuditLogger` dependency optional.
- 7 unit tests covering per-run processing, recent-feedback queries, correction application, and run isolation.

#### Facade (`brainwires`)
- `learning` convenience feature now includes `permissions` and `brainwires-seal/feedback`, completing the full feedback loop: `AuditLogger → FeedbackBridge → LearningCoordinator → BKS promotion`.

### Changed

#### Framework-wide
- **MSRV bumped from 1.88 to 1.91** — required by updated AWS SDK dependencies (`aws-config`, `aws-sigv4`, `aws-smithy-*`, etc.).
- Updated CI toolchain from Rust 1.88 to 1.91 across all 5 GitHub Actions jobs.
- Added `protoc` installation step to CI (required by `lance-encoding` build dependency).
- Applied `cargo fmt --all` across workspace.

#### Dependencies
- **rmcp** 0.8 → 1.1 (non-exhaustive structs, renamed features/types)
- **tokio-tungstenite** 0.21 → 0.26 (`Message::Text` now wraps `Utf8Bytes`)
- **rand** 0.8 → 0.10 (`thread_rng` → `rng`, `RngCore` → `Rng`, `gen_range` → `random_range`)
- **bincode** 1 → 2 (new serde encode/decode API)
- **serde_yaml** → **serde_yml** 0.0.12 (crate rename)
- **tonic** 0.12 → 0.13, **prost** 0.13 → 0.14 (removed `async_trait` macro)
- **lancedb** 0.23 → 0.26, **arrow** 56 → 57
- **toml** 0.8 → 1.0, **git2** 0.19 → 0.20, **lru** 0.12 → 0.16
- **boa_engine** 0.20 → 0.21, **tokenizers** 0.21 → 0.22, **tiktoken-rs** 0.7 → 0.9

### Fixed
- Fixed invalid crates.io category slug (`science::ml` → `artificial-intelligence`) on `brainwires-training`.
- Updated publish script rate limits for existing-crate version publishes (burst 30, then 1/min).

## [0.1.0] - 2026-03-09

### Added

#### A2A (`brainwires-a2a`)
- New crate: full Agent-to-Agent protocol — JSON-RPC 2.0, HTTP/REST, and gRPC bindings.
- `A2aClient` with unified transport selection, `A2aServer` with `A2aHandler` trait.
- AgentCard discovery at `/.well-known/agent-card.json`, SSE streaming, push notification CRUD.
- gRPC support via tonic-build from official `a2a.proto` with full type conversions.
- 71 tests covering serde roundtrips, SSE parsing, streaming, HTTP integration.

#### Core (`brainwires-core`)
- `Provider` trait with streaming support (`stream_chat`) and `ChatOptions` builder
- `Message`, `Role`, `ContentBlock`, `ChatResponse`, `StreamChunk` types
- `Tool`, `ToolUse`, `ToolResult`, `ToolRegistry` for tool definitions
- `EmbeddingProvider` trait with batch support
- `VectorStore` trait (backend-agnostic vector database interface)
- `Task`, `WorkingSet`, `PlanMetadata` types
- `FrameworkError` hierarchy with `thiserror`
- Graph types: `GraphNode`, `GraphEdge`, `EntityType`, `EdgeType`

#### Providers (`brainwires-providers`)
- Anthropic, OpenAI, Google (Gemini), Ollama provider implementations
- Groq, Together, Fireworks, Anyscale via OpenAI-compatible protocol
- `ChatProviderFactory` for dynamic provider creation from config
- Rate limiting, model listing, streaming responses
- Optional local LLM support via `llama-cpp-2` feature
- Optional Bedrock and Vertex AI authentication
- Ollama multimodal image support (base64 extraction from `ContentBlock::Image`)
- **OpenAI Responses API**: Full-spec coverage — all 7 tool types, 11 output item types, 35+ streaming event types, structured outputs, reasoning config, and all 6 REST endpoints.
- `ProviderType::OpenAiResponses` with registry entry, factory integration, model listing support, and `base_url` passthrough.
- Response ID tracking for automatic conversation chaining.

#### Agents (`brainwires-agent`)
- `AgentRuntime` with communication hub and file lock coordination
- `TaskManager` and `TaskQueue` for agent task lifecycle
- `ValidationConfig` with file existence, syntax, duplicate, and build checks
- `AccessControlManager` with contention strategies
- `GitCoordinator` for multi-agent git operations
- `PlanExecutorAgent` for structured plan execution
- Extended reasoning support (feature-gated)
- Evaluation framework for benchmarking (feature-gated)
- **Workflow Graph Builder**: Declarative DAG workflows with `WorkflowBuilder`, parallel fan-out/fan-in, conditional routing, shared `WorkflowContext` state, and failure propagation. Topological validation via `petgraph`.
- **Named Reasoning Strategies** (feature-gated `reasoning`): `ReActStrategy`, `ReflexionStrategy`, `ChainOfThoughtStrategy`, `TreeOfThoughtsStrategy` — each with system prompts, completion detection, and step limits. `StrategyPreset` enum for factory creation.
- **OpenTelemetry Export** (feature-gated `otel`): `export_to_otel()` maps `ExecutionGraph` to hierarchical OTel spans (`agent.run` → `agent.iteration.N` → `agent.tool.name`). `telemetry_attributes()` for attaching metrics to existing spans.
- `AgentLifecycleHooks` trait with 10 hook points: before/after iteration, provider call, tool execution, completion, and context pressure.
- `ToolDecision::Delegate` for sub-agent spawning, `ConversationView` for history manipulation, `DefaultDelegationHandler` wrapping `AgentPool`.
- `#[non_exhaustive]` on `AgentContext` and `TaskAgentConfig`.

#### MDAP (`brainwires-mdap`)
- Multi-Dimensional Adaptive Planning with k-agent voting
- `Composer` for aggregating multi-agent results
- `FirstToAheadByKVoter` voting strategy
- Red flag validation and microagent configuration
- Recursive task decomposition

#### Brain (`brainwires-brain`)
- Personal Knowledge Store (PKS) and Behavioral Knowledge Store (BKS)
- Entity extraction and relationship graphs
- Persistent thought storage
- Knowledge integration with prompting system

#### Storage (`brainwires-storage`)
- LanceDB-backed tiered memory (hot/warm/cold)
- Semantic search across conversation history
- Lock store for concurrent access

#### Prompting (`brainwires-prompting`)
- `PromptGenerator` with technique library
- `TemperatureOptimizer` for adaptive temperature selection
- `TaskClusterManager` for grouping similar tasks
- Knowledge-aware prompt construction (feature-gated)

#### Permissions (`brainwires-permissions`)
- `PolicyEngine` with capability profiles
- `TrustManager` with trust levels and escalation
- `AuditLogger` for security audit trails
- Anomaly detection for unusual tool usage

#### Model Tools (`brainwires-tool-system`)
- File operations (read, write, edit, delete, list)
- Bash command execution
- Git operations
- Web fetch and search
- Code search with semantic queries
- Validation tools (syntax, duplicates, build)
- Tool orchestration engine (feature-gated)
- Smart router for tool selection (feature-gated)
- **OpenAPI Tool Generation** (feature-gated `openapi`): `openapi_to_tools()` parses OpenAPI 3.x JSON/YAML specs into `Tool` definitions. `execute_openapi_tool()` handles path/query param substitution and Bearer/API-key/Basic auth.

#### MCP (`brainwires-mcp`)
- MCP client for connecting to external MCP servers
- `McpConfigManager` for server configuration

#### Relay (`brainwires-relay`)
- MCP server mode for exposing agents as tools
- IPC and remote relay for cross-process communication
- Agent-to-Agent (A2A) protocol support (feature-gated)
- Heartbeat monitoring and attachment transfer

#### RAG (`brainwires-rag`)
- AST-aware code chunking with tree-sitter
- Hybrid vector + BM25 keyword search
- Git-aware indexing with blame and history
- LanceDB and Qdrant vector backends
- Relation extraction and storage
- MCP server integration
- `indexed_at` field on `SearchResult` — exposes the chunk indexing timestamp (Unix epoch seconds) from the vector database.
- Upgraded `zip` dependency from v2 to v8 (pure-Rust `lzma-rust2`).

#### Skills (`brainwires-skills`)
- Pluggable skill definitions
- Slash command registry

#### Code Interpreters (`brainwires-code-interpreters`)
- Sandboxed JavaScript execution (Rhai)
- Sandboxed Lua execution
- Python and additional language support (feature-gated)

#### WASM (`brainwires-wasm`)
- Browser-compatible WASM bindings for core agent functionality

#### SEAL (`brainwires-seal`)
- Self-Evolving Agentic Learning system
- Feedback-driven prompt improvement
- Coreference resolution and query analysis
- Knowledge integration (feature-gated)
- Structured `PatternHint` for BKS-to-SEAL pattern transfer
- `QueryCore::resolved` field for tracking coreference-resolved queries
- Execution timing propagation through `record_outcome`

#### Mesh (`brainwires-mesh`)
- Distributed agent mesh networking
- Topology management (star, ring, full mesh)
- Message routing with configurable strategies
- Peer discovery protocols
- Federation gateways for cross-mesh communication

#### Hardware (`brainwires-hardware`)
- Hardware audio capture and playback (CPAL)
- Speech-to-text and text-to-speech traits
- FLAC encoding/decoding support
- Local STT support (feature-gated)
- Unit tests for types, device, and error modules

#### Datasets (`brainwires-datasets`)
- JSONL I/O for training data
- Tokenization (HuggingFace tokenizers, tiktoken)
- Deduplication pipelines
- Format conversion between training formats

#### Training (`brainwires-training`)
- Cloud fine-tuning for 6 providers (OpenAI, Anthropic, Google, Together, Fireworks, Anyscale)
- Local LoRA/QLoRA/DoRA training via Burn
- Training job management and monitoring
- **BPE tokenizer integration**: `Tokenizer` trait with `ModelTokenizer` (HuggingFace `tokenizers` crate) and `SimpleTokenizer` (byte-level fallback). New `tokenizer_path` config option on `LocalTrainingConfig`.
- **SafeTensors model weight loading**: `weight_loader.rs` with `SafeTensorsLoader` for loading pre-trained base weights (f32/f16/bf16 dtype conversion). `LoraLinearConfig::init_with_base_weights()` and `DoraLinearConfig::init_with_base_weights()`.
- **QLoRA quantized base weight loading**: `QLoraLinear` and `QLoraLinearConfig` Burn modules with `init_quantized()` for INT4/INT8 dequantized base weights. Full training loop in `train_qlora()`.
- **DPO/ORPO alignment training**: `PreferenceExample` and `PreferenceDataset` (JSONL: `{"prompt", "chosen", "rejected"}`). `train_dpo_alignment()` with frozen reference model and `train_orpo_alignment()` with single-pass odds ratio loss.
- `TrainingError::NotImplemented` variant for clear stub errors on unimplemented provider methods.
- Dataset loading: JSONL parser supporting prompt/completion and chat message formats (`dataset_loader.rs`).
- Learning rate scheduling: warmup phase + constant/linear/cosine/cosine-warm-restarts strategies (`lr_schedule.rs`).
- Multi-adapter dispatch: LoRA and DoRA training paths with QLoRA/QDoRA fallbacks.
- Validation loop: optional eval dataset evaluated each epoch during local training.
- Weight serialization: adapter weights (A, B, magnitude) written as binary for export.
- Token count tracking in training metrics.
- Weight accessor methods on `LoraLinear` and `DoraLinear` for export.

#### Autonomy (`brainwires-autonomy`)
- Self-improvement strategies
- Evaluation-driven optimization
- Supervisor agent patterns
- Attention mechanisms for context prioritization
- Unit tests for config, error, metrics, attention, health, parallel, training loop, forge, branch manager, investigator, and strategies

#### Facade (`brainwires`)
- Unified re-exports of all 22 sub-crates via feature flags
- `prelude` module with commonly needed types
- Convenience feature bundles: `full`, `researcher`, `agent-full`, `learning`

### Changed
- Upgraded `#![warn(missing_docs)]` to `#![deny(missing_docs)]` across all 22 crates
- Added doc comments to all previously undocumented public items (~155 warnings resolved)

### Refactored
- Renamed `brainwires-model-tools` to `brainwires-tool-system` to better reflect the crate's scope (registry, execution, built-in implementations, error taxonomy, sanitization, orchestration, code execution, semantic search, OpenAPI generation, smart routing).

#### Agents (`brainwires-agent`)
- Replaced `panic!()`/`unwrap()` in eval suite with graceful `TrialResult::failure` conversions.
- Implemented `TextMerge` (line-by-line dedup) and `JsonMerge` (recursive deep merge) optimistic concurrency strategies.
- Replaced silent `let _ =` broadcast/send drops with `tracing::warn` logging across contract_net, task_orchestrator, and validator_agent.

#### Providers (`brainwires-providers`)
- Refactored monolithic `openai_responses/mod.rs` into structured modules (`client.rs`, `convert.rs`, `provider.rs`, `types/`).
- 54 new tests covering serde round-trips for all wire types.

#### Training (`brainwires-training`)
- Upgraded Burn from 0.16 to 0.20. Switched from umbrella `burn` crate to individual crates (`burn-core`, `burn-nn`, `burn-optim`, `burn-autodiff`, `burn-wgpu`, `burn-ndarray`) to avoid `cubecl-cpu` links="lzma" conflict with `xz2` from datafusion/lancedb.
- Fixed `squeeze`/`unsqueeze` API calls for Burn 0.19+ compatibility.
- Added `extern crate burn_core as burn` shim for derive macro resolution.
- Cloud providers (Together, Fireworks, Anyscale): extracted `extract_error()` and `parse_job_status()` helpers; `list_jobs()` now parses actual job status instead of hardcoding `Pending`.
- Cloud providers (Bedrock, Vertex): all methods now return explicit `TrainingError::NotImplemented` errors instead of ad-hoc strings.

#### Framework-wide
- Production-readiness audit across 15 crates (40 files): replaced 121 `unwrap()` calls with `context()`/`expect()`/`LazyLock`; fixed 10 clippy warnings; removed 3 deprecated zero-caller functions; removed 3 dead code items; resolved 2 TODO comments.

### Fixed

#### A2A (`brainwires-a2a`)
- Capped SSE stream buffers at 16MB to prevent unbounded memory growth.
- Added bearer token auth on all transports.
- Fixed gRPC error code mapping, mutex for streaming, and bind error propagation.
- Added CORS headers, resilient accept loop, and graceful shutdown.
- Incremental SSE parser with multi-line data support.

#### Hardware (`brainwires-hardware`)
- Proper error handling for non-UTF-8 model paths in `WhisperStt`.

#### RAG (`brainwires-rag`)
- Fixed use-after-move of `symbol_name` in `find_references`.
- Git search results now return the actual commit date instead of hardcoded `0`.
- Dirty flag is now cleared immediately after embeddings + cache are flushed to disk in both full and incremental indexing paths.

[0.4.1]: https://github.com/Brainwires/brainwires-framework/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Brainwires/brainwires-framework/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Brainwires/brainwires-framework/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Brainwires/brainwires-framework/releases/tag/v0.2.0
[0.1.0]: Untagged initial release
