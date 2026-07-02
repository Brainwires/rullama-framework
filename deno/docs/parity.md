# Parity matrix — Deno port vs. Rust crates

How this doc works: for every Rust crate under `crates/`, we list the Deno
package (if any) and the status. A **partial** crate links to the `SKIPPED.md`
under the corresponding package so it's clear what stays Rust-side and why.

Run `deno task parity` (see `scripts/parity.ts`) to regenerate the crate ↔
package diff from the actual filesystem.

## Summary

The v0.11.0 restructure split the old monolithic crates
(`rullama-providers`, `rullama-agents`, `rullama-tools`) into focused
packages and renamed several others. The current package set is 27 packages,
each 1:1 with a Rust crate under `crates/`. Run `deno task parity` to
regenerate this diff.

| Rust crate                | Deno package                                                       | Status                                 |
| ------------------------- | ------------------------------------------------------------------ | -------------------------------------- |
| `rullama-a2a`             | [`@rullama/a2a`](../packages/a2a/)                                 | Faithful (no gRPC, by design).         |
| `rullama-agent`           | [`@rullama/agent`](../packages/agent/)                             | Faithful — coordination primitives.    |
| `rullama-call-policy`     | [`@rullama/call-policy`](../packages/call-policy/)                 | Faithful (was `resilience`).           |
| `rullama-core`            | [`@rullama/core`](../packages/core/)                               | Faithful.                              |
| `rullama-eval`            | [`@rullama/eval`](../packages/eval/)                               | Faithful — evaluation harness.         |
| `rullama-finetune`        | [`@rullama/finetune`](../packages/finetune/)                       | Partial — cloud slice only (was `training`). |
| `rullama-inference`       | [`@rullama/inference`](../packages/inference/)                     | Faithful — TaskAgent/Chat/Planner/etc. |
| `rullama-knowledge`       | [`@rullama/knowledge`](../packages/knowledge/)                     | Partial — RAG/BKS/PKS are client-only. |
| `rullama-mcp-client`      | [`@rullama/mcp-client`](../packages/mcp-client/)                   | Faithful (was `mcp`).                  |
| `rullama-mcp-server`      | [`@rullama/mcp-server`](../packages/mcp-server/)                   | Faithful — own package (unfolded in v0.11.0). |
| `rullama-mdap`            | [`@rullama/mdap`](../packages/mdap/)                               | Faithful — MDAP/MAKER voting.          |
| `rullama-memory`          | [`@rullama/memory`](../packages/memory/)                           | Faithful.                              |
| `rullama-network`         | [`@rullama/network`](../packages/network/)                         | Faithful.                              |
| `rullama-permission`      | [`@rullama/permission`](../packages/permission/)                   | Faithful (was `permissions`).          |
| `rullama-prompting`       | [`@rullama/prompting`](../packages/prompting/)                     | Faithful.                              |
| `rullama-provider`        | [`@rullama/provider`](../packages/provider/)                       | Partial — see §Providers (was `providers`). |
| `rullama-provider-speech` | [`@rullama/provider-speech`](../packages/provider-speech/)         | Faithful — HTTP audio clients only.    |
| `rullama-rag`             | [`@rullama/rag`](../packages/rag/)                                 | Partial — client-only.                 |
| `rullama-reasoning`       | [`@rullama/reasoning`](../packages/reasoning/)                     | Partial — Tier 1 only.                 |
| `rullama-seal`            | [`@rullama/seal`](../packages/seal/)                               | Faithful — SEAL learning loop.         |
| `rullama-session`         | [`@rullama/session`](../packages/session/)                         | Faithful (SQLite → Deno KV).           |
| `rullama-skills`          | [`@rullama/skills`](../packages/skills/)                           | Faithful — SKILL.md system.            |
| `rullama-storage`         | [`@rullama/storage`](../packages/storage/)                         | Faithful.                              |
| `rullama-stores`          | [`@rullama/stores`](../packages/stores/)                           | Faithful.                              |
| `rullama-telemetry`       | [`@rullama/telemetry`](../packages/telemetry/)                     | Partial — see §Telemetry.              |
| `rullama-tool-builtins`   | [`@rullama/tool-builtins`](../packages/tool-builtins/)             | Partial — see §Tools.                  |
| `rullama-tool-runtime`    | [`@rullama/tool-runtime`](../packages/tool-runtime/)               | Partial — see §Tools.                  |

Rust-only crates on the runtime boundary (no Deno package, intentional):
`rullama` (meta), `rullama-hardware`, `rullama-sandbox`,
`rullama-sandbox-proxy`, `rullama-datasets`, `rullama-test-fixtures`,
`rullama-test-harness`. See below.

## Runtime boundary — not ported on purpose

These are marked off-limits at the package layer. The Deno port does not try to
approximate any of them; drive the Rust binary from Deno instead and communicate
over `@rullama/network` or `@rullama/a2a`. This set is kept in sync with the
`RUST_ONLY` list in `scripts/parity.ts`.

- **`rullama`** — the meta-crate. No Deno equivalent; the JSR packages are
  independent.
- **`rullama-hardware`** — needs kernel access (ALSA/PulseAudio, libusb,
  bluez, GPIO sysfs, Zigbee, Z-Wave, Matter). Not portable.
- **`rullama-sandbox`** — Bollard Docker client driving container
  orchestration. Run the Rust sidecar.
- **`rullama-sandbox-proxy`** — Hyper-based HTTP egress proxy. Run the Rust
  sidecar.
- **`rullama-datasets`** — local training-data pipeline (GPU/disk). Not a Deno
  concern; callers construct JSONL themselves and upload via the finetune API.
- **`rullama-test-fixtures`** — internal test infrastructure, unpublished.
- **`rullama-test-harness`** — internal test infrastructure, unpublished.
- **`local_llm` provider** — llama-cpp FFI. Use `OllamaChatProvider` for local
  inference from Deno.
- **`interpreters` / `orchestrator` tools** — Rhai, Boa, RustPython embedded
  runtimes. Not worth emulating in the Deno layer.
- **`sandbox_executor` / `code_exec`** — depend on the Rust sandbox crate.
- **`browser` tool** — pairs with the Rust Thalora headless browser. Drive it
  over IPC from Deno when needed.
- **LanceDB / ONNX / tantivy RAG** — native indexing stays Rust-side. The Deno
  `knowledge` package keeps its client role and talks to a Rust RAG service over
  the existing `RagClient` interface.
- **Burn-based local training** — the Deno `finetune` package exposes cloud
  fine-tuning only.

## Providers — partial

Ported:

- `AnthropicChatProvider`, `OpenAiChatProvider`, `OpenAiResponsesProvider`,
  `BedrockProvider`, `VertexAiProvider`, `GoogleChatProvider`,
  `OllamaChatProvider`
  via response_id metadata
- Audio clients: `AzureSpeechClient`, `DeepgramClient`, `ElevenLabsClient`,
  `GoogleTtsClient`, `MurfClient`, `CartesiaClient`, `FishClient` — all
  fetch-based, return `Uint8Array`

Not ported:

- `local_llm` — llama-cpp FFI provider, runtime-boundary.

## Tools — partial

The old `rullama-tools` crate was split in v0.11.0 into
[`@rullama/tool-runtime`](../packages/tool-runtime/) (registry, executor,
error taxonomy, sanitization, smart routing, transactions, OpenAPI/OAuth,
validation, tool-search/-embedding) and
[`@rullama/tool-builtins`](../packages/tool-builtins/) (the concrete built-in
tools). Ported:

- runtime: `registry`, `executor`, `error`, `sanitization`, `smart_router`,
  `transaction`, `validation`, `openapi`, `oauth`, `tool_search`,
  `tool_embedding`
- builtins: `bash`, `file_ops`, `git`, `web`, `search`, `semantic_search`,
  `calendar/{types,google,caldav,mod}`, `sessions/{broker,sessions_tool}`

Skipped (runtime boundary) — documented in
[packages/tool-builtins/SKIPPED.md](../packages/tool-builtins/SKIPPED.md):

- `email/` (IMAP/SMTP/Gmail-push)
- `system/services/` (systemd, docker shell, process)
- `system/reactor/` (filesystem watchers with ripgrep-style matching)
- `browser` (pairs with Rust Thalora)
- `interpreters/`, `orchestrator/`, `sandbox_executor`, `code_exec`

## Telemetry — partial

Ported:

- `AnalyticsEvent` (10 variants), `UsageEvent` (5 variants), `BillingHook`,
  `AnalyticsCollector`, `MemoryAnalyticsSink`, `MetricsRegistry` + Prometheus
  exposition, PII helpers (`hashSessionId`, `redactSecrets`).

Intentionally omitted:

- **SQLite sink + SQL query layer** — implement `AnalyticsSink` against Deno KV
  / Postgres / OTLP instead.
- **`tracing` crate layer** — no Deno equivalent. Use `collector.onEvent(cb)` to
  pipe events to your logger or OTLP exporter.
- **Heavy PII detectors** — email / phone / SSN pattern libraries stay Rust-side
  until explicitly requested.

## Reasoning — Tier 1 only

Ported: `parsePlanSteps`, `stepsToTasks`, `ComplexityScorer`, `LocalRouter`,
`LocalValidator`, `RetrievalClassifier`, `LocalInferenceConfig`,
`InferenceTimer`, plus `OutputParser` re-exported from core.

Deferred (planned follow-up): `strategies` (CoT/ReAct/Reflexion/ToT),
`strategy_selector`, `summarizer`, `relevance_scorer`, `entity_enhancer`.

## Finetune — cloud only

The `@rullama/finetune` package (was `@rullama/training` pre-0.11.0) ports the
cloud fine-tuning slice.

Ported: shared types / hyperparams / LoRA / alignment config, `TrainingError`,
`FineTuneProvider` interface, `OpenAiFineTune`, `TogetherFineTune`,
`FireworksFineTune`, `JobPoller`, `TrainingManager`.

Not ported:

- **Bedrock / Vertex** — need vendor SDKs for request signing; implement
  `FineTuneProvider` directly if needed.
- **Anyscale, `cost.rs`** — niche; add as needed.
- **`datasets/` subtree** — callers construct JSONL themselves and upload via
  `uploadDataset`.
- **`local/` subtree** — Burn-based training stays Rust-side.

## Knowledge — client-interface only

`@rullama/knowledge` ships the full prompting-technique catalog, code
analysis helpers, and the client interfaces (`BrainClient`, `RagClient`, plus
request/response types). The concrete RAG / BKS / PKS implementations stay in
Rust because they depend on LanceDB + tantivy + ONNX which are not portable to
Deno; Deno consumers proxy to the Rust service over the existing interfaces.
