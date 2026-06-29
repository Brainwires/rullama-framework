# Parity matrix — Deno port vs. Rust crates

How this doc works: for every Rust crate under `crates/`, we list the Deno
package (if any) and the status. A **partial** crate links to the `SKIPPED.md`
under the corresponding package so it's clear what stays Rust-side and why.

Run `deno task parity` (see `scripts/parity.ts`) to regenerate the crate ↔
package diff from the actual filesystem.

## Summary

| Rust crate                              | Deno package                                              | Status                                 |
| --------------------------------------- | --------------------------------------------------------- | -------------------------------------- |
| `rullama` (meta)                     | —                                                         | n/a — JSR packages are independent.    |
| `rullama-core`                       | [`@rullama/core`](../packages/core/)                   | Faithful.                              |
| `rullama-providers`                  | [`@rullama/providers`](../packages/providers/)         | Partial — see §Providers.              |
| `rullama-agent`                      | [`@rullama/agents`](../packages/agents/)               | Faithful.                              |
| `rullama-mcp`                        | [`@rullama/mcp`](../packages/mcp/)                     | Faithful.                              |
| `rullama-mcp-server`                 | folded into [`@rullama/network`](../packages/network/) | Faithful.                              |
| `rullama-a2a`                        | [`@rullama/a2a`](../packages/a2a/)                     | Faithful (no gRPC, by design).         |
| `rullama-storage`                    | [`@rullama/storage`](../packages/storage/)             | Faithful.                              |
| `rullama-permissions`                | [`@rullama/permissions`](../packages/permissions/)     | Faithful.                              |
| `rullama-tools`                      | [`@rullama/tools`](../packages/tools/)                 | Partial — see §Tools.                  |
| `rullama-knowledge`                  | [`@rullama/knowledge`](../packages/knowledge/)         | Partial — RAG/BKS/PKS are client-only. |
| `rullama-network`                    | [`@rullama/network`](../packages/network/)             | Faithful.                              |
| `rullama-session`                    | [`@rullama/session`](../packages/session/)             | Faithful (SQLite → Deno KV).           |
| `rullama-resilience`                 | [`@rullama/resilience`](../packages/resilience/)       | Faithful.                              |
| `rullama-telemetry`                  | [`@rullama/telemetry`](../packages/telemetry/)         | Partial — see §Telemetry.              |
| `rullama-reasoning`                  | [`@rullama/reasoning`](../packages/reasoning/)         | Partial — Tier 1 only.                 |
| `rullama-training`                   | [`@rullama/training`](../packages/training/)           | Partial — cloud slice only.            |
| `rullama-hardware`                   | —                                                         | Runtime boundary.                      |
| `rullama-sandbox` · `-sandbox-proxy` | —                                                         | Runtime boundary.                      |

## Runtime boundary — not ported on purpose

These are marked off-limits at the package layer. The Deno port does not try to
approximate any of them; drive the Rust binary from Deno instead and communicate
over `@rullama/network` or `@rullama/a2a`.

- **`rullama-hardware`** — needs kernel access (ALSA/PulseAudio, libusb,
  bluez, GPIO sysfs, Zigbee, Z-Wave, Matter). Not portable.
- **`rullama-sandbox`** — Bollard Docker client driving container
  orchestration. Run the Rust sidecar.
- **`rullama-sandbox-proxy`** — Hyper-based HTTP egress proxy. Run the Rust
  sidecar.
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
- **Burn-based local training** — the Deno `training` package exposes cloud
  fine-tuning only.

## Providers — partial

Ported:

- `AnthropicChatProvider`, `OpenAiChatProvider`, `OpenAiResponsesProvider`,
  `BedrockProvider`, `VertexAiProvider`, `GoogleChatProvider`,
  `OllamaChatProvider`
- `BrainwiresRelayProvider` — Studio backend, SSE streaming, tool-call threading
  via response_id metadata
- Audio clients: `AzureSpeechClient`, `DeepgramClient`, `ElevenLabsClient`,
  `GoogleTtsClient`, `MurfClient`, `CartesiaClient`, `FishClient` — all
  fetch-based, return `Uint8Array`

Not ported:

- `local_llm` — llama-cpp FFI provider, runtime-boundary.

## Tools — partial

Ported (see [packages/tools/tools/](../packages/tools/tools/) for files):

- `bash`, `file_ops`, `git`, `web`, `search`, `validation`, `openapi`
- `oauth`, `calendar/{types,google,caldav,mod}`, `tool_search`,
  `tool_embedding`, `semantic_search`, `sessions/{broker,sessions_tool}`

Skipped (runtime boundary) — documented in
[packages/tools/tools/SKIPPED.md](../packages/tools/tools/SKIPPED.md):

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

## Training — cloud only

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
