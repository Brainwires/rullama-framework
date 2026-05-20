# @brainwires/reasoning

Provider-agnostic reasoning primitives — parsers plus the Tier-1 local-inference
scorers (routing, validation, complexity, retrieval gating).

## What ships today

| Module | Purpose |
|---|---|
| `OutputParser` + friends | Re-exported from `@brainwires/core` so consumers import one symbol from reasoning. |
| `parsePlanSteps` / `stepsToTasks` | Turn LLM plan output into `Task` objects. |
| `ComplexityScorer` | 0.0–1.0 task complexity score. LLM-backed with a keyword + length heuristic fallback. |
| `LocalRouter` | Semantic query → `ToolCategory` classification. |
| `LocalValidator` | Semantic validation of agent responses. Also ships the pure heuristic. |
| `RetrievalClassifier` | Decides whether a query needs earlier conversation context. |
| `LocalInferenceConfig` | Feature flags + per-task model selection. |
| `InferenceTimer` | Lightweight latency measurement. |

Every scorer takes a `Provider` (from `@brainwires/core`) in its constructor.
The LLM-backed methods (`score`, `classify`, `validate`) return `null` on
failure so callers can fall through to the heuristic variant without a
try/catch.

## Not yet ported

Covered in a follow-up — scope is tracked in the alignment plan:

- `strategies` (CoT, ReAct, Reflexion, Tree-of-Thoughts)
- `strategy_selector`
- `summarizer` (+ fact extraction)
- `relevance_scorer`
- `entity_enhancer`

The Deno slice here covers the Tier-1 fast path plus the parsers the rest of
the framework consumes. The remaining modules are pure-logic and tractable,
just larger — they'll land as a second commit rather than stuffing everything
into the first ship.

## Equivalent Rust crate

`brainwires-reasoning` — same scorer shape, same provider-first design. The
`lfm2-350m` / `lfm2-1.2b` default model ids are preserved but are advisory;
the Deno workspace doesn't ship a local-model runner.
