# brainwires-reasoning

Layer 3 — Intelligence. Provider-agnostic reasoning primitives for the
Brainwires Agent Framework: plan / output parsers, local-LLM scorers for
routing and validation, and named reasoning strategies (CoT, ReAct,
Reflexion, Tree-of-Thoughts). Every scorer accepts `Arc<dyn Provider>`
and falls back to pattern-based logic when the provider is unavailable,
so a missing local model degrades gracefully instead of crashing.

## Layout

```
brainwires-reasoning
├── output_parser / plan_parser    # LLM-output → structured data
├── LocalRouter                    # fast model-size selection
├── ComplexityScorer               # score task difficulty
├── RetrievalClassifier            # needs-retrieval gate
├── RelevanceScorer                # rerank retrieved chunks
├── StrategySelector               # pick CoT / ReAct / ...
├── EntityEnhancer                 # extract typed entities + relations
├── LocalSummarizer                # offline summarization / fact extract
├── Validator                      # output sanity checks
└── strategies::{ChainOfThought, ReAct, Reflexion, TreeOfThoughts}
```

## Features

This crate has no Cargo features — every scorer is always compiled.
Individual scorers are enabled at runtime via `LocalInferenceConfig`, so
callers pay the dependency cost once and pick which of the nine
components to wire up.

## Quick start

```rust
use std::sync::Arc;
use brainwires_reasoning::{
    LocalInferenceConfig, LocalRouterBuilder, JsonOutputParser, OutputParser,
    parse_plan_steps,
};

# async fn demo(provider: Arc<dyn brainwires_core::Provider>) -> anyhow::Result<()> {
// Parse a numbered plan.
let steps = parse_plan_steps("1. Research topic\n2. Draft summary\n3. Review");
assert_eq!(steps.len(), 3);

// Parse JSON from LLM output (strips ```json fences automatically).
let parser = JsonOutputParser::new();
let value = parser.parse("```json\n{\"result\": 42}\n```")?;

// Route a prompt to the right model tier.
let router = LocalRouterBuilder::new(provider.clone()).build();
let choice = router.route("summarise the last 50 chat turns").await?;
println!("model = {:?}, confidence = {}", choice.model, choice.confidence);
# Ok(()) }
```

## Configuration

Enable scorers via `LocalInferenceConfig`:

```toml
[local_llm]
enabled = true
use_for_routing = true
use_for_validation = true
use_for_complexity = true
use_for_summarization = true
```

Convenience presets live on the type:

```rust
use brainwires_reasoning::LocalInferenceConfig;

let tier1  = LocalInferenceConfig::tier1_enabled();    // routing + complexity
let tier2  = LocalInferenceConfig::tier2_enabled();    // + validation + summary
let all    = LocalInferenceConfig::all_enabled();
let routing_only       = LocalInferenceConfig::routing_only();
let validation_only    = LocalInferenceConfig::validation_only();
let summarization_only = LocalInferenceConfig::summarization_only();
```

## Components

- **`plan_parser`** — extract numbered steps (`1. foo\n2. bar`) from an
  LLM plan into `ParsedStep { index, text }`. Survives leading prose and
  inter-step blank lines; `steps_to_tasks` converts to core `Task` values.
- **`output_parser`** — `JsonOutputParser` (fence-aware),
  `JsonListParser`, `RegexOutputParser`. All implement the shared
  `OutputParser` trait.
- **`LocalRouter`** — pick a small/large model based on prompt length,
  entropy, and task-type cues.
- **`ComplexityScorer`** — multi-factor task-difficulty score used by
  `StrategySelector` to choose a reasoning strategy.
- **`RetrievalClassifier`** — gate retrieval behind a cheap pre-classifier
  so trivially-answerable questions skip the RAG roundtrip.
- **`RelevanceScorer`** — rerank retrieved chunks by query-aware
  relevance; pairs well with `brainwires-knowledge::SpectralReranker` for
  diversity-aware final selection.
- **`StrategySelector`** — input task → recommended strategy
  (`ChainOfThought`, `ReAct`, `Reflexion`, `TreeOfThoughts`) with a
  rationale.
- **`EntityEnhancer`** — pull typed entities and `RelationType`
  relationships out of free text.
- **`LocalSummarizer`** — offline summarisation and `ExtractedFact`
  extraction (typed by `FactCategory`).
- **`Validator`** — safety / format sanity checks on generated output.
- **`strategies`** — concrete implementations of the four reasoning
  strategies behind a shared `ReasoningStrategy` trait, selectable via
  `StrategyPreset`.

## Scope note

An earlier architectural plan (`sleepy-popping-falcon.md`) also proposed
moving the `prompting` subsystem here. It stayed in `brainwires-knowledge`
because `prompting` is tightly coupled to `bks_pks` inside that crate;
moving it would have pulled the entire knowledge-store dependency into
this layer-3 crate. The deviation is intentional — consumers of prompting
should continue to target `brainwires_knowledge::prompting`.

## Observability

Every scorer call emits a `tracing` span with `task`, `model`, `success`
and duration. `log_inference(task, model, latency_ms, success)` and the
RAII `InferenceTimer` (`finish(success)`) are the lower-level entry
points when you need to instrument custom code paths.

## Status

`#[deny(missing_docs)]`. Stable trait surface; scorers evolve additively.
