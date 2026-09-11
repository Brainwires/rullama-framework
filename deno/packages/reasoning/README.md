# @rullama/reasoning

Provider-agnostic reasoning primitives — parsers plus the Tier-1 local-inference
scorers (routing, validation, complexity, retrieval gating).

## Install

```sh
deno add @rullama/reasoning
```

## What ships today

| Module                            | Purpose                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| `OutputParser` + friends          | Re-exported from `@rullama/core` so consumers import one symbol from reasoning.       |
| `parsePlanSteps` / `stepsToTasks` | Turn LLM plan output into `Task` objects.                                             |
| `ComplexityScorer`                | 0.0–1.0 task complexity score. LLM-backed with a keyword + length heuristic fallback. |
| `LocalRouter`                     | Semantic query → `ToolCategory` classification.                                       |
| `LocalValidator`                  | Semantic validation of agent responses. Also ships the pure heuristic.                |
| `RetrievalClassifier`             | Decides whether a query needs earlier conversation context.                           |
| `LocalInferenceConfig`            | Feature flags + per-task model selection.                                             |
| `InferenceTimer`                  | Lightweight latency measurement.                                                      |

Every scorer takes a `Provider` (from `@rullama/core`; `@rullama/provider`
supplies concrete ones) in its constructor. The LLM-backed methods return a "no
answer" value instead of throwing so callers can fall through to the heuristic
variant without a try/catch: `ComplexityScorer.score` and `LocalRouter.classify`
/ `RetrievalClassifier.classify` return `null`, while `LocalValidator.validate`
returns `{ kind: "skipped" }`.

## Quick Example

```ts
import type { Provider } from "@rullama/core";
import {
  ComplexityScorer,
  LocalRouter,
  LocalValidator,
  parsePlanSteps,
  RetrievalClassifier,
  shouldRetrieve,
  stepsToTasks,
  tier1Enabled,
} from "@rullama/reasoning";

export async function triage(provider: Provider, query: string) {
  const config = tier1Enabled(); // routing + validation + complexity on
  const modelId = config.routing_model ?? "lfm2-350m";

  // Which tool categories does the query need? (LLM, then heuristic)
  const router = new LocalRouter(provider, modelId);
  const route = await router.classify(query);
  const categories = route?.categories ?? ["FileOps"];

  // How hard is it? `score` returns null on failure; fall back to the heuristic.
  const scorer = new ComplexityScorer(provider, modelId);
  const complexity = (await scorer.score(query)) ??
    scorer.scoreHeuristic(query);

  // Does it depend on earlier conversation context?
  const classifier = new RetrievalClassifier(provider, modelId);
  const need = classifier.classifyHeuristic(query, /* context_len */ 2);

  // Validate an agent reply; `validate` returns { kind: "skipped" } on failure.
  const validator = new LocalValidator(provider, modelId);
  const verdict = await validator.validate(query, "Sure — here is the plan…");

  // Turn a numbered plan into Task objects with sequential dependencies.
  const steps = parsePlanSteps(
    "1. Read the config\n2. Update the schema\n3. Run tests",
  );
  const tasks = stepsToTasks(steps, "plan-42");

  return {
    categories,
    complexity: complexity.score,
    retrieve: shouldRetrieve(need.need),
    verdict: verdict.kind,
    taskIds: tasks.map((t) => t.id),
  };
}
```

## Not yet ported

Covered in a follow-up — scope is tracked in the alignment plan:

- `strategies` (CoT, ReAct, Reflexion, Tree-of-Thoughts)
- `strategy_selector`
- `summarizer` (+ fact extraction)
- `relevance_scorer`
- `entity_enhancer`

The Deno slice here covers the Tier-1 fast path plus the parsers the rest of the
framework consumes. `LocalInferenceConfig` already carries the Tier-2 flags
(`summarization_enabled`, `relevance_scoring_enabled`, …) for parity with the
Rust crate, but only `retrieval_gating_enabled` has a consumer in this package.
The remaining modules are pure-logic and tractable, just larger — they'll land
as a second commit rather than stuffing everything into the first ship.

## Equivalent Rust crate

`rullama-reasoning` — same scorer shape, same provider-first design. The
`lfm2-350m` / `lfm2-1.2b` default model ids are preserved but are advisory; the
Deno workspace doesn't ship a local-model runner.
