# @rullama/seal

Self-Evolving Agentic Learning (SEAL) for the rullama: coreference resolution,
query-core extraction, in-process pattern learning, and post-execution
reflection for conversational question answering over a relationship graph.

Equivalent to Rust's `rullama-seal` crate. The Deno port keeps the learning loop
in-process; the Rust crate ships a LanceDB-backed pattern store, which Deno
consumers can plug in via the `RagClient` interface in `@rullama/rag`.

## Install

```sh
deno add jsr:@rullama/seal
```

## Quick Example

```ts
import {
  DialogState,
  InMemoryEntityStore,
  queryCoreToSexp,
  SealProcessor,
} from "@rullama/seal";

// Entities the conversation already knows about.
const entities = new InMemoryEntityStore();
entities.add("main.rs", "file");
entities.add("parse_args", "function");

// Dialog state tracks what was mentioned on which turn.
const dialog = new DialogState();
dialog.nextTurn();
dialog.mentionEntity("main.rs", "file");

const seal = SealProcessor.withDefaults();
seal.initConversation("conv-1");

// "the file" is resolved to `[main.rs]`, then a query core is extracted.
const result = seal.process("what does the file use", dialog, entities);

console.log(result.resolved_query); // "what does [main.rs] use"
console.log(result.resolutions.map((r) => r.antecedent)); // ["main.rs"]
if (result.query_core !== undefined) {
  console.log(queryCoreToSexp(result.query_core)); // (JOIN DependsOn "main.rs" ?dependency)
}

// Feed execution outcomes back so successful shapes become reusable patterns.
seal.recordOutcome(result.matched_pattern, true, 3, result.query_core, 12);
console.log(seal.getLearningContext());
```

## Components

| Component                  | Description                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------ |
| `SealProcessor`            | Orchestrates the four stages below over a `DialogState` + `EntityStoreT` (+ optional graph)      |
| `CoreferenceResolver`      | Detects pronouns / definite NPs / demonstratives and ranks antecedents by salience               |
| `DialogState`              | Focus stack, mention history and turn counter that drive recency/frequency scores                |
| `QueryCoreExtractor`       | Classifies a question and builds an S-expression `QueryCore` from the entities it mentions       |
| `QueryExecutor`            | Runs a `QueryCore` against a `RelationshipGraphT` from `@rullama/core`                           |
| `LearningCoordinator`      | Per-session `LocalMemory` + cross-session `GlobalMemory` (query patterns, tool stats, hints)     |
| `ReflectionModule`         | Post-execution analysis: empty/overflow results, missing entities, custom relations, corrections |
| `FeedbackBridge`           | Pulls thumbs-up/down + corrections from a `@rullama/permission` `AuditLogger` into learning      |
| `SealKnowledgeCoordinator` | Wires SEAL to injected behavioral (BKS) / personal (PKS) knowledge caches                        |
| `InMemoryEntityStore`      | Minimal `EntityStoreT` implementation for tests and examples                                     |

## Pipeline

`SealProcessor.process(query, dialogState, entityStore, graph?)` runs, per
`SealConfig` flag:

1. **Coreference** — references with confidence ≥ `min_coreference_confidence`
   (default 0.5) are rewritten to `[antecedent]` markers.
2. **Query core** — the resolved query is classified (definition, location,
   dependency, count, superlative, enumeration, boolean) and turned into a
   `QueryCore`.
3. **Learning** — the best matching learned `QueryPattern` (if any) is reported
   as `matched_pattern`; call `recordOutcome` after execution to learn new
   patterns.
4. **Reflection** — the core is validated structurally and a `quality_score` is
   assigned.

`SealProcessor.reflect(core, result, graph)` produces a `ReflectionReport` for
an executed query; `ReflectionModule.attemptCorrection` can substitute a
similarly named entity when the original is missing from the graph.
