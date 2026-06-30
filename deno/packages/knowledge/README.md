# @rullama/knowledge

Unified intelligence layer for the rullama. Provides
prompting technique selection, knowledge graph interfaces, and RAG
(Retrieval-Augmented Generation) client types.

Equivalent to the Rust `rullama-knowledge` crate.

## Install

```sh
deno add @rullama/knowledge
```

## Quick Example

```ts
import {
  ALL_TECHNIQUES,
  getTechniqueMetadata,
  getTechniquesByCategory,
  getTechniquesByComplexity,
} from "@rullama/knowledge";

// List all 15 prompting techniques
for (const technique of ALL_TECHNIQUES) {
  const meta = getTechniqueMetadata(technique);
  console.log(`${technique}: ${meta.description} [${meta.complexity}]`);
}

// Filter techniques by complexity
const advanced = getTechniquesByComplexity("Advanced");
console.log("Advanced techniques:", advanced);

// Filter by category
const reasoning = getTechniquesByCategory("Reasoning");
console.log("Reasoning techniques:", reasoning);
```

## Modules

| Module        | Description                                                                                                                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Prompting** | 15 techniques from "Adaptive Selection of Prompting Techniques" (arXiv:2510.18162) with SEAL quality integration. Includes `ChainOfThought`, `PlanAndSolve`, `DecomposedPrompting`, and more. |
| **Knowledge** | `BrainClient` interface for persistent thought storage, entity/relationship types, and thought capture/search.                                                                                |
| **RAG**       | `RagClient` interface for semantic code search with `IndexRequest`, `QueryRequest`, `AdvancedSearchRequest`, and statistics.                                                                  |
