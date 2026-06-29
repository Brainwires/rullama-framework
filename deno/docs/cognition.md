# Cognition

The `@rullama/knowledge` package provides prompting techniques, a knowledge
graph interface, RAG (Retrieval-Augmented Generation) client, and code analysis
tools.

## Prompting Techniques

15 techniques from the adaptive prompting selection paper, each with metadata
including complexity level, task characteristics, and SEAL quality integration.

```ts
import {
  ALL_TECHNIQUES,
  bestTechnique,
  getTechniqueMetadata,
  getTechniquesByCategory,
  getTechniquesByComplexity,
  PromptGenerator,
} from "@rullama/knowledge";

// Browse all techniques
const metadata = getTechniqueMetadata("chain_of_thought");

// Select by category or complexity
const reasoning = getTechniquesByCategory("reasoning");
const simple = getTechniquesByComplexity("low");

// Auto-select the best technique for a task
const technique = bestTechnique(taskCharacteristics);

// Generate a prompt using a technique
const generator = new PromptGenerator();
const prompt = generator.generate(
  "chain_of_thought",
  "Solve this math problem...",
);
```

Supporting components:

- `PromptingLearningCoordinator` -- tracks technique effectiveness over time
- `TaskClusterManager` -- groups similar tasks for technique selection
- `TemperatureOptimizer` -- finds optimal temperature per technique

See: `../examples/cognition/prompting_techniques.ts`.

## Knowledge (BrainClient)

The `BrainClient` interface provides persistent thought storage,
entity/relationship management, and knowledge search.

```ts
import type { BrainClient, Entity, Thought } from "@rullama/knowledge";
import { createThought } from "@rullama/knowledge";

// Create a thought
const thought = createThought({
  content: "The auth module uses JWT tokens",
  category: "observation",
  source: "code_review",
});

// Use with a BrainClient implementation
// await client.captureThought({ thought });
// const results = await client.searchKnowledge({ query: "authentication" });
```

Types: `Thought`, `ThoughtCategory`, `Entity`, `Relationship`,
`KnowledgeResult`, `MemorySearchResult`.

See: `../examples/cognition/knowledge_graph.ts`.

## RAG (RagClient)

The `RagClient` interface defines semantic code search operations: index, query,
advanced search, and statistics.

```ts
import type {
  IndexRequest,
  QueryRequest,
  RagClient,
} from "@rullama/knowledge";

// Index a codebase
const indexReq: IndexRequest = { path: "/path/to/project", mode: "full" };
// await ragClient.index(indexReq);

// Semantic search
const queryReq: QueryRequest = {
  query: "authentication middleware",
  limit: 10,
};
// const results = await ragClient.query(queryReq);
```

Types: `IndexRequest`, `IndexResponse`, `QueryRequest`, `QueryResponse`,
`AdvancedSearchRequest`, `SearchResult`, `ChunkMetadata`.

See: `../examples/cognition/rag_search.ts`.

## Code Analysis

Regex-based symbol extraction and call graph generation for TypeScript,
JavaScript, Python, and Rust.

```ts
import {
  buildCallGraph,
  CallGraph,
  findReferences,
  RepoMap,
} from "@rullama/knowledge";

// Generate a repository map
const repoMap = new RepoMap();
// repoMap.addFile(filePath, content);
// const map = repoMap.format();

// Build a call graph
const graph = buildCallGraph(definitions, references);

// Find references to a symbol
const refs = findReferences(symbolId, allReferences);
```

Types: `SymbolId`, `SymbolKind`, `CallEdge`, `CallGraphNode`,
`CodeAnalysisDefinition`, `CodeAnalysisReference`, `Visibility`.

See: `../examples/cognition/code_analysis.ts`.

## Further Reading

- [Agents](./agents.md) for using cognition in agent loops
- [Extensibility](./extensibility.md) for implementing custom BrainClient or
  RagClient
