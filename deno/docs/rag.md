# RAG and Code Analysis

The `@rullama/rag` package is the client-side contract for retrieval-augmented
generation over a codebase. The indexing service itself (LanceDB + tantivy +
ONNX) stays in the Rust `rullama-rag` crate; Deno callers implement or proxy the
`RagClient` interface. Code analysis (symbol extraction, repo maps, call graphs)
is co-located here under the `@rullama/rag/code-analysis` sub-path and is also
re-exported from the package root.

## RagClient

```ts
import type { IndexRequest, QueryRequest, RagClient } from "@rullama/rag";
import { DEFAULT_LIMIT, DEFAULT_MIN_SCORE } from "@rullama/rag";

// Index a codebase
const indexReq: IndexRequest = {
  path: "/path/to/project",
  includePatterns: ["**/*.ts"],
};
// await ragClient.indexCodebase(indexReq);

// Hybrid semantic + keyword search
const queryReq: QueryRequest = {
  query: "authentication middleware",
  limit: DEFAULT_LIMIT,
  minScore: DEFAULT_MIN_SCORE,
  hybrid: true,
};
// const { results } = await ragClient.queryCodebase(queryReq);
```

Interface methods: `indexCodebase`, `queryCodebase`, `advancedSearch` (filter by
extensions / languages / path patterns), `searchGitHistory`, `getStatistics`,
`clearIndex`.

Types: `IndexRequest`, `IndexResponse`, `QueryRequest`, `QueryResponse`,
`SearchResult`, `AdvancedSearchRequest`, `SearchGitHistoryRequest` /
`SearchGitHistoryResponse`, `StatisticsResponse`, `ClearResponse`,
`ChunkMetadata`. Limits: `DEFAULT_LIMIT`, `DEFAULT_MIN_SCORE`,
`DEFAULT_MAX_FILE_SIZE`.

See: `../examples/knowledge/rag_search.ts` (a mock `RagClient`).

The `@rullama/tool-builtins` `SemanticSearchTool` exposes the same operations as
agent tools (`index_codebase`, `query_codebase`, `search_with_filters`, …) and
takes an injected `RagClient`.

## Code Analysis

Regex-based symbol extraction and call-graph construction for TypeScript,
JavaScript, Python and Rust. `RepoMap` is a namespace of static helpers.

```ts
import { buildCallGraph, findReferences, RepoMap } from "@rullama/rag";

const files = new Map([["src/app.ts", source]]);

// Extract definitions from one file
const definitions = RepoMap.extractSymbols({
  filePath: "src/app.ts",
  content: source,
});

// Aider-style repository map
const map = RepoMap.formatRepoMap(definitions);

// Call graph across files
const graph = buildCallGraph(definitions, files);

// References to known symbols in one file
const index = new Map(definitions.map((d) => [d.symbolId.name, [d]]));
const refs = findReferences("src/app.ts", source, index);
```

Types: `SymbolId`, `SymbolKind`, `Visibility`, `CodeAnalysisDefinition`,
`CodeAnalysisReference`, `ReferenceKind`, `CallEdge`, `CallGraphNode`,
`CallGraph`, `ExtractOptions`. Helpers: `createSymbolId`, `symbolIdToStorageId`,
`definitionToStorageId`, `symbolKindDisplayName`,
`RepoMap.supportedExtensions()`.

See: `../examples/knowledge/code_analysis.ts`.

## Further Reading

- [Prompting](./prompting.md) for the technique catalog and `BrainClient`
- [Storage](./storage.md) for the `VectorDatabase` interface backing an index
- [Extensibility](./extensibility.md) for implementing a custom `RagClient`
