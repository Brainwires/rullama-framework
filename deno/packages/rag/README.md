# @rullama/rag

RAG client interface + code-analysis surface for the rullama framework. No
`@rullama/*` dependencies.

Extracted from `@rullama/knowledge` in v0.11.0 to mirror Rust's `rullama-rag`
crate. Code analysis (symbol extraction, repo maps, call graphs, reference
tracking) lives alongside the RAG client because both share embedding pipelines
and storage in Rust; in Deno it is the `@rullama/rag/code-analysis` sub-path
export and is also re-exported from the package root.

The actual indexing service is Rust-side (`rullama-rag` over LanceDB + ONNX +
tantivy). This package ships the **`RagClient` interface and its
request/response types only** -- implement it against your own service or proxy
to the Rust one. `@rullama/tool-builtins`' `SemanticSearchTool` accepts any
`RagClient`.

## Install

```sh
deno add jsr:@rullama/rag
```

## Quick Example

```ts
import type { QueryRequest, RagClient } from "@rullama/rag";
import {
  buildCallGraph,
  DEFAULT_LIMIT,
  DEFAULT_MIN_SCORE,
  findReferences,
  RepoMap,
} from "@rullama/rag";

// --- Code analysis (pure, in-process) ---
const source = `
export function parse(input: string): Ast { return tokenize(input); }
export function tokenize(input: string): Ast { return { input }; }
`;
const files = new Map([["src/parser.ts", source]]);

const definitions = RepoMap.extractSymbols({
  filePath: "src/parser.ts",
  content: source,
});
console.log(RepoMap.formatRepoMap(definitions)); // aider-style repo map

const graph = buildCallGraph(definitions, files);
console.log(graph.nodes.size, graph.edges.length);

const index = new Map(definitions.map((d) => [d.symbolId.name, [d]]));
const refs = findReferences("src/parser.ts", source, index);
console.log(refs.map((r) => `${r.targetSymbolId}@${r.startLine}`));

// --- RAG client (interface; implementation is yours) ---
const req: QueryRequest = {
  query: "where is the tokenizer",
  limit: DEFAULT_LIMIT,
  minScore: DEFAULT_MIN_SCORE,
  hybrid: true,
};
async function search(client: RagClient) {
  const { results } = await client.queryCodebase(req);
  return results.map((r) => `${r.filePath}:${r.startLine} (${r.score})`);
}
console.log(typeof search);
```
