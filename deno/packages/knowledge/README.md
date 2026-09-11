# @rullama/knowledge

Type contract for the rullama "Open Brain" knowledge layer: the `Thought` model,
the entity/relationship knowledge-graph types, the request/response shapes for
thought capture, memory search and PKS/BKS knowledge search, and the
`BrainClient` interface that ties them together.

**This package ships types plus a few tiny helpers only.** There is no
`BrainClient` implementation here — a concrete client needs a storage backend
and an embedding provider, which you supply. Prompting techniques
(`ALL_TECHNIQUES`, `getTechniqueMetadata`, …) live in `@rullama/prompting`; RAG
and code analysis live in `@rullama/rag`.

Equivalent to the Rust `rullama-knowledge` crate.

## Install

```sh
deno add @rullama/knowledge
```

## Quick Example

Implement `BrainClient` against whatever store you have. The sketch below keeps
thoughts in a `Map` and does substring search; a real implementation would embed
content and query a vector store.

```ts
import {
  type BrainClient,
  type CaptureThoughtRequest,
  createThought,
  parseThoughtCategory,
  parseThoughtSource,
  type Thought,
} from "@rullama/knowledge";

class InMemoryBrain implements BrainClient {
  #thoughts = new Map<string, Thought>();

  captureThought(req: CaptureThoughtRequest) {
    const thought = createThought(req.content);
    thought.category = parseThoughtCategory(req.category ?? "general");
    thought.source = parseThoughtSource(req.source ?? "manual");
    thought.tags = req.tags ?? [];
    thought.importance = req.importance ?? 0.5;
    this.#thoughts.set(thought.id, thought);
    return Promise.resolve({
      id: thought.id,
      category: thought.category,
      tags: thought.tags,
      importance: thought.importance,
      factsExtracted: 0,
    });
  }

  searchMemory(req: Parameters<BrainClient["searchMemory"]>[0]) {
    const results = [...this.#thoughts.values()]
      .filter((t) => !t.deleted && t.content.includes(req.query))
      .slice(0, req.limit ?? 10)
      .map((t) => ({
        content: t.content,
        score: 1,
        source: "thoughts",
        thoughtId: t.id,
        category: t.category,
        tags: t.tags,
        createdAt: t.createdAt,
      }));
    return Promise.resolve({ results, total: results.length });
  }

  listRecent(req: Parameters<BrainClient["listRecent"]>[0]) {
    const thoughts = [...this.#thoughts.values()]
      .filter((t) => !t.deleted)
      .sort((a, b) => b.createdAt - a.createdAt)
      .slice(0, req.limit ?? 20);
    return Promise.resolve({ thoughts, total: thoughts.length });
  }

  getThought(id: string) {
    return Promise.resolve(this.#thoughts.get(id) ?? null);
  }

  searchKnowledge() {
    return Promise.resolve({ results: [], total: 0 });
  }

  memoryStats() {
    const total = this.#thoughts.size;
    return Promise.resolve({
      thoughts: {
        total,
        byCategory: {},
        recent24h: total,
        recent7d: total,
        recent30d: total,
        topTags: [],
      },
      pks: { totalFacts: 0, byCategory: {}, avgConfidence: 0 },
      bks: { totalTruths: 0, byCategory: {} },
    });
  }

  deleteThought(id: string) {
    const t = this.#thoughts.get(id);
    if (t) t.deleted = true;
    return Promise.resolve({ deleted: t !== undefined, id });
  }
}

const brain: BrainClient = new InMemoryBrain();
const { id } = await brain.captureThought({
  content: "Ship the storage refactor before the 0.13 cut.",
  category: "action_item",
  tags: ["release"],
});
const hits = await brain.searchMemory({ query: "storage", limit: 5 });
console.log(id, hits.results.map((r) => r.content));
```

## What's exported

| Export                                                                                               | Description                                                                               |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `Thought`, `ThoughtCategory`, `ThoughtSource`                                                        | The persistent thought record and its category / capture-source enums                     |
| `createThought`, `parseThoughtCategory`, `parseThoughtSource`, `ALL_THOUGHT_CATEGORIES`              | Helpers: new thought with defaults, lenient string parsers, the category list             |
| `Entity`, `EntityType`, `Relationship`, `ExtractionResult`                                           | Knowledge-graph types produced by entity extraction                                       |
| `ContradictionEvent`, `ContradictionKind`                                                            | Flags raised when a new fact conflicts with a stored one                                  |
| `Capture*` / `SearchMemory*` / `ListRecent*` / `GetThought*` / `DeleteThought*` / `SearchKnowledge*` | Request and response shapes for each `BrainClient` operation                              |
| `MemoryStatsResponse`, `ThoughtStats`, `PksStats`, `BksStats`                                        | Aggregate statistics over the thought store and the PKS / BKS knowledge stores            |
| `BrainClient`                                                                                        | The interface a concrete Open Brain client implements (no implementation in this package) |
