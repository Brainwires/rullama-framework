# @rullama/stores

Domain stores built on top of `@rullama/storage`'s `StorageBackend` interface.

In v0.11.0 these were extracted out of `@rullama/storage` to mirror the Rust
restructure (`rullama-stores`). The schemas live here; the backend interfaces
and adapters (Postgres / MySQL / SurrealDB / Qdrant / Pinecone / Weaviate /
Milvus / in-memory) remain in `@rullama/storage`.

## Stores

| Store                                             | Purpose                                                  |
| ------------------------------------------------- | -------------------------------------------------------- |
| `MessageStore` / `InMemoryMessageStore`           | Chat history with metadata + embedding search            |
| `ConversationStore` / `InMemoryConversationStore` | Multi-turn conversation aggregates                       |
| `TaskStore` / `InMemoryTaskStore`                 | Task graph + status                                      |
| `AgentStateStore` / `InMemoryAgentStateStore`     | Persisted agent state                                    |
| `PlanStore` / `InMemoryPlanStore`                 | Saved Plan-Work-Judge plan instances                     |
| `TemplateStore`                                   | Reusable plan templates with `{{variable}}` substitution |

Each backend-based store takes a `StorageBackend` (and, for `MessageStore`, an
`EmbeddingProvider` whose `dimension` sizes the vector column); call
`ensureTable()` before use. The `InMemory*` variants need no backend.

Tiered memory orchestration lives in `@rullama/memory`.

## Install

```sh
deno add jsr:@rullama/stores jsr:@rullama/storage
```

## Quick Example

```ts
import type { EmbeddingProvider } from "@rullama/core";
import { InMemoryStorageBackend } from "@rullama/storage";
import {
  createTemplate,
  InMemoryMessageStore,
  instantiateTemplate,
  MessageStore,
  TemplateStore,
} from "@rullama/stores";

const now = Math.floor(Date.now() / 1000);

// In-memory store: no backend, no embeddings
const mem = new InMemoryMessageStore();
await mem.ensureTable();
await mem.add({
  messageId: "msg-1",
  conversationId: "conv-1",
  role: "user",
  content: "How do I implement a binary search tree?",
  tokenCount: 9,
  createdAt: now,
});
console.log((await mem.getByConversation("conv-1")).length);

// Backend-based store with vector search
const embeddings: EmbeddingProvider = {
  dimension: 3,
  modelName: "toy",
  embed: (text) => Promise.resolve([text.length, 0, 0]),
  embedBatch: (texts) => Promise.resolve(texts.map((t) => [t.length, 0, 0])),
};
const store = new MessageStore(new InMemoryStorageBackend(), embeddings);
await store.ensureTable();
await store.add({
  messageId: "msg-2",
  conversationId: "conv-1",
  role: "assistant",
  content: "Use a node class with left/right children.",
  tokenCount: 10,
  createdAt: now + 1,
});
const hits = await store.search("binary tree", 5, 0.0);
console.log(hits.map(([m, score]) => `${m.messageId}: ${score.toFixed(2)}`));

// Plan templates
const templates = new TemplateStore();
const tpl = createTemplate(
  "feature",
  "Implement a feature end to end",
  "1. Read {{module}}\n2. Implement {{feature}}\n3. Add tests",
);
templates.save(tpl);
const plan = instantiateTemplate(tpl, { module: "src/api", feature: "search" });
console.log(templates.list().length, plan);
```
