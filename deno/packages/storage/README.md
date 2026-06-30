# @rullama/storage

Backend-agnostic persistent storage for the rullama. Provides
storage interfaces, an in-memory backend, domain-specific stores, and a tiered
memory hierarchy.

Equivalent to the Rust `rullama-storage` crate.

## Install

```sh
deno add @rullama/storage
```

## Quick Example

```ts
import {
  FieldTypes,
  InMemoryMessageStore,
  InMemoryStorageBackend,
  MessageStore,
  requiredField,
} from "@rullama/storage";

// Use the in-memory backend directly
const backend = new InMemoryStorageBackend();
await backend.ensureTable("notes", [
  requiredField("id", FieldTypes.Text),
  requiredField("content", FieldTypes.Text),
]);
await backend.insert("notes", [
  { fields: new Map([["id", { Text: "1" }], ["content", { Text: "Hello" }]]) },
]);
const results = await backend.query("notes");
console.log(results);

// Or use a domain store for messages
const messageStore = new InMemoryMessageStore();
await messageStore.store("conv-1", {
  id: "msg-1",
  conversationId: "conv-1",
  role: "user",
  content: "Hi there",
  timestamp: Date.now(),
});
const messages = await messageStore.listByConversation("conv-1");
console.log(messages);
```

## Key Exports

| Export                                            | Description                                           |
| ------------------------------------------------- | ----------------------------------------------------- |
| `StorageBackend`                                  | Interface for table-based key-value storage           |
| `VectorDatabase`                                  | Interface for vector similarity search                |
| `InMemoryStorageBackend`                          | In-memory implementation for testing                  |
| `MessageStore` / `InMemoryMessageStore`           | Message persistence                                   |
| `ConversationStore` / `InMemoryConversationStore` | Conversation metadata                                 |
| `TaskStore` / `InMemoryTaskStore`                 | Task persistence                                      |
| `PlanStore` / `InMemoryPlanStore`                 | Plan persistence                                      |
| `TemplateStore`                                   | Plan template storage and instantiation               |
| `TieredMemory`                                    | Hot/warm/cold memory hierarchy with retention scoring |
| `CachedEmbeddingProvider`                         | Embedding provider with caching layer                 |
