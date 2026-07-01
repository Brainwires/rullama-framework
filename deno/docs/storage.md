# Storage

The `@rullama/storage` package provides backend-agnostic persistent storage
with domain-specific stores and tiered memory.

## StorageBackend Interface

All backends implement this interface for table creation, record CRUD, and
querying:

```ts
interface StorageBackend {
  createTable(name: string, fields: FieldDef[]): Promise<void>;
  insert(table: string, record: Record): Promise<void>;
  get(table: string, id: string): Promise<Record | null>;
  update(table: string, id: string, record: Record): Promise<void>;
  delete(table: string, id: string): Promise<void>;
  query(table: string, filter: Filter): Promise<Record[]>;
  list(table: string, limit?: number, offset?: number): Promise<Record[]>;
}
```

The `VectorDatabase` interface extends this with vector search capabilities
(`upsert`, `search`, `count`).

## Backends

| Class                    | Backend    | Notes                               |
| ------------------------ | ---------- | ----------------------------------- |
| `InMemoryStorageBackend` | In-memory  | Zero config, great for testing      |
| `PostgresDatabase`       | PostgreSQL | Production-ready relational backend |
| `MySqlDatabase`          | MySQL      | MySQL/MariaDB backend               |
| `QdrantDatabase`         | Qdrant     | Vector search database              |
| `PineconeDatabase`       | Pinecone   | Managed vector search               |
| `WeaviateDatabase`       | Weaviate   | Vector + keyword search             |
| `MilvusDatabase`         | Milvus     | High-performance vector DB          |
| `SurrealDatabase`        | SurrealDB  | Multi-model database                |

```ts
import { InMemoryStorageBackend } from "@rullama/storage";

const backend = new InMemoryStorageBackend();
```

## Domain Stores

Higher-level stores wrap a `StorageBackend` with domain-specific logic:

| Store               | Interface            | Purpose                                   |
| ------------------- | -------------------- | ----------------------------------------- |
| `MessageStore`      | `MessageStoreI`      | Chat message persistence with metadata    |
| `ConversationStore` | `ConversationStoreI` | Conversation session management           |
| `TaskStore`         | `TaskStoreI`         | Task tracking and agent state             |
| `PlanStore`         | `PlanStoreI`         | Plan persistence and versioning           |
| `TemplateStore`     | --                   | Plan templates with variable substitution |

```ts
import { InMemoryStorageBackend, MessageStore } from "@rullama/storage";

const backend = new InMemoryStorageBackend();
const messageStore = new MessageStore(backend);
await messageStore.save({
  conversationId: "conv-1",
  role: "user",
  content: "Hello",
});
```

In-memory variants (`InMemoryMessageStore`, `InMemoryConversationStore`, etc.)
are also available for testing.

See: `../examples/storage/message_store.ts`,
`../examples/storage/plan_templates.ts`.

## Tiered Memory

`TieredMemory` organizes data into hot/warm/cold tiers with automatic promotion
and demotion based on access patterns.

```ts
import { defaultTieredMemoryConfig, TieredMemory } from "@rullama/storage";

const memory = new TieredMemory(defaultTieredMemoryConfig());
```

Key functions: `promoteTier`, `demoteTier`, `retentionScore`,
`computeMultiFactorScore`, `recencyFromHours`.

See: `../examples/storage/tiered_memory.ts`.

## Embeddings

`CachedEmbeddingProvider` wraps an `EmbeddingProvider` with LRU caching to avoid
redundant embedding calls.

```ts
import { CachedEmbeddingProvider } from "@rullama/storage";

const cached = new CachedEmbeddingProvider(baseProvider, { maxSize: 1000 });
```

## Further Reading

- [Architecture](./architecture.md) for where storage fits in the dependency
  graph
- [Extensibility](./extensibility.md) for implementing custom storage backends
- Example: `../examples/storage/lock_coordination.ts`
