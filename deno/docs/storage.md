# Storage

Persistence is split across three packages since v0.11.0:

- `@rullama/storage` -- the backend-agnostic substrate: the `StorageBackend` and
  `VectorDatabase` interfaces, the shared schema/record/filter types,
  `InMemoryStorageBackend`, `CachedEmbeddingProvider`, and the concrete
  adapters.
- `@rullama/stores` -- domain stores (messages, conversations, tasks, plans,
  agent state, plan templates) built on `StorageBackend`.
- `@rullama/memory` -- tiered memory (hot/warm/cold) and retention scoring.

## StorageBackend Interface

`StorageBackend` (from `packages/storage/traits.ts`) is a typed-table store with
structured filters and vector search. Records are keyed by the `id` field of the
table schema; there is no per-row `get`/`update` -- use `query` with an `Eq`
filter and `insert` (backends upsert on `id`).

```ts
interface StorageBackend {
  ensureTable(tableName: string, schema: FieldDef[]): Promise<void>;
  insert(tableName: string, records: Record[]): Promise<void>;
  query(tableName: string, filter?: Filter, limit?: number): Promise<Record[]>;
  delete(tableName: string, filter: Filter): Promise<void>;
  count(tableName: string, filter?: Filter): Promise<number>;
  vectorSearch(
    tableName: string,
    vectorColumn: string,
    vector: number[],
    limit: number,
    filter?: Filter,
  ): Promise<ScoredRecord[]>;
}
```

A `Record` is a list of `[name, FieldValue]` pairs (build values with the
`FieldValues` helpers, read them back with `recordGet`). `Filter` is a
discriminated union built with the `Filters` helpers: `Eq`, `Ne`, `Lt`, `Lte`,
`Gt`, `Gte`, `IsNull`, `NotNull`, `In`, `And`, `Or`, and `Raw` (a verbatim
expression in the backend's query language).

```ts
import {
  FieldTypes,
  FieldValues,
  Filters,
  InMemoryStorageBackend,
  requiredField,
} from "@rullama/storage";

const backend = new InMemoryStorageBackend();
await backend.ensureTable("locks", [
  requiredField("id", FieldTypes.Utf8),
  requiredField("holder", FieldTypes.Utf8),
]);
await backend.insert("locks", [[
  ["id", FieldValues.Utf8("src/app.ts")],
  ["holder", FieldValues.Utf8("agent-1")],
]]);
const rows = await backend.query(
  "locks",
  Filters.Eq("holder", FieldValues.Utf8("agent-1")),
);
```

## VectorDatabase Interface

`VectorDatabase` is the RAG embedding store. It is a separate interface, not an
extension of `StorageBackend`:

```ts
interface VectorDatabase {
  initialize(dimension: number): Promise<void>;
  storeEmbeddings(
    embeddings: number[][],
    metadata: ChunkMetadata[],
    contents: string[],
    rootPath: string,
  ): Promise<number>;
  search(
    queryVector: number[],
    queryText: string,
    limit: number,
    minScore: number,
    project?: string,
    rootPath?: string,
    hybrid?: boolean,
  ): Promise<SearchResult[]>;
  searchFiltered(
    /* ...search args..., */ fileExtensions?,
    languages?,
    pathPatterns?,
  ): Promise<SearchResult[]>;
  searchWithEmbeddings(/* ...search args... */): Promise<
    [SearchResult[], number[][]]
  >;
  deleteByFile(filePath: string): Promise<number>;
  clear(): Promise<void>;
  flush(): Promise<void>;
  getStatistics(): Promise<DatabaseStats>;
  countByRootPath(rootPath: string): Promise<number>;
  getIndexedFiles(rootPath: string): Promise<string[]>;
}
```

`ChunkMetadata`, `SearchResult` and `DatabaseStats` come from `@rullama/core`.

## Backends

| Class                    | Backend    | Implements                         | Transport                 |
| ------------------------ | ---------- | ---------------------------------- | ------------------------- |
| `InMemoryStorageBackend` | In-memory  | `StorageBackend`                   | none (tests, prototyping) |
| `PostgresDatabase`       | PostgreSQL | `StorageBackend`, `VectorDatabase` | `npm:pg` (pgvector)       |
| `MySqlDatabase`          | MySQL      | `StorageBackend`, `VectorDatabase` | `npm:mysql2`              |
| `SurrealDatabase`        | SurrealDB  | `StorageBackend`, `VectorDatabase` | `npm:surrealdb`           |
| `QdrantDatabase`         | Qdrant     | `VectorDatabase`                   | `fetch()`                 |
| `PineconeDatabase`       | Pinecone   | `VectorDatabase`                   | `fetch()`                 |
| `WeaviateDatabase`       | Weaviate   | `VectorDatabase`                   | `fetch()`                 |
| `MilvusDatabase`         | Milvus     | `VectorDatabase`                   | `fetch()`                 |

### Raw filters and identifier validation

Since v0.12.0 the SQL/SurrealQL backends treat model- or user-supplied strings
defensively:

- **`Raw` filters are disabled by default.** A `Filter` of kind `"Raw"` is
  refused with
  `Raw filters are disabled; pass { allowRaw: true } (or construct
  the backend with allowRawFilters: true) to enable them`.
  Opt in per backend with the `allowRawFilters` config flag (`PostgresConfig`,
  `MySqlConfig`, `SurrealConfig`); the query builders then receive
  `{ allowRaw: true }`.
- **Identifiers are validated.** Table and column names must match
  `^[A-Za-z_][A-Za-z0-9_]{0,62}$`; anything else throws before a query is built,
  so schema names cannot smuggle SQL.

```ts
import { PostgresDatabase } from "@rullama/storage";

// Raw filters stay off unless you ask for them.
const db = new PostgresDatabase({
  connectionString: Deno.env.get("DATABASE_URL"),
  allowRawFilters: true,
});
```

## Domain Stores (`@rullama/stores`)

Each store has an interface (`*StoreI`), a backend-based implementation and an
in-memory variant for tests:

| Store               | Interface            | In-memory                   | Purpose                                         |
| ------------------- | -------------------- | --------------------------- | ----------------------------------------------- |
| `MessageStore`      | `MessageStoreI`      | `InMemoryMessageStore`      | Chat messages with metadata + vector search     |
| `ConversationStore` | `ConversationStoreI` | `InMemoryConversationStore` | Conversation session management                 |
| `TaskStore`         | `TaskStoreI`         | `InMemoryTaskStore`         | Task tracking                                   |
| `AgentStateStore`   | `AgentStateStoreI`   | `InMemoryAgentStateStore`   | Persisted agent state                           |
| `PlanStore`         | `PlanStoreI`         | `InMemoryPlanStore`         | Plan persistence                                |
| `TemplateStore`     | --                   | (in-memory only)            | Plan templates with `{{variable}}` substitution |

`MessageStore` takes the backend **and** an `EmbeddingProvider` (its `dimension`
sizes the vector column); call `ensureTable()` before use.

```ts
import { InMemoryMessageStore, MessageStore } from "@rullama/stores";
import { InMemoryStorageBackend } from "@rullama/storage";

const store = new MessageStore(new InMemoryStorageBackend(), embeddings);
await store.ensureTable();
await store.add({
  messageId: "msg-1",
  conversationId: "conv-1",
  role: "user",
  content: "Hello",
  tokenCount: 1,
  createdAt: Math.floor(Date.now() / 1000),
});

// No backend needed for tests:
const mem = new InMemoryMessageStore();
```

See: `../examples/storage/message_store.ts`,
`../examples/storage/plan_templates.ts`.

## Tiered Memory (`@rullama/memory`)

`TieredMemory` organizes messages into hot/warm/cold tiers with promotion and
demotion based on importance, recency and access counts; persistence is left to
the caller.

```ts
import { defaultTieredMemoryConfig, TieredMemory } from "@rullama/memory";

const memory = new TieredMemory(defaultTieredMemoryConfig());
```

Key functions: `promoteTier`, `demoteTier`, `retentionScore`,
`computeMultiFactorScore`, `recencyFromHours`, `recordAccess`,
`createTierMetadata`.

See: `../examples/storage/tiered_memory.ts`.

## Embeddings

`CachedEmbeddingProvider` wraps any `EmbeddingProvider` with an LRU memo; the
second argument is the maximum number of cached texts (default 1000).

```ts
import { CachedEmbeddingProvider } from "@rullama/storage";

const cached = new CachedEmbeddingProvider(baseProvider, 1000);
```

## Further Reading

- [Architecture](./architecture.md) for where storage fits in the dependency
  graph
- [Extensibility](./extensibility.md) for implementing custom storage backends
- Example: `../examples/storage/lock_coordination.ts`
