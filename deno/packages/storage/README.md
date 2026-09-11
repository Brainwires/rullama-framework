# @rullama/storage

Backend-agnostic persistent storage for rullama. Provides the `StorageBackend`
and `VectorDatabase` interfaces, the typed schema/record/filter model they
share, an in-memory backend for tests, an LRU-cached embedding provider, and
seven concrete database adapters.

Domain stores (`MessageStore`, `ConversationStore`, `TaskStore`, `PlanStore`,
`TemplateStore`) live in `@rullama/stores`; the tiered memory hierarchy lives in
`@rullama/memory`. This package does not re-export them.

Equivalent to the Rust `rullama-storage` crate.

## Install

```sh
deno add @rullama/storage
```

## Quick Example

```ts
import {
  FieldTypes,
  fieldValueAsStr,
  FieldValues,
  Filters,
  InMemoryStorageBackend,
  recordGet,
  requiredField,
  type StorageBackend,
} from "@rullama/storage";

const backend: StorageBackend = new InMemoryStorageBackend();

// A table is a list of typed fields; a record is a list of (column, value).
await backend.ensureTable("notes", [
  requiredField("id", FieldTypes.Utf8),
  requiredField("content", FieldTypes.Utf8),
  requiredField("embedding", FieldTypes.Vector(3)),
]);

await backend.insert("notes", [
  [
    ["id", FieldValues.Utf8("1")],
    ["content", FieldValues.Utf8("Hello")],
    ["embedding", FieldValues.Vector([1, 0, 0])],
  ],
  [
    ["id", FieldValues.Utf8("2")],
    ["content", FieldValues.Utf8("World")],
    ["embedding", FieldValues.Vector([0, 1, 0])],
  ],
]);

// Structured filters translate to each backend's native syntax.
const rows = await backend.query(
  "notes",
  Filters.Eq("id", FieldValues.Utf8("1")),
);
console.log(fieldValueAsStr(recordGet(rows[0], "content")!)); // "Hello"

// Vector similarity search over a Vector column.
const nearest = await backend.vectorSearch(
  "notes",
  "embedding",
  [0.9, 0.1, 0],
  1,
);
console.log(
  nearest[0].score,
  fieldValueAsStr(recordGet(nearest[0].record, "id")!),
);
```

## Backends

| Class              | Implements                         | Talks to                                  |
| ------------------ | ---------------------------------- | ----------------------------------------- |
| `PostgresDatabase` | `StorageBackend`, `VectorDatabase` | PostgreSQL + pgvector via `npm:pg`        |
| `MySqlDatabase`    | `StorageBackend`                   | MySQL / MariaDB via `npm:mysql2`          |
| `SurrealDatabase`  | `StorageBackend`                   | SurrealDB via `npm:surrealdb` (WebSocket) |
| `QdrantDatabase`   | `VectorDatabase`                   | Qdrant REST API, plain `fetch`            |
| `PineconeDatabase` | `VectorDatabase`                   | Pinecone REST API, plain `fetch`          |
| `WeaviateDatabase` | `VectorDatabase`                   | Weaviate REST + GraphQL, plain `fetch`    |
| `MilvusDatabase`   | `VectorDatabase`                   | Milvus REST API v2, plain `fetch`         |

Postgres, MySQL and SurrealDB pull an npm driver (`pg`, `mysql2`, `surrealdb`)
when imported. Qdrant, Pinecone, Weaviate and Milvus have no driver dependency —
they only use `fetch`, so they need `--allow-net` and nothing else.
`MySqlDatabase` stores vectors as JSON and computes cosine similarity
client-side.

Each adapter's default endpoint is exposed as a static `defaultUrl()`
(`postgresql://localhost:5432/rullama`, `mysql://localhost:3306/rullama`,
`ws://localhost:8000`, `http://localhost:6333`, `http://localhost:8080`,
`http://localhost:19530`). Pinecone takes an index host and API key instead.

## Raw filters are off by default

`Filter` is a structured union (`Eq`, `Ne`, `Lt`, `Lte`, `Gt`, `Gte`, `In`,
`IsNull`, `NotNull`, `And`, `Or`), built with the `Filters` helpers. Field
values are always bound as parameters; table and column names are validated as
plain identifiers (`[A-Za-z_][A-Za-z0-9_]*`, at most 63 characters) rather than
quoted.

The escape hatch `Filters.Raw(expression)` splices `expression` into the query
verbatim. Because a filter deserialized from JSON, a request body or a model's
tool call must never carry raw SQL, the SQL and SurrealQL backends **reject
`Raw` filters unless you opt in** when constructing the backend:

```ts
import { PostgresDatabase } from "@rullama/storage";

// Without allowRawFilters, a Raw filter throws:
//   "Raw filters are disabled; pass { allowRaw: true } (or construct the
//    backend with allowRawFilters: true) to enable them"
const db = new PostgresDatabase({
  connectionString: "postgresql://user:pass@localhost:5432/app",
  allowRawFilters: true,
});
```

`allowRawFilters` is accepted by `PostgresConfig`, `MySqlConfig` and
`SurrealConfig`. The exported SQL builder helpers (`filterToSql`, `buildSelect`,
…) take the same switch as `{ allowRaw: true }` in their `options` argument.
`InMemoryStorageBackend` and the fetch-only vector stores never evaluate `Raw`
expressions.

## Key Exports

| Export                                                                               | Description                                                                         |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `StorageBackend`                                                                     | Interface: `ensureTable` / `insert` / `query` / `delete` / `count` / `vectorSearch` |
| `VectorDatabase`                                                                     | Interface for RAG embedding stores with hybrid (vector + keyword) search            |
| `InMemoryStorageBackend`                                                             | Map-backed `StorageBackend` for tests and lightweight use                           |
| `FieldType` / `FieldTypes`, `FieldDef`, `requiredField` / `optionalField`            | Table schema model                                                                  |
| `FieldValue` / `FieldValues`, `Record`, `ScoredRecord`, `recordGet`, `fieldValueAs*` | Typed row model and accessors                                                       |
| `Filter` / `Filters`                                                                 | Structured query filters (see above for `Raw`)                                      |
| `BackendCapabilities` / `defaultCapabilities`                                        | Feature flags a backend advertises (currently `vectorSearch`)                       |
| `EmbeddingProvider`, `CachedEmbeddingProvider`                                       | Core embedding interface plus an LRU-memoized wrapper                               |
| `PostgresDatabase` / `PostgresConfig`                                                | Postgres + pgvector adapter                                                         |
| `MySqlDatabase` / `MySqlConfig`                                                      | MySQL / MariaDB adapter                                                             |
| `SurrealDatabase` / `SurrealConfig`                                                  | SurrealDB adapter                                                                   |
| `QdrantDatabase`, `PineconeDatabase`, `WeaviateDatabase`, `MilvusDatabase`           | Fetch-only vector store adapters                                                    |
| `ChunkMetadata`, `SearchResult`, `DatabaseStats`                                     | `VectorDatabase` result types (re-exported from `@rullama/core`)                    |
