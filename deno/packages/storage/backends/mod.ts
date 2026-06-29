/**
 * Database backend implementations for @rullama/storage.
 *
 * Re-exports all available backends:
 * - PostgresDatabase (pg + pgvector) -- StorageBackend + VectorDatabase
 * - QdrantDatabase (REST API) -- VectorDatabase
 * - SurrealDatabase (SurrealDB SDK) -- StorageBackend
 * - PineconeDatabase (REST API) -- VectorDatabase
 * - WeaviateDatabase (REST + GraphQL) -- VectorDatabase
 * - MilvusDatabase (REST API v2) -- VectorDatabase
 * - MySqlDatabase (mysql2) -- StorageBackend
 * @module
 */

export {
  buildCount,
  buildCreateTable,
  buildDelete,
  buildInsert,
  buildSelect,
  fieldValueToParam,
  // SQL helpers exported for testing
  filterToSql,
  type PostgresConfig,
  PostgresDatabase,
} from "./postgres.ts";

export {
  // Helpers exported for testing
  buildQdrantFilter,
  buildSearchBody,
  buildUpsertBody,
  parseSearchPoint,
  QdrantDatabase,
} from "./qdrant.ts";

export {
  // Helpers exported for testing
  fieldTypeToSurrealQL,
  fieldValueToJson,
  filterToSurrealQL,
  jsonRowToRecord,
  type SurrealConfig,
  SurrealDatabase,
} from "./surrealdb.ts";

export {
  // Helpers exported for testing
  buildMetadataFilter as buildPineconeFilter,
  buildQueryBody as buildPineconeQueryBody,
  buildUpsertBody as buildPineconeUpsertBody,
  extractFilePathsFromIds,
  parseMatch as parsePineconeMatch,
  PineconeDatabase,
} from "./pinecone.ts";

export {
  buildAggregateQuery as buildWeaviateAggregateQuery,
  buildBatchObject as buildWeaviateBatchObject,
  buildSearchQuery as buildWeaviateSearchQuery,
  // Helpers exported for testing
  buildWhereFilter as buildWeaviateWhereFilter,
  deterministicUuid,
  parseWeaviateResult,
  WeaviateDatabase,
} from "./weaviate.ts";

export {
  buildFilterExpr as buildMilvusFilterExpr,
  buildInsertBody as buildMilvusInsertBody,
  buildSearchBody as buildMilvusSearchBody,
  // Helpers exported for testing
  escapeFilterValue as escapeMilvusFilterValue,
  MilvusDatabase,
  parseMilvusResult,
} from "./milvus.ts";

export {
  buildCount as mysqlBuildCount,
  buildCreateTable as mysqlBuildCreateTable,
  buildDelete as mysqlBuildDelete,
  buildInsert as mysqlBuildInsert,
  buildSelect as mysqlBuildSelect,
  cosineSimilarity,
  fieldValueToParam as mysqlFieldValueToParam,
  filterToSql as mysqlFilterToSql,
  // SQL helpers exported for testing
  mapFieldType as mysqlMapFieldType,
  type MySqlConfig,
  MySqlDatabase,
} from "./mysql.ts";
