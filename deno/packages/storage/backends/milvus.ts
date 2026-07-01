/**
 * Milvus vector database backend implementing VectorDatabase.
 *
 * Port of the Rust `rullama-storage/src/databases/milvus/mod.rs`.
 *
 * Uses `fetch()` to Milvus REST API v2 -- zero npm dependencies.
 * @module
 */

import type {
  ChunkMetadata,
  DatabaseStats,
  SearchResult,
} from "@rullama/core";
import type { VectorDatabase } from "../traits.ts";

const DEFAULT_URL = "http://localhost:19530";
const DEFAULT_COLLECTION = "code_embeddings";
const INSERT_BATCH_SIZE = 1000;
const QUERY_LIMIT = 16384;

// ---------------------------------------------------------------------------
// Milvus REST helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Escape a string value for use in a Milvus filter expression. */
export function escapeFilterValue(value: string): string {
  return value.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

/** Build a Milvus filter expression from optional search parameters. */
export function buildFilterExpr(
  project?: string,
  rootPath?: string,
  fileExtensions?: string[],
  languages?: string[],
): string {
  const clauses: string[] = [];

  if (project) {
    clauses.push(`project == "${escapeFilterValue(project)}"`);
  }
  if (rootPath) {
    clauses.push(`root_path == "${escapeFilterValue(rootPath)}"`);
  }
  if (fileExtensions && fileExtensions.length > 0) {
    const items = fileExtensions
      .map((e) => `"${escapeFilterValue(e)}"`)
      .join(", ");
    clauses.push(`extension in [${items}]`);
  }
  if (languages && languages.length > 0) {
    const items = languages
      .map((l) => `"${escapeFilterValue(l)}"`)
      .join(", ");
    clauses.push(`language in [${items}]`);
  }

  return clauses.join(" and ");
}

/** Build a Milvus search request body. */
export function buildSearchBody(
  collectionName: string,
  queryVector: number[],
  limit: number,
  filterExpr: string,
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    collectionName,
    data: [queryVector],
    annsField: "embedding",
    limit,
    outputFields: [
      "file_path",
      "root_path",
      "project",
      "start_line",
      "end_line",
      "language",
      "extension",
      "indexed_at",
      "content",
    ],
  };
  if (filterExpr) {
    body.filter = filterExpr;
  }
  return body;
}

/** Build an insert request body for a batch of rows. */
export function buildInsertBody(
  collectionName: string,
  embeddings: number[][],
  metadata: ChunkMetadata[],
  contents: string[],
  rootPath: string,
): Record<string, unknown> {
  const data = embeddings.map((emb, idx) => {
    const meta = metadata[idx];
    return {
      embedding: emb,
      file_path: meta.file_path,
      root_path: meta.root_path ?? rootPath,
      project: meta.project ?? "",
      start_line: meta.start_line,
      end_line: meta.end_line,
      language: meta.language ?? "Unknown",
      extension: meta.extension ?? "",
      file_hash: meta.file_hash,
      indexed_at: meta.indexed_at,
      content: contents[idx],
    };
  });
  return { collectionName, data };
}

/** Parse a Milvus search result item into a SearchResult. */
export function parseMilvusResult(
  item: Record<string, unknown>,
  minScore: number,
): SearchResult | null {
  // Milvus COSINE metric returns `distance` in [0, 2]; 0 = identical.
  const distance = (item.distance as number) ?? 1.0;
  const vectorScore = 1.0 - distance;

  if (vectorScore < minScore) return null;

  const content = item.content as string | undefined;
  if (!content) return null;

  const filePath = item.file_path as string | undefined;
  if (!filePath) return null;

  const projectVal = item.project as string | undefined;

  return {
    file_path: filePath,
    root_path: item.root_path as string | undefined,
    content,
    score: vectorScore,
    vector_score: vectorScore,
    keyword_score: undefined,
    start_line: (item.start_line as number) ?? 0,
    end_line: (item.end_line as number) ?? 0,
    language: (item.language as string) ?? "Unknown",
    project: (projectVal && projectVal.length > 0) ? projectVal : undefined,
    indexed_at: (item.indexed_at as number) ?? 0,
  };
}

// ---------------------------------------------------------------------------
// MilvusDatabase
// ---------------------------------------------------------------------------

/**
 * Milvus-backed vector database for code embeddings.
 *
 * Implements the VectorDatabase interface using the Milvus REST API v2.
 * Does not implement StorageBackend -- it is RAG-only.
 */
export class MilvusDatabase implements VectorDatabase {
  readonly baseUrl: string;
  readonly collectionName: string;

  constructor(url?: string, collectionName?: string) {
    this.baseUrl = (url ?? DEFAULT_URL).replace(/\/$/, "");
    this.collectionName = collectionName ?? DEFAULT_COLLECTION;
  }

  /** Return the default Milvus URL. */
  static defaultUrl(): string {
    return DEFAULT_URL;
  }

  // -- private helpers ----------------------------------------------------

  private async apiPost(
    path: string,
    body: unknown,
  ): Promise<Record<string, unknown>> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(text) as Record<string, unknown>;
    } catch {
      throw new Error(
        `Milvus returned non-JSON (HTTP ${res.status}): ${text}`,
      );
    }

    // Milvus REST v2 uses a `code` field -- 0 or 200 means success.
    const code = parsed.code as number | undefined;
    if (code !== undefined && code !== 0 && code !== 200) {
      const message = (parsed.message as string) ?? "unknown error";
      throw new Error(
        `Milvus API error on ${path}: code=${code}, message=${message}`,
      );
    }

    return parsed;
  }

  private async collectionExists(): Promise<boolean> {
    const resp = await this.apiPost("/v2/vectordb/collections/has", {
      collectionName: this.collectionName,
    });
    return (resp.data as Record<string, unknown>)?.has === true;
  }

  // -- VectorDatabase -----------------------------------------------------

  async initialize(dimension: number): Promise<void> {
    if (await this.collectionExists()) {
      // Ensure collection is loaded.
      await this.apiPost("/v2/vectordb/collections/load", {
        collectionName: this.collectionName,
      });
      return;
    }

    await this.apiPost("/v2/vectordb/collections/create", {
      collectionName: this.collectionName,
      schema: {
        autoId: true,
        enableDynamicField: true,
        fields: [
          { fieldName: "id", dataType: "Int64", isPrimary: true, autoID: true },
          {
            fieldName: "embedding",
            dataType: "FloatVector",
            elementTypeParams: { dim: dimension },
          },
          {
            fieldName: "file_path",
            dataType: "VarChar",
            elementTypeParams: { max_length: 2048 },
          },
          {
            fieldName: "root_path",
            dataType: "VarChar",
            elementTypeParams: { max_length: 2048 },
          },
          {
            fieldName: "project",
            dataType: "VarChar",
            elementTypeParams: { max_length: 512 },
          },
          { fieldName: "start_line", dataType: "Int64" },
          { fieldName: "end_line", dataType: "Int64" },
          {
            fieldName: "language",
            dataType: "VarChar",
            elementTypeParams: { max_length: 128 },
          },
          {
            fieldName: "extension",
            dataType: "VarChar",
            elementTypeParams: { max_length: 32 },
          },
          {
            fieldName: "file_hash",
            dataType: "VarChar",
            elementTypeParams: { max_length: 128 },
          },
          { fieldName: "indexed_at", dataType: "Int64" },
          {
            fieldName: "content",
            dataType: "VarChar",
            elementTypeParams: { max_length: 65535 },
          },
        ],
      },
      indexParams: [
        {
          fieldName: "embedding",
          indexName: "embedding_index",
          metricType: "COSINE",
        },
      ],
    });

    // Load collection into memory.
    await this.apiPost("/v2/vectordb/collections/load", {
      collectionName: this.collectionName,
    });
  }

  async storeEmbeddings(
    embeddings: number[][],
    metadata: ChunkMetadata[],
    contents: string[],
    rootPath: string,
  ): Promise<number> {
    if (embeddings.length === 0) return 0;

    let inserted = 0;

    for (let start = 0; start < embeddings.length; start += INSERT_BATCH_SIZE) {
      const end = Math.min(start + INSERT_BATCH_SIZE, embeddings.length);
      const body = buildInsertBody(
        this.collectionName,
        embeddings.slice(start, end),
        metadata.slice(start, end),
        contents.slice(start, end),
        rootPath,
      );

      const resp = await this.apiPost("/v2/vectordb/entities/insert", body);
      const batchCount =
        ((resp.data as Record<string, unknown>)?.insertCount as number) ??
          (end - start);
      inserted += batchCount;
    }

    return inserted;
  }

  // deno-lint-ignore require-await
  async search(
    queryVector: number[],
    queryText: string,
    limit: number,
    minScore: number,
    project?: string,
    rootPath?: string,
    hybrid?: boolean,
  ): Promise<SearchResult[]> {
    return this.searchFiltered(
      queryVector,
      queryText,
      limit,
      minScore,
      project,
      rootPath,
      hybrid,
    );
  }

  async searchFiltered(
    queryVector: number[],
    _queryText: string,
    limit: number,
    minScore: number,
    project?: string,
    rootPath?: string,
    _hybrid?: boolean,
    fileExtensions?: string[],
    languages?: string[],
    pathPatterns?: string[],
  ): Promise<SearchResult[]> {
    const filterExpr = buildFilterExpr(
      project,
      rootPath,
      fileExtensions,
      languages,
    );
    const body = buildSearchBody(
      this.collectionName,
      queryVector,
      limit,
      filterExpr,
    );

    const resp = await this.apiPost("/v2/vectordb/entities/search", body);
    const data = (resp.data as Record<string, unknown>[]) ?? [];

    let results = data
      .map((item) => parseMilvusResult(item, minScore))
      .filter((r): r is SearchResult => r !== null);

    // Post-filter by path patterns.
    if (pathPatterns && pathPatterns.length > 0) {
      results = results.filter((r) =>
        pathPatterns.some((p) => r.file_path.includes(p))
      );
    }

    return results;
  }

  async deleteByFile(filePath: string): Promise<number> {
    const filter = `file_path == "${escapeFilterValue(filePath)}"`;
    await this.apiPost("/v2/vectordb/entities/delete", {
      collectionName: this.collectionName,
      filter,
    });
    return 0;
  }

  async clear(): Promise<void> {
    await this.apiPost("/v2/vectordb/collections/drop", {
      collectionName: this.collectionName,
    });
  }

  async getStatistics(): Promise<DatabaseStats> {
    const resp = await this.apiPost("/v2/vectordb/collections/describe", {
      collectionName: this.collectionName,
    });

    const data = resp.data as Record<string, unknown> | undefined;
    let rowCount = 0;
    if (data?.rowCount !== undefined) {
      const rc = data.rowCount;
      if (typeof rc === "string") {
        rowCount = parseInt(rc, 10) || 0;
      } else if (typeof rc === "number") {
        rowCount = rc;
      }
    }

    return {
      total_points: rowCount,
      total_vectors: rowCount,
      language_breakdown: [],
    };
  }

  async flush(): Promise<void> {
    // Milvus REST API v2 does not expose a flush endpoint.
  }

  async countByRootPath(rootPath: string): Promise<number> {
    const filter = `root_path == "${escapeFilterValue(rootPath)}"`;
    const resp = await this.apiPost("/v2/vectordb/entities/query", {
      collectionName: this.collectionName,
      filter,
      outputFields: ["id"],
      limit: QUERY_LIMIT,
    });

    const data = (resp.data as unknown[]) ?? [];
    return data.length;
  }

  async getIndexedFiles(rootPath: string): Promise<string[]> {
    const filter = `root_path == "${escapeFilterValue(rootPath)}"`;
    const resp = await this.apiPost("/v2/vectordb/entities/query", {
      collectionName: this.collectionName,
      filter,
      outputFields: ["file_path"],
      limit: QUERY_LIMIT,
    });

    const data = (resp.data as Record<string, unknown>[]) ?? [];
    const paths = new Set<string>();
    for (const item of data) {
      const fp = item.file_path as string | undefined;
      if (fp) paths.add(fp);
    }
    return [...paths];
  }

  async searchWithEmbeddings(
    queryVector: number[],
    queryText: string,
    limit: number,
    minScore: number,
    project?: string,
    rootPath?: string,
    hybrid?: boolean,
  ): Promise<[SearchResult[], number[][]]> {
    const results = await this.search(
      queryVector,
      queryText,
      limit,
      minScore,
      project,
      rootPath,
      hybrid,
    );
    return [results, results.map(() => [] as number[])];
  }
}
