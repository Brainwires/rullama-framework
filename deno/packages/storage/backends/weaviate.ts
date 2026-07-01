/**
 * Weaviate vector database backend implementing VectorDatabase.
 *
 * Port of the Rust `rullama-storage/src/databases/weaviate/mod.rs`.
 *
 * Uses `fetch()` to Weaviate REST + GraphQL APIs -- zero npm dependencies.
 * @module
 */

import type {
  ChunkMetadata,
  DatabaseStats,
  SearchResult,
} from "@rullama/core";
import type { VectorDatabase } from "../traits.ts";

const DEFAULT_URL = "http://localhost:8080";
const DEFAULT_CLASS_NAME = "CodeEmbedding";
const BATCH_SIZE = 100;

// ---------------------------------------------------------------------------
// Weaviate helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Build a Weaviate `where` filter from optional query parameters. */
export function buildWhereFilter(
  project?: string,
  rootPath?: string,
  fileExtensions?: string[],
  languages?: string[],
): Record<string, unknown> | undefined {
  const operands: Record<string, unknown>[] = [];

  if (project) {
    operands.push({
      path: ["project"],
      operator: "Equal",
      valueText: project,
    });
  }
  if (rootPath) {
    operands.push({
      path: ["root_path"],
      operator: "Equal",
      valueText: rootPath,
    });
  }
  if (fileExtensions && fileExtensions.length > 0) {
    operands.push({
      path: ["extension"],
      operator: "ContainsAny",
      valueTextArray: fileExtensions,
    });
  }
  if (languages && languages.length > 0) {
    operands.push({
      path: ["language"],
      operator: "ContainsAny",
      valueTextArray: languages,
    });
  }

  if (operands.length === 0) return undefined;
  if (operands.length === 1) return operands[0];
  return { operator: "And", operands };
}

/** Build a GraphQL query string for Weaviate Get queries. */
export function buildSearchQuery(
  className: string,
  queryVector: number[],
  limit: number,
  hybrid: boolean,
  queryText: string,
  whereFilter?: Record<string, unknown>,
): string {
  const vectorStr = `[${queryVector.join(", ")}]`;

  let searchOperator: string;
  if (hybrid) {
    const escaped = queryText.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
    searchOperator =
      `hybrid: { query: "${escaped}", vector: ${vectorStr}, alpha: 0.7 }`;
  } else {
    searchOperator = `nearVector: { vector: ${vectorStr} }`;
  }

  const whereClause = whereFilter
    ? `, where: ${JSON.stringify(whereFilter)}`
    : "";

  const fields =
    "file_path root_path content project start_line end_line language extension indexed_at _additional { score }";

  return `{ Get { ${className}(${searchOperator}, limit: ${limit}${whereClause}) { ${fields} } } }`;
}

/** Build a GraphQL Aggregate query for counting objects. */
export function buildAggregateQuery(
  className: string,
  whereFilter?: Record<string, unknown>,
): string {
  const whereClause = whereFilter
    ? `(where: ${JSON.stringify(whereFilter)})`
    : "";
  return `{ Aggregate { ${className}${whereClause} { meta { count } } } }`;
}

/** Parse a single Weaviate GraphQL result object into a SearchResult. */
export function parseWeaviateResult(
  obj: Record<string, unknown>,
): SearchResult | null {
  const filePath = obj.file_path as string | undefined;
  if (!filePath) return null;

  const content = obj.content as string | undefined;
  if (!content) return null;

  const startLine = obj.start_line as number | undefined;
  if (startLine === undefined) return null;

  const endLine = obj.end_line as number | undefined;
  if (endLine === undefined) return null;

  const additional = obj._additional as Record<string, unknown> | undefined;
  const scoreStr = additional?.score as string | undefined;
  const score = scoreStr ? parseFloat(scoreStr) : 0;

  return {
    file_path: filePath,
    root_path: obj.root_path as string | undefined,
    content,
    score,
    vector_score: score,
    keyword_score: undefined,
    start_line: startLine,
    end_line: endLine,
    language: (obj.language as string) ?? "Unknown",
    project: obj.project as string | undefined,
    indexed_at: (obj.indexed_at as number) ?? 0,
  };
}

/** Build a batch object for Weaviate upsert. */
export function buildBatchObject(
  className: string,
  embedding: number[],
  meta: ChunkMetadata,
  content: string,
  rootPath: string,
): Record<string, unknown> {
  // Deterministic UUID from file_path + line range.
  const id = deterministicUuid(meta.file_path, meta.start_line, meta.end_line);
  return {
    id,
    class: className,
    properties: {
      file_path: meta.file_path,
      root_path: meta.root_path ?? rootPath,
      project: meta.project ?? "",
      start_line: meta.start_line,
      end_line: meta.end_line,
      language: meta.language ?? "Unknown",
      extension: meta.extension ?? "",
      file_hash: meta.file_hash,
      indexed_at: meta.indexed_at,
      content,
    },
    vector: embedding,
  };
}

/**
 * Generate a deterministic UUID-like string from file path and line range.
 * Uses a simple hash approach (no crypto deps needed for testing).
 */
export function deterministicUuid(
  filePath: string,
  startLine: number,
  endLine: number,
): string {
  const input = `${filePath}:${startLine}:${endLine}`;
  // Simple FNV-1a-like hash producing a UUID-formatted string.
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  let h3 = 0xdeadbeef;
  let h4 = 0xcafebabe;
  for (let i = 0; i < input.length; i++) {
    const c = input.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ c, 0x811c9dc5) >>> 0;
    h3 = Math.imul(h3 ^ c, 0x01000193) >>> 0;
    h4 = Math.imul(h4 ^ c, 0x811c9dc5) >>> 0;
  }
  const hex = (n: number, len: number) =>
    (n >>> 0).toString(16).padStart(len, "0").slice(-len);
  return `${hex(h1, 8)}-${hex(h2, 4)}-${hex((h3 & 0x0fff) | 0x5000, 4)}-${
    hex((h4 & 0x3fff) | 0x8000, 4)
  }-${hex(h1 ^ h3, 8)}${hex(h2 ^ h4, 4)}`;
}

// ---------------------------------------------------------------------------
// WeaviateDatabase
// ---------------------------------------------------------------------------

/**
 * Weaviate-backed vector database for code embeddings.
 *
 * Implements the VectorDatabase interface using Weaviate REST + GraphQL APIs.
 * Does not implement StorageBackend -- it is RAG-only.
 */
export class WeaviateDatabase implements VectorDatabase {
  readonly baseUrl: string;
  readonly className: string;

  constructor(url?: string, className?: string) {
    this.baseUrl = (url ?? DEFAULT_URL).replace(/\/$/, "");
    this.className = className ?? DEFAULT_CLASS_NAME;
  }

  /** Return the default Weaviate URL. */
  static defaultUrl(): string {
    return DEFAULT_URL;
  }

  // -- private helpers ----------------------------------------------------

  private async graphql(query: string): Promise<Record<string, unknown>> {
    const res = await fetch(`${this.baseUrl}/v1/graphql`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(
        `Weaviate GraphQL failed (${res.status}): ${text}`,
      );
    }
    return await res.json() as Record<string, unknown>;
  }

  private async restRequest(
    path: string,
    method: string,
    body?: unknown,
  ): Promise<Record<string, unknown>> {
    const opts: RequestInit = {
      method,
      headers: { "Content-Type": "application/json" },
    };
    if (body !== undefined) {
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(`${this.baseUrl}${path}`, opts);
    if (!res.ok && res.status !== 404) {
      const text = await res.text();
      throw new Error(
        `Weaviate ${method} ${path} failed (${res.status}): ${text}`,
      );
    }
    if (res.status === 204 || res.status === 404) {
      return { status: res.status };
    }
    return await res.json() as Record<string, unknown>;
  }

  private async classExists(): Promise<boolean> {
    const res = await fetch(
      `${this.baseUrl}/v1/schema/${this.className}`,
    );
    return res.ok;
  }

  // -- VectorDatabase -----------------------------------------------------

  async initialize(_dimension: number): Promise<void> {
    if (await this.classExists()) return;

    const schema = {
      class: this.className,
      vectorizer: "none",
      vectorIndexConfig: { distance: "cosine" },
      properties: [
        { name: "file_path", dataType: ["text"] },
        { name: "root_path", dataType: ["text"] },
        { name: "project", dataType: ["text"] },
        { name: "start_line", dataType: ["int"] },
        { name: "end_line", dataType: ["int"] },
        { name: "language", dataType: ["text"] },
        { name: "extension", dataType: ["text"] },
        { name: "file_hash", dataType: ["text"] },
        { name: "indexed_at", dataType: ["int"] },
        { name: "content", dataType: ["text"] },
      ],
    };

    await this.restRequest("/v1/schema", "POST", schema);
  }

  async storeEmbeddings(
    embeddings: number[][],
    metadata: ChunkMetadata[],
    contents: string[],
    rootPath: string,
  ): Promise<number> {
    if (embeddings.length === 0) return 0;

    const objects = embeddings.map((emb, idx) =>
      buildBatchObject(
        this.className,
        emb,
        metadata[idx],
        contents[idx],
        rootPath,
      )
    );

    let stored = 0;
    for (let start = 0; start < objects.length; start += BATCH_SIZE) {
      const chunk = objects.slice(start, start + BATCH_SIZE);
      await this.restRequest("/v1/batch/objects", "POST", {
        objects: chunk,
      });
      stored += chunk.length;
    }

    return stored;
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
    queryText: string,
    limit: number,
    minScore: number,
    project?: string,
    rootPath?: string,
    hybrid?: boolean,
    fileExtensions?: string[],
    languages?: string[],
    pathPatterns?: string[],
  ): Promise<SearchResult[]> {
    const whereFilter = buildWhereFilter(
      project,
      rootPath,
      fileExtensions,
      languages,
    );

    const gql = buildSearchQuery(
      this.className,
      queryVector,
      limit,
      hybrid ?? false,
      queryText,
      whereFilter,
    );

    const response = await this.graphql(gql);

    const items = (
      (response.data as Record<string, unknown>)
        ?.Get as Record<string, unknown>
    )?.[this.className] as Record<string, unknown>[] | undefined;

    let results = (items ?? [])
      .map(parseWeaviateResult)
      .filter((r): r is SearchResult => r !== null)
      .filter((r) => r.score >= minScore);

    // Post-filter by path patterns.
    if (pathPatterns && pathPatterns.length > 0) {
      results = results.filter((r) =>
        pathPatterns.some((p) => r.file_path.includes(p))
      );
    }

    // Sort descending by score.
    results.sort((a, b) => b.score - a.score);

    return results;
  }

  async deleteByFile(filePath: string): Promise<number> {
    await this.restRequest("/v1/batch/objects/delete", "POST", {
      match: {
        class: this.className,
        where: {
          path: ["file_path"],
          operator: "Equal",
          valueText: filePath,
        },
      },
    });
    return 0;
  }

  async clear(): Promise<void> {
    await this.restRequest(`/v1/schema/${this.className}`, "DELETE");
  }

  async getStatistics(): Promise<DatabaseStats> {
    const gql = buildAggregateQuery(this.className);
    const response = await this.graphql(gql);

    const agg = (
      (response.data as Record<string, unknown>)
        ?.Aggregate as Record<string, unknown>
    )?.[this.className] as Record<string, unknown>[] | undefined;

    const count = (agg?.[0]?.meta as Record<string, unknown>)
      ?.count as number ?? 0;

    return {
      total_points: count,
      total_vectors: count,
      language_breakdown: [],
    };
  }

  async flush(): Promise<void> {
    // Weaviate persists automatically.
  }

  async countByRootPath(rootPath: string): Promise<number> {
    const whereFilter = {
      path: ["root_path"],
      operator: "Equal",
      valueText: rootPath,
    };
    const gql = buildAggregateQuery(this.className, whereFilter);
    const response = await this.graphql(gql);

    const agg = (
      (response.data as Record<string, unknown>)
        ?.Aggregate as Record<string, unknown>
    )?.[this.className] as Record<string, unknown>[] | undefined;

    return (agg?.[0]?.meta as Record<string, unknown>)?.count as number ?? 0;
  }

  async getIndexedFiles(rootPath: string): Promise<string[]> {
    const whereFilter = {
      path: ["root_path"],
      operator: "Equal",
      valueText: rootPath,
    };
    const whereStr = JSON.stringify(whereFilter);

    const gql =
      `{ Get { ${this.className}(where: ${whereStr}, limit: 10000) { file_path } } }`;
    const response = await this.graphql(gql);

    const items = (
      (response.data as Record<string, unknown>)
        ?.Get as Record<string, unknown>
    )?.[this.className] as Record<string, unknown>[] | undefined;

    const paths = new Set<string>();
    for (const item of items ?? []) {
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
