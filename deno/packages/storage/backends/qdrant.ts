/**
 * Qdrant vector database backend implementing VectorDatabase.
 *
 * Port of the Rust `rullama-storage/src/databases/qdrant/mod.rs`.
 *
 * Uses the Qdrant REST API via `fetch()` -- no npm dependency required.
 * @module
 */

import type {
  ChunkMetadata,
  DatabaseStats,
  SearchResult,
} from "@rullama/core";
import type { VectorDatabase } from "../traits.ts";

const COLLECTION_NAME = "code_embeddings";
const DEFAULT_URL = "http://localhost:6333";

// ---------------------------------------------------------------------------
// Qdrant REST helpers
// ---------------------------------------------------------------------------

/** Qdrant filter condition. */
interface QdrantCondition {
  key: string;
  match?: { value: string | string[] };
}

/** Qdrant filter object. */
interface QdrantFilter {
  must?: QdrantCondition[];
}

/** Build a Qdrant filter object from search parameters. */
export function buildQdrantFilter(
  project?: string,
  rootPath?: string,
  fileExtensions?: string[],
  languages?: string[],
): QdrantFilter | undefined {
  const must: QdrantCondition[] = [];

  if (project) {
    must.push({ key: "project", match: { value: project } });
  }
  if (rootPath) {
    must.push({ key: "root_path", match: { value: rootPath } });
  }
  if (fileExtensions && fileExtensions.length > 0) {
    must.push({ key: "extension", match: { value: fileExtensions } });
  }
  if (languages && languages.length > 0) {
    must.push({ key: "language", match: { value: languages } });
  }

  return must.length > 0 ? { must } : undefined;
}

/** Build a Qdrant upsert request body. */
export function buildUpsertBody(
  embeddings: number[][],
  metadata: ChunkMetadata[],
  contents: string[],
): { points: unknown[] } {
  const points = embeddings.map((embedding, idx) => {
    const meta = metadata[idx];
    return {
      id: idx,
      vector: embedding,
      payload: {
        file_path: meta.file_path,
        root_path: meta.root_path ?? null,
        project: meta.project ?? null,
        start_line: meta.start_line,
        end_line: meta.end_line,
        language: meta.language ?? null,
        extension: meta.extension ?? null,
        file_hash: meta.file_hash,
        indexed_at: meta.indexed_at,
        content: contents[idx],
      },
    };
  });
  return { points };
}

/** Build a Qdrant search request body. */
export function buildSearchBody(
  queryVector: number[],
  limit: number,
  minScore: number,
  filter?: QdrantFilter,
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    vector: queryVector,
    limit,
    score_threshold: minScore,
    with_payload: true,
  };
  if (filter) {
    body.filter = filter;
  }
  return body;
}

/** Parse a Qdrant search result point into a SearchResult. */
export function parseSearchPoint(
  point: Record<string, unknown>,
): SearchResult | null {
  const payload = point.payload as Record<string, unknown> | undefined;
  if (!payload) return null;

  const content = payload.content as string | undefined;
  if (!content) return null;

  const filePath = payload.file_path as string | undefined;
  if (!filePath) return null;

  const startLine = payload.start_line as number | undefined;
  if (startLine === undefined) return null;

  const endLine = payload.end_line as number | undefined;
  if (endLine === undefined) return null;

  const vectorScore = point.score as number ?? 0;

  return {
    file_path: filePath,
    root_path: payload.root_path as string | undefined,
    content,
    score: vectorScore,
    vector_score: vectorScore,
    keyword_score: undefined,
    start_line: startLine,
    end_line: endLine,
    language: (payload.language as string) ?? "Unknown",
    project: payload.project as string | undefined,
    indexed_at: (payload.indexed_at as number) ?? 0,
  };
}

// ---------------------------------------------------------------------------
// QdrantDatabase
// ---------------------------------------------------------------------------

/**
 * Qdrant-backed vector database for code embeddings.
 *
 * Implements the VectorDatabase interface using Qdrant's REST API.
 * Does not implement StorageBackend -- it is RAG-only.
 */
export class QdrantDatabase implements VectorDatabase {
  readonly baseUrl: string;

  constructor(url?: string) {
    this.baseUrl = (url ?? DEFAULT_URL).replace(/\/$/, "");
  }

  /** Return the default Qdrant URL. */
  static defaultUrl(): string {
    return DEFAULT_URL;
  }

  // ── private helpers ────────────────────────────────────────────────

  private async request(
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
    if (!res.ok) {
      const text = await res.text();
      throw new Error(
        `Qdrant ${method} ${path} failed (${res.status}): ${text}`,
      );
    }
    return await res.json() as Record<string, unknown>;
  }

  private async collectionExists(): Promise<boolean> {
    const data = await this.request("/collections", "GET");
    const result = data.result as
      | { collections?: { name: string }[] }
      | undefined;
    return result?.collections?.some((c) => c.name === COLLECTION_NAME) ??
      false;
  }

  // ── VectorDatabase ─────────────────────────────────────────────────

  async initialize(dimension: number): Promise<void> {
    if (await this.collectionExists()) return;

    await this.request(`/collections/${COLLECTION_NAME}`, "PUT", {
      vectors: {
        size: dimension,
        distance: "Cosine",
      },
    });
  }

  async storeEmbeddings(
    embeddings: number[][],
    metadata: ChunkMetadata[],
    contents: string[],
    _rootPath: string,
  ): Promise<number> {
    if (embeddings.length === 0) return 0;
    const body = buildUpsertBody(embeddings, metadata, contents);
    await this.request(
      `/collections/${COLLECTION_NAME}/points`,
      "PUT",
      body,
    );
    return embeddings.length;
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
    const filter = buildQdrantFilter(
      project,
      rootPath,
      fileExtensions,
      languages,
    );
    const body = buildSearchBody(queryVector, limit, minScore, filter);

    const data = await this.request(
      `/collections/${COLLECTION_NAME}/points/search`,
      "POST",
      body,
    );

    const points = (data.result as Record<string, unknown>[]) ?? [];
    let results = points
      .map(parseSearchPoint)
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
    await this.request(
      `/collections/${COLLECTION_NAME}/points/delete`,
      "POST",
      {
        filter: {
          must: [{ key: "file_path", match: { value: filePath } }],
        },
      },
    );
    // Qdrant doesn't return the count of deleted points.
    return 0;
  }

  async clear(): Promise<void> {
    await this.request(`/collections/${COLLECTION_NAME}`, "DELETE");
  }

  async getStatistics(): Promise<DatabaseStats> {
    const data = await this.request(
      `/collections/${COLLECTION_NAME}`,
      "GET",
    );
    const result = data.result as Record<string, unknown> | undefined;
    const pointsCount = (result?.points_count as number) ?? 0;

    return {
      total_points: pointsCount,
      total_vectors: pointsCount,
      language_breakdown: [],
    };
  }

  async flush(): Promise<void> {
    // Qdrant persists automatically.
  }

  async countByRootPath(rootPath: string): Promise<number> {
    const data = await this.request(
      `/collections/${COLLECTION_NAME}/points/count`,
      "POST",
      {
        filter: {
          must: [{ key: "root_path", match: { value: rootPath } }],
        },
        exact: true,
      },
    );
    const result = data.result as { count?: number } | undefined;
    return result?.count ?? 0;
  }

  async getIndexedFiles(rootPath: string): Promise<string[]> {
    const filePaths = new Set<string>();
    let offset: number | string | null = null;

    for (;;) {
      const body: Record<string, unknown> = {
        filter: {
          must: [{ key: "root_path", match: { value: rootPath } }],
        },
        with_payload: true,
        limit: 1000,
      };
      if (offset !== null) body.offset = offset;

      const data = await this.request(
        `/collections/${COLLECTION_NAME}/points/scroll`,
        "POST",
        body,
      );

      const result = data.result as {
        points?: { payload?: { file_path?: string } }[];
        next_page_offset?: number | string | null;
      } | undefined;

      if (!result?.points?.length) break;
      for (const point of result.points) {
        if (point.payload?.file_path) {
          filePaths.add(point.payload.file_path);
        }
      }

      offset = result.next_page_offset ?? null;
      if (offset === null) break;
    }

    return [...filePaths];
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
