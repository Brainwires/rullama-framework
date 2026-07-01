/**
 * Pinecone managed cloud vector database backend implementing VectorDatabase.
 *
 * Port of the Rust `rullama-storage/src/databases/pinecone/mod.rs`.
 *
 * Uses `fetch()` to Pinecone REST API -- zero npm dependencies.
 * @module
 */

import type {
  ChunkMetadata,
  DatabaseStats,
  SearchResult,
} from "@rullama/core";
import type { VectorDatabase } from "../traits.ts";

const BATCH_SIZE = 100;

// ---------------------------------------------------------------------------
// Pinecone REST helpers (exported for testing)
// ---------------------------------------------------------------------------

/** Build a Pinecone metadata filter from search parameters. */
export function buildMetadataFilter(
  project?: string,
  rootPath?: string,
  fileExtensions?: string[],
  languages?: string[],
): Record<string, unknown> | undefined {
  const conditions: Record<string, unknown>[] = [];

  if (project) {
    conditions.push({ project: { $eq: project } });
  }
  if (rootPath) {
    conditions.push({ root_path: { $eq: rootPath } });
  }
  if (fileExtensions && fileExtensions.length > 0) {
    conditions.push({ extension: { $in: fileExtensions } });
  }
  if (languages && languages.length > 0) {
    conditions.push({ language: { $in: languages } });
  }

  if (conditions.length === 0) return undefined;
  if (conditions.length === 1) return conditions[0];
  return { $and: conditions };
}

/** Build a Pinecone upsert request body. */
export function buildUpsertBody(
  embeddings: number[][],
  metadata: ChunkMetadata[],
  contents: string[],
  rootPath: string,
  namespace: string,
): { vectors: unknown[]; namespace: string } {
  const vectors = embeddings.map((values, idx) => {
    const meta = metadata[idx];
    const id = `${rootPath}:${meta.file_path}:${meta.start_line}`;
    return {
      id,
      values,
      metadata: {
        file_path: meta.file_path,
        root_path: meta.root_path ?? rootPath,
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
  return { vectors, namespace };
}

/** Build a Pinecone query request body. */
export function buildQueryBody(
  vector: number[],
  topK: number,
  namespace: string,
  filter?: Record<string, unknown>,
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    vector,
    topK,
    namespace,
    includeMetadata: true,
  };
  if (filter) {
    body.filter = filter;
  }
  return body;
}

/** Parse a Pinecone match object into a SearchResult. */
export function parseMatch(
  match: Record<string, unknown>,
  minScore: number,
): SearchResult | null {
  const score = match.score as number ?? 0;
  if (score < minScore) return null;

  const meta = match.metadata as Record<string, unknown> | undefined;
  if (!meta) return null;

  const filePath = meta.file_path as string | undefined;
  if (!filePath) return null;

  const content = (meta.content as string) ?? "";

  return {
    file_path: filePath,
    root_path: meta.root_path as string | undefined,
    content,
    score,
    vector_score: score,
    keyword_score: undefined,
    start_line: (meta.start_line as number) ?? 0,
    end_line: (meta.end_line as number) ?? 0,
    language: (meta.language as string) ?? "Unknown",
    project: meta.project as string | undefined,
    indexed_at: (meta.indexed_at as number) ?? 0,
  };
}

/** Extract unique file paths from Pinecone list vector IDs. */
export function extractFilePathsFromIds(
  ids: string[],
  prefix: string,
): string[] {
  const paths = new Set<string>();
  for (const id of ids) {
    if (!id.startsWith(prefix)) continue;
    const rest = id.slice(prefix.length);
    const lastColon = rest.lastIndexOf(":");
    if (lastColon > 0) {
      paths.add(rest.slice(0, lastColon));
    }
  }
  return [...paths].sort();
}

// ---------------------------------------------------------------------------
// PineconeDatabase
// ---------------------------------------------------------------------------

/**
 * Pinecone-backed vector database for code embeddings.
 *
 * Implements the VectorDatabase interface using Pinecone's REST API.
 * Does not implement StorageBackend -- Pinecone is a pure vector store.
 */
export class PineconeDatabase implements VectorDatabase {
  readonly indexHost: string;
  readonly apiKey: string;
  readonly namespace: string;
  private dimension: number | null = null;

  constructor(indexHost: string, apiKey: string, namespace = "") {
    this.indexHost = indexHost.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.namespace = namespace;
  }

  // -- private helpers ----------------------------------------------------

  private async request(
    path: string,
    method: string,
    body?: unknown,
  ): Promise<Record<string, unknown>> {
    const opts: RequestInit = {
      method,
      headers: {
        "Api-Key": this.apiKey,
        "Content-Type": "application/json",
      },
    };
    if (body !== undefined) {
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(`${this.indexHost}${path}`, opts);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(
        `Pinecone ${method} ${path} failed (${res.status}): ${text}`,
      );
    }
    return await res.json() as Record<string, unknown>;
  }

  // -- VectorDatabase -----------------------------------------------------

  async initialize(dimension: number): Promise<void> {
    this.dimension = dimension;
    // Verify connectivity by fetching index stats.
    await this.request("/describe_index_stats", "POST", {});
  }

  async storeEmbeddings(
    embeddings: number[][],
    metadata: ChunkMetadata[],
    contents: string[],
    rootPath: string,
  ): Promise<number> {
    if (embeddings.length === 0) return 0;

    const total = embeddings.length;
    let stored = 0;

    for (let start = 0; start < total; start += BATCH_SIZE) {
      const end = Math.min(start + BATCH_SIZE, total);
      const batchEmbeddings = embeddings.slice(start, end);
      const batchMeta = metadata.slice(start, end);
      const batchContents = contents.slice(start, end);

      const body = buildUpsertBody(
        batchEmbeddings,
        batchMeta,
        batchContents,
        rootPath,
        this.namespace,
      );
      await this.request("/vectors/upsert", "POST", body);
      stored += end - start;
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
    const extra = (pathPatterns && pathPatterns.length > 0) ? 3 : 1;
    const filter = buildMetadataFilter(
      project,
      rootPath,
      fileExtensions,
      languages,
    );
    const body = buildQueryBody(
      queryVector,
      limit * extra,
      this.namespace,
      filter,
    );

    const data = await this.request("/query", "POST", body);
    const matches = (data.matches as Record<string, unknown>[]) ?? [];

    let results = matches
      .map((m) => parseMatch(m, minScore))
      .filter((r): r is SearchResult => r !== null);

    // Post-filter by path patterns.
    if (pathPatterns && pathPatterns.length > 0) {
      results = results.filter((r) =>
        pathPatterns.some((p) => r.file_path.includes(p))
      );
    }

    return results.slice(0, limit);
  }

  async deleteByFile(filePath: string): Promise<number> {
    await this.request("/vectors/delete", "POST", {
      filter: { file_path: { $eq: filePath } },
      namespace: this.namespace,
    });
    return 0;
  }

  async clear(): Promise<void> {
    await this.request("/vectors/delete", "POST", {
      deleteAll: true,
      namespace: this.namespace,
    });
  }

  async getStatistics(): Promise<DatabaseStats> {
    const data = await this.request("/describe_index_stats", "POST", {});
    const namespaces = data.namespaces as
      | Record<string, { vectorCount?: number }>
      | undefined;
    const totalVectors = namespaces?.[this.namespace]?.vectorCount ?? 0;

    return {
      total_points: totalVectors,
      total_vectors: totalVectors,
      language_breakdown: [],
    };
  }

  async flush(): Promise<void> {
    // Pinecone is managed -- writes are durable immediately.
  }

  async countByRootPath(rootPath: string): Promise<number> {
    const data = await this.request("/describe_index_stats", "POST", {
      filter: { root_path: { $eq: rootPath } },
    });
    const namespaces = data.namespaces as
      | Record<string, { vectorCount?: number }>
      | undefined;
    return namespaces?.[this.namespace]?.vectorCount ?? 0;
  }

  async getIndexedFiles(rootPath: string): Promise<string[]> {
    const prefix = `${rootPath}:`;
    try {
      const url = new URL(`${this.indexHost}/vectors/list`);
      url.searchParams.set("namespace", this.namespace);
      url.searchParams.set("prefix", prefix);
      url.searchParams.set("limit", "10000");

      const res = await fetch(url.toString(), {
        method: "GET",
        headers: { "Api-Key": this.apiKey },
      });

      if (!res.ok) return [];

      const data = await res.json() as Record<string, unknown>;
      const vectors = (data.vectors as { id: string }[]) ?? [];
      return extractFilePathsFromIds(
        vectors.map((v) => v.id),
        prefix,
      );
    } catch {
      return [];
    }
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
