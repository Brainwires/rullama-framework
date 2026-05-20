/**
 * ToolEmbedding — semantic tool discovery via embedding similarity.
 *
 * Embeds tool names and descriptions as vectors, then uses cosine similarity
 * to find semantically relevant tools for a given query.
 *
 * Equivalent to Rust's `brainwires_tools::tool_embedding` module. The Rust
 * version uses `FastEmbedManager` (ONNX runtime) directly; the Deno port
 * accepts any {@link EmbeddingProvider} from `@brainwires/core`, letting the
 * host inject whichever embedding backend is available (e.g. an HTTP proxy
 * over the Rust service, or an in-process JS implementation).
 */

import type { EmbeddingProvider } from "@brainwires/core";

interface ToolEmbeddingEntry {
  name: string;
  embedding: number[];
}

/**
 * Pre-computed embedding index for semantic tool discovery.
 *
 * Stores embeddings of tool `"{name}: {description}"` strings and supports
 * cosine-similarity search against user queries.
 */
export class ToolEmbeddingIndex {
  private readonly entries: ToolEmbeddingEntry[];
  private readonly _tool_count: number;
  private readonly provider: EmbeddingProvider;

  private constructor(
    entries: ToolEmbeddingEntry[],
    tool_count: number,
    provider: EmbeddingProvider,
  ) {
    this.entries = entries;
    this._tool_count = tool_count;
    this.provider = provider;
  }

  /**
   * Build an index from tool (name, description) pairs.
   * Each tool is embedded as `"{name}: {description}"`.
   * Returns an empty index if no tools are provided.
   */
  static async build(
    tools: ReadonlyArray<readonly [string, string]>,
    provider: EmbeddingProvider,
  ): Promise<ToolEmbeddingIndex> {
    if (tools.length === 0) {
      return new ToolEmbeddingIndex([], 0, provider);
    }

    const texts = tools.map(([name, desc]) => `${name}: ${desc}`);
    const embeddings = await provider.embedBatch(texts);

    const entries: ToolEmbeddingEntry[] = tools.map(([name, _], i) => ({
      name,
      embedding: embeddings[i],
    }));

    return new ToolEmbeddingIndex(entries, tools.length, provider);
  }

  /**
   * Search for tools semantically similar to the query.
   * Returns [tool_name, similarity_score] pairs, sorted desc, filtered by
   * `min_score`, capped at `limit`.
   */
  async search(
    query: string,
    limit: number,
    min_score: number,
  ): Promise<Array<[string, number]>> {
    if (this.entries.length === 0) return [];

    const queryVec = await this.provider.embed(query);

    const scored: Array<[string, number]> = [];
    for (const entry of this.entries) {
      const score = cosineSimilarity(queryVec, entry.embedding);
      if (score >= min_score) scored.push([entry.name, score]);
    }
    scored.sort((a, b) => b[1] - a[1]);
    return scored.slice(0, limit);
  }

  /** Number of tools in the index (for staleness detection). */
  tool_count(): number {
    return this._tool_count;
  }
}

/** Cosine similarity between two vectors. */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i];
    const bi = b[i];
    dot += ai * bi;
    normA += ai * ai;
    normB += bi * bi;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
