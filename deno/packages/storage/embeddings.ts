/**
 * Embedding provider wrapper interface for the storage layer.
 *
 * Re-exports the core EmbeddingProvider and adds a cached variant.
 * Equivalent to Rust's `embeddings.rs` in rullama-storage.
 * @module
 */

import type { EmbeddingProvider } from "@rullama/core";

export type { EmbeddingProvider };

/**
 * LRU-cached embedding provider.
 *
 * Wraps any EmbeddingProvider and memoizes results to reduce latency
 * for repeated queries in agent loops.
 */
export class CachedEmbeddingProvider implements EmbeddingProvider {
  private readonly inner: EmbeddingProvider;
  private readonly cache: Map<string, number[]>;
  private readonly order: string[];
  private readonly maxSize: number;

  readonly dimension: number;
  readonly modelName: string;

  constructor(inner: EmbeddingProvider, maxSize = 1000) {
    this.inner = inner;
    this.cache = new Map();
    this.order = [];
    this.maxSize = maxSize;
    this.dimension = inner.dimension;
    this.modelName = inner.modelName;
  }

  /** Embed with caching. */
  async embed(text: string): Promise<number[]> {
    const cached = this.cache.get(text);
    if (cached !== undefined) {
      return cached;
    }

    const embedding = await this.inner.embed(text);

    // Evict oldest if at capacity
    if (this.order.length >= this.maxSize) {
      const evicted = this.order.shift()!;
      this.cache.delete(evicted);
    }

    this.cache.set(text, embedding);
    this.order.push(text);
    return embedding;
  }

  /** Batch embed (delegates to inner, no per-item caching). */
  async embedBatch(texts: string[]): Promise<number[][]> {
    return await this.inner.embedBatch(texts);
  }

  /** Number of cached embeddings. */
  get cacheLength(): number {
    return this.cache.size;
  }

  /** Clear the cache. */
  clearCache(): void {
    this.cache.clear();
    this.order.length = 0;
  }
}
