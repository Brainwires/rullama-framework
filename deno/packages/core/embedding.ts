/** Interface for text embedding generation.
 * Equivalent to Rust's `EmbeddingProvider` trait in rullama-core. */
export interface EmbeddingProvider {
  /** Generate an embedding for a single text. */
  embed(text: string): Promise<number[]>;

  /** Generate embeddings for a batch of texts. Default: calls embed in a loop. */
  embedBatch(texts: string[]): Promise<number[][]>;

  /** Get the dimensionality of the embedding vectors. */
  readonly dimension: number;

  /** Get the model name (e.g. "all-MiniLM-L6-v2"). */
  readonly modelName: string;
}
