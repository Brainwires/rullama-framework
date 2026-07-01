// Example: RAG pipeline with custom embedding and vector search
// Shows how to implement EmbeddingProvider and VectorStore for retrieval-augmented generation.
// Run: deno run deno/examples/core/rag_pipeline.ts

import {
  type ChunkMetadata,
  type EmbeddingProvider,
  type SearchResult,
  type VectorSearchResult,
  type VectorStore,
} from "@rullama/core";

// 1. Implement a custom embedding provider
// A toy hash-based embedding (deterministic, not semantic).
// Replace with a real model (e.g., ONNX Runtime, OpenAI Embeddings API).

class HashEmbedding implements EmbeddingProvider {
  readonly dimension: number;
  readonly modelName: string;

  constructor(dim: number) {
    this.dimension = dim;
    this.modelName = "hash-embedding-v1";
  }

  async embed(text: string): Promise<number[]> {
    const vec = new Array<number>(this.dimension).fill(0);
    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);
    for (let i = 0; i < bytes.length; i++) {
      vec[i % this.dimension] += bytes[i] / 255.0;
    }
    // Normalize to unit length
    const norm = Math.sqrt(vec.reduce((sum, x) => sum + x * x, 0));
    if (norm > 0) {
      for (let i = 0; i < vec.length; i++) {
        vec[i] /= norm;
      }
    }
    return vec;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const results: number[][] = [];
    for (const text of texts) {
      results.push(await this.embed(text));
    }
    return results;
  }
}

// 2. Implement an in-memory vector store

interface StoredItem {
  id: string;
  embedding: number[];
  content: string;
  metadata: ChunkMetadata;
}

class InMemoryVectorStore implements VectorStore {
  private items: StoredItem[] = [];
  private dim = 0;

  async initialize(dimension: number): Promise<void> {
    this.dim = dimension;
    console.log(`  Vector store initialized (dimension=${dimension})`);
  }

  async upsert(
    ids: string[],
    embeddings: number[][],
    contents: string[],
    metadata: ChunkMetadata[],
  ): Promise<number> {
    for (let i = 0; i < ids.length; i++) {
      // Remove existing item with same id
      this.items = this.items.filter((item) => item.id !== ids[i]);
      this.items.push({
        id: ids[i],
        embedding: embeddings[i],
        content: contents[i],
        metadata: metadata[i],
      });
    }
    return ids.length;
  }

  async search(
    queryVector: number[],
    limit: number,
    minScore: number,
  ): Promise<VectorSearchResult[]> {
    const scored = this.items.map((item) => ({
      item,
      score: cosineSimilarity(queryVector, item.embedding),
    }));

    return scored
      .filter((s) => s.score >= minScore)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map((s) => ({
        id: s.item.id,
        score: s.score,
        content: s.item.content,
        metadata: s.item.metadata,
      }));
  }

  async delete(ids: string[]): Promise<number> {
    const idSet = new Set(ids);
    const before = this.items.length;
    this.items = this.items.filter((item) => !idSet.has(item.id));
    return before - this.items.length;
  }

  async clear(): Promise<void> {
    this.items = [];
  }

  async count(): Promise<number> {
    return this.items.length;
  }
}

// Helper: cosine similarity between two vectors
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}

// 3. Build a simple RAG pipeline

async function main() {
  console.log("=== RAG Pipeline Example ===");

  // Setup embedding provider
  const embedder = new HashEmbedding(64);
  console.log(`\nModel: ${embedder.modelName}`);
  console.log(`Dimension: ${embedder.dimension}`);

  // 4. Demonstrate embedding similarity
  console.log("\n=== Embedding Similarity ===");
  const a = await embedder.embed("machine learning algorithms");
  const b = await embedder.embed("deep learning neural networks");
  const c = await embedder.embed("banana smoothie recipe");

  console.log(
    `  'machine learning' vs 'deep learning': ${
      cosineSimilarity(a, b).toFixed(4)
    }`,
  );
  console.log(
    `  'machine learning' vs 'banana smoothie': ${
      cosineSimilarity(a, c).toFixed(4)
    }`,
  );

  // Batch embedding
  const batchTexts = ["first document", "second document", "third document"];
  const batchResults = await embedder.embedBatch(batchTexts);
  console.log(
    `  Batch: embedded ${batchResults.length} texts, each ${
      batchResults[0].length
    } dims`,
  );

  // 5. Index code chunks into the vector store
  console.log("\n=== Indexing Code Chunks ===");
  const store = new InMemoryVectorStore();
  await store.initialize(embedder.dimension);

  const codeChunks: { id: string; content: string; metadata: ChunkMetadata }[] =
    [
      {
        id: "chunk-1",
        content:
          "export async function authenticate(token: string): Promise<User> {\n  const decoded = jwt.verify(token, SECRET);\n  return findUser(decoded.sub);\n}",
        metadata: {
          file_path: "src/auth.ts",
          start_line: 10,
          end_line: 14,
          language: "TypeScript",
          extension: "ts",
          file_hash: "abc123",
          indexed_at: Math.floor(Date.now() / 1000),
        },
      },
      {
        id: "chunk-2",
        content:
          "export function hashPassword(password: string): string {\n  return bcrypt.hashSync(password, SALT_ROUNDS);\n}",
        metadata: {
          file_path: "src/auth.ts",
          start_line: 20,
          end_line: 23,
          language: "TypeScript",
          extension: "ts",
          file_hash: "abc123",
          indexed_at: Math.floor(Date.now() / 1000),
        },
      },
      {
        id: "chunk-3",
        content:
          "export class DatabaseConnection {\n  constructor(private url: string) {}\n  async query(sql: string): Promise<Row[]> { /* ... */ }\n}",
        metadata: {
          file_path: "src/db.ts",
          start_line: 1,
          end_line: 5,
          language: "TypeScript",
          extension: "ts",
          file_hash: "def456",
          indexed_at: Math.floor(Date.now() / 1000),
        },
      },
      {
        id: "chunk-4",
        content:
          "const router = new Router();\nrouter.get('/api/users', listUsers);\nrouter.post('/api/users', createUser);\nrouter.delete('/api/users/:id', deleteUser);",
        metadata: {
          file_path: "src/routes.ts",
          start_line: 5,
          end_line: 9,
          language: "TypeScript",
          extension: "ts",
          file_hash: "ghi789",
          indexed_at: Math.floor(Date.now() / 1000),
        },
      },
    ];

  // Embed and store all chunks
  const ids = codeChunks.map((c) => c.id);
  const contents = codeChunks.map((c) => c.content);
  const metadatas = codeChunks.map((c) => c.metadata);
  const embeddings = await embedder.embedBatch(contents);

  const stored = await store.upsert(ids, embeddings, contents, metadatas);
  console.log(`  Stored ${stored} chunks (total: ${await store.count()})`);

  // 6. Semantic search
  console.log("\n=== Semantic Search ===");
  const queries = [
    "authentication and token verification",
    "database connection and SQL queries",
    "REST API routes and endpoints",
  ];

  for (const query of queries) {
    const queryVec = await embedder.embed(query);
    const results = await store.search(queryVec, 2, 0.0);

    console.log(`\n  Query: "${query}"`);
    for (const result of results) {
      const meta = result.metadata as ChunkMetadata;
      const preview = result.content.split("\n")[0].slice(0, 60);
      console.log(
        `    [${
          result.score.toFixed(4)
        }] ${meta.file_path}:${meta.start_line} — ${preview}...`,
      );
    }
  }

  // 7. Build search results in the standard format
  console.log("\n=== Formatted Search Results ===");
  const queryVec = await embedder.embed("user authentication");
  const topResults = await store.search(queryVec, 3, 0.0);

  const searchResults: SearchResult[] = topResults.map((r) => {
    const meta = r.metadata as ChunkMetadata;
    return {
      file_path: meta.file_path,
      root_path: meta.root_path,
      content: r.content,
      score: r.score,
      vector_score: r.score,
      start_line: meta.start_line,
      end_line: meta.end_line,
      language: meta.language ?? "unknown",
      project: meta.project,
      indexed_at: meta.indexed_at,
    };
  });

  for (const result of searchResults) {
    console.log(
      `  [${
        result.score.toFixed(4)
      }] ${result.file_path}:${result.start_line}-${result.end_line} (${result.language})`,
    );
  }

  // 8. Cleanup
  const deleted = await store.delete(["chunk-1"]);
  console.log(
    `\n  Deleted ${deleted} chunk(s), remaining: ${await store.count()}`,
  );
  await store.clear();
  console.log(`  Cleared store, remaining: ${await store.count()}`);

  console.log(
    "\nDone! Plug in a real embedding model and vector database for production RAG.",
  );
}

await main();
