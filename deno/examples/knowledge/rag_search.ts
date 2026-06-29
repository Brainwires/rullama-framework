// Example: RAG Search
// Demonstrates the RagClient interface for codebase indexing and hybrid semantic+keyword search.
// Run: deno run deno/examples/cognition/rag_search.ts

import type {
  IndexRequest,
  IndexResponse,
  QueryRequest,
  QueryResponse,
  RagClient,
  SearchResult,
  StatisticsResponse,
} from "@rullama/knowledge";
import {
  DEFAULT_LIMIT,
  DEFAULT_MAX_FILE_SIZE,
  DEFAULT_MIN_SCORE,
} from "@rullama/knowledge";

// ---------------------------------------------------------------------------
// Mock RagClient — in production, supply a real vector-DB-backed implementation
// ---------------------------------------------------------------------------

class MockRagClient implements RagClient {
  private indexed = false;

  async indexCodebase(req: IndexRequest): Promise<IndexResponse> {
    console.log(
      `  [mock] Indexing ${req.path} (patterns: ${
        req.includePatterns?.join(", ") ?? "*"
      })`,
    );
    this.indexed = true;
    return {
      mode: "full",
      filesIndexed: 42,
      chunksCreated: 318,
      embeddingsGenerated: 318,
      durationMs: 1234,
      errors: [],
      filesUpdated: 0,
      filesRemoved: 0,
    };
  }

  async queryCodebase(req: QueryRequest): Promise<QueryResponse> {
    const results: SearchResult[] = [
      {
        filePath: "src/knowledge/entity.ts",
        content:
          "export interface Entity {\n  name: string;\n  entityType: EntityType;",
        score: 0.92,
        vectorScore: 0.88,
        keywordScore: req.hybrid ? 0.95 : undefined,
        startLine: 10,
        endLine: 25,
        language: "TypeScript",
        indexedAt: Date.now(),
      },
      {
        filePath: "src/rag/types.ts",
        content:
          "export interface SearchResult {\n  filePath: string;\n  score: number;",
        score: 0.85,
        vectorScore: 0.85,
        startLine: 40,
        endLine: 60,
        language: "TypeScript",
        indexedAt: Date.now(),
      },
      {
        filePath: "src/core/error.ts",
        content:
          "export class FrameworkError extends Error {\n  constructor(kind, message)",
        score: 0.78,
        vectorScore: 0.78,
        startLine: 5,
        endLine: 20,
        language: "TypeScript",
        indexedAt: Date.now(),
      },
    ];

    return {
      results: results.slice(0, req.limit ?? DEFAULT_LIMIT),
      durationMs: 23,
      thresholdUsed: req.minScore ?? DEFAULT_MIN_SCORE,
      thresholdLowered: false,
    };
  }

  async getStatistics(): Promise<StatisticsResponse> {
    return {
      totalFiles: 42,
      totalChunks: 318,
      totalEmbeddings: 318,
      databaseSizeBytes: 2_456_000,
      languageBreakdown: [
        { language: "TypeScript", fileCount: 30, chunkCount: 220 },
        { language: "Rust", fileCount: 12, chunkCount: 98 },
      ],
    };
  }

  async clearIndex() {
    this.indexed = false;
    return { success: true, message: "Index cleared" };
  }

  async advancedSearch(
    req: import("@rullama/knowledge").AdvancedSearchRequest,
  ): Promise<QueryResponse> {
    return this.queryCodebase({
      query: req.query,
      limit: req.limit,
      minScore: req.minScore,
    });
  }

  async searchGitHistory(
    req: import("@rullama/knowledge").SearchGitHistoryRequest,
  ) {
    return {
      results: [],
      commitsIndexed: 0,
      totalCachedCommits: 0,
      durationMs: 5,
    };
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log("=== Brainwires RAG Search Example ===\n");

  // 1. Create the RagClient
  console.log("--- Step 1: Initialize RagClient ---\n");
  const client: RagClient = new MockRagClient();
  console.log("RagClient initialized successfully.\n");

  // 2. Index a codebase directory
  console.log("--- Step 2: Index a Codebase ---\n");

  const indexPath = "./src";
  console.log(`Indexing directory: ${indexPath}`);

  const indexRequest: IndexRequest = {
    path: indexPath,
    project: "rag-example",
    includePatterns: ["**/*.ts", "**/*.rs"],
    excludePatterns: ["**/target/**", "**/node_modules/**"],
    maxFileSize: DEFAULT_MAX_FILE_SIZE,
  };

  const indexResponse = await client.indexCodebase(indexRequest);
  console.log("Indexing complete:");
  console.log(`  Mode:       ${indexResponse.mode}`);
  console.log(`  Files:      ${indexResponse.filesIndexed}`);
  console.log(`  Chunks:     ${indexResponse.chunksCreated}`);
  console.log(`  Embeddings: ${indexResponse.embeddingsGenerated}`);
  console.log(`  Duration:   ${indexResponse.durationMs} ms`);
  if (indexResponse.errors.length > 0) {
    console.log(`  Errors:     ${indexResponse.errors.length}`);
    for (const err of indexResponse.errors) {
      console.log(`    - ${err}`);
    }
  }
  console.log();

  // 3. Check statistics
  console.log("--- Step 3: Index Statistics ---\n");

  const stats = await client.getStatistics();
  console.log(`Total files:      ${stats.totalFiles}`);
  console.log(`Total chunks:     ${stats.totalChunks}`);
  console.log(`Total embeddings: ${stats.totalEmbeddings}`);
  console.log(`Database size:    ${stats.databaseSizeBytes} bytes`);
  console.log("Language breakdown:");
  for (const lang of stats.languageBreakdown) {
    console.log(
      `  ${lang.language}: ${lang.fileCount} files, ${lang.chunkCount} chunks`,
    );
  }
  console.log();

  // 4. Perform hybrid semantic + keyword queries
  console.log("--- Step 4: Semantic Search Queries ---\n");

  const queries = [
    "entity relationship graph traversal",
    "embedding vector search",
    "error handling and result types",
  ];

  for (const queryText of queries) {
    console.log(`Query: "${queryText}"`);

    const query: QueryRequest = {
      query: queryText,
      path: indexPath,
      project: "rag-example",
      limit: 3,
      minScore: 0.5,
      hybrid: true,
    };

    const response = await client.queryCodebase(query);
    console.log(
      `  Found ${response.results.length} results in ${response.durationMs} ms ` +
        `(threshold: ${
          response.thresholdUsed.toFixed(2)
        }, lowered: ${response.thresholdLowered})`,
    );

    for (let i = 0; i < response.results.length; i++) {
      const result = response.results[i];
      let preview = result.content.split("\n").slice(0, 2).join(" ");
      if (preview.length > 100) {
        preview = preview.slice(0, 100) + "...";
      }

      const kwLabel = result.keywordScore != null
        ? result.keywordScore.toFixed(3)
        : "n/a";

      console.log(
        `  [${i + 1}] score=${result.score.toFixed(3)} ` +
          `(vec=${result.vectorScore.toFixed(3)}, kw=${kwLabel}) | ` +
          `${result.filePath}:${result.startLine}-${result.endLine} | ${preview}`,
      );
    }
    console.log();
  }

  // 5. Vector-only search (disable hybrid)
  console.log("--- Step 5: Vector-Only Search ---\n");

  const vectorQuery: QueryRequest = {
    query: "how are thoughts stored and retrieved",
    path: indexPath,
    project: "rag-example",
    limit: 3,
    minScore: 0.5,
    hybrid: false,
  };

  const vectorResponse = await client.queryCodebase(vectorQuery);
  console.log(`Vector-only results (${vectorResponse.results.length} found):`);
  for (let i = 0; i < vectorResponse.results.length; i++) {
    const result = vectorResponse.results[i];
    console.log(
      `  [${i + 1}] score=${result.score.toFixed(3)} | ` +
        `${result.filePath} (lines ${result.startLine}-${result.endLine})`,
    );
  }

  console.log("\n=== Done ===");
}

await main();
