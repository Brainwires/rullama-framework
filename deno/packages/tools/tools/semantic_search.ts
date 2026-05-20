/**
 * Semantic Search Tool — RAG-powered codebase search.
 *
 * Provides semantic code search using vector embeddings via the
 * `@brainwires/knowledge` RAG interfaces. Supports indexing, querying,
 * filtered search, statistics, and git history search.
 *
 * Equivalent to Rust's `brainwires_tools::semantic_search` module. The Rust
 * version carries a global `OnceCell<RagClient>`; the Deno port accepts an
 * injected {@link RagClient} on each call so the host picks the transport
 * (in-process stub, HTTP RPC to the Rust service, etc.).
 */

import {
  objectSchema,
  type Tool,
  type ToolContext,
  ToolResult,
} from "@brainwires/core";
import type {
  AdvancedSearchRequest,
  IndexRequest,
  QueryRequest,
  RagClient,
  SearchGitHistoryRequest,
} from "@brainwires/knowledge";

/** Tool definitions and executor for semantic codebase search powered by RAG. */
export class SemanticSearchTool {
  /** Get all semantic search tool definitions. */
  static getTools(): Tool[] {
    return [
      {
        name: "index_codebase",
        description:
          "Index a codebase directory for semantic search using embeddings. Automatically performs full or incremental indexing. After indexing, use query_codebase to search.",
        input_schema: objectSchema({
          path: {
            type: "string",
            description: "Path to the codebase directory to index",
          },
          project: {
            type: "string",
            description:
              "Optional project name for multi-project support",
          },
          include_patterns: {
            type: "array",
            items: { type: "string" },
            description: "Optional glob patterns to include",
            default: [],
          },
          exclude_patterns: {
            type: "array",
            items: { type: "string" },
            description: "Optional glob patterns to exclude",
            default: [],
          },
          max_file_size: {
            type: "integer",
            description:
              "Maximum file size in bytes to index (default: 1MB)",
            default: 1_048_576,
          },
        }, ["path"]),
        requires_approval: false,
        defer_loading: true,
      },
      {
        name: "query_codebase",
        description:
          "Search the indexed codebase using semantic search. Returns relevant code chunks ranked by similarity.",
        input_schema: objectSchema({
          query: { type: "string", description: "The search query" },
          project: {
            type: "string",
            description: "Optional project name to filter by",
          },
          limit: {
            type: "integer",
            description: "Number of results (default: 10)",
            default: 10,
          },
          min_score: {
            type: "number",
            description:
              "Minimum similarity score 0-1 (default: 0.7)",
            default: 0.7,
          },
          hybrid: {
            type: "boolean",
            description:
              "Enable hybrid search (vector + keyword) (default: true)",
            default: true,
          },
        }, ["query"]),
        requires_approval: false,
        defer_loading: true,
      },
      {
        name: "search_with_filters",
        description:
          "Advanced semantic search with filters for file type, language, and path patterns.",
        input_schema: objectSchema({
          query: { type: "string", description: "The search query" },
          project: {
            type: "string",
            description: "Optional project name",
          },
          limit: {
            type: "integer",
            description: "Number of results (default: 10)",
            default: 10,
          },
          min_score: {
            type: "number",
            description: "Minimum score (default: 0.7)",
            default: 0.7,
          },
          file_extensions: {
            type: "array",
            items: { type: "string" },
            description: "Filter by extensions",
            default: [],
          },
          languages: {
            type: "array",
            items: { type: "string" },
            description: "Filter by languages",
            default: [],
          },
          path_patterns: {
            type: "array",
            items: { type: "string" },
            description: "Filter by path patterns",
            default: [],
          },
        }, ["query"]),
        requires_approval: false,
        defer_loading: true,
      },
      {
        name: "get_rag_statistics",
        description:
          "Get statistics about the indexed codebase (file counts, chunk counts, languages).",
        input_schema: objectSchema({
          project: {
            type: "string",
            description: "Optional project name",
          },
        }, []),
        requires_approval: false,
        defer_loading: true,
      },
      {
        name: "clear_rag_index",
        description:
          "Clear all indexed data from the vector database. Use before reindexing from scratch.",
        input_schema: objectSchema({}, []),
        requires_approval: true,
        defer_loading: true,
      },
      {
        name: "search_git_history",
        description:
          "Search git commit history using semantic search with on-demand indexing.",
        input_schema: objectSchema({
          query: { type: "string", description: "The search query" },
          path: {
            type: "string",
            description: "Path to the git repository (default: .)",
            default: ".",
          },
          project: {
            type: "string",
            description: "Optional project name",
          },
          branch: {
            type: "string",
            description: "Optional branch name",
          },
          max_commits: {
            type: "integer",
            description: "Max commits to index (default: 10)",
            default: 10,
          },
          limit: {
            type: "integer",
            description: "Number of results (default: 10)",
            default: 10,
          },
          min_score: {
            type: "number",
            description: "Minimum score (default: 0.7)",
            default: 0.7,
          },
          author: {
            type: "string",
            description: "Filter by author (regex)",
          },
          since: {
            type: "string",
            description: "Filter since date (ISO 8601)",
          },
          until: {
            type: "string",
            description: "Filter until date (ISO 8601)",
          },
          file_pattern: {
            type: "string",
            description: "Filter by file path pattern (regex)",
          },
        }, ["query"]),
        requires_approval: false,
        defer_loading: true,
      },
    ];
  }

  /** Execute a semantic search tool. */
  static async execute(
    tool_use_id: string,
    tool_name: string,
    input: Record<string, unknown>,
    _context: ToolContext,
    client: RagClient,
  ): Promise<ToolResult> {
    try {
      const output = await SemanticSearchTool.dispatch(
        tool_name,
        input,
        client,
      );
      return ToolResult.success(tool_use_id, output);
    } catch (e) {
      return ToolResult.error(
        tool_use_id,
        `Semantic search operation failed: ${(e as Error).message}`,
      );
    }
  }

  private static dispatch(
    tool_name: string,
    input: Record<string, unknown>,
    client: RagClient,
  ): Promise<string> {
    switch (tool_name) {
      case "index_codebase":
        return SemanticSearchTool.indexCodebase(input, client);
      case "query_codebase":
        return SemanticSearchTool.queryCodebase(input, client);
      case "search_with_filters":
        return SemanticSearchTool.searchWithFilters(input, client);
      case "get_rag_statistics":
        return SemanticSearchTool.getStatistics(client);
      case "clear_rag_index":
        return SemanticSearchTool.clearIndex(client);
      case "search_git_history":
        return SemanticSearchTool.searchGitHistory(input, client);
      default:
        return Promise.reject(new Error(`Unknown tool: ${tool_name}`));
    }
  }

  static async indexCodebase(
    input: Record<string, unknown>,
    client: RagClient,
  ): Promise<string> {
    const path = input.path as string | undefined;
    if (!path) throw new Error("Missing 'path' parameter");

    const req: IndexRequest = {
      path,
      project: input.project as string | undefined,
      includePatterns: Array.isArray(input.include_patterns)
        ? (input.include_patterns as string[])
        : undefined,
      excludePatterns: Array.isArray(input.exclude_patterns)
        ? (input.exclude_patterns as string[])
        : undefined,
      maxFileSize: (input.max_file_size as number | undefined) ?? 1_048_576,
    };

    const resp = await client.indexCodebase(req);
    return `Indexed ${resp.filesIndexed} files, ${resp.chunksCreated} chunks in ${resp.durationMs}ms (mode: ${resp.mode})`;
  }

  static async queryCodebase(
    input: Record<string, unknown>,
    client: RagClient,
  ): Promise<string> {
    const query = input.query as string | undefined;
    if (!query) throw new Error("Missing 'query' parameter");

    const req: QueryRequest = {
      query,
      path: input.path as string | undefined,
      project: input.project as string | undefined,
      limit: (input.limit as number | undefined) ?? 10,
      minScore: (input.min_score as number | undefined) ?? 0.7,
      hybrid: (input.hybrid as boolean | undefined) ?? true,
    };

    const resp = await client.queryCodebase(req);
    let out = `Found ${resp.results.length} results:\n\n`;
    resp.results.forEach((r, i) => {
      out += `${i + 1}. ${r.filePath} (score: ${r.score.toFixed(3)})\n`;
      out += `   Lines ${r.startLine}-${r.endLine}\n`;
      const firstLine = r.content.split("\n")[0] ?? "";
      out += `   ${firstLine}\n\n`;
    });
    return out;
  }

  static async searchWithFilters(
    input: Record<string, unknown>,
    client: RagClient,
  ): Promise<string> {
    const query = input.query as string | undefined;
    if (!query) throw new Error("Missing 'query' parameter");

    const req: AdvancedSearchRequest = {
      query,
      path: input.path as string | undefined,
      project: input.project as string | undefined,
      limit: (input.limit as number | undefined) ?? 10,
      minScore: (input.min_score as number | undefined) ?? 0.7,
      fileExtensions: Array.isArray(input.file_extensions)
        ? (input.file_extensions as string[])
        : undefined,
      languages: Array.isArray(input.languages)
        ? (input.languages as string[])
        : undefined,
      pathPatterns: Array.isArray(input.path_patterns)
        ? (input.path_patterns as string[])
        : undefined,
    };

    const resp = await client.advancedSearch(req);
    let out = `Found ${resp.results.length} filtered results:\n\n`;
    resp.results.forEach((r, i) => {
      out += `${i + 1}. ${r.filePath} (score: ${r.score.toFixed(3)})\n`;
      out += `   Language: ${r.language}\n`;
      out += `   Lines ${r.startLine}-${r.endLine}\n\n`;
    });
    return out;
  }

  static async getStatistics(client: RagClient): Promise<string> {
    const resp = await client.getStatistics();
    let out = "RAG Index Statistics:\n";
    out += `  Total chunks: ${resp.totalChunks}\n`;
    out += `  Total files: ${resp.totalFiles}\n\n`;
    if (resp.languageBreakdown.length > 0) {
      out += "Languages:\n";
      for (const s of resp.languageBreakdown) {
        out += `  ${s.language}: ${s.fileCount} files, ${s.chunkCount} chunks\n`;
      }
    }
    return out;
  }

  static async clearIndex(client: RagClient): Promise<string> {
    const resp = await client.clearIndex();
    return `Cleared index: ${resp.message}`;
  }

  static async searchGitHistory(
    input: Record<string, unknown>,
    client: RagClient,
  ): Promise<string> {
    const query = input.query as string | undefined;
    if (!query) throw new Error("Missing 'query' parameter");

    const req: SearchGitHistoryRequest = {
      query,
      path: (input.path as string | undefined) ?? ".",
      project: input.project as string | undefined,
      branch: input.branch as string | undefined,
      maxCommits: (input.max_commits as number | undefined) ?? 10,
      limit: (input.limit as number | undefined) ?? 10,
      minScore: (input.min_score as number | undefined) ?? 0.7,
      author: input.author as string | undefined,
      since: input.since as string | undefined,
      until: input.until as string | undefined,
      filePattern: input.file_pattern as string | undefined,
    };

    const resp = await client.searchGitHistory(req);
    let out = `Found ${resp.results.length} commits:\n\n`;
    resp.results.forEach((r, i) => {
      out += `${i + 1}. ${r.commitHash.slice(0, 8)} (score: ${
        r.score.toFixed(3)
      })\n`;
      out += `   Author: ${r.author}\n`;
      out += `   Date: ${r.commitDate}\n`;
      out += `   Message: ${r.commitMessage}\n\n`;
    });
    return out;
  }
}
