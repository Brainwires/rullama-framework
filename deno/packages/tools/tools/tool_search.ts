/**
 * Tool Search — meta-tool for discovering available tools dynamically.
 *
 * Equivalent to Rust's `brainwires_tools::tool_search` module. The semantic
 * search mode is deferred until an embedding backend lands in the Deno
 * `@brainwires/knowledge` package; requesting it returns a clear error.
 */

import {
  objectSchema,
  type Tool,
  type ToolContext,
  ToolResult,
} from "@brainwires/core";

import type { ToolRegistry } from "../registry.ts";
import type { ToolEmbeddingIndex } from "./tool_embedding.ts";

/** Search mode for tool discovery. */
export type SearchMode = "keyword" | "regex" | "semantic";

/** Default search mode. */
export const DEFAULT_SEARCH_MODE: SearchMode = "keyword";

const MAX_REGEX_LENGTH = 200;

/** Meta-tool for discovering available tools dynamically. */
export class ToolSearchTool {
  /** Return tool definitions for tool search. */
  static getTools(): Tool[] {
    return [
      {
        name: "search_tools",
        description:
          "Search for available tools by name or description.",
        input_schema: objectSchema({
          query: {
            type: "string",
            description: "Search query to find relevant tools",
          },
          mode: {
            type: "string",
            enum: ["keyword", "regex", "semantic"],
            description:
              "Search mode: keyword (substring match), regex (pattern match), or semantic (embedding similarity, requires rag feature)",
            default: "keyword",
          },
          include_deferred: {
            type: "boolean",
            description: "Include deferred tools",
            default: true,
          },
          limit: {
            type: "integer",
            description:
              "Maximum number of results to return (semantic mode only)",
            default: 10,
          },
          min_score: {
            type: "number",
            description:
              "Minimum similarity score 0.0-1.0 (semantic mode only)",
            default: 0.3,
          },
        }, ["query"]),
        requires_approval: false,
      },
    ];
  }

  /** Execute the tool search tool by name. */
  static async execute(
    tool_use_id: string,
    tool_name: string,
    input: Record<string, unknown>,
    _context: ToolContext,
    registry: ToolRegistry,
    embeddingIndex?: ToolEmbeddingIndex,
  ): Promise<ToolResult> {
    if (tool_name !== "search_tools") {
      return ToolResult.error(
        tool_use_id,
        `Tool search failed: Unknown tool search tool: ${tool_name}`,
      );
    }
    try {
      const body = await ToolSearchTool.searchTools(
        input,
        registry,
        embeddingIndex,
      );
      return ToolResult.success(tool_use_id, body);
    } catch (e) {
      return ToolResult.error(
        tool_use_id,
        `Tool search failed: ${(e as Error).message}`,
      );
    }
  }

  private static async searchTools(
    input: Record<string, unknown>,
    registry: ToolRegistry,
    embeddingIndex: ToolEmbeddingIndex | undefined,
  ): Promise<string> {
    const query = String(input.query ?? "");
    const mode = (input.mode as SearchMode | undefined) ?? "keyword";
    const include_deferred = (input.include_deferred as boolean | undefined) ??
      true;
    const limit = (input.limit as number | undefined) ?? 10;
    const min_score = (input.min_score as number | undefined) ?? 0.3;

    if (mode === "semantic") {
      if (!embeddingIndex) {
        throw new Error(
          "Semantic search mode requires a ToolEmbeddingIndex. Pass one to ToolSearchTool.execute or use 'keyword'/'regex' mode instead.",
        );
      }
      return ToolSearchTool.searchSemantic(
        query,
        registry,
        include_deferred,
        limit,
        min_score,
        embeddingIndex,
      );
    }

    if (mode === "regex" && query.length > MAX_REGEX_LENGTH) {
      throw new Error(
        `Regex pattern exceeds maximum length of ${MAX_REGEX_LENGTH} characters (got ${query.length})`,
      );
    }

    let regex: RegExp | null = null;
    if (mode === "regex") {
      try {
        regex = new RegExp(query);
      } catch (e) {
        throw new Error(
          `Invalid regex pattern '${query}': ${(e as Error).message}`,
        );
      }
    }

    const queryLower = query.toLowerCase();
    const queryTerms = queryLower.split(/\s+/).filter((t) => t.length > 0);

    const matching = registry.getAll().filter((tool) => {
      if (tool.defer_loading && !include_deferred) return false;
      const searchText = `${tool.name} ${tool.description}`;
      if (regex) {
        return regex.test(searchText);
      }
      const nameLower = tool.name.toLowerCase();
      const descLower = tool.description.toLowerCase();
      return queryTerms.some(
        (term) => nameLower.includes(term) || descLower.includes(term),
      );
    });

    if (matching.length === 0) {
      return `No tools found matching query: "${query}"`;
    }

    let result = `Found ${matching.length} tools matching "${query}":\n\n`;
    for (const tool of matching) {
      result += ToolSearchTool.formatTool(tool, null);
    }
    return result;
  }

  private static async searchSemantic(
    query: string,
    registry: ToolRegistry,
    include_deferred: boolean,
    limit: number,
    min_score: number,
    index: ToolEmbeddingIndex,
  ): Promise<string> {
    const tools = registry
      .getAll()
      .filter((t) => include_deferred || !t.defer_loading);

    const results = await index.search(query, limit, min_score);

    if (results.length === 0) {
      return `No tools found semantically matching query: "${query}" (min_score: ${
        min_score.toFixed(2)
      })`;
    }

    let out = `Found ${results.length} tools semantically matching "${query}":\n\n`;
    for (const [name, score] of results) {
      const tool = tools.find((t) => t.name === name);
      if (tool) out += ToolSearchTool.formatTool(tool, score);
    }
    return out;
  }

  /** Format a single tool entry for output. */
  static formatTool(tool: Tool, score: number | null): string {
    let out = `## ${tool.name}\n`;
    if (score !== null) {
      out += `**Similarity:** ${score.toFixed(2)}\n`;
    }
    out += `**Description:** ${tool.description}\n`;
    if (tool.input_schema.properties) {
      out += "**Parameters:**\n";
      for (const [name, schema] of Object.entries(tool.input_schema.properties)) {
        const s = schema as Record<string, unknown>;
        const desc = typeof s.description === "string"
          ? s.description
          : "No description";
        const ptype = typeof s.type === "string" ? s.type : "unknown";
        out += `  - \`${name}\` (${ptype}): ${desc}\n`;
      }
    }
    out += "\n";
    return out;
  }
}
