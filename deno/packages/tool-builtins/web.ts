/**
 * Web fetching tool implementation.
 * Uses the global fetch() API.
 */

// deno-lint-ignore-file no-explicit-any

import { objectSchema, type ToolContext, ToolResult } from "@rullama/core";
import type { Tool } from "@rullama/core";

/** Web fetching tool. */
export class WebTool {
  /** Return tool definitions for web operations. */
  static getTools(): Tool[] {
    return [WebTool.fetchUrlTool()];
  }

  private static fetchUrlTool(): Tool {
    return {
      name: "fetch_url",
      description: "Fetch content from a URL on the internet.",
      input_schema: objectSchema(
        {
          url: {
            type: "string",
            description: "URL to fetch",
          },
        },
        ["url"],
      ),
      requires_approval: false,
    };
  }

  /** Execute a web tool by name. */
  static async execute(
    toolUseId: string,
    toolName: string,
    input: any,
    _context: ToolContext,
  ): Promise<ToolResult> {
    if (toolName !== "fetch_url") {
      return ToolResult.error(
        toolUseId,
        `Unknown web tool: ${toolName}`,
      );
    }

    try {
      const output = await WebTool.fetchUrl(input);
      return ToolResult.success(toolUseId, output);
    } catch (e) {
      return ToolResult.error(
        toolUseId,
        `Web operation failed: ${(e as Error).message}`,
      );
    }
  }

  private static async fetchUrl(input: any): Promise<string> {
    const url: string = input.url;
    const response = await fetch(url);
    const text = await response.text();
    return `URL: ${url}\nContent length: ${text.length} bytes\n\n${text}`;
  }

  /** Fetch URL content (helper for orchestrator integration). */
  static async fetchUrlContent(url: string): Promise<string> {
    const response = await fetch(url);
    const text = await response.text();
    return `URL: ${url}\nContent length: ${text.length} bytes\n\n${text}`;
  }
}
