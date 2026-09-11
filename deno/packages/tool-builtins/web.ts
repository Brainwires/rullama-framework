/**
 * The `fetch_url` tool: fetches a model-chosen URL through `safeFetch` from
 * `@rullama/tool-runtime` (http/https only, private / loopback / link-local
 * addresses refused, every redirect re-checked, deadline enforced) and returns
 * the body capped by `readCappedText`.
 * Equivalent to Rust's `rullama_tool_builtins::web`.
 *
 * @module
 */

// deno-lint-ignore-file no-explicit-any

import { objectSchema, type ToolContext, ToolResult } from "@rullama/core";
import type { Tool } from "@rullama/core";
import { readCappedText, safeFetch } from "@rullama/tool-runtime";

/** Web fetching tool. */
export class WebTool {
  /** Return tool definitions for web operations. */
  static getTools(): Tool[] {
    return [WebTool.fetchUrlTool()];
  }

  /** Definition of the `fetch_url` tool. */
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

  /** Run `fetch_url` through `safeFetch` and cap the body. */
  private static fetchUrl(input: any): Promise<string> {
    return WebTool.fetchUrlContent(String(input.url));
  }

  /**
   * Fetch URL content (helper for orchestrator integration). Only public
   * `http(s)` destinations are reachable (every redirect hop is re-checked),
   * the request has a 30 s deadline and the body is capped at 1 MiB.
   */
  static async fetchUrlContent(url: string): Promise<string> {
    const response = await safeFetch(url);
    const text = await readCappedText(response);
    return `URL: ${url}\nStatus: ${response.status}\nContent length: ${text.length} bytes\n\n${text}`;
  }
}
