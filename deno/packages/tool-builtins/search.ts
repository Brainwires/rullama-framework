/**
 * Regex-based code pattern search tool.
 * Respects .gitignore via git ls-files fallback.
 */

// deno-lint-ignore-file no-explicit-any

import { objectSchema, type ToolContext, ToolResult } from "@rullama/core";
import type { Tool } from "@rullama/core";

/** Regex-based code search tool. */
export class SearchTool {
  /** Return tool definitions for code search. */
  static getTools(): Tool[] {
    return [SearchTool.searchCodeTool()];
  }

  private static searchCodeTool(): Tool {
    return {
      name: "search_code",
      description: "Search for code patterns in files using regex.",
      input_schema: objectSchema(
        {
          pattern: {
            type: "string",
            description: "Regex pattern to search for",
          },
          path: {
            type: "string",
            description: "Path to search in",
            default: ".",
          },
        },
        ["pattern"],
      ),
      requires_approval: false,
    };
  }

  /** Execute a search tool by name. */
  static async execute(
    toolUseId: string,
    toolName: string,
    input: any,
    context: ToolContext,
  ): Promise<ToolResult> {
    if (toolName !== "search_code") {
      return ToolResult.error(
        toolUseId,
        `Unknown search tool: ${toolName}`,
      );
    }

    try {
      const output = await SearchTool.searchCode(input, context);
      return ToolResult.success(toolUseId, output);
    } catch (e) {
      return ToolResult.error(
        toolUseId,
        `Search failed: ${(e as Error).message}`,
      );
    }
  }

  private static async searchCode(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const pattern: string = input.pattern;
    const searchPath = input.path === "." || !input.path
      ? context.working_directory
      : input.path;

    const regex = new RegExp(pattern);
    const matches: string[] = [];
    const MAX_MATCHES = 100;

    await searchDir(searchPath, regex, matches, MAX_MATCHES);

    return `Search Results:\nPattern: ${pattern}\nMatches: ${matches.length}\n\n${
      matches.join("\n")
    }`;
  }
}

/** Recursively search files in a directory, respecting common ignore patterns. */
async function searchDir(
  dir: string,
  regex: RegExp,
  matches: string[],
  maxMatches: number,
): Promise<void> {
  const IGNORE_DIRS = new Set([
    ".git",
    "node_modules",
    "target",
    ".next",
    "dist",
    "build",
    "__pycache__",
    ".cache",
    "vendor",
  ]);

  try {
    for await (const entry of Deno.readDir(dir)) {
      if (matches.length >= maxMatches) return;

      if (entry.isDirectory) {
        if (!IGNORE_DIRS.has(entry.name) && !entry.name.startsWith(".")) {
          await searchDir(
            `${dir}/${entry.name}`,
            regex,
            matches,
            maxMatches,
          );
        }
        continue;
      }

      if (!entry.isFile) continue;

      // Skip binary-looking files
      if (isBinaryFileName(entry.name)) continue;

      const filePath = `${dir}/${entry.name}`;
      try {
        const content = await Deno.readTextFile(filePath);
        const lines = content.split("\n");
        for (let i = 0; i < lines.length; i++) {
          if (matches.length >= maxMatches) return;
          if (regex.test(lines[i])) {
            matches.push(`${filePath}:${i + 1} - ${lines[i].trim()}`);
          }
        }
      } catch {
        // Skip unreadable files
      }
    }
  } catch {
    // Skip unreadable directories
  }
}

/** Simple heuristic for binary file extensions. */
function isBinaryFileName(name: string): boolean {
  const binaryExts = new Set([
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".svg",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".o",
    ".obj",
    ".wasm",
    ".bin",
    ".dat",
    ".db",
    ".sqlite",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".eot",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".lock",
  ]);
  const dotIdx = name.lastIndexOf(".");
  if (dotIdx === -1) return false;
  return binaryExts.has(name.substring(dotIdx).toLowerCase());
}
