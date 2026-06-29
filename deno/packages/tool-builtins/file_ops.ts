/**
 * File operations tool implementation.
 * Uses Deno's native FS APIs.
 */

// deno-lint-ignore-file no-explicit-any

import { objectSchema, type ToolContext, ToolResult } from "@rullama/core";
import type { Tool } from "@rullama/core";
import { join, resolve } from "@std/path";

/** File operations tool. */
export class FileOpsTool {
  /** Get all file operation tool definitions. */
  static getTools(): Tool[] {
    return [
      FileOpsTool.readFileTool(),
      FileOpsTool.writeFileTool(),
      FileOpsTool.editFileTool(),
      FileOpsTool.listDirectoryTool(),
      FileOpsTool.searchFilesTool(),
      FileOpsTool.deleteFileTool(),
      FileOpsTool.createDirectoryTool(),
    ];
  }

  private static readFileTool(): Tool {
    return {
      name: "read_file",
      description: "Read the contents of a local file.",
      input_schema: objectSchema(
        {
          path: {
            type: "string",
            description: "Path to the file to read (relative or absolute)",
          },
        },
        ["path"],
      ),
      requires_approval: false,
    };
  }

  private static writeFileTool(): Tool {
    return {
      name: "write_file",
      description: "Create or overwrite a file with the given content.",
      input_schema: objectSchema(
        {
          path: {
            type: "string",
            description: "Path to the file to write",
          },
          content: {
            type: "string",
            description: "Content to write to the file",
          },
        },
        ["path", "content"],
      ),
      requires_approval: true,
    };
  }

  private static editFileTool(): Tool {
    return {
      name: "edit_file",
      description:
        "Replace the first occurrence of old_text with new_text in a file.",
      input_schema: objectSchema(
        {
          path: {
            type: "string",
            description: "Path to the file to edit",
          },
          old_text: {
            type: "string",
            description: "Exact text to find in the file",
          },
          new_text: {
            type: "string",
            description: "Text to replace old_text with",
          },
        },
        ["path", "old_text", "new_text"],
      ),
      requires_approval: true,
    };
  }

  private static listDirectoryTool(): Tool {
    return {
      name: "list_directory",
      description: "List files and directories in a local path.",
      input_schema: objectSchema(
        {
          path: {
            type: "string",
            description: "Path to the directory to list",
          },
          recursive: {
            type: "boolean",
            description: "Whether to list recursively",
            default: false,
          },
        },
        ["path"],
      ),
      requires_approval: false,
    };
  }

  private static searchFilesTool(): Tool {
    return {
      name: "search_files",
      description: "Search for files matching a glob pattern.",
      input_schema: objectSchema(
        {
          path: {
            type: "string",
            description: "Directory to search in",
          },
          pattern: {
            type: "string",
            description: "File name pattern to match (glob pattern)",
          },
        },
        ["path", "pattern"],
      ),
      requires_approval: false,
    };
  }

  private static deleteFileTool(): Tool {
    return {
      name: "delete_file",
      description: "Delete a file or directory.",
      input_schema: objectSchema(
        {
          path: {
            type: "string",
            description: "Path to the file or directory to delete",
          },
        },
        ["path"],
      ),
      requires_approval: true,
    };
  }

  private static createDirectoryTool(): Tool {
    return {
      name: "create_directory",
      description: "Create a new directory (including parent directories).",
      input_schema: objectSchema(
        {
          path: {
            type: "string",
            description: "Path to the directory to create",
          },
        },
        ["path"],
      ),
      requires_approval: true,
    };
  }

  /** Execute a file operation tool. */
  static async execute(
    toolUseId: string,
    toolName: string,
    input: any,
    context: ToolContext,
  ): Promise<ToolResult> {
    try {
      let output: string;
      switch (toolName) {
        case "read_file":
          output = await FileOpsTool.readFile(input, context);
          break;
        case "write_file":
          output = await FileOpsTool.writeFile(input, context);
          break;
        case "edit_file":
          output = await FileOpsTool.editFile(input, context);
          break;
        case "list_directory":
          output = await FileOpsTool.listDirectory(input, context);
          break;
        case "search_files":
          output = await FileOpsTool.searchFiles(input, context);
          break;
        case "delete_file":
          output = await FileOpsTool.deleteFile(input, context);
          break;
        case "create_directory":
          output = await FileOpsTool.createDirectory(input, context);
          break;
        default:
          return ToolResult.error(
            toolUseId,
            `Unknown file operation tool: ${toolName}`,
          );
      }
      return ToolResult.success(toolUseId, output);
    } catch (e) {
      return ToolResult.error(
        toolUseId,
        `File operation failed: ${(e as Error).message}`,
      );
    }
  }

  /** Resolve a path relative to the working directory. */
  static resolvePath(path: string, context: ToolContext): string {
    if (path.startsWith("/")) {
      return resolve(path);
    }
    return resolve(join(context.working_directory, path));
  }

  private static async readFile(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const fullPath = FileOpsTool.resolvePath(input.path, context);
    const content = await Deno.readTextFile(fullPath);
    return `File: ${fullPath}\nSize: ${content.length} bytes\n\n${content}`;
  }

  private static async writeFile(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const fullPath = FileOpsTool.resolvePath(input.path, context);
    const content: string = input.content;

    // Ensure parent directory exists
    const parent = fullPath.substring(0, fullPath.lastIndexOf("/"));
    if (parent) {
      await Deno.mkdir(parent, { recursive: true }).catch(() => {});
    }

    await Deno.writeTextFile(fullPath, content);
    return `Successfully wrote ${content.length} bytes to ${fullPath}`;
  }

  private static async editFile(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const fullPath = FileOpsTool.resolvePath(input.path, context);
    const oldText: string = input.old_text;
    const newText: string = input.new_text;

    const current = await Deno.readTextFile(fullPath);
    if (!current.includes(oldText)) {
      throw new Error(`Text not found in file: '${oldText}'`);
    }

    // Replace first occurrence only
    const idx = current.indexOf(oldText);
    const newContent = current.substring(0, idx) + newText +
      current.substring(idx + oldText.length);

    await Deno.writeTextFile(fullPath, newContent);
    return `Successfully replaced 1 occurrence(s) in ${fullPath}`;
  }

  private static async listDirectory(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const fullPath = FileOpsTool.resolvePath(input.path, context);
    const recursive = input.recursive ?? false;

    const entries: string[] = [];

    if (recursive) {
      for await (const entry of walkDir(fullPath)) {
        const relative = entry.path.startsWith(fullPath + "/")
          ? entry.path.substring(fullPath.length + 1)
          : entry.path;
        const typeStr = entry.isDirectory ? "dir" : "file";
        entries.push(`${typeStr} - ${relative}`);
      }
    } else {
      for await (const entry of Deno.readDir(fullPath)) {
        const typeStr = entry.isDirectory ? "dir" : "file";
        entries.push(`${typeStr} - ${entry.name}`);
      }
    }

    entries.sort();
    return `Directory: ${fullPath}\nEntries: ${entries.length}\n\n${
      entries.join("\n")
    }`;
  }

  private static async searchFiles(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const fullPath = FileOpsTool.resolvePath(input.path, context);
    const pattern: string = input.pattern;

    // Convert glob pattern to regex
    const regexStr = pattern
      .replace(/\./g, "\\.")
      .replace(/\*/g, ".*")
      .replace(/\?/g, ".");
    const regex = new RegExp(regexStr);

    const matches: string[] = [];
    for await (const entry of walkDir(fullPath)) {
      if (!entry.isDirectory) {
        const name = entry.path.substring(
          entry.path.lastIndexOf("/") + 1,
        );
        if (regex.test(name)) {
          const relative = entry.path.startsWith(fullPath + "/")
            ? entry.path.substring(fullPath.length + 1)
            : entry.path;
          matches.push(relative);
        }
      }
    }

    matches.sort();
    return `Search pattern: ${pattern}\nMatches: ${matches.length}\n\n${
      matches.join("\n")
    }`;
  }

  private static async deleteFile(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const fullPath = FileOpsTool.resolvePath(input.path, context);

    try {
      const stat = await Deno.stat(fullPath);
      if (stat.isDirectory) {
        await Deno.remove(fullPath, { recursive: true });
        return `Successfully deleted directory: ${fullPath}`;
      }
    } catch {
      // Not a directory, try as file
    }

    await Deno.remove(fullPath);
    return `Successfully deleted file: ${fullPath}`;
  }

  private static async createDirectory(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const fullPath = FileOpsTool.resolvePath(input.path, context);
    await Deno.mkdir(fullPath, { recursive: true });
    return `Successfully created directory: ${fullPath}`;
  }
}

/** Simple recursive directory walker. */
async function* walkDir(
  path: string,
): AsyncGenerator<{ path: string; isDirectory: boolean }> {
  for await (const entry of Deno.readDir(path)) {
    const fullPath = `${path}/${entry.name}`;
    if (entry.isDirectory) {
      yield { path: fullPath, isDirectory: true };
      yield* walkDir(fullPath);
    } else {
      yield { path: fullPath, isDirectory: false };
    }
  }
}
