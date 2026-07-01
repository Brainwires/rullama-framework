/**
 * Validation tool implementation.
 * Provides build/compile checks, syntax validation, and duplicate detection.
 */

// deno-lint-ignore-file no-explicit-any

import { objectSchema, type ToolContext, ToolResult } from "@rullama/core";
import type { Tool } from "@rullama/core";

const BUILD_TIMEOUT_MS = 600_000;
const SYNTAX_CHECK_TIMEOUT_MS = 30_000;

/** Supported build types for verify_build. */
const BUILD_TYPES = [
  "npm",
  "yarn",
  "pnpm",
  "bun",
  "cargo",
  "typescript",
  "go",
  "python",
  "gradle",
  "maven",
  "make",
] as const;

type BuildType = (typeof BUILD_TYPES)[number];

/** Validation tool for agents to verify their work. */
export class ValidationTool {
  /** Get all validation tool definitions. */
  static getTools(): Tool[] {
    return [
      ValidationTool.checkDuplicatesTool(),
      ValidationTool.verifyBuildTool(),
      ValidationTool.checkSyntaxTool(),
    ];
  }

  private static checkDuplicatesTool(): Tool {
    return {
      name: "check_duplicates",
      description:
        "Check a file for duplicate exports, constants, or function definitions.",
      input_schema: objectSchema(
        {
          file_path: {
            type: "string",
            description: "Path to file",
          },
        },
        ["file_path"],
      ),
      requires_approval: false,
    };
  }

  private static verifyBuildTool(): Tool {
    return {
      name: "verify_build",
      description:
        "Run a build command and verify it succeeds. Supports: npm, yarn, pnpm, bun, cargo, typescript, go, python, gradle, maven, make.",
      input_schema: objectSchema(
        {
          working_directory: {
            type: "string",
            description: "Directory to run the build in",
          },
          build_type: {
            type: "string",
            enum: [...BUILD_TYPES],
            description: "Build system to use",
          },
        },
        ["working_directory", "build_type"],
      ),
      requires_approval: false,
    };
  }

  private static checkSyntaxTool(): Tool {
    return {
      name: "check_syntax",
      description:
        "Check syntax of a single file without running a full build.",
      input_schema: objectSchema(
        {
          file_path: {
            type: "string",
            description: "Path to file",
          },
        },
        ["file_path"],
      ),
      requires_approval: false,
    };
  }

  /** Execute a validation tool by name. */
  static async execute(
    toolUseId: string,
    toolName: string,
    input: any,
    _context: ToolContext,
  ): Promise<ToolResult> {
    try {
      switch (toolName) {
        case "check_duplicates":
          return await ValidationTool.checkDuplicates(
            toolUseId,
            input.file_path ?? "",
          );
        case "verify_build":
          return await ValidationTool.verifyBuild(
            toolUseId,
            input.working_directory ?? ".",
            input.build_type ?? "cargo",
          );
        case "check_syntax":
          return await ValidationTool.checkSyntax(
            toolUseId,
            input.file_path ?? "",
          );
        default:
          return ToolResult.error(
            toolUseId,
            `Unknown validation tool: ${toolName}`,
          );
      }
    } catch (e) {
      return ToolResult.error(
        toolUseId,
        `Validation failed: ${(e as Error).message}`,
      );
    }
  }

  // ── check_duplicates ────────────────────────────────────────────────

  private static async checkDuplicates(
    toolUseId: string,
    filePath: string,
  ): Promise<ToolResult> {
    if (!filePath) {
      return ToolResult.error(toolUseId, "File path cannot be empty");
    }

    const content = await Deno.readTextFile(filePath);
    const lines = content.split("\n");

    const exports = new Map<string, number>();
    const duplicates: {
      name: string;
      first_line: number;
      duplicate_line: number;
      code: string;
    }[] = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (isExportLine(line)) {
        const name = extractExportName(line);
        if (name) {
          const firstLine = exports.get(name);
          if (firstLine !== undefined) {
            duplicates.push({
              name,
              first_line: firstLine,
              duplicate_line: i + 1,
              code: line.trim(),
            });
          } else {
            exports.set(name, i + 1);
          }
        }
      }
    }

    const result = {
      file: filePath,
      has_duplicates: duplicates.length > 0,
      duplicate_count: duplicates.length,
      duplicates,
      total_exports: exports.size,
    };

    return new ToolResult(
      toolUseId,
      JSON.stringify(result, null, 2),
      false,
    );
  }

  // ── verify_build ────────────────────────────────────────────────────

  private static async verifyBuild(
    toolUseId: string,
    workingDirectory: string,
    buildType: string,
  ): Promise<ToolResult> {
    const spec = getBuildCommand(buildType);
    if (!spec) {
      return ToolResult.error(toolUseId, `Unknown build type: ${buildType}`);
    }

    const [command, args] = spec;

    let output: Deno.CommandOutput;
    try {
      const cmd = new Deno.Command(command, {
        args,
        cwd: workingDirectory,
        stdout: "piped",
        stderr: "piped",
      });

      const child = cmd.spawn();
      const id = setTimeout(() => {
        try {
          child.kill();
        } catch { /* already exited */ }
      }, BUILD_TIMEOUT_MS);

      output = await child.output();
      clearTimeout(id);
    } catch (e) {
      const result = { success: false, error: `Failed to execute: ${e}` };
      return new ToolResult(toolUseId, JSON.stringify(result), true);
    }

    const stdout = new TextDecoder().decode(output.stdout);
    const stderr = new TextDecoder().decode(output.stderr);
    const success = output.success;
    const errors = parseBuildErrors(stderr, stdout, buildType);

    const result = {
      success,
      exit_code: output.code,
      error_count: errors.length,
      errors,
      working_directory: workingDirectory,
    };

    return new ToolResult(
      toolUseId,
      JSON.stringify(result, null, 2),
      !success,
    );
  }

  // ── check_syntax ────────────────────────────────────────────────────

  private static async checkSyntax(
    toolUseId: string,
    filePath: string,
  ): Promise<ToolResult> {
    if (!filePath) {
      return ToolResult.error(toolUseId, "File path cannot be empty");
    }

    const ext = filePath.split(".").pop() ?? "";

    // For TypeScript files, do a quick heuristic check
    if (ext === "ts" || ext === "tsx") {
      const content = await Deno.readTextFile(filePath);
      const errors: { message: string; type: string }[] = [];

      if (content.includes("export export")) {
        errors.push({
          message: "Duplicate 'export' keyword",
          type: "syntax_error",
        });
      }
      if (content.includes("import import")) {
        errors.push({
          message: "Duplicate 'import' keyword",
          type: "syntax_error",
        });
      }

      const open = (content.match(/{/g) ?? []).length;
      const close = (content.match(/}/g) ?? []).length;
      if (open !== close) {
        errors.push({
          message: `Unmatched braces: ${open} open, ${close} close`,
          type: "syntax_error",
        });
      }

      if (errors.length > 0) {
        const result = { file: filePath, valid_syntax: false, errors };
        return new ToolResult(toolUseId, JSON.stringify(result), true);
      }

      const result = { file: filePath, valid_syntax: true, skipped: true };
      return new ToolResult(
        toolUseId,
        JSON.stringify(result, null, 2),
        false,
      );
    }

    // For other file types, shell out to a syntax checker
    const spec = getSyntaxCommand(ext, filePath);
    if (!spec) {
      return ToolResult.error(
        toolUseId,
        `Unsupported file type: ${ext}`,
      );
    }

    const [command, args] = spec;
    let output: Deno.CommandOutput;
    try {
      const cmd = new Deno.Command(command, {
        args,
        stdout: "piped",
        stderr: "piped",
      });
      const child = cmd.spawn();
      const id = setTimeout(() => {
        try {
          child.kill();
        } catch { /* already exited */ }
      }, SYNTAX_CHECK_TIMEOUT_MS);
      output = await child.output();
      clearTimeout(id);
    } catch (e) {
      const result = {
        file: filePath,
        valid_syntax: false,
        error: `${e}`,
      };
      return new ToolResult(toolUseId, JSON.stringify(result), true);
    }

    const success = output.success;
    const result = { file: filePath, valid_syntax: success };
    return new ToolResult(
      toolUseId,
      JSON.stringify(result, null, 2),
      !success,
    );
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Returns true if the line is an export declaration. */
export function isExportLine(line: string): boolean {
  const t = line.trim();
  return (
    t.startsWith("export const ") ||
    t.startsWith("export let ") ||
    t.startsWith("export var ") ||
    t.startsWith("export function ") ||
    t.startsWith("export async function ") ||
    t.startsWith("export class ") ||
    t.startsWith("export interface ") ||
    t.startsWith("export type ") ||
    t.startsWith("export enum ") ||
    t.startsWith("export namespace ") ||
    t.startsWith("export default class ") ||
    t.startsWith("export default function ") ||
    t.startsWith("export default async function ")
  );
}

/** Extract the exported name from a line. */
export function extractExportName(line: string): string | null {
  const t = line.trim();

  for (const prefix of ["export const ", "export let ", "export var "]) {
    if (t.startsWith(prefix)) {
      const after = t.slice(prefix.length);
      const token = after.split(/\s/)[0];
      return token ? token.replace(/[^a-zA-Z0-9_$]/g, "") : null;
    }
  }

  if (t.startsWith("export async function ")) {
    const after = t.slice("export async function ".length);
    return after.split("(")[0]?.trim() || null;
  }
  if (t.startsWith("export function ")) {
    const after = t.slice("export function ".length);
    return after.split("(")[0]?.trim() || null;
  }
  if (t.startsWith("export default async function ")) {
    const after = t.slice("export default async function ".length);
    const name = after.split("(")[0]?.trim();
    return name || "default";
  }
  if (t.startsWith("export default function ")) {
    const after = t.slice("export default function ".length);
    const name = after.split("(")[0]?.trim();
    return name || "default";
  }
  if (t.startsWith("export default class ")) {
    const after = t.slice("export default class ".length);
    const name = after.split(/\s/)[0];
    return (!name || name === "{") ? "default" : name;
  }
  if (t.startsWith("export class ")) {
    const after = t.slice("export class ".length);
    return after.split(/\s/)[0] || null;
  }
  if (t.startsWith("export interface ")) {
    const after = t.slice("export interface ".length);
    return after.split(/\s/)[0] || null;
  }
  if (t.startsWith("export type ")) {
    const after = t.slice("export type ".length);
    return after.split(/[\s=<]/)[0]?.trim() || null;
  }
  if (t.startsWith("export enum ")) {
    const after = t.slice("export enum ".length);
    return after.split(/\s/)[0] || null;
  }
  if (t.startsWith("export namespace ")) {
    const after = t.slice("export namespace ".length);
    return after.split(/\s/)[0] || null;
  }

  return null;
}

function getBuildCommand(
  buildType: string,
): [string, string[]] | null {
  switch (buildType) {
    case "npm":
      return ["npm", ["run", "build"]];
    case "yarn":
      return ["yarn", ["build"]];
    case "pnpm":
      return ["pnpm", ["build"]];
    case "bun":
      return ["bun", ["run", "build"]];
    case "cargo":
      return ["cargo", ["build"]];
    case "typescript":
      return ["npx", ["tsc", "--noEmit"]];
    case "go":
      return ["go", ["build", "./..."]];
    case "python":
      return ["python", ["-m", "py_compile"]];
    case "gradle":
      return ["gradle", ["build"]];
    case "maven":
      return ["mvn", ["compile"]];
    case "make":
      return ["make", []];
    default:
      return null;
  }
}

function getSyntaxCommand(
  ext: string,
  filePath: string,
): [string, string[]] | null {
  switch (ext) {
    case "js":
    case "jsx":
      return [
        "npx",
        [
          "eslint",
          "--no-eslintrc",
          "--parser",
          "@babel/eslint-parser",
          filePath,
        ],
      ];
    case "rs":
      return [
        "rustc",
        ["--crate-type", "lib", "--error-format", "json", filePath],
      ];
    case "py":
      return ["python", ["-m", "py_compile", filePath]];
    case "go":
      return ["gofmt", ["-e", filePath]];
    default:
      return null;
  }
}

interface BuildError {
  message: string;
  type: string;
  [key: string]: unknown;
}

function parseBuildErrors(
  stderr: string,
  stdout: string,
  buildType: string,
): BuildError[] {
  const errors: BuildError[] = [];
  const seen = new Set<string>();
  const combined = `${stderr}\n${stdout}`;

  for (const line of combined.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const lower = trimmed.toLowerCase();
    if (
      lower.startsWith("warning:") ||
      lower.startsWith("note:") ||
      lower.startsWith("help:") ||
      lower.startsWith("-->")
    ) {
      continue;
    }

    let error: BuildError | null = null;
    switch (buildType) {
      case "typescript":
      case "npm":
      case "yarn":
      case "pnpm":
      case "bun": {
        const parts = line.split(" - error ");
        if (parts.length === 2) {
          error = {
            location: parts[0].trim(),
            message: parts[1].trim(),
            type: "typescript",
          };
        }
        break;
      }
      case "cargo":
        if (line.includes("error[E") || trimmed.startsWith("error:")) {
          error = { message: trimmed, type: "rust", severity: "error" };
        }
        break;
      case "go":
        if (trimmed.includes(".go:") && trimmed.includes(": ")) {
          const goParts = trimmed.split(": ");
          if (goParts.length >= 2) {
            error = {
              location: goParts[0].trim(),
              message: goParts.slice(1).join(": ").trim(),
              type: "go",
            };
          }
        }
        break;
      case "python":
        if (
          trimmed.startsWith('File "') ||
          (trimmed.includes("Error") &&
            (trimmed.includes("SyntaxError") ||
              trimmed.includes("IndentationError")))
        ) {
          error = { message: trimmed, type: "python", severity: "error" };
        }
        break;
      case "gradle":
      case "maven":
        if (trimmed.includes(".java:") && trimmed.includes("error:")) {
          const javaParts = trimmed.split("error:");
          if (javaParts.length >= 2) {
            error = {
              location: javaParts[0].trim(),
              message: javaParts[1].trim(),
              type: "java",
            };
          }
        } else if (trimmed.startsWith("[ERROR]")) {
          error = {
            message: trimmed.replace(/^\[ERROR\]\s*/, ""),
            type: "java",
          };
        }
        break;
    }

    if (error) {
      const key = error.message;
      if (!seen.has(key)) {
        seen.add(key);
        error.build_type = buildType;
        errors.push(error);
      }
      continue;
    }

    if (lower.includes("error") && !lower.includes("0 error")) {
      if (!seen.has(trimmed)) {
        seen.add(trimmed);
        errors.push({
          message: trimmed,
          type: "generic",
          build_type: buildType,
        });
      }
    }
  }

  return errors.slice(0, 25);
}
