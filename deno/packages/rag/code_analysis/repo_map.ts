/**
 * RepoMap — regex-based symbol extraction from source files.
 *
 * Provides AST-aware code understanding without requiring tree-sitter.
 * Supports TypeScript, JavaScript, Python, and Rust via regex patterns.
 *
 * Ported from `rullama-knowledge/src/code_analysis/repomap/`.
 *
 * @module
 */

import type {
  Definition,
  ReferenceKind,
  SymbolId,
  SymbolKind,
} from "./types.ts";
import { visibilityFromKeywords } from "./types.ts";

// ── Language patterns ────────────────────────────────────────────────────────

interface SymbolPattern {
  /** Regex that captures the symbol name in group 1. */
  regex: RegExp;
  /** What kind of symbol this pattern matches. */
  kind: SymbolKind;
}

/** Patterns for TypeScript / JavaScript. */
const TS_JS_PATTERNS: SymbolPattern[] = [
  // export function name(
  {
    regex: /^[ \t]*(?:export\s+)?(?:async\s+)?function\s+(\w+)/gm,
    kind: "function",
  },
  // export class Name
  {
    regex: /^[ \t]*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)/gm,
    kind: "class",
  },
  // export interface Name
  { regex: /^[ \t]*(?:export\s+)?interface\s+(\w+)/gm, kind: "interface" },
  // export type Name =
  { regex: /^[ \t]*(?:export\s+)?type\s+(\w+)\s*[=<]/gm, kind: "type_alias" },
  // export enum Name
  { regex: /^[ \t]*(?:export\s+)?(?:const\s+)?enum\s+(\w+)/gm, kind: "enum" },
  // export const/let/var name  (module-level only — no indentation or exactly 0 spaces)
  { regex: /^(?:export\s+)?(?:const|let|var)\s+(\w+)/gm, kind: "variable" },
];

/** Patterns for Python. */
const PYTHON_PATTERNS: SymbolPattern[] = [
  // def name(
  { regex: /^[ \t]*(?:async\s+)?def\s+(\w+)/gm, kind: "function" },
  // class Name
  { regex: /^[ \t]*class\s+(\w+)/gm, kind: "class" },
  // import ... (captured for completeness)
  { regex: /^[ \t]*(?:from\s+\S+\s+)?import\s+(\w+)/gm, kind: "import" },
];

/** Patterns for Rust. */
const RUST_PATTERNS: SymbolPattern[] = [
  // pub fn name  or  fn name
  {
    regex:
      /^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?fn\s+(\w+)/gm,
    kind: "function",
  },
  // struct Name
  { regex: /^[ \t]*(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)/gm, kind: "struct" },
  // enum Name
  { regex: /^[ \t]*(?:pub(?:\([^)]*\))?\s+)?enum\s+(\w+)/gm, kind: "enum" },
  // trait Name
  { regex: /^[ \t]*(?:pub(?:\([^)]*\))?\s+)?trait\s+(\w+)/gm, kind: "trait" },
  // impl Name or impl Trait for Name
  {
    regex: /^[ \t]*impl(?:<[^>]*>)?\s+(?:\w+\s+for\s+)?(\w+)/gm,
    kind: "class",
  },
  // mod name
  { regex: /^[ \t]*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)/gm, kind: "module" },
  // const NAME or static NAME
  {
    regex: /^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+(\w+)/gm,
    kind: "constant",
  },
  // type Name =
  {
    regex: /^[ \t]*(?:pub(?:\([^)]*\))?\s+)?type\s+(\w+)/gm,
    kind: "type_alias",
  },
  // use ...
  { regex: /^[ \t]*(?:pub(?:\([^)]*\))?\s+)?use\s+(\S+)/gm, kind: "import" },
];

/** Map of file extension to pattern list. */
const LANGUAGE_PATTERNS: Record<string, SymbolPattern[]> = {
  ts: TS_JS_PATTERNS,
  tsx: TS_JS_PATTERNS,
  js: TS_JS_PATTERNS,
  jsx: TS_JS_PATTERNS,
  mjs: TS_JS_PATTERNS,
  cjs: TS_JS_PATTERNS,
  py: PYTHON_PATTERNS,
  rs: RUST_PATTERNS,
};

/** Map of file extension to language name. */
const EXTENSION_TO_LANGUAGE: Record<string, string> = {
  ts: "TypeScript",
  tsx: "TypeScript",
  js: "JavaScript",
  jsx: "JavaScript",
  mjs: "JavaScript",
  cjs: "JavaScript",
  py: "Python",
  rs: "Rust",
};

// ── RepoMap class ────────────────────────────────────────────────────────────

/** Options passed to extractSymbols. */
export interface ExtractOptions {
  /** Relative file path from the project root. */
  filePath: string;
  /** Source code contents. */
  content: string;
  /** Absolute root path (optional, stored in definitions). */
  rootPath?: string;
  /** Project name (optional). */
  project?: string;
}

/**
 * Regex-based symbol extractor.
 *
 * Extracts function, class, type, and variable definitions from source
 * code using language-specific regex patterns. This is the zero-dependency
 * fallback mode; a tree-sitter mode can be layered on top for higher
 * precision.
 */
export class RepoMap {
  /**
   * Return the list of supported file extensions.
   */
  static supportedExtensions(): string[] {
    return Object.keys(LANGUAGE_PATTERNS);
  }

  /**
   * Check whether a file extension is supported.
   */
  static supportsExtension(ext: string): boolean {
    return ext.toLowerCase().replace(/^\./, "") in LANGUAGE_PATTERNS;
  }

  /**
   * Detect the language name from a file extension.
   * Returns `undefined` for unsupported extensions.
   */
  static languageForExtension(ext: string): string | undefined {
    return EXTENSION_TO_LANGUAGE[ext.toLowerCase().replace(/^\./, "")];
  }

  /**
   * Extract all symbol definitions from a source file.
   */
  static extractSymbols(opts: ExtractOptions): Definition[] {
    const ext = extensionOf(opts.filePath);
    if (!ext) return [];

    const patterns = LANGUAGE_PATTERNS[ext];
    if (!patterns) return [];

    const definitions: Definition[] = [];
    const now = Date.now();

    for (const pattern of patterns) {
      // Reset the regex (global flag means state is stored)
      const re = new RegExp(pattern.regex.source, pattern.regex.flags);
      let match: RegExpExecArray | null;

      while ((match = re.exec(opts.content)) !== null) {
        const name = match[1];
        if (!name) continue;

        const startOffset = match.index;
        const { line, col } = offsetToLineCol(opts.content, startOffset);

        // Determine visibility from the matched text
        const matchedText = match[0];
        const visibility = visibilityFromKeywords(matchedText);

        // Extract signature: the matched line (capped at 200 chars)
        const lineEnd = opts.content.indexOf("\n", startOffset);
        const sigEnd = lineEnd === -1 ? opts.content.length : lineEnd;
        let signature = opts.content.slice(startOffset, sigEnd).trimEnd();
        if (signature.length > 200) {
          signature = signature.slice(0, 200) + "...";
        }

        // Guess endLine — look for balanced braces or next blank line
        const endLine = guessEndLine(opts.content, line, ext);

        // Extract preceding doc comment
        const docComment = extractDocComment(opts.content, line, ext);

        const symbolId: SymbolId = {
          filePath: opts.filePath,
          name,
          kind: pattern.kind,
          startLine: line,
          startCol: col,
        };

        definitions.push({
          symbolId,
          rootPath: opts.rootPath,
          project: opts.project,
          endLine,
          endCol: 0,
          signature,
          docComment,
          visibility,
          indexedAt: now,
        });
      }
    }

    // Sort by start line for deterministic output
    definitions.sort((a, b) => a.symbolId.startLine - b.symbolId.startLine);

    return definitions;
  }

  /**
   * Format a set of definitions into a repo-map style string.
   *
   * The output groups symbols by file and displays them as a tree:
   * ```
   * src/foo.ts
   *   function greet (line 3)
   *   class Person (line 10)
   *     method constructor (line 11)
   * ```
   */
  static formatRepoMap(definitions: Definition[]): string {
    // Group by file
    const byFile = new Map<string, Definition[]>();
    for (const def of definitions) {
      const key = def.symbolId.filePath;
      const list = byFile.get(key) ?? [];
      list.push(def);
      byFile.set(key, list);
    }

    const lines: string[] = [];

    for (
      const [filePath, defs] of [...byFile.entries()].sort((a, b) =>
        a[0].localeCompare(b[0])
      )
    ) {
      lines.push(filePath);
      const sorted = [...defs].sort((a, b) =>
        a.symbolId.startLine - b.symbolId.startLine
      );
      for (const def of sorted) {
        const indent = def.parentId ? "    " : "  ";
        const vis = def.visibility === "public" ? "pub " : "";
        lines.push(
          `${indent}${vis}${def.symbolId.kind} ${def.symbolId.name} (line ${def.symbolId.startLine})`,
        );
      }
    }

    return lines.join("\n");
  }
}

// ── Reference detection helpers ──────────────────────────────────────────────

/**
 * Determine the kind of reference from surrounding context.
 *
 * This is a heuristic; it checks the characters around the identifier
 * occurrence on its line.
 */
export function determineReferenceKind(
  line: string,
  position: number,
  name: string,
): ReferenceKind {
  const before = line.slice(0, position);
  const afterStart = position + name.length;
  const after = afterStart <= line.length ? line.slice(afterStart) : "";
  const lower = line.toLowerCase();

  // Import patterns
  if (
    lower.includes("import ") ||
    lower.includes("from ") ||
    lower.includes("require(") ||
    lower.includes("use ")
  ) {
    return "import";
  }

  // Instantiation
  if (before.includes("new ")) return "instantiation";

  // Inheritance
  if (before.includes("extends") || before.includes("implements")) {
    return "inheritance";
  }

  // Call
  if (after.trimStart().startsWith("(")) return "call";

  // Write
  const trimmedAfter = after.trimStart();
  if (
    trimmedAfter.startsWith("=") &&
    !trimmedAfter.startsWith("==") &&
    !trimmedAfter.startsWith("=>")
  ) {
    return "write";
  }

  // Type reference
  if (before.includes(":") || before.includes("->") || before.includes("<")) {
    return "type_reference";
  }

  return "read";
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/** Extract file extension (without leading dot) from a path. */
function extensionOf(filePath: string): string | undefined {
  const dotIdx = filePath.lastIndexOf(".");
  if (dotIdx === -1 || dotIdx === filePath.length - 1) return undefined;
  return filePath.slice(dotIdx + 1).toLowerCase();
}

/** Convert a byte offset in `source` to a 1-based line and 0-based column. */
function offsetToLineCol(
  source: string,
  offset: number,
): { line: number; col: number } {
  let line = 1;
  let lastNewline = -1;
  for (let i = 0; i < offset; i++) {
    if (source[i] === "\n") {
      line++;
      lastNewline = i;
    }
  }
  return { line, col: offset - lastNewline - 1 };
}

/**
 * Guess the ending line of a definition.
 *
 * Uses a simple brace-counting heuristic for brace-delimited languages
 * (TS/JS/Rust) and indentation for Python.
 */
function guessEndLine(source: string, startLine: number, ext: string): number {
  const lines = source.split("\n");
  const idx = startLine - 1; // Convert to 0-based

  if (idx >= lines.length) return startLine;

  if (ext === "py") {
    // Python: look for next line with same or less indentation (non-empty)
    const baseIndent = leadingSpaces(lines[idx]);
    for (let i = idx + 1; i < lines.length; i++) {
      const line = lines[i];
      if (line.trim() === "") continue;
      if (leadingSpaces(line) <= baseIndent) return i; // 1-based (i is 0-based, next line)
    }
    return lines.length;
  }

  // Brace-delimited languages
  let depth = 0;
  let foundOpen = false;
  for (let i = idx; i < lines.length; i++) {
    for (const ch of lines[i]) {
      if (ch === "{") {
        depth++;
        foundOpen = true;
      } else if (ch === "}") {
        depth--;
        if (foundOpen && depth === 0) return i + 1; // 1-based
      }
    }
  }

  // Fallback: single-line definition
  return startLine;
}

function leadingSpaces(line: string): number {
  const match = line.match(/^(\s*)/);
  return match ? match[1].replace(/\t/g, "    ").length : 0;
}

/**
 * Extract a doc comment preceding a definition line.
 *
 * Walks upward from the line before `defLine` collecting comment lines.
 */
function extractDocComment(
  source: string,
  defLine: number,
  ext: string,
): string | undefined {
  const lines = source.split("\n");
  const idx = defLine - 2; // line before the definition (0-based)
  if (idx < 0) return undefined;

  const commentLines: string[] = [];

  for (let i = idx; i >= 0; i--) {
    const trimmed = lines[i].trim();
    if (trimmed === "") break; // stop at blank lines

    let isComment = false;
    let cleaned = trimmed;

    if (ext === "py") {
      if (trimmed.startsWith("#")) {
        isComment = true;
        cleaned = trimmed.replace(/^#+\s?/, "");
      }
    } else if (ext === "rs") {
      if (trimmed.startsWith("///") || trimmed.startsWith("//!")) {
        isComment = true;
        cleaned = trimmed.replace(/^\/\/[/!]\s?/, "");
      } else if (trimmed.startsWith("//")) {
        isComment = true;
        cleaned = trimmed.replace(/^\/\/\s?/, "");
      }
    } else {
      // TS/JS
      if (trimmed.startsWith("//")) {
        isComment = true;
        cleaned = trimmed.replace(/^\/\/\s?/, "");
      } else if (
        trimmed.startsWith("*") || trimmed.startsWith("/**") ||
        trimmed.startsWith("*/")
      ) {
        isComment = true;
        cleaned = trimmed.replace(/^\/?\*+\/?\s?/, "").replace(
          /\*+\/?\s?$/,
          "",
        );
      }
    }

    if (!isComment) break;
    if (cleaned.trim()) commentLines.push(cleaned.trim());
  }

  if (commentLines.length === 0) return undefined;
  commentLines.reverse();
  return commentLines.join("\n");
}
