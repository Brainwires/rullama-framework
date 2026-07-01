/**
 * Type definitions for code analysis — symbol extraction, references, and call graphs.
 *
 * Ported from `rullama-knowledge/src/code_analysis/types.rs`.
 *
 * @module
 */

// ── Symbol kinds ─────────────────────────────────────────────────────────────

/** Kind of symbol in the codebase. */
export type SymbolKind =
  | "function"
  | "method"
  | "class"
  | "struct"
  | "interface"
  | "trait"
  | "enum"
  | "module"
  | "variable"
  | "constant"
  | "parameter"
  | "field"
  | "import"
  | "export"
  | "enum_variant"
  | "type_alias"
  | "unknown";

/** Human-readable display names for each symbol kind. */
export function symbolKindDisplayName(kind: SymbolKind): string {
  const map: Record<SymbolKind, string> = {
    function: "function",
    method: "method",
    class: "class",
    struct: "struct",
    interface: "interface",
    trait: "trait",
    enum: "enum",
    module: "module",
    variable: "variable",
    constant: "constant",
    parameter: "parameter",
    field: "field",
    import: "import",
    export: "export",
    enum_variant: "enum variant",
    type_alias: "type alias",
    unknown: "unknown",
  };
  return map[kind];
}

// ── Visibility ───────────────────────────────────────────────────────────────

/** Visibility / access modifier for a symbol. */
export type Visibility = "public" | "private" | "protected" | "internal";

/** Infer visibility from source-code keywords. */
export function visibilityFromKeywords(text: string): Visibility {
  const lower = text.toLowerCase();
  if (
    lower.includes("pub ") || lower.includes("public ") ||
    lower.includes("export ")
  ) {
    return "public";
  }
  if (lower.includes("protected ")) return "protected";
  if (lower.includes("internal ") || lower.includes("package ")) {
    return "internal";
  }
  return "private";
}

// ── Reference kinds ──────────────────────────────────────────────────────────

/** Kind of reference to a symbol. */
export type ReferenceKind =
  | "call"
  | "read"
  | "write"
  | "import"
  | "type_reference"
  | "inheritance"
  | "instantiation"
  | "unknown";

// ── Core data structures ─────────────────────────────────────────────────────

/** Unique identifier for a symbol in the codebase. */
export interface SymbolId {
  /** Relative file path from the project root. */
  filePath: string;
  /** Symbol name. */
  name: string;
  /** Kind of symbol. */
  kind: SymbolKind;
  /** Starting line number (1-based). */
  startLine: number;
  /** Starting column (0-based). */
  startCol: number;
}

/** Create a new SymbolId. */
export function createSymbolId(
  filePath: string,
  name: string,
  kind: SymbolKind,
  startLine: number,
  startCol: number,
): SymbolId {
  return { filePath, name, kind, startLine, startCol };
}

/** Generate a unique storage-key string for a SymbolId. */
export function symbolIdToStorageId(id: SymbolId): string {
  return `${id.filePath}:${id.name}:${id.startLine}:${id.startCol}`;
}

/** A definition of a symbol in the codebase. */
export interface Definition {
  symbolId: SymbolId;
  /** Absolute root path of the indexed codebase. */
  rootPath?: string;
  /** Project name (for multi-project support). */
  project?: string;
  /** Ending line number (1-based). */
  endLine: number;
  /** Ending column (0-based). */
  endCol: number;
  /** Full signature or declaration text. */
  signature: string;
  /** Documentation comment if available. */
  docComment?: string;
  /** Visibility modifier. */
  visibility: Visibility;
  /** Parent symbol storage ID (e.g., containing class for a method). */
  parentId?: string;
  /** Timestamp when this definition was indexed (epoch ms). */
  indexedAt: number;
}

/** Generate a unique storage ID for a definition. */
export function definitionToStorageId(def: Definition): string {
  return `def:${def.symbolId.filePath}:${def.symbolId.name}:${def.symbolId.startLine}`;
}

/** A reference to a symbol from another location in the codebase. */
export interface Reference {
  /** File path where the reference occurs. */
  filePath: string;
  rootPath?: string;
  project?: string;
  startLine: number;
  endLine: number;
  startCol: number;
  endCol: number;
  /** Storage ID of the target symbol being referenced. */
  targetSymbolId: string;
  referenceKind: ReferenceKind;
  indexedAt: number;
}

/** Generate a unique storage ID for a reference. */
export function referenceToStorageId(ref_: Reference): string {
  return `ref:${ref_.filePath}:${ref_.startLine}:${ref_.startCol}`;
}

// ── Call graph types ─────────────────────────────────────────────────────────

/** An edge in the call graph. */
export interface CallEdge {
  /** Storage ID of the caller. */
  callerId: string;
  /** Storage ID of the callee. */
  calleeId: string;
  /** File where the call occurs. */
  callSiteFile: string;
  /** Line where the call occurs. */
  callSiteLine: number;
  /** Column where the call occurs. */
  callSiteCol: number;
}

/** A node in the call graph (for tree display). */
export interface CallGraphNode {
  name: string;
  kind: SymbolKind;
  filePath: string;
  line: number;
  children: CallGraphNode[];
}

// ── Language statistics ──────────────────────────────────────────────────────

/** Per-language statistics from a code analysis run. */
export interface LanguageStats {
  language: string;
  fileCount: number;
  symbolCount: number;
  referenceCount: number;
}
