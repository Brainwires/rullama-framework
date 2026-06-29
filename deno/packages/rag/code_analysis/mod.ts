/**
 * Code analysis — AST-aware symbol extraction, call graphs, and reference tracking.
 *
 * Provides two extraction modes:
 * 1. **Regex mode** (default, zero dependencies) — pattern-based extraction that
 *    works everywhere. Supports TypeScript, JavaScript, Python, and Rust.
 * 2. **Tree-sitter mode** (optional) — if `npm:web-tree-sitter` is available,
 *    use it for precise AST parsing. (Not yet implemented.)
 *
 * Ported from `rullama-knowledge/src/code_analysis/`.
 *
 * @module
 */

// ── Types ────────────────────────────────────────────────────────────────────
export type {
  CallEdge,
  CallGraphNode,
  Definition,
  LanguageStats,
  Reference,
  ReferenceKind,
  SymbolId,
  SymbolKind,
  Visibility,
} from "./types.ts";

export {
  createSymbolId,
  definitionToStorageId,
  referenceToStorageId,
  symbolIdToStorageId,
  symbolKindDisplayName,
  visibilityFromKeywords,
} from "./types.ts";

// ── RepoMap (regex-based symbol extraction) ──────────────────────────────────
export { RepoMap } from "./repo_map.ts";
export type { ExtractOptions } from "./repo_map.ts";
export { determineReferenceKind } from "./repo_map.ts";

// ── Relations (call graph & reference tracking) ──────────────────────────────
export { buildCallGraph, CallGraph, findReferences } from "./relations.ts";
