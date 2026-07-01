/**
 * Call graph construction and reference tracking.
 *
 * Tracks which functions call which, building a directed graph of
 * caller/callee relationships from extracted definitions and references.
 *
 * Ported from `rullama-knowledge/src/code_analysis/repomap/reference_finder.rs`
 * and the call-graph logic in `mod.rs`.
 *
 * @module
 */

import type {
  CallEdge,
  CallGraphNode,
  Definition,
  Reference,
  ReferenceKind,
  SymbolKind,
} from "./types.ts";
import { definitionToStorageId } from "./types.ts";
import { determineReferenceKind } from "./repo_map.ts";

// ── Reference finder ─────────────────────────────────────────────────────────

/** Regex for valid identifiers. */
const IDENTIFIER_RE = /\b[a-zA-Z_][a-zA-Z0-9_]*\b/g;

/**
 * Find all references to known symbols in a source file.
 *
 * @param filePath  Relative path of the file being scanned.
 * @param content   Source text of the file.
 * @param symbolIndex  Map from symbol name to its definitions.
 * @param rootPath  Optional root path to store on references.
 * @param project   Optional project name.
 * @returns Array of references found in the file.
 */
export function findReferences(
  filePath: string,
  content: string,
  symbolIndex: Map<string, Definition[]>,
  rootPath?: string,
  project?: string,
): Reference[] {
  if (symbolIndex.size === 0) return [];

  const references: Reference[] = [];
  const now = Date.now();
  const lines = content.split("\n");

  for (let lineIdx = 0; lineIdx < lines.length; lineIdx++) {
    const line = lines[lineIdx];
    const lineNumber = lineIdx + 1; // 1-based

    // Reset global regex
    IDENTIFIER_RE.lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = IDENTIFIER_RE.exec(line)) !== null) {
      const name = match[0];
      const defs = symbolIndex.get(name);
      if (!defs) continue;

      // Skip if this is a definition site in the same file
      if (isDefinitionSite(defs, filePath, lineNumber)) continue;

      const referenceKind: ReferenceKind = determineReferenceKind(
        line,
        match.index,
        name,
      );

      const targetDef = defs[0];
      references.push({
        filePath,
        rootPath,
        project,
        startLine: lineNumber,
        endLine: lineNumber,
        startCol: match.index,
        endCol: match.index + name.length,
        targetSymbolId: definitionToStorageId(targetDef),
        referenceKind,
        indexedAt: now,
      });
    }
  }

  return references;
}

/** Check whether a line falls within a definition site in the same file. */
function isDefinitionSite(
  defs: Definition[],
  filePath: string,
  lineNumber: number,
): boolean {
  return defs.some(
    (d) =>
      d.symbolId.filePath === filePath &&
      lineNumber >= d.symbolId.startLine &&
      lineNumber <= d.endLine,
  );
}

// ── Call graph ───────────────────────────────────────────────────────────────

/**
 * Directed call graph built from definitions and references.
 *
 * Nodes are definitions (identified by their storage ID) and edges are
 * call-site references.
 */
export class CallGraph {
  /** Map from storage ID to definition. */
  readonly nodes: Map<string, Definition> = new Map();
  /** All call edges. */
  readonly edges: CallEdge[] = [];

  /** Add a definition as a node. */
  addNode(def: Definition): void {
    this.nodes.set(definitionToStorageId(def), def);
  }

  /** Add a call edge. */
  addEdge(edge: CallEdge): void {
    this.edges.push(edge);
  }

  /** Get all callees of a given caller (outgoing edges). */
  calleesOf(callerId: string): CallEdge[] {
    return this.edges.filter((e) => e.callerId === callerId);
  }

  /** Get all callers of a given callee (incoming edges). */
  callersOf(calleeId: string): CallEdge[] {
    return this.edges.filter((e) => e.calleeId === calleeId);
  }

  /**
   * Build a tree of callees from a root symbol, up to `maxDepth` levels.
   */
  calleeTree(rootId: string, maxDepth = 3): CallGraphNode | undefined {
    const def = this.nodes.get(rootId);
    if (!def) return undefined;
    return this.buildTree(rootId, maxDepth, new Set());
  }

  private buildTree(
    nodeId: string,
    depth: number,
    visited: Set<string>,
  ): CallGraphNode | undefined {
    const def = this.nodes.get(nodeId);
    if (!def) return undefined;

    const node: CallGraphNode = {
      name: def.symbolId.name,
      kind: def.symbolId.kind,
      filePath: def.symbolId.filePath,
      line: def.symbolId.startLine,
      children: [],
    };

    if (depth <= 0 || visited.has(nodeId)) return node;
    visited.add(nodeId);

    for (const edge of this.calleesOf(nodeId)) {
      const child = this.buildTree(edge.calleeId, depth - 1, visited);
      if (child) node.children.push(child);
    }

    return node;
  }
}

/**
 * Build a CallGraph from a set of definitions and a source-file map.
 *
 * @param definitions  All known definitions across the codebase.
 * @param files        Map of relative file path to source content.
 * @returns A populated CallGraph.
 */
export function buildCallGraph(
  definitions: Definition[],
  files: Map<string, string>,
): CallGraph {
  const graph = new CallGraph();

  // Index definitions by name for reference lookup
  const symbolIndex = new Map<string, Definition[]>();
  for (const def of definitions) {
    graph.addNode(def);
    const name = def.symbolId.name;
    const list = symbolIndex.get(name) ?? [];
    list.push(def);
    symbolIndex.set(name, list);
  }

  // Build a map from storage ID to definition for quick callee lookup
  const defById = new Map<string, Definition>();
  for (const def of definitions) {
    defById.set(definitionToStorageId(def), def);
  }

  // For each file, find references and create call edges
  for (const [filePath, content] of files) {
    const refs = findReferences(filePath, content, symbolIndex);

    // Determine which definition (if any) contains each reference line
    const fileDefs = definitions.filter((d) =>
      d.symbolId.filePath === filePath
    );

    for (const ref of refs) {
      if (ref.referenceKind !== "call") continue;

      // Find the enclosing function / method for this call site
      const caller = findEnclosingDefinition(fileDefs, ref.startLine);
      if (!caller) continue;

      const callerId = definitionToStorageId(caller);

      graph.addEdge({
        callerId,
        calleeId: ref.targetSymbolId,
        callSiteFile: filePath,
        callSiteLine: ref.startLine,
        callSiteCol: ref.startCol,
      });
    }
  }

  return graph;
}

/**
 * Find the innermost definition that contains the given line.
 *
 * Prefers the smallest range (most specific scope).
 */
function findEnclosingDefinition(
  defs: Definition[],
  line: number,
): Definition | undefined {
  const callableKinds: Set<SymbolKind> = new Set([
    "function",
    "method",
    "class",
  ]);

  let best: Definition | undefined;
  let bestSize = Infinity;

  for (const def of defs) {
    if (!callableKinds.has(def.symbolId.kind)) continue;
    if (line >= def.symbolId.startLine && line <= def.endLine) {
      const size = def.endLine - def.symbolId.startLine;
      if (size < bestSize) {
        bestSize = size;
        best = def;
      }
    }
  }

  return best;
}
