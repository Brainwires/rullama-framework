/**
 * Coreference Resolution for Multi-Turn Conversations.
 *
 * Resolves anaphoric references (pronouns, definite NPs, ellipsis) to concrete
 * entities from the conversation history using salience-based ranking.
 *
 * Equivalent to Rust's `brainwires_agents::seal::coreference` module.
 */

import type {
  EntityStoreT,
  EntityType,
  RelationshipGraphT,
} from "@brainwires/core";

// ─── Regex statics ──────────────────────────────────────────────────────────

const RE_SINGULAR_NEUTRAL = /\b(it|this|that)\b/g;
const RE_PLURAL = /\b(they|them|those|these)\b/g;

const RE_THE_FILE = /\bthe\s+(file|files)\b/g;
const RE_THE_FUNCTION = /\bthe\s+(function|method|fn)\b/g;
const RE_THE_TYPE = /\bthe\s+(type|struct|class|enum|interface)\b/g;
const RE_THE_ERROR = /\bthe\s+(error|bug|issue)\b/g;
const RE_THE_VARIABLE = /\bthe\s+(variable|var|const|let)\b/g;
const RE_THE_COMMAND = /\bthe\s+(command|cmd)\b/g;

const RE_DEMO_FILE = /\b(that|this)\s+(file)\b/g;
const RE_DEMO_FUNCTION = /\b(that|this)\s+(function|method|fn)\b/g;
const RE_DEMO_TYPE = /\b(that|this)\s+(type|struct|class|enum)\b/g;
const RE_DEMO_ERROR = /\b(that|this)\s+(error|bug|issue)\b/g;

// ─── Types ──────────────────────────────────────────────────────────────────

/** Types of anaphoric references. */
export type ReferenceType =
  | { kind: "singular_neutral" }
  | { kind: "plural" }
  | { kind: "definite_np"; entity_type: EntityType }
  | { kind: "demonstrative"; entity_type: EntityType }
  | { kind: "ellipsis" };

/** Get compatible entity types for this reference type. */
export function compatibleTypes(ref: ReferenceType): EntityType[] {
  switch (ref.kind) {
    case "singular_neutral":
      return [
        "file",
        "function",
        "type",
        "variable",
        "error",
        "concept",
        "command",
      ];
    case "plural":
      return ["file", "function", "type", "variable", "error"];
    case "definite_np":
    case "demonstrative":
      return [ref.entity_type];
    case "ellipsis":
      return ["file", "function", "type", "command"];
  }
}

/** An unresolved reference detected in user input. */
export interface UnresolvedReference {
  text: string;
  ref_type: ReferenceType;
  start: number;
  end: number;
}

/** Salience factors for ranking antecedent candidates. */
export interface SalienceScore {
  recency: number;
  frequency: number;
  graph_centrality: number;
  type_match: number;
  syntactic_prominence: number;
}

/** Compute the weighted total salience. */
export function salienceTotal(s: SalienceScore): number {
  return s.recency * 0.35 +
    s.frequency * 0.15 +
    s.graph_centrality * 0.2 +
    s.type_match * 0.2 +
    s.syntactic_prominence * 0.1;
}

/** A resolved reference with its antecedent. */
export interface ResolvedReference {
  reference: UnresolvedReference;
  antecedent: string;
  entity_type: EntityType;
  confidence: number;
  salience: SalienceScore;
}

// ─── DialogState ────────────────────────────────────────────────────────────

/** Dialog state for tracking entities across conversation turns. */
export class DialogState {
  focus_stack: string[] = [];
  mention_history: Map<string, number[]> = new Map();
  current_turn = 0;
  recently_modified: string[] = [];
  private entity_types: Map<string, EntityType> = new Map();

  /** Advance to the next turn. */
  nextTurn(): void {
    this.current_turn += 1;
  }

  /** Record a mention of an entity. */
  mentionEntity(name: string, entityType: EntityType): void {
    this.focus_stack = this.focus_stack.filter((n) => n !== name);
    this.focus_stack.unshift(name);
    if (this.focus_stack.length > 20) {
      this.focus_stack.length = 20;
    }

    const turns = this.mention_history.get(name) ?? [];
    turns.push(this.current_turn);
    this.mention_history.set(name, turns);

    this.entity_types.set(name, entityType);
  }

  /** Mark an entity as recently modified. */
  markModified(name: string): void {
    this.recently_modified = this.recently_modified.filter((n) => n !== name);
    this.recently_modified.unshift(name);
    if (this.recently_modified.length > 10) {
      this.recently_modified.length = 10;
    }
  }

  /** Get the entity type for a name (if known). */
  getEntityType(name: string): EntityType | undefined {
    return this.entity_types.get(name);
  }

  /** Get the recency score for an entity (1.0 for most recent, decays with age). */
  recencyScore(name: string): number {
    const pos = this.focus_stack.indexOf(name);
    if (pos !== -1) {
      const focusScore = 1.0 - pos / this.focus_stack.length;
      const modifiedBonus = this.recently_modified.includes(name) ? 0.2 : 0.0;
      return Math.min(focusScore + modifiedBonus, 1.0);
    }

    const turns = this.mention_history.get(name);
    if (turns !== undefined && turns.length > 0) {
      const lastTurn = turns[turns.length - 1];
      const age = Math.max(0, this.current_turn - lastTurn);
      return Math.exp(-0.1 * age);
    }
    return 0.0;
  }

  /** Get the frequency score for an entity. */
  frequencyScore(name: string): number {
    const turns = this.mention_history.get(name);
    if (turns === undefined) return 0.0;
    return Math.min(Math.log1p(turns.length) / 3.0, 1.0);
  }

  /** Clear all state. */
  clear(): void {
    this.focus_stack = [];
    this.mention_history.clear();
    this.current_turn = 0;
    this.recently_modified = [];
    this.entity_types.clear();
  }
}

// ─── CoreferenceResolver ────────────────────────────────────────────────────

interface ReferencePattern {
  regex: RegExp;
  makeRef: () => ReferenceType;
}

/** Coreference resolver for multi-turn conversations. */
export class CoreferenceResolver {
  private pronounPatterns: ReferencePattern[];
  private definiteNpPatterns: ReferencePattern[];
  private demonstrativePatterns: ReferencePattern[];

  constructor() {
    this.pronounPatterns = [
      { regex: RE_SINGULAR_NEUTRAL, makeRef: () => ({ kind: "singular_neutral" }) },
      { regex: RE_PLURAL, makeRef: () => ({ kind: "plural" }) },
    ];

    this.definiteNpPatterns = [
      { regex: RE_THE_FILE, makeRef: () => ({ kind: "definite_np", entity_type: "file" }) },
      { regex: RE_THE_FUNCTION, makeRef: () => ({ kind: "definite_np", entity_type: "function" }) },
      { regex: RE_THE_TYPE, makeRef: () => ({ kind: "definite_np", entity_type: "type" }) },
      { regex: RE_THE_ERROR, makeRef: () => ({ kind: "definite_np", entity_type: "error" }) },
      { regex: RE_THE_VARIABLE, makeRef: () => ({ kind: "definite_np", entity_type: "variable" }) },
      { regex: RE_THE_COMMAND, makeRef: () => ({ kind: "definite_np", entity_type: "command" }) },
    ];

    this.demonstrativePatterns = [
      { regex: RE_DEMO_FILE, makeRef: () => ({ kind: "demonstrative", entity_type: "file" }) },
      { regex: RE_DEMO_FUNCTION, makeRef: () => ({ kind: "demonstrative", entity_type: "function" }) },
      { regex: RE_DEMO_TYPE, makeRef: () => ({ kind: "demonstrative", entity_type: "type" }) },
      { regex: RE_DEMO_ERROR, makeRef: () => ({ kind: "demonstrative", entity_type: "error" }) },
    ];
  }

  /** Detect unresolved references in a message. */
  detectReferences(message: string): UnresolvedReference[] {
    const references: UnresolvedReference[] = [];
    const lower = message.toLowerCase();

    const scan = (
      patterns: ReferencePattern[],
      skipOverlaps: boolean,
    ): void => {
      for (const p of patterns) {
        p.regex.lastIndex = 0;
        let m: RegExpExecArray | null;
        while ((m = p.regex.exec(lower)) !== null) {
          const start = m.index;
          const end = m.index + m[0].length;
          if (
            skipOverlaps &&
            references.some((r) => r.start <= start && r.end >= end)
          ) {
            continue;
          }
          references.push({
            text: m[0],
            ref_type: p.makeRef(),
            start,
            end,
          });
        }
      }
    };

    scan(this.demonstrativePatterns, false);
    scan(this.definiteNpPatterns, true);
    scan(this.pronounPatterns, true);

    references.sort((a, b) => a.start - b.start);
    return references;
  }

  /** Resolve references using dialog state and entity store. */
  resolve(
    references: UnresolvedReference[],
    dialogState: DialogState,
    entityStore: EntityStoreT,
    graph?: RelationshipGraphT,
  ): ResolvedReference[] {
    const resolved: ResolvedReference[] = [];
    for (const ref of references) {
      const r = this.resolveSingle(ref, dialogState, entityStore, graph);
      if (r !== undefined) resolved.push(r);
    }
    return resolved;
  }

  private resolveSingle(
    reference: UnresolvedReference,
    dialogState: DialogState,
    entityStore: EntityStoreT,
    graph: RelationshipGraphT | undefined,
  ): ResolvedReference | undefined {
    const compatible = compatibleTypes(reference.ref_type);
    const candidates: {
      name: string;
      entity_type: EntityType;
      salience: SalienceScore;
    }[] = [];

    // Check focus stack first.
    for (const name of dialogState.focus_stack) {
      const et = dialogState.getEntityType(name);
      if (et !== undefined && compatible.includes(et)) {
        candidates.push({
          name,
          entity_type: et,
          salience: this.computeSalience(name, dialogState, graph),
        });
      }
    }

    // Expand with entity store.
    const existing = new Set(candidates.map((c) => c.name));
    for (const et of compatible) {
      for (const name of entityStore.entityNamesByType(et)) {
        if (existing.has(name)) continue;
        existing.add(name);
        candidates.push({
          name,
          entity_type: et,
          salience: this.computeSalience(name, dialogState, graph),
        });
      }
    }

    candidates.sort((a, b) => salienceTotal(b.salience) - salienceTotal(a.salience));
    const best = candidates[0];
    if (best === undefined) return undefined;

    return {
      reference,
      antecedent: best.name,
      entity_type: best.entity_type,
      confidence: salienceTotal(best.salience),
      salience: best.salience,
    };
  }

  private computeSalience(
    name: string,
    dialogState: DialogState,
    graph: RelationshipGraphT | undefined,
  ): SalienceScore {
    const recency = dialogState.recencyScore(name);
    const frequency = dialogState.frequencyScore(name);

    let graphCentrality = 0.5;
    if (graph !== undefined) {
      const node = graph.getNode(name);
      graphCentrality = node !== undefined ? node.importance : 0.0;
    }

    const syntacticProminence = dialogState.focus_stack[0] === name
      ? 1.0
      : dialogState.focus_stack.includes(name)
      ? 0.5
      : 0.0;

    return {
      recency,
      frequency,
      graph_centrality: graphCentrality,
      type_match: 1.0,
      syntactic_prominence: syntacticProminence,
    };
  }

  /** Rewrite a message, replacing references with [antecedent] markers. */
  rewriteWithResolutions(
    message: string,
    resolutions: ResolvedReference[],
  ): string {
    if (resolutions.length === 0) return message;

    const sorted = [...resolutions].sort(
      (a, b) => b.reference.start - a.reference.start,
    );

    let result = message;
    const lower = message.toLowerCase();

    for (const resolution of sorted) {
      const start = resolution.reference.start;
      const end = resolution.reference.end;
      if (end > lower.length || start >= end) continue;

      const refText = lower.slice(start, end);
      const replacement = `[${resolution.antecedent}]`;
      const pos = result.toLowerCase().indexOf(refText);
      if (pos !== -1) {
        result = result.slice(0, pos) + replacement +
          result.slice(pos + (end - start));
      }
    }
    return result;
  }
}

// ─── Minimal EntityStore test helper ───────────────────────────────────────

/**
 * In-memory EntityStore suitable for tests. Equivalent to Rust's
 * `brainwires_knowledge::knowledge::EntityStore` used in the original
 * coreference tests.
 */
export class InMemoryEntityStore implements EntityStoreT {
  private byType: Map<EntityType, Set<string>> = new Map();

  /** Register an entity. */
  add(name: string, entityType: EntityType): void {
    const existing = this.byType.get(entityType) ?? new Set<string>();
    existing.add(name);
    this.byType.set(entityType, existing);
  }

  entityNamesByType(entityType: EntityType): string[] {
    return Array.from(this.byType.get(entityType) ?? []);
  }

  topEntityInfo(limit: number): [string, EntityType][] {
    const out: [string, EntityType][] = [];
    for (const [et, names] of this.byType.entries()) {
      for (const n of names) {
        out.push([n, et]);
        if (out.length >= limit) return out;
      }
    }
    return out;
  }
}
