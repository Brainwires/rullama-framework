/**
 * Semantic Query Core Extraction.
 *
 * Extracts structured "query cores" (S-expression-inspired) from natural
 * language questions so they can be executed against a relationship graph.
 *
 * Equivalent to Rust's `brainwires_agents::seal::query_core` module.
 */

import type {
  EdgeType,
  EntityType,
  RelationshipGraphT,
} from "@brainwires/core";

// ─── Regex statics ──────────────────────────────────────────────────────────

const RE_WHAT_IS = /what\s+is\s+(\w+)/i;
const RE_EXPLAIN = /explain\s+(\w+)/i;
const RE_WHERE_IS = /where\s+is\s+(.+?)\s*(defined|declared|located)/i;
const RE_WHICH_FILE = /which\s+file\s+(contains|has|defines)\s+(.+)/i;
const RE_FIND_IN = /find\s+(.+?)\s+in/i;
const RE_WHAT_USES = /what\s+(uses|depends\s+on|calls|imports)\s+(.+)/i;
const RE_WHAT_DOES_USE =
  /what\s+does\s+(.+?)\s+(use|depend\s+on|call|import)/i;
const RE_SHOW_DEPS = /show\s+(dependencies|usages)\s+(of|for)\s+(.+)/i;
const RE_HOW_MANY = /how\s+many\s+(.+)/i;
const RE_COUNT = /count\s+(.+)/i;
const RE_WHICH_MOST = /which\s+(.+?)\s+has\s+the\s+(most|least|highest|lowest)/i;
const RE_LARGEST = /(largest|smallest|biggest)\s+(.+)/i;
const RE_LIST = /list\s+(all\s+)?(.+)/i;
const RE_SHOW = /show\s+(all\s+)?(.+)/i;
const RE_DOES_USE =
  /does\s+(.+?)\s+(use|depend|call|import|contain)\s+(.+)/i;
const RE_IS_USED_BY = /is\s+(.+?)\s+(used|called|imported)\s+by\s+(.+)/i;

// ─── Types ──────────────────────────────────────────────────────────────────

/** Relation types that map to graph edge types. */
export type RelationType =
  | { kind: "contains" }
  | { kind: "references" }
  | { kind: "depends_on" }
  | { kind: "modifies" }
  | { kind: "defines" }
  | { kind: "co_occurs" }
  | { kind: "has_type" }
  | { kind: "has_error" }
  | { kind: "created_at" }
  | { kind: "modified_at" }
  | { kind: "custom"; name: string };

/** Relation helpers. */
export function relationToEdgeType(rel: RelationType): EdgeType | undefined {
  switch (rel.kind) {
    case "contains":
      return "contains";
    case "references":
      return "references";
    case "depends_on":
      return "depends_on";
    case "modifies":
      return "modifies";
    case "defines":
      return "defines";
    case "co_occurs":
      return "co_occurs";
    default:
      return undefined;
  }
}

/** Return the inverse relation (if applicable). */
export function relationInverse(rel: RelationType): RelationType | undefined {
  switch (rel.kind) {
    case "contains":
      return { kind: "custom", name: "ContainedBy" };
    case "depends_on":
      return { kind: "custom", name: "DependedOnBy" };
    case "defines":
      return { kind: "custom", name: "DefinedBy" };
    case "modifies":
      return { kind: "custom", name: "ModifiedBy" };
    case "references":
      return { kind: "custom", name: "ReferencedBy" };
    default:
      return undefined;
  }
}

/** Human-readable name for a relation (matches Rust Debug formatting). */
export function relationName(rel: RelationType): string {
  switch (rel.kind) {
    case "contains":
      return "Contains";
    case "references":
      return "References";
    case "depends_on":
      return "DependsOn";
    case "modifies":
      return "Modifies";
    case "defines":
      return "Defines";
    case "co_occurs":
      return "CoOccurs";
    case "has_type":
      return "HasType";
    case "has_error":
      return "HasError";
    case "created_at":
      return "CreatedAt";
    case "modified_at":
      return "ModifiedAt";
    case "custom":
      return `Custom("${rel.name}")`;
  }
}

/** Direction for superlative queries. */
export type SuperlativeDir = "max" | "min";

/** Comparison operators. */
export type CompareOp =
  | "eq"
  | "ne"
  | "lt"
  | "le"
  | "gt"
  | "ge"
  | "contains"
  | "starts_with"
  | "ends_with";

/** Filter predicates for query results. */
export type FilterPredicate =
  | { kind: "has_type"; entity_type: EntityType }
  | { kind: "name_matches"; pattern: string }
  | { kind: "in"; values: string[] }
  | { kind: "not_in"; values: string[] }
  | { kind: "property"; name: string; op: CompareOp; value: string };

/** A query expression (variable, constant, or operation). */
export type QueryExpr =
  | { kind: "variable"; name: string }
  | { kind: "constant"; value: string; entity_type: EntityType }
  | { kind: "op"; op: QueryOp };

/** Core operations in the query language. */
export type QueryOp =
  | {
    kind: "join";
    relation: RelationType;
    subject: QueryExpr;
    object: QueryExpr;
  }
  | { kind: "and"; exprs: QueryExpr[] }
  | { kind: "or"; exprs: QueryExpr[] }
  | { kind: "values"; values: string[] }
  | { kind: "filter"; source: QueryExpr; predicate: FilterPredicate }
  | { kind: "count"; inner: QueryExpr }
  | {
    kind: "superlative";
    source: QueryExpr;
    property: string;
    direction: SuperlativeDir;
  };

/** Create a new variable expression (prefixes '?' if absent). */
export function queryVar(name: string): QueryExpr {
  const trimmed = name.startsWith("?") ? name.slice(1) : name;
  return { kind: "variable", name: `?${trimmed}` };
}

/** Create a new constant expression. */
export function queryConstant(
  value: string,
  entity_type: EntityType,
): QueryExpr {
  return { kind: "constant", value, entity_type };
}

/** Create a join operation. */
export function queryJoin(
  relation: RelationType,
  subject: QueryExpr,
  object: QueryExpr,
): QueryExpr {
  return { kind: "op", op: { kind: "join", relation, subject, object } };
}

/** Create a count operation. */
export function queryCount(inner: QueryExpr): QueryExpr {
  return { kind: "op", op: { kind: "count", inner } };
}

/** Check if this is a variable. */
export function isVariable(expr: QueryExpr): boolean {
  return expr.kind === "variable";
}

/** Get the variable name if this is a variable. */
export function asVariable(expr: QueryExpr): string | undefined {
  return expr.kind === "variable" ? expr.name : undefined;
}

/** Question type classification. */
export type QuestionType =
  | "definition"
  | "location"
  | "dependency"
  | "count"
  | "superlative"
  | "enumeration"
  | "boolean"
  | "multi_hop"
  | "unknown";

/** A complete query core with metadata. */
export interface QueryCore {
  question_type: QuestionType;
  root: QueryExpr;
  entities: [string, EntityType][];
  original: string;
  resolved: string | undefined;
  confidence: number;
}

/** Create a new query core. */
export function newQueryCore(
  question_type: QuestionType,
  root: QueryExpr,
  entities: [string, EntityType][],
  original: string,
): QueryCore {
  return {
    question_type,
    root,
    entities,
    original,
    resolved: undefined,
    confidence: 1.0,
  };
}

/** Convert a query core to a human-readable S-expression string. */
export function queryCoreToSexp(core: QueryCore): string {
  return exprToSexp(core.root);
}

function exprToSexp(expr: QueryExpr): string {
  switch (expr.kind) {
    case "variable":
      return expr.name;
    case "constant":
      return `"${expr.value}"`;
    case "op":
      return opToSexp(expr.op);
  }
}

function opToSexp(op: QueryOp): string {
  switch (op.kind) {
    case "join":
      return `(JOIN ${relationName(op.relation)} ${exprToSexp(op.subject)} ${
        exprToSexp(op.object)
      })`;
    case "and":
      return `(AND ${op.exprs.map(exprToSexp).join(" ")})`;
    case "or":
      return `(OR ${op.exprs.map(exprToSexp).join(" ")})`;
    case "values":
      return `(VALUES ${op.values.join(" ")})`;
    case "filter":
      return `(FILTER ${exprToSexp(op.source)} ${
        predicateToDebug(op.predicate)
      })`;
    case "count":
      return `(COUNT ${exprToSexp(op.inner)})`;
    case "superlative": {
      const dir = op.direction === "max" ? "ARGMAX" : "ARGMIN";
      return `(${dir} ${exprToSexp(op.source)} ${op.property})`;
    }
  }
}

function predicateToDebug(p: FilterPredicate): string {
  switch (p.kind) {
    case "has_type":
      return `HasType(${p.entity_type})`;
    case "name_matches":
      return `NameMatches("${p.pattern}")`;
    case "in":
      return `In([${p.values.join(", ")}])`;
    case "not_in":
      return `NotIn([${p.values.join(", ")}])`;
    case "property":
      return `Property { name: "${p.name}", op: ${p.op}, value: "${p.value}" }`;
  }
}

// ─── Query results ──────────────────────────────────────────────────────────

/** A single result value. */
export interface QueryResultValue {
  value: string;
  entity_type: EntityType | undefined;
  score: number;
  metadata: Map<string, string>;
}

/** Result of executing a query core. */
export interface QueryResult {
  values: QueryResultValue[];
  count: number | undefined;
  success: boolean;
  error: string | undefined;
}

/** Create an empty successful result. */
export function queryResultEmpty(): QueryResult {
  return { values: [], count: undefined, success: true, error: undefined };
}

/** Create an error result. */
export function queryResultError(msg: string): QueryResult {
  return { values: [], count: undefined, success: false, error: msg };
}

/** Create a result with values. */
export function queryResultWithValues(values: QueryResultValue[]): QueryResult {
  return { count: values.length, values, success: true, error: undefined };
}

// ─── Extractor ──────────────────────────────────────────────────────────────

interface QuestionPattern {
  regex: RegExp;
  question_type: QuestionType;
  relation: RelationType | undefined;
}

/** Query core extractor. */
export class QueryCoreExtractor {
  private patterns: QuestionPattern[] = [
    { regex: RE_WHAT_IS, question_type: "definition", relation: { kind: "defines" } },
    { regex: RE_EXPLAIN, question_type: "definition", relation: { kind: "defines" } },
    { regex: RE_WHERE_IS, question_type: "location", relation: { kind: "contains" } },
    { regex: RE_WHICH_FILE, question_type: "location", relation: { kind: "contains" } },
    { regex: RE_FIND_IN, question_type: "location", relation: { kind: "contains" } },
    { regex: RE_WHAT_USES, question_type: "dependency", relation: { kind: "depends_on" } },
    { regex: RE_WHAT_DOES_USE, question_type: "dependency", relation: { kind: "depends_on" } },
    { regex: RE_SHOW_DEPS, question_type: "dependency", relation: { kind: "depends_on" } },
    { regex: RE_HOW_MANY, question_type: "count", relation: undefined },
    { regex: RE_COUNT, question_type: "count", relation: undefined },
    { regex: RE_WHICH_MOST, question_type: "superlative", relation: undefined },
    { regex: RE_LARGEST, question_type: "superlative", relation: undefined },
    { regex: RE_LIST, question_type: "enumeration", relation: undefined },
    { regex: RE_SHOW, question_type: "enumeration", relation: undefined },
    { regex: RE_DOES_USE, question_type: "boolean", relation: { kind: "depends_on" } },
    { regex: RE_IS_USED_BY, question_type: "boolean", relation: { kind: "depends_on" } },
  ];

  /** Classify a question by type. */
  classifyQuestion(
    query: string,
  ): [QuestionType, RelationType | undefined] {
    for (const p of this.patterns) {
      if (p.regex.test(query)) {
        return [p.question_type, p.relation];
      }
    }
    return ["unknown", undefined];
  }

  /** Extract a query core from natural language. */
  extract(
    query: string,
    entities: [string, EntityType][],
  ): QueryCore | undefined {
    const [question_type, relation] = this.classifyQuestion(query);

    if (question_type === "unknown") return undefined;

    const queryLower = query.toLowerCase();
    const mentioned: [string, EntityType][] = entities.filter(([name]) =>
      queryLower.includes(name.toLowerCase())
    );

    let root: QueryExpr;

    switch (question_type) {
      case "definition": {
        if (mentioned.length === 0) return undefined;
        const [name, et] = mentioned[0];
        root = queryJoin(
          { kind: "defines" },
          queryVar("definer"),
          queryConstant(name, et),
        );
        break;
      }
      case "location": {
        if (mentioned.length === 0) return undefined;
        const [name, et] = mentioned[0];
        root = queryJoin(
          { kind: "contains" },
          queryVar("container"),
          queryConstant(name, et),
        );
        break;
      }
      case "dependency": {
        if (mentioned.length === 0) return undefined;
        const [name, et] = mentioned[0];
        const rel: RelationType = relation ?? { kind: "depends_on" };
        if (
          queryLower.includes("what uses") ||
          queryLower.includes("what depends on")
        ) {
          root = queryJoin(rel, queryVar("dependent"), queryConstant(name, et));
        } else {
          root = queryJoin(rel, queryConstant(name, et), queryVar("dependency"));
        }
        break;
      }
      case "count": {
        if (mentioned.length > 0) {
          const [name, et] = mentioned[0];
          root = queryCount(
            queryJoin(
              { kind: "contains" },
              queryVar("container"),
              queryConstant(name, et),
            ),
          );
        } else {
          root = queryCount(queryVar("entity"));
        }
        break;
      }
      case "superlative": {
        const direction: SuperlativeDir =
          queryLower.includes("most") ||
            queryLower.includes("largest") ||
            queryLower.includes("highest")
            ? "max"
            : "min";
        root = {
          kind: "op",
          op: {
            kind: "superlative",
            source: queryVar("entity"),
            property: "mention_count",
            direction,
          },
        };
        break;
      }
      case "enumeration": {
        if (mentioned.length > 0) {
          const [name, et] = mentioned[0];
          root = queryJoin(
            { kind: "contains" },
            queryVar("container"),
            queryConstant(name, et),
          );
        } else {
          root = queryVar("entity");
        }
        break;
      }
      case "boolean": {
        if (mentioned.length < 2) return undefined;
        const rel: RelationType = relation ?? { kind: "depends_on" };
        root = queryJoin(
          rel,
          queryConstant(mentioned[0][0], mentioned[0][1]),
          queryConstant(mentioned[1][0], mentioned[1][1]),
        );
        break;
      }
      case "multi_hop":
        return undefined;
    }

    return newQueryCore(question_type, root, mentioned, query);
  }
}

// ─── Executor ───────────────────────────────────────────────────────────────

/** Query executor for running query cores against a relationship graph. */
export class QueryExecutor {
  constructor(private graph: RelationshipGraphT) {}

  execute(query: QueryCore): QueryResult {
    return this.executeExpr(query.root);
  }

  private executeExpr(expr: QueryExpr): QueryResult {
    switch (expr.kind) {
      case "variable": {
        const values: QueryResultValue[] = this.graph.search("", 100).map(
          (node) => ({
            value: node.entity_name,
            entity_type: node.entity_type,
            score: node.importance,
            metadata: new Map(),
          }),
        );
        return queryResultWithValues(values);
      }
      case "constant":
        return queryResultWithValues([{
          value: expr.value,
          entity_type: expr.entity_type,
          score: 1.0,
          metadata: new Map(),
        }]);
      case "op":
        return this.executeOp(expr.op);
    }
  }

  private executeOp(op: QueryOp): QueryResult {
    switch (op.kind) {
      case "join": {
        const edgeType = relationToEdgeType(op.relation);
        const pickName = (e: QueryExpr): string | undefined =>
          e.kind === "constant" ? e.value : undefined;
        const objName = pickName(op.object);
        const subName = pickName(op.subject);
        const name = objName ?? subName;
        if (name === undefined) return queryResultEmpty();

        const neighbors = this.graph.getNeighbors(name);
        const edges = this.graph.getEdges(name);
        const values: QueryResultValue[] = [];
        for (let i = 0; i < Math.min(neighbors.length, edges.length); i++) {
          if (edgeType !== undefined && edges[i].edge_type !== edgeType) {
            continue;
          }
          values.push({
            value: neighbors[i].entity_name,
            entity_type: neighbors[i].entity_type,
            score: edges[i].weight,
            metadata: new Map(),
          });
        }
        return queryResultWithValues(values);
      }
      case "and": {
        let results: QueryResultValue[] | undefined;
        for (const e of op.exprs) {
          const r = this.executeExpr(e);
          if (!r.success) return r;
          if (results === undefined) {
            results = r.values;
          } else {
            const rset = new Set(r.values.map((v) => v.value));
            results = results.filter((v) => rset.has(v.value));
          }
        }
        return queryResultWithValues(results ?? []);
      }
      case "or": {
        const values: QueryResultValue[] = [];
        const seen = new Set<string>();
        for (const e of op.exprs) {
          for (const v of this.executeExpr(e).values) {
            if (!seen.has(v.value)) {
              seen.add(v.value);
              values.push(v);
            }
          }
        }
        return queryResultWithValues(values);
      }
      case "values":
        return queryResultWithValues(op.values.map((v) => ({
          value: v,
          entity_type: undefined,
          score: 1.0,
          metadata: new Map(),
        })));
      case "filter": {
        const result = this.executeExpr(op.source);
        result.values = result.values.filter((v) =>
          applyPredicate(v, op.predicate)
        );
        result.count = result.values.length;
        return result;
      }
      case "count": {
        const inner = this.executeExpr(op.inner);
        return {
          values: [],
          count: inner.values.length,
          success: inner.success,
          error: inner.error,
        };
      }
      case "superlative": {
        const result = this.executeExpr(op.source);
        result.values.sort((a, b) =>
          op.direction === "max" ? b.score - a.score : a.score - b.score
        );
        result.values = result.values.slice(0, 1);
        result.count = result.values.length;
        return result;
      }
    }
  }
}

function applyPredicate(
  v: QueryResultValue,
  p: FilterPredicate,
): boolean {
  switch (p.kind) {
    case "has_type":
      return v.entity_type === p.entity_type;
    case "name_matches":
      try {
        return new RegExp(p.pattern).test(v.value);
      } catch {
        return false;
      }
    case "in":
      return p.values.includes(v.value);
    case "not_in":
      return !p.values.includes(v.value);
    case "property": {
      const propValue = v.metadata.get(p.name);
      if (propValue === undefined) return false;
      switch (p.op) {
        case "eq":
          return propValue === p.value;
        case "ne":
          return propValue !== p.value;
        case "contains":
          return propValue.includes(p.value);
        case "starts_with":
          return propValue.startsWith(p.value);
        case "ends_with":
          return propValue.endsWith(p.value);
        default:
          return false;
      }
    }
  }
}
