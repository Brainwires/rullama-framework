/** Types of entities tracked in the knowledge graph.
 * Equivalent to Rust's `EntityType` in rullama-core. */
export type EntityType =
  | "file"
  | "function"
  | "type"
  | "variable"
  | "concept"
  | "error"
  | "command";

/** Types of edges in the relationship graph.
 * Equivalent to Rust's `EdgeType` in rullama-core. */
export type EdgeType =
  | "co_occurs"
  | "contains"
  | "references"
  | "depends_on"
  | "modifies"
  | "defines";

/** Get the default weight for an edge type. */
export function edgeTypeWeight(edgeType: EdgeType): number {
  switch (edgeType) {
    case "defines":
      return 1.0;
    case "contains":
      return 0.9;
    case "depends_on":
      return 0.8;
    case "modifies":
      return 0.7;
    case "references":
      return 0.6;
    case "co_occurs":
      return 0.3;
  }
}

/** A node in the relationship graph.
 * Equivalent to Rust's `GraphNode` in rullama-core. */
export interface GraphNode {
  /** Name of the entity (file path, function name, concept, ...). */
  entity_name: string;
  /** What kind of entity the node represents. */
  entity_type: EntityType;
  /** IDs of the messages in which the entity was mentioned. */
  message_ids: string[];
  /** How many times the entity has been mentioned. */
  mention_count: number;
  /** Relative importance score used to rank entities. */
  importance: number;
}

/** An edge in the relationship graph.
 * Equivalent to Rust's `GraphEdge` in rullama-core. */
export interface GraphEdge {
  /** Entity name of the source node. */
  from: string;
  /** Entity name of the target node. */
  to: string;
  /** Kind of relationship the edge encodes. */
  edge_type: EdgeType;
  /** Strength of the relationship (see {@link edgeTypeWeight} for the defaults). */
  weight: number;
  /** Message in which the relationship was observed, if known. */
  message_id?: string;
}

/** Interface for querying an entity store.
 * Equivalent to Rust's `EntityStoreT` trait in rullama-core. */
export interface EntityStoreT {
  /** Names of every stored entity of the given type. */
  entityNamesByType(entityType: EntityType): string[];
  /** The `limit` most important entities as `[name, type]` pairs. */
  topEntityInfo(limit: number): [string, EntityType][];
}

/** Interface for querying a relationship graph.
 * Equivalent to Rust's `RelationshipGraphT` trait in rullama-core. */
export interface RelationshipGraphT {
  /** Look up a node by entity name. */
  getNode(name: string): GraphNode | undefined;
  /** Nodes directly connected to the named entity. */
  getNeighbors(name: string): GraphNode[];
  /** Edges incident to the named entity. */
  getEdges(name: string): GraphEdge[];
  /** Up to `limit` nodes whose entity name matches `query`. */
  search(query: string, limit: number): GraphNode[];
  /** Entity names along a path from `from` to `to`, or undefined if none exists. */
  findPath(from: string, to: string): string[] | undefined;
}
