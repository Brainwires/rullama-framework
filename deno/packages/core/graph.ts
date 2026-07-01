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
  entity_name: string;
  entity_type: EntityType;
  message_ids: string[];
  mention_count: number;
  importance: number;
}

/** An edge in the relationship graph.
 * Equivalent to Rust's `GraphEdge` in rullama-core. */
export interface GraphEdge {
  from: string;
  to: string;
  edge_type: EdgeType;
  weight: number;
  message_id?: string;
}

/** Interface for querying an entity store.
 * Equivalent to Rust's `EntityStoreT` trait in rullama-core. */
export interface EntityStoreT {
  entityNamesByType(entityType: EntityType): string[];
  topEntityInfo(limit: number): [string, EntityType][];
}

/** Interface for querying a relationship graph.
 * Equivalent to Rust's `RelationshipGraphT` trait in rullama-core. */
export interface RelationshipGraphT {
  getNode(name: string): GraphNode | undefined;
  getNeighbors(name: string): GraphNode[];
  getEdges(name: string): GraphEdge[];
  search(query: string, limit: number): GraphNode[];
  findPath(from: string, to: string): string[] | undefined;
}
