//! Graph types and traits for knowledge graph abstraction.
//!
//! Defines entity types, edge types, and trait interfaces for entity stores
//! and relationship graphs. These abstractions allow consumers (like SEAL)
//! to depend on traits rather than concrete storage implementations.

use serde::{Deserialize, Serialize};

/// Types of entities tracked in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// A source file entity.
    File,
    /// A function or method entity.
    Function,
    /// A type (struct, enum, class) entity.
    Type,
    /// A variable or constant entity.
    Variable,
    /// An abstract concept entity.
    Concept,
    /// An error or exception entity.
    Error,
    /// A CLI command entity.
    Command,
}

impl EntityType {
    /// Returns the string representation of this entity type.
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityType::File => "file",
            EntityType::Function => "function",
            EntityType::Type => "type",
            EntityType::Variable => "variable",
            EntityType::Concept => "concept",
            EntityType::Error => "error",
            EntityType::Command => "command",
        }
    }
}

/// Types of edges in the relationship graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// Entities co-occur in the same context.
    CoOccurs,
    /// One entity contains another.
    Contains,
    /// One entity references another.
    References,
    /// One entity depends on another.
    DependsOn,
    /// One entity modifies another.
    Modifies,
    /// One entity defines another.
    Defines,
}

impl EdgeType {
    /// Get the default weight for this edge type.
    pub fn weight(&self) -> f32 {
        match self {
            EdgeType::Defines => 1.0,
            EdgeType::Contains => 0.9,
            EdgeType::DependsOn => 0.8,
            EdgeType::Modifies => 0.7,
            EdgeType::References => 0.6,
            EdgeType::CoOccurs => 0.3,
        }
    }
}

/// A node in the relationship graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Name of the entity.
    pub entity_name: String,
    /// Type classification of the entity.
    pub entity_type: EntityType,
    /// IDs of messages where this entity was mentioned.
    pub message_ids: Vec<String>,
    /// Total number of mentions across all messages.
    pub mention_count: u32,
    /// Computed importance score.
    pub importance: f32,
}

/// An edge in the relationship graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source entity name.
    pub from: String,
    /// Target entity name.
    pub to: String,
    /// Type of relationship.
    pub edge_type: EdgeType,
    /// Strength of the relationship.
    pub weight: f32,
    /// Message ID where this edge was established, if any.
    pub message_id: Option<String>,
}

/// Trait for querying an entity store.
///
/// Provides access to entity information without coupling to a specific storage
/// implementation.
pub trait EntityStoreT: Send + Sync {
    /// Get entity names that match a given type.
    fn entity_names_by_type(&self, entity_type: &EntityType) -> Vec<String>;

    /// Get the top entities by mention count, returning (name, type) pairs.
    fn top_entity_info(&self, limit: usize) -> Vec<(String, EntityType)>;
}

/// Trait for querying a relationship graph.
///
/// Provides read-only access to the graph structure without coupling to a
/// specific implementation.
pub trait RelationshipGraphT: Send + Sync {
    /// Get a node by name.
    fn get_node(&self, name: &str) -> Option<&GraphNode>;

    /// Get all neighbor nodes.
    fn get_neighbors(&self, name: &str) -> Vec<&GraphNode>;

    /// Get all edges for a node.
    fn get_edges(&self, name: &str) -> Vec<&GraphEdge>;

    /// Search for nodes matching a query string.
    fn search(&self, query: &str, limit: usize) -> Vec<&GraphNode>;

    /// Find the shortest path between two nodes.
    fn find_path(&self, from: &str, to: &str) -> Option<Vec<String>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_as_str() {
        assert_eq!(EntityType::File.as_str(), "file");
        assert_eq!(EntityType::Function.as_str(), "function");
        assert_eq!(EntityType::Type.as_str(), "type");
        assert_eq!(EntityType::Variable.as_str(), "variable");
        assert_eq!(EntityType::Concept.as_str(), "concept");
        assert_eq!(EntityType::Error.as_str(), "error");
        assert_eq!(EntityType::Command.as_str(), "command");
    }

    #[test]
    fn test_edge_type_weight() {
        assert_eq!(EdgeType::Defines.weight(), 1.0);
        assert_eq!(EdgeType::Contains.weight(), 0.9);
        assert_eq!(EdgeType::DependsOn.weight(), 0.8);
        assert_eq!(EdgeType::CoOccurs.weight(), 0.3);
    }
}
