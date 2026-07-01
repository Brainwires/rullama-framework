//! # rullama Brain — Central Knowledge Module
//!
//! The canonical home for all knowledge systems in the rullama:
//!
//! - **Knowledge Systems**: BKS (behavioral truths) and PKS (personal facts)
//! - **Entity Graph**: Entity types, entity store, relationship graph
//! - **Brain Client**: Persistent thought storage with semantic search
//! - **Thought Types**: Categories, sources, and metadata
//! - **Fact Extraction**: Automatic categorization and tag extraction
//!
//! ## Library Usage
//!
//! ```no_run
//! use rullama_knowledge::knowledge::BrainClient;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = BrainClient::new().await?;
//!     Ok(())
//! }
//! ```

/// Knowledge systems (BKS and PKS).
pub mod bks_pks;
/// Persistent thought storage with semantic search.
pub mod brain_client;
/// Memory bank configuration: mission, directives, disposition traits.
pub mod config;
/// Entity types and store for the knowledge graph.
pub mod entity;
/// Automatic fact extraction from text.
pub mod fact_extractor;
/// Entity relationship graph storage and queries.
pub mod relationship_graph;
/// Thought types, categories, and sources.
pub mod thought;
/// Request/response types for MCP tool endpoints.
pub mod types;

// Re-export main types
pub use brain_client::BrainClient;
pub use config::{DispositionTrait, MemoryBankConfig};
pub use entity::{
    ContradictionEvent, ContradictionKind, Entity, EntityStore, EntityStoreStats, EntityType,
    ExtractionResult, Relationship,
};
pub use relationship_graph::{EdgeType, EntityContext, GraphEdge, GraphNode, RelationshipGraph};
pub use thought::{Thought, ThoughtCategory, ThoughtSource};
pub use types::{
    CaptureThoughtRequest, CaptureThoughtResponse, DeleteThoughtRequest, DeleteThoughtResponse,
    GetThoughtRequest, GetThoughtResponse, ListRecentRequest, ListRecentResponse,
    MemoryStatsRequest, MemoryStatsResponse, SearchKnowledgeRequest, SearchKnowledgeResponse,
    SearchMemoryRequest, SearchMemoryResponse,
};
