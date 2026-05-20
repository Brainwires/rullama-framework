#![deny(missing_docs)]
//! # Brainwires Core
//!
//! Foundation types, traits, and error handling for the Brainwires Agent Framework.
//!
//! This crate provides the core data structures used across all framework crates:
//! - Message types for AI conversations
//! - Tool definitions and execution results
//! - Task and agent context types
//! - Plan metadata and status
//! - Working set for file context management
//! - Chat options and provider configuration
//! - Permission modes

/// Response confidence extraction (CISC heuristics) — used by SEAL learning
/// and validation policies to score model output quality.
pub mod confidence;
/// Content source types for tracking where content originates.
pub mod content_source;
/// Embedding provider trait for vector operations.
pub mod embedding;
/// Framework error types and result aliases.
pub mod error;
/// Unified event trait and `EventEnvelope<E>` with trace IDs and sequence numbers.
pub mod event;
/// File chunking + content extraction primitives — `FileContextManager`,
/// `FileContent`, `FileChunk`. Moved from `brainwires-storage` in Phase 9.
/// Native-only (uses `tokio::fs`).
#[cfg(not(target_arch = "wasm32"))]
pub mod file_context;
/// Knowledge graph types: entities, edges, and trait interfaces.
pub mod graph;
/// Lifecycle hooks for intercepting framework events.
pub mod lifecycle;
/// Message, role, and streaming types for AI conversations.
pub mod message;
/// Platform-specific path helpers — `PlatformPaths`. Moved from
/// `brainwires-storage` in Phase 9.
pub mod paths;
/// Permission mode definitions.
pub mod permission;
/// Plan metadata, steps, budgets, and serializable plans.
pub mod plan;
/// Provider configuration and chat options.
pub mod provider;
/// Shared search types (SearchResult, ChunkMetadata, DatabaseStats).
pub mod search;
/// Task, priority, and agent response types.
pub mod task;
/// Tool definitions, schemas, contexts, and idempotency.
pub mod tool;
/// Vector store trait for similarity search.
pub mod vector_store;
/// Persistent workflow state for crash-safe agent retry.
///
/// Native-only: the filesystem-backed checkpoint store uses `tokio::fs` and
/// `dirs::home_dir`, neither of which are available on `wasm32-unknown-unknown`.
#[cfg(not(target_arch = "wasm32"))]
pub mod workflow_state;
/// Working set for file context management with LRU eviction.
pub mod working_set;

// Re-export core types at crate root
pub use content_source::ContentSource;
pub use embedding::EmbeddingProvider;
pub use error::*;
pub use graph::*;
pub use message::*;
pub use permission::*;
pub use plan::*;
pub use provider::*;
pub use search::{ChunkMetadata, DatabaseStats, SearchResult};
pub use task::*;
pub use tool::*;
pub use vector_store::{VectorSearchResult, VectorStore};
pub use working_set::{
    WorkingSet, WorkingSetConfig, WorkingSetEntry, estimate_tokens, estimate_tokens_from_size,
};
