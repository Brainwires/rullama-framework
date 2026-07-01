//! Embedding provider re-exports.
//!
//! Canonical embedding infrastructure lives in `rullama-storage`; this module
//! re-exports it so RAG code can import from a single path. The trait itself is
//! `rullama_core::EmbeddingProvider`.

pub use rullama_core::EmbeddingProvider;
pub use rullama_storage::embeddings::{CachedEmbeddingProvider, FastEmbedManager};
