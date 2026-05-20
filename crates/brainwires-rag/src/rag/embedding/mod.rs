//! Embedding provider re-exports.
//!
//! Canonical embedding infrastructure lives in `brainwires-storage`; this module
//! re-exports it so RAG code can import from a single path. The trait itself is
//! `brainwires_core::EmbeddingProvider`.

pub use brainwires_core::EmbeddingProvider;
pub use brainwires_storage::embeddings::{CachedEmbeddingProvider, FastEmbedManager};
