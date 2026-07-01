//! Backend capability declarations.
//!
//! [`BackendCapabilities`] lets callers discover what a database backend
//! supports at runtime (e.g. whether vector search is available).

/// Describes the capabilities of a database backend.
///
/// Backends that implement
/// [`StorageBackend`](super::traits::StorageBackend) can override the default
/// `capabilities` method to advertise what they support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    /// Whether the backend supports vector similarity search.
    ///
    /// Backends without native vector support (e.g. MySQL) should return
    /// `false` here; calling `vector_search` on them will return an error.
    pub vector_search: bool,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            vector_search: true,
        }
    }
}
