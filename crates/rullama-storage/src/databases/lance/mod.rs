//! LanceDB unified database backend.
//!
//! [`LanceDatabase`] implements both [`StorageBackend`] and [`VectorDatabase`]
//! using a single shared `lancedb::Connection`. This replaces the former
//! `LanceBackend` + `LanceVectorDB` split.
//!
//! # Feature flag
//!
//! Requires `lance-backend` (included in `native` by default).
//!
//! # Module layout
//!
//! | Module | Contents |
//! |--------|----------|
//! | `arrow_convert` | Arrow ↔ domain-type conversions |
//! | `database` | `LanceDatabase` struct and helper methods |
//! | `storage_backend` | `StorageBackend` impl |
//! | `vector_database` | `VectorDatabase` impl |

pub mod arrow_convert;

mod database;
mod storage_backend;
mod vector_database;

#[cfg(test)]
mod tests;

// Re-export the primary type so callers continue to use the same path.
pub use database::LanceDatabase;
