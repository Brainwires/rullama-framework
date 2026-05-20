//! # brainwires-stores
//!
//! Schema + CRUD for the framework's opinionated minimum data-store set.
//!
//! Every store is built on the [`brainwires_storage::StorageBackend`] trait —
//! consumers can swap backends without touching store code. Each store
//! family is gated behind a Cargo feature so consumers only pay for what
//! they use.
//!
//! This crate is **schema only**. Orchestration (multi-tier search,
//! promotion / demotion logic) and the offline `dream` consolidation
//! engine live in the **`brainwires-memory`** crate, which depends on
//! the schema types here.
//!
//! ## Feature flags
//!
//! - `session` *(default)* — `SessionStore` trait + `InMemorySessionStore`
//!   (and `SqliteSessionStore` with the `sqlite` feature). Full-transcript
//!   persistence keyed by session id.
//! - `task` *(default)* — `TaskStore` + `AgentStateStore`.
//! - `plan` *(default)* — `PlanStore` + `TemplateStore`.
//! - `conversation` *(default)* — `ConversationStore` (catalog metadata).
//! - `memory` — tier schema stores: `MessageStore`, `SummaryStore`,
//!   `FactStore`, `MentalModelStore`, `TierMetadataStore`, plus the
//!   shared `tier_types` (`MemoryTier`, `MemoryAuthority`, `TierMetadata`,
//!   `MessageSummary`, `KeyFact`, `FactType`).
//! - `lock` — `LockStore`. Coordination locks (rusqlite-backed).
//! - `image` — `ImageStore` with hashing + metadata.
//! - `sqlite` — pulls rusqlite for backends that need it.

#[cfg(feature = "session")]
pub mod session;

#[cfg(feature = "memory")]
pub mod memory;

#[cfg(feature = "conversation")]
pub mod conversation_store;

#[cfg(feature = "image")]
pub mod image_store;

#[cfg(feature = "lock")]
pub mod lock_store;

#[cfg(feature = "plan")]
pub mod plan_store;

#[cfg(feature = "task")]
pub mod task_store;

#[cfg(feature = "plan")]
pub mod template_store;

#[cfg(feature = "session")]
pub use session::{
    ArcSessionStore, InMemorySessionStore, ListOptions, Message, SessionBroker, SessionError,
    SessionId, SessionMessage, SessionRecord, SessionStore, SessionSummary, SpawnRequest,
    SpawnedSession,
};

#[cfg(all(feature = "session", feature = "sqlite"))]
pub use session::SqliteSessionStore;

#[cfg(feature = "memory")]
pub use memory::{
    FactStore, FactType, KeyFact, MemoryAuthority, MemoryTier, MentalModel, MentalModelStore,
    MessageMetadata, MessageStore, MessageSummary, ModelType, SummaryStore, TierMetadata,
    TierMetadataStore, facts_field_defs, facts_schema, messages_schema, summaries_field_defs,
    summaries_schema, tier_types,
};

#[cfg(feature = "conversation")]
pub use conversation_store::{ConversationMetadata, ConversationStore};

#[cfg(feature = "image")]
pub use image_store::ImageStore;

#[cfg(feature = "lock")]
pub use lock_store::{LockRecord, LockStats, LockStore};

#[cfg(feature = "plan")]
pub use plan_store::PlanStore;

#[cfg(feature = "task")]
pub use task_store::{AgentStateMetadata, AgentStateStore, TaskMetadata, TaskStore};

#[cfg(feature = "plan")]
pub use template_store::{PlanTemplate, TemplateStore};
