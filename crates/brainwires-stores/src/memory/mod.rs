//! Tiered hot/warm/cold agent memory — schema + CRUD.
//!
//! - **Hot tier** — full messages with embeddings ([`MessageStore`]).
//! - **Warm tier** — compressed summaries ([`SummaryStore`]).
//! - **Cold tier** — key-fact extracts ([`FactStore`]).
//! - **Mental models** — synthesised behavioural / structural / causal /
//!   procedural beliefs ([`MentalModelStore`]).
//! - **Tier metadata** — placement, access counts, importance scores
//!   ([`TierMetadataStore`]).
//!
//! All stores are generic over `brainwires_storage::StorageBackend`.
//!
//! Orchestration over these tiers (`TieredMemory`, multi-factor adaptive
//! search) and the offline `dream` consolidation engine (summarisation,
//! fact extraction, demotion) live in the **`brainwires-memory`** crate
//! — this module is schema only.

pub mod fact_store;
pub mod mental_model_store;
pub mod message_store;
pub mod summary_store;
pub mod tier_metadata_store;
pub mod tier_types;

pub use fact_store::{FactStore, facts_field_defs, facts_schema};
pub use mental_model_store::{MentalModel, MentalModelStore, ModelType};
pub use message_store::{MessageMetadata, MessageStore, messages_schema};
pub use summary_store::{SummaryStore, summaries_field_defs, summaries_schema};
pub use tier_metadata_store::TierMetadataStore;
pub use tier_types::{
    FactType, KeyFact, MemoryAuthority, MemoryTier, MessageSummary, TierMetadata,
};
