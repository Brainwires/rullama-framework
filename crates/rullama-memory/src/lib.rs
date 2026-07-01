//! # rullama-memory
//!
//! Tiered hot/warm/cold agent memory **orchestration**.
//!
//! The schema layer (the five tier stores — `MessageStore`, `SummaryStore`,
//! `FactStore`, `MentalModelStore`, `TierMetadataStore`, plus the shared
//! `tier_types`) lives in [`rullama_stores`]. This crate adds:
//!
//! - [`TieredMemory`] — multi-factor adaptive search across all four tiers
//!   (similarity × recency × importance), plus promotion / demotion of
//!   entries when access patterns change.
//! - [`dream`] — offline consolidation engine that summarises hot-tier
//!   messages into warm-tier summaries, extracts cold-tier facts, and
//!   demotes by retention score. Feature-gated behind `dream`.
//!
//! [`TieredMemory`]: tiered_memory::TieredMemory
//! [`rullama_stores`]: https://docs.rs/rullama-stores

#[cfg(feature = "dream")]
pub mod dream;

pub mod tiered_memory;

pub use tiered_memory::{
    CanonicalWriteToken, MultiFactorScore, TieredMemory, TieredMemoryConfig, TieredMemoryStats,
    TieredSearchResult,
};

// Re-export the schema types from rullama-stores so a consumer that only
// pulls rullama-memory still gets the tier_types it'll need to interact
// with the orchestrator's API.
pub use rullama_stores::{
    FactStore, FactType, KeyFact, MemoryAuthority, MemoryTier, MentalModel, MentalModelStore,
    MessageMetadata, MessageStore, MessageSummary, ModelType, SummaryStore, TierMetadata,
    TierMetadataStore,
};
