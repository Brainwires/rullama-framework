//! Offline memory consolidation — the write path for warm/cold tiers.
//!
//! `dream` runs periodically (via cron or on-demand) and turns older
//! conversation messages into summaries and durable facts, reducing token
//! pressure while preserving important knowledge. It's the consolidation
//! engine paired with the data structures in [`crate::tiered_memory`] plus
//! the `SummaryStore` / `FactStore` re-exported from `brainwires-stores`.
//!
//! Lives in `brainwires-memory` rather than a separate crate because dream
//! and the tiered memory stores it writes to are one concern — splitting
//! them creates two halves that don't stand alone.

/// Orchestrates summarisation + fact extraction over a [`consolidator::DreamSessionStore`].
pub mod consolidator;
/// Extracts durable facts from message transcripts via a `Provider`.
pub mod fact_extractor;
/// Counters + per-cycle reports for instrumentation.
pub mod metrics;
/// Tier transition policy (when to demote hot → warm, warm → cold).
pub mod policy;
/// Summariser that compresses message batches into one long-form note.
pub mod summarizer;
/// Async task wrapper that runs a consolidation cycle.
pub mod task;
