//! Full billing implementation for Brainwires agents.
//!
//! This crate provides concrete implementations of the hook surface defined in
//! `brainwires-billing` (the framework crate). None of this storage or payment
//! logic belongs in the framework itself.
//!
//! # Components
//!
//! - **[`BillingLedger`]** — async trait for pluggable event storage
//! - **[`InMemoryLedger`]** — in-process ledger for tests and short-lived processes
//! - **[`SqliteLedger`]** *(feature `sqlite`)* — WAL-mode SQLite at
//!   `~/.brainwires/billing/billing.db`
//! - **[`AgentWallet`]** — implements [`BillingHook`]; accumulates spend per
//!   customer, enforces a USD budget ceiling, and persists every event to a ledger
//! - **[`StripeClient`]** *(feature `stripe`)* — reports metered usage, creates
//!   payment links, and queries customer balance via the Stripe REST API
//!
//! # Quick start
//!
//! ```rust,ignore
//! use brainwires_billing_impl::{AgentWallet, SqliteLedger};
//! use brainwires_inference::task_agent::TaskAgentConfig;
//! use std::sync::Arc;
//!
//! let ledger = Arc::new(SqliteLedger::new_default()?);
//! let wallet = Arc::new(AgentWallet::new("customer-42".into(), Some(5.00), ledger));
//!
//! let config = TaskAgentConfig {
//!     billing_hook: Some(wallet),
//!     ..Default::default()
//! };
//! ```

pub mod error;
pub mod ledger;
pub mod wallet;

#[cfg(feature = "sqlite")]
pub mod schema;

#[cfg(feature = "stripe")]
pub mod stripe;

pub use error::BillingImplError;
pub use ledger::{BillingLedger, InMemoryLedger};
pub use wallet::AgentWallet;

#[cfg(feature = "sqlite")]
pub use ledger::SqliteLedger;

#[cfg(feature = "stripe")]
pub use stripe::StripeClient;
