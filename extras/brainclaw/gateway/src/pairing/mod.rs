//! DM pairing policy for the gateway.
//!
//! By default the gateway ignores any direct message from an unknown user
//! until an operator explicitly pairs that user to the bot. A peer whose
//! `<channel>:<user_id>` is not on the approval list triggers a pairing
//! code flow: the peer receives a neutral one-liner with a 6-digit code,
//! and the operator approves (or rejects) the code out-of-band via the
//! admin API or the `brainclaw pairing` CLI.
//!
//! This module contains:
//! - [`policy::PairingPolicy`] — the per-channel policy enum.
//! - [`store::PairingStore`] — JSON-backed state with pending + approved peers.
//! - [`handler::PairingHandler`] — the interception point used by
//!   [`crate::agent_handler::AgentInboundHandler`].

pub mod handler;
pub mod policy;
pub mod store;

pub use handler::{PairingHandler, PairingOutcome};
pub use policy::{PairingPolicy, default_policy};
pub use store::{PairingStore, PendingCode};
