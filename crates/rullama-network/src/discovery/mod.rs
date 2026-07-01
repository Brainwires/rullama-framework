//! # Discovery Layer
//!
//! How agents find each other on the network. The [`Discovery`](crate::discovery::Discovery) trait
//! provides a uniform interface for registering an agent's presence,
//! discovering peers, and looking up specific agents.
//!
//! ## Provided implementations
//!
//! | Implementation | Feature flag | Description |
//! |---------------|-------------|-------------|
//! | [`ManualDiscovery`](crate::discovery::ManualDiscovery) | *(always)* | Explicit peer list — no network calls |
//! | `RegistryDiscovery` | `registry-discovery` | HTTP-backed central agent registry |

mod manual;
mod traits;

#[cfg(feature = "registry-discovery")]
mod registry;

pub use manual::ManualDiscovery;
pub use traits::{Discovery, DiscoveryProtocol};

#[cfg(feature = "registry-discovery")]
pub use registry::RegistryDiscovery;
