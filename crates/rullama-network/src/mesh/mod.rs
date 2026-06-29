//! Distributed agent mesh networking.
//!
//! Cross-mesh federation and topology management.

/// Federation gateways and policies for cross-mesh communication.
pub mod federation;
/// Mesh topology management and layout types.
pub mod topology;

pub use federation::{FederationGateway, FederationPolicy};
pub use topology::{MeshTopology, TopologyType};
