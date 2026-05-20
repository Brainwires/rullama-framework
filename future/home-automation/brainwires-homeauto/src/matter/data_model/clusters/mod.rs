//! Cluster server implementations for required Matter commissioning clusters.
//!
//! Each sub-module implements [`ClusterServer`](super::ClusterServer) for one cluster:
//!
//! | Module                      | Cluster ID | Description                               |
//! |-----------------------------|-----------|-------------------------------------------|
//! | [`basic_information`]       | 0x0028     | Device identity attributes                |
//! | [`general_commissioning`]   | 0x0030     | FailSafe, regulatory config               |
//! | [`operational_credentials`] | 0x003E     | NOC, fabrics, attestation                 |
//! | [`network_commissioning`]   | 0x0031     | Network interface config (on-network)     |

/// Basic Information cluster (0x0028) — device identity attributes.
pub mod basic_information;
/// General Commissioning cluster (0x0030) — FailSafe + regulatory config.
pub mod general_commissioning;
/// Network Commissioning cluster (0x0031) — network-interface provisioning.
pub mod network_commissioning;
/// Operational Credentials cluster (0x003E) — NOC / fabric / attestation.
pub mod operational_credentials;
