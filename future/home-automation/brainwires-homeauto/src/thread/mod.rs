//! Thread protocol support via the OpenThread Border Router (OTBR) REST API.

/// HTTP client for the OTBR REST endpoints (`/node`, `/node/neighbors`, etc.).
pub mod border_router;
/// Response DTOs for the OTBR REST API.
pub mod types;

pub use border_router::ThreadBorderRouter;
pub use types::{ThreadNeighbor, ThreadNetworkDataset, ThreadNodeInfo, ThreadRole};
