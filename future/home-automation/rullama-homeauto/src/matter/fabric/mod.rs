/// Matter TLV-encoded certificate (NOC / RCAC / ICAC) codec.
pub mod cert;
/// Fabric lifecycle management (root CA, NOC issuance, storage).
pub mod manager;
/// Matter 1.3 fabric management.
///
/// A Matter **fabric** is an administrative domain composed of a root Certificate
/// Authority (RCAC), one or more Node Operational Certificates (NOCs), and the
/// identifiers that bind nodes together.
///
/// This module provides:
/// - `types` — `FabricIndex`, `OperationalNodeId`, `CompressedFabricId`, `FabricDescriptor`
/// - `cert`  — `MatterCert` / `MatterCertSubject`: TLV encoding + decoding
/// - `manager` — `FabricManager`: root-CA generation, NOC issuance, persistence
///
/// Identity types for Matter fabrics.
pub mod types;

pub use cert::{MatterCert, MatterCertSubject};
pub use manager::{FabricManager, StoredFabricEntry};
pub use types::{CompressedFabricId, FabricDescriptor, FabricIndex, OperationalNodeId};
