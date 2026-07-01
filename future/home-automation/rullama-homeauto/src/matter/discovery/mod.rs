/// Matter device discovery — commissionable and operational DNS-SD advertisement
/// and browsing.
///
/// ## Overview
///
/// Matter uses DNS-SD (mDNS) for two distinct discovery phases:
///
/// 1. **Commissionable discovery** (`_matterc._udp`) — before commissioning.
///    A new device broadcasts its discriminator and pairing metadata so that
///    a commissioner app can find it.  See [`CommissionableAdvertiser`].
///
/// 2. **Operational discovery** (`_matter._tcp`) — after commissioning.
///    A commissioned node advertises its compressed-fabric-id + node-id so
///    that controllers on the same fabric can reach it.
///    See [`OperationalAdvertiser`] and [`OperationalBrowser`].
///
/// ## References
///
/// - Matter Core Specification §4.3 — Device Discovery
/// - Matter Core Specification §4.3.1.2 — Commissionable Node Discovery
/// - Matter Core Specification §4.3.2 — Operational Node Discovery
///
/// Commissionable device advertisement (`_matterc._udp`).
pub mod commissionable;
/// Operational device advertisement and browsing (`_matter._tcp`).
pub mod operational;

pub use commissionable::CommissionableAdvertiser;
pub use operational::{OperationalAdvertiser, OperationalBrowser, derive_compressed_fabric_id};
