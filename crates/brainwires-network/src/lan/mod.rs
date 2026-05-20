//! Network hardware discovery, interface enumeration, and port scanning.
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | `interfaces` | Enumerate wired/wireless NICs and their IP addresses |
//! | `ipconfig` | IP configuration and default gateway per interface |
//! | `discovery` | ARP-based host discovery on local subnets |
//! | `portscan` | Async TCP connect-based port scanning |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use brainwires_network::lan;
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() {
//!     // List interfaces
//!     for iface in lan::list_interfaces() {
//!         println!("{} ({:?}) — {:?}", iface.name, iface.kind, iface.addrs);
//!     }
//!
//!     // IP config with gateways
//!     for cfg in lan::get_ip_configs() {
//!         println!("{}: gateway={:?}", cfg.interface, cfg.gateway);
//!     }
//!
//!     // Port scan
//!     let results = lan::scan_common_ports(
//!         "192.168.1.1".parse().unwrap(),
//!         Duration::from_millis(500),
//!     ).await;
//!     for r in results.iter().filter(|r| r.state == lan::PortState::Open) {
//!         println!("Open: {}", r.port);
//!     }
//! }
//! ```

/// Layer-2/3 host discovery: ARP probing and subnet-scoped sweeps.
pub mod discovery;
/// Enumerate the machine's network interfaces (physical + virtual).
pub mod interfaces;
/// Read per-interface IP configuration (addresses, CIDR, gateway).
pub mod ipconfig;
/// TCP port scanning helpers (single-port, common-port set, range).
pub mod portscan;
/// Typed inputs/outputs for the network module (interfaces, hosts, scan results).
pub mod types;

pub use discovery::{arp_probe, arp_scan};
pub use interfaces::list_interfaces;
pub use ipconfig::{get_interface_addrs, get_ip_configs};
pub use portscan::{scan_common_ports, scan_ports, scan_range};
pub use types::{
    DiscoveredHost, InterfaceKind, IpConfig, NetworkInterface, PortScanResult, PortState,
};
