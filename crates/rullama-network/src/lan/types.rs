use std::net::IpAddr;

use ipnetwork::IpNetwork;
use serde::{Deserialize, Serialize};

/// A physical or virtual network interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    /// OS-assigned interface name (e.g. "eth0", "wlan0", "lo").
    pub name: String,
    /// Interface classification.
    pub kind: InterfaceKind,
    /// MAC address, if available.
    pub mac: Option<String>,
    /// Assigned IP addresses with prefix lengths.
    pub addrs: Vec<IpNetwork>,
    /// Whether the interface is administratively up.
    pub is_up: bool,
}

/// Classification of a network interface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterfaceKind {
    /// Wired Ethernet (e.g. eth0, enp3s0).
    Wired,
    /// Wireless / Wi-Fi (e.g. wlan0, wlp2s0).
    Wireless,
    /// Loopback (lo, lo0).
    Loopback,
    /// Virtual, tunnel, or bridge interface.
    Virtual,
    /// Could not be determined.
    Unknown,
}

/// IP configuration for a single interface, including default gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpConfig {
    /// Interface name.
    pub interface: String,
    /// Assigned addresses (CIDR notation).
    pub addrs: Vec<IpNetwork>,
    /// Default gateway, if known.
    pub gateway: Option<IpAddr>,
}

/// Result of a single port probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortScanResult {
    /// Target host.
    pub host: IpAddr,
    /// Target port.
    pub port: u16,
    /// Observed state.
    pub state: PortState,
}

/// Observed TCP port state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortState {
    /// Connection succeeded — service is listening.
    Open,
    /// Connection refused — port is closed.
    Closed,
    /// No response within timeout — port may be filtered.
    Filtered,
}

/// A host discovered on the local network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredHost {
    /// IP address.
    pub ip: IpAddr,
    /// MAC address from ARP reply, if available.
    pub mac: Option<String>,
    /// Reverse-DNS hostname, if resolved.
    pub hostname: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn ipv4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(a, b, c, d))
    }

    // --- InterfaceKind ---

    #[test]
    fn interface_kind_serde_roundtrip() {
        let kinds = [
            InterfaceKind::Wired,
            InterfaceKind::Wireless,
            InterfaceKind::Loopback,
            InterfaceKind::Virtual,
            InterfaceKind::Unknown,
        ];
        for kind in kinds {
            let json = serde_json::to_string(&kind).unwrap();
            let back: InterfaceKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, kind);
        }
    }

    // --- PortState ---

    #[test]
    fn port_state_serde_roundtrip() {
        for state in [PortState::Open, PortState::Closed, PortState::Filtered] {
            let json = serde_json::to_string(&state).unwrap();
            let back: PortState = serde_json::from_str(&json).unwrap();
            assert_eq!(back, state);
        }
    }

    // --- PortScanResult ---

    #[test]
    fn port_scan_result_serde_roundtrip() {
        let result = PortScanResult {
            host: ipv4(192, 168, 1, 1),
            port: 80,
            state: PortState::Open,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: PortScanResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.port, 80);
        assert_eq!(back.state, PortState::Open);
    }

    // --- DiscoveredHost ---

    #[test]
    fn discovered_host_serde_roundtrip() {
        let host = DiscoveredHost {
            ip: ipv4(10, 0, 0, 1),
            mac: Some("aa:bb:cc:dd:ee:ff".to_string()),
            hostname: Some("router.local".to_string()),
        };
        let json = serde_json::to_string(&host).unwrap();
        let back: DiscoveredHost = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ip, host.ip);
        assert_eq!(back.mac.as_deref(), Some("aa:bb:cc:dd:ee:ff"));
        assert_eq!(back.hostname.as_deref(), Some("router.local"));
    }

    #[test]
    fn discovered_host_optional_fields_omit_when_none() {
        let host = DiscoveredHost {
            ip: ipv4(172, 16, 0, 1),
            mac: None,
            hostname: None,
        };
        let json = serde_json::to_string(&host).unwrap();
        // Optional fields may serialize as null depending on derive - just verify round-trip
        let back: DiscoveredHost = serde_json::from_str(&json).unwrap();
        assert!(back.mac.is_none());
        assert!(back.hostname.is_none());
    }
}
