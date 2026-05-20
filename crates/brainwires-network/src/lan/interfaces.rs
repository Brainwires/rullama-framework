#[cfg(target_os = "linux")]
use std::path::Path;

use ipnetwork::IpNetwork;
use network_interface::{Addr, NetworkInterface as NI, NetworkInterfaceConfig};
use tracing::warn;

use super::types::{InterfaceKind, NetworkInterface};

/// Enumerate all network interfaces on this system.
///
/// Includes loopback, wired, wireless, and virtual interfaces.
pub fn list_interfaces() -> Vec<NetworkInterface> {
    let raw = match NI::show() {
        Ok(r) => r,
        Err(e) => {
            warn!("Failed to enumerate network interfaces: {e}");
            return Vec::new();
        }
    };

    raw.into_iter()
        .map(|iface| {
            let name = iface.name.clone();
            let kind = detect_kind(&name);
            let mac = iface.mac_addr.as_deref().map(str::to_string);
            let is_up = true; // network-interface only returns active interfaces

            let addrs: Vec<IpNetwork> = iface.addr.iter().filter_map(addr_to_network).collect();

            NetworkInterface {
                name,
                kind,
                mac,
                addrs,
                is_up,
            }
        })
        .collect()
}

/// Detect interface kind from name and (on Linux) sysfs.
fn detect_kind(name: &str) -> InterfaceKind {
    let lower = name.to_lowercase();

    if lower == "lo" || lower.starts_with("lo0") {
        return InterfaceKind::Loopback;
    }

    // Check Linux sysfs for wireless flag
    #[cfg(target_os = "linux")]
    if Path::new(&format!("/sys/class/net/{name}/wireless")).exists() {
        return InterfaceKind::Wireless;
    }

    if lower.starts_with("wl") || lower.starts_with("wifi") || lower.starts_with("ath") {
        return InterfaceKind::Wireless;
    }

    if lower.starts_with("eth")
        || lower.starts_with("en")
        || lower.starts_with("eno")
        || lower.starts_with("enp")
        || lower.starts_with("ens")
    {
        return InterfaceKind::Wired;
    }

    if lower.starts_with("veth")
        || lower.starts_with("docker")
        || lower.starts_with("br-")
        || lower.starts_with("virbr")
        || lower.starts_with("tun")
        || lower.starts_with("tap")
        || lower.starts_with("vlan")
    {
        return InterfaceKind::Virtual;
    }

    InterfaceKind::Unknown
}

fn addr_to_network(addr: &Addr) -> Option<IpNetwork> {
    match addr {
        Addr::V4(v4) => {
            let prefix = v4
                .netmask
                .map(|m| ipv4_netmask_to_prefix(m.octets()))
                .unwrap_or(32);
            IpNetwork::new(v4.ip.into(), prefix).ok()
        }
        Addr::V6(v6) => {
            let prefix = v6
                .netmask
                .map(|m| ipv6_netmask_to_prefix(m.segments()))
                .unwrap_or(128);
            IpNetwork::new(v6.ip.into(), prefix).ok()
        }
    }
}

fn ipv4_netmask_to_prefix(octets: [u8; 4]) -> u8 {
    u32::from_be_bytes(octets).count_ones() as u8
}

fn ipv6_netmask_to_prefix(segments: [u16; 8]) -> u8 {
    segments.iter().map(|s| s.count_ones() as u8).sum()
}
