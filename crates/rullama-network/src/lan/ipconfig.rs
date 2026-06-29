use std::net::IpAddr;

use ipnetwork::IpNetwork;

use super::interfaces::list_interfaces;
use super::types::IpConfig;

/// Return IP configuration for all active interfaces, including default
/// gateways where detectable.
pub fn get_ip_configs() -> Vec<IpConfig> {
    let interfaces = list_interfaces();
    let gateways = read_default_gateways();

    interfaces
        .into_iter()
        .map(|iface| {
            let gateway = gateways.get(&iface.name).copied();
            IpConfig {
                interface: iface.name,
                addrs: iface.addrs,
                gateway,
            }
        })
        .collect()
}

/// Return all IP addresses assigned to a specific interface.
pub fn get_interface_addrs(name: &str) -> Vec<IpNetwork> {
    list_interfaces()
        .into_iter()
        .find(|i| i.name == name)
        .map(|i| i.addrs)
        .unwrap_or_default()
}

/// Parse `/proc/net/route` (Linux) to find default gateways per interface.
/// Returns a map of interface name → gateway IP.
/// On non-Linux platforms returns an empty map.
fn read_default_gateways() -> std::collections::HashMap<String, IpAddr> {
    #[allow(unused_mut)]
    let mut map = std::collections::HashMap::new();

    #[cfg(target_os = "linux")]
    {
        use std::net::Ipv4Addr;
        use tracing::warn;
        let content = match std::fs::read_to_string("/proc/net/route") {
            Ok(c) => c,
            Err(e) => {
                warn!("Could not read /proc/net/route: {e}");
                return map;
            }
        };

        for line in content.lines().skip(1) {
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 8 {
                continue;
            }
            let iface = cols[0];
            let dest = u32::from_str_radix(cols[1], 16).unwrap_or(1);
            let gw_hex = u32::from_str_radix(cols[2], 16).unwrap_or(0);
            let flags = u32::from_str_radix(cols[3], 16).unwrap_or(0);

            // Flag 0x3 = RTF_UP | RTF_GATEWAY, dest == 0 means default route
            const RTF_UP: u32 = 0x1;
            const RTF_GATEWAY: u32 = 0x2;
            if dest == 0 && (flags & RTF_UP != 0) && (flags & RTF_GATEWAY != 0) && gw_hex != 0 {
                let gw_bytes = gw_hex.to_le_bytes();
                let gw = IpAddr::V4(Ipv4Addr::from(gw_bytes));
                map.insert(iface.to_string(), gw);
            }
        }
    }

    map
}
