use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

use ipnetwork::IpNetwork;
use pnet::packet::Packet;
use pnet::packet::arp::{ArpHardwareTypes, ArpOperations, ArpPacket, MutableArpPacket};
use pnet::packet::ethernet::{EtherTypes, EthernetPacket, MutableEthernetPacket};
use pnet::util::MacAddr;
use pnet_datalink::{self as datalink, Channel};
use tokio::task;
use tracing::{debug, warn};

use super::types::DiscoveredHost;

/// Send ARP requests for all hosts in `subnet` and collect replies.
///
/// Requires the calling process to have `CAP_NET_RAW` (or root) on Linux.
/// Uses the first interface whose assigned addresses overlap with `subnet`.
///
/// Returns discovered hosts with MAC addresses from ARP replies.
pub async fn arp_scan(subnet: IpNetwork) -> Vec<DiscoveredHost> {
    let subnet_copy = subnet;
    task::spawn_blocking(move || arp_scan_blocking(subnet_copy, Duration::from_millis(500)))
        .await
        .unwrap_or_default()
}

/// Probe a list of specific hosts with ARP (useful when subnet is unknown).
pub async fn arp_probe(hosts: Vec<IpAddr>) -> Vec<DiscoveredHost> {
    task::spawn_blocking(move || {
        let mut results = Vec::new();
        for host in hosts {
            if let IpAddr::V4(v4) = host {
                // Build a /32 network just to satisfy arp_scan_blocking's interface lookup
                if let Ok(net) = IpNetwork::new(host, 32) {
                    let found = arp_scan_blocking(net, Duration::from_millis(300));
                    results.extend(found);
                } else {
                    let _ = v4; // suppress unused warning on non-Linux
                }
            }
        }
        results
    })
    .await
    .unwrap_or_default()
}

fn arp_scan_blocking(subnet: IpNetwork, timeout: Duration) -> Vec<DiscoveredHost> {
    let IpNetwork::V4(v4_net) = subnet else {
        warn!("arp_scan only supports IPv4 subnets");
        return Vec::new();
    };

    // Find a suitable interface
    let interfaces = datalink::interfaces();
    let interface = interfaces.iter().find(|iface| {
        iface.ips.iter().any(|ip| {
            if let IpAddr::V4(addr) = ip.ip() {
                v4_net.contains(addr) || iface.ips.iter().any(|i| i.contains(IpAddr::V4(addr)))
            } else {
                false
            }
        })
    });

    // Fall back to first non-loopback interface with an IPv4 address
    let interface = interface.or_else(|| {
        interfaces
            .iter()
            .find(|i| !i.is_loopback() && i.ips.iter().any(|ip| ip.is_ipv4()))
    });

    let interface = match interface {
        Some(i) => i.clone(),
        None => {
            warn!("No suitable interface found for ARP scan");
            return Vec::new();
        }
    };

    let src_mac = match interface.mac {
        Some(m) => m,
        None => {
            warn!("Interface {} has no MAC address", interface.name);
            return Vec::new();
        }
    };

    let src_ip: Ipv4Addr = interface
        .ips
        .iter()
        .find_map(|ip| {
            if let IpAddr::V4(v4) = ip.ip() {
                Some(v4)
            } else {
                None
            }
        })
        .unwrap_or(Ipv4Addr::UNSPECIFIED);

    let (mut tx, mut rx) = match datalink::channel(&interface, Default::default()) {
        Ok(Channel::Ethernet(tx, rx)) => (tx, rx),
        _ => {
            warn!("Could not open datalink channel on {}", interface.name);
            return Vec::new();
        }
    };

    let broadcast = MacAddr::broadcast();

    // Send ARP request for each host in subnet
    for target_ip in v4_net.iter() {
        let mut eth_buf = [0u8; 42];
        let mut arp_buf = [0u8; 28];

        let mut arp = MutableArpPacket::new(&mut arp_buf).unwrap();
        arp.set_hardware_type(ArpHardwareTypes::Ethernet);
        arp.set_protocol_type(EtherTypes::Ipv4);
        arp.set_hw_addr_len(6);
        arp.set_proto_addr_len(4);
        arp.set_operation(ArpOperations::Request);
        arp.set_sender_hw_addr(src_mac);
        arp.set_sender_proto_addr(src_ip);
        arp.set_target_hw_addr(MacAddr::zero());
        arp.set_target_proto_addr(target_ip);

        let mut eth = MutableEthernetPacket::new(&mut eth_buf).unwrap();
        eth.set_destination(broadcast);
        eth.set_source(src_mac);
        eth.set_ethertype(EtherTypes::Arp);
        eth.set_payload(arp.packet());

        let _ = tx.send_to(eth.packet(), None);
    }

    // Collect replies
    let deadline = std::time::Instant::now() + timeout;
    let mut discovered: std::collections::HashMap<Ipv4Addr, String> =
        std::collections::HashMap::new();

    while std::time::Instant::now() < deadline {
        match rx.next() {
            Ok(frame) => {
                if let Some(eth) = EthernetPacket::new(frame)
                    && eth.get_ethertype() == EtherTypes::Arp
                    && let Some(arp) = ArpPacket::new(eth.payload())
                    && arp.get_operation() == ArpOperations::Reply
                {
                    let ip = arp.get_sender_proto_addr();
                    let mac = arp.get_sender_hw_addr().to_string();
                    debug!("ARP reply: {ip} is at {mac}");
                    discovered.insert(ip, mac);
                }
            }
            Err(_) => break,
        }
    }

    discovered
        .into_iter()
        .map(|(ip, mac)| DiscoveredHost {
            ip: IpAddr::V4(ip),
            mac: Some(mac),
            hostname: reverse_lookup(IpAddr::V4(ip)),
        })
        .collect()
}

/// Attempt a reverse DNS lookup. Returns `None` if resolution fails or times out.
fn reverse_lookup(addr: IpAddr) -> Option<String> {
    use std::net::ToSocketAddrs;
    let sa = std::net::SocketAddr::new(addr, 0);
    // stdlib does not expose reverse DNS directly; use dns_lookup on a
    // best-effort basis — return None for now to avoid heavy deps.
    let _ = format!("{sa}").to_socket_addrs().ok()?.next();
    None
}
