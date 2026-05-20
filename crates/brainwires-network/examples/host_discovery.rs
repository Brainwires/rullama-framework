//! ARP scan a local subnet to discover hosts.
//!
//! Requires `CAP_NET_RAW` (or root) for raw packet access.
//!
//! Run with:
//! ```bash
//! sudo cargo run -p brainwires-hardware --example host_discovery --features network -- 192.168.1.0/24
//! ```

use ipnetwork::IpNetwork;

#[tokio::main]
async fn main() {
    let subnet: IpNetwork = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "192.168.1.0/24".to_string())
        .parse()
        .expect("Invalid subnet (use CIDR notation, e.g. 192.168.1.0/24)");

    println!("ARP scanning {subnet}...");
    let hosts = brainwires_network::lan::arp_scan(subnet).await;

    if hosts.is_empty() {
        println!("No hosts discovered. (Note: requires CAP_NET_RAW / root)");
    } else {
        println!("Discovered {} host(s):", hosts.len());
        for h in &hosts {
            println!("  {}  mac={:?}  hostname={:?}", h.ip, h.mac, h.hostname);
        }
    }
}
