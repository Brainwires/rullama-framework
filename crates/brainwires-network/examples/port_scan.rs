//! Scan common ports on a host and print open ones.
//!
//! Run with:
//! ```bash
//! cargo run -p brainwires-hardware --example port_scan --features network -- 192.168.1.1
//! ```

use std::net::IpAddr;
use std::time::Duration;

#[tokio::main]
async fn main() {
    let target: IpAddr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1".to_string())
        .parse()
        .expect("Invalid IP address");

    println!("Scanning common ports on {target}...");

    let results =
        brainwires_network::lan::scan_common_ports(target, Duration::from_millis(500)).await;

    let open: Vec<_> = results
        .iter()
        .filter(|r| r.state == brainwires_network::lan::PortState::Open)
        .collect();

    if open.is_empty() {
        println!("No open ports found.");
    } else {
        println!("Open ports:");
        for r in &open {
            println!("  {}", r.port);
        }
    }
}
