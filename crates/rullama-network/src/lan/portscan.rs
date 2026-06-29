use std::net::{IpAddr, SocketAddr};
use std::time::Duration;

use futures::stream::{self, StreamExt};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tracing::debug;

use super::types::{PortScanResult, PortState};

/// Scan a list of specific ports on `host`, returning the state of each.
///
/// Uses async TCP connect — no raw sockets required. Ports that refuse the
/// connection are `Closed`; ports that don't respond within `connect_timeout`
/// are `Filtered`.
///
/// `concurrency` controls how many simultaneous connect attempts are in flight.
pub async fn scan_ports(
    host: IpAddr,
    ports: &[u16],
    connect_timeout: Duration,
    concurrency: usize,
) -> Vec<PortScanResult> {
    let ports = ports.to_vec();
    stream::iter(ports)
        .map(|port| {
            let addr = SocketAddr::new(host, port);
            async move {
                let state = probe_tcp(addr, connect_timeout).await;
                debug!("{addr} => {state:?}");
                PortScanResult { host, port, state }
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await
}

/// Scan a contiguous port range on `host`.
///
/// `start` and `end` are both inclusive.
pub async fn scan_range(
    host: IpAddr,
    start: u16,
    end: u16,
    connect_timeout: Duration,
    concurrency: usize,
) -> Vec<PortScanResult> {
    let ports: Vec<u16> = (start..=end).collect();
    scan_ports(host, &ports, connect_timeout, concurrency).await
}

/// Scan a list of well-known service ports on `host`.
pub async fn scan_common_ports(host: IpAddr, connect_timeout: Duration) -> Vec<PortScanResult> {
    const COMMON: &[u16] = &[
        21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 443, 445, 993, 995, 1723, 3306, 3389,
        5900, 8080, 8443,
    ];
    scan_ports(host, COMMON, connect_timeout, 32).await
}

async fn probe_tcp(addr: SocketAddr, connect_timeout: Duration) -> PortState {
    match timeout(connect_timeout, TcpStream::connect(addr)).await {
        Ok(Ok(_)) => PortState::Open,
        Ok(Err(e)) => {
            // "Connection refused" → closed; anything else → filtered
            if e.kind() == std::io::ErrorKind::ConnectionRefused {
                PortState::Closed
            } else {
                PortState::Filtered
            }
        }
        Err(_) => PortState::Filtered, // timed out
    }
}
