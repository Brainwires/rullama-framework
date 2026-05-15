use crate::output::Output;
use anyhow::Result;
use mdns_sd::{ServiceDaemon, ServiceEvent};
use std::time::Duration;

/// Browse for Matter devices on the local network via mDNS.
///
/// Listens on both:
/// - `_matterc._udp` — commissionable devices (not yet commissioned)
/// - `_matter._tcp`  — operational devices (already commissioned)
pub async fn run(timeout_secs: u64, out: &Output) -> Result<()> {
    let daemon = ServiceDaemon::new()?;
    let timeout = Duration::from_secs(timeout_secs);

    if !out.json {
        println!("Browsing for Matter devices ({timeout_secs}s)…");
    }

    let mut found: Vec<DiscoveredDevice> = Vec::new();

    // Browse commissionable devices (_matterc._udp)
    let rx_c = daemon.browse("_matterc._udp")?;
    // Browse operational nodes (_matter._tcp)
    let rx_o = daemon.browse("_matter._tcp")?;

    let deadline = std::time::Instant::now() + timeout;

    loop {
        let remaining = deadline
            .checked_duration_since(std::time::Instant::now())
            .unwrap_or(Duration::ZERO);
        if remaining.is_zero() {
            break;
        }

        // Poll commissionable
        if let Ok(ServiceEvent::ServiceResolved(info)) =
            rx_c.recv_timeout(Duration::from_millis(50))
        {
            found.push(DiscoveredDevice {
                kind: DeviceKind::Commissionable,
                instance: info.get_fullname().to_owned(),
                addresses: info.get_addresses().iter().map(|a| a.to_string()).collect(),
                port: info.get_port(),
                txt: info
                    .get_properties()
                    .iter()
                    .map(|p| format!("{}={}", p.key(), p.val_str()))
                    .collect(),
            });
        }

        // Poll operational
        if let Ok(ServiceEvent::ServiceResolved(info)) =
            rx_o.recv_timeout(Duration::from_millis(50))
        {
            found.push(DiscoveredDevice {
                kind: DeviceKind::Operational,
                instance: info.get_fullname().to_owned(),
                addresses: info.get_addresses().iter().map(|a| a.to_string()).collect(),
                port: info.get_port(),
                txt: info
                    .get_properties()
                    .iter()
                    .map(|p| format!("{}={}", p.key(), p.val_str()))
                    .collect(),
            });
        }
    }

    let _ = daemon.stop_browse("_matterc._udp");
    let _ = daemon.stop_browse("_matter._tcp");

    if out.json {
        let items: Vec<serde_json::Value> = found
            .iter()
            .map(|d| {
                serde_json::json!({
                    "kind": d.kind.as_str(),
                    "instance": d.instance,
                    "addresses": d.addresses,
                    "port": d.port,
                    "txt": d.txt,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".into())
        );
    } else if found.is_empty() {
        println!("No Matter devices found.");
    } else {
        for d in &found {
            println!(
                "[{}] {}  {}:{}",
                d.kind.as_str(),
                d.instance,
                d.addresses.first().map(String::as_str).unwrap_or("-"),
                d.port
            );
            for t in &d.txt {
                println!("    {t}");
            }
        }
    }

    Ok(())
}

enum DeviceKind {
    Commissionable,
    Operational,
}

impl DeviceKind {
    fn as_str(&self) -> &'static str {
        match self {
            DeviceKind::Commissionable => "commissionable",
            DeviceKind::Operational => "operational",
        }
    }
}

struct DiscoveredDevice {
    kind: DeviceKind,
    instance: String,
    addresses: Vec<String>,
    port: u16,
    txt: Vec<String>,
}
