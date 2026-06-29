//! Enumerate network interfaces and print IP configuration.
//!
//! Run with:
//! ```bash
//! cargo run -p rullama-hardware --example network_interfaces --features network
//! ```

fn main() {
    let interfaces = rullama_network::lan::list_interfaces();
    println!("Network interfaces ({}):", interfaces.len());
    for iface in &interfaces {
        println!(
            "  {:10} [{:?}]  mac={:?}  up={}",
            iface.name, iface.kind, iface.mac, iface.is_up
        );
        for addr in &iface.addrs {
            println!("             {addr}");
        }
    }

    println!("\nIP configuration (with gateways):");
    for cfg in rullama_network::lan::get_ip_configs() {
        println!(
            "  {:10} gateway={:?}  addrs={:?}",
            cfg.interface, cfg.gateway, cfg.addrs
        );
    }
}
