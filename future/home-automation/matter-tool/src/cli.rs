/// All clap CLI structs for matter-tool.
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "matter-tool",
    about = "Commission and control Matter 1.3 devices using the Brainwires Matter stack",
    version
)]
pub struct Cli {
    /// Fabric storage directory.
    #[arg(long, value_name = "DIR", global = true)]
    pub fabric_dir: Option<PathBuf>,

    /// Enable debug-level tracing.
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Emit machine-readable JSON output instead of pretty text.
    #[arg(long, global = true)]
    pub json: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Commission (pair) a Matter device into the local fabric.
    Pair {
        #[command(subcommand)]
        action: PairAction,
    },
    /// OnOff cluster commands.
    Onoff {
        #[command(subcommand)]
        action: OnoffAction,
    },
    /// LevelControl cluster commands.
    Level {
        #[command(subcommand)]
        action: LevelAction,
    },
    /// Thermostat cluster commands.
    Thermostat {
        #[command(subcommand)]
        action: ThermostatAction,
    },
    /// DoorLock cluster commands.
    Doorlock {
        #[command(subcommand)]
        action: DoorlockAction,
    },
    /// Send a raw cluster command (TLV payload).
    Invoke {
        /// Node ID of the target device.
        node_id: u64,
        /// Endpoint ID (e.g. 1).
        endpoint: u16,
        /// Cluster ID in hex (e.g. 0x0006).
        #[arg(value_parser = parse_hex_u32)]
        cluster_id: u32,
        /// Command ID in hex (e.g. 0x01).
        #[arg(value_parser = parse_hex_u32)]
        command_id: u32,
        /// Optional TLV payload bytes in hex (e.g. 2801).
        payload_hex: Option<String>,
    },
    /// Read a raw cluster attribute.
    Read {
        /// Node ID of the target device.
        node_id: u64,
        /// Endpoint ID (e.g. 1).
        endpoint: u16,
        /// Cluster ID in hex (e.g. 0x0006).
        #[arg(value_parser = parse_hex_u32)]
        cluster_id: u32,
        /// Attribute ID in hex (e.g. 0x0000).
        #[arg(value_parser = parse_hex_u32)]
        attribute_id: u32,
    },
    /// Browse for Matter devices on the local network via mDNS.
    Discover {
        /// How many seconds to listen for mDNS responses.
        #[arg(short, long, default_value = "5")]
        timeout: u64,
    },
    /// Run as a Matter device server (use another controller to commission us).
    Serve {
        /// Device name broadcast in mDNS.
        #[arg(long, default_value = "Brainwires Matter Device")]
        device_name: String,
        /// Vendor ID (hex, e.g. 0xFFF1).
        #[arg(long, default_value = "0xFFF1", value_parser = parse_hex_u16)]
        vendor_id: u16,
        /// Product ID (hex, e.g. 0x8001).
        #[arg(long, default_value = "0x8001", value_parser = parse_hex_u16)]
        product_id: u16,
        /// 12-bit discriminator (0–4095).
        #[arg(long, default_value = "3840")]
        discriminator: u16,
        /// Commissioning passcode.
        #[arg(long, default_value = "20202021")]
        passcode: u32,
        /// UDP port to listen on.
        #[arg(long, default_value = "5540")]
        port: u16,
        /// Storage path for server state.
        #[arg(long)]
        storage: Option<PathBuf>,
    },
    /// List all commissioned devices in the local fabric.
    Devices,
    /// Fabric management.
    Fabric {
        #[command(subcommand)]
        action: FabricAction,
    },
}

// ── Pair ─────────────────────────────────────────────────────────────────────

#[derive(Subcommand, Debug)]
pub enum PairAction {
    /// Commission via QR code (MT:...).
    Qr {
        /// Node ID to assign to this device.
        node_id: u64,
        /// QR code string starting with "MT:".
        qr_code: String,
    },
    /// Commission via 11-digit manual pairing code.
    Code {
        /// Node ID to assign to this device.
        node_id: u64,
        /// 11-digit decimal manual pairing code.
        manual_code: String,
    },
    /// Commission via BLE (requires --features ble).
    Ble {
        /// Node ID to assign to this device.
        node_id: u64,
        /// Commissioning passcode.
        passcode: u32,
        /// 12-bit discriminator.
        discriminator: u16,
    },
    /// Remove a commissioned device from the local fabric.
    Unpair {
        /// Node ID to remove.
        node_id: u64,
    },
}

// ── OnOff ────────────────────────────────────────────────────────────────────

#[derive(Subcommand, Debug)]
pub enum OnoffAction {
    /// Turn the device on.
    On { node_id: u64, endpoint: u16 },
    /// Turn the device off.
    Off { node_id: u64, endpoint: u16 },
    /// Toggle the device.
    Toggle { node_id: u64, endpoint: u16 },
    /// Read the current on/off state.
    Read { node_id: u64, endpoint: u16 },
}

// ── Level ────────────────────────────────────────────────────────────────────

#[derive(Subcommand, Debug)]
pub enum LevelAction {
    /// Set the level (0–254).
    Set {
        node_id: u64,
        endpoint: u16,
        /// Level value 0–254.
        level: u8,
        /// Transition time in tenths of a second (0 = immediate).
        #[arg(long, default_value = "0")]
        transition: u16,
    },
    /// Read the current level.
    Read { node_id: u64, endpoint: u16 },
}

// ── Thermostat ───────────────────────────────────────────────────────────────

#[derive(Subcommand, Debug)]
pub enum ThermostatAction {
    /// Set the occupied heating setpoint (°C).
    Setpoint {
        node_id: u64,
        endpoint: u16,
        /// Target temperature in degrees Celsius (e.g. 21.5).
        celsius: f32,
    },
    /// Read current local temperature and setpoints.
    Read { node_id: u64, endpoint: u16 },
}

// ── DoorLock ─────────────────────────────────────────────────────────────────

#[derive(Subcommand, Debug)]
pub enum DoorlockAction {
    /// Lock the door.
    Lock { node_id: u64, endpoint: u16 },
    /// Unlock the door.
    Unlock { node_id: u64, endpoint: u16 },
    /// Read the current lock state.
    Read { node_id: u64, endpoint: u16 },
}

// ── Fabric ───────────────────────────────────────────────────────────────────

#[derive(Subcommand, Debug)]
pub enum FabricAction {
    /// Print fabric ID, root CA fingerprint, and commissioned node count.
    Info,
    /// Wipe all fabric storage (interactive confirmation required).
    Reset,
}

// ── Hex parsers ───────────────────────────────────────────────────────────────

fn parse_hex_u16(s: &str) -> Result<u16, String> {
    let s = s.trim_start_matches("0x").trim_start_matches("0X");
    u16::from_str_radix(s, 16).map_err(|e| format!("invalid hex u16 '{s}': {e}"))
}

fn parse_hex_u32(s: &str) -> Result<u32, String> {
    let s = s.trim_start_matches("0x").trim_start_matches("0X");
    u32::from_str_radix(s, 16).map_err(|e| format!("invalid hex u32 '{s}': {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("matter-tool").chain(args.iter().copied()))
    }

    // ── pair ─────────────────────────────────────────────────────────────────

    #[test]
    fn pair_qr() {
        let cli = parse(&["pair", "qr", "1", "MT:Y.K9042C00KA0648G00"]).unwrap();
        match cli.command {
            Command::Pair {
                action: PairAction::Qr { node_id, qr_code },
            } => {
                assert_eq!(node_id, 1);
                assert_eq!(qr_code, "MT:Y.K9042C00KA0648G00");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn pair_code() {
        let cli = parse(&["pair", "code", "7", "34970112332"]).unwrap();
        match cli.command {
            Command::Pair {
                action:
                    PairAction::Code {
                        node_id,
                        manual_code,
                    },
            } => {
                assert_eq!(node_id, 7);
                assert_eq!(manual_code, "34970112332");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn pair_unpair() {
        let cli = parse(&["pair", "unpair", "3"]).unwrap();
        match cli.command {
            Command::Pair {
                action: PairAction::Unpair { node_id },
            } => {
                assert_eq!(node_id, 3);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    // ── onoff ────────────────────────────────────────────────────────────────

    #[test]
    fn onoff_on() {
        let cli = parse(&["onoff", "on", "1", "1"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Onoff {
                action: OnoffAction::On {
                    node_id: 1,
                    endpoint: 1
                }
            }
        ));
    }

    #[test]
    fn onoff_toggle() {
        let cli = parse(&["onoff", "toggle", "2", "3"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Onoff {
                action: OnoffAction::Toggle {
                    node_id: 2,
                    endpoint: 3
                }
            }
        ));
    }

    // ── level ────────────────────────────────────────────────────────────────

    #[test]
    fn level_set_default_transition() {
        let cli = parse(&["level", "set", "1", "1", "200"]).unwrap();
        match cli.command {
            Command::Level {
                action:
                    LevelAction::Set {
                        node_id,
                        endpoint,
                        level,
                        transition,
                    },
            } => {
                assert_eq!(node_id, 1);
                assert_eq!(endpoint, 1);
                assert_eq!(level, 200);
                assert_eq!(transition, 0); // default
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn level_set_with_transition() {
        let cli = parse(&["level", "set", "1", "1", "128", "--transition", "10"]).unwrap();
        match cli.command {
            Command::Level {
                action: LevelAction::Set { transition, .. },
            } => {
                assert_eq!(transition, 10);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    // ── invoke / read ─────────────────────────────────────────────────────────

    #[test]
    fn invoke_hex_cluster_and_cmd() {
        let cli = parse(&["invoke", "1", "1", "0x0006", "0x01"]).unwrap();
        match cli.command {
            Command::Invoke {
                cluster_id,
                command_id,
                payload_hex,
                ..
            } => {
                assert_eq!(cluster_id, 0x0006);
                assert_eq!(command_id, 0x01);
                assert!(payload_hex.is_none());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn invoke_with_payload() {
        let cli = parse(&["invoke", "1", "1", "0x0006", "0x01", "DEADBEEF"]).unwrap();
        match cli.command {
            Command::Invoke { payload_hex, .. } => {
                assert_eq!(payload_hex.as_deref(), Some("DEADBEEF"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn read_hex_cluster_and_attr() {
        let cli = parse(&["read", "5", "2", "0x0006", "0x0000"]).unwrap();
        match cli.command {
            Command::Read {
                node_id,
                endpoint,
                cluster_id,
                attribute_id,
            } => {
                assert_eq!(node_id, 5);
                assert_eq!(endpoint, 2);
                assert_eq!(cluster_id, 0x0006);
                assert_eq!(attribute_id, 0x0000);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    // ── global flags ──────────────────────────────────────────────────────────

    #[test]
    fn global_json_flag() {
        let cli = parse(&["--json", "devices"]).unwrap();
        assert!(cli.json);
    }

    #[test]
    fn global_verbose_flag() {
        let cli = parse(&["-v", "devices"]).unwrap();
        assert!(cli.verbose);
    }

    #[test]
    fn global_fabric_dir() {
        let cli = parse(&["--fabric-dir", "/tmp/myfabric", "devices"]).unwrap();
        assert_eq!(cli.fabric_dir, Some(PathBuf::from("/tmp/myfabric")));
    }

    // ── hex parsers ───────────────────────────────────────────────────────────

    #[test]
    fn parse_hex_u32_with_prefix() {
        assert_eq!(parse_hex_u32("0x0006").unwrap(), 6);
        assert_eq!(parse_hex_u32("0XFFF1").unwrap(), 0xFFF1);
        assert_eq!(parse_hex_u32("DEAD").unwrap(), 0xDEAD);
    }

    #[test]
    fn parse_hex_u32_invalid() {
        assert!(parse_hex_u32("0xGGGG").is_err());
    }

    #[test]
    fn parse_hex_u16_limits() {
        assert_eq!(parse_hex_u16("0xFFFF").unwrap(), 0xFFFF);
        assert!(parse_hex_u16("0x10000").is_err()); // overflows u16
    }

    // ── discover / serve / fabric ─────────────────────────────────────────────

    #[test]
    fn discover_default_timeout() {
        let cli = parse(&["discover"]).unwrap();
        assert!(matches!(cli.command, Command::Discover { timeout: 5 }));
    }

    #[test]
    fn discover_custom_timeout() {
        let cli = parse(&["discover", "--timeout", "30"]).unwrap();
        assert!(matches!(cli.command, Command::Discover { timeout: 30 }));
    }

    #[test]
    fn fabric_info() {
        let cli = parse(&["fabric", "info"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Fabric {
                action: FabricAction::Info
            }
        ));
    }

    #[test]
    fn fabric_reset() {
        let cli = parse(&["fabric", "reset"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Fabric {
                action: FabricAction::Reset
            }
        ));
    }

    #[test]
    fn serve_defaults() {
        let cli = parse(&["serve"]).unwrap();
        match cli.command {
            Command::Serve {
                device_name,
                vendor_id,
                product_id,
                discriminator,
                passcode,
                port,
                storage,
            } => {
                assert_eq!(device_name, "Brainwires Matter Device");
                assert_eq!(vendor_id, 0xFFF1);
                assert_eq!(product_id, 0x8001);
                assert_eq!(discriminator, 3840);
                assert_eq!(passcode, 20202021);
                assert_eq!(port, 5540);
                assert!(storage.is_none());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
}
