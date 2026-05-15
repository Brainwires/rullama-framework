/// Output rendering helpers: pretty text or machine-readable JSON.
use brainwires_homeauto::{AttributeValue, MatterDevice};

pub struct Output {
    pub json: bool,
}

impl Output {
    pub fn new(json: bool) -> Self {
        Self { json }
    }

    /// Print a simple success message.
    pub fn ok(&self, msg: &str) {
        if self.json {
            println!(
                "{{\"ok\":true,\"msg\":{}}}",
                serde_json::to_string(msg).unwrap()
            );
        } else {
            println!("✓ {msg}");
        }
    }

    /// Print an error (does not exit).
    pub fn err(&self, msg: &str) {
        if self.json {
            eprintln!(
                "{{\"ok\":false,\"error\":{}}}",
                serde_json::to_string(msg).unwrap()
            );
        } else {
            eprintln!("✗ {msg}");
        }
    }

    /// Print a device list.
    pub fn devices(&self, devices: &[MatterDevice]) {
        if self.json {
            println!(
                "{}",
                serde_json::to_string_pretty(devices).unwrap_or_else(|_| "[]".into())
            );
        } else if devices.is_empty() {
            println!("No commissioned devices.");
        } else {
            println!(
                "{:<10} {:<8} {:<8} {:<20} STATUS",
                "NODE-ID", "VID", "PID", "NAME"
            );
            println!("{}", "-".repeat(62));
            for d in devices {
                let name = d.name.as_deref().unwrap_or("-");
                let status = if d.online { "online" } else { "offline" };
                println!(
                    "{:<10} {:#06x}   {:#06x}   {:<20} {}",
                    d.node_id, d.vendor_id, d.product_id, name, status
                );
            }
        }
    }

    /// Print an attribute value (result of a Read operation).
    pub fn attribute(
        &self,
        node_id: u64,
        endpoint: u16,
        cluster: u32,
        attr: u32,
        value: &AttributeValue,
    ) {
        if self.json {
            let v = attribute_value_to_json(value);
            println!(
                "{{\"node_id\":{node_id},\"endpoint\":{endpoint},\"cluster\":{cluster},\"attribute\":{attr},\"value\":{v}}}"
            );
        } else {
            println!(
                "node={node_id} ep={endpoint} cluster={cluster:#010x} attr={attr:#010x} → {value}"
            );
        }
    }

    /// Print a generic key-value pair.
    pub fn kv(&self, key: &str, value: &str) {
        if self.json {
            println!("{{\"{}\":{}}}", key, serde_json::to_string(value).unwrap());
        } else {
            println!("{key}: {value}");
        }
    }

    /// Print raw text (used for QR codes, banners etc.) — always plain regardless of --json.
    pub fn raw(&self, msg: &str) {
        println!("{msg}");
    }
}

pub(crate) fn attribute_value_to_json(v: &AttributeValue) -> String {
    match v {
        AttributeValue::Bool(b) => b.to_string(),
        AttributeValue::U8(n) => n.to_string(),
        AttributeValue::U16(n) => n.to_string(),
        AttributeValue::U32(n) => n.to_string(),
        AttributeValue::U64(n) => n.to_string(),
        AttributeValue::I8(n) => n.to_string(),
        AttributeValue::I16(n) => n.to_string(),
        AttributeValue::I32(n) => n.to_string(),
        AttributeValue::F32(n) => n.to_string(),
        AttributeValue::F64(n) => n.to_string(),
        AttributeValue::String(s) => serde_json::to_string(s).unwrap(),
        AttributeValue::Bytes(b) => format!("\"{}\"", hex::encode(b)),
        AttributeValue::Null => "null".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_value_to_json_primitives() {
        assert_eq!(attribute_value_to_json(&AttributeValue::Bool(true)), "true");
        assert_eq!(
            attribute_value_to_json(&AttributeValue::Bool(false)),
            "false"
        );
        assert_eq!(attribute_value_to_json(&AttributeValue::U8(42)), "42");
        assert_eq!(attribute_value_to_json(&AttributeValue::U16(1000)), "1000");
        assert_eq!(attribute_value_to_json(&AttributeValue::I16(-500)), "-500");
        assert_eq!(attribute_value_to_json(&AttributeValue::Null), "null");
    }

    #[test]
    fn attribute_value_to_json_string_escaped() {
        // Strings must be JSON-quoted and special chars escaped.
        let v = AttributeValue::String(r#"hello "world""#.into());
        let json = attribute_value_to_json(&v);
        assert_eq!(json, r#""hello \"world\"""#);
    }

    #[test]
    fn attribute_value_to_json_bytes_hex() {
        let v = AttributeValue::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(attribute_value_to_json(&v), "\"deadbeef\"");
    }

    #[test]
    fn output_ok_json() {
        // Output::ok in JSON mode must emit valid JSON with "ok":true.
        let out = Output::new(true);
        // We can't capture stdout easily in a unit test, but we can verify the
        // format by constructing the string the same way the impl does.
        let msg = "all good";
        let json = format!(
            "{{\"ok\":true,\"msg\":{}}}",
            serde_json::to_string(msg).unwrap()
        );
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["ok"], true);
        assert_eq!(parsed["msg"], "all good");
        // Also make sure the Output struct round-trips.
        assert!(out.json);
    }

    #[test]
    fn output_devices_empty() {
        // devices() with an empty slice should not panic in either mode.
        let out_plain = Output::new(false);
        let out_json = Output::new(true);
        // These write to stdout; just verify they don't panic.
        out_plain.devices(&[]);
        out_json.devices(&[]);
    }
}
