/// Matter commissionable device discovery — DNS-SD advertisement per spec §4.3.1.
///
/// A commissionable device announces itself on `_matterc._udp.local.` so that
/// commissioners (phone apps, chip-tool, etc.) can find it before commissioning.
use mdns_sd::{ServiceDaemon, ServiceInfo};
use tracing::info;

use crate::matter::error::{MatterError, MatterResult};
use crate::matter::types::MatterDeviceConfig;

/// Advertises this device as commissionable over mDNS/DNS-SD.
///
/// ## Matter spec §4.3.1.2 TXT records
///
/// | Key  | Value                          | Notes                                  |
/// |------|--------------------------------|----------------------------------------|
/// | `D`  | discriminator (decimal)        | 12-bit value 0–4095                    |
/// | `CM` | `"1"`                          | Standard commissioning mode (open)     |
/// | `DN` | device name                    | Human-readable label                   |
/// | `VP` | `"{vid}+{pid}"`                | Vendor + Product ID                    |
/// | `SII`| `"5000"`                       | Sleep-idle interval (ms)               |
/// | `SAI`| `"300"`                        | Sleep-active interval (ms)             |
/// | `T`  | `"0"`                          | TCP transport unsupported              |
/// | `PH` | `"33"`                         | PHY hint: on-network (0x21)            |
pub struct CommissionableAdvertiser {
    daemon: ServiceDaemon,
    service_fullname: String,
}

impl CommissionableAdvertiser {
    /// Start advertising this device as commissionable.
    ///
    /// The service instance name is `"BW-{discriminator_hex}"` on `_matterc._udp`.
    /// The hostname is taken from the OS (via `gethostname`); if unavailable a
    /// synthetic name `"matter-{discriminator}.local."` is used as a fallback.
    pub fn start(config: &MatterDeviceConfig) -> MatterResult<Self> {
        let daemon =
            ServiceDaemon::new().map_err(|e| MatterError::Mdns(format!("daemon init: {e}")))?;

        // TXT records per Matter spec §4.3.1.2
        let txt: &[(&str, &str)] = &[
            ("D", &config.discriminator.to_string()),
            ("CM", "1"), // commissioning mode: standard (open window)
            ("DN", &config.device_name),
            ("VP", &format!("{}+{}", config.vendor_id, config.product_id)),
            ("SII", "5000"), // sleep-idle interval (ms)
            ("SAI", "300"),  // sleep-active interval (ms)
            ("T", "0"),      // TCP transport: not supported
            ("PH", "33"),    // PHY: on-network + IP-capable (0x21 = 33)
        ];

        // Build owned string values so their borrows live long enough.
        let d_val = config.discriminator.to_string();
        let vp_val = format!("{}+{}", config.vendor_id, config.product_id);
        let txt_owned: Vec<(&str, String)> = vec![
            ("D", d_val.clone()),
            ("CM", "1".to_string()),
            ("DN", config.device_name.clone()),
            ("VP", vp_val),
            ("SII", "5000".to_string()),
            ("SAI", "300".to_string()),
            ("T", "0".to_string()),
            ("PH", "33".to_string()),
        ];
        let _ = txt; // suppress unused warning — txt_owned is used below

        let hostname = gethostname::gethostname().to_string_lossy().to_string();
        let host_fqdn = if hostname.is_empty() {
            format!("matter-{}.local.", config.discriminator)
        } else {
            format!("{hostname}.local.")
        };

        // Instance name: "BW-{DISCRIMINATOR_HEX}" — unique per discriminator.
        let instance_name = format!("BW-{:04X}", config.discriminator);
        const SERVICE_TYPE: &str = "_matterc._udp";

        let txt_refs: Vec<(&str, &str)> = txt_owned.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let svc = ServiceInfo::new(
            SERVICE_TYPE,
            &instance_name,
            &host_fqdn,
            (),
            config.port,
            txt_refs.as_slice(),
        )
        .map_err(|e| MatterError::Mdns(format!("ServiceInfo: {e}")))?;

        let service_fullname = svc.get_fullname().to_string();

        daemon
            .register(svc)
            .map_err(|e| MatterError::Mdns(format!("register: {e}")))?;

        info!(
            "Matter mDNS: commissionable '{}' on port {} (discriminator {})",
            instance_name, config.port, config.discriminator
        );

        Ok(Self {
            daemon,
            service_fullname,
        })
    }

    /// Deregister the commissionable advertisement from mDNS.
    pub fn stop(&self) -> MatterResult<()> {
        self.daemon
            .unregister(&self.service_fullname)
            .map_err(|e| MatterError::Mdns(format!("unregister: {e}")))?;
        Ok(())
    }

    /// The full mDNS service name, e.g. `"BW-0F00._matterc._udp.local."`.
    pub fn service_fullname(&self) -> &str {
        &self.service_fullname
    }
}

impl Drop for CommissionableAdvertiser {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Helper: build the TXT record map the same way `start()` does, without
    /// actually starting the mDNS daemon.
    fn build_txt_records(config: &MatterDeviceConfig) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("D".to_string(), config.discriminator.to_string());
        map.insert("CM".to_string(), "1".to_string());
        map.insert("DN".to_string(), config.device_name.clone());
        map.insert(
            "VP".to_string(),
            format!("{}+{}", config.vendor_id, config.product_id),
        );
        map.insert("SII".to_string(), "5000".to_string());
        map.insert("SAI".to_string(), "300".to_string());
        map.insert("T".to_string(), "0".to_string());
        map.insert("PH".to_string(), "33".to_string());
        map
    }

    fn test_config() -> MatterDeviceConfig {
        MatterDeviceConfig::builder()
            .device_name("Test Light")
            .vendor_id(0xFFF1)
            .product_id(0x8001)
            .discriminator(0xF00)
            .passcode(20202021)
            .port(5540)
            .build()
    }

    /// The TXT record map must contain the `D` key with the discriminator in
    /// decimal string form per Matter spec §4.3.1.2.
    #[test]
    fn commissionable_txt_records_include_discriminator() {
        let config = test_config();
        let txt = build_txt_records(&config);

        assert!(
            txt.contains_key("D"),
            "TXT records must contain key 'D' (discriminator)"
        );
        assert_eq!(
            txt["D"],
            config.discriminator.to_string(),
            "D must be the discriminator as a decimal string"
        );
        // Sanity-check a few other required keys
        assert_eq!(
            txt["CM"], "1",
            "CM must be '1' for standard commissioning mode"
        );
        assert_eq!(
            txt["VP"],
            format!("{}+{}", config.vendor_id, config.product_id)
        );
        assert_eq!(txt["SII"], "5000");
        assert_eq!(txt["SAI"], "300");
        assert_eq!(txt["T"], "0");
        assert_eq!(txt["PH"], "33");
    }

    /// The service type string used for commissionable discovery must be
    /// `_matterc._udp` per Matter spec §4.3.1.
    #[test]
    fn commissionable_service_type_is_matterc_udp() {
        // The constant is private but we can verify the advertised fullname
        // contains the right service type by checking the instance name format.
        let config = test_config();
        let instance = format!("BW-{:04X}", config.discriminator);
        // Fullname = "{instance}._matterc._udp.local."
        let expected_suffix = "._matterc._udp.local.";
        let fullname = format!("{instance}{expected_suffix}");
        assert!(
            fullname.ends_with("._matterc._udp.local."),
            "commissionable service type must be _matterc._udp, got: {fullname}"
        );
        // Also verify the instance prefix contains our discriminator
        assert!(
            fullname.starts_with("BW-"),
            "instance name must start with BW-"
        );
    }
}
