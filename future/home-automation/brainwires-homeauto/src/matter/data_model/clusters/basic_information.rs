//! BasicInformation cluster server (cluster ID 0x0028).
//!
//! Serves device identity attributes such as VendorID, ProductID, and name.
//! Matter spec §11.1.

use async_trait::async_trait;

use crate::matter::clusters::tlv;
use crate::matter::data_model::ClusterServer;
use crate::matter::error::{MatterError, MatterResult};
use crate::matter::types::MatterDeviceConfig;

// ── Attribute IDs ─────────────────────────────────────────────────────────────

/// `0x0000` — DataModelRevision attribute.
pub const ATTR_DATA_MODEL_REVISION: u32 = 0x0000;
/// `0x0001` — VendorName attribute (string).
pub const ATTR_VENDOR_NAME: u32 = 0x0001;
/// `0x0002` — VendorID attribute.
pub const ATTR_VENDOR_ID: u32 = 0x0002;
/// `0x0003` — ProductName attribute (string).
pub const ATTR_PRODUCT_NAME: u32 = 0x0003;
/// `0x0004` — ProductID attribute.
pub const ATTR_PRODUCT_ID: u32 = 0x0004;
/// `0x0005` — NodeLabel attribute (user-assigned name).
pub const ATTR_NODE_LABEL: u32 = 0x0005;
/// `0x0006` — Location attribute (ISO 3166-1 country code).
pub const ATTR_LOCATION: u32 = 0x0006;
/// `0x0007` — HardwareVersion attribute.
pub const ATTR_HARDWARE_VERSION: u32 = 0x0007;
/// `0x000A` — SoftwareVersion attribute (numeric).
pub const ATTR_SOFTWARE_VERSION: u32 = 0x000A;
/// `0x000B` — SoftwareVersionString attribute.
pub const ATTR_SOFTWARE_VERSION_STRING: u32 = 0x000B;
/// `0x000F` — CapabilityMinima attribute (min SDK capability thresholds).
pub const ATTR_CAPABILITY_MINIMA: u32 = 0x000F;

const CLUSTER_ID: u32 = 0x0028;

// ── TLV encoding helpers (local) ──────────────────────────────────────────────

/// Encode a context-tagged uint16 element.
fn tlv_uint16(tag: u8, val: u16) -> Vec<u8> {
    let mut v = vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_2, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

/// Encode a context-tagged uint32 element.
fn tlv_uint32(tag: u8, val: u32) -> Vec<u8> {
    let mut v = vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_4, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

/// Encode a UTF-8 string as a TLV element with a 1-byte length prefix.
///
/// Matter TLV type byte `0x0C` = UTF-8 string, 1-byte length.
fn tlv_utf8_string(tag: u8, s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    // Control: TAG_CONTEXT_1 | 0x0C  (UTF-8 string, 1-byte length)
    let mut v = vec![tlv::TAG_CONTEXT_1 | 0x0C, tag, bytes.len() as u8];
    v.extend_from_slice(bytes);
    v
}

/// Wrap bytes in an anonymous TLV structure.
fn wrap_struct(inner: &[u8]) -> Vec<u8> {
    let mut v = vec![tlv::TYPE_STRUCTURE];
    v.extend_from_slice(inner);
    v.push(tlv::TYPE_END_OF_CONTAINER);
    v
}

// ── BasicInformationCluster ───────────────────────────────────────────────────

/// Server for the BasicInformation cluster (0x0028).
pub struct BasicInformationCluster {
    vendor_name: String,
    vendor_id: u16,
    product_name: String,
    product_id: u16,
    node_label: String,
}

impl BasicInformationCluster {
    /// Create a new server populated from a [`MatterDeviceConfig`].
    pub fn new(config: &MatterDeviceConfig) -> Self {
        Self {
            vendor_name: "Brainwires".to_string(),
            vendor_id: config.vendor_id,
            product_name: config.device_name.clone(),
            product_id: config.product_id,
            node_label: config.device_name.clone(),
        }
    }
}

#[async_trait]
impl ClusterServer for BasicInformationCluster {
    fn cluster_id(&self) -> u32 {
        CLUSTER_ID
    }

    async fn read_attribute(&self, attr_id: u32) -> MatterResult<Vec<u8>> {
        match attr_id {
            ATTR_DATA_MODEL_REVISION => {
                // uint16 value = 1
                Ok(tlv_uint16(0, 1))
            }
            ATTR_VENDOR_NAME => Ok(tlv_utf8_string(0, &self.vendor_name)),
            ATTR_VENDOR_ID => Ok(tlv_uint16(0, self.vendor_id)),
            ATTR_PRODUCT_NAME => Ok(tlv_utf8_string(0, &self.product_name)),
            ATTR_PRODUCT_ID => Ok(tlv_uint16(0, self.product_id)),
            ATTR_NODE_LABEL => Ok(tlv_utf8_string(0, &self.node_label)),
            ATTR_LOCATION => Ok(tlv_utf8_string(0, "XX")),
            ATTR_HARDWARE_VERSION => Ok(tlv_uint16(0, 0)),
            ATTR_SOFTWARE_VERSION => Ok(tlv_uint32(0, 1)),
            ATTR_SOFTWARE_VERSION_STRING => Ok(tlv_utf8_string(0, "1.0.0")),
            ATTR_CAPABILITY_MINIMA => {
                // struct { CaseSessionsPerFabric(0): uint16=3, SubscriptionsPerFabric(1): uint16=3 }
                let mut inner = tlv_uint16(0, 3);
                inner.extend_from_slice(&tlv_uint16(1, 3));
                Ok(wrap_struct(&inner))
            }
            _ => Err(MatterError::Transport("unsupported attribute".into())),
        }
    }

    async fn write_attribute(&self, _attr_id: u32, _value: &[u8]) -> MatterResult<()> {
        Err(MatterError::Transport(
            "BasicInformation attributes are read-only".into(),
        ))
    }

    async fn invoke_command(&self, _cmd_id: u32, _args: &[u8]) -> MatterResult<Vec<u8>> {
        Err(MatterError::Transport(
            "BasicInformation has no commands".into(),
        ))
    }

    fn attribute_ids(&self) -> Vec<u32> {
        vec![
            ATTR_DATA_MODEL_REVISION,
            ATTR_VENDOR_NAME,
            ATTR_VENDOR_ID,
            ATTR_PRODUCT_NAME,
            ATTR_PRODUCT_ID,
            ATTR_NODE_LABEL,
            ATTR_LOCATION,
            ATTR_HARDWARE_VERSION,
            ATTR_SOFTWARE_VERSION,
            ATTR_SOFTWARE_VERSION_STRING,
            ATTR_CAPABILITY_MINIMA,
        ]
    }

    fn command_ids(&self) -> Vec<u32> {
        vec![]
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matter::types::MatterDeviceConfig;

    fn make_cluster() -> BasicInformationCluster {
        let config = MatterDeviceConfig::builder()
            .device_name("Test Light")
            .vendor_id(0xFFF1)
            .product_id(0x8001)
            .build();
        BasicInformationCluster::new(&config)
    }

    #[tokio::test]
    async fn basic_info_vendor_id_attribute_returns_correct_tlv() {
        let cluster = make_cluster();
        let data = cluster
            .read_attribute(ATTR_VENDOR_ID)
            .await
            .expect("VendorID read failed");

        // Expect: [TAG_CONTEXT_1 | TYPE_UNSIGNED_INT_2, 0, 0xF1, 0xFF]
        assert_eq!(data.len(), 4, "VendorID TLV should be 4 bytes");
        // Extract the u16 LE value from bytes 2-3.
        let value = u16::from_le_bytes([data[2], data[3]]);
        assert_eq!(value, 0xFFF1);
    }

    #[tokio::test]
    async fn basic_info_unknown_attribute_returns_error() {
        let cluster = make_cluster();
        let result = cluster.read_attribute(0xFFFF).await;
        assert!(result.is_err(), "Unknown attribute should return an error");
    }
}
