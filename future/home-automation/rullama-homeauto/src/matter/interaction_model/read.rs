/// Matter IM Read and Report interactions.
///
/// Implements TLV encode/decode for:
/// - `ReadRequest` (opcode 0x02) — ask for one or more attributes.
/// - `ReportData`  (opcode 0x05) — server reply carrying attribute values.
///
/// TLV layout (Matter spec §8.6):
///
/// ReadRequest
/// ```text
/// struct {
///   tag 0: list of AttributePath  // attribute_requests
///   tag 3: bool                   // fabric_filtered
/// }
/// ```
///
/// ReportData
/// ```text
/// struct {
///   tag 0: uint32?                // subscription_id (present if subscribed)
///   tag 1: list of {              // attribute_reports
///     tag 1: struct {             //   AttributeDataIB
///       tag 0: AttributePath
///       tag 1: <raw TLV>          //   attribute value
///     }
///   }
///   tag 4: bool                   // suppress_response
/// }
/// ```
use super::super::clusters::{
    AttributePath, tlv, tlv_bool, tlv_uint32, wrap_list_tagged, wrap_struct, wrap_struct_tagged,
};
use super::super::error::{MatterError, MatterResult};

// ── ReadRequest ───────────────────────────────────────────────────────────────

/// Request to read one or more attributes (opcode 0x02).
#[derive(Debug, Clone)]
pub struct ReadRequest {
    /// The attribute paths to read.  Use [`AttributePath::wildcard`] to read all.
    pub attribute_requests: Vec<AttributePath>,
    /// When `true`, only return attributes visible to the accessing fabric.
    pub fabric_filtered: bool,
}

impl ReadRequest {
    /// Construct a new read request for the given paths.
    pub fn new(paths: Vec<AttributePath>) -> Self {
        Self {
            attribute_requests: paths,
            fabric_filtered: false,
        }
    }

    /// TLV-encode the `ReadRequest`.
    ///
    /// Layout:
    /// ```text
    /// struct {
    ///   tag 0: list { AttributePath... }
    ///   tag 3: bool  (fabric_filtered)
    /// }
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        // Encode attribute path list (tag 0)
        let mut paths_inner = Vec::new();
        for path in &self.attribute_requests {
            // Each path is an anonymous-tagged struct inside the list.
            let path_bytes = path.encode();
            paths_inner.extend_from_slice(&path_bytes);
        }
        let paths_list = wrap_list_tagged(0, &paths_inner);

        // fabric_filtered (tag 3)
        let ff = tlv_bool(3, self.fabric_filtered);

        let mut inner = paths_list;
        inner.extend_from_slice(&ff);
        wrap_struct(&inner)
    }

    /// Decode a `ReadRequest` from TLV bytes.
    pub fn decode(bytes: &[u8]) -> MatterResult<Self> {
        if bytes.is_empty() || bytes[0] != tlv::TYPE_STRUCTURE {
            return Err(MatterError::Transport(
                "ReadRequest: expected structure".into(),
            ));
        }
        let mut attribute_requests = Vec::new();
        let mut fabric_filtered = false;

        let mut i = 1;
        while i < bytes.len() {
            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                break;
            }
            if i + 1 >= bytes.len() {
                return Err(MatterError::Transport("ReadRequest: truncated".into()));
            }
            let ctrl = bytes[i];
            let tag = bytes[i + 1];
            i += 2;
            let type_bits = ctrl & 0x1F;

            match (tag, type_bits) {
                (0, t) if t == tlv::TYPE_LIST => {
                    // Read list of AttributePaths
                    while i < bytes.len() && bytes[i] != tlv::TYPE_END_OF_CONTAINER {
                        // Each path is a structure
                        let start = i;
                        if bytes[i] != tlv::TYPE_STRUCTURE {
                            return Err(MatterError::Transport(
                                "ReadRequest: expected path struct".into(),
                            ));
                        }
                        // Scan to matching END_OF_CONTAINER
                        i += 1;
                        let mut depth = 1u32;
                        while i < bytes.len() && depth > 0 {
                            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                                depth -= 1;
                            } else if bytes[i] == tlv::TYPE_STRUCTURE
                                || bytes[i] == tlv::TYPE_LIST
                                || bytes[i] == (tlv::TAG_CONTEXT_1 | tlv::TYPE_STRUCTURE)
                                || bytes[i] == (tlv::TAG_CONTEXT_1 | tlv::TYPE_LIST)
                            {
                                depth += 1;
                            }
                            i += 1;
                        }
                        let path = AttributePath::decode(&bytes[start..i]).ok_or_else(|| {
                            MatterError::Transport("ReadRequest: bad AttributePath".into())
                        })?;
                        attribute_requests.push(path);
                    }
                    if i < bytes.len() {
                        i += 1;
                    } // consume END_OF_CONTAINER for list
                }
                (3, t) if t == tlv::TYPE_BOOL_TRUE || t == tlv::TYPE_BOOL_FALSE => {
                    fabric_filtered = t == tlv::TYPE_BOOL_TRUE;
                }
                _ => {
                    return Err(MatterError::Transport(format!(
                        "ReadRequest: unexpected field tag={tag} ctrl={ctrl:#04x}"
                    )));
                }
            }
        }
        Ok(Self {
            attribute_requests,
            fabric_filtered,
        })
    }
}

// ── AttributeData ─────────────────────────────────────────────────────────────

/// An attribute path + its raw TLV-encoded value.
#[derive(Debug, Clone)]
pub struct AttributeData {
    /// Path identifying which attribute this data belongs to.
    pub path: AttributePath,
    /// Raw TLV-encoded attribute value (opaque bytes).
    pub data: Vec<u8>,
}

impl AttributeData {
    /// TLV-encode as an AttributeDataIB struct.
    ///
    /// Layout:
    /// ```text
    /// struct {
    ///   tag 0: AttributePath (struct)
    ///   tag 1: <raw data bytes>
    /// }
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        let path_tagged =
            wrap_struct_tagged(0, &self.path.encode()[1..self.path.encode().len() - 1]);
        // tag 1: raw data — wrap data bytes as a context-tagged anonymous blob
        // We treat data as a pre-encoded TLV octet — embed verbatim with tag 1 prefix
        let data_tagged = {
            let mut v = Vec::new();
            // Wrap data in a context-tagged structure shell (tag 1)
            v.push(tlv::TAG_CONTEXT_1 | tlv::TYPE_STRUCTURE);
            v.push(1u8);
            v.extend_from_slice(&self.data);
            v.push(tlv::TYPE_END_OF_CONTAINER);
            v
        };
        let mut inner = path_tagged;
        inner.extend_from_slice(&data_tagged);
        wrap_struct(&inner)
    }

    /// Decode an `AttributeData` from TLV bytes produced by [`AttributeData::encode`].
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() || bytes[0] != tlv::TYPE_STRUCTURE {
            return None;
        }
        let mut path: Option<AttributePath> = None;
        let mut data: Option<Vec<u8>> = None;
        let mut i = 1;

        while i < bytes.len() {
            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                break;
            }
            if i + 1 >= bytes.len() {
                return None;
            }
            let ctrl = bytes[i];
            let tag = bytes[i + 1];
            i += 2;
            let type_bits = ctrl & 0x1F;

            match (tag, type_bits) {
                (0, t) if t == tlv::TYPE_STRUCTURE => {
                    // Reconstruct AttributePath — starts at (i-2), but we skipped past ctrl+tag.
                    // We need to read the inner struct body and wrap it.
                    let start = i;
                    let mut depth = 1u32;
                    while i < bytes.len() && depth > 0 {
                        if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                            depth -= 1;
                        } else if bytes[i] == tlv::TYPE_STRUCTURE {
                            depth += 1;
                        }
                        i += 1;
                    }
                    // Reconstruct: prepend TYPE_STRUCTURE, the body is bytes[start..i-1], then END
                    let mut path_bytes = vec![tlv::TYPE_STRUCTURE];
                    path_bytes.extend_from_slice(&bytes[start..i - 1]);
                    path_bytes.push(tlv::TYPE_END_OF_CONTAINER);
                    path = AttributePath::decode(&path_bytes);
                }
                (1, t) if t == tlv::TYPE_STRUCTURE => {
                    // Raw data is the contents of this struct
                    let start = i;
                    let mut depth = 1u32;
                    while i < bytes.len() && depth > 0 {
                        if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                            depth -= 1;
                        } else if bytes[i] == tlv::TYPE_STRUCTURE {
                            depth += 1;
                        }
                        i += 1;
                    }
                    data = Some(bytes[start..i - 1].to_vec());
                }
                _ => return None,
            }
        }
        Some(Self {
            path: path?,
            data: data.unwrap_or_default(),
        })
    }
}

// ── ReportData ────────────────────────────────────────────────────────────────

/// Attribute report from device → controller (opcode 0x05).
#[derive(Debug, Clone)]
pub struct ReportData {
    /// Present when this report is part of a subscription.
    pub subscription_id: Option<u32>,
    /// The reported attribute values.
    pub attribute_reports: Vec<AttributeData>,
    /// When `true`, the controller must not send a `StatusResponse`.
    pub suppress_response: bool,
}

impl ReportData {
    /// TLV-encode the `ReportData`.
    ///
    /// Layout:
    /// ```text
    /// struct {
    ///   tag 0: uint32?          (subscription_id)
    ///   tag 1: list { AttributeData... }
    ///   tag 4: bool             (suppress_response)
    /// }
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        let mut inner = Vec::new();
        if let Some(sid) = self.subscription_id {
            inner.extend_from_slice(&tlv_uint32(0, sid));
        }
        let mut reports_inner = Vec::new();
        for attr in &self.attribute_reports {
            reports_inner.extend_from_slice(&attr.encode());
        }
        inner.extend_from_slice(&wrap_list_tagged(1, &reports_inner));
        inner.extend_from_slice(&tlv_bool(4, self.suppress_response));
        wrap_struct(&inner)
    }

    /// Decode a `ReportData` from TLV bytes.
    pub fn decode(bytes: &[u8]) -> MatterResult<Self> {
        if bytes.is_empty() || bytes[0] != tlv::TYPE_STRUCTURE {
            return Err(MatterError::Transport(
                "ReportData: expected structure".into(),
            ));
        }
        let mut subscription_id = None;
        let mut attribute_reports = Vec::new();
        let mut suppress_response = false;
        let mut i = 1;

        while i < bytes.len() {
            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                break;
            }
            if i + 1 >= bytes.len() {
                return Err(MatterError::Transport("ReportData: truncated".into()));
            }
            let ctrl = bytes[i];
            let tag = bytes[i + 1];
            i += 2;
            let type_bits = ctrl & 0x1F;

            match (tag, type_bits) {
                (0, t) if t == tlv::TYPE_UNSIGNED_INT_4 => {
                    use super::super::clusters::read_u32_le;
                    let (v, next) = read_u32_le(bytes, i).ok_or_else(|| {
                        MatterError::Transport("ReportData: bad subscription_id".into())
                    })?;
                    subscription_id = Some(v);
                    i = next;
                }
                (1, t) if t == tlv::TYPE_LIST => {
                    // Read list of AttributeData structs
                    while i < bytes.len() && bytes[i] != tlv::TYPE_END_OF_CONTAINER {
                        if bytes[i] != tlv::TYPE_STRUCTURE {
                            return Err(MatterError::Transport(
                                "ReportData: expected AttributeData struct".into(),
                            ));
                        }
                        let start = i;
                        i += 1;
                        let mut depth = 1u32;
                        while i < bytes.len() && depth > 0 {
                            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                                depth -= 1;
                            } else if bytes[i] == tlv::TYPE_STRUCTURE
                                || bytes[i] == (tlv::TAG_CONTEXT_1 | tlv::TYPE_STRUCTURE)
                            {
                                depth += 1;
                            }
                            i += 1;
                        }
                        let attr = AttributeData::decode(&bytes[start..i]).ok_or_else(|| {
                            MatterError::Transport("ReportData: bad AttributeData".into())
                        })?;
                        attribute_reports.push(attr);
                    }
                    if i < bytes.len() {
                        i += 1;
                    } // consume END_OF_CONTAINER
                }
                (4, t) if t == tlv::TYPE_BOOL_TRUE || t == tlv::TYPE_BOOL_FALSE => {
                    suppress_response = t == tlv::TYPE_BOOL_TRUE;
                }
                _ => {
                    return Err(MatterError::Transport(format!(
                        "ReportData: unexpected field tag={tag} ctrl={ctrl:#04x}"
                    )));
                }
            }
        }
        Ok(Self {
            subscription_id,
            attribute_reports,
            suppress_response,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::super::clusters::AttributePath;
    use super::*;

    #[test]
    fn read_request_single_path_roundtrip() {
        let path = AttributePath::specific(1, 0x0006, 0x0000);
        let req = ReadRequest::new(vec![path.clone()]);
        let encoded = req.encode();
        let decoded = ReadRequest::decode(&encoded).expect("decode failed");
        assert_eq!(decoded.attribute_requests.len(), 1);
        assert_eq!(decoded.attribute_requests[0], path);
        assert!(!decoded.fabric_filtered);
    }

    #[test]
    fn report_data_multiple_attributes_roundtrip() {
        let reports = vec![
            AttributeData {
                path: AttributePath::specific(1, 0x0006, 0x0000),
                data: vec![tlv::TYPE_BOOL_TRUE],
            },
            AttributeData {
                path: AttributePath::specific(1, 0x0008, 0x0000),
                data: vec![tlv::TAG_CONTEXT_1 | tlv::TYPE_UNSIGNED_INT_1, 0, 128],
            },
        ];
        let report = ReportData {
            subscription_id: Some(42),
            attribute_reports: reports,
            suppress_response: false,
        };
        let encoded = report.encode();
        let decoded = ReportData::decode(&encoded).expect("decode failed");
        assert_eq!(decoded.subscription_id, Some(42));
        assert_eq!(decoded.attribute_reports.len(), 2);
        assert_eq!(
            decoded.attribute_reports[0].path,
            AttributePath::specific(1, 0x0006, 0x0000)
        );
        assert_eq!(
            decoded.attribute_reports[1].path,
            AttributePath::specific(1, 0x0008, 0x0000)
        );
        assert!(!decoded.suppress_response);
    }
}
