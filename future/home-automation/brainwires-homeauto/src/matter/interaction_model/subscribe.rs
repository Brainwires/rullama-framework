use super::super::clusters::read_u16_le;
/// Matter IM Subscribe interactions.
///
/// Implements TLV encode/decode for:
/// - `SubscribeRequest`  (opcode 0x03) — set up a subscription.
/// - `SubscribeResponse` (opcode 0x04) — confirm subscription parameters.
///
/// TLV layout (Matter spec §8.5):
///
/// SubscribeRequest
/// ```text
/// struct {
///   tag 0: bool               // keep_subscriptions
///   tag 1: uint16             // min_interval_floor_seconds
///   tag 2: uint16             // max_interval_ceiling_seconds
///   tag 3: list { AttributePath... }  // attribute_requests
///   tag 4: list { AttributePath... }  // event_requests (simplified)
///   tag 7: bool               // fabric_filtered
/// }
/// ```
///
/// SubscribeResponse
/// ```text
/// struct {
///   tag 0: uint32  // subscription_id
///   tag 2: uint16  // max_interval (negotiated)
/// }
/// ```
use super::super::clusters::{
    AttributePath, tlv, tlv_bool, tlv_uint16, tlv_uint32, wrap_list_tagged, wrap_struct,
};
use super::super::error::{MatterError, MatterResult};

// ── SubscribeRequest ──────────────────────────────────────────────────────────

/// Request to establish or refresh a subscription (opcode 0x03).
#[derive(Debug, Clone)]
pub struct SubscribeRequest {
    /// When `true`, preserve existing subscriptions on reconnect.
    pub keep_subscriptions: bool,
    /// Minimum reporting interval floor (seconds).
    pub min_interval_floor_seconds: u16,
    /// Maximum reporting interval ceiling (seconds).
    pub max_interval_ceiling_seconds: u16,
    /// Attribute paths to subscribe to.
    pub attribute_requests: Vec<AttributePath>,
    /// Event paths to subscribe to (simplified — uses `AttributePath` type).
    pub event_requests: Vec<AttributePath>,
    /// When `true`, only report attributes visible to the accessing fabric.
    pub fabric_filtered: bool,
}

impl SubscribeRequest {
    /// TLV-encode the `SubscribeRequest`.
    ///
    /// Layout:
    /// ```text
    /// struct {
    ///   tag 0: bool   (keep_subscriptions)
    ///   tag 1: uint16 (min_interval_floor_seconds)
    ///   tag 2: uint16 (max_interval_ceiling_seconds)
    ///   tag 3: list { AttributePath... }  (attribute_requests)
    ///   tag 4: list { AttributePath... }  (event_requests)
    ///   tag 7: bool   (fabric_filtered)
    /// }
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        let mut inner = Vec::new();
        inner.extend_from_slice(&tlv_bool(0, self.keep_subscriptions));
        inner.extend_from_slice(&tlv_uint16(1, self.min_interval_floor_seconds));
        inner.extend_from_slice(&tlv_uint16(2, self.max_interval_ceiling_seconds));

        let mut attr_inner = Vec::new();
        for path in &self.attribute_requests {
            attr_inner.extend_from_slice(&path.encode());
        }
        inner.extend_from_slice(&wrap_list_tagged(3, &attr_inner));

        let mut evt_inner = Vec::new();
        for path in &self.event_requests {
            evt_inner.extend_from_slice(&path.encode());
        }
        inner.extend_from_slice(&wrap_list_tagged(4, &evt_inner));

        inner.extend_from_slice(&tlv_bool(7, self.fabric_filtered));
        wrap_struct(&inner)
    }

    /// Decode a `SubscribeRequest` from TLV bytes.
    pub fn decode(bytes: &[u8]) -> MatterResult<Self> {
        if bytes.is_empty() || bytes[0] != tlv::TYPE_STRUCTURE {
            return Err(MatterError::Transport(
                "SubscribeRequest: expected structure".into(),
            ));
        }
        let mut keep_subscriptions = false;
        let mut min_interval_floor_seconds = 0u16;
        let mut max_interval_ceiling_seconds = 0u16;
        let mut attribute_requests = Vec::new();
        let mut event_requests = Vec::new();
        let mut fabric_filtered = false;
        let mut i = 1;

        while i < bytes.len() {
            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                break;
            }
            if i + 1 >= bytes.len() {
                return Err(MatterError::Transport("SubscribeRequest: truncated".into()));
            }
            let ctrl = bytes[i];
            let tag = bytes[i + 1];
            i += 2;
            let type_bits = ctrl & 0x1F;

            match (tag, type_bits) {
                (0, t) if t == tlv::TYPE_BOOL_TRUE || t == tlv::TYPE_BOOL_FALSE => {
                    keep_subscriptions = t == tlv::TYPE_BOOL_TRUE;
                }
                (1, t) if t == tlv::TYPE_UNSIGNED_INT_2 => {
                    let (v, next) = read_u16_le(bytes, i).ok_or_else(|| {
                        MatterError::Transport("SubscribeRequest: bad min_interval".into())
                    })?;
                    min_interval_floor_seconds = v;
                    i = next;
                }
                (2, t) if t == tlv::TYPE_UNSIGNED_INT_2 => {
                    let (v, next) = read_u16_le(bytes, i).ok_or_else(|| {
                        MatterError::Transport("SubscribeRequest: bad max_interval".into())
                    })?;
                    max_interval_ceiling_seconds = v;
                    i = next;
                }
                (3, t) if t == tlv::TYPE_LIST => {
                    decode_path_list(bytes, &mut i, &mut attribute_requests).map_err(|e| {
                        MatterError::Transport(format!("SubscribeRequest attr_list: {e}"))
                    })?;
                }
                (4, t) if t == tlv::TYPE_LIST => {
                    decode_path_list(bytes, &mut i, &mut event_requests).map_err(|e| {
                        MatterError::Transport(format!("SubscribeRequest evt_list: {e}"))
                    })?;
                }
                (7, t) if t == tlv::TYPE_BOOL_TRUE || t == tlv::TYPE_BOOL_FALSE => {
                    fabric_filtered = t == tlv::TYPE_BOOL_TRUE;
                }
                _ => {
                    return Err(MatterError::Transport(format!(
                        "SubscribeRequest: unexpected field tag={tag} ctrl={ctrl:#04x}"
                    )));
                }
            }
        }
        Ok(Self {
            keep_subscriptions,
            min_interval_floor_seconds,
            max_interval_ceiling_seconds,
            attribute_requests,
            event_requests,
            fabric_filtered,
        })
    }
}

/// Shared helper: decode a list of `AttributePath` structs from a TLV list body.
fn decode_path_list(
    bytes: &[u8],
    i: &mut usize,
    out: &mut Vec<AttributePath>,
) -> Result<(), String> {
    while *i < bytes.len() && bytes[*i] != tlv::TYPE_END_OF_CONTAINER {
        if bytes[*i] != tlv::TYPE_STRUCTURE {
            return Err(format!("expected AttributePath struct at byte {}", *i));
        }
        let start = *i;
        *i += 1;
        let mut depth = 1u32;
        while *i < bytes.len() && depth > 0 {
            if bytes[*i] == tlv::TYPE_END_OF_CONTAINER {
                depth -= 1;
            } else if bytes[*i] == tlv::TYPE_STRUCTURE {
                depth += 1;
            }
            *i += 1;
        }
        let path = AttributePath::decode(&bytes[start..*i])
            .ok_or_else(|| "bad AttributePath".to_string())?;
        out.push(path);
    }
    if *i < bytes.len() {
        *i += 1;
    } // consume END_OF_CONTAINER
    Ok(())
}

// ── SubscribeResponse ─────────────────────────────────────────────────────────

/// Subscription confirmation from device → controller (opcode 0x04).
#[derive(Debug, Clone)]
pub struct SubscribeResponse {
    /// Assigned subscription identifier.
    pub subscription_id: u32,
    /// Negotiated maximum reporting interval (seconds).
    pub max_interval: u16,
}

impl SubscribeResponse {
    /// TLV-encode the `SubscribeResponse`.
    ///
    /// Layout:
    /// ```text
    /// struct {
    ///   tag 0: uint32 (subscription_id)
    ///   tag 2: uint16 (max_interval)
    /// }
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        let mut inner = Vec::new();
        inner.extend_from_slice(&tlv_uint32(0, self.subscription_id));
        inner.extend_from_slice(&tlv_uint16(2, self.max_interval));
        wrap_struct(&inner)
    }

    /// Decode a `SubscribeResponse` from TLV bytes.
    pub fn decode(bytes: &[u8]) -> MatterResult<Self> {
        if bytes.is_empty() || bytes[0] != tlv::TYPE_STRUCTURE {
            return Err(MatterError::Transport(
                "SubscribeResponse: expected structure".into(),
            ));
        }
        let mut subscription_id = None::<u32>;
        let mut max_interval = None::<u16>;
        let mut i = 1;

        while i < bytes.len() {
            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                break;
            }
            if i + 1 >= bytes.len() {
                return Err(MatterError::Transport(
                    "SubscribeResponse: truncated".into(),
                ));
            }
            let ctrl = bytes[i];
            let tag = bytes[i + 1];
            i += 2;
            let type_bits = ctrl & 0x1F;

            match (tag, type_bits) {
                (0, t) if t == tlv::TYPE_UNSIGNED_INT_4 => {
                    use super::super::clusters::read_u32_le;
                    let (v, next) = read_u32_le(bytes, i).ok_or_else(|| {
                        MatterError::Transport("SubscribeResponse: bad subscription_id".into())
                    })?;
                    subscription_id = Some(v);
                    i = next;
                }
                (2, t) if t == tlv::TYPE_UNSIGNED_INT_2 => {
                    let (v, next) = read_u16_le(bytes, i).ok_or_else(|| {
                        MatterError::Transport("SubscribeResponse: bad max_interval".into())
                    })?;
                    max_interval = Some(v);
                    i = next;
                }
                _ => {
                    return Err(MatterError::Transport(format!(
                        "SubscribeResponse: unexpected field tag={tag} ctrl={ctrl:#04x}"
                    )));
                }
            }
        }
        Ok(Self {
            subscription_id: subscription_id.ok_or_else(|| {
                MatterError::Transport("SubscribeResponse: missing subscription_id".into())
            })?,
            max_interval: max_interval.ok_or_else(|| {
                MatterError::Transport("SubscribeResponse: missing max_interval".into())
            })?,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::super::clusters::AttributePath;
    use super::*;

    #[test]
    fn subscribe_request_roundtrip() {
        let req = SubscribeRequest {
            keep_subscriptions: false,
            min_interval_floor_seconds: 1,
            max_interval_ceiling_seconds: 30,
            attribute_requests: vec![AttributePath::specific(1, 0x0006, 0x0000)],
            event_requests: vec![],
            fabric_filtered: false,
        };
        let encoded = req.encode();
        let decoded = SubscribeRequest::decode(&encoded).expect("decode failed");
        assert_eq!(decoded.min_interval_floor_seconds, 1);
        assert_eq!(decoded.max_interval_ceiling_seconds, 30);
        assert_eq!(decoded.attribute_requests.len(), 1);
        assert_eq!(
            decoded.attribute_requests[0],
            AttributePath::specific(1, 0x0006, 0x0000)
        );
        assert!(decoded.event_requests.is_empty());
        assert!(!decoded.fabric_filtered);
    }

    #[test]
    fn subscribe_response_roundtrip() {
        let resp = SubscribeResponse {
            subscription_id: 7,
            max_interval: 30,
        };
        let encoded = resp.encode();
        let decoded = SubscribeResponse::decode(&encoded).expect("decode failed");
        assert_eq!(decoded.subscription_id, 7);
        assert_eq!(decoded.max_interval, 30);
    }
}
