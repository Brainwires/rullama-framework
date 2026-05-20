/// Matter IM Write interactions.
///
/// Implements TLV encode/decode for:
/// - `WriteRequest`  (opcode 0x06) — write one or more attributes.
/// - `WriteResponse` (opcode 0x07) — per-attribute status results.
///
/// TLV layout (Matter spec §8.7):
///
/// WriteRequest
/// ```text
/// struct {
///   tag 0: bool                   // suppress_response
///   tag 1: bool                   // timed_request
///   tag 2: list of AttributeData  // write_requests
/// }
/// ```
///
/// WriteResponse
/// ```text
/// struct {
///   tag 0: list of AttributeStatus  // write_responses
/// }
/// ```
use super::super::clusters::{
    AttributePath, tlv, tlv_bool, tlv_uint8, wrap_list_tagged, wrap_struct,
};
use super::super::error::{MatterError, MatterResult};
use super::read::AttributeData;

// ── InteractionStatus ─────────────────────────────────────────────────────────

/// Status codes used in IM responses (Matter spec §8.10.22).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionStatus {
    /// `0x00` — Operation succeeded.
    Success = 0x00,
    /// `0x01` — Operation failed with no more specific code.
    Failure = 0x01,
    /// `0x7d` — Subscription ID is unknown.
    InvalidSubscription = 0x7d,
    /// `0x7e` — Caller's privilege level is insufficient.
    UnsupportedAccess = 0x7e,
    /// `0x7f` — Requested endpoint does not exist.
    UnsupportedEndpoint = 0x7f,
    /// `0x80` — Action is not permitted in the current state.
    InvalidAction = 0x80,
    /// `0x81` — Cluster does not support this command ID.
    UnsupportedCommand = 0x81,
    /// `0x85` — Command invocation payload is malformed.
    InvalidCommand = 0x85,
    /// `0x86` — Cluster does not support this attribute ID.
    UnsupportedAttribute = 0x86,
    /// `0x87` — Value violated an attribute constraint.
    ConstraintError = 0x87,
    /// `0x88` — Attribute is read-only.
    UnsupportedWrite = 0x88,
    /// `0x89` — Resource limit (e.g. subscription count) reached.
    ResourceExhausted = 0x89,
    /// `0x8b` — Requested entry does not exist.
    NotFound = 0x8b,
    /// `0x8c` — Attribute cannot be subscribed to.
    UnreportableAttribute = 0x8c,
    /// `0x8d` — Value's TLV type does not match the attribute schema.
    InvalidDataType = 0x8d,
    /// `0x8f` — Attribute is not readable.
    UnsupportedRead = 0x8f,
    /// `0x92` — Submitted DataVersion did not match the current value.
    DataVersionMismatch = 0x92,
    /// `0x94` — Operation took too long to complete.
    Timeout = 0x94,
    /// `0x9c` — Node is busy and cannot process the request right now.
    Busy = 0x9c,
    /// `0xc3` — Endpoint does not support this cluster.
    UnsupportedCluster = 0xc3,
    /// `0xc5` — Proxy has no upstream subscription to fulfil the request.
    NoUpstreamSubscription = 0xc5,
    /// `0xc6` — Command must be sent inside a TimedRequest envelope.
    NeedsTimedInteraction = 0xc6,
    /// `0xc7` — Cluster does not support this event ID.
    UnsupportedEvent = 0xc7,
    /// `0xc8` — Too many paths submitted in a single interaction.
    PathsExhausted = 0xc8,
    /// `0xc9` — TimedRequest deadline does not match the follow-up action.
    TimedRequestMismatch = 0xc9,
    /// `0xca` — Operation requires an armed FailSafe window.
    FailsafeRequired = 0xca,
}

impl InteractionStatus {
    /// Try to convert a raw `u8` to an `InteractionStatus`.
    pub fn from_u8(v: u8) -> Option<Self> {
        use InteractionStatus::*;
        match v {
            0x00 => Some(Success),
            0x01 => Some(Failure),
            0x7d => Some(InvalidSubscription),
            0x7e => Some(UnsupportedAccess),
            0x7f => Some(UnsupportedEndpoint),
            0x80 => Some(InvalidAction),
            0x81 => Some(UnsupportedCommand),
            0x85 => Some(InvalidCommand),
            0x86 => Some(UnsupportedAttribute),
            0x87 => Some(ConstraintError),
            0x88 => Some(UnsupportedWrite),
            0x89 => Some(ResourceExhausted),
            0x8b => Some(NotFound),
            0x8c => Some(UnreportableAttribute),
            0x8d => Some(InvalidDataType),
            0x8f => Some(UnsupportedRead),
            0x92 => Some(DataVersionMismatch),
            0x94 => Some(Timeout),
            0x9c => Some(Busy),
            0xc3 => Some(UnsupportedCluster),
            0xc5 => Some(NoUpstreamSubscription),
            0xc6 => Some(NeedsTimedInteraction),
            0xc7 => Some(UnsupportedEvent),
            0xc8 => Some(PathsExhausted),
            0xc9 => Some(TimedRequestMismatch),
            0xca => Some(FailsafeRequired),
            _ => None,
        }
    }
}

// ── WriteRequest ──────────────────────────────────────────────────────────────

/// Request to write one or more attributes (opcode 0x06).
#[derive(Debug, Clone)]
pub struct WriteRequest {
    /// When `true`, the device must not send a `WriteResponse`.
    pub suppress_response: bool,
    /// When `true`, this write was preceded by a `TimedRequest`.
    pub timed_request: bool,
    /// Attribute path + value pairs to write.
    pub write_requests: Vec<AttributeData>,
}

impl WriteRequest {
    /// TLV-encode the `WriteRequest`.
    ///
    /// Layout:
    /// ```text
    /// struct {
    ///   tag 0: bool (suppress_response)
    ///   tag 1: bool (timed_request)
    ///   tag 2: list { AttributeData... }
    /// }
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        let mut inner = Vec::new();
        inner.extend_from_slice(&tlv_bool(0, self.suppress_response));
        inner.extend_from_slice(&tlv_bool(1, self.timed_request));
        let mut list_inner = Vec::new();
        for attr in &self.write_requests {
            list_inner.extend_from_slice(&attr.encode());
        }
        inner.extend_from_slice(&wrap_list_tagged(2, &list_inner));
        wrap_struct(&inner)
    }

    /// Decode a `WriteRequest` from TLV bytes.
    pub fn decode(bytes: &[u8]) -> MatterResult<Self> {
        if bytes.is_empty() || bytes[0] != tlv::TYPE_STRUCTURE {
            return Err(MatterError::Transport(
                "WriteRequest: expected structure".into(),
            ));
        }
        let mut suppress_response = false;
        let mut timed_request = false;
        let mut write_requests = Vec::new();
        let mut i = 1;

        while i < bytes.len() {
            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                break;
            }
            if i + 1 >= bytes.len() {
                return Err(MatterError::Transport("WriteRequest: truncated".into()));
            }
            let ctrl = bytes[i];
            let tag = bytes[i + 1];
            i += 2;
            let type_bits = ctrl & 0x1F;

            match (tag, type_bits) {
                (0, t) if t == tlv::TYPE_BOOL_TRUE || t == tlv::TYPE_BOOL_FALSE => {
                    suppress_response = t == tlv::TYPE_BOOL_TRUE;
                }
                (1, t) if t == tlv::TYPE_BOOL_TRUE || t == tlv::TYPE_BOOL_FALSE => {
                    timed_request = t == tlv::TYPE_BOOL_TRUE;
                }
                (2, t) if t == tlv::TYPE_LIST => {
                    while i < bytes.len() && bytes[i] != tlv::TYPE_END_OF_CONTAINER {
                        if bytes[i] != tlv::TYPE_STRUCTURE {
                            return Err(MatterError::Transport(
                                "WriteRequest: expected AttributeData struct".into(),
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
                            MatterError::Transport("WriteRequest: bad AttributeData".into())
                        })?;
                        write_requests.push(attr);
                    }
                    if i < bytes.len() {
                        i += 1;
                    }
                }
                _ => {
                    return Err(MatterError::Transport(format!(
                        "WriteRequest: unexpected field tag={tag} ctrl={ctrl:#04x}"
                    )));
                }
            }
        }
        Ok(Self {
            suppress_response,
            timed_request,
            write_requests,
        })
    }
}

// ── AttributeStatus ───────────────────────────────────────────────────────────

/// Result for a single attribute in a `WriteResponse`.
#[derive(Debug, Clone)]
pub struct AttributeStatus {
    /// The attribute this status applies to.
    pub path: AttributePath,
    /// The status code.
    pub status: InteractionStatus,
}

impl AttributeStatus {
    /// TLV-encode as a struct:
    /// ```text
    /// struct {
    ///   tag 0: AttributePath (struct)
    ///   tag 1: uint8 (status code)
    /// }
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        let path_enc = self.path.encode();
        // Embed path struct body inside a context-tagged struct shell (tag 0)
        let path_inner = &path_enc[1..path_enc.len() - 1]; // strip outer TYPE_STRUCTURE / END
        let mut inner = Vec::new();
        // context-tagged struct tag 0
        inner.push(tlv::TAG_CONTEXT_1 | tlv::TYPE_STRUCTURE);
        inner.push(0u8);
        inner.extend_from_slice(path_inner);
        inner.push(tlv::TYPE_END_OF_CONTAINER);
        // status code (tag 1)
        inner.extend_from_slice(&tlv_uint8(1, self.status as u8));
        wrap_struct(&inner)
    }

    /// Decode an `AttributeStatus` from TLV bytes.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() || bytes[0] != tlv::TYPE_STRUCTURE {
            return None;
        }
        let mut path: Option<AttributePath> = None;
        let mut status_code: Option<u8> = None;
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
                    let mut path_bytes = vec![tlv::TYPE_STRUCTURE];
                    path_bytes.extend_from_slice(&bytes[start..i - 1]);
                    path_bytes.push(tlv::TYPE_END_OF_CONTAINER);
                    path = AttributePath::decode(&path_bytes);
                }
                (1, t) if t == tlv::TYPE_UNSIGNED_INT_1 => {
                    if i < bytes.len() {
                        status_code = Some(bytes[i]);
                        i += 1;
                    }
                }
                _ => return None,
            }
        }
        Some(Self {
            path: path?,
            status: InteractionStatus::from_u8(status_code?)?,
        })
    }
}

// ── WriteResponse ─────────────────────────────────────────────────────────────

/// Per-attribute status reply from device → controller (opcode 0x07).
#[derive(Debug, Clone)]
pub struct WriteResponse {
    /// Per-path status results.
    pub write_responses: Vec<AttributeStatus>,
}

impl WriteResponse {
    /// TLV-encode the `WriteResponse`.
    ///
    /// Layout:
    /// ```text
    /// struct {
    ///   tag 0: list { AttributeStatus... }
    /// }
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        let mut list_inner = Vec::new();
        for status in &self.write_responses {
            list_inner.extend_from_slice(&status.encode());
        }
        let list = wrap_list_tagged(0, &list_inner);
        wrap_struct(&list)
    }

    /// Decode a `WriteResponse` from TLV bytes.
    pub fn decode(bytes: &[u8]) -> MatterResult<Self> {
        if bytes.is_empty() || bytes[0] != tlv::TYPE_STRUCTURE {
            return Err(MatterError::Transport(
                "WriteResponse: expected structure".into(),
            ));
        }
        let mut write_responses = Vec::new();
        let mut i = 1;

        while i < bytes.len() {
            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                break;
            }
            if i + 1 >= bytes.len() {
                return Err(MatterError::Transport("WriteResponse: truncated".into()));
            }
            let ctrl = bytes[i];
            let tag = bytes[i + 1];
            i += 2;
            let type_bits = ctrl & 0x1F;

            match (tag, type_bits) {
                (0, t) if t == tlv::TYPE_LIST => {
                    while i < bytes.len() && bytes[i] != tlv::TYPE_END_OF_CONTAINER {
                        if bytes[i] != tlv::TYPE_STRUCTURE {
                            return Err(MatterError::Transport(
                                "WriteResponse: expected status struct".into(),
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
                        let st = AttributeStatus::decode(&bytes[start..i]).ok_or_else(|| {
                            MatterError::Transport("WriteResponse: bad AttributeStatus".into())
                        })?;
                        write_responses.push(st);
                    }
                    if i < bytes.len() {
                        i += 1;
                    }
                }
                _ => {
                    return Err(MatterError::Transport(format!(
                        "WriteResponse: unexpected field tag={tag} ctrl={ctrl:#04x}"
                    )));
                }
            }
        }
        Ok(Self { write_responses })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::super::clusters::AttributePath;
    use super::super::read::AttributeData;
    use super::*;

    #[test]
    fn write_request_roundtrip() {
        let req = WriteRequest {
            suppress_response: false,
            timed_request: false,
            write_requests: vec![AttributeData {
                path: AttributePath::specific(1, 0x0006, 0x0000),
                data: vec![tlv::TYPE_BOOL_TRUE],
            }],
        };
        let encoded = req.encode();
        let decoded = WriteRequest::decode(&encoded).expect("decode failed");
        assert!(!decoded.suppress_response);
        assert!(!decoded.timed_request);
        assert_eq!(decoded.write_requests.len(), 1);
        assert_eq!(
            decoded.write_requests[0].path,
            AttributePath::specific(1, 0x0006, 0x0000)
        );
    }

    #[test]
    fn write_response_success_roundtrip() {
        let resp = WriteResponse {
            write_responses: vec![AttributeStatus {
                path: AttributePath::specific(1, 0x0006, 0x0000),
                status: InteractionStatus::Success,
            }],
        };
        let encoded = resp.encode();
        let decoded = WriteResponse::decode(&encoded).expect("decode failed");
        assert_eq!(decoded.write_responses.len(), 1);
        assert_eq!(
            decoded.write_responses[0].status,
            InteractionStatus::Success
        );
        assert_eq!(
            decoded.write_responses[0].path,
            AttributePath::specific(1, 0x0006, 0x0000)
        );
    }
}
