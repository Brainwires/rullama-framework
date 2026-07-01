/// Matter IM Invoke interactions.
///
/// Implements TLV encode/decode for:
/// - `InvokeRequest`  (opcode 0x08) — invoke one or more commands.
/// - `InvokeResponse` (opcode 0x09) — command result(s).
///
/// TLV layout (Matter spec §8.8):
///
/// InvokeRequest
/// ```text
/// struct {
///   tag 0: bool                         // suppress_response
///   tag 1: bool                         // timed_request
///   tag 2: list of {                    // invoke_requests
///     struct {
///       tag 0: CommandPath (struct)
///       tag 1: struct { <command args TLV> }
///     }
///   }
/// }
/// ```
///
/// InvokeResponse
/// ```text
/// struct {
///   tag 0: bool                         // suppress_response
///   tag 1: list of {                    // invoke_responses
///     struct {
///       // one of:
///       tag 0: CommandDataIB  { tag 0: CommandPath, tag 1: struct{data} }
///       tag 1: CommandStatusIB{ tag 0: CommandPath, tag 1: uint8 status }
///     }
///   }
/// }
/// ```
use super::super::clusters::{
    CommandPath, tlv, tlv_bool, tlv_uint8, wrap_list_tagged, wrap_struct, wrap_struct_tagged,
};
use super::super::error::{MatterError, MatterResult};
use super::write::InteractionStatus;

// ── InvokeRequest ─────────────────────────────────────────────────────────────

/// Request to invoke one or more commands (opcode 0x08).
#[derive(Debug, Clone)]
pub struct InvokeRequest {
    /// When `true`, the device must not send an `InvokeResponse`.
    pub suppress_response: bool,
    /// When `true`, this invoke was preceded by a `TimedRequest`.
    pub timed_request: bool,
    /// The commands to invoke, each paired with its TLV-encoded argument struct.
    pub invoke_requests: Vec<(CommandPath, Vec<u8>)>,
}

impl InvokeRequest {
    /// Construct a single-command invoke request (not timed, expects response).
    pub fn new(path: CommandPath, args: Vec<u8>) -> Self {
        Self {
            suppress_response: false,
            timed_request: false,
            invoke_requests: vec![(path, args)],
        }
    }

    /// TLV-encode the `InvokeRequest`.
    ///
    /// Layout:
    /// ```text
    /// struct {
    ///   tag 0: bool   (suppress_response)
    ///   tag 1: bool   (timed_request)
    ///   tag 2: list {
    ///     struct {
    ///       tag 0: CommandPath (struct)
    ///       tag 1: struct { <args> }
    ///     }
    ///   }
    /// }
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        let mut inner = Vec::new();
        inner.extend_from_slice(&tlv_bool(0, self.suppress_response));
        inner.extend_from_slice(&tlv_bool(1, self.timed_request));

        let mut list_inner = Vec::new();
        for (path, args) in &self.invoke_requests {
            // Inner struct: tag 0 = CommandPath, tag 1 = args struct
            let path_enc = path.encode();
            let path_inner = &path_enc[1..path_enc.len() - 1];
            let cmd_inner = {
                let mut v = Vec::new();
                // tag 0: CommandPath as context-tagged struct
                v.push(tlv::TAG_CONTEXT_1 | tlv::TYPE_STRUCTURE);
                v.push(0u8);
                v.extend_from_slice(path_inner);
                v.push(tlv::TYPE_END_OF_CONTAINER);
                // tag 1: args embedded in a context-tagged struct
                v.extend_from_slice(&wrap_struct_tagged(1, args));
                v
            };
            list_inner.extend_from_slice(&wrap_struct(&cmd_inner));
        }
        inner.extend_from_slice(&wrap_list_tagged(2, &list_inner));
        wrap_struct(&inner)
    }

    /// Decode an `InvokeRequest` from TLV bytes.
    pub fn decode(bytes: &[u8]) -> MatterResult<Self> {
        if bytes.is_empty() || bytes[0] != tlv::TYPE_STRUCTURE {
            return Err(MatterError::Transport(
                "InvokeRequest: expected structure".into(),
            ));
        }
        let mut suppress_response = false;
        let mut timed_request = false;
        let mut invoke_requests = Vec::new();
        let mut i = 1;

        while i < bytes.len() {
            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                break;
            }
            if i + 1 >= bytes.len() {
                return Err(MatterError::Transport("InvokeRequest: truncated".into()));
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
                    // Each element is a CommandDataIB struct
                    while i < bytes.len() && bytes[i] != tlv::TYPE_END_OF_CONTAINER {
                        if bytes[i] != tlv::TYPE_STRUCTURE {
                            return Err(MatterError::Transport(
                                "InvokeRequest: expected CommandDataIB struct".into(),
                            ));
                        }
                        // Parse CommandDataIB
                        i += 1; // skip TYPE_STRUCTURE
                        let mut path: Option<CommandPath> = None;
                        let mut args: Vec<u8> = Vec::new();

                        while i < bytes.len() && bytes[i] != tlv::TYPE_END_OF_CONTAINER {
                            if i + 1 >= bytes.len() {
                                return Err(MatterError::Transport(
                                    "InvokeRequest: truncated CommandDataIB".into(),
                                ));
                            }
                            let inner_ctrl = bytes[i];
                            let inner_tag = bytes[i + 1];
                            i += 2;
                            let inner_type = inner_ctrl & 0x1F;

                            match (inner_tag, inner_type) {
                                (0, t) if t == tlv::TYPE_STRUCTURE => {
                                    // CommandPath struct body
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
                                    let mut cp_bytes = vec![tlv::TYPE_STRUCTURE];
                                    cp_bytes.extend_from_slice(&bytes[start..i - 1]);
                                    cp_bytes.push(tlv::TYPE_END_OF_CONTAINER);
                                    path = CommandPath::decode(&cp_bytes);
                                }
                                (1, t) if t == tlv::TYPE_STRUCTURE => {
                                    // Args struct body
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
                                    args = bytes[start..i - 1].to_vec();
                                }
                                _ => {
                                    return Err(MatterError::Transport(format!(
                                        "InvokeRequest: unexpected CommandDataIB field tag={inner_tag}"
                                    )));
                                }
                            }
                        }
                        if i < bytes.len() {
                            i += 1;
                        } // consume END_OF_CONTAINER of CommandDataIB
                        let p = path.ok_or_else(|| {
                            MatterError::Transport("InvokeRequest: missing CommandPath".into())
                        })?;
                        invoke_requests.push((p, args));
                    }
                    if i < bytes.len() {
                        i += 1;
                    } // consume END_OF_CONTAINER of list
                }
                _ => {
                    return Err(MatterError::Transport(format!(
                        "InvokeRequest: unexpected field tag={tag} ctrl={ctrl:#04x}"
                    )));
                }
            }
        }
        Ok(Self {
            suppress_response,
            timed_request,
            invoke_requests,
        })
    }
}

// ── InvokeResponseItem ────────────────────────────────────────────────────────

/// A single entry in an `InvokeResponse` — either command data or a status.
#[derive(Debug, Clone)]
pub enum InvokeResponseItem {
    /// The command returned data.
    Command {
        /// Endpoint / cluster / command identifying this result.
        path: CommandPath,
        /// TLV-encoded response payload.
        data: Vec<u8>,
    },
    /// The command returned a status code.
    Status {
        /// Endpoint / cluster / command identifying this result.
        path: CommandPath,
        /// IM status code (Success or an error code).
        status: InteractionStatus,
    },
}

// ── InvokeResponse ────────────────────────────────────────────────────────────

/// Command result from device → controller (opcode 0x09).
#[derive(Debug, Clone)]
pub struct InvokeResponse {
    /// When `true`, the controller must not send a `StatusResponse`.
    pub suppress_response: bool,
    /// Per-command results.
    pub invoke_responses: Vec<InvokeResponseItem>,
}

impl InvokeResponse {
    /// TLV-encode the `InvokeResponse`.
    ///
    /// Layout:
    /// ```text
    /// struct {
    ///   tag 0: bool (suppress_response)
    ///   tag 1: list {
    ///     struct {
    ///       // either:
    ///       tag 0: struct { tag 0: CommandPath, tag 1: struct{data} }  // Command
    ///       // or:
    ///       tag 1: struct { tag 0: CommandPath, tag 1: uint8 status }  // Status
    ///     }
    ///   }
    /// }
    /// ```
    pub fn encode(&self) -> Vec<u8> {
        let mut inner = Vec::new();
        inner.extend_from_slice(&tlv_bool(0, self.suppress_response));

        let mut list_inner = Vec::new();
        for item in &self.invoke_responses {
            match item {
                InvokeResponseItem::Command { path, data } => {
                    let path_enc = path.encode();
                    let path_body = &path_enc[1..path_enc.len() - 1];
                    // CommandDataIB: { tag 0: CommandPath, tag 1: data }
                    let mut cib = Vec::new();
                    cib.push(tlv::TAG_CONTEXT_1 | tlv::TYPE_STRUCTURE);
                    cib.push(0u8);
                    cib.extend_from_slice(path_body);
                    cib.push(tlv::TYPE_END_OF_CONTAINER);
                    cib.extend_from_slice(&wrap_struct_tagged(1, data));
                    // Outer CommandDataIB in tag 0 struct
                    let outer = wrap_struct_tagged(0, &cib);
                    list_inner.extend_from_slice(&wrap_struct(&outer));
                }
                InvokeResponseItem::Status { path, status } => {
                    let path_enc = path.encode();
                    let path_body = &path_enc[1..path_enc.len() - 1];
                    // CommandStatusIB: { tag 0: CommandPath, tag 1: status_code }
                    let mut sib = Vec::new();
                    sib.push(tlv::TAG_CONTEXT_1 | tlv::TYPE_STRUCTURE);
                    sib.push(0u8);
                    sib.extend_from_slice(path_body);
                    sib.push(tlv::TYPE_END_OF_CONTAINER);
                    sib.extend_from_slice(&tlv_uint8(1, *status as u8));
                    let outer = wrap_struct_tagged(1, &sib);
                    list_inner.extend_from_slice(&wrap_struct(&outer));
                }
            }
        }
        inner.extend_from_slice(&wrap_list_tagged(1, &list_inner));
        wrap_struct(&inner)
    }

    /// Decode an `InvokeResponse` from TLV bytes.
    pub fn decode(bytes: &[u8]) -> MatterResult<Self> {
        if bytes.is_empty() || bytes[0] != tlv::TYPE_STRUCTURE {
            return Err(MatterError::Transport(
                "InvokeResponse: expected structure".into(),
            ));
        }
        let mut suppress_response = false;
        let mut invoke_responses = Vec::new();
        let mut i = 1;

        while i < bytes.len() {
            if bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                break;
            }
            if i + 1 >= bytes.len() {
                return Err(MatterError::Transport("InvokeResponse: truncated".into()));
            }
            let ctrl = bytes[i];
            let tag = bytes[i + 1];
            i += 2;
            let type_bits = ctrl & 0x1F;

            match (tag, type_bits) {
                (0, t) if t == tlv::TYPE_BOOL_TRUE || t == tlv::TYPE_BOOL_FALSE => {
                    suppress_response = t == tlv::TYPE_BOOL_TRUE;
                }
                (1, t) if t == tlv::TYPE_LIST => {
                    while i < bytes.len() && bytes[i] != tlv::TYPE_END_OF_CONTAINER {
                        // Each is a struct with either tag 0 (Command) or tag 1 (Status) field
                        if bytes[i] != tlv::TYPE_STRUCTURE {
                            return Err(MatterError::Transport(
                                "InvokeResponse: expected response item struct".into(),
                            ));
                        }
                        i += 1; // skip TYPE_STRUCTURE
                        // Read the discriminant: ctrl + tag
                        if i + 1 >= bytes.len() {
                            return Err(MatterError::Transport(
                                "InvokeResponse: truncated item".into(),
                            ));
                        }
                        let item_ctrl = bytes[i];
                        let item_tag = bytes[i + 1];
                        i += 2;
                        let item_type = item_ctrl & 0x1F;

                        if item_type != tlv::TYPE_STRUCTURE {
                            return Err(MatterError::Transport(
                                "InvokeResponse: expected inner struct".into(),
                            ));
                        }

                        // Parse inner struct body
                        let inner_start = i;
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
                        let inner_bytes = &bytes[inner_start..i - 1]; // body without END

                        // Parse inner struct: { tag 0: CommandPath, tag 1: data/status }
                        let (path, extra) =
                            parse_command_response_inner(inner_bytes).ok_or_else(|| {
                                MatterError::Transport("InvokeResponse: bad inner struct".into())
                            })?;

                        let item = match item_tag {
                            0 => InvokeResponseItem::Command { path, data: extra },
                            1 => {
                                let code = extra.first().copied().unwrap_or(0);
                                let status = InteractionStatus::from_u8(code)
                                    .unwrap_or(InteractionStatus::Failure);
                                InvokeResponseItem::Status { path, status }
                            }
                            _ => {
                                return Err(MatterError::Transport(format!(
                                    "InvokeResponse: unknown item_tag={item_tag}"
                                )));
                            }
                        };
                        invoke_responses.push(item);

                        // consume END_OF_CONTAINER for outer item struct
                        if i < bytes.len() && bytes[i] == tlv::TYPE_END_OF_CONTAINER {
                            i += 1;
                        }
                    }
                    if i < bytes.len() {
                        i += 1;
                    } // consume list END_OF_CONTAINER
                }
                _ => {
                    return Err(MatterError::Transport(format!(
                        "InvokeResponse: unexpected field tag={tag} ctrl={ctrl:#04x}"
                    )));
                }
            }
        }
        Ok(Self {
            suppress_response,
            invoke_responses,
        })
    }
}

/// Parse the body of a CommandDataIB or CommandStatusIB inner struct.
/// Returns `(CommandPath, data_or_status_bytes)`.
fn parse_command_response_inner(body: &[u8]) -> Option<(CommandPath, Vec<u8>)> {
    let mut path: Option<CommandPath> = None;
    let mut extra: Vec<u8> = Vec::new();
    let mut i = 0;

    while i < body.len() {
        if body[i] == tlv::TYPE_END_OF_CONTAINER {
            break;
        }
        if i + 1 >= body.len() {
            return None;
        }
        let ctrl = body[i];
        let tag = body[i + 1];
        i += 2;
        let type_bits = ctrl & 0x1F;

        match (tag, type_bits) {
            (0, t) if t == tlv::TYPE_STRUCTURE => {
                let start = i;
                let mut depth = 1u32;
                while i < body.len() && depth > 0 {
                    if body[i] == tlv::TYPE_END_OF_CONTAINER {
                        depth -= 1;
                    } else if body[i] == tlv::TYPE_STRUCTURE {
                        depth += 1;
                    }
                    i += 1;
                }
                let mut cp_bytes = vec![tlv::TYPE_STRUCTURE];
                cp_bytes.extend_from_slice(&body[start..i - 1]);
                cp_bytes.push(tlv::TYPE_END_OF_CONTAINER);
                path = CommandPath::decode(&cp_bytes);
            }
            (1, t) if t == tlv::TYPE_STRUCTURE => {
                // For Command: data struct body
                let start = i;
                let mut depth = 1u32;
                while i < body.len() && depth > 0 {
                    if body[i] == tlv::TYPE_END_OF_CONTAINER {
                        depth -= 1;
                    } else if body[i] == tlv::TYPE_STRUCTURE {
                        depth += 1;
                    }
                    i += 1;
                }
                extra = body[start..i - 1].to_vec();
            }
            (1, t) if t == tlv::TYPE_UNSIGNED_INT_1 => {
                // For Status: u8 status code
                if i < body.len() {
                    extra = vec![body[i]];
                    i += 1;
                }
            }
            _ => return None,
        }
    }
    Some((path?, extra))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::super::clusters::{CommandPath, on_off};
    use super::*;

    #[test]
    fn invoke_request_single_command_roundtrip() {
        let path = CommandPath::new(1, 0x0006, on_off::CMD_ON);
        let args = on_off::on_tlv();
        let req = InvokeRequest::new(path.clone(), args.clone());
        let encoded = req.encode();
        let decoded = InvokeRequest::decode(&encoded).expect("decode failed");
        assert!(!decoded.suppress_response);
        assert!(!decoded.timed_request);
        assert_eq!(decoded.invoke_requests.len(), 1);
        let (dec_path, dec_args) = &decoded.invoke_requests[0];
        assert_eq!(*dec_path, path);
        assert_eq!(*dec_args, args);
    }

    #[test]
    fn invoke_request_encode_has_correct_opcode_structure() {
        let path = CommandPath::new(0, 0x0006, 0x01);
        let req = InvokeRequest::new(path, vec![]);
        let encoded = req.encode();
        // Starts with TYPE_STRUCTURE
        assert_eq!(encoded[0], tlv::TYPE_STRUCTURE);
        // Ends with END_OF_CONTAINER
        assert_eq!(*encoded.last().unwrap(), tlv::TYPE_END_OF_CONTAINER);
        // Contains suppress_response (tag 0) and timed_request (tag 1) booleans,
        // and the invoke list (tag 2).
        // Look for tag 0 false bool: [TAG_CONTEXT_1 | TYPE_BOOL_FALSE, 0]
        let sr_ctrl = tlv::TAG_CONTEXT_1 | tlv::TYPE_BOOL_FALSE;
        assert!(
            encoded.windows(2).any(|w| w == [sr_ctrl, 0]),
            "suppress_response field not found"
        );
        // Look for tag 1 false bool
        assert!(
            encoded.windows(2).any(|w| w == [sr_ctrl, 1]),
            "timed_request field not found"
        );
        // Contains list marker (TAG_CONTEXT_1 | TYPE_LIST = 0x37) with tag 2
        let list_ctrl = tlv::TAG_CONTEXT_1 | tlv::TYPE_LIST;
        assert!(
            encoded.windows(2).any(|w| w == [list_ctrl, 2]),
            "invoke list not found"
        );
    }
}
