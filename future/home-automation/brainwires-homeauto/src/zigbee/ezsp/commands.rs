//! EZSP v8 command ID constants and typed request/response helpers.
//!
//! Sources: Silicon Labs UG100, AN0042, EZSP Reference Guide (EmberZNet 7.x).

// ── System / Configuration ────────────────────────────────────────────────────

/// `0x0000` — VERSION: NCP firmware + EZSP protocol version.
pub const VERSION: u16 = 0x0000;
/// `0x0052` — GET_CONFIG_VALUE: read a configuration parameter.
pub const GET_CONFIG_VALUE: u16 = 0x0052;
/// `0x0053` — SET_CONFIG_VALUE: write a configuration parameter.
pub const SET_CONFIG_VALUE: u16 = 0x0053;
/// `0x0056` — GET_POLICY: read a stack policy.
pub const GET_POLICY: u16 = 0x0056;
/// `0x0055` — SET_POLICY: write a stack policy.
pub const SET_POLICY: u16 = 0x0055;
/// `0x00AA` — GET_VALUE: read an extended configuration value.
pub const GET_VALUE: u16 = 0x00AA;
/// `0x00AB` — SET_VALUE: write an extended configuration value.
pub const SET_VALUE: u16 = 0x00AB;

// ── Network ───────────────────────────────────────────────────────────────────

/// `0x001E` — FORM_NETWORK: create a new PAN.
pub const FORM_NETWORK: u16 = 0x001E;
/// `0x001F` — JOIN_NETWORK: associate with an existing PAN.
pub const JOIN_NETWORK: u16 = 0x001F;
/// `0x0020` — LEAVE_NETWORK: disassociate from the current PAN.
pub const LEAVE_NETWORK: u16 = 0x0020;
/// `0x0022` — PERMIT_JOINING: open the join window.
pub const PERMIT_JOINING: u16 = 0x0022;
/// `0x0028` — GET_NETWORK_PARAMETERS: read PAN ID / channel / security.
pub const GET_NETWORK_PARAMETERS: u16 = 0x0028;
/// `0x0018` — NETWORK_STATE: current stack state.
pub const NETWORK_STATE: u16 = 0x0018;
/// `0x0019` — STACK_STATUS_HANDLER callback (NCP→host).
pub const STACK_STATUS_HANDLER: u16 = 0x0019;

// ── Node identity ─────────────────────────────────────────────────────────────

/// `0x0026` — GET_EUI64: this node's IEEE 64-bit address.
pub const GET_EUI64: u16 = 0x0026;
/// `0x0027` — GET_NODE_ID: this node's 16-bit network address.
pub const GET_NODE_ID: u16 = 0x0027;

// ── Messaging ─────────────────────────────────────────────────────────────────

/// `0x0034` — SEND_UNICAST: send a packet to one address.
pub const SEND_UNICAST: u16 = 0x0034;
/// `0x0036` — SEND_BROADCAST: send to all nodes on the PAN.
pub const SEND_BROADCAST: u16 = 0x0036;
/// `0x0038` — SEND_MULTICAST: send to a group/multicast address.
pub const SEND_MULTICAST: u16 = 0x0038;
/// `0x003F` — MESSAGE_SENT_HANDLER callback (delivery status).
pub const MESSAGE_SENT_HANDLER: u16 = 0x003F;
/// `0x0045` — INCOMING_MESSAGE_HANDLER callback (NCP→host).
pub const INCOMING_MESSAGE_HANDLER: u16 = 0x0045;

// ── Trust Center / Security ───────────────────────────────────────────────────

/// `0x0024` — TRUST_CENTER_JOIN_HANDLER: device-join authorization callback.
pub const TRUST_CENTER_JOIN_HANDLER: u16 = 0x0024;
/// `0x0068` — SET_INITIAL_SECURITY_STATE: install pre-config keys.
pub const SET_INITIAL_SECURITY_STATE: u16 = 0x0068;
/// `0x0069` — GET_CURRENT_SECURITY_STATE: read security bitmap.
pub const GET_CURRENT_SECURITY_STATE: u16 = 0x0069;
/// `0x006A` — GET_KEY: read a named security key.
pub const GET_KEY: u16 = 0x006A;
/// `0x00A9` — SET_KEY: install a named security key.
pub const SET_KEY: u16 = 0x00A9;

// ── Neighbor / device management ─────────────────────────────────────────────

/// `0x0079` — GET_NEIGHBOR: read a neighbor-table entry.
pub const GET_NEIGHBOR: u16 = 0x0079;
/// `0x007A` — NEIGHBOR_COUNT: number of entries in the neighbor table.
pub const NEIGHBOR_COUNT: u16 = 0x007A;
/// `0x007B` — GET_ROUTE_TABLE_ENTRY: read a route-table entry.
pub const GET_ROUTE_TABLE_ENTRY: u16 = 0x007B;
/// `0x0077` — ADDRESS_TABLE_ENTRY: read an address-table entry.
pub const ADDRESS_TABLE_ENTRY: u16 = 0x0077;
/// `0x004E` — GET_ADDRESS_TABLE_REMOTE_EUI64.
pub const GET_ADDRESS_TABLE_REMOTE_EUI64: u16 = 0x004E;
/// `0x004F` — GET_ADDRESS_TABLE_REMOTE_NODE_ID.
pub const GET_ADDRESS_TABLE_REMOTE_NODE_ID: u16 = 0x004F;

// ── EZSP status codes ─────────────────────────────────────────────────────────

/// `0x00` — Success.
pub const STATUS_SUCCESS: u8 = 0x00;
/// `0x01` — Fatal NCP error.
pub const STATUS_ERR_FATAL: u8 = 0x01;
/// `0x28` — Frame-ID not recognised by this NCP firmware.
pub const STATUS_INVALID_FRAME_ID: u8 = 0x28;
/// `0x31` — EZSP protocol version mismatch.
pub const STATUS_VERSION_NOT_SUPPORTED: u8 = 0x31;

// ── Typed helpers ─────────────────────────────────────────────────────────────

/// Encode a PERMIT_JOINING command payload.
/// `duration`: 0 = disable, 0xFF = forever, 1–254 = seconds.
pub fn permit_joining_payload(duration: u8) -> Vec<u8> {
    vec![duration]
}

/// Encode a SEND_UNICAST payload header.
///
/// Full unicast frame:
/// `type(1) | indexOrDest(2) | apsFrame(11+) | msgTag(1) | msgLen(1) | msg(msgLen)`
// reason: the 8 args correspond 1:1 to the EZSP wire format fields; bundling
// them into a struct only relocates the same field list.
#[allow(clippy::too_many_arguments)]
pub fn send_unicast_payload(
    dest_nwk: u16,
    src_endpoint: u8,
    dst_endpoint: u8,
    cluster_id: u16,
    profile_id: u16,
    sequence: u8,
    msg_tag: u8,
    payload: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::new();
    // type = EMBER_OUTGOING_DIRECT = 0x00
    buf.push(0x00);
    // indexOrDest = destination NWK address
    buf.extend_from_slice(&dest_nwk.to_le_bytes());
    // APS frame: options(2) | profileId(2) | clusterId(2) | srcEp(1) | dstEp(1) | groupId(2) | seq(1)
    buf.extend_from_slice(&0x0000u16.to_le_bytes()); // options
    buf.extend_from_slice(&profile_id.to_le_bytes());
    buf.extend_from_slice(&cluster_id.to_le_bytes());
    buf.push(src_endpoint);
    buf.push(dst_endpoint);
    buf.extend_from_slice(&0x0000u16.to_le_bytes()); // groupId
    buf.push(sequence);
    // msgTag
    buf.push(msg_tag);
    // message length + content
    buf.push(payload.len() as u8);
    buf.extend_from_slice(payload);
    buf
}

/// Decode an INCOMING_MESSAGE_HANDLER callback payload.
/// Returns (message_type, aps_frame_cluster_id, src_nwk, src_endpoint, payload) or None.
pub fn decode_incoming_message(params: &[u8]) -> Option<(u8, u16, u16, u8, &[u8])> {
    if params.len() < 12 {
        return None;
    }
    let msg_type = params[0];
    // APS frame starts at byte 1: options(2)|profileId(2)|clusterId(2)|srcEp(1)|dstEp(1)|groupId(2)|seq(1)
    let cluster_id = u16::from_le_bytes([params[3], params[4]]);
    let src_endpoint = params[6];
    // srcNwkAddr at offset 11
    let src_nwk = u16::from_le_bytes([params[11], params[12]]);
    let msg_len = *params.get(13)? as usize;
    let msg_start = 14;
    let payload = params.get(msg_start..msg_start + msg_len)?;
    Some((msg_type, cluster_id, src_nwk, src_endpoint, payload))
}
