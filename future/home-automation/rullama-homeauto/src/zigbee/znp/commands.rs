//! ZNP command constants for TI Z-Stack 3.x.
//!
//! Subsystem IDs and command bytes from the TI Z-Stack Monitor and Test API
//! (MT API, Z-Stack 3.x).

// ── Subsystem IDs ─────────────────────────────────────────────────────────────

/// `0x21` — System subsystem.
pub const SYS: u8 = 0x21;
/// `0x22` — MAC-layer subsystem.
pub const MAC: u8 = 0x22;
/// `0x26` — Network-layer subsystem.
pub const NWK: u8 = 0x26;
/// `0x24` — Application Framework subsystem.
pub const AF: u8 = 0x24;
/// `0x25` — Zigbee Device Object subsystem.
pub const ZDO: u8 = 0x25;
/// `0x2F` — Simple API subsystem.
pub const SAPI: u8 = 0x2F;
/// `0x27` — Utilities subsystem.
pub const UTIL: u8 = 0x27;
/// `0x26` — App config (overlaps NWK per Z-Stack 3.x).
pub const APP_CNF: u8 = 0x26;
/// `0x29` — Application-layer subsystem.
pub const APP: u8 = 0x29;

// ── SYS subsystem commands ────────────────────────────────────────────────────

/// `0x09` — SYS_RESET_REQ (SREQ): request NCP reset.
pub const SYS_RESET_REQ: u8 = 0x09;
/// `0x80` — SYS_RESET_IND (AREQ): reset completion indication.
pub const SYS_RESET_IND: u8 = 0x80;
/// `0x02` — SYS_VERSION: firmware version query.
pub const SYS_VERSION: u8 = 0x02;
/// `0x01` — SYS_PING: connectivity check.
pub const SYS_PING: u8 = 0x01;
/// `0x08` — SYS_OSAL_NV_READ: read an NV item.
pub const SYS_OSAL_NV_READ: u8 = 0x08;
/// `0x09` — SYS_OSAL_NV_WRITE: write an NV item.
pub const SYS_OSAL_NV_WRITE: u8 = 0x09;
/// `0x04` — SYS_GET_EXTADDR: get this device's IEEE address.
pub const SYS_GET_EXTADDR: u8 = 0x04;

// ── ZDO subsystem commands ────────────────────────────────────────────────────

/// `0x40` — ZDO_STARTUP_FROM_APP (SREQ): start the Zigbee stack.
pub const ZDO_STARTUP_FROM_APP: u8 = 0x40;
/// `0x02` — ZDO_NODE_DESC_REQ: request a node descriptor.
pub const ZDO_NODE_DESC_REQ: u8 = 0x02;
/// `0x05` — ZDO_ACTIVE_EP_REQ: enumerate active endpoints on a node.
pub const ZDO_ACTIVE_EP_REQ: u8 = 0x05;
/// `0x04` — ZDO_SIMPLE_DESC_REQ: read a simple descriptor for an endpoint.
pub const ZDO_SIMPLE_DESC_REQ: u8 = 0x04;
/// `0xFF` — ZDO_END_DEVICE_ANNCE_IND (AREQ): new device joined.
pub const ZDO_END_DEVICE_ANNCE_IND: u8 = 0xFF;
/// `0xCA` — ZDO_TC_DEV_IND (AREQ): trust-center device indication.
pub const ZDO_TC_DEV_IND: u8 = 0xCA;
/// `0x36` — ZDO_PERMIT_JOIN_REQ: open/close the join window.
pub const ZDO_PERMIT_JOIN_REQ: u8 = 0x36;
/// `0xCB` — ZDO_PERMIT_JOIN_IND (AREQ): permit-join state changed.
pub const ZDO_PERMIT_JOIN_IND: u8 = 0xCB;
/// `0x80` — ZDO_NWK_ADDR_RSP: response to network-address request.
pub const ZDO_NWK_ADDR_RSP: u8 = 0x80;
/// `0x81` — ZDO_IEEE_ADDR_RSP: response to IEEE-address request.
pub const ZDO_IEEE_ADDR_RSP: u8 = 0x81;
/// `0xC0` — ZDO_STATE_CHANGE_IND (AREQ): stack state transition.
pub const ZDO_STATE_CHANGE_IND: u8 = 0xC0;
/// `0xC9` — ZDO_LEAVE_IND (AREQ): device left the network.
pub const ZDO_LEAVE_IND: u8 = 0xC9;

// ── AF subsystem commands ─────────────────────────────────────────────────────

/// `0x00` — AF_REGISTER: register an endpoint with the stack.
pub const AF_REGISTER: u8 = 0x00;
/// `0x01` — AF_DATA_REQUEST (SREQ): send a ZCL message.
pub const AF_DATA_REQUEST: u8 = 0x01;
/// `0x05` — AF_DATA_CONFIRM (AREQ): delivery confirmation.
pub const AF_DATA_CONFIRM: u8 = 0x05;
/// `0x81` — AF_INCOMING_MSG (AREQ): inbound application frame.
pub const AF_INCOMING_MSG: u8 = 0x81;
/// `0x02` — AF_DATA_REQUEST_EXT: extended data request (long addressing).
pub const AF_DATA_REQUEST_EXT: u8 = 0x02;

// ── APP_CNF subsystem commands ────────────────────────────────────────────────

/// `0x00` — APP_CNF_BDB_START_COMMISSIONING: kick off BDB commissioning.
pub const APP_CNF_BDB_START_COMMISSIONING: u8 = 0x00;
/// `0x08` — APP_CNF_BDB_SET_CHANNEL: set the Zigbee channel mask.
pub const APP_CNF_BDB_SET_CHANNEL: u8 = 0x08;
/// `0x80` — APP_CNF_BDB_COMMISSIONING_NOTIFICATION (AREQ): commissioning done.
pub const APP_CNF_BDB_COMMISSIONING_NOTIFICATION: u8 = 0x80;

// ── ZDO network state values ─────────────────────────────────────────────────

/// Network state: coordinator role (`0x09`).
pub const DEV_COORDINATOR: u8 = 0x09;
/// Network state: router role (`0x08`).
pub const DEV_ROUTER: u8 = 0x08;
/// Network state: end-device role (`0x07`).
pub const DEV_END_DEVICE: u8 = 0x07;
/// Network state: initialized but holding (`0x00`).
pub const DEV_HOLD: u8 = 0x00;
/// Network state: initialisation in progress (`0x01`).
pub const DEV_INIT: u8 = 0x01;

// ── Status codes ─────────────────────────────────────────────────────────────

/// ZNP status `0x00` — Success.
pub const ZNP_STATUS_SUCCESS: u8 = 0x00;
/// ZNP status `0x01` — Generic failure.
pub const ZNP_STATUS_FAILED: u8 = 0x01;
/// ZNP status `0x02` — Invalid parameter.
pub const ZNP_STATUS_INVALID_PARAM: u8 = 0x02;

// ── Typed payload helpers ─────────────────────────────────────────────────────

/// Build the payload for ZDO_STARTUP_FROM_APP (start-delay in ms).
pub fn startup_payload(start_delay_ms: u16) -> Vec<u8> {
    start_delay_ms.to_le_bytes().to_vec()
}

/// Build the payload for AF_DATA_REQUEST (send a ZCL message).
///
/// Layout: dstAddr(2) | dstEndpoint(1) | srcEndpoint(1) | clusterId(2) |
///         transId(1) | options(1) | radius(1) | len(1) | data(len)
pub fn af_data_request(
    dst_nwk: u16,
    dst_ep: u8,
    src_ep: u8,
    cluster_id: u16,
    trans_id: u8,
    data: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&dst_nwk.to_le_bytes());
    buf.push(dst_ep);
    buf.push(src_ep);
    buf.extend_from_slice(&cluster_id.to_le_bytes());
    buf.push(trans_id);
    buf.push(0x00); // options: none
    buf.push(0x0F); // radius: 15 hops
    buf.push(data.len() as u8);
    buf.extend_from_slice(data);
    buf
}

/// Build the payload for ZDO_PERMIT_JOIN_REQ.
/// `dest` = 0xFFFC (all routers + coordinator), duration = 0–254 s or 0xFF (forever).
pub fn permit_join_payload(dest: u16, duration: u8) -> Vec<u8> {
    let mut buf = dest.to_le_bytes().to_vec();
    buf.push(duration);
    buf.push(0); // TCSignificance
    buf
}

/// Decode an AF_INCOMING_MSG AREQ payload.
/// Returns (groupId, clusterId, srcAddr, srcEp, dstEp, transId, payload) or None.
// reason: the tuple maps 1:1 to the AF_INCOMING_MSG wire layout; a named
// struct would just rename the same fields without making the call site
// clearer at the single decode point.
#[allow(clippy::type_complexity)]
pub fn decode_af_incoming(params: &[u8]) -> Option<(u16, u16, u16, u8, u8, u8, &[u8])> {
    if params.len() < 11 {
        return None;
    }
    let group_id = u16::from_le_bytes([params[0], params[1]]);
    let cluster_id = u16::from_le_bytes([params[2], params[3]]);
    let src_addr = u16::from_le_bytes([params[4], params[5]]);
    let src_ep = params[6];
    let dst_ep = params[7];
    let trans_id = params[8];
    // params[9] = broadcast radius, params[10] = link quality
    let data_len = *params.get(11)? as usize;
    let data = params.get(12..12 + data_len)?;
    Some((
        group_id, cluster_id, src_addr, src_ep, dst_ep, trans_id, data,
    ))
}
