/// Z-Wave Serial API (ZAPI) framing and controller implementation.
///
/// Implements Z-Wave Plus v2 / SDK 7.x Serial API (ZAPI2) as documented in
/// Silicon Labs INS13954 "Z-Wave Serial API Host Application Programming Guide".
///
/// Frame layout:
/// ```text
/// SOF (0x01) | LEN | TYPE (REQ=0x00 / RES=0x01) | CMD_ID | DATA... | CS
/// ```
///
/// CS = 0xFF XOR LEN XOR TYPE XOR CMD_ID XOR DATA...
///
/// Flow control (single byte, no SOF):
/// - ACK = 0x06 (frame received OK)
/// - NAK = 0x15 (frame rejected, retransmit)
/// - CAN = 0x18 (host cancelled / collision)
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::time::timeout;
use tokio_serial::SerialStream;
use tracing::{debug, error, warn};

use super::ZWaveController;
use super::command_class::CommandClass;
use super::types::{NodeId, ZWaveNode};
use crate::error::{HomeAutoError, HomeAutoResult};
use crate::types::{HomeAutoEvent, Protocol};

// ── Wire constants ────────────────────────────────────────────────────────────

/// Start-of-Frame byte — every ZAPI message begins with this byte.
pub const SOF: u8 = 0x01;
/// Acknowledge flow-control byte (frame received intact).
pub const ACK: u8 = 0x06;
/// Negative-acknowledge flow-control byte (frame rejected, retransmit).
pub const NAK: u8 = 0x15;
/// Cancel flow-control byte (host-originated cancellation or collision).
pub const CAN: u8 = 0x18;

/// Frame type: host → controller request (`0x00`).
pub const FRAME_TYPE_REQ: u8 = 0x00;
/// Frame type: controller → host response (`0x01`).
pub const FRAME_TYPE_RES: u8 = 0x01;

// ── ZAPI command IDs ─────────────────────────────────────────────────────────

/// `GET_CAPABILITIES` (`0x07`) — controller capability query.
pub const GET_CAPABILITIES: u8 = 0x07;
/// `SERIAL_API_STARTED` (`0x0A`) — controller sends after a soft reset.
pub const SERIAL_API_STARTED: u8 = 0x0A;
/// `GET_VERSION` (`0x15`) — controller firmware/protocol version.
pub const GET_VERSION: u8 = 0x15;
/// `SEND_DATA` (`0x13`) — unicast command to a single node.
pub const SEND_DATA: u8 = 0x13;
/// `SEND_DATA_MULTI` (`0x14`) — multicast command to a node group.
pub const SEND_DATA_MULTI: u8 = 0x14;
/// `GET_INIT_DATA` (`0x02`) — initial node-list snapshot.
pub const GET_INIT_DATA: u8 = 0x02;
/// `APPLICATION_COMMAND_HANDLER` (`0x04`) — controller-originated node report.
pub const APPLICATION_COMMAND_HANDLER: u8 = 0x04;
/// `ADD_NODE_TO_NETWORK` (`0x4A`) — inclusion state-machine kickoff.
pub const ADD_NODE_TO_NETWORK: u8 = 0x4A;
/// `REMOVE_NODE_FROM_NETWORK` (`0x4B`) — exclusion state-machine kickoff.
pub const REMOVE_NODE_FROM_NETWORK: u8 = 0x4B;
/// `SET_DEFAULT` (`0x42`) — controller factory reset.
pub const SET_DEFAULT: u8 = 0x42;
/// `GET_NODE_PROTOCOL_INFO` (`0x41`) — basic/generic/specific device type.
pub const GET_NODE_PROTOCOL_INFO: u8 = 0x41;
/// `REQUEST_NODE_INFO` (`0x60`) — ask a node to re-send its command-class list.
pub const REQUEST_NODE_INFO: u8 = 0x60;
/// `APPLICATION_SLAVE_COMMAND_HANDLER` (`0xA1`) — slave-role frame variant.
pub const APPLICATION_SLAVE_COMMAND_HANDLER: u8 = 0xA1;

/// `ADD_NODE_TO_NETWORK` mode byte: any (`0x01`).
pub const ADD_NODE_ANY: u8 = 0x01;
/// `ADD_NODE_TO_NETWORK` mode byte: stop inclusion (`0x05`).
pub const ADD_NODE_STOP: u8 = 0x05;

/// `REMOVE_NODE_FROM_NETWORK` mode byte: any (`0x01`).
pub const REMOVE_NODE_ANY: u8 = 0x01;
/// `REMOVE_NODE_FROM_NETWORK` mode byte: stop exclusion (`0x05`).
pub const REMOVE_NODE_STOP: u8 = 0x05;

/// `SEND_DATA` tx option: request an ACK from the destination (`0x01`).
pub const TRANSMIT_OPTION_ACK: u8 = 0x01;
/// `SEND_DATA` tx option: auto-route via mesh (`0x04`).
pub const TRANSMIT_OPTION_AUTO_ROUTE: u8 = 0x04;
/// `SEND_DATA` tx option: use Explorer frames for route discovery (`0x20`).
pub const TRANSMIT_OPTION_EXPLORE: u8 = 0x20;

// ── Frame encoding / decoding ─────────────────────────────────────────────────

/// Compute Z-Wave Serial API checksum: `0xFF XOR (all bytes from LEN through last data byte)`.
pub fn checksum(data: &[u8]) -> u8 {
    data.iter().fold(0xFF_u8, |acc, &b| acc ^ b)
}

/// A decoded ZAPI frame.
#[derive(Debug, Clone)]
pub struct ZApiFrame {
    /// `FRAME_TYPE_REQ` (0x00) or `FRAME_TYPE_RES` (0x01).
    pub frame_type: u8,
    /// ZAPI command-ID byte (see the `GET_*` / `SEND_*` constants above).
    pub cmd_id: u8,
    /// Command-specific payload (excludes type, cmd_id, and checksum).
    pub data: Vec<u8>,
}

impl ZApiFrame {
    /// Build a request frame with `cmd_id` and payload `data`.
    pub fn new_request(cmd_id: u8, data: Vec<u8>) -> Self {
        Self {
            frame_type: FRAME_TYPE_REQ,
            cmd_id,
            data,
        }
    }

    /// Encode to wire bytes (SOF | LEN | TYPE | CMD_ID | DATA | CS).
    pub fn encode(&self) -> Vec<u8> {
        let len = (2 + self.data.len()) as u8; // TYPE + CMD_ID + DATA
        let mut buf = vec![SOF, len, self.frame_type, self.cmd_id];
        buf.extend_from_slice(&self.data);
        // CS computed over LEN..last data byte
        buf.push(checksum(&buf[1..]));
        buf
    }

    /// Decode from a raw byte slice (starting from the LEN byte, *after* SOF).
    /// Returns the frame and the total bytes consumed (including SOF if present).
    pub fn decode_after_sof(data: &[u8]) -> Result<(Self, usize), &'static str> {
        if data.len() < 4 {
            return Err("ZAPI frame too short");
        }
        let len = data[0] as usize;
        if len < 2 {
            return Err("ZAPI LEN < 2");
        }
        if data.len() < 1 + len + 1 {
            return Err("ZAPI frame incomplete");
        }
        let frame_type = data[1];
        let cmd_id = data[2];
        let payload = data[3..len - 1 + 2].to_vec(); // LEN includes TYPE + CMD_ID + DATA
        let received_cs = data[1 + len];
        let computed_cs = checksum(&data[0..=len]);
        if received_cs != computed_cs {
            return Err("ZAPI checksum mismatch");
        }
        let total = 1 + len + 1; // LEN byte + content + CS
        Ok((
            Self {
                frame_type,
                cmd_id,
                data: payload,
            },
            total,
        ))
    }
}

// ── Pending response tracking ─────────────────────────────────────────────────

struct PendingCmd {
    tx: oneshot::Sender<ZApiFrame>,
}

struct ZApiInner {
    pending: HashMap<u8, PendingCmd>,
    writer: Option<tokio::io::WriteHalf<SerialStream>>,
    event_tx: mpsc::Sender<HomeAutoEvent>,
    nodes: HashMap<NodeId, ZWaveNode>,
    callback_id: u8,
}

/// Z-Wave Serial API (ZAPI2) controller.
///
/// Connects directly to a Z-Wave USB stick (Aeotec Z-Stick Gen5+, Nortek HUSBZB-1, etc.)
/// via serial port. Implements Z-Wave Plus v2 (Specification 7.x).
pub struct ZWaveSerialController {
    port_path: String,
    baud_rate: u32,
    inner: Arc<Mutex<ZApiInner>>,
}

impl ZWaveSerialController {
    /// Create a controller for `port` at `baud_rate` (usually 115200).
    pub fn new(port: impl Into<String>, baud_rate: u32) -> Self {
        let (event_tx, _) = mpsc::channel(64);
        Self {
            port_path: port.into(),
            baud_rate,
            inner: Arc::new(Mutex::new(ZApiInner {
                pending: HashMap::new(),
                writer: None,
                event_tx,
                nodes: HashMap::new(),
                callback_id: 1,
            })),
        }
    }

    /// Send a ZAPI request frame and await the matching response (2 s, up to 3 retries on NAK).
    async fn send_request(&self, cmd_id: u8, data: Vec<u8>) -> HomeAutoResult<ZApiFrame> {
        const MAX_RETRIES: u8 = 3;
        for attempt in 0..MAX_RETRIES {
            let frame = ZApiFrame::new_request(cmd_id, data.clone());
            let wire = frame.encode();

            let rx = {
                let mut inner = self.inner.lock().await;
                let (tx, rx) = oneshot::channel();
                inner.pending.insert(cmd_id, PendingCmd { tx });
                match inner.writer.as_mut() {
                    Some(w) => w.write_all(&wire).await.map_err(HomeAutoError::Io)?,
                    None => {
                        inner.pending.remove(&cmd_id);
                        return Err(HomeAutoError::ZWaveController(
                            "controller not started — call start() first".into(),
                        ));
                    }
                }
                rx
            };

            match timeout(Duration::from_secs(2), rx).await {
                Ok(Ok(resp)) => return Ok(resp),
                Ok(Err(_)) => return Err(HomeAutoError::ChannelClosed),
                Err(_) if attempt < MAX_RETRIES - 1 => {
                    warn!("ZAPI timeout for cmd {cmd_id:#04x}, retry {}", attempt + 1);
                }
                Err(_) => return Err(HomeAutoError::Timeout),
            }
        }
        Err(HomeAutoError::ZWaveNak {
            retries: MAX_RETRIES,
        })
    }

    /// Allocate a new callback function ID (1–255, wraps).
    async fn next_callback_id(&self) -> u8 {
        let mut inner = self.inner.lock().await;
        let id = inner.callback_id;
        inner.callback_id = inner.callback_id.wrapping_add(1).max(1);
        id
    }

    /// Spawn the serial reader task.
    async fn spawn_reader(&self, mut reader: tokio::io::ReadHalf<SerialStream>) {
        let inner = Arc::clone(&self.inner);
        tokio::spawn(async move {
            let mut buf: Vec<u8> = Vec::with_capacity(256);
            let mut byte = [0u8; 1];
            loop {
                match reader.read_exact(&mut byte).await {
                    Err(e) => {
                        error!("ZAPI serial read error: {e}");
                        break;
                    }
                    Ok(_) => {
                        let b = byte[0];
                        match b {
                            ACK => debug!("ZAPI ACK received"),
                            NAK => warn!("ZAPI NAK received"),
                            CAN => warn!("ZAPI CAN received"),
                            SOF => {
                                buf.clear();
                                buf.push(b);
                            }
                            _ => {
                                buf.push(b);
                                // Try to parse once we have at least LEN + min bytes
                                if buf.len() >= 5 && buf.first() == Some(&SOF) {
                                    let payload = &buf[1..]; // skip SOF
                                    if !payload.is_empty() {
                                        let expected = payload[0] as usize + 2; // LEN byte + LEN + CS
                                        if payload.len() >= expected {
                                            match ZApiFrame::decode_after_sof(payload) {
                                                Ok((frame, _)) => {
                                                    // Send ACK
                                                    let mut g = inner.lock().await;
                                                    if let Some(w) = g.writer.as_mut() {
                                                        let _ = w.write_all(&[ACK]).await;
                                                    }
                                                    // Dispatch
                                                    match frame.frame_type {
                                                        FRAME_TYPE_RES => {
                                                            if let Some(p) =
                                                                g.pending.remove(&frame.cmd_id)
                                                            {
                                                                let _ = p.tx.send(frame);
                                                            }
                                                        }
                                                        FRAME_TYPE_REQ => {
                                                            Self::dispatch_req(&mut g, &frame);
                                                        }
                                                        _ => {}
                                                    }
                                                }
                                                Err(e) => warn!("ZAPI decode: {e}"),
                                            }
                                            buf.clear();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    /// Handle unsolicited REQ frames from the controller.
    fn dispatch_req(inner: &mut ZApiInner, frame: &ZApiFrame) {
        match frame.cmd_id {
            APPLICATION_COMMAND_HANDLER => {
                // rxStatus(1) | srcNodeId(1) | cmdLen(1) | cmdData(cmdLen)
                if frame.data.len() >= 3 {
                    let src_node_id = frame.data[1];
                    let cmd_len = frame.data[2] as usize;
                    if let Some(cmd_data) = frame.data.get(3..3 + cmd_len) {
                        debug!(
                            "ZAPI app command from node {src_node_id}: cc={:#04x} cmd={:#04x}",
                            cmd_data.first().copied().unwrap_or(0),
                            cmd_data.get(1).copied().unwrap_or(0)
                        );
                    }
                }
            }
            ADD_NODE_TO_NETWORK => {
                // bStatus(1) | bSource(1) | ...
                if frame.data.len() >= 2 {
                    let status = frame.data[0];
                    let node_id = frame.data[1];
                    if status == 0x05 && node_id > 0 {
                        // Node added
                        let node = ZWaveNode::new(node_id);
                        inner.nodes.insert(node_id, node);
                        debug!("ZAPI: node {node_id} added to network");
                    }
                }
            }
            REMOVE_NODE_FROM_NETWORK => {
                if frame.data.len() >= 2 {
                    let status = frame.data[0];
                    let node_id = frame.data[1];
                    if status == 0x06 && node_id > 0 {
                        inner.nodes.remove(&node_id);
                        let _ = inner.event_tx.try_send(HomeAutoEvent::DeviceLeft {
                            id: node_id.to_string(),
                            protocol: Protocol::ZWave,
                        });
                        debug!("ZAPI: node {node_id} removed from network");
                    }
                }
            }
            _ => {}
        }
    }
}

#[async_trait]
impl ZWaveController for ZWaveSerialController {
    async fn start(&self) -> HomeAutoResult<()> {
        let port = SerialStream::open(&tokio_serial::new(&self.port_path, self.baud_rate))
            .map_err(HomeAutoError::Serial)?;
        let (reader, writer) = tokio::io::split(port);
        self.inner.lock().await.writer = Some(writer);
        self.spawn_reader(reader).await;

        // Check connectivity
        let resp = self.send_request(GET_VERSION, vec![]).await?;
        debug!(
            "ZAPI connected: version={}",
            String::from_utf8_lossy(&resp.data)
        );

        // Load existing node list from GET_INIT_DATA
        let init = self.send_request(GET_INIT_DATA, vec![]).await?;
        // init.data: ver(1) | capabilities(1) | nodeListLen(1) | nodeList(29 bytes = 232 bits)
        if init.data.len() >= 3 {
            let node_list_len = init.data[2] as usize;
            let node_bytes = init.data.get(3..3 + node_list_len).unwrap_or(&[]);
            let mut inner = self.inner.lock().await;
            for (byte_idx, &byte) in node_bytes.iter().enumerate() {
                for bit in 0..8 {
                    if byte & (1 << bit) != 0 {
                        let node_id = (byte_idx * 8 + bit + 1) as NodeId;
                        inner
                            .nodes
                            .entry(node_id)
                            .or_insert_with(|| ZWaveNode::new(node_id));
                    }
                }
            }
            debug!("ZAPI: {} nodes discovered", inner.nodes.len());
        }
        Ok(())
    }

    async fn stop(&self) -> HomeAutoResult<()> {
        self.inner.lock().await.writer = None;
        Ok(())
    }

    async fn include_node(&self, timeout_secs: u8) -> HomeAutoResult<ZWaveNode> {
        // Start inclusion (ADD_NODE_ANY)
        let cb_id = self.next_callback_id().await;
        self.send_request(ADD_NODE_TO_NETWORK, vec![ADD_NODE_ANY, cb_id])
            .await?;
        // Wait for the node to be added (event arrives via APPLICATION_COMMAND_HANDLER)
        tokio::time::sleep(Duration::from_secs(timeout_secs as u64)).await;
        // Stop inclusion
        self.send_request(ADD_NODE_TO_NETWORK, vec![ADD_NODE_STOP, 0])
            .await?;
        // Return the most recently added node
        let inner = self.inner.lock().await;
        inner
            .nodes
            .values()
            .max_by_key(|n| n.node_id)
            .cloned()
            .ok_or_else(|| HomeAutoError::ZWaveController("no node was added".into()))
    }

    async fn exclude_node(&self, timeout_secs: u8) -> HomeAutoResult<()> {
        let cb_id = self.next_callback_id().await;
        self.send_request(REMOVE_NODE_FROM_NETWORK, vec![REMOVE_NODE_ANY, cb_id])
            .await?;
        tokio::time::sleep(Duration::from_secs(timeout_secs as u64)).await;
        self.send_request(REMOVE_NODE_FROM_NETWORK, vec![REMOVE_NODE_STOP, 0])
            .await?;
        Ok(())
    }

    async fn nodes(&self) -> HomeAutoResult<Vec<ZWaveNode>> {
        Ok(self.inner.lock().await.nodes.values().cloned().collect())
    }

    async fn send_cc(&self, node_id: NodeId, cc: CommandClass, data: &[u8]) -> HomeAutoResult<()> {
        let cb_id = self.next_callback_id().await;
        let tx_opts = TRANSMIT_OPTION_ACK | TRANSMIT_OPTION_AUTO_ROUTE | TRANSMIT_OPTION_EXPLORE;
        // SEND_DATA payload: nodeId(1) | dataLen(1) | cc(1) | data... | txOptions(1) | callbackId(1)
        let mut payload = vec![node_id, (1 + data.len()) as u8, cc.id()];
        payload.extend_from_slice(data);
        payload.push(tx_opts);
        payload.push(cb_id);

        let resp = self.send_request(SEND_DATA, payload).await?;
        let send_ok = resp.data.first().copied().unwrap_or(0);
        if send_ok == 0 {
            return Err(HomeAutoError::ZWaveTransmit {
                node_id,
                msg: "SEND_DATA rejected".into(),
            });
        }
        Ok(())
    }

    fn events(&self) -> crate::BoxStream<'static, HomeAutoEvent> {
        let inner = Arc::clone(&self.inner);
        let (new_tx, mut rx) = mpsc::channel::<HomeAutoEvent>(64);
        tokio::spawn(async move {
            inner.lock().await.event_tx = new_tx;
        });
        Box::pin(async_stream::stream! {
            while let Some(event) = rx.recv().await {
                yield event;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zapi_xor_checksum_known_vector() {
        // From ZAPI spec: SOF | LEN=3 | TYPE=0x00 | CMD=0x07 | ...
        // CS = 0xFF ^ 3 ^ 0x00 ^ 0x07 = 0xFF ^ 3 ^ 7
        let data: &[u8] = &[0x03, 0x00, 0x07]; // LEN | TYPE | CMD_ID
        let cs = checksum(data);
        // reason: keep the explicit `^ 0x00` for symmetry with the spec
        // citation in the comment block above (LEN ^ TYPE ^ CMD_ID).
        #[allow(clippy::identity_op)]
        let expected = 0xFF ^ 0x03 ^ 0x00 ^ 0x07;
        assert_eq!(cs, expected);
    }

    #[test]
    fn zapi_single_byte_ack_nak_can() {
        assert_eq!(ACK, 0x06);
        assert_eq!(NAK, 0x15);
        assert_eq!(CAN, 0x18);
    }

    #[test]
    fn zapi_frame_encode_send_data() {
        // GET_VERSION request: SOF | LEN=2 | REQ | 0x15 | CS
        let frame = ZApiFrame::new_request(GET_VERSION, vec![]);
        let encoded = frame.encode();
        assert_eq!(encoded[0], SOF);
        assert_eq!(encoded[1], 2); // LEN = TYPE + CMD_ID
        assert_eq!(encoded[2], FRAME_TYPE_REQ);
        assert_eq!(encoded[3], GET_VERSION);
        assert_eq!(encoded[4], checksum(&encoded[1..4]));
    }

    #[test]
    fn zapi_frame_decode_app_command_handler() {
        // Construct a minimal APPLICATION_COMMAND_HANDLER RES frame
        let cmd_id = APPLICATION_COMMAND_HANDLER;
        let data: Vec<u8> = vec![0x00, 0x05, 0x03, 0x25, 0x03, 0xFF]; // rxStat|src|len|SWITCH_BINARY|REPORT|value
        let frame = ZApiFrame {
            frame_type: FRAME_TYPE_REQ,
            cmd_id,
            data: data.clone(),
        };
        let encoded = frame.encode();
        // Decode starting after SOF
        let (decoded, _) = ZApiFrame::decode_after_sof(&encoded[1..]).unwrap();
        assert_eq!(decoded.cmd_id, APPLICATION_COMMAND_HANDLER);
        assert_eq!(decoded.data, data);
    }

    #[test]
    fn zapi_frame_roundtrip() {
        let data = vec![0x05, 0xAA, 0xBB];
        let frame = ZApiFrame::new_request(SEND_DATA, data.clone());
        let encoded = frame.encode();
        let (decoded, consumed) = ZApiFrame::decode_after_sof(&encoded[1..]).unwrap();
        assert_eq!(decoded.cmd_id, SEND_DATA);
        assert_eq!(decoded.data, data);
        assert_eq!(consumed, encoded.len() - 1); // minus SOF
    }

    #[test]
    fn zapi_checksum_mismatch() {
        let mut encoded = ZApiFrame::new_request(GET_VERSION, vec![]).encode();
        // Corrupt checksum
        let last = encoded.len() - 1;
        encoded[last] ^= 0xFF;
        assert!(ZApiFrame::decode_after_sof(&encoded[1..]).is_err());
    }
}
