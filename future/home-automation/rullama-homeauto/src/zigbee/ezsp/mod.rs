/// ASH (Asynchronous Serial Host) framing layer for EZSP over UART.
pub mod ash;
/// EZSP v8 command-ID constants + typed payload helpers.
pub mod commands;
/// EZSP frame encode/decode (sequence, frame-control, cmd-ID, params).
pub mod frame;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::time::timeout;
use tokio_serial::SerialStream;
use tracing::{debug, error, warn};

use super::ZigbeeCoordinator;
use super::types::{ZigbeeAddr, ZigbeeAttrId, ZigbeeClusterId, ZigbeeDevice, ZigbeeDeviceKind};
use crate::error::{HomeAutoError, HomeAutoResult};
use crate::types::{AttributeValue, HomeAutoEvent, HomeDevice, Protocol};
use frame::EzspFrame;

/// Pending synchronous EZSP response correlation entry.
struct Pending {
    tx: oneshot::Sender<EzspFrame>,
}

struct EzspInner {
    /// Sequence counter for outgoing frames (wraps 0–255).
    seq: u8,
    /// Pending synchronous response waiters keyed by seq.
    pending: HashMap<u8, Pending>,
    /// Writer half of the serial port.
    writer: Option<tokio::io::WriteHalf<SerialStream>>,
    /// Channel to broadcast incoming events to subscribers.
    event_tx: mpsc::Sender<HomeAutoEvent>,
    /// Known device table (IEEE address → device).
    devices: HashMap<u64, ZigbeeDevice>,
}

/// Silicon Labs EZSP v8 coordinator over ASH/UART.
///
/// Targets Zigbee 3.0 devices via EFR32-based USB dongles
/// (Sonoff Zigbee 3.0 USB Dongle Plus, Aeotec USB 7, etc.).
pub struct EzspCoordinator {
    port_path: String,
    baud_rate: u32,
    inner: Arc<Mutex<EzspInner>>,
}

impl EzspCoordinator {
    /// Create a coordinator for `port` (e.g. `/dev/ttyUSB0`) at `baud_rate` (usually 115200).
    /// Call [`ZigbeeCoordinator::start`] to open the port and connect to the NCP.
    pub fn new(port: impl Into<String>, baud_rate: u32) -> Self {
        let (event_tx, _) = mpsc::channel(64);
        Self {
            port_path: port.into(),
            baud_rate,
            inner: Arc::new(Mutex::new(EzspInner {
                seq: 0,
                pending: HashMap::new(),
                writer: None,
                event_tx,
                devices: HashMap::new(),
            })),
        }
    }

    /// Send an EZSP command and await the NCP's synchronous response (2 s timeout).
    async fn ezsp_cmd(&self, cmd_id: u16, params: Vec<u8>) -> HomeAutoResult<EzspFrame> {
        let (_seq, rx) = {
            let mut inner = self.inner.lock().await;
            let seq = inner.seq;
            inner.seq = inner.seq.wrapping_add(1);
            let (tx, rx) = oneshot::channel();
            inner.pending.insert(seq, Pending { tx });
            let frame = EzspFrame::command(seq, cmd_id, params);
            let wire = ash::encode_frame(&frame.encode());
            match inner.writer.as_mut() {
                Some(w) => w.write_all(&wire).await.map_err(HomeAutoError::Io)?,
                None => {
                    inner.pending.remove(&seq);
                    return Err(HomeAutoError::ZigbeeCoordinator(
                        "coordinator not started — call start() first".into(),
                    ));
                }
            }
            (seq, rx)
        };

        timeout(Duration::from_secs(2), rx)
            .await
            .map_err(|_| HomeAutoError::Timeout)?
            .map_err(|_| HomeAutoError::ChannelClosed)
    }

    /// Spawn the serial reader task that dispatches incoming ASH/EZSP frames.
    async fn spawn_reader(&self, mut reader: tokio::io::ReadHalf<SerialStream>) {
        let inner = Arc::clone(&self.inner);
        tokio::spawn(async move {
            let mut buf: Vec<u8> = Vec::with_capacity(256);
            let mut byte = [0u8; 1];
            loop {
                match reader.read_exact(&mut byte).await {
                    Err(e) => {
                        error!("EZSP serial read error: {e}");
                        break;
                    }
                    Ok(_) => {
                        let b = byte[0];
                        if b == ash::FLAG {
                            if buf.len() >= 2 {
                                match ash::decode_frame(&buf) {
                                    Ok(payload) => match EzspFrame::decode(&payload) {
                                        Ok(frame) => {
                                            let mut g = inner.lock().await;
                                            // Wake any pending waiter for this sequence
                                            if let Some(pending) = g.pending.remove(&frame.seq) {
                                                let _ = pending.tx.send(frame.clone());
                                            }
                                            Self::dispatch_callback(&mut g, &frame);
                                        }
                                        Err(e) => warn!("EZSP decode error: {e}"),
                                    },
                                    Err(e) => warn!("ASH decode error: {e}"),
                                }
                            }
                            buf.clear();
                        } else {
                            buf.push(b);
                        }
                    }
                }
            }
        });
    }

    /// Dispatch a NCP→host callback frame to the event broadcast channel.
    fn dispatch_callback(inner: &mut EzspInner, frame: &EzspFrame) {
        match frame.cmd_id {
            commands::TRUST_CENTER_JOIN_HANDLER => {
                // params: newNodeId(2) | newNodeEui64(8) | status(1) | policyDecision(1) | parentId(2)
                if frame.params.len() >= 13 {
                    let nwk = u16::from_le_bytes([frame.params[0], frame.params[1]]);
                    let mut ieee_bytes = [0u8; 8];
                    ieee_bytes.copy_from_slice(&frame.params[2..10]);
                    let ieee = u64::from_le_bytes(ieee_bytes);
                    let addr = ZigbeeAddr::new(ieee, nwk);
                    let dev = ZigbeeDevice::new(addr, ZigbeeDeviceKind::Other(0));
                    inner.devices.insert(ieee, dev);
                    let home_dev = HomeDevice {
                        id: format!("{ieee:016x}"),
                        name: None,
                        protocol: Protocol::Zigbee,
                        manufacturer: None,
                        model: None,
                        firmware_version: None,
                        capabilities: Vec::new(),
                    };
                    let _ = inner
                        .event_tx
                        .try_send(HomeAutoEvent::DeviceJoined(home_dev));
                }
            }
            commands::INCOMING_MESSAGE_HANDLER => {
                if let Some((_, cluster_id, src_nwk, _, payload)) =
                    commands::decode_incoming_message(&frame.params)
                {
                    debug!(
                        "EZSP incoming: cluster={cluster_id:#06x} src={src_nwk:#06x} len={}",
                        payload.len()
                    );
                }
            }
            _ => {}
        }
    }
}

#[async_trait]
impl ZigbeeCoordinator for EzspCoordinator {
    async fn start(&self) -> HomeAutoResult<()> {
        let port = SerialStream::open(&tokio_serial::new(&self.port_path, self.baud_rate))
            .map_err(HomeAutoError::Serial)?;
        let (reader, writer) = tokio::io::split(port);
        self.inner.lock().await.writer = Some(writer);
        self.spawn_reader(reader).await;

        // Reset the NCP first
        let rst = ash::build_rst();
        self.inner
            .lock()
            .await
            .writer
            .as_mut()
            .unwrap()
            .write_all(&rst)
            .await
            .map_err(HomeAutoError::Io)?;
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Negotiate EZSP version (must be the first command)
        let resp = self.ezsp_cmd(commands::VERSION, vec![8]).await?;
        if resp.params.first().copied() != Some(8) {
            return Err(HomeAutoError::ZigbeeCoordinator(
                "EZSP version negotiation failed — NCP does not support v8".into(),
            ));
        }
        debug!("EZSP coordinator started on {}", self.port_path);
        Ok(())
    }

    async fn stop(&self) -> HomeAutoResult<()> {
        self.inner.lock().await.writer = None;
        Ok(())
    }

    async fn permit_join(&self, duration_secs: u8) -> HomeAutoResult<()> {
        let resp = self
            .ezsp_cmd(commands::PERMIT_JOINING, vec![duration_secs])
            .await?;
        let status = resp.params.first().copied().unwrap_or(0xFF);
        if status != commands::STATUS_SUCCESS {
            return Err(HomeAutoError::EzspStatus {
                status,
                msg: "permit_joining rejected".into(),
            });
        }
        Ok(())
    }

    async fn devices(&self) -> HomeAutoResult<Vec<ZigbeeDevice>> {
        Ok(self.inner.lock().await.devices.values().cloned().collect())
    }

    async fn read_attribute(
        &self,
        addr: ZigbeeAddr,
        cluster: ZigbeeClusterId,
        attr: ZigbeeAttrId,
    ) -> HomeAutoResult<AttributeValue> {
        // ZCL read attribute command (cmd=0x00): frame_control | seq | cmd | attr_id(2)
        let zcl_seq = 0u8;
        let mut zcl = vec![0x00u8, zcl_seq, 0x00];
        zcl.extend_from_slice(&attr.to_le_bytes());

        let payload = commands::send_unicast_payload(
            addr.nwk, 0x01, 0x01, cluster, 0x0104, zcl_seq, zcl_seq, &zcl,
        );
        let resp = self.ezsp_cmd(commands::SEND_UNICAST, payload).await?;
        let status = resp.params.first().copied().unwrap_or(0xFF);
        if status != commands::STATUS_SUCCESS {
            return Err(HomeAutoError::ZigbeeAttribute {
                cluster,
                attr,
                msg: format!("send status {status:#04x}"),
            });
        }
        // The attribute value arrives asynchronously via INCOMING_MESSAGE_HANDLER.
        // Subscribe to events() to receive it.
        Ok(AttributeValue::Null)
    }

    async fn write_attribute(
        &self,
        addr: ZigbeeAddr,
        cluster: ZigbeeClusterId,
        attr: ZigbeeAttrId,
        value: AttributeValue,
    ) -> HomeAutoResult<()> {
        let (zcl_type, encoded) = encode_zcl_value(&value)?;
        let zcl_seq = 1u8;
        let mut zcl = vec![0x00u8, zcl_seq, 0x02]; // cmd=0x02 write attr
        zcl.extend_from_slice(&attr.to_le_bytes());
        zcl.push(zcl_type);
        zcl.extend_from_slice(&encoded);

        let payload = commands::send_unicast_payload(
            addr.nwk, 0x01, 0x01, cluster, 0x0104, zcl_seq, zcl_seq, &zcl,
        );
        let resp = self.ezsp_cmd(commands::SEND_UNICAST, payload).await?;
        let status = resp.params.first().copied().unwrap_or(0xFF);
        if status != commands::STATUS_SUCCESS {
            return Err(HomeAutoError::ZigbeeAttribute {
                cluster,
                attr,
                msg: format!("write status {status:#04x}"),
            });
        }
        Ok(())
    }

    async fn invoke_command(
        &self,
        addr: ZigbeeAddr,
        cluster: ZigbeeClusterId,
        cmd: u8,
        cmd_payload: &[u8],
    ) -> HomeAutoResult<()> {
        // ZCL cluster-specific command: frame_control=0x01 | seq | cmd | payload
        let zcl_seq = 2u8;
        let mut zcl = vec![0x01u8, zcl_seq, cmd];
        zcl.extend_from_slice(cmd_payload);

        let payload = commands::send_unicast_payload(
            addr.nwk, 0x01, 0x01, cluster, 0x0104, zcl_seq, zcl_seq, &zcl,
        );
        let resp = self.ezsp_cmd(commands::SEND_UNICAST, payload).await?;
        let status = resp.params.first().copied().unwrap_or(0xFF);
        if status != commands::STATUS_SUCCESS {
            return Err(HomeAutoError::ZigbeeCoordinator(format!(
                "send_command status {status:#04x}"
            )));
        }
        Ok(())
    }

    fn events(&self) -> crate::BoxStream<'static, HomeAutoEvent> {
        let inner = Arc::clone(&self.inner);
        let (new_tx, mut rx) = mpsc::channel::<HomeAutoEvent>(64);
        // Replace the stored sender so future events go to our new channel
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

/// Encode an [`AttributeValue`] to a `(ZCL type id, encoded bytes)` pair.
fn encode_zcl_value(value: &AttributeValue) -> HomeAutoResult<(u8, Vec<u8>)> {
    Ok(match value {
        AttributeValue::Bool(v) => (0x10, vec![*v as u8]),
        AttributeValue::U8(v) => (0x20, vec![*v]),
        AttributeValue::U16(v) => (0x21, v.to_le_bytes().to_vec()),
        AttributeValue::U32(v) => (0x23, v.to_le_bytes().to_vec()),
        AttributeValue::I8(v) => (0x28, vec![*v as u8]),
        AttributeValue::I16(v) => (0x29, v.to_le_bytes().to_vec()),
        AttributeValue::I32(v) => (0x2B, v.to_le_bytes().to_vec()),
        AttributeValue::String(s) => {
            let bytes = s.as_bytes();
            let mut out = vec![bytes.len() as u8];
            out.extend_from_slice(bytes);
            (0x42, out)
        }
        AttributeValue::Bytes(b) => {
            let mut out = vec![b.len() as u8];
            out.extend_from_slice(b);
            (0x41, out)
        }
        _ => {
            return Err(HomeAutoError::Unsupported(format!(
                "ZCL encoding not implemented for {value:?}"
            )));
        }
    })
}
