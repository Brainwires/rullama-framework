/// ZNP Monitor-and-Test (MT) command-ID constants + payload helpers.
pub mod commands;
/// ZNP frame encode/decode (SOF, LEN, TYPE/subsystem, CMD, payload, FCS).
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
use frame::{TYPE_AREQ, TYPE_SRSP, ZnpFrame};

/// Pending synchronous SRSP waiter keyed by (subsystem, cmd).
struct PendingSrsp {
    tx: oneshot::Sender<ZnpFrame>,
}

struct ZnpInner {
    /// Pending SRSP waiters keyed by (subsystem, cmd).
    pending: HashMap<(u8, u8), PendingSrsp>,
    /// Writer half of the serial port.
    writer: Option<tokio::io::WriteHalf<SerialStream>>,
    /// Channel for broadcasting AREQ callbacks as [`HomeAutoEvent`]s.
    event_tx: mpsc::Sender<HomeAutoEvent>,
    /// Known device table.
    devices: HashMap<u64, ZigbeeDevice>,
}

/// TI Z-Stack 3.x ZNP coordinator.
///
/// Targets Zigbee 3.0 devices via CC2652/CC1352-based USB dongles (e.g. Sonoff Zigbee 3.0
/// USB Dongle-E, TI LaunchPad with Z-Stack 3 firmware).
pub struct ZnpCoordinator {
    port_path: String,
    baud_rate: u32,
    inner: Arc<Mutex<ZnpInner>>,
}

impl ZnpCoordinator {
    /// Create a coordinator for `port` (e.g. `/dev/ttyUSB0`) at `baud_rate` (usually 115200).
    pub fn new(port: impl Into<String>, baud_rate: u32) -> Self {
        let (event_tx, _) = mpsc::channel(64);
        Self {
            port_path: port.into(),
            baud_rate,
            inner: Arc::new(Mutex::new(ZnpInner {
                pending: HashMap::new(),
                writer: None,
                event_tx,
                devices: HashMap::new(),
            })),
        }
    }

    /// Send a SREQ and await the SRSP (2 s timeout).
    async fn sreq(&self, subsystem: u8, cmd: u8, payload: Vec<u8>) -> HomeAutoResult<ZnpFrame> {
        let rx = {
            let mut inner = self.inner.lock().await;
            let (tx, rx) = oneshot::channel();
            inner.pending.insert((subsystem, cmd), PendingSrsp { tx });
            let frame = ZnpFrame::sreq(subsystem, cmd, payload);
            let wire = frame.encode();
            match inner.writer.as_mut() {
                Some(w) => w.write_all(&wire).await.map_err(HomeAutoError::Io)?,
                None => {
                    inner.pending.remove(&(subsystem, cmd));
                    return Err(HomeAutoError::ZigbeeCoordinator(
                        "coordinator not started — call start() first".into(),
                    ));
                }
            }
            rx
        };

        timeout(Duration::from_secs(2), rx)
            .await
            .map_err(|_| HomeAutoError::Timeout)?
            .map_err(|_| HomeAutoError::ChannelClosed)
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
                        error!("ZNP serial read error: {e}");
                        break;
                    }
                    Ok(_) => {
                        let b = byte[0];
                        if b == frame::SOF {
                            buf.clear();
                            buf.push(b);
                        } else {
                            buf.push(b);
                            if buf.len() >= 5 {
                                let expected_len = buf.get(1).copied().unwrap_or(0) as usize + 5;
                                if buf.len() == expected_len {
                                    match ZnpFrame::decode(&buf) {
                                        Ok((f, _)) => {
                                            let mut g = inner.lock().await;
                                            match f.msg_type {
                                                TYPE_SRSP => {
                                                    if let Some(p) =
                                                        g.pending.remove(&(f.subsystem, f.cmd))
                                                    {
                                                        let _ = p.tx.send(f.clone());
                                                    }
                                                }
                                                TYPE_AREQ => {
                                                    Self::dispatch_areq(&mut g, &f);
                                                }
                                                _ => {}
                                            }
                                        }
                                        Err(e) => warn!("ZNP decode error: {e}"),
                                    }
                                    buf.clear();
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    /// Handle an asynchronous AREQ from the NCP.
    fn dispatch_areq(inner: &mut ZnpInner, frame: &ZnpFrame) {
        match (frame.subsystem & 0x1F, frame.cmd) {
            // ZDO end-device announce: NWK(2)|IEEE(8)|capabilities(1)
            (0x05, commands::ZDO_END_DEVICE_ANNCE_IND) => {
                if frame.payload.len() >= 11 {
                    let nwk = u16::from_le_bytes([frame.payload[0], frame.payload[1]]);
                    let mut ieee_bytes = [0u8; 8];
                    ieee_bytes.copy_from_slice(&frame.payload[2..10]);
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
            // ZDO leave indication
            (0x05, commands::ZDO_LEAVE_IND) => {
                if frame.payload.len() >= 8 {
                    let mut ieee_bytes = [0u8; 8];
                    ieee_bytes.copy_from_slice(&frame.payload[0..8]);
                    let ieee = u64::from_le_bytes(ieee_bytes);
                    inner.devices.remove(&ieee);
                    let _ = inner.event_tx.try_send(HomeAutoEvent::DeviceLeft {
                        id: format!("{ieee:016x}"),
                        protocol: Protocol::Zigbee,
                    });
                }
            }
            // AF incoming message
            (0x04, commands::AF_INCOMING_MSG) => {
                if let Some((_, cluster_id, src_addr, _, _, _, data)) =
                    commands::decode_af_incoming(&frame.payload)
                {
                    debug!(
                        "ZNP AF incoming: cluster={cluster_id:#06x} src={src_addr:#06x} len={}",
                        data.len()
                    );
                }
            }
            _ => {}
        }
    }
}

#[async_trait]
impl ZigbeeCoordinator for ZnpCoordinator {
    async fn start(&self) -> HomeAutoResult<()> {
        let port = SerialStream::open(&tokio_serial::new(&self.port_path, self.baud_rate))
            .map_err(HomeAutoError::Serial)?;
        let (reader, writer) = tokio::io::split(port);
        self.inner.lock().await.writer = Some(writer);
        self.spawn_reader(reader).await;

        // Ping to verify connectivity
        let resp = self.sreq(commands::SYS, commands::SYS_PING, vec![]).await?;
        if resp.payload.len() < 2 {
            return Err(HomeAutoError::ZigbeeCoordinator(
                "SYS_PING response too short".into(),
            ));
        }
        debug!(
            "ZNP coordinator started on {}, capabilities={:#06x}",
            self.port_path,
            u16::from_le_bytes([resp.payload[0], resp.payload[1]])
        );

        // Start the BDB commissioning (coordinator mode)
        let startup_payload = commands::startup_payload(100);
        let resp = self
            .sreq(
                commands::ZDO,
                commands::ZDO_STARTUP_FROM_APP,
                startup_payload,
            )
            .await?;
        let status = resp.payload.first().copied().unwrap_or(0xFF);
        debug!("ZDO startup status: {status:#04x}");
        Ok(())
    }

    async fn stop(&self) -> HomeAutoResult<()> {
        self.inner.lock().await.writer = None;
        Ok(())
    }

    async fn permit_join(&self, duration_secs: u8) -> HomeAutoResult<()> {
        let payload = commands::permit_join_payload(0xFFFC, duration_secs);
        let resp = self
            .sreq(commands::ZDO, commands::ZDO_PERMIT_JOIN_REQ, payload)
            .await?;
        let status = resp.payload.first().copied().unwrap_or(0xFF);
        if status != commands::ZNP_STATUS_SUCCESS {
            return Err(HomeAutoError::ZnpStatus {
                status,
                msg: "permit_join rejected".into(),
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
        // ZCL read attribute request
        let zcl_seq = 0u8;
        let mut zcl = vec![0x00u8, zcl_seq, 0x00]; // FC | seq | cmd=read_attr
        zcl.extend_from_slice(&attr.to_le_bytes());

        let payload = commands::af_data_request(addr.nwk, 0x01, 0x01, cluster, zcl_seq, &zcl);
        let resp = self
            .sreq(commands::AF, commands::AF_DATA_REQUEST, payload)
            .await?;
        let status = resp.payload.first().copied().unwrap_or(0xFF);
        if status != commands::ZNP_STATUS_SUCCESS {
            return Err(HomeAutoError::ZigbeeAttribute {
                cluster,
                attr,
                msg: format!("AF_DATA_REQUEST status {status:#04x}"),
            });
        }
        Ok(AttributeValue::Null)
    }

    async fn write_attribute(
        &self,
        addr: ZigbeeAddr,
        cluster: ZigbeeClusterId,
        attr: ZigbeeAttrId,
        value: AttributeValue,
    ) -> HomeAutoResult<()> {
        // Encode ZCL type and value
        let (zcl_type, encoded_val) = encode_zcl_value(&value)?;
        let zcl_seq = 1u8;
        let mut zcl = vec![0x00u8, zcl_seq, 0x02]; // cmd=write_attr
        zcl.extend_from_slice(&attr.to_le_bytes());
        zcl.push(zcl_type);
        zcl.extend_from_slice(&encoded_val);

        let payload = commands::af_data_request(addr.nwk, 0x01, 0x01, cluster, zcl_seq, &zcl);
        let resp = self
            .sreq(commands::AF, commands::AF_DATA_REQUEST, payload)
            .await?;
        let status = resp.payload.first().copied().unwrap_or(0xFF);
        if status != commands::ZNP_STATUS_SUCCESS {
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
        let zcl_seq = 2u8;
        let mut zcl = vec![0x01u8, zcl_seq, cmd]; // frame_control=cluster-specific | seq | cmd
        zcl.extend_from_slice(cmd_payload);

        let payload = commands::af_data_request(addr.nwk, 0x01, 0x01, cluster, zcl_seq, &zcl);
        let resp = self
            .sreq(commands::AF, commands::AF_DATA_REQUEST, payload)
            .await?;
        let status = resp.payload.first().copied().unwrap_or(0xFF);
        if status != commands::ZNP_STATUS_SUCCESS {
            return Err(HomeAutoError::ZigbeeCoordinator(format!(
                "invoke_command AF status {status:#04x}"
            )));
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
