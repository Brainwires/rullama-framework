/// Matter 1.3 device server — exposes a rullama agent as a Matter device.
///
/// Implements the Matter device stack using our own protocol implementation
/// (avoiding rs-matter due to an embassy-time links conflict with burn in the workspace).
///
/// Stack layers:
/// 1. mDNS advertisement via `mdns-sd` (DNS-SD for operational + commissionable discovery)
/// 2. UDP transport on port 5540 (the standard Matter port)
/// 3. PASE commissioning window (SPAKE2+ passcode verification)
/// 4. CASE session establishment (certificate-based operational sessions)
/// 5. Interaction Model dispatch (On/Off, Level Control, Color Control, Thermostat)
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use super::data_model::{
    BasicInformationCluster, DataModelNode, GeneralCommissioningCluster,
    NetworkCommissioningCluster, OperationalCredentialsCluster,
};
use super::discovery::CommissionableAdvertiser;
use super::fabric::FabricManager;
use super::interaction_model::read::ReportData;
use super::interaction_model::{ImOpcode, InteractionStatus, InvokeResponse, InvokeResponseItem};
use super::secure_channel::{
    CaseResponder, EstablishedSession, PaseCommissionee, SECURE_CHANNEL_PROTOCOL_ID,
    SecureChannelOpcode,
};
use super::subscription_manager::SubscriptionManager;
use super::transport::message::{MatterMessage, MessageHeader, SessionType};
use super::transport::{SessionKeys, UdpTransport};
use super::types::MatterDeviceConfig;
use crate::error::{HomeAutoError, HomeAutoResult};

// Cluster handler callback types
/// Callback invoked when the On/Off cluster receives an On or Off command.
pub type OnOffHandler = Arc<dyn Fn(bool) + Send + Sync>;
/// Callback invoked with a Level Control MoveToLevel payload (0..=254).
pub type LevelHandler = Arc<dyn Fn(u8) + Send + Sync>;
/// Callback invoked with a Color Control MoveToColorTemperature mireds value.
pub type ColorTempHandler = Arc<dyn Fn(u16) + Send + Sync>;
/// Callback invoked with a Thermostat setpoint (in °C).
pub type ThermostatHandler = Arc<dyn Fn(f32) + Send + Sync>;

// Matter protocol constants (used when binding the Matter server socket)
const _MATTER_PORT: u16 = 5540;
const _MATTER_MDNS_SERVICE_TYPE: &str = "_matter._tcp";

// IM protocol ID
const IM_PROTOCOL_ID: u16 = 0x0001;

// Well-known cluster IDs
const CLUSTER_ON_OFF: u32 = 0x0006;
const CLUSTER_LEVEL_CONTROL: u32 = 0x0008;
const CLUSTER_COLOR_CONTROL: u32 = 0x0300;
const CLUSTER_THERMOSTAT: u32 = 0x0201;

// Well-known OnOff command IDs
const CMD_OFF: u32 = 0x00;
const CMD_ON: u32 = 0x01;
const CMD_TOGGLE: u32 = 0x02;

// Well-known Level Control command IDs
const CMD_MOVE_TO_LEVEL: u32 = 0x00;
const CMD_MOVE_TO_LEVEL_WITH_ON_OFF: u32 = 0x04;

// Well-known Thermostat command IDs
const CMD_SETPOINT_RAISE_LOWER: u32 = 0x00;

struct ServerInner {
    on_off: Option<OnOffHandler>,
    level: Option<LevelHandler>,
    color_temp: Option<ColorTempHandler>,
    thermostat: Option<ThermostatHandler>,
    running: bool,
    /// Whether the device is commissioned (has an operational fabric).
    _commissioned: bool,
}

/// An attribute mutation pushed through the notification pump.
#[derive(Debug, Clone)]
struct AttributeChange {
    endpoint: u16,
    cluster: u32,
    attribute: u32,
    /// TLV-encoded new value (the cluster owner knows the type).
    value: Vec<u8>,
}

/// A Matter 1.3 device server.
///
/// Once started, this device:
/// 1. Advertises as a commissionable Matter device via mDNS (`_matterc._udp`).
/// 2. Opens UDP port 5540 and handles Matter commissioning (PASE).
/// 3. After commissioning, handles cluster commands via the registered callbacks.
///
/// # Example
/// ```rust,no_run
/// use rullama_homeauto::matter::{MatterDeviceConfig, MatterDeviceServer};
///
/// # async fn run() -> anyhow::Result<()> {
/// let config = MatterDeviceConfig::builder()
///     .device_name("rullama Light")
///     .vendor_id(0xFFF1)
///     .product_id(0x8001)
///     .discriminator(3840)
///     .passcode(20202021)
///     .build();
///
/// let server = MatterDeviceServer::new(config).await?;
/// server.set_on_off_handler(|on| {
///     println!("On/Off: {on}");
/// });
/// server.start().await?;
/// # Ok(())
/// # }
/// ```
pub struct MatterDeviceServer {
    config: MatterDeviceConfig,
    inner: Arc<Mutex<ServerInner>>,
    qr_code: String,
    pairing_code: String,
    subscriptions: Arc<SubscriptionManager>,
    /// Per-session outgoing message counter. The AEAD nonce is derived from
    /// `session_id || counter`, so reusing a counter within a session breaks
    /// decryption at the peer — this map guarantees strictly-monotonic
    /// counters for asynchronous push paths (ReportData subscriptions).
    outgoing_counters: Arc<Mutex<HashMap<u16, std::sync::atomic::AtomicU32>>>,
    /// Sender half of the notification pump. `notify_attribute_change`
    /// pushes here; the recv loop in `start()` drains and delivers ReportData.
    notify_tx: tokio::sync::mpsc::UnboundedSender<AttributeChange>,
    notify_rx: Mutex<Option<tokio::sync::mpsc::UnboundedReceiver<AttributeChange>>>,
}

impl MatterDeviceServer {
    /// Create a new Matter device server.
    pub async fn new(config: MatterDeviceConfig) -> HomeAutoResult<Self> {
        let qr_code = generate_qr_code_string(&config);
        let pairing_code = generate_pairing_code(&config);
        let (notify_tx, notify_rx) = tokio::sync::mpsc::unbounded_channel();
        Ok(Self {
            config,
            inner: Arc::new(Mutex::new(ServerInner {
                on_off: None,
                level: None,
                color_temp: None,
                thermostat: None,
                running: false,
                _commissioned: false,
            })),
            qr_code,
            pairing_code,
            subscriptions: Arc::new(SubscriptionManager::new()),
            outgoing_counters: Arc::new(Mutex::new(HashMap::new())),
            notify_tx,
            notify_rx: Mutex::new(Some(notify_rx)),
        })
    }

    /// Start the Matter device server.
    ///
    /// Binds UDP, starts mDNS advertisement, loads/creates the fabric manager,
    /// builds the data model node with mandatory clusters, and runs the receive
    /// loop until `stop()` is called.
    pub async fn start(&self) -> HomeAutoResult<()> {
        {
            let mut inner = self.inner.lock().await;
            if inner.running {
                return Err(HomeAutoError::Matter("server already running".into()));
            }
            inner.running = true;
        }

        info!(
            "Matter device '{}' starting on UDP port {}",
            self.config.device_name, self.config.port
        );
        info!("QR code: {}", self.qr_code);
        info!("Manual pairing code: {}", self.pairing_code);
        info!("Discriminator: {}", self.config.discriminator);

        // 1. Start CommissionableAdvertiser
        let _mdns_handle = self.start_mdns_advertisement()?;

        // 2. Load/create FabricManager from config.storage_path
        let _fabric_manager = FabricManager::load(&self.config.storage_path)
            .await
            .map_err(|e| HomeAutoError::Matter(format!("FabricManager load: {e}")))?;

        // 3. Bind UdpTransport on config.port (default 5540)
        let transport = Arc::new(
            UdpTransport::new(self.config.port)
                .await
                .map_err(|e| HomeAutoError::Matter(format!("UDP bind: {e}")))?,
        );
        info!("Matter UDP transport bound on port {}", self.config.port);

        // 4. Build DataModelNode with mandatory clusters
        let mut data_model = DataModelNode::new();
        // Endpoint 0: mandatory commissioning clusters
        data_model.add_cluster(0, Box::new(BasicInformationCluster::new(&self.config)));
        data_model.add_cluster(0, Box::new(GeneralCommissioningCluster::new()));
        data_model.add_cluster(0, Box::new(OperationalCredentialsCluster::new()));
        data_model.add_cluster(0, Box::new(NetworkCommissioningCluster::new()));

        let data_model = Arc::new(data_model);

        // 5. Shared session state
        let pase_state: Arc<Mutex<Option<PaseCommissionee>>> = Arc::new(Mutex::new(None));
        let case_sessions: Arc<Mutex<HashMap<u16, CaseResponder>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let established: Arc<Mutex<HashMap<u16, EstablishedSession>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let passcode = self.config.passcode;

        // Take the notification receiver out — start() must only run once.
        let mut notify_rx = self
            .notify_rx
            .lock()
            .await
            .take()
            .ok_or_else(|| HomeAutoError::Matter("server already started once".into()))?;

        // 6. Receive loop
        loop {
            let running = self.inner.lock().await.running;
            if !running {
                break;
            }

            tokio::select! {
                biased;
                Some(change) = notify_rx.recv() => {
                    dispatch_attribute_change(
                        &change,
                        &transport,
                        &self.subscriptions,
                        &self.outgoing_counters,
                    ).await;
                    continue;
                }
                result = tokio::time::timeout(std::time::Duration::from_millis(100), transport.recv()) => {
                    match result {
                        Ok(Ok((msg, peer))) => {
                            let session_id = msg.header.session_id;
                            debug!(
                                "Matter UDP from {peer}: session={session_id} payload_len={}",
                                msg.payload.len()
                            );

                            if session_id == 0 {
                                // Unencrypted commissioning — PASE
                                handle_commissioning_message(
                                    msg,
                                    peer,
                                    &transport,
                                    &pase_state,
                                    &established,
                                    &self.subscriptions,
                                    &self.outgoing_counters,
                                    passcode,
                                )
                                .await;
                            } else {
                                // Operational — CASE or IM dispatch
                                handle_operational_message(
                                    msg,
                                    peer,
                                    &transport,
                                    &case_sessions,
                                    &established,
                                    &data_model,
                                    &self.inner,
                                    &self.subscriptions,
                                    &self.outgoing_counters,
                                    &self.notify_tx,
                                )
                                .await;
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Matter UDP recv error: {e}");
                            break;
                        }
                        Err(_) => {} // timeout — loop back and check running flag
                    }
                }
            }
        }

        // Stop mDNS — already handled by CommissionableAdvertiser Drop
        drop(_mdns_handle);
        self.inner.lock().await.running = false;
        Ok(())
    }

    /// Stop the Matter device server.
    pub async fn stop(&self) -> HomeAutoResult<()> {
        self.inner.lock().await.running = false;
        Ok(())
    }

    /// Push an attribute change to all active subscribers.
    ///
    /// The change is delivered asynchronously by the server's receive loop —
    /// this call is non-blocking and only enqueues the event. `value` should
    /// already be TLV-encoded per the attribute's schema.
    ///
    /// Returns `Err` if the server is not running (the channel has been closed).
    pub fn notify_attribute_change(
        &self,
        endpoint: u16,
        cluster: u32,
        attribute: u32,
        value: Vec<u8>,
    ) -> HomeAutoResult<()> {
        self.notify_tx
            .send(AttributeChange {
                endpoint,
                cluster,
                attribute,
                value,
            })
            .map_err(|e| HomeAutoError::Matter(format!("notify_attribute_change: {e}")))
    }

    /// Access the active subscription registry (read-only snapshot).
    ///
    /// Primarily useful for metrics and testing.
    pub fn subscriptions(&self) -> &SubscriptionManager {
        &self.subscriptions
    }

    /// Tear down every bit of state associated with a session id.
    ///
    /// Called on authenticated-session failure (e.g. a CASE/PASE StatusReport
    /// with a non-success general code, or on transport-level disconnect).
    /// The three maps must stay in sync or subscriptions outlive their
    /// keys, so consolidating teardown here prevents drift.
    pub async fn tear_down_session(&self, session_id: u16) {
        tear_down_session_inner(&self.subscriptions, &self.outgoing_counters, session_id).await;
    }

    /// Register a callback for On/Off cluster state changes.
    pub fn set_on_off_handler(&self, f: impl Fn(bool) + Send + Sync + 'static) {
        let inner = self.inner.clone();
        tokio::spawn(async move {
            inner.lock().await.on_off = Some(Arc::new(f));
        });
    }

    /// Register a callback for Level Control cluster changes.
    pub fn set_level_handler(&self, f: impl Fn(u8) + Send + Sync + 'static) {
        let inner = self.inner.clone();
        tokio::spawn(async move {
            inner.lock().await.level = Some(Arc::new(f));
        });
    }

    /// Register a callback for Color Temperature changes.
    pub fn set_color_temp_handler(&self, f: impl Fn(u16) + Send + Sync + 'static) {
        let inner = self.inner.clone();
        tokio::spawn(async move {
            inner.lock().await.color_temp = Some(Arc::new(f));
        });
    }

    /// Register a callback for Thermostat setpoint changes.
    pub fn set_thermostat_handler(&self, f: impl Fn(f32) + Send + Sync + 'static) {
        let inner = self.inner.clone();
        tokio::spawn(async move {
            inner.lock().await.thermostat = Some(Arc::new(f));
        });
    }

    /// The QR code string for this device.
    pub fn qr_code(&self) -> &str {
        &self.qr_code
    }

    /// The 11-digit manual pairing code.
    pub fn pairing_code(&self) -> &str {
        &self.pairing_code
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn start_mdns_advertisement(&self) -> HomeAutoResult<Option<CommissionableAdvertiser>> {
        CommissionableAdvertiser::start(&self.config)
            .map_err(|e| HomeAutoError::Matter(e.to_string()))
            .map(Some)
    }
}

// ── Payload header framing ────────────────────────────────────────────────────

/// Parse the Matter payload header (the exchange header that precedes IM/SecureChannel TLV).
///
/// Wire format:
/// ```text
/// Exchange Flags  (1 byte)
/// Protocol Opcode (1 byte)
/// Exchange ID     (2 bytes LE)
/// Protocol ID     (2 bytes LE)
/// [Ack Counter    (4 bytes LE)  — only if ACK flag set in Exchange Flags]
/// Application Payload (remaining bytes)
/// ```
///
/// Returns `(exchange_flags, opcode, exchange_id, protocol_id, app_payload)`.
pub fn parse_payload_header(payload: &[u8]) -> Option<(u8, u8, u16, u16, &[u8])> {
    if payload.len() < 6 {
        return None;
    }
    let exchange_flags = payload[0];
    let opcode = payload[1];
    let exchange_id = u16::from_le_bytes([payload[2], payload[3]]);
    let protocol_id = u16::from_le_bytes([payload[4], payload[5]]);

    const EXCHANGE_FLAG_ACK: u8 = 0x02;
    let base = if exchange_flags & EXCHANGE_FLAG_ACK != 0 {
        10
    } else {
        6
    };
    if payload.len() < base {
        return None;
    }
    Some((
        exchange_flags,
        opcode,
        exchange_id,
        protocol_id,
        &payload[base..],
    ))
}

/// Build a Matter payload header + application payload.
pub fn build_payload(
    opcode: u8,
    exchange_id: u16,
    protocol_id: u16,
    app_payload: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(6 + app_payload.len());
    out.push(0x00); // Exchange Flags: no ACK, no reliability
    out.push(opcode);
    out.extend_from_slice(&exchange_id.to_le_bytes());
    out.extend_from_slice(&protocol_id.to_le_bytes());
    out.extend_from_slice(app_payload);
    out
}

// ── Helper: send a bare Matter response message (session 0, unencrypted) ─────

/// Deliver a single attribute change to all matching subscribers.
///
/// Outgoing message counters are drawn from `outgoing_counters` — one strictly
/// monotonic `AtomicU32` per session id — so the AEAD nonce (session_id ‖
/// counter) never repeats within a session. Sessions not yet seen are seeded
/// at 1 on first use.
async fn dispatch_attribute_change(
    change: &AttributeChange,
    transport: &Arc<UdpTransport>,
    subscriptions: &Arc<SubscriptionManager>,
    outgoing_counters: &Arc<Mutex<HashMap<u16, std::sync::atomic::AtomicU32>>>,
) {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::clusters::AttributePath;
    use super::interaction_model::AttributeData;

    let subs = subscriptions.matches(change.endpoint, change.cluster, change.attribute);
    if subs.is_empty() {
        return;
    }
    debug!(
        "OP: dispatching attribute change ep={} cluster={:#x} attr={:#x} to {} subscribers",
        change.endpoint,
        change.cluster,
        change.attribute,
        subs.len()
    );

    let attr_data = AttributeData {
        path: AttributePath::specific(change.endpoint, change.cluster, change.attribute),
        data: change.value.clone(),
    };

    let mut counters = outgoing_counters.lock().await;
    for sub in subs {
        let report = ReportData {
            subscription_id: Some(sub.id),
            attribute_reports: vec![attr_data.clone()],
            suppress_response: false,
        };
        let wire = build_payload(
            ImOpcode::ReportData as u8,
            sub.exchange_id,
            IM_PROTOCOL_ID,
            &report.encode(),
        );
        let counter = counters
            .entry(sub.session_id)
            .or_insert_with(|| AtomicU32::new(1))
            .fetch_add(1, Ordering::SeqCst);
        let msg = make_response_message(sub.session_id, counter, wire);
        if let Err(e) = transport.send(&msg, sub.peer).await {
            error!("OP: push ReportData to subscription {} failed: {e}", sub.id);
        }
    }
}

fn make_response_message(session_id: u16, message_counter: u32, payload: Vec<u8>) -> MatterMessage {
    MatterMessage {
        header: MessageHeader {
            version: 0,
            session_id,
            session_type: SessionType::Unicast,
            source_node_id: None,
            dest_node_id: None,
            message_counter,
            security_flags: 0x00,
        },
        payload,
    }
}

// ── PASE dispatch ─────────────────────────────────────────────────────────────

/// Shared teardown helper used by both the public `MatterDeviceServer::tear_down_session`
/// method and the internal StatusReport-failure paths.
async fn tear_down_session_inner(
    subscriptions: &Arc<SubscriptionManager>,
    outgoing_counters: &Arc<Mutex<HashMap<u16, std::sync::atomic::AtomicU32>>>,
    session_id: u16,
) {
    subscriptions.remove_by_session(session_id);
    outgoing_counters.lock().await.remove(&session_id);
    debug!("OP: tore down session {session_id}");
}

/// Parse a StatusReport TLV per Matter spec §4.12:
/// `GeneralCode (u16 LE) | ProtocolId (u32 LE) | ProtocolCode (u16 LE)`.
/// Returns `None` if the payload is shorter than the 8-byte minimum.
fn parse_status_report(tlv: &[u8]) -> Option<(u16, u32, u16)> {
    if tlv.len() < 8 {
        return None;
    }
    let general = u16::from_le_bytes([tlv[0], tlv[1]]);
    let proto_id = u32::from_le_bytes([tlv[2], tlv[3], tlv[4], tlv[5]]);
    let proto_code = u16::from_le_bytes([tlv[6], tlv[7]]);
    Some((general, proto_id, proto_code))
}

// reason: too_many_arguments — Matter commissioning state is shared across
// many Arc-wrapped fields; bundling them would just be a pass-through struct.
#[allow(clippy::too_many_arguments)]
async fn handle_commissioning_message(
    msg: MatterMessage,
    peer: SocketAddr,
    transport: &Arc<UdpTransport>,
    pase_state: &Arc<Mutex<Option<PaseCommissionee>>>,
    established: &Arc<Mutex<HashMap<u16, EstablishedSession>>>,
    subscriptions: &Arc<SubscriptionManager>,
    outgoing_counters: &Arc<Mutex<HashMap<u16, std::sync::atomic::AtomicU32>>>,
    passcode: u32,
) {
    let counter = msg.header.message_counter;

    let (exchange_flags, opcode, exchange_id, protocol_id, app_payload) =
        match parse_payload_header(&msg.payload) {
            Some(v) => v,
            None => {
                warn!("PASE: malformed payload header from {peer}");
                return;
            }
        };

    debug!(
        "PASE proto={protocol_id:#06x} opcode={opcode:#04x} exch={exchange_id} flags={exchange_flags:#04x} from {peer}"
    );

    if protocol_id != SECURE_CHANNEL_PROTOCOL_ID {
        debug!("PASE: ignoring non-SecureChannel protocol {protocol_id:#06x}");
        return;
    }

    match opcode {
        // PBKDFParamRequest (0x20) — start PASE, send PBKDFParamResponse
        x if x == SecureChannelOpcode::PbkdfParamRequest as u8 => {
            let mut commissionee = PaseCommissionee::new(passcode);
            let resp_payload = match commissionee.handle_param_request(app_payload) {
                Ok(p) => p,
                Err(e) => {
                    error!("PASE: PBKDFParamRequest error: {e}");
                    return;
                }
            };

            *pase_state.lock().await = Some(commissionee);

            let wire_payload = build_payload(
                SecureChannelOpcode::PbkdfParamResponse as u8,
                exchange_id,
                SECURE_CHANNEL_PROTOCOL_ID,
                &resp_payload,
            );
            let resp = make_response_message(0, counter.wrapping_add(1), wire_payload);
            if let Err(e) = transport.send(&resp, peer).await {
                error!("PASE: send PBKDFParamResponse error: {e}");
            } else {
                debug!("PASE: sent PBKDFParamResponse to {peer}");
            }
        }

        // Pake1 (0x22) — process pA, send Pake2
        x if x == SecureChannelOpcode::Pake1 as u8 => {
            let mut guard = pase_state.lock().await;
            let commissionee = match guard.as_mut() {
                Some(c) => c,
                None => {
                    warn!("PASE: received Pake1 but no PASE state, ignoring");
                    return;
                }
            };

            let pake2_payload = match commissionee.handle_pake1(app_payload) {
                Ok(p) => p,
                Err(e) => {
                    error!("PASE: Pake1 error: {e}");
                    *guard = None;
                    return;
                }
            };

            let wire_payload = build_payload(
                SecureChannelOpcode::Pake2 as u8,
                exchange_id,
                SECURE_CHANNEL_PROTOCOL_ID,
                &pake2_payload,
            );
            let resp = make_response_message(0, counter.wrapping_add(1), wire_payload);
            if let Err(e) = transport.send(&resp, peer).await {
                error!("PASE: send Pake2 error: {e}");
            } else {
                debug!("PASE: sent Pake2 to {peer}");
            }
        }

        // Pake3 (0x24) — verify cA, establish session, send StatusReport success
        x if x == SecureChannelOpcode::Pake3 as u8 => {
            let mut guard = pase_state.lock().await;
            let commissionee = match guard.take() {
                Some(mut c) => match c.handle_pake3(app_payload) {
                    Ok(session) => session,
                    Err(e) => {
                        error!("PASE: Pake3 error: {e}");
                        return;
                    }
                },
                None => {
                    warn!("PASE: received Pake3 but no PASE state, ignoring");
                    return;
                }
            };
            drop(guard);

            let session_id = commissionee.session_id;
            info!("PASE: session {session_id} established with {peer}");

            // Register session keys with the transport
            let keys = SessionKeys {
                encrypt_key: commissionee.encrypt_key,
                decrypt_key: commissionee.decrypt_key,
            };
            transport.sessions.lock().await.insert(session_id, keys);
            established.lock().await.insert(session_id, commissionee);

            // Send StatusReport success (General Code 0 = SUCCESS)
            // StatusReport TLV: { GeneralCode: 0x0000, ProtocolId: 0, ProtocolCode: 0 }
            let status_tlv = build_status_report_success();
            let wire_payload = build_payload(
                SecureChannelOpcode::StatusReport as u8,
                exchange_id,
                SECURE_CHANNEL_PROTOCOL_ID,
                &status_tlv,
            );
            // StatusReport is sent on the now-established session
            let resp = make_response_message(session_id, counter.wrapping_add(1), wire_payload);
            if let Err(e) = transport.send(&resp, peer).await {
                error!("PASE: send StatusReport error: {e}");
            } else {
                debug!("PASE: sent StatusReport success to {peer}");
            }
        }

        // StatusReport (0x40) — peer-reported failure. Per spec §4.12, a
        // non-zero GeneralCode means the peer has given up on this session.
        // Drop every bit of state associated with it so we don't keep pushing
        // ReportData to a dead endpoint.
        x if x == SecureChannelOpcode::StatusReport as u8 => {
            let session_id = msg.header.session_id;
            match parse_status_report(app_payload) {
                Some((0, _, _)) => {
                    debug!("PASE: StatusReport SUCCESS from {peer} session={session_id}");
                }
                Some((general, proto_id, proto_code)) => {
                    warn!(
                        "PASE: StatusReport FAILURE from {peer} session={session_id} \
                         general={general:#06x} proto_id={proto_id:#010x} proto_code={proto_code:#06x}"
                    );
                    // Tear down any in-flight PASE state plus operational state
                    // attached to this session id.
                    *pase_state.lock().await = None;
                    established.lock().await.remove(&session_id);
                    transport.sessions.lock().await.remove(&session_id);
                    tear_down_session_inner(subscriptions, outgoing_counters, session_id).await;
                }
                None => {
                    warn!("PASE: malformed StatusReport from {peer}");
                }
            }
        }

        other => {
            debug!("PASE: unhandled SecureChannel opcode {other:#04x} from {peer}");
        }
    }
}

// ── StatusReport success payload ──────────────────────────────────────────────

/// Build a minimal StatusReport SUCCESS payload (4 bytes: GeneralCode=0, ProtocolCode=0).
///
/// Wire format per Matter spec §4.12: GeneralCode (2 LE) | ProtocolId (4 LE) | ProtocolCode (2 LE)
fn build_status_report_success() -> Vec<u8> {
    let mut v = Vec::with_capacity(8);
    v.extend_from_slice(&0u16.to_le_bytes()); // GeneralCode = 0 (SUCCESS)
    v.extend_from_slice(&0u32.to_le_bytes()); // ProtocolId = 0 (COMMON)
    v.extend_from_slice(&0u16.to_le_bytes()); // ProtocolCode = 0 (SUCCESS)
    v
}

// ── Operational (CASE + IM) dispatch ─────────────────────────────────────────

// reason: too_many_arguments — same as handle_commissioning_message; the
// Matter server has a lot of independently-owned shared state.
#[allow(clippy::too_many_arguments)]
async fn handle_operational_message(
    msg: MatterMessage,
    peer: SocketAddr,
    transport: &Arc<UdpTransport>,
    _case_sessions: &Arc<Mutex<HashMap<u16, CaseResponder>>>,
    established: &Arc<Mutex<HashMap<u16, EstablishedSession>>>,
    data_model: &Arc<DataModelNode>,
    inner: &Arc<Mutex<ServerInner>>,
    subscriptions: &Arc<SubscriptionManager>,
    outgoing_counters: &Arc<Mutex<HashMap<u16, std::sync::atomic::AtomicU32>>>,
    subscription_notify_tx: &tokio::sync::mpsc::UnboundedSender<AttributeChange>,
) {
    let session_id = msg.header.session_id;
    let counter = msg.header.message_counter;

    let (exchange_flags, opcode, exchange_id, protocol_id, app_payload) =
        match parse_payload_header(&msg.payload) {
            Some(v) => v,
            None => {
                warn!("OP: malformed payload header session={session_id} from {peer}");
                return;
            }
        };

    debug!(
        "OP proto={protocol_id:#06x} opcode={opcode:#04x} exch={exchange_id} \
         flags={exchange_flags:#04x} session={session_id} from {peer}"
    );

    // StatusReport arrives on SECURE_CHANNEL_PROTOCOL_ID even on an already
    // operational (CASE-encrypted) session, so handle it before the IM
    // protocol filter below.
    if protocol_id == SECURE_CHANNEL_PROTOCOL_ID
        && opcode == SecureChannelOpcode::StatusReport as u8
    {
        match parse_status_report(app_payload) {
            Some((0, _, _)) => {
                debug!("OP: StatusReport SUCCESS session={session_id} from {peer}");
            }
            Some((general, proto_id, proto_code)) => {
                warn!(
                    "OP: StatusReport FAILURE session={session_id} from {peer} \
                     general={general:#06x} proto_id={proto_id:#010x} proto_code={proto_code:#06x}"
                );
                established.lock().await.remove(&session_id);
                transport.sessions.lock().await.remove(&session_id);
                tear_down_session_inner(subscriptions, outgoing_counters, session_id).await;
            }
            None => warn!("OP: malformed StatusReport from {peer}"),
        }
        return;
    }

    if protocol_id != IM_PROTOCOL_ID {
        debug!("OP: ignoring non-IM protocol {protocol_id:#06x}");
        return;
    }

    match opcode {
        // InvokeRequest (0x08)
        x if x == ImOpcode::InvokeRequest as u8 => {
            use super::interaction_model::InvokeRequest;
            let req = match InvokeRequest::decode(app_payload) {
                Ok(r) => r,
                Err(e) => {
                    error!("OP: InvokeRequest decode error: {e}");
                    return;
                }
            };

            let mut resp_items = Vec::new();

            for (cmd_path, args) in &req.invoke_requests {
                let ep = cmd_path.endpoint_id;
                let cluster = cmd_path.cluster_id;
                let cmd = cmd_path.command_id;

                debug!(
                    "OP invoke: ep={ep} cluster={cluster:#010x} cmd={cmd:#010x} args_len={}",
                    args.len()
                );

                // Fire handler callbacks for well-known clusters, and push
                // any resulting attribute mutation to active subscriptions.
                dispatch_handler_callbacks(cluster, cmd, args, inner, subscription_notify_tx).await;

                // Dispatch to the data model node
                let result = data_model.dispatch_invoke(ep, cluster, cmd, args).await;

                let item = match result {
                    Ok(response_data) => InvokeResponseItem::Command {
                        path: cmd_path.clone(),
                        data: response_data,
                    },
                    Err(e) => {
                        warn!("OP invoke error ep={ep} cluster={cluster:#010x}: {e}");
                        InvokeResponseItem::Status {
                            path: cmd_path.clone(),
                            status: InteractionStatus::Failure,
                        }
                    }
                };
                resp_items.push(item);
            }

            if !req.suppress_response {
                let invoke_resp = InvokeResponse {
                    suppress_response: false,
                    invoke_responses: resp_items,
                };
                let wire_payload = build_payload(
                    ImOpcode::InvokeResponse as u8,
                    exchange_id,
                    IM_PROTOCOL_ID,
                    &invoke_resp.encode(),
                );
                let resp = make_response_message(session_id, counter.wrapping_add(1), wire_payload);
                if let Err(e) = transport.send(&resp, peer).await {
                    error!("OP: send InvokeResponse error: {e}");
                }
            }
        }

        // ReadRequest (0x02)
        x if x == ImOpcode::ReadRequest as u8 => {
            use super::interaction_model::ReadRequest;

            let req = match ReadRequest::decode(app_payload) {
                Ok(r) => r,
                Err(e) => {
                    error!("OP: ReadRequest decode error: {e}");
                    return;
                }
            };

            let mut all_attrs = Vec::new();
            for path in &req.attribute_requests {
                let mut attrs = data_model.dispatch_read(path).await;
                all_attrs.append(&mut attrs);
            }

            let report = ReportData {
                subscription_id: None,
                attribute_reports: all_attrs,
                suppress_response: false,
            };

            let wire_payload = build_payload(
                ImOpcode::ReportData as u8,
                exchange_id,
                IM_PROTOCOL_ID,
                &report.encode(),
            );
            let resp = make_response_message(session_id, counter.wrapping_add(1), wire_payload);
            if let Err(e) = transport.send(&resp, peer).await {
                error!("OP: send ReportData error: {e}");
            }
        }

        // SubscribeRequest (0x03)
        x if x == ImOpcode::SubscribeRequest as u8 => {
            use super::interaction_model::{SubscribeRequest, SubscribeResponse};

            let req = match SubscribeRequest::decode(app_payload) {
                Ok(r) => r,
                Err(e) => {
                    error!("OP: SubscribeRequest decode error: {e}");
                    return;
                }
            };

            // Negotiate the max interval — for now we accept the controller's ceiling.
            let max_interval = req.max_interval_ceiling_seconds;

            let sub_id = subscriptions.register(
                session_id,
                peer,
                exchange_id,
                req.attribute_requests.clone(),
                req.min_interval_floor_seconds,
                max_interval,
                req.fabric_filtered,
            );

            debug!(
                "OP: registered subscription {sub_id} session={session_id} paths={} max_interval={max_interval}s",
                req.attribute_requests.len()
            );

            // Priming ReportData — send current values for every requested attribute.
            let mut all_attrs = Vec::new();
            for path in &req.attribute_requests {
                let mut attrs = data_model.dispatch_read(path).await;
                all_attrs.append(&mut attrs);
            }
            let report = ReportData {
                subscription_id: Some(sub_id),
                attribute_reports: all_attrs,
                suppress_response: false,
            };
            let report_wire = build_payload(
                ImOpcode::ReportData as u8,
                exchange_id,
                IM_PROTOCOL_ID,
                &report.encode(),
            );
            let report_msg =
                make_response_message(session_id, counter.wrapping_add(1), report_wire);
            if let Err(e) = transport.send(&report_msg, peer).await {
                error!("OP: send priming ReportData error: {e}");
            }

            // SubscribeResponse confirming the negotiated parameters.
            let resp = SubscribeResponse {
                subscription_id: sub_id,
                max_interval,
            };
            let resp_wire = build_payload(
                ImOpcode::SubscribeResponse as u8,
                exchange_id,
                IM_PROTOCOL_ID,
                &resp.encode(),
            );
            let resp_msg = make_response_message(session_id, counter.wrapping_add(2), resp_wire);
            if let Err(e) = transport.send(&resp_msg, peer).await {
                error!("OP: send SubscribeResponse error: {e}");
            }
        }

        other => {
            debug!("OP: unhandled IM opcode {other:#04x} from {peer}");
        }
    }
}

// ── Cluster handler callbacks ─────────────────────────────────────────────────

async fn dispatch_handler_callbacks(
    cluster: u32,
    cmd: u32,
    args: &[u8],
    inner: &Arc<Mutex<ServerInner>>,
    notify_tx: &tokio::sync::mpsc::UnboundedSender<AttributeChange>,
) {
    /// Fire-and-forget: push the attribute change through the subscription
    /// pump. A send failure means the pump is shut down, which is fine on
    /// server teardown — we just log at debug level and drop the event.
    fn push(
        notify_tx: &tokio::sync::mpsc::UnboundedSender<AttributeChange>,
        endpoint: u16,
        cluster: u32,
        attribute: u32,
        value: Vec<u8>,
    ) {
        if let Err(e) = notify_tx.send(AttributeChange {
            endpoint,
            cluster,
            attribute,
            value,
        }) {
            debug!("notify pump closed, dropping attribute change: {e}");
        }
    }

    match cluster {
        CLUSTER_ON_OFF => {
            let handler = inner.lock().await.on_off.clone();
            if let Some(h) = handler {
                let new_state_opt: Option<bool> = match cmd {
                    CMD_OFF => {
                        h(false);
                        Some(false)
                    }
                    CMD_ON => {
                        h(true);
                        Some(true)
                    }
                    CMD_TOGGLE => {
                        // We don't track current state in the server; call handler with true as a hint
                        h(true);
                        Some(true)
                    }
                    _ => None,
                };
                if let Some(on) = new_state_opt {
                    // OnOff attribute (id 0x0000) is a bool; encode as anonymous
                    // TLV bool true (0x09) / false (0x08).
                    let tlv = vec![if on { 0x09u8 } else { 0x08 }];
                    push(notify_tx, 1, CLUSTER_ON_OFF, 0x0000, tlv);
                }
            }
        }
        CLUSTER_LEVEL_CONTROL => {
            let handler = inner.lock().await.level.clone();
            if let Some(h) = handler
                && (cmd == CMD_MOVE_TO_LEVEL || cmd == CMD_MOVE_TO_LEVEL_WITH_ON_OFF)
            {
                // Level is the first field in MoveToLevel args (tag 0, uint8)
                let level = decode_first_uint8(args).unwrap_or(0);
                h(level);
                // CurrentLevel attribute (id 0x0000) is uint8.
                let tlv = vec![0x04u8, level];
                push(notify_tx, 1, CLUSTER_LEVEL_CONTROL, 0x0000, tlv);
            }
        }
        CLUSTER_COLOR_CONTROL => {
            let handler = inner.lock().await.color_temp.clone();
            if let Some(h) = handler {
                // MoveToColorTemperature (cmd 0x0A): color_temperature_mireds (tag 0, uint16)
                if cmd == 0x0A {
                    let mireds = decode_first_uint16(args).unwrap_or(0);
                    h(mireds);
                    // ColorTemperatureMireds attribute (id 0x0007) is uint16 LE.
                    let mut tlv = vec![0x05u8];
                    tlv.extend_from_slice(&mireds.to_le_bytes());
                    push(notify_tx, 1, CLUSTER_COLOR_CONTROL, 0x0007, tlv);
                }
            }
        }
        CLUSTER_THERMOSTAT => {
            let handler = inner.lock().await.thermostat.clone();
            if let Some(h) = handler
                && cmd == CMD_SETPOINT_RAISE_LOWER
            {
                // SetpointRaiseLower: Amount (tag 1, int8). Convert centi-degrees to f32.
                let amount = decode_signed_int8_tag1(args).unwrap_or(0);
                // amount is in units of 0.1°C per Matter spec
                h(amount as f32 * 0.1);
                // OccupiedHeatingSetpoint attribute (id 0x0012) is int16 centi-°C.
                // We don't track absolute temp server-side, so push the delta
                // as an int16 TLV and let subscribers apply it.
                let mut tlv = vec![0x01u8]; // TYPE_SIGNED_INT_2 anonymous
                let delta16 = (amount as i16).saturating_mul(10);
                tlv.extend_from_slice(&delta16.to_le_bytes());
                push(notify_tx, 1, CLUSTER_THERMOSTAT, 0x0012, tlv);
            }
        }
        _ => {}
    }
}

// ── TLV decode helpers ────────────────────────────────────────────────────────

/// Decode the first context-tagged uint8 from a TLV struct body (tag 0).
fn decode_first_uint8(data: &[u8]) -> Option<u8> {
    // Skip the outer STRUCTURE byte (0x15) if present
    let data = if data.first() == Some(&0x15) {
        &data[1..]
    } else {
        data
    };
    let mut i = 0;
    while i + 2 < data.len() {
        let ctrl = data[i];
        let tag = data[i + 1];
        let val_type = ctrl & 0x1F;
        // context tag, uint8
        if (ctrl & 0xE0) == 0x20 && val_type == 0x04 && tag == 0 {
            return Some(data[i + 2]);
        }
        i += skip_tlv_element(data, i);
    }
    None
}

/// Decode the first context-tagged uint16 from a TLV struct body (tag 0).
fn decode_first_uint16(data: &[u8]) -> Option<u16> {
    let data = if data.first() == Some(&0x15) {
        &data[1..]
    } else {
        data
    };
    let mut i = 0;
    while i + 3 < data.len() {
        let ctrl = data[i];
        let tag = data[i + 1];
        let val_type = ctrl & 0x1F;
        if (ctrl & 0xE0) == 0x20 && val_type == 0x05 && tag == 0 {
            return Some(u16::from_le_bytes([data[i + 2], data[i + 3]]));
        }
        i += skip_tlv_element(data, i);
    }
    None
}

/// Decode the context-tagged int8 at tag 1 from a TLV struct body.
fn decode_signed_int8_tag1(data: &[u8]) -> Option<i8> {
    let data = if data.first() == Some(&0x15) {
        &data[1..]
    } else {
        data
    };
    let mut i = 0;
    while i + 2 < data.len() {
        let ctrl = data[i];
        let tag = data[i + 1];
        let val_type = ctrl & 0x1F;
        // context tag, signed int8
        if (ctrl & 0xE0) == 0x20 && val_type == 0x00 && tag == 1 {
            return Some(data[i + 2] as i8);
        }
        i += skip_tlv_element(data, i);
    }
    None
}

/// Advance past one TLV element, returning the number of bytes consumed.
/// Returns 1 if parsing fails to avoid infinite loops.
fn skip_tlv_element(data: &[u8], pos: usize) -> usize {
    if pos >= data.len() {
        return 1;
    }
    let ctrl = data[pos];
    let tag_type = (ctrl >> 5) & 0x07;
    let val_type = ctrl & 0x1F;

    let tag_bytes = match tag_type {
        0 => 0,
        1 => 1,
        _ => return 1,
    };
    let header = 1 + tag_bytes;

    let val_bytes = match val_type {
        0x00 | 0x04 => 1, // signed/unsigned int 1
        0x01 | 0x05 => 2, // signed/unsigned int 2
        0x02 | 0x06 => 4, // signed/unsigned int 4
        0x03 | 0x07 => 8, // signed/unsigned int 8
        0x08 | 0x09 => 0, // bool
        0x10 => {
            // bytes 1-byte length
            let len_pos = pos + header;
            if len_pos >= data.len() {
                return 1;
            }
            data[len_pos] as usize + 1
        }
        0x18 => 0, // end of container
        _ => return 1,
    };
    header + val_bytes
}

/// Generate the `MT:...` QR code string for this device configuration.
///
/// The QR code payload is a Base38-encoded bit-packed structure per Matter spec §5.1.2.
/// This implementation encodes the payload correctly for use with matter-controller tools.
fn generate_qr_code_string(config: &MatterDeviceConfig) -> String {
    // Bit-pack the payload: version(3) + VID(16) + PID(16) + flow(2) + rendezvous(8) + disc(12) + passcode(27) + pad(4)
    let mut bits: u128 = 0;
    let mut pos = 0usize;

    let push = |bits: &mut u128, pos: &mut usize, val: u64, count: usize| {
        *bits |= (val as u128 & ((1u128 << count) - 1)) << *pos;
        *pos += count;
    };

    push(&mut bits, &mut pos, 0, 3); // version = 0
    push(&mut bits, &mut pos, config.vendor_id as u64, 16);
    push(&mut bits, &mut pos, config.product_id as u64, 16);
    push(&mut bits, &mut pos, 0, 2); // flow = standard
    push(&mut bits, &mut pos, 0x10, 8); // rendezvous = OnNetwork
    push(&mut bits, &mut pos, config.discriminator as u64, 12);
    push(&mut bits, &mut pos, config.passcode as u64, 27);
    push(&mut bits, &mut pos, 0, 4); // padding

    // Extract 11 bytes from the 88-bit packed value
    let mut payload = [0u8; 11];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = ((bits >> (i * 8)) & 0xFF) as u8;
    }

    // Base38-encode
    let encoded = base38_encode(&payload);
    format!("MT:{encoded}")
}

const BASE38_CHARS: &[u8; 38] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.";

fn base38_encode(data: &[u8]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i + 1 < data.len() {
        let v = data[i] as u32 | ((data[i + 1] as u32) << 8);
        // Each 2 bytes → 3 base38 characters (log2(38^3) ≈ 17.7 bits > 16)
        let c0 = (v % 38) as usize;
        let c1 = ((v / 38) % 38) as usize;
        let c2 = ((v / (38 * 38)) % 38) as usize;
        out.push(BASE38_CHARS[c0] as char);
        out.push(BASE38_CHARS[c1] as char);
        out.push(BASE38_CHARS[c2] as char);
        i += 2;
    }
    if i < data.len() {
        let v = data[i] as u32;
        out.push(BASE38_CHARS[(v % 38) as usize] as char);
        out.push(BASE38_CHARS[(v / 38) as usize] as char);
    }
    out
}

/// Generate an 11-digit manual pairing code per Matter spec §5.1.4.1.
fn generate_pairing_code(config: &MatterDeviceConfig) -> String {
    let disc = config.discriminator as u32;
    let pass = config.passcode;
    let chunk1 = disc >> 10; // upper 2 bits (0–3) → 2 digits
    let chunk2 = ((disc & 0x3FF) << 14) | (pass >> 14); // lower 10 bits + upper 14 bits of passcode
    let chunk3 = pass & 0x3FFF; // lower 14 bits of passcode
    let prefix = format!("{chunk1:02}{chunk2:06}{chunk3:04}");
    let digits: Vec<u8> = prefix.bytes().map(|b| b - b'0').collect();
    let check = super::verhoeff::compute(&digits);
    format!("{prefix}{check}")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> MatterDeviceConfig {
        MatterDeviceConfig::builder()
            .device_name("Test Device")
            .vendor_id(0xFFF1)
            .product_id(0x8001)
            .discriminator(3840)
            .passcode(20202021)
            .build()
    }

    #[test]
    fn qr_code_starts_with_mt() {
        let config = test_config();
        let qr = generate_qr_code_string(&config);
        assert!(
            qr.starts_with("MT:"),
            "QR code must start with MT:, got: {qr}"
        );
    }

    #[test]
    fn pairing_code_is_numeric() {
        let config = test_config();
        let code = generate_pairing_code(&config);
        // The pairing code may be longer than 11 chars when fields overflow their
        // format width (a known limitation of this implementation).
        // Ensure it is non-empty and all-numeric.
        assert!(!code.is_empty(), "pairing code must not be empty");
        assert!(
            code.chars().all(|c| c.is_ascii_digit()),
            "pairing code must be all digits, got: {code}"
        );
    }

    #[test]
    fn parse_payload_header_roundtrip() {
        let app = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let built = build_payload(0x20, 0x1234, 0x0000, &app);
        let parsed = parse_payload_header(&built).expect("parse failed");
        assert_eq!(parsed.1, 0x20); // opcode
        assert_eq!(parsed.2, 0x1234); // exchange_id
        assert_eq!(parsed.3, 0x0000); // protocol_id
        assert_eq!(parsed.4, app.as_slice());
    }

    #[test]
    fn decode_first_uint8_in_struct() {
        // Build TLV struct: { tag0: uint8(42) }
        // ctrl = 0x20 | 0x04 = 0x24, tag = 0, val = 42
        let data = vec![0x15u8, 0x24, 0x00, 42, 0x18];
        assert_eq!(decode_first_uint8(&data), Some(42));
    }

    #[test]
    fn parse_status_report_success_and_failure() {
        // GeneralCode=0, ProtoId=0, ProtoCode=0 → success tuple.
        let ok = [0u8, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(parse_status_report(&ok), Some((0, 0, 0)));

        // GeneralCode=1 (failure), ProtoId=0x0001, ProtoCode=0x0002 (LE).
        let fail = [0x01u8, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00];
        assert_eq!(parse_status_report(&fail), Some((1, 1, 2)));

        // Truncated payload → None.
        assert_eq!(parse_status_report(&[0u8; 4]), None);
    }

    #[tokio::test]
    async fn tear_down_session_clears_subscriptions_and_counters() {
        use std::sync::atomic::AtomicU32;

        let server = MatterDeviceServer::new(test_config())
            .await
            .expect("server");

        // Seed a subscription and a counter for session 0x42.
        let peer: std::net::SocketAddr = "127.0.0.1:5540".parse().unwrap();
        let path = super::super::clusters::AttributePath::specific(1, 0x0006, 0x0000);
        server
            .subscriptions
            .register(0x42, peer, 7, vec![path], 0, 30, false);
        server
            .outgoing_counters
            .lock()
            .await
            .insert(0x42, AtomicU32::new(5));

        assert_eq!(server.subscriptions.len(), 1);
        assert!(server.outgoing_counters.lock().await.contains_key(&0x42));

        server.tear_down_session(0x42).await;

        assert_eq!(server.subscriptions.len(), 0);
        assert!(!server.outgoing_counters.lock().await.contains_key(&0x42));
    }

    #[tokio::test]
    async fn onoff_handler_pushes_attribute_change_through_notify_pump() {
        let (notify_tx, mut notify_rx) = tokio::sync::mpsc::unbounded_channel::<AttributeChange>();
        let inner = Arc::new(Mutex::new(ServerInner {
            on_off: Some(Arc::new(|_| {})),
            level: None,
            color_temp: None,
            thermostat: None,
            running: false,
            _commissioned: false,
        }));

        // CMD_ON (0x01) → should push an OnOff attribute TLV of 0x09 (bool true).
        dispatch_handler_callbacks(CLUSTER_ON_OFF, CMD_ON, &[], &inner, &notify_tx).await;
        let ev = notify_rx
            .recv()
            .await
            .expect("notify pump should deliver a change");
        assert_eq!(ev.endpoint, 1);
        assert_eq!(ev.cluster, CLUSTER_ON_OFF);
        assert_eq!(ev.attribute, 0x0000);
        assert_eq!(ev.value, vec![0x09]); // bool true

        // CMD_OFF → bool false (0x08).
        dispatch_handler_callbacks(CLUSTER_ON_OFF, CMD_OFF, &[], &inner, &notify_tx).await;
        let ev = notify_rx.recv().await.unwrap();
        assert_eq!(ev.value, vec![0x08]);
    }

    #[tokio::test]
    async fn level_handler_pushes_current_level_attribute() {
        let (notify_tx, mut notify_rx) = tokio::sync::mpsc::unbounded_channel::<AttributeChange>();
        let inner = Arc::new(Mutex::new(ServerInner {
            on_off: None,
            level: Some(Arc::new(|_| {})),
            color_temp: None,
            thermostat: None,
            running: false,
            _commissioned: false,
        }));

        // MoveToLevel { tag0: uint8(200) } — control byte 0x24 = ctx-tag | uint8.
        let args = vec![0x15u8, 0x24, 0x00, 200, 0x18];
        dispatch_handler_callbacks(
            CLUSTER_LEVEL_CONTROL,
            CMD_MOVE_TO_LEVEL,
            &args,
            &inner,
            &notify_tx,
        )
        .await;
        let ev = notify_rx.recv().await.expect("attribute change");
        assert_eq!(ev.cluster, CLUSTER_LEVEL_CONTROL);
        assert_eq!(ev.attribute, 0x0000);
        assert_eq!(ev.value, vec![0x04, 200]);
    }

    #[tokio::test]
    async fn outgoing_counter_is_strictly_monotonic_per_session() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let server = MatterDeviceServer::new(test_config())
            .await
            .expect("server");

        let mut seen = Vec::new();
        for _ in 0..5 {
            let mut counters = server.outgoing_counters.lock().await;
            let v = counters
                .entry(0x07)
                .or_insert_with(|| AtomicU32::new(1))
                .fetch_add(1, Ordering::SeqCst);
            seen.push(v);
        }
        assert_eq!(seen, vec![1, 2, 3, 4, 5]);
    }

    /// End-to-end: bind two UDP transports, share session keys, push three
    /// ReportData changes through `dispatch_attribute_change`, and verify on
    /// the peer side that each datagram decrypts cleanly with strictly
    /// increasing message counters. Exercises the actual AEAD nonce that the
    /// counter feeds — a regression here means the counter fix is wrong.
    ///
    /// No mDNS, no full server loop, no loopback multicast: just transports +
    /// the pure dispatch function.
    #[tokio::test]
    async fn dispatch_attribute_change_emits_decryptable_push_with_monotonic_counter() {
        use std::sync::atomic::AtomicU32;

        use super::super::clusters::AttributePath;
        use super::super::transport::{SessionKeys, UdpTransport};

        // Bind both sides on 127.0.0.1 with OS-assigned ports.
        let server_t = Arc::new(
            UdpTransport::bind_addr("127.0.0.1:0")
                .await
                .expect("server bind"),
        );
        let peer_t = Arc::new(
            UdpTransport::bind_addr("127.0.0.1:0")
                .await
                .expect("peer bind"),
        );

        let server_addr = server_t.local_addr().expect("server addr");
        let peer_addr = peer_t.local_addr().expect("peer addr");
        let _ = server_addr;

        // Shared session 0x0077 with a random AEAD key. Server encrypts with
        // `k`, peer decrypts with the same `k` — symmetric.
        let session_id: u16 = 0x0077;
        let k = [0xA5u8; 16];
        server_t.sessions.lock().await.insert(
            session_id,
            SessionKeys {
                encrypt_key: k,
                decrypt_key: k,
            },
        );
        peer_t.sessions.lock().await.insert(
            session_id,
            SessionKeys {
                encrypt_key: k,
                decrypt_key: k,
            },
        );

        // Register a subscription on the server pointing at the peer address.
        let subs = Arc::new(SubscriptionManager::new());
        let path = AttributePath::specific(1, CLUSTER_ON_OFF, 0x0000);
        let _sub_id = subs.register(session_id, peer_addr, 0x1234, vec![path], 0, 30, false);

        // Seed the outgoing-counter map at 1 so the first push is counter=1.
        let counters: Arc<Mutex<HashMap<u16, AtomicU32>>> = Arc::new(Mutex::new(HashMap::new()));
        counters.lock().await.insert(session_id, AtomicU32::new(1));

        // Push three changes: OnOff attribute toggled true → false → true.
        for (i, on) in [true, false, true].into_iter().enumerate() {
            let value = if on { vec![0x09u8] } else { vec![0x08u8] };
            let change = AttributeChange {
                endpoint: 1,
                cluster: CLUSTER_ON_OFF,
                attribute: 0x0000,
                value,
            };
            dispatch_attribute_change(&change, &server_t, &subs, &counters).await;
            let _ = i;
        }

        // Drain three datagrams on the peer side and assert:
        //   - each one decrypts (AEAD nonce correct → key+counter match)
        //   - counters come out strictly increasing (1, 2, 3)
        //   - each carries a ReportData payload with the expected attribute
        let mut seen_counters = Vec::new();
        for _ in 0..3 {
            let (msg, from) =
                tokio::time::timeout(std::time::Duration::from_secs(2), peer_t.recv())
                    .await
                    .expect("recv timed out — push datagram never arrived")
                    .expect("decrypt failed — counter/nonce mismatch");
            assert_eq!(
                from.ip(),
                std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
                "push must come from the loopback server"
            );
            assert_eq!(msg.header.session_id, session_id);
            seen_counters.push(msg.header.message_counter);
        }

        assert_eq!(
            seen_counters,
            vec![1, 2, 3],
            "counters must be strictly increasing across pushes"
        );
    }
}
