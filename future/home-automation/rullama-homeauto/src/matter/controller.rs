/// Matter controller — commissions and controls Matter devices.
///
/// Implements:
/// - Commissioning payload parsing (complete)
/// - PASE session establishment via SPAKE2+ over UDP (complete)
/// - CASE session establishment via SIGMA over UDP (complete)
/// - Cluster TLV encoding (complete)
/// - Invoke and read operations over established sessions (complete)
/// - Session caching to reuse CASE sessions across calls
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use super::clusters;
use super::clusters::{AttributePath, CommandPath};
use super::commissioning::{parse_manual_code, parse_qr_code};
use super::commissioning_session::CommissioningSession;
use super::discovery::operational::{OperationalBrowser, derive_compressed_fabric_id};
use super::fabric::{FabricIndex, FabricManager};
use super::interaction_model::{
    ImOpcode, InteractionStatus, InvokeRequest, InvokeResponse, InvokeResponseItem, ReadRequest,
    ReportData,
};
use super::secure_channel::{
    CaseInitiator, EstablishedSession, PaseCommissioner, SECURE_CHANNEL_PROTOCOL_ID,
    SecureChannelOpcode,
};
use super::server::{build_payload, parse_payload_header};
use super::transport::message::{MatterMessage, MessageHeader, SessionType};
use super::transport::{SessionKeys, UdpTransport};
use super::types::MatterDevice;
use crate::BoxStream;
use crate::error::{HomeAutoError, HomeAutoResult};
use crate::types::{AttributeValue, HomeAutoEvent};

// IM protocol ID
const IM_PROTOCOL_ID: u16 = 0x0001;

// ── OperationalCredentials cluster constants (client side) ────────────────────
const CLUSTER_OPERATIONAL_CREDENTIALS: u32 = 0x003E;
// CertificateChainRequest (cmd 0x02) is server-side-only for now — the
// controller doesn't currently request the DAC/PAI chain. Left documented
// here for future commissioner-side attestation validation.
const CMD_CSR_REQUEST: u32 = 0x04;
const CMD_ADD_NOC: u32 = 0x06;

// Admin identity seeds for the controller's own admin fabric.
const ADMIN_VENDOR_ID: u16 = 0xFFF1;
const ADMIN_NODE_ID: u64 = 0x0000_0000_0000_0001;

/// Monotonic message counter (global, per-process).
static MSG_COUNTER: AtomicU32 = AtomicU32::new(1);

fn next_counter() -> u32 {
    MSG_COUNTER.fetch_add(1, Ordering::SeqCst)
}

// ── Session cache entry ───────────────────────────────────────────────────────

struct CachedSession {
    addr: SocketAddr,
    session: EstablishedSession,
}

// ── ControllerInner ───────────────────────────────────────────────────────────

struct ControllerInner {
    /// Commissioned devices keyed by node_id.
    devices: HashMap<u64, MatterDevice>,
    /// Next node ID to assign on commissioning (reserved for future auto-assignment).
    _next_node_id: u64,
    /// CASE session cache keyed by node_id.
    session_cache: HashMap<u64, CachedSession>,
}

/// A Matter commissioner and cluster client.
///
/// Supports commissioning devices via QR code or manual pairing code,
/// operational device discovery via mDNS, and cluster command invocation.
pub struct MatterController {
    /// Fabric label stored in NOC (informational).
    _fabric_name: String,
    storage_path: std::path::PathBuf,
    inner: Arc<Mutex<ControllerInner>>,
}

impl MatterController {
    /// Create a new controller. `fabric_name` is stored in the fabric label.
    /// `storage_path` is where the fabric certificate and node data are persisted.
    pub async fn new(fabric_name: impl Into<String>, storage_path: &Path) -> HomeAutoResult<Self> {
        tokio::fs::create_dir_all(storage_path)
            .await
            .map_err(HomeAutoError::Io)?;
        let fabric_name = fabric_name.into();
        info!("MatterController initialised (fabric: {})", fabric_name);
        Ok(Self {
            _fabric_name: fabric_name,
            storage_path: storage_path.to_path_buf(),
            inner: Arc::new(Mutex::new(ControllerInner {
                devices: HashMap::new(),
                _next_node_id: 1,
                session_cache: HashMap::new(),
            })),
        })
    }

    /// Commission a device using its QR code (`MT:...`), returning both the
    /// device handle and a [`CommissioningSession`] subscribers can observe.
    ///
    /// Drives the full commissioning chain:
    ///   Parsed → Discovered → PaseEstablished → CsrReceived → NocInstalled →
    ///   CaseEstablished.
    ///
    /// On any failure, `CommissioningSession::fail(reason)` is called with a
    /// static tag before the error is propagated, so observers can tell which
    /// phase broke.
    pub async fn commission_qr_with_session(
        &self,
        qr_code: &str,
        node_id: u64,
    ) -> HomeAutoResult<(MatterDevice, CommissioningSession)> {
        let payload = parse_qr_code(qr_code)
            .map_err(|e| HomeAutoError::MatterCommissioning(e.to_string()))?;
        self.commission_payload(payload, node_id).await
    }

    /// Commission a device using its 11-digit manual pairing code, returning
    /// both the device handle and a [`CommissioningSession`] subscribers can
    /// observe. Drives the same chain as
    /// [`commission_qr_with_session`](Self::commission_qr_with_session).
    pub async fn commission_code_with_session(
        &self,
        pairing_code: &str,
        node_id: u64,
    ) -> HomeAutoResult<(MatterDevice, CommissioningSession)> {
        let payload = parse_manual_code(pairing_code)
            .map_err(|e| HomeAutoError::MatterCommissioning(e.to_string()))?;
        self.commission_payload(payload, node_id).await
    }

    /// Internal payload-driven commissioning flow shared by both QR and
    /// manual-code entry points.
    async fn commission_payload(
        &self,
        payload: super::commissioning::CommissioningPayload,
        node_id: u64,
    ) -> HomeAutoResult<(MatterDevice, CommissioningSession)> {
        let mut session = CommissioningSession::new(payload.clone(), node_id);

        let peer_addr = match self.discover_commissionable(payload.discriminator).await {
            Ok(addr) => {
                session.advance_discovered(addr);
                addr
            }
            Err(e) => {
                session.fail("discovery_timeout");
                return Err(e);
            }
        };

        info!("Commissioning: found device at {peer_addr}");

        let transport = UdpTransport::bind_addr("0.0.0.0:0")
            .await
            .map_err(|e| HomeAutoError::MatterCommissioning(format!("UDP bind: {e}")))?;
        let transport = Arc::new(transport);

        let pase = match self.run_pase(&transport, peer_addr, payload.passcode).await {
            Ok(s) => {
                session.advance_pase_established(s.session_id);
                s
            }
            Err(e) => {
                session.fail("pase_failed");
                return Err(e);
            }
        };

        transport.sessions.lock().await.insert(
            pase.session_id,
            SessionKeys {
                encrypt_key: pase.encrypt_key,
                decrypt_key: pase.decrypt_key,
            },
        );

        info!("PASE commissioned: session_id={}", pase.session_id);

        // ── Phase 4: CSRRequest over PASE ────────────────────────────────────
        let csr_nonce: [u8; 32] = {
            use rand_core::RngCore;
            let mut buf = [0u8; 32];
            rand_core::OsRng.fill_bytes(&mut buf);
            buf
        };
        let csr_args = build_struct(&[tlv_octet_string_ctx(0, &csr_nonce)]);
        let csr_response = match self
            .invoke_over_session(
                &transport,
                pase.session_id,
                peer_addr,
                CLUSTER_OPERATIONAL_CREDENTIALS,
                CMD_CSR_REQUEST,
                &csr_args,
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                session.fail("csr_invoke_failed");
                return Err(HomeAutoError::MatterCommissioning(format!(
                    "CSRRequest: {e}"
                )));
            }
        };

        let csr_pubkey = match parse_csr_response(&csr_response) {
            Some(k) => {
                session.advance_csr_received();
                k
            }
            None => {
                session.fail("csr_malformed");
                return Err(HomeAutoError::MatterCommissioning(
                    "CSRRequest: response missing NOCSRElements or pubkey".into(),
                ));
            }
        };

        // ── Phase 5: issue NOC on the controller's admin fabric, send AddNOC ──
        let mut fabric_manager = FabricManager::load(&self.storage_path).await.map_err(|e| {
            session.fail("fabric_load_failed");
            HomeAutoError::MatterCommissioning(format!("FabricManager load: {e}"))
        })?;

        // Ensure we have an admin fabric. If none exists, mint one and persist
        // (using the RCAC as a placeholder NOC — this is the same pattern the
        // existing `generate_root_ca_and_issue_noc` test uses). Otherwise reuse
        // the first stored entry as the admin identity for issuing device NOCs.
        let fabric_index: FabricIndex = if fabric_manager.fabrics().is_empty() {
            let fabric_id = rand_fabric_id();
            let (sk_bytes, rcac, descriptor) = fabric_manager
                .generate_root_ca(
                    ADMIN_VENDOR_ID,
                    fabric_id,
                    ADMIN_NODE_ID,
                    &self._fabric_name,
                )
                .map_err(|e| {
                    session.fail("generate_root_ca_failed");
                    HomeAutoError::MatterCommissioning(format!("generate_root_ca: {e}"))
                })?;
            let idx = descriptor.fabric_index;
            let noc_placeholder = rcac.clone();
            fabric_manager.add_fabric_entry(descriptor, &rcac, &noc_placeholder, None, sk_bytes);
            fabric_manager.save().await.map_err(|e| {
                session.fail("fabric_persist_failed");
                HomeAutoError::MatterCommissioning(format!("FabricManager save: {e}"))
            })?;
            idx
        } else {
            fabric_manager.fabrics()[0].descriptor.fabric_index
        };

        let device_noc = fabric_manager
            .issue_noc(fabric_index, &csr_pubkey, node_id)
            .map_err(|e| {
                session.fail("issue_noc_failed");
                HomeAutoError::MatterCommissioning(format!("issue_noc: {e}"))
            })?;
        let device_noc_tlv = device_noc.encode();

        // AddNOC { tag0: NOCValue, tag1?: ICACValue, tag2: IPK (16 bytes),
        //          tag3: CaseAdminSubject (uint64), tag4: AdminVendorId (uint16) }
        let ipk = [0u8; 16];
        let add_noc_args = build_struct(&[
            tlv_octet_string_ctx(0, &device_noc_tlv),
            tlv_octet_string_ctx(2, &ipk),
            tlv_uint64_ctx(3, ADMIN_NODE_ID),
            tlv_uint16_ctx(4, ADMIN_VENDOR_ID),
        ]);

        let add_noc_response = match self
            .invoke_over_session(
                &transport,
                pase.session_id,
                peer_addr,
                CLUSTER_OPERATIONAL_CREDENTIALS,
                CMD_ADD_NOC,
                &add_noc_args,
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                session.fail("add_noc_invoke_failed");
                return Err(HomeAutoError::MatterCommissioning(format!("AddNOC: {e}")));
            }
        };

        let add_noc_status = parse_noc_response_status(&add_noc_response).unwrap_or(0xFF);
        if add_noc_status != 0 {
            session.fail("add_noc_nonzero_status");
            return Err(HomeAutoError::MatterCommissioning(format!(
                "AddNOC returned status {add_noc_status:#x}"
            )));
        }
        session.advance_noc_installed();

        // ── Phase 6: CASE handshake on the now-commissioned device ───────────
        let device = MatterDevice {
            node_id,
            fabric_index: fabric_index.0,
            name: None,
            vendor_id: payload.vendor_id,
            product_id: payload.product_id,
            endpoints: Vec::new(),
            online: true,
        };
        self.inner
            .lock()
            .await
            .devices
            .insert(node_id, device.clone());

        match self.get_or_establish_session(&device).await {
            Ok((_t, established, _peer)) => {
                session.advance_case_established(established.session_id);
            }
            Err(e) => {
                session.fail("case_establish_failed");
                return Err(e);
            }
        }

        Ok((device, session))
    }

    /// Send an InvokeRequest over an *already-established* secure session
    /// (PASE or CASE). Returns the first `Command { data }` payload in the
    /// InvokeResponse, or errors if the response is a Status with failure.
    ///
    /// Unlike [`Self::invoke`], this does not create or cache a CASE
    /// session — the caller must have already registered keys with
    /// `transport.sessions` for the given `session_id`.
    async fn invoke_over_session(
        &self,
        transport: &Arc<UdpTransport>,
        session_id: u16,
        peer: SocketAddr,
        cluster: u32,
        cmd: u32,
        args_tlv: &[u8],
    ) -> HomeAutoResult<Vec<u8>> {
        // Endpoint 0 is the commissioner-addressable administrative endpoint
        // for the operational-credentials cluster.
        let path = CommandPath::new(0, cluster, cmd);
        let req = InvokeRequest::new(path, args_tlv.to_vec());
        let exchange_id = (next_counter() & 0xFFFF) as u16;

        let wire = build_payload(
            ImOpcode::InvokeRequest as u8,
            exchange_id,
            IM_PROTOCOL_ID,
            &req.encode(),
        );
        let msg = build_matter_message(session_id, next_counter(), wire);
        let resp = send_and_recv(transport, peer, msg).await?;

        let (_, opcode, _, proto, app) = parse_payload_header(&resp.payload).ok_or_else(|| {
            HomeAutoError::Matter("invoke_over_session: bad response header".into())
        })?;
        if proto != IM_PROTOCOL_ID {
            return Err(HomeAutoError::Matter(format!(
                "invoke_over_session: unexpected protocol {proto:#06x}"
            )));
        }
        if opcode != ImOpcode::InvokeResponse as u8 {
            return Err(HomeAutoError::Matter(format!(
                "invoke_over_session: expected InvokeResponse (0x09), got {opcode:#04x}"
            )));
        }

        let ir = InvokeResponse::decode(app).map_err(|e| {
            HomeAutoError::Matter(format!("invoke_over_session decode InvokeResponse: {e}"))
        })?;
        for item in ir.invoke_responses {
            match item {
                InvokeResponseItem::Command { data, .. } => return Ok(data),
                InvokeResponseItem::Status { status, .. } => {
                    if status != InteractionStatus::Success {
                        return Err(HomeAutoError::Matter(format!(
                            "invoke_over_session: status {:?}",
                            status
                        )));
                    }
                }
            }
        }
        Err(HomeAutoError::Matter(
            "invoke_over_session: response had no Command or success Status".into(),
        ))
    }

    /// Commission a device using its QR code (`MT:...`).
    ///
    /// Drives the full QR → Discovery → PASE → CSR → AddNOC → CASE chain via
    /// [`Self::commission_qr_with_session`] and returns the commissioned
    /// device handle. The observable session is dropped; use
    /// `commission_qr_with_session` directly if you need progress events.
    pub async fn commission_qr(&self, qr_code: &str, node_id: u64) -> HomeAutoResult<MatterDevice> {
        let (device, _session) = self.commission_qr_with_session(qr_code, node_id).await?;
        Ok(device)
    }

    /// Commission a device using its 11-digit manual pairing code.
    ///
    /// Drives the same full chain as [`Self::commission_qr`]. Vendor and
    /// product IDs are not carried by manual pairing codes and end up zero
    /// on the returned [`MatterDevice`]; read them via BasicInformation if
    /// needed.
    pub async fn commission_code(
        &self,
        pairing_code: &str,
        node_id: u64,
    ) -> HomeAutoResult<MatterDevice> {
        let (device, _session) = self
            .commission_code_with_session(pairing_code, node_id)
            .await?;
        Ok(device)
    }

    /// Return all commissioned devices.
    pub async fn devices(&self) -> HomeAutoResult<Vec<MatterDevice>> {
        Ok(self.inner.lock().await.devices.values().cloned().collect())
    }

    // ── Convenience cluster helpers ───────────────────────────────────────────

    /// Turn a device's On/Off endpoint on or off.
    pub async fn on_off(
        &self,
        device: &MatterDevice,
        endpoint: u16,
        on: bool,
    ) -> HomeAutoResult<()> {
        let (cmd, tlv) = if on {
            (clusters::on_off::CMD_ON, clusters::on_off::on_tlv())
        } else {
            (clusters::on_off::CMD_OFF, clusters::on_off::off_tlv())
        };
        self.invoke(device, endpoint, clusters::on_off::CLUSTER_ID, cmd, &tlv)
            .await
    }

    /// Set the level on a Level Control endpoint (0–254).
    pub async fn set_level(
        &self,
        device: &MatterDevice,
        endpoint: u16,
        level: u8,
    ) -> HomeAutoResult<()> {
        let tlv = clusters::level_control::move_to_level_tlv(level, None);
        self.invoke(
            device,
            endpoint,
            clusters::level_control::CLUSTER_ID,
            clusters::level_control::CMD_MOVE_TO_LEVEL_WITH_ON_OFF,
            &tlv,
        )
        .await
    }

    /// Move a window covering up or down.
    pub async fn window_covering(
        &self,
        device: &MatterDevice,
        endpoint: u16,
        up: bool,
    ) -> HomeAutoResult<()> {
        let cmd = if up {
            clusters::window_covering::CMD_UP_OR_OPEN
        } else {
            clusters::window_covering::CMD_DOWN_OR_CLOSE
        };
        self.invoke(
            device,
            endpoint,
            clusters::window_covering::CLUSTER_ID,
            cmd,
            &[],
        )
        .await
    }

    /// Lock or unlock a door lock endpoint.
    pub async fn door_lock(
        &self,
        device: &MatterDevice,
        endpoint: u16,
        lock: bool,
        pin: Option<&[u8]>,
    ) -> HomeAutoResult<()> {
        let cmd = if lock {
            clusters::door_lock::CMD_LOCK_DOOR
        } else {
            clusters::door_lock::CMD_UNLOCK_DOOR
        };
        let tlv = clusters::door_lock::lock_tlv(pin);
        self.invoke(device, endpoint, clusters::door_lock::CLUSTER_ID, cmd, &tlv)
            .await
    }

    // ── Generic interaction model operations ──────────────────────────────────

    /// Invoke a cluster command on a device endpoint.
    ///
    /// Establishes or reuses a CASE session, sends an InvokeRequest, and
    /// awaits the InvokeResponse.
    pub async fn invoke(
        &self,
        device: &MatterDevice,
        endpoint: u16,
        cluster: u32,
        cmd: u32,
        tlv: &[u8],
    ) -> HomeAutoResult<()> {
        debug!(
            "Matter invoke: node={} ep={endpoint} cluster={cluster:#010x} cmd={cmd:#010x} payload_len={}",
            device.node_id,
            tlv.len()
        );

        let (transport, session, peer) = self.get_or_establish_session(device).await?;

        let path = CommandPath::new(endpoint, cluster, cmd);
        let req = InvokeRequest::new(path, tlv.to_vec());
        let exchange_id = (next_counter() & 0xFFFF) as u16;

        let wire_payload = build_payload(
            ImOpcode::InvokeRequest as u8,
            exchange_id,
            IM_PROTOCOL_ID,
            &req.encode(),
        );
        let msg = build_matter_message(session.session_id, next_counter(), wire_payload);

        let resp_msg = send_and_recv(&transport, peer, msg).await?;

        // Parse the response payload header
        let (_, resp_opcode, _, resp_proto, resp_app) = parse_payload_header(&resp_msg.payload)
            .ok_or_else(|| HomeAutoError::Matter("invoke: bad response payload header".into()))?;

        if resp_proto != IM_PROTOCOL_ID {
            return Err(HomeAutoError::Matter(format!(
                "invoke: unexpected response protocol {resp_proto:#06x}"
            )));
        }

        if resp_opcode != ImOpcode::InvokeResponse as u8 {
            return Err(HomeAutoError::Matter(format!(
                "invoke: expected InvokeResponse (0x09), got {resp_opcode:#04x}"
            )));
        }

        let resp = InvokeResponse::decode(resp_app)
            .map_err(|e| HomeAutoError::Matter(format!("invoke: decode InvokeResponse: {e}")))?;

        // Check for any failure status in the response
        for item in &resp.invoke_responses {
            if let InvokeResponseItem::Status { path: _, status } = item
                && *status != InteractionStatus::Success
            {
                return Err(HomeAutoError::MatterCluster {
                    cluster,
                    cmd,
                    msg: format!("invoke failed with status {:?}", status),
                });
            }
        }

        Ok(())
    }

    /// Read an attribute from a device endpoint.
    ///
    /// Establishes or reuses a CASE session, sends a ReadRequest, and
    /// returns the decoded attribute value.
    pub async fn read_attr(
        &self,
        device: &MatterDevice,
        endpoint: u16,
        cluster: u32,
        attr: u32,
    ) -> HomeAutoResult<AttributeValue> {
        debug!(
            "Matter read_attr: node={} ep={endpoint} cluster={cluster:#010x} attr={attr:#010x}",
            device.node_id
        );

        let (transport, session, peer) = self.get_or_establish_session(device).await?;

        let path = AttributePath::specific(endpoint, cluster, attr);
        let req = ReadRequest::new(vec![path.clone()]);
        let exchange_id = (next_counter() & 0xFFFF) as u16;

        let wire_payload = build_payload(
            ImOpcode::ReadRequest as u8,
            exchange_id,
            IM_PROTOCOL_ID,
            &req.encode(),
        );
        let msg = build_matter_message(session.session_id, next_counter(), wire_payload);

        let resp_msg = send_and_recv(&transport, peer, msg).await?;

        let (_, resp_opcode, _, resp_proto, resp_app) = parse_payload_header(&resp_msg.payload)
            .ok_or_else(|| HomeAutoError::Matter("read_attr: bad response header".into()))?;

        if resp_proto != IM_PROTOCOL_ID || resp_opcode != ImOpcode::ReportData as u8 {
            return Err(HomeAutoError::Matter(format!(
                "read_attr: expected ReportData, got proto={resp_proto:#06x} opcode={resp_opcode:#04x}"
            )));
        }

        let report = ReportData::decode(resp_app)
            .map_err(|e| HomeAutoError::Matter(format!("read_attr: decode ReportData: {e}")))?;

        let attr_data = report
            .attribute_reports
            .into_iter()
            .find(|d| {
                d.path.endpoint_id == Some(endpoint)
                    && d.path.cluster_id == Some(cluster)
                    && d.path.attribute_id == Some(attr)
            })
            .ok_or_else(|| HomeAutoError::Matter(format!(
                "read_attr: attribute ep={endpoint} cluster={cluster:#010x} attr={attr:#010x} not in response"
            )))?;

        // Convert raw TLV bytes to AttributeValue
        Ok(tlv_to_attribute_value(&attr_data.data))
    }

    /// Subscribe to a stream of events from all commissioned devices.
    ///
    /// **Stub**: Currently returns an empty stream. Event subscription is not
    /// yet implemented.
    pub fn events(&self) -> BoxStream<'static, HomeAutoEvent> {
        Box::pin(futures::stream::empty())
    }

    // ── Internal session management ───────────────────────────────────────────

    /// Get or establish a CASE session to the given device.
    ///
    /// Returns `(transport, established_session, peer_addr)`.
    async fn get_or_establish_session(
        &self,
        device: &MatterDevice,
    ) -> HomeAutoResult<(Arc<UdpTransport>, EstablishedSession, SocketAddr)> {
        // Check session cache
        {
            let inner = self.inner.lock().await;
            if let Some(cached) = inner.session_cache.get(&device.node_id) {
                let transport = UdpTransport::bind_addr("0.0.0.0:0")
                    .await
                    .map_err(|e| HomeAutoError::Matter(format!("UDP bind: {e}")))?;
                let transport = Arc::new(transport);
                transport.sessions.lock().await.insert(
                    cached.session.session_id,
                    SessionKeys {
                        encrypt_key: cached.session.encrypt_key,
                        decrypt_key: cached.session.decrypt_key,
                    },
                );
                return Ok((transport, cached.session.clone(), cached.addr));
            }
        }

        // Need to establish a new CASE session — discover node and load fabric
        let fabric_manager = FabricManager::load(&self.storage_path)
            .await
            .map_err(|e| HomeAutoError::Matter(format!("FabricManager load: {e}")))?;

        // We need at least one fabric to do CASE
        let fabrics = fabric_manager.fabrics();
        if fabrics.is_empty() {
            return Err(HomeAutoError::Matter(
                "no fabric found — commission the device first".into(),
            ));
        }
        let fabric_entry = &fabrics[0];
        let fabric = &fabric_entry.descriptor;

        // Discover the device via mDNS operational browsing
        let cfid = derive_compressed_fabric_id(fabric);
        let browser = OperationalBrowser::new()
            .map_err(|e| HomeAutoError::Matter(format!("OperationalBrowser: {e}")))?;
        let peer = browser
            .discover_node(cfid, device.node_id, 10_000)
            .await
            .map_err(|e| HomeAutoError::Matter(format!("discover_node: {e}")))?;

        // Load the node's private key from fabric entry
        let sk_bytes: [u8; 32] = fabric_entry.private_key_bytes[..32]
            .try_into()
            .map_err(|_| HomeAutoError::Matter("invalid private key length".into()))?;
        let node_key = p256::SecretKey::from_bytes(&sk_bytes.into())
            .map_err(|e| HomeAutoError::Matter(format!("parse node key: {e}")))?;

        let noc = super::fabric::MatterCert::decode(&fabric_entry.noc_der)
            .map_err(|e| HomeAutoError::Matter(format!("decode NOC: {e}")))?;
        let icac = fabric_entry
            .icac_der
            .as_deref()
            .and_then(|d| super::fabric::MatterCert::decode(d).ok());

        let transport = Arc::new(
            UdpTransport::bind_addr("0.0.0.0:0")
                .await
                .map_err(|e| HomeAutoError::Matter(format!("UDP bind: {e}")))?,
        );

        // Run CASE (SIGMA protocol)
        let mut initiator = CaseInitiator::new(node_key, noc, icac, fabric.clone());

        let (session_id, sigma1) = initiator
            .build_sigma1()
            .map_err(|e| HomeAutoError::Matter(format!("CASE Sigma1: {e}")))?;

        // Send Sigma1, receive Sigma2
        let exchange_id = (next_counter() & 0xFFFF) as u16;
        let wire1 = build_payload(
            SecureChannelOpcode::Sigma1 as u8,
            exchange_id,
            SECURE_CHANNEL_PROTOCOL_ID,
            &sigma1,
        );
        let sigma1_msg = build_matter_message(0, next_counter(), wire1);
        let sigma2_resp = send_and_recv(&transport, peer, sigma1_msg).await?;

        let (_, op2, _, _, sigma2_app) = parse_payload_header(&sigma2_resp.payload)
            .ok_or_else(|| HomeAutoError::Matter("CASE: bad Sigma2 header".into()))?;
        if op2 != SecureChannelOpcode::Sigma2 as u8 {
            return Err(HomeAutoError::Matter(format!(
                "CASE: expected Sigma2, got opcode {op2:#04x}"
            )));
        }

        // Process Sigma2, produce Sigma3
        let sigma3 = initiator
            .handle_sigma2(sigma2_app)
            .map_err(|e| HomeAutoError::Matter(format!("CASE handle_sigma2: {e}")))?;

        // Send Sigma3
        let wire3 = build_payload(
            SecureChannelOpcode::Sigma3 as u8,
            exchange_id,
            SECURE_CHANNEL_PROTOCOL_ID,
            &sigma3,
        );
        let sigma3_msg = build_matter_message(0, next_counter(), wire3);
        let status_resp = send_and_recv(&transport, peer, sigma3_msg).await?;

        // Parse StatusReport to confirm success
        let (_, op_sr, _, _, _) = parse_payload_header(&status_resp.payload)
            .ok_or_else(|| HomeAutoError::Matter("CASE: bad StatusReport header".into()))?;
        if op_sr != SecureChannelOpcode::StatusReport as u8 {
            warn!("CASE: expected StatusReport, got {op_sr:#04x}");
        }

        // Extract the established session
        let session = initiator
            .established_session()
            .ok_or_else(|| HomeAutoError::Matter("CASE: session not established".into()))?
            .clone();

        // Register session keys
        transport.sessions.lock().await.insert(
            session_id,
            SessionKeys {
                encrypt_key: session.encrypt_key,
                decrypt_key: session.decrypt_key,
            },
        );

        info!("CASE: session {session_id} established with {peer}");

        // Cache the session
        self.inner.lock().await.session_cache.insert(
            device.node_id,
            CachedSession {
                addr: peer,
                session: session.clone(),
            },
        );

        Ok((transport, session, peer))
    }

    // ── PASE commissioning helper ─────────────────────────────────────────────

    /// Run the full PASE handshake against `peer` using `passcode`.
    ///
    /// Returns the established PASE session.
    async fn run_pase(
        &self,
        transport: &UdpTransport,
        peer: SocketAddr,
        passcode: u32,
    ) -> HomeAutoResult<EstablishedSession> {
        let mut commissioner = PaseCommissioner::new(passcode);

        // Step 1: send PBKDFParamRequest
        let (_session_id, param_req) = commissioner
            .build_param_request()
            .map_err(|e| HomeAutoError::MatterCommissioning(format!("PBKDFParamRequest: {e}")))?;

        let exchange_id = (next_counter() & 0xFFFF) as u16;
        let wire_req = build_payload(
            SecureChannelOpcode::PbkdfParamRequest as u8,
            exchange_id,
            SECURE_CHANNEL_PROTOCOL_ID,
            &param_req,
        );
        let param_req_msg = build_matter_message(0, next_counter(), wire_req);
        let param_resp_msg = send_and_recv(transport, peer, param_req_msg)
            .await
            .map_err(|e| {
                HomeAutoError::MatterCommissioning(format!("PBKDFParamResponse recv: {e}"))
            })?;

        let (_, op_r, _, _, param_resp_app) = parse_payload_header(&param_resp_msg.payload)
            .ok_or_else(|| {
                HomeAutoError::MatterCommissioning("bad PBKDFParamResponse header".into())
            })?;
        if op_r != SecureChannelOpcode::PbkdfParamResponse as u8 {
            return Err(HomeAutoError::MatterCommissioning(format!(
                "expected PBKDFParamResponse, got {op_r:#04x}"
            )));
        }

        // Step 2: send Pake1
        let pake1 = commissioner
            .handle_param_response(param_resp_app)
            .map_err(|e| {
                HomeAutoError::MatterCommissioning(format!("handle_param_response: {e}"))
            })?;

        let wire_pake1 = build_payload(
            SecureChannelOpcode::Pake1 as u8,
            exchange_id,
            SECURE_CHANNEL_PROTOCOL_ID,
            &pake1,
        );
        let pake1_msg = build_matter_message(0, next_counter(), wire_pake1);
        let pake2_resp = send_and_recv(transport, peer, pake1_msg)
            .await
            .map_err(|e| HomeAutoError::MatterCommissioning(format!("Pake2 recv: {e}")))?;

        let (_, op_2, _, _, pake2_app) = parse_payload_header(&pake2_resp.payload)
            .ok_or_else(|| HomeAutoError::MatterCommissioning("bad Pake2 header".into()))?;
        if op_2 != SecureChannelOpcode::Pake2 as u8 {
            return Err(HomeAutoError::MatterCommissioning(format!(
                "expected Pake2, got {op_2:#04x}"
            )));
        }

        // Step 3: send Pake3
        let pake3 = commissioner
            .handle_pake2(pake2_app)
            .map_err(|e| HomeAutoError::MatterCommissioning(format!("handle_pake2: {e}")))?;

        let wire_pake3 = build_payload(
            SecureChannelOpcode::Pake3 as u8,
            exchange_id,
            SECURE_CHANNEL_PROTOCOL_ID,
            &pake3,
        );
        let pake3_msg = build_matter_message(0, next_counter(), wire_pake3);
        let status_msg = send_and_recv(transport, peer, pake3_msg)
            .await
            .map_err(|e| HomeAutoError::MatterCommissioning(format!("StatusReport recv: {e}")))?;

        // Parse StatusReport for success
        let (_, op_sr, _, _, _) = parse_payload_header(&status_msg.payload)
            .ok_or_else(|| HomeAutoError::MatterCommissioning("bad StatusReport header".into()))?;
        if op_sr != SecureChannelOpcode::StatusReport as u8 {
            warn!("PASE: expected StatusReport, got {op_sr:#04x}");
        }

        commissioner.established_session().cloned().ok_or_else(|| {
            HomeAutoError::MatterCommissioning("PASE: session not established after Pake3".into())
        })
    }

    // ── mDNS discovery helper ─────────────────────────────────────────────────

    /// Discover a commissionable device by discriminator via mDNS.
    ///
    /// Browses `_matterc._udp` for up to 10 seconds.  Returns a `SocketAddr`
    /// suitable for PASE commissioning.
    async fn discover_commissionable(&self, discriminator: u16) -> HomeAutoResult<SocketAddr> {
        use mdns_sd::{ServiceDaemon, ServiceEvent};
        use std::time::Duration;

        let daemon =
            ServiceDaemon::new().map_err(|e| HomeAutoError::Matter(format!("mDNS daemon: {e}")))?;

        let receiver = daemon
            .browse("_matterc._udp.local.")
            .map_err(|e| HomeAutoError::Matter(format!("mDNS browse: {e}")))?;

        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let disc_str = discriminator.to_string();

        loop {
            let remaining = deadline
                .checked_duration_since(std::time::Instant::now())
                .unwrap_or(Duration::ZERO);
            if remaining.is_zero() {
                break;
            }

            match receiver.recv_timeout(remaining) {
                Ok(ServiceEvent::ServiceResolved(info)) => {
                    // Check the D TXT record for a discriminator match
                    let d_val_owned;
                    let d_val = if let Some(prop) = info.get_properties().get("D") {
                        d_val_owned = prop.val_str().to_string();
                        d_val_owned.as_str()
                    } else {
                        ""
                    };
                    if d_val == disc_str {
                        let port = info.get_port();
                        let addr = info
                            .get_addresses()
                            .iter()
                            .find(|a| matches!(a, std::net::IpAddr::V4(_)))
                            .or_else(|| info.get_addresses().iter().next())
                            .copied()
                            .ok_or_else(|| {
                                HomeAutoError::Matter("mDNS: no address for device".into())
                            })?;
                        let _ = daemon.stop_browse("_matterc._udp.local.");
                        return Ok(SocketAddr::new(addr, port));
                    }
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }

        let _ = daemon.stop_browse("_matterc._udp.local.");
        Err(HomeAutoError::Matter(format!(
            "commissionable device with discriminator={discriminator} not found within 10s"
        )))
    }
}

// ── Transport helpers ─────────────────────────────────────────────────────────

/// Build a MatterMessage with the given session_id, counter, and payload.
fn build_matter_message(session_id: u16, counter: u32, payload: Vec<u8>) -> MatterMessage {
    MatterMessage {
        header: MessageHeader {
            version: 0,
            session_id,
            session_type: SessionType::Unicast,
            source_node_id: None,
            dest_node_id: None,
            message_counter: counter,
            security_flags: 0x00,
        },
        payload,
    }
}

/// Send a `MatterMessage` and wait for one response datagram.
///
/// Uses a 5-second timeout.
async fn send_and_recv(
    transport: &UdpTransport,
    peer: SocketAddr,
    msg: MatterMessage,
) -> HomeAutoResult<MatterMessage> {
    transport
        .send(&msg, peer)
        .await
        .map_err(|e| HomeAutoError::Matter(format!("send_and_recv: send: {e}")))?;

    // Wait for response with timeout
    match tokio::time::timeout(std::time::Duration::from_secs(5), transport.recv()).await {
        Ok(Ok((resp, _))) => Ok(resp),
        Ok(Err(e)) => Err(HomeAutoError::Matter(format!("send_and_recv: recv: {e}"))),
        Err(_) => Err(HomeAutoError::Timeout),
    }
}

// ── TLV helpers for commissioning args / responses ──────────────────────────

/// Wrap an inner byte slice as `struct { <inner> }` in Matter TLV.
fn build_struct(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut v = Vec::new();
    v.push(0x15u8); // TYPE_STRUCTURE
    for p in parts {
        v.extend_from_slice(p);
    }
    v.push(0x18u8); // TYPE_END_OF_CONTAINER
    v
}

/// TLV: context-tagged 1-byte length octet string `{ tag: bytes }`.
fn tlv_octet_string_ctx(tag: u8, data: &[u8]) -> Vec<u8> {
    // TYPE_OCTET_STRING_1 = 0x10, CONTEXT tag class = 0x20
    assert!(
        data.len() <= 255,
        "octet_string over 255 bytes needs a wider length header"
    );
    let mut v = vec![0x20 | 0x10, tag, data.len() as u8];
    v.extend_from_slice(data);
    v
}

/// TLV: context-tagged uint64 (tag | TYPE_UNSIGNED_INT_8 = 0x07).
fn tlv_uint64_ctx(tag: u8, val: u64) -> Vec<u8> {
    let mut v = vec![0x20 | 0x07, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

/// TLV: context-tagged uint16 (TYPE_UNSIGNED_INT_2 = 0x05).
fn tlv_uint16_ctx(tag: u8, val: u16) -> Vec<u8> {
    let mut v = vec![0x20 | 0x05, tag];
    v.extend_from_slice(&val.to_le_bytes());
    v
}

/// Parse a CSRResponse body and return the 65-byte P-256 CSR public key.
///
/// The server layout is `struct { tag 0: octet_string(NOCSRElements), tag 1:
/// octet_string(signature) }` where NOCSRElements is itself
/// `struct { tag 1: octet_string(pubkey, 65 bytes), tag 2: octet_string(nonce) }`.
fn parse_csr_response(resp: &[u8]) -> Option<Vec<u8>> {
    let nocsr_elements = extract_outer_octet_string(resp, 0)?;
    // NOCSRElements is a struct; scan its body for tag 1 octet_string.
    let body = if nocsr_elements.first() == Some(&0x15u8) {
        &nocsr_elements[1..]
    } else {
        &nocsr_elements[..]
    };
    let csr_pubkey = extract_first_octet_string_at_tag(body, 1)?;
    if csr_pubkey.len() != 65 {
        return None;
    }
    Some(csr_pubkey)
}

/// Parse a NOCResponse body and return the status code at tag 0.
fn parse_noc_response_status(resp: &[u8]) -> Option<u8> {
    let body = if resp.first() == Some(&0x15u8) {
        &resp[1..]
    } else {
        resp
    };
    let mut i = 0;
    while i + 2 < body.len() {
        // ctrl = 0x20 | TYPE_UNSIGNED_INT_1 = 0x24, tag = 0
        if body[i] == 0x24 && body[i + 1] == 0 {
            return Some(body[i + 2]);
        }
        i += 1;
    }
    None
}

/// Find the first octet-string TLV at the outer level of `resp`, at
/// the given context tag. `resp` is typically a `struct { ... }` body.
fn extract_outer_octet_string(resp: &[u8], tag: u8) -> Option<Vec<u8>> {
    let body = if resp.first() == Some(&0x15u8) {
        &resp[1..]
    } else {
        resp
    };
    // Walk top-level fields only (don't descend into nested structs).
    let mut i = 0;
    while i + 2 < body.len() {
        if body[i] == 0x18 {
            break;
        }
        let ctrl = body[i];
        let cur_tag = body[i + 1];
        // octet_string with 1-byte length = 0x30 (context tag 1 | TYPE_OCTET_STRING_1)
        if ctrl == 0x30 && cur_tag == tag {
            let len = body[i + 2] as usize;
            let start = i + 3;
            if start + len > body.len() {
                return None;
            }
            return Some(body[start..start + len].to_vec());
        }
        // Skip this element. For octet strings with 1-byte length we can
        // compute the stride; for others we scan one byte at a time.
        if ctrl == 0x30 {
            let len = body[i + 2] as usize;
            i = i + 3 + len;
        } else {
            i += 1;
        }
    }
    None
}

/// Scan `body` for the first octet-string at the given context tag.
fn extract_first_octet_string_at_tag(body: &[u8], tag: u8) -> Option<Vec<u8>> {
    let mut i = 0;
    while i + 2 < body.len() {
        if body[i] == 0x30 && body[i + 1] == tag {
            let len = body[i + 2] as usize;
            let start = i + 3;
            if start + len > body.len() {
                return None;
            }
            return Some(body[start..start + len].to_vec());
        }
        i += 1;
    }
    None
}

/// Generate a random Matter fabric-id. Keep it in the 64-bit range but
/// avoid the well-known all-zero and all-ones values.
fn rand_fabric_id() -> u64 {
    use rand_core::RngCore;
    let mut buf = [0u8; 8];
    rand_core::OsRng.fill_bytes(&mut buf);
    let v = u64::from_le_bytes(buf);
    if v == 0 || v == u64::MAX {
        0x0000_0001_0000_0001
    } else {
        v
    }
}

// ── TLV → AttributeValue conversion ──────────────────────────────────────────

/// Convert raw TLV bytes (attribute value blob) to an `AttributeValue`.
///
/// This handles the common cases: uint8, uint16, uint32, bool, and raw bytes.
/// Unknown encodings are returned as `AttributeValue::Bytes`.
fn tlv_to_attribute_value(data: &[u8]) -> AttributeValue {
    if data.is_empty() {
        return AttributeValue::Null;
    }

    // The data may be wrapped in a TLV struct — skip outer struct wrapper if present
    let inner = if data[0] == 0x15 {
        // Anonymous struct: peek inside at the first element
        if data.len() >= 3 { &data[1..] } else { data }
    } else {
        data
    };

    if inner.is_empty() {
        return AttributeValue::Null;
    }

    let ctrl = inner[0];
    let val_type = ctrl & 0x1F;
    let tag_type = (ctrl >> 5) & 0x07;

    // Skip tag bytes
    let value_start = 1 + if tag_type == 1 { 1usize } else { 0usize };

    match val_type {
        0x08 => AttributeValue::Bool(false),
        0x09 => AttributeValue::Bool(true),
        0x04 if inner.len() > value_start => AttributeValue::U8(inner[value_start]),
        0x05 if inner.len() >= value_start + 2 => AttributeValue::U16(u16::from_le_bytes([
            inner[value_start],
            inner[value_start + 1],
        ])),
        0x06 if inner.len() >= value_start + 4 => {
            let bytes: [u8; 4] = inner[value_start..value_start + 4].try_into().unwrap();
            AttributeValue::U32(u32::from_le_bytes(bytes))
        }
        _ => AttributeValue::Bytes(data.to_vec()),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_matter_message_has_correct_session_id() {
        let msg = build_matter_message(0x0042, 1, vec![0xDE, 0xAD]);
        assert_eq!(msg.header.session_id, 0x0042);
        assert_eq!(msg.header.message_counter, 1);
        assert_eq!(msg.payload, vec![0xDE, 0xAD]);
    }

    #[test]
    fn tlv_to_attribute_value_uint8() {
        // ctrl = 0x04 (anonymous uint8), val = 127
        let data = vec![0x04u8, 127];
        assert_eq!(tlv_to_attribute_value(&data), AttributeValue::U8(127));
    }

    #[test]
    fn tlv_to_attribute_value_bool_true() {
        let data = vec![0x09u8]; // anonymous bool true
        assert_eq!(tlv_to_attribute_value(&data), AttributeValue::Bool(true));
    }

    #[test]
    fn tlv_to_attribute_value_bool_false() {
        let data = vec![0x08u8]; // anonymous bool false
        assert_eq!(tlv_to_attribute_value(&data), AttributeValue::Bool(false));
    }

    #[test]
    fn tlv_to_attribute_value_uint16() {
        let data = vec![0x05u8, 0x01, 0x00]; // uint16 = 1
        assert_eq!(tlv_to_attribute_value(&data), AttributeValue::U16(1));
    }

    #[test]
    fn tlv_to_attribute_value_empty_is_null() {
        assert_eq!(tlv_to_attribute_value(&[]), AttributeValue::Null);
    }

    #[test]
    fn parse_noc_response_status_reads_tag0_uint8() {
        // struct { tag0: uint8(0), tag1: uint8(1) }
        // ctrl=0x24 (ctx | uint8), tag, val; open/close with 0x15/0x18
        let body = vec![0x15u8, 0x24, 0, 0, 0x24, 1, 1, 0x18];
        assert_eq!(parse_noc_response_status(&body), Some(0));

        let body_err = vec![0x15u8, 0x24, 0, 0x20, 0x18];
        assert_eq!(parse_noc_response_status(&body_err), Some(0x20));
    }

    #[test]
    fn parse_csr_response_extracts_65_byte_pubkey() {
        // Build a plausible CSR response:
        //   struct {
        //     tag 0: octet_string(NOCSRElements)  ← 0x30
        //     tag 1: octet_string(sig)            ← 0x30
        //   }
        // NOCSRElements = struct { tag 1: octet_string(pubkey, 65 bytes),
        //                          tag 2: octet_string(nonce, 32 bytes) }
        let mut pubkey = vec![0x04u8]; // uncompressed prefix
        pubkey.extend_from_slice(&[0xAA; 64]);
        assert_eq!(pubkey.len(), 65);
        let nonce = vec![0x11u8; 32];

        let mut nocsr_inner = vec![
            0x15, // TYPE_STRUCTURE
            // tag 1: pubkey
            0x30,
            1,
            pubkey.len() as u8,
        ];
        nocsr_inner.extend_from_slice(&pubkey);
        // tag 2: nonce
        nocsr_inner.push(0x30);
        nocsr_inner.push(2);
        nocsr_inner.push(nonce.len() as u8);
        nocsr_inner.extend_from_slice(&nonce);
        nocsr_inner.push(0x18); // END_OF_CONTAINER

        let mut resp = vec![
            0x15, // outer struct
            // tag 0: NOCSRElements bytes
            0x30,
            0,
            nocsr_inner.len() as u8,
        ];
        resp.extend_from_slice(&nocsr_inner);
        // tag 1: signature bytes (64)
        let sig = vec![0xBB; 64];
        resp.push(0x30);
        resp.push(1);
        resp.push(sig.len() as u8);
        resp.extend_from_slice(&sig);
        resp.push(0x18);

        let got = parse_csr_response(&resp).expect("pubkey extraction should succeed");
        assert_eq!(got, pubkey);
    }

    #[test]
    fn parse_csr_response_rejects_wrong_pubkey_length() {
        // 60 bytes instead of 65 → must be rejected
        let mut short_key = vec![0x04u8];
        short_key.extend_from_slice(&[0xCC; 59]);
        assert_eq!(short_key.len(), 60);

        let mut nocsr_inner = vec![0x15, 0x30, 1, short_key.len() as u8];
        nocsr_inner.extend_from_slice(&short_key);
        nocsr_inner.push(0x18);

        let mut resp = vec![0x15, 0x30, 0, nocsr_inner.len() as u8];
        resp.extend_from_slice(&nocsr_inner);
        resp.push(0x18);

        assert!(parse_csr_response(&resp).is_none());
    }

    #[test]
    fn tlv_helpers_roundtrip() {
        // Smoke test: helpers produce the expected byte layouts.
        assert_eq!(
            tlv_uint16_ctx(3, 0x0102),
            vec![0x25, 3, 0x02, 0x01],
            "uint16 LE"
        );
        assert_eq!(
            tlv_uint64_ctx(4, 0x0807_0605_0403_0201),
            vec![0x27, 4, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
            "uint64 LE"
        );
        let s = tlv_octet_string_ctx(0, &[0xAA, 0xBB]);
        assert_eq!(s, vec![0x30, 0, 2, 0xAA, 0xBB]);

        let wrapped = build_struct(&[tlv_uint16_ctx(0, 5)]);
        assert_eq!(wrapped.first(), Some(&0x15));
        assert_eq!(wrapped.last(), Some(&0x18));
    }
}
