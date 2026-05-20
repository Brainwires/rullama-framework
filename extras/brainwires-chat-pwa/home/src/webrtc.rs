//! WebRTC peer wrapper around `webrtc-rs` (the Brainwires fork pinned by
//! the workspace at `0.20.0-alpha.1`).
//!
//! M1 exercises the create-offer / set-answer dance between two local peers
//! and confirms a JSON frame round-trips on a `"a2a"` data channel. The
//! length-prefixed frame codec, ICE-restart reconnect, and Cloudflare Calls
//! TURN credential minting land in M3 and M7 respectively.
//!
//! ## API note
//!
//! Upstream webrtc-rs (the crates.io tree) wires events with closures:
//! `pc.on_ice_candidate(Box::new(|c| async move { ... }))`. The Brainwires
//! fork uses an event-handler trait passed at builder time:
//! `PeerConnectionBuilder::new().with_handler(Arc<dyn PeerConnectionEventHandler>)`.
//! That trait drives the [`build_peer`] / [`PeerHandler`] split below.
//! DataChannel reads are also poll-based on the fork (`dc.poll()`), not
//! callback-based.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use bytes::BytesMut;
use serde_json::Value;
use tokio::sync::broadcast;
use webrtc::data_channel::{DataChannel as WrtcDataChannel, DataChannelEvent, RTCDataChannelInit};
use webrtc::media_stream::track_remote::TrackRemote;
use webrtc::peer_connection::{
    MediaEngine, PeerConnection, PeerConnectionBuilder, PeerConnectionEventHandler,
    RTCConfigurationBuilder, RTCIceConnectionState, RTCIceServer, RTCPeerConnectionIceEvent,
    RTCPeerConnectionState, RTCSessionDescription, RTCSignalingState, Registry,
    register_default_interceptors,
};

use crate::a2a::{self, A2aBridge};
use crate::binary::{
    BinBeginParams, BinChunkParams, BinEndParams, BinaryError, ERR_SEQ_OUT_OF_ORDER,
    ERR_SHA256_MISMATCH, ERR_UNKNOWN_BIN_ID,
};
use crate::signaling::SessionState;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use brainwires_a2a::{JsonRpcRequest, JsonRpcResponse};

/// The canonical data-channel label used by the dial-home protocol.
pub const A2A_CHANNEL_LABEL: &str = "a2a";

/// Lightweight event broadcast for an [`HomePeer`]. Keeps the M1 test (and
/// future signaling-route plumbing) out of the `webrtc-rs` event-handler
/// trait directly.
///
/// `Arc<dyn DataChannel>` does not implement `Debug`, so we hand-roll the
/// formatter rather than `#[derive(Debug)]` it.
#[derive(Clone)]
pub enum PeerEvent {
    /// New ICE candidate the local peer wants to send to the remote.
    LocalIceCandidate {
        candidate: String,
        sdp_mid: Option<String>,
        sdp_mline_index: Option<u16>,
    },
    /// Connection state changed.
    ConnectionState(RTCPeerConnectionState),
    /// ICE connection state changed.
    IceConnectionState(RTCIceConnectionState),
    /// Signaling state changed.
    SignalingState(RTCSignalingState),
    /// Remote opened a data channel (answerer side).
    DataChannel(Arc<dyn WrtcDataChannel>),
}

impl std::fmt::Debug for PeerEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LocalIceCandidate {
                candidate,
                sdp_mid,
                sdp_mline_index,
            } => f
                .debug_struct("LocalIceCandidate")
                .field("candidate", candidate)
                .field("sdp_mid", sdp_mid)
                .field("sdp_mline_index", sdp_mline_index)
                .finish(),
            Self::ConnectionState(s) => f.debug_tuple("ConnectionState").field(s).finish(),
            Self::IceConnectionState(s) => f.debug_tuple("IceConnectionState").field(s).finish(),
            Self::SignalingState(s) => f.debug_tuple("SignalingState").field(s).finish(),
            Self::DataChannel(_) => f
                .debug_tuple("DataChannel")
                .field(&"<dyn DataChannel>")
                .finish(),
        }
    }
}

struct PeerHandler {
    tx: broadcast::Sender<PeerEvent>,
}

#[async_trait::async_trait]
impl PeerConnectionEventHandler for PeerHandler {
    async fn on_ice_candidate(&self, event: RTCPeerConnectionIceEvent) {
        if let Ok(init) = event.candidate.to_json() {
            let _ = self.tx.send(PeerEvent::LocalIceCandidate {
                candidate: init.candidate,
                sdp_mid: init.sdp_mid,
                sdp_mline_index: init.sdp_mline_index,
            });
        }
    }

    async fn on_connection_state_change(&self, state: RTCPeerConnectionState) {
        let _ = self.tx.send(PeerEvent::ConnectionState(state));
    }

    async fn on_ice_connection_state_change(&self, state: RTCIceConnectionState) {
        let _ = self.tx.send(PeerEvent::IceConnectionState(state));
    }

    async fn on_signaling_state_change(&self, state: RTCSignalingState) {
        let _ = self.tx.send(PeerEvent::SignalingState(state));
    }

    async fn on_track(&self, _track: Arc<dyn TrackRemote>) {
        // M1 is data-channel only; ignore media tracks.
    }

    async fn on_data_channel(&self, dc: Arc<dyn WrtcDataChannel>) {
        let _ = self.tx.send(PeerEvent::DataChannel(dc));
    }
}

/// One end of a WebRTC connection plus a broadcast bus for its events.
pub struct HomePeer {
    pub pc: Arc<dyn PeerConnection>,
    tx: broadcast::Sender<PeerEvent>,
}

impl HomePeer {
    /// Subscribe to this peer's event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<PeerEvent> {
        self.tx.subscribe()
    }
}

/// Build a default-configured peer.
///
/// `ice_servers` is a list of STUN/TURN URLs. Pass an empty `Vec` to use the
/// default Google STUN. M7 will replace this with a Cloudflare-Calls-minted
/// TURN credential.
pub async fn build_peer(ice_servers: Vec<String>) -> Result<HomePeer> {
    let urls = if ice_servers.is_empty() {
        vec!["stun:stun.l.google.com:19302".to_string()]
    } else {
        ice_servers
    };

    let mut media_engine = MediaEngine::default();
    media_engine
        .register_default_codecs()
        .map_err(|e| anyhow!("register_default_codecs: {e}"))?;
    let registry = register_default_interceptors(Registry::new(), &mut media_engine)
        .map_err(|e| anyhow!("register_default_interceptors: {e}"))?;

    let cfg = RTCConfigurationBuilder::new()
        .with_ice_servers(vec![RTCIceServer {
            urls,
            username: String::new(),
            credential: String::new(),
        }])
        .build();

    let (tx, _) = broadcast::channel::<PeerEvent>(64);
    let handler = Arc::new(PeerHandler { tx: tx.clone() });

    // `PeerConnectionBuilder` is generic over the address type used for UDP/TCP
    // candidate bindings (`A: ToSocketAddrs`). We bind on ephemeral ports on
    // `0.0.0.0` so host-candidate gathering works between two in-process peers
    // without needing TURN. Using `&'static str` lets the inference resolve.
    let pc: Arc<dyn PeerConnection> = Arc::new(
        PeerConnectionBuilder::<&'static str>::new()
            .with_configuration(cfg)
            .with_media_engine(media_engine)
            .with_interceptor_registry(registry)
            .with_handler(handler.clone() as Arc<dyn PeerConnectionEventHandler>)
            .with_udp_addrs(vec!["0.0.0.0:0"])
            .build()
            .await
            .map_err(|e| anyhow!("PeerConnectionBuilder::build: {e}"))?,
    );

    Ok(HomePeer { pc, tx })
}

/// Open the canonical `"a2a"` data channel as the offerer.
pub async fn open_a2a_channel(peer: &HomePeer) -> Result<Arc<dyn WrtcDataChannel>> {
    let init = RTCDataChannelInit {
        ordered: true,
        max_retransmits: None,
        max_packet_life_time: None,
        protocol: String::new(),
        negotiated: None,
    };
    peer.pc
        .create_data_channel(A2A_CHANNEL_LABEL, Some(init))
        .await
        .map_err(|e| anyhow!("create_data_channel({A2A_CHANNEL_LABEL}): {e}"))
}

/// Send a UTF-8 text frame on a data channel.
pub async fn send_text(dc: &Arc<dyn WrtcDataChannel>, s: &str) -> Result<()> {
    dc.send_text(s).await.map_err(|e| anyhow!("send_text: {e}"))
}

/// Send a binary frame on a data channel.
pub async fn send_bytes(dc: &Arc<dyn WrtcDataChannel>, data: &[u8]) -> Result<()> {
    dc.send(BytesMut::from(data))
        .await
        .map_err(|e| anyhow!("send_bytes: {e}"))
}

/// Read the next text/binary message off a data channel. Returns `None` when
/// the channel closes.
pub async fn recv_text(dc: &Arc<dyn WrtcDataChannel>) -> Option<String> {
    loop {
        match dc.poll().await {
            Some(DataChannelEvent::OnMessage(msg)) => {
                return Some(String::from_utf8_lossy(&msg.data).into_owned());
            }
            Some(DataChannelEvent::OnClose) | None => return None,
            _ => continue,
        }
    }
}

/// Drive a peer to [`RTCPeerConnectionState::Connected`] (or an error). Used
/// in tests to gate on connection establishment.
pub async fn wait_connected(peer: &HomePeer) -> Result<()> {
    let mut rx = peer.subscribe();
    loop {
        match rx.recv().await {
            Ok(PeerEvent::ConnectionState(RTCPeerConnectionState::Connected)) => return Ok(()),
            Ok(PeerEvent::ConnectionState(RTCPeerConnectionState::Failed)) => {
                return Err(anyhow!("peer entered Failed state before Connected"));
            }
            Ok(PeerEvent::ConnectionState(RTCPeerConnectionState::Closed)) => {
                return Err(anyhow!("peer Closed before Connected"));
            }
            Ok(_) => continue,
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(_) => return Err(anyhow!("peer event stream ended before Connected")),
        }
    }
}

/// Forward local ICE candidates from `from` into `into.add_ice_candidate`.
/// Returns a JoinHandle that exits when `from`'s broadcast closes or the peer
/// reaches Connected/Failed.
pub fn spawn_ice_relay(
    from: &HomePeer,
    into: Arc<dyn PeerConnection>,
) -> tokio::task::JoinHandle<()> {
    let mut rx = from.subscribe();
    tokio::spawn(async move {
        use webrtc::peer_connection::RTCIceCandidateInit;
        loop {
            match rx.recv().await {
                Ok(PeerEvent::LocalIceCandidate {
                    candidate,
                    sdp_mid,
                    sdp_mline_index,
                }) => {
                    let _ = into
                        .add_ice_candidate(RTCIceCandidateInit {
                            candidate,
                            sdp_mid,
                            sdp_mline_index,
                            username_fragment: None,
                            url: None,
                        })
                        .await;
                }
                Ok(PeerEvent::ConnectionState(
                    RTCPeerConnectionState::Connected
                    | RTCPeerConnectionState::Failed
                    | RTCPeerConnectionState::Closed,
                )) => {
                    break;
                }
                Err(broadcast::error::RecvError::Closed) => break,
                _ => continue,
            }
        }
    })
}

/// Build a fresh answerer peer (no SDP applied yet). Caller subscribes to its
/// event stream BEFORE calling [`apply_offer_and_create_answer`] so that no
/// ICE candidates are missed.
pub async fn build_answerer() -> Result<HomePeer> {
    build_peer(vec![]).await
}

/// Apply an offer SDP to an answerer peer and produce the answer SDP.
///
/// Sequence (matches the WebRTC spec):
///   1. `set_remote_description(offer)`.
///   2. `create_answer()`.
///   3. `set_local_description(answer)` — *this kicks off ICE gathering*.
///
/// Local ICE candidates surface via the peer's event broadcast immediately
/// after step 3. The caller must already be subscribed.
pub async fn apply_offer_and_create_answer(peer: &HomePeer, offer_sdp: String) -> Result<String> {
    let offer = RTCSessionDescription::offer(offer_sdp)
        .map_err(|e| anyhow!("RTCSessionDescription::offer: {e}"))?;
    peer.pc
        .set_remote_description(offer)
        .await
        .map_err(|e| anyhow!("answerer.set_remote_description(offer): {e}"))?;
    let answer = peer
        .pc
        .create_answer(None)
        .await
        .map_err(|e| anyhow!("answerer.create_answer: {e}"))?;
    let answer_sdp = answer.sdp.clone();
    peer.pc
        .set_local_description(answer)
        .await
        .map_err(|e| anyhow!("answerer.set_local_description(answer): {e}"))?;
    Ok(answer_sdp)
}

/// Wait for the offerer to expose the `"a2a"` data channel, then drive a
/// message loop that routes every inbound JSON-RPC frame through the
/// supplied [`A2aBridge`].
///
/// Returns the resolved data channel as soon as it shows up so the caller
/// can stash it on the per-session state. The actual message-pump loop is
/// spawned onto the runtime — its handle is `tokio::spawn`'d and dropped,
/// so it lives until the data channel closes.
///
/// If `bridge` is `None`, only the legacy `ping` method is answered (the
/// M3 smoke-test path). This is the fallback for tests that construct an
/// [`AppState`] without a bridge attached.
pub async fn run_a2a_loop(
    peer: &HomePeer,
    bridge: Option<Arc<A2aBridge>>,
) -> Result<Arc<dyn WrtcDataChannel>> {
    run_a2a_loop_with_session(peer, bridge, None, None).await
}

/// Variant of [`run_a2a_loop`] that also pushes every reply frame onto the
/// supplied [`SessionState`]'s outbox and answers the transport-level
/// `system/resume` method directly without going through the bridge.
///
/// Wired by `signaling::post_offer` in M10. The plain [`run_a2a_loop`] is
/// retained for tests and the legacy ping-only path.
pub async fn run_a2a_loop_with_session(
    peer: &HomePeer,
    bridge: Option<Arc<A2aBridge>>,
    session: Option<Arc<SessionState>>,
    sync_store: Option<Arc<crate::sync::SyncStore>>,
) -> Result<Arc<dyn WrtcDataChannel>> {
    let mut rx = peer.subscribe();
    let dc = loop {
        match rx.recv().await {
            Ok(PeerEvent::DataChannel(dc)) => {
                let label = dc.label().await.unwrap_or_default();
                if label == A2A_CHANNEL_LABEL {
                    break dc;
                }
            }
            Ok(_) => continue,
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(_) => {
                return Err(anyhow!(
                    "peer event stream ended before data channel arrived"
                ));
            }
        }
    };

    // Spawn the message-pump task. Lives until the data channel closes.
    let pump_dc = dc.clone();
    tokio::spawn(async move {
        loop {
            match pump_dc.poll().await {
                Some(DataChannelEvent::OnMessage(msg)) => {
                    let text = String::from_utf8_lossy(&msg.data).into_owned();
                    if let Some(reply) = dispatch_jsonrpc(
                        &text,
                        bridge.as_deref(),
                        session.as_deref(),
                        sync_store.as_deref(),
                    )
                    .await
                    {
                        // Push onto the outbox before sending so a successful
                        // send is always reflected in the resume buffer. Worst
                        // case: we buffer a frame the PWA already received,
                        // which the resume-cursor filter then drops. Better
                        // than the alternative (sent-but-not-buffered).
                        if let Some(s) = session.as_deref()
                            && let Some(id) = parse_outbox_id(&reply)
                        {
                            s.push_outbox(id, reply.clone()).await;
                        }
                        if let Err(e) = pump_dc.send_text(&reply).await {
                            tracing::warn!(error = %e, "a2a: send_text failed");
                            break;
                        }
                    }
                }
                Some(DataChannelEvent::OnClose) | None => break,
                _ => continue,
            }
        }
    });

    Ok(dc)
}

/// Extract a numeric JSON-RPC id from an outbound reply frame for outbox
/// indexing. Replies with string ids are still buffered but only at the
/// tail of the queue (caller-side `resume` only filters by numeric `>`,
/// so string-id frames would never replay; we therefore skip them).
fn parse_outbox_id(frame: &str) -> Option<i64> {
    let v: Value = serde_json::from_str(frame).ok()?;
    v.get("id").and_then(|i| i.as_i64())
}

/// Transport-level JSON-RPC method that asks the daemon to replay any
/// reply frames the PWA missed during a brief disconnect. Handled here
/// (not via the [`A2aBridge`]) because it's a pure transport concern —
/// the agent never sees a `system/resume` call.
pub const METHOD_SYSTEM_RESUME: &str = "system/resume";

/// M11 — declare a binary upload's id, content-type, total size, and
/// chunk count. Allocates a per-session pending buffer.
pub const METHOD_BIN_BEGIN: &str = "bin/begin";

/// M11 — append one base64-encoded chunk to a pending buffer. Must
/// arrive in order (`seq` strictly equals the next expected slot).
pub const METHOD_BIN_CHUNK: &str = "bin/chunk";

/// M11 — finalize a pending buffer, optionally checking sha256, and
/// move the assembled bytes to the finalized-blobs map. A subsequent
/// `message/send` whose A2A `Part.metadata.bin_id` matches will consume
/// the blob and have it inlined as `Part.raw` (base64) before the
/// agent sees the message.
pub const METHOD_BIN_END: &str = "bin/end";

/// Metadata key recognized on an A2A `Part` to pull a finalized blob
/// from the per-session binary store and inline it. Lives under
/// `Part.metadata` so we don't need to extend the typed `Part` struct.
pub const BIN_ID_METADATA_KEY: &str = "bin_id";

/// Cross-device sync: push local changelog entries to the daemon.
pub const METHOD_SYNC_PUSH: &str = "sync/push";
/// Cross-device sync: pull remote changelog entries from the daemon.
pub const METHOD_SYNC_PULL: &str = "sync/pull";
/// Cross-device sync: acknowledge received entries (enables compaction).
pub const METHOD_SYNC_ACK: &str = "sync/ack";

/// Dispatch one inbound text frame and serialize the reply.
///
/// Order of precedence:
///   1. Transport-level methods handled here directly (M10: `system/resume`,
///      M11: `bin/begin`, `bin/chunk`, `bin/end`).
///   2. `message/send` is intercepted to resolve any `bin_id`-bearing file
///      parts against the session's [`crate::binary::BinaryStore`] before
///      forwarding to the bridge.
///   3. Typed bridge dispatch when an [`A2aBridge`] is attached.
///   4. Untyped fallback for the M3 smoke-test path (raw `ping`).
async fn dispatch_jsonrpc(
    text: &str,
    bridge: Option<&A2aBridge>,
    session: Option<&SessionState>,
    sync_store: Option<&crate::sync::SyncStore>,
) -> Option<String> {
    // M10/M11 — transport-level methods. Peek at `method` before dispatching
    // to the bridge so we don't round-trip system/* or bin/* calls through
    // the agent.
    let parsed_value: Option<Value> = serde_json::from_str(text).ok();
    if let Some(v) = parsed_value.as_ref()
        && let Some(method) = v.get("method").and_then(|m| m.as_str())
    {
        match method {
            METHOD_SYSTEM_RESUME => return handle_resume(v, session).await,
            METHOD_BIN_BEGIN | METHOD_BIN_CHUNK | METHOD_BIN_END => {
                return handle_bin(method, v, session).await;
            }
            METHOD_SYNC_PUSH | METHOD_SYNC_PULL | METHOD_SYNC_ACK => {
                return handle_sync(method, v, sync_store);
            }
            _ => {}
        }
    }

    if let Some(bridge) = bridge {
        // M11 — pre-process `message/send` to resolve `bin_id`-bearing
        // file parts to inline bytes before the typed bridge sees them.
        // For every other method this returns the original frame text
        // unchanged.
        let resolved = match (parsed_value, session) {
            (Some(v), Some(s)) => rewrite_message_send_with_bins(v, s)
                .await
                .unwrap_or(text.to_string()),
            _ => text.to_string(),
        };

        // Typed path. If the frame is a valid JSON-RPC envelope, hand it to
        // the bridge — even errors (method-not-found, invalid-params, ...)
        // come back as well-formed JsonRpcResponse frames.
        match serde_json::from_str::<JsonRpcRequest>(&resolved) {
            Ok(req) => {
                let resp: JsonRpcResponse = bridge.dispatch(req).await;
                return serde_json::to_string(&resp).ok();
            }
            Err(e) => {
                tracing::debug!(error = %e, frame = %resolved, "a2a: bridge couldn't parse frame, falling back");
            }
        }
    }

    // Untyped fallback. This is what the M3 smoke-test exercises (raw
    // `{ jsonrpc, id, method: "ping" }` without a typed envelope check).
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(error = %e, frame = %text, "a2a: dropping non-JSON frame");
            return None;
        }
    };
    let method = v.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let id = v.get("id").cloned().unwrap_or(Value::Null);
    match method {
        "ping" => Some(a2a::handle_jsonrpc_ping(id).to_string()),
        other => {
            tracing::debug!(method = %other, "a2a: no bridge attached and method != ping; dropping");
            None
        }
    }
}

/// Build a JSON-RPC error reply for the given id with a custom code.
fn make_error_reply(id_value: &Value, code: i32, message: &str) -> Option<String> {
    let resp = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id_value,
        "error": { "code": code, "message": message },
    });
    serde_json::to_string(&resp).ok()
}

fn make_success_reply(id_value: &Value, result: Value) -> Option<String> {
    let resp = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id_value,
        "result": result,
    });
    serde_json::to_string(&resp).ok()
}

/// Map a [`BinaryError`] to a JSON-RPC error code per the M11 wire spec.
fn binary_error_to_code(err: &BinaryError) -> i32 {
    match err {
        BinaryError::UnknownBinId(_) => ERR_UNKNOWN_BIN_ID,
        BinaryError::SeqOutOfOrder { .. } => ERR_SEQ_OUT_OF_ORDER,
        BinaryError::Sha256Mismatch => ERR_SHA256_MISMATCH,
        // Everything else is invalid-params — base64 garbage, oversized
        // chunks, declared total exceeded, ...
        _ => brainwires_a2a::error::INVALID_PARAMS,
    }
}

/// Handle one of the three `bin/*` methods. Always returns `Some(reply)`
/// (never `None`) so the data-channel pump always emits a response.
async fn handle_bin(method: &str, req: &Value, session: Option<&SessionState>) -> Option<String> {
    let id_value = req.get("id").cloned().unwrap_or(Value::Null);
    let Some(session) = session else {
        // Without a session we can't track buffers. Reply with an
        // INVALID_PARAMS error rather than silently dropping the frame.
        return make_error_reply(
            &id_value,
            brainwires_a2a::error::INVALID_PARAMS,
            "binary chunking unavailable: no session context",
        );
    };
    let params = req.get("params").cloned().unwrap_or(Value::Null);

    match method {
        METHOD_BIN_BEGIN => {
            let parsed: BinBeginParams = match serde_json::from_value(params) {
                Ok(p) => p,
                Err(e) => {
                    return make_error_reply(
                        &id_value,
                        brainwires_a2a::error::INVALID_PARAMS,
                        &format!("bin/begin: malformed params: {e}"),
                    );
                }
            };
            match session.binaries.handle_begin(parsed).await {
                Ok(ok) => {
                    let v = serde_json::to_value(&ok).unwrap_or(Value::Null);
                    make_success_reply(&id_value, v)
                }
                Err(e) => make_error_reply(&id_value, binary_error_to_code(&e), &e.to_string()),
            }
        }
        METHOD_BIN_CHUNK => {
            let parsed: BinChunkParams = match serde_json::from_value(params) {
                Ok(p) => p,
                Err(e) => {
                    return make_error_reply(
                        &id_value,
                        brainwires_a2a::error::INVALID_PARAMS,
                        &format!("bin/chunk: malformed params: {e}"),
                    );
                }
            };
            match session.binaries.handle_chunk(parsed).await {
                Ok(ok) => {
                    let v = serde_json::to_value(&ok).unwrap_or(Value::Null);
                    make_success_reply(&id_value, v)
                }
                Err(e) => make_error_reply(&id_value, binary_error_to_code(&e), &e.to_string()),
            }
        }
        METHOD_BIN_END => {
            let parsed: BinEndParams = match serde_json::from_value(params) {
                Ok(p) => p,
                Err(e) => {
                    return make_error_reply(
                        &id_value,
                        brainwires_a2a::error::INVALID_PARAMS,
                        &format!("bin/end: malformed params: {e}"),
                    );
                }
            };
            match session.binaries.handle_end(parsed).await {
                Ok(blob) => {
                    let v = serde_json::json!({
                        "ok": true,
                        "size": blob.bytes.len() as u64,
                    });
                    make_success_reply(&id_value, v)
                }
                Err(e) => make_error_reply(&id_value, binary_error_to_code(&e), &e.to_string()),
            }
        }
        _ => unreachable!("handle_bin called with non-bin method"),
    }
}

/// Handle `sync/push` and `sync/pull` methods.
fn handle_sync(
    method: &str,
    req: &Value,
    sync_store: Option<&crate::sync::SyncStore>,
) -> Option<String> {
    let id_value = req.get("id").cloned().unwrap_or(Value::Null);
    let Some(store) = sync_store else {
        return make_error_reply(
            &id_value,
            brainwires_a2a::error::INVALID_PARAMS,
            "sync not configured on this daemon",
        );
    };
    let params = req.get("params").cloned().unwrap_or(Value::Null);

    match method {
        METHOD_SYNC_PUSH => {
            let device_id = params
                .get("device_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if device_id.is_empty() {
                return make_error_reply(
                    &id_value,
                    brainwires_a2a::error::INVALID_PARAMS,
                    "sync/push: device_id required",
                );
            }
            let entries: Vec<crate::sync::SyncEntry> = match params.get("entries") {
                Some(arr) => match serde_json::from_value(arr.clone()) {
                    Ok(e) => e,
                    Err(e) => {
                        return make_error_reply(
                            &id_value,
                            brainwires_a2a::error::INVALID_PARAMS,
                            &format!("sync/push: invalid entries: {e}"),
                        );
                    }
                },
                None => {
                    return make_error_reply(
                        &id_value,
                        brainwires_a2a::error::INVALID_PARAMS,
                        "sync/push: entries array required",
                    );
                }
            };
            match store.push(device_id, &entries) {
                Ok(seq) => {
                    let v = serde_json::json!({ "stored": entries.len(), "seq": seq });
                    make_success_reply(&id_value, v)
                }
                Err(e) => make_error_reply(&id_value, -32000, &format!("sync/push failed: {e}")),
            }
        }
        METHOD_SYNC_PULL => {
            let device_id = params
                .get("device_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if device_id.is_empty() {
                return make_error_reply(
                    &id_value,
                    brainwires_a2a::error::INVALID_PARAMS,
                    "sync/pull: device_id required",
                );
            }
            let since = params.get("since").and_then(|v| v.as_i64()).unwrap_or(0);
            let limit = params.get("limit").and_then(|v| v.as_i64()).unwrap_or(100) as usize;
            match store.pull(device_id, since, limit) {
                Ok(result) => {
                    let v = serde_json::json!({
                        "entries": result.entries,
                        "has_more": result.has_more,
                        "latest_seq": result.latest_seq,
                    });
                    make_success_reply(&id_value, v)
                }
                Err(e) => make_error_reply(&id_value, -32000, &format!("sync/pull failed: {e}")),
            }
        }
        METHOD_SYNC_ACK => {
            let device_id = params
                .get("device_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if device_id.is_empty() {
                return make_error_reply(
                    &id_value,
                    brainwires_a2a::error::INVALID_PARAMS,
                    "sync/ack: device_id required",
                );
            }
            let seq = params.get("seq").and_then(|v| v.as_i64()).unwrap_or(0);
            match store.ack(device_id, seq) {
                Ok(()) => make_success_reply(&id_value, serde_json::json!({ "ok": true })),
                Err(e) => make_error_reply(&id_value, -32000, &format!("sync/ack failed: {e}")),
            }
        }
        _ => unreachable!("handle_sync called with non-sync method"),
    }
}

/// M11 — when the inbound frame is a `message/send`, walk the message's
/// `parts` array and replace any part whose `metadata.bin_id` matches a
/// finalized blob with an inline part carrying `raw` (base64) + the
/// blob's content-type + the original filename. Consuming the blob is
/// one-shot (the `take` call removes it from the store).
///
/// Returns `Some(rewritten_text)` when the frame is a `message/send` and
/// at least one part was rewritten, else `None`. The caller should fall
/// back to the original text in either case where this returns `None`.
///
/// Errors during rewrite (e.g. unknown `bin_id`) are intentionally
/// non-fatal at this layer: they fall through to the bridge as the
/// original-with-`bin_id` part, which the agent will see as an empty
/// file part. This keeps the rewrite layer transparent — the daemon
/// doesn't reject the message just because a buffer expired between
/// `bin/end` and `message/send`. The agent layer (or future
/// validation) decides what to do.
async fn rewrite_message_send_with_bins(mut v: Value, session: &SessionState) -> Option<String> {
    let method = v.get("method").and_then(|m| m.as_str())?;
    if method != "message/send" && method != brainwires_a2a::jsonrpc::METHOD_MESSAGE_SEND {
        return None;
    }
    let parts = v
        .get_mut("params")
        .and_then(|p| p.get_mut("message"))
        .and_then(|m| m.get_mut("parts"))
        .and_then(|p| p.as_array_mut())?;
    let mut rewrote_any = false;
    for part in parts.iter_mut() {
        let bin_id_opt = part
            .get("metadata")
            .and_then(|md| md.get(BIN_ID_METADATA_KEY))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let Some(bin_id) = bin_id_opt else { continue };
        let Some(blob) = session.binaries.take(&bin_id).await else {
            tracing::warn!(bin_id = %bin_id, "message/send: bin_id not found in finalized store; leaving placeholder part");
            continue;
        };
        // Encode bytes to base64 — A2A `Part.raw` is base64-encoded
        // raw content per the type docs. Slot it in alongside whatever
        // mediaType / filename the PWA already supplied (don't clobber
        // what the caller intentionally set).
        if let Some(obj) = part.as_object_mut() {
            obj.insert("raw".to_string(), Value::String(BASE64.encode(&blob.bytes)));
            if !obj.contains_key("mediaType")
                && let Some(ct) = blob.content_type.as_deref()
            {
                obj.insert("mediaType".to_string(), Value::String(ct.to_string()));
            }
            // Strip the `bin_id` key so it doesn't round-trip into the
            // agent's view of the message. Leave any other metadata
            // entries in place.
            if let Some(md) = obj.get_mut("metadata").and_then(|m| m.as_object_mut()) {
                md.remove(BIN_ID_METADATA_KEY);
                if md.is_empty() {
                    obj.remove("metadata");
                }
            }
            rewrote_any = true;
        }
    }
    if !rewrote_any {
        return None;
    }
    serde_json::to_string(&v).ok()
}

/// Build the `system/resume` reply for the given session state.
///
/// Reply shape: `{ jsonrpc, id, result: { replayed: [<frame strings>], dropped: <bool> } }`.
/// When no session is in scope (legacy `run_a2a_loop` path) we return an
/// empty replay so the PWA still gets a well-formed response.
async fn handle_resume(req: &Value, session: Option<&SessionState>) -> Option<String> {
    // Prefer numeric ids (the PWA's dispatcher always mints i64), but echo
    // whatever form arrived to keep JsonRpcDispatcher routing happy.
    let id_value = req.get("id").cloned().unwrap_or(Value::Null);
    let last_seen_id = req
        .get("params")
        .and_then(|p| p.get("last_seen_id"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    let (replayed, dropped) = match session {
        Some(s) => s.resume_from(last_seen_id).await,
        None => (Vec::new(), false),
    };

    // Emit `replayed` as raw JSON so the PWA can re-feed each entry back
    // into its dispatcher untouched. Each element is the original frame
    // string; we wrap them as `Value::String(...)` here.
    let replayed_json: Vec<Value> = replayed.into_iter().map(Value::String).collect();
    let resp = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id_value,
        "result": {
            "replayed": replayed_json,
            "dropped": dropped,
        }
    });
    serde_json::to_string(&resp).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::mpsc;

    /// Spin up two peers in-process, do the offer/answer dance manually,
    /// open the `"a2a"` data channel, and round-trip a single ping/pong
    /// frame. Validates the webrtc-rs (Brainwires fork) scaffolding before
    /// it gets wired into the signaling server in M3.
    ///
    /// Requires `flavor = "multi_thread"` because webrtc-rs spawns
    /// background tasks that need a real thread pool.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ping_roundtrip_two_local_peers() -> Result<()> {
        let alice = build_peer(vec![]).await?;
        let bob = build_peer(vec![]).await?;

        // Forward ICE candidates between peers locally.
        let _alice_to_bob = spawn_ice_relay(&alice, bob.pc.clone());
        let _bob_to_alice = spawn_ice_relay(&bob, alice.pc.clone());

        // Bob (answerer): when the data channel arrives, spawn a poll task
        // that echoes back any text it receives.
        let (bob_got_tx, mut bob_got_rx) = mpsc::channel::<String>(1);
        let mut bob_events = bob.subscribe();
        let bob_dc_task = tokio::spawn(async move {
            while let Ok(ev) = bob_events.recv().await {
                if let PeerEvent::DataChannel(dc) = ev {
                    let bob_got_tx = bob_got_tx.clone();
                    tokio::spawn(async move {
                        loop {
                            match dc.poll().await {
                                Some(DataChannelEvent::OnMessage(msg)) => {
                                    let s = String::from_utf8_lossy(&msg.data).into_owned();
                                    let _ = bob_got_tx.send(s).await;
                                    let _ = dc.send_text("pong").await;
                                }
                                Some(DataChannelEvent::OnClose) | None => break,
                                _ => continue,
                            }
                        }
                    });
                    break;
                }
            }
        });

        // Alice (offerer): open the channel, send ping when open, capture pong.
        let dc = open_a2a_channel(&alice).await?;
        let (alice_got_tx, mut alice_got_rx) = mpsc::channel::<String>(1);
        let dc_for_reader = dc.clone();
        let alice_reader = tokio::spawn(async move {
            loop {
                match dc_for_reader.poll().await {
                    Some(DataChannelEvent::OnOpen) => {
                        let _ = dc_for_reader.send_text("ping").await;
                    }
                    Some(DataChannelEvent::OnMessage(msg)) => {
                        let s = String::from_utf8_lossy(&msg.data).into_owned();
                        let _ = alice_got_tx.send(s).await;
                        break;
                    }
                    Some(DataChannelEvent::OnClose) | None => break,
                    _ => continue,
                }
            }
        });

        // Offer / answer.
        let offer = alice
            .pc
            .create_offer(None)
            .await
            .map_err(|e| anyhow!("alice.create_offer: {e}"))?;
        let offer_sdp = offer.sdp.clone();
        alice
            .pc
            .set_local_description(offer)
            .await
            .map_err(|e| anyhow!("alice.set_local_description: {e}"))?;
        bob.pc
            .set_remote_description(
                RTCSessionDescription::offer(offer_sdp).map_err(|e| anyhow!("offer: {e}"))?,
            )
            .await
            .map_err(|e| anyhow!("bob.set_remote_description(offer): {e}"))?;
        let answer = bob
            .pc
            .create_answer(None)
            .await
            .map_err(|e| anyhow!("bob.create_answer: {e}"))?;
        let answer_sdp = answer.sdp.clone();
        bob.pc
            .set_local_description(answer)
            .await
            .map_err(|e| anyhow!("bob.set_local_description: {e}"))?;
        alice
            .pc
            .set_remote_description(
                RTCSessionDescription::answer(answer_sdp).map_err(|e| anyhow!("answer: {e}"))?,
            )
            .await
            .map_err(|e| anyhow!("alice.set_remote_description(answer): {e}"))?;

        // Bob should receive "ping".
        let bob_got = tokio::time::timeout(Duration::from_secs(15), bob_got_rx.recv())
            .await
            .map_err(|_| anyhow!("timed out waiting for bob to receive ping"))?
            .ok_or_else(|| anyhow!("bob channel closed before receiving ping"))?;
        assert_eq!(bob_got, "ping");

        // Alice should receive "pong".
        let alice_got = tokio::time::timeout(Duration::from_secs(15), alice_got_rx.recv())
            .await
            .map_err(|_| anyhow!("timed out waiting for alice to receive pong"))?
            .ok_or_else(|| anyhow!("alice channel closed before receiving pong"))?;
        assert_eq!(alice_got, "pong");

        // Cleanup.
        let _ = dc.close().await;
        let _ = alice.pc.close().await;
        let _ = bob.pc.close().await;
        let _ = alice_reader.await;
        let _ = bob_dc_task.await;
        Ok(())
    }

    #[tokio::test]
    async fn build_peer_smoke() -> Result<()> {
        let p = build_peer(vec![]).await?;
        // Just confirm we can create a data channel with the canonical label.
        let dc = open_a2a_channel(&p).await?;
        assert_eq!(dc.label().await.unwrap_or_default(), A2A_CHANNEL_LABEL);
        let _ = p.pc.close().await;
        Ok(())
    }

    // ───────── M11: bin/* dispatch + message/send rewrite ─────────

    use crate::a2a::A2aBridge;
    use crate::a2a::test_support::echo_chat_agent;
    use base64::engine::general_purpose::STANDARD as B64;
    use serde_json::json;
    use sha2::{Digest as _, Sha256};
    use std::sync::Arc;

    /// Push the M11 chunk methods through `dispatch_jsonrpc` end-to-end:
    /// bin/begin → bin/chunk → bin/end → message/send (with bin_id) →
    /// assert the agent sees the inlined bytes.
    #[tokio::test]
    async fn message_send_with_bin_ref_resolves_to_inline() {
        let session = Arc::new(SessionState::new("sess".to_string()));
        let bridge = A2aBridge::new(echo_chat_agent());

        let payload = b"hello binary world".to_vec();
        let mut hasher = Sha256::new();
        hasher.update(&payload);
        let sha = hex::encode(hasher.finalize());
        let b64 = B64.encode(&payload);
        let bin_id = "blob-1";

        // bin/begin
        let begin = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "bin/begin",
            "params": {
                "bin_id": bin_id,
                "content_type": "image/png",
                "total_size": payload.len(),
                "total_chunks": 1
            }
        })
        .to_string();
        let r = dispatch_jsonrpc(&begin, Some(&bridge), Some(&session), None)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&r).unwrap();
        assert_eq!(v["result"]["ok"], json!(true));

        // bin/chunk
        let chunk = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "bin/chunk",
            "params": { "bin_id": bin_id, "seq": 0, "data": b64 }
        })
        .to_string();
        let r = dispatch_jsonrpc(&chunk, Some(&bridge), Some(&session), None)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&r).unwrap();
        assert_eq!(v["result"]["ok"], json!(true));

        // bin/end with sha256
        let end = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "bin/end",
            "params": { "bin_id": bin_id, "sha256": sha }
        })
        .to_string();
        let r = dispatch_jsonrpc(&end, Some(&bridge), Some(&session), None)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&r).unwrap();
        assert_eq!(v["result"]["ok"], json!(true));
        assert_eq!(v["result"]["size"].as_u64().unwrap(), payload.len() as u64);

        // The blob should be in the finalized map awaiting a message/send.
        // message/send referencing the bin_id via metadata.
        let msg = json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": "m-1",
                    "role": "ROLE_USER",
                    "parts": [
                        { "text": "describe this" },
                        {
                            "filename": "image.png",
                            "mediaType": "image/png",
                            "metadata": { "bin_id": bin_id }
                        }
                    ]
                }
            }
        })
        .to_string();
        let reply = dispatch_jsonrpc(&msg, Some(&bridge), Some(&session), None)
            .await
            .unwrap();
        let reply_v: Value = serde_json::from_str(&reply).unwrap();
        assert!(
            reply_v.get("error").is_none(),
            "message/send should succeed: {reply_v}"
        );

        // The bin_id should now be consumed (one-shot). A second message/send
        // with the same bin_id should leave the part untouched (no `raw`),
        // because the blob has been taken.
        assert!(
            session.binaries.take(bin_id).await.is_none(),
            "blob must be consumed by message/send"
        );
    }

    #[tokio::test]
    async fn bin_chunk_unknown_bin_id_yields_neg_32001() {
        let session = Arc::new(SessionState::new("sess".to_string()));
        let chunk = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "bin/chunk",
            "params": { "bin_id": "ghost", "seq": 0, "data": "AAA=" }
        })
        .to_string();
        let r = dispatch_jsonrpc(&chunk, None, Some(&session), None)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&r).unwrap();
        assert_eq!(v["error"]["code"].as_i64().unwrap(), -32001);
    }

    #[tokio::test]
    async fn bin_chunk_out_of_order_yields_neg_32002() {
        let session = Arc::new(SessionState::new("sess".to_string()));
        let begin = json!({
            "jsonrpc":"2.0","id":1,"method":"bin/begin",
            "params": { "bin_id": "ooo", "total_size": 100, "total_chunks": 2 }
        })
        .to_string();
        dispatch_jsonrpc(&begin, None, Some(&session), None)
            .await
            .unwrap();
        let chunk = json!({
            "jsonrpc":"2.0","id":2,"method":"bin/chunk",
            "params": { "bin_id":"ooo", "seq":1, "data":"AAA=" }
        })
        .to_string();
        let r = dispatch_jsonrpc(&chunk, None, Some(&session), None)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&r).unwrap();
        assert_eq!(v["error"]["code"].as_i64().unwrap(), -32002);
    }

    #[tokio::test]
    async fn bin_end_sha256_mismatch_yields_neg_32003() {
        let session = Arc::new(SessionState::new("sess".to_string()));
        let begin = json!({
            "jsonrpc":"2.0","id":1,"method":"bin/begin",
            "params": { "bin_id": "h", "total_size": 4, "total_chunks": 1 }
        })
        .to_string();
        dispatch_jsonrpc(&begin, None, Some(&session), None)
            .await
            .unwrap();
        let chunk = json!({
            "jsonrpc":"2.0","id":2,"method":"bin/chunk",
            "params": { "bin_id":"h", "seq":0, "data": B64.encode(b"abcd") }
        })
        .to_string();
        dispatch_jsonrpc(&chunk, None, Some(&session), None)
            .await
            .unwrap();
        let end = json!({
            "jsonrpc":"2.0","id":3,"method":"bin/end",
            "params": { "bin_id":"h", "sha256": "00".repeat(32) }
        })
        .to_string();
        let r = dispatch_jsonrpc(&end, None, Some(&session), None)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&r).unwrap();
        assert_eq!(v["error"]["code"].as_i64().unwrap(), -32003);
    }

    /// `rewrite_message_send_with_bins` should leave non-`message/send`
    /// frames untouched (returns `None`) and should consume only the
    /// matching bin_id parts.
    #[tokio::test]
    async fn rewrite_returns_none_for_unrelated_methods() {
        let session = SessionState::new("sess".to_string());
        let v: Value =
            serde_json::from_str(r#"{"jsonrpc":"2.0","id":1,"method":"system/ping","params":{}}"#)
                .unwrap();
        let out = rewrite_message_send_with_bins(v, &session).await;
        assert!(out.is_none(), "non message/send frames must pass through");
    }

    #[tokio::test]
    async fn rewrite_resolves_bin_ref_to_raw_part() {
        let session = SessionState::new("sess".to_string());
        // Stage a finalized blob directly via the binary store.
        session
            .binaries
            .handle_begin(crate::binary::BinBeginParams {
                bin_id: "rb".to_string(),
                content_type: Some("image/jpeg".to_string()),
                total_size: 4,
                total_chunks: 1,
            })
            .await
            .unwrap();
        session
            .binaries
            .handle_chunk(crate::binary::BinChunkParams {
                bin_id: "rb".to_string(),
                seq: 0,
                data: B64.encode(b"abcd"),
            })
            .await
            .unwrap();
        session
            .binaries
            .handle_end(crate::binary::BinEndParams {
                bin_id: "rb".to_string(),
                sha256: None,
            })
            .await
            .unwrap();

        let frame = json!({
            "jsonrpc":"2.0","id":1,"method":"message/send",
            "params":{"message":{
                "messageId":"m","role":"ROLE_USER",
                "parts":[
                    {"text":"hi"},
                    {"filename":"a.jpg","metadata":{"bin_id":"rb"}}
                ]
            }}
        });
        let out = rewrite_message_send_with_bins(frame, &session)
            .await
            .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let parts = v["params"]["message"]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], json!("hi"));
        assert_eq!(parts[1]["raw"].as_str().unwrap(), B64.encode(b"abcd"));
        assert_eq!(parts[1]["mediaType"], json!("image/jpeg"));
        // bin_id metadata should be stripped after consumption.
        assert!(
            parts[1].get("metadata").is_none()
                || !parts[1]["metadata"]
                    .as_object()
                    .unwrap()
                    .contains_key("bin_id")
        );
    }
}
