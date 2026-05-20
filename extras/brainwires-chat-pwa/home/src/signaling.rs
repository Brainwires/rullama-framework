//! Signaling endpoints — `/signal/*` HTTP for SDP offer/answer + ICE
//! long-poll, and `/.well-known/agent-card.json`. Wired up in M2.
//!
//! State is **in-memory only** (an `Arc<DashMap<String, Arc<SessionState>>>`).
//! The home daemon serves a single user; if it restarts, the PWA re-pairs and
//! mints a new session. M3 wires the offer/answer that flows through these
//! routes into a real `RTCPeerConnection`.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::{HeaderName, HeaderValue, Method, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use brainwires_a2a::{
    A2A_PROTOCOL_VERSION, AgentCapabilities, AgentCard, AgentInterface, AgentProvider,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, RwLock, broadcast};
use tower_http::cors::{AllowOrigin, CorsLayer};
use uuid::Uuid;
use webrtc::data_channel::DataChannel as WrtcDataChannel;
use webrtc::peer_connection::{PeerConnection, RTCIceCandidateInit};

use crate::TurnConfig;
use crate::a2a::A2aBridge;
use crate::binary::BinaryStore;
use crate::turn::{IceServerJson, mint_ice_servers};
use crate::webrtc::{
    PeerEvent, apply_offer_and_create_answer, build_answerer, run_a2a_loop_with_session,
};

/// Maximum number of reply frames retained in the per-session outbox ring
/// buffer. Keeps a generous-but-bounded backlog for resume after a brief
/// network blip; not a durable queue (M10).
pub const OUTBOX_CAPACITY: usize = 64;

/// One entry in the per-session outbox.
///
/// `id` is the JSON-RPC reply id (mirrored from the originating request).
/// We keep it as `i64` so resume cursors compare cleanly even when the PWA
/// has only seen one of many in-flight requests. Frames are stored as the
/// already-serialized JSON text — no need to re-serialize on replay.
#[derive(Debug, Clone)]
pub struct OutboxEntry {
    pub id: i64,
    pub frame: String,
}

/// Default long-poll wait. The PWA retries on 204.
pub const DEFAULT_LONG_POLL: Duration = Duration::from_secs(25);

/// Default session TTL. Sessions are GC'd after this much idle time.
pub const DEFAULT_SESSION_TTL: Duration = Duration::from_secs(30 * 60);

/// How often the GC sweep runs.
pub const GC_INTERVAL: Duration = Duration::from_secs(60);

/// Crate version, baked at compile time. Surfaced in the AgentCard.
pub const HOME_VERSION: &str = env!("CARGO_PKG_VERSION");

/// `Access-Control-Max-Age` value advertised by the CORS layer (10 min).
pub const CORS_MAX_AGE: Duration = Duration::from_secs(600);

/// Build the `reqwest::Client` the daemon uses for outbound HTTPS (today
/// only Cloudflare TURN minting). 10 s connect/total timeout — TURN
/// minting on the request path; we'd rather fall back to STUN-only than
/// hold a `POST /signal/session` open for tens of seconds.
pub fn default_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(10))
        .build()
        .expect("default reqwest::Client builds")
}

/// Default allow-list when neither `--cors-origin` nor `--cors-permissive`
/// is supplied. Matches the chat-PWA dev container (`HOST_PORT` defaults to
/// 8080 in `extras/brainwires-chat-pwa/docker-compose.yml`) plus the
/// esbuild-serve fallback on 5173, on both `localhost` and `127.0.0.1`.
pub const DEFAULT_DEV_ORIGINS: &[&str] = &[
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
];

// ---------- wire types ----------

/// SDP description sent over signaling. Mirrors the shape `RTCPeerConnection`
/// produces in the browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdpDesc {
    pub sdp: String,
    /// `"offer"` or `"answer"` — kept as a string so we don't tightly couple
    /// to a Rust enum the browser doesn't share.
    #[serde(rename = "type")]
    pub kind: String,
}

/// One ICE candidate relayed over signaling. `candidate == null` is the
/// end-of-candidates marker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IceCandidate {
    pub candidate: Option<String>,
    #[serde(rename = "sdpMid", default)]
    pub sdp_mid: Option<String>,
    #[serde(rename = "sdpMLineIndex", default)]
    pub sdp_m_line_index: Option<u16>,
}

#[derive(Debug, Serialize)]
struct SessionCreatedResponse {
    session_id: String,
    ice_servers: Vec<IceServerJson>,
}

#[derive(Debug, Serialize)]
struct IcePollResponse {
    candidates: Vec<IceCandidate>,
    cursor: usize,
}

#[derive(Debug, Deserialize)]
pub struct IceQuery {
    #[serde(default)]
    pub since: usize,
}

// ---------- session state ----------

/// Per-session in-memory state.
///
/// `peer` and `data_channel` are populated by [`post_offer`] once the home
/// daemon accepts an SDP offer and runs the answerer handshake. The
/// `ice_candidates` buffer is the **home → PWA** direction (the PWA
/// long-polls it). Inbound PWA → home candidates go straight into
/// `peer.add_ice_candidate(...)` — see [`post_ice`] — and are counted in
/// `peer_input_count` for diagnostics only.
pub struct SessionState {
    pub session_id: String,
    pub created_at: Instant,
    pub offer: RwLock<Option<SdpDesc>>,
    pub answer: RwLock<Option<SdpDesc>>,
    pub ice_candidates: RwLock<Vec<IceCandidate>>,
    pub answer_notify: Notify,
    pub ice_notify: Notify,
    pub peer: RwLock<Option<Arc<dyn PeerConnection>>>,
    pub data_channel: RwLock<Option<Arc<dyn WrtcDataChannel>>>,
    /// Count of PWA-originated ICE candidates we've forwarded into the peer.
    /// Logging only — the wire protocol doesn't expose this.
    pub peer_input_count: AtomicUsize,
    /// M10 — bounded outbox of reply frames the daemon has emitted on the
    /// data channel. The PWA's `system/resume` cursors against this on
    /// reconnect. Capped at [`OUTBOX_CAPACITY`]; oldest entries roll off.
    pub outbox: RwLock<VecDeque<OutboxEntry>>,
    /// M11 — per-session binary chunking store. `bin/begin` allocates a
    /// pending buffer; `bin/end` finalizes; a subsequent `message/send`
    /// with a `bin_id` part consumes the blob. See [`crate::binary`].
    pub binaries: BinaryStore,
}

impl SessionState {
    /// Visible to the rest of the crate so the M11 webrtc tests can stage
    /// a session by hand without going through the signaling routes.
    pub(crate) fn new(session_id: String) -> Self {
        Self {
            session_id,
            created_at: Instant::now(),
            offer: RwLock::new(None),
            answer: RwLock::new(None),
            ice_candidates: RwLock::new(Vec::new()),
            answer_notify: Notify::new(),
            ice_notify: Notify::new(),
            peer: RwLock::new(None),
            data_channel: RwLock::new(None),
            peer_input_count: AtomicUsize::new(0),
            outbox: RwLock::new(VecDeque::with_capacity(OUTBOX_CAPACITY)),
            binaries: BinaryStore::new(),
        }
    }

    /// Push a reply frame onto the outbox. Drops the oldest entry when at
    /// capacity. Frames are expected in monotonic-id order (the dispatcher
    /// minted ids are JSON-RPC request ids that the PWA allocated
    /// monotonically — see `home-transport.js::JsonRpcDispatcher`).
    pub async fn push_outbox(&self, id: i64, frame: String) {
        let mut q = self.outbox.write().await;
        if q.len() >= OUTBOX_CAPACITY {
            q.pop_front();
        }
        q.push_back(OutboxEntry { id, frame });
    }

    /// Build the `system/resume` reply payload for a given cursor.
    ///
    /// Returns `(replayed_frames, dropped)`:
    ///   - `replayed_frames` = every outbox entry with `id > last_seen_id`,
    ///     in original order.
    ///   - `dropped = true` when the cursor predates the oldest retained
    ///     frame (i.e. the PWA missed at least one frame that's now gone).
    ///     The PWA treats this as a hard reset and re-issues anything
    ///     in-flight.
    pub async fn resume_from(&self, last_seen_id: i64) -> (Vec<String>, bool) {
        let q = self.outbox.read().await;
        let oldest_id = q.front().map(|e| e.id);
        let dropped = match oldest_id {
            // If the oldest retained id is greater than `last_seen_id + 1`
            // we've lost frames in between. Strict greater-than because the
            // cursor is exclusive (we're returning frames with id > cursor).
            Some(o) => o > last_seen_id.saturating_add(1),
            None => false,
        };
        let frames = q
            .iter()
            .filter(|e| e.id > last_seen_id)
            .map(|e| e.frame.clone())
            .collect();
        (frames, dropped)
    }
}

/// Shared application state passed to every handler via `State<AppState>`.
///
/// `bridge` is the A2A → agent dispatcher used by the WebRTC data-channel
/// loop (M4). It's optional so tests that don't exercise the agent path
/// can construct an [`AppState`] without spinning up a `ChatAgent`. The
/// data-channel loop falls back to the M3 ping/pong-only echo when the
/// bridge is `None`.
#[derive(Clone)]
pub struct AppState {
    pub sessions: Arc<DashMap<String, Arc<SessionState>>>,
    pub long_poll_timeout: Duration,
    pub session_ttl: Duration,
    pub bridge: Option<Arc<A2aBridge>>,
    /// TURN minting config. Default = STUN-only fallback.
    pub turn: TurnConfig,
    /// Shared `reqwest::Client` reused across TURN mint calls. Built once
    /// at server start; never per-request. Held in an `Arc` so cloning
    /// `AppState` (axum does this on every request) is cheap.
    pub http: Arc<reqwest::Client>,
    /// Sync store for cross-device data synchronization. When set, the
    /// daemon acts as a store-and-forward hub for `sync/*` JSON-RPC methods.
    pub sync_store: Option<Arc<crate::sync::SyncStore>>,
}

impl AppState {
    pub fn new(long_poll_timeout: Duration, session_ttl: Duration) -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            long_poll_timeout,
            session_ttl,
            bridge: None,
            turn: TurnConfig::default(),
            http: Arc::new(default_http_client()),
            sync_store: None,
        }
    }

    /// Attach an [`A2aBridge`]. Panics if one is already set — the daemon
    /// builder calls this exactly once.
    pub fn with_bridge(mut self, bridge: Arc<A2aBridge>) -> Self {
        assert!(self.bridge.is_none(), "A2aBridge already set on AppState");
        self.bridge = Some(bridge);
        self
    }

    /// Replace the TURN config. The daemon builder calls this once.
    pub fn with_turn(mut self, turn: TurnConfig) -> Self {
        self.turn = turn;
        self
    }

    /// Attach a [`crate::sync::SyncStore`] for cross-device sync.
    pub fn with_sync(mut self, sync: Arc<crate::sync::SyncStore>) -> Self {
        self.sync_store = Some(sync);
        self
    }

    /// Drop sessions older than `session_ttl`.
    pub fn gc_expired(&self) {
        let now = Instant::now();
        let ttl = self.session_ttl;
        self.sessions
            .retain(|_, s| now.saturating_duration_since(s.created_at) < ttl);
    }

    /// Spawn a background task that GCs every [`GC_INTERVAL`]. M11 piggybacks
    /// on this tick to also sweep each surviving session's [`BinaryStore`]
    /// for expired pending buffers (30 s) and finalized blobs (5 min).
    pub fn spawn_gc(&self) -> tokio::task::JoinHandle<()> {
        let me = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(GC_INTERVAL);
            // First tick fires immediately; skip it.
            tick.tick().await;
            loop {
                tick.tick().await;
                me.gc_expired();
                // Snapshot the surviving sessions then sweep each one's
                // BinaryStore. The snapshot avoids holding the dashmap
                // shard locks across the .await.
                let snapshot: Vec<Arc<SessionState>> =
                    me.sessions.iter().map(|e| e.value().clone()).collect();
                for s in snapshot {
                    s.binaries.gc_expired().await;
                }
            }
        })
    }
}

// ---------- CORS ----------

/// CORS policy applied to the axum router.
///
/// The chat PWA is served from a different origin than the home daemon
/// (PWA on the tunnel hostname or `localhost:8080` dev; daemon on
/// `127.0.0.1:7878`), so the browser issues preflight `OPTIONS` requests
/// against `/signal/*` + `/.well-known/agent-card.json`. Without an
/// `Access-Control-Allow-Origin` matching the PWA origin, the browser
/// blocks the actual call.
///
/// Three modes:
///   1. **Default** ([`CorsConfig::default`]): allow `DEFAULT_DEV_ORIGINS`
///      (chat-PWA's dev container + esbuild-serve loopback origins).
///   2. **Allow-list** ([`CorsConfig::allow_origin`], repeatable): exact-
///      match origin strings — typical production wiring is one PWA URL.
///   3. **Permissive** ([`CorsConfig::permissive`]): allow any origin.
///      Dev only — wide-open CORS in production is a footgun.
///
/// No credentials mode is enabled — the PWA carries Bearer tokens (M8),
/// not cookies, so `Access-Control-Allow-Credentials` is intentionally off.
#[derive(Debug, Clone)]
pub struct CorsConfig {
    origins: Vec<String>,
    permissive: bool,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            origins: DEFAULT_DEV_ORIGINS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            permissive: false,
        }
    }
}

impl CorsConfig {
    /// Append an origin to the allow-list. Replaces the
    /// [`DEFAULT_DEV_ORIGINS`] defaults on first call — explicit callers
    /// opt in to exactly the origins they list, no surprise loopbacks.
    pub fn allow_origin(mut self, origin: impl Into<String>) -> Self {
        if self
            .origins
            .iter()
            .any(|o| DEFAULT_DEV_ORIGINS.contains(&o.as_str()))
        {
            // Wipe the dev defaults the first time the caller adds an
            // explicit origin. Otherwise they'd unknowingly inherit
            // `http://localhost:8080` in production.
            self.origins.clear();
        }
        self.origins.push(origin.into());
        self
    }

    /// Allow any origin (wide-open). **Dev only.** Use this when the
    /// daemon and PWA are both on the same dev host but you don't want
    /// to enumerate ports.
    pub fn permissive(mut self) -> Self {
        self.permissive = true;
        self
    }

    /// Build the [`CorsLayer`] reflecting this config.
    pub fn into_layer(self) -> CorsLayer {
        let methods = [Method::GET, Method::POST, Method::DELETE, Method::OPTIONS];
        // Headers the PWA + the M8 pairing flow need to send. Listed
        // explicitly rather than `mirror_request` because some browsers
        // are stricter when credentials are involved.
        let headers = [
            HeaderName::from_static("content-type"),
            HeaderName::from_static("authorization"),
            HeaderName::from_static("cf-access-client-id"),
            HeaderName::from_static("cf-access-client-secret"),
        ];
        let layer = CorsLayer::new()
            .allow_methods(methods)
            .allow_headers(headers)
            .max_age(CORS_MAX_AGE);
        if self.permissive {
            layer.allow_origin(AllowOrigin::any())
        } else {
            let values: Vec<HeaderValue> = self
                .origins
                .iter()
                .filter_map(|o| HeaderValue::from_str(o).ok())
                .collect();
            layer.allow_origin(AllowOrigin::list(values))
        }
    }
}

// ---------- router ----------

/// Build the axum `Router` for all `/signal/*` + agent-card routes.
///
/// This wires the default CORS policy (chat-PWA dev origins). For
/// production, callers should go through [`router_with_cors`] with an
/// explicit allow-list — or via [`crate::HomeServerBuilder`] which
/// exposes the same knobs as CLI flags.
pub fn router(state: AppState) -> Router {
    router_with_cors(state, CorsConfig::default())
}

/// Build the axum `Router` with an explicit CORS configuration.
pub fn router_with_cors(state: AppState, cors: CorsConfig) -> Router {
    Router::new()
        .route("/signal/session", post(create_session))
        .route("/signal/offer/{session}", post(post_offer))
        .route("/signal/answer/{session}", get(get_answer))
        .route("/signal/ice/{session}", post(post_ice).get(get_ice))
        .route("/signal/{session}", axum::routing::delete(delete_session))
        .route("/.well-known/agent-card.json", get(agent_card))
        .with_state(state)
        .layer(cors.into_layer())
}

// ---------- handlers ----------

async fn create_session(State(state): State<AppState>) -> impl IntoResponse {
    let session_id = Uuid::new_v4().simple().to_string();
    let s = Arc::new(SessionState::new(session_id.clone()));
    state.sessions.insert(session_id.clone(), s);
    // Mint per-session ICE servers. With Cloudflare TURN configured this
    // hits the Calls API; otherwise it returns the two STUN fallbacks
    // synchronously (no I/O). On TURN API failure we fall back to STUN-only
    // — connection-establishment must not hard-depend on the TURN API.
    let ice_servers = mint_ice_servers(&state.turn, state.http.as_ref()).await;
    (
        StatusCode::OK,
        Json(SessionCreatedResponse {
            session_id,
            ice_servers,
        }),
    )
}

async fn post_offer(
    State(state): State<AppState>,
    Path(session): Path<String>,
    Json(body): Json<SdpDesc>,
) -> StatusCode {
    let Some(s) = state.sessions.get(&session).map(|e| e.value().clone()) else {
        return StatusCode::NOT_FOUND;
    };

    // M10 — renegotiation path. If a peer already exists on this session
    // (the PWA initiated an ICE restart), reuse it: just apply the new
    // remote offer, mint a fresh answer, replace the stored answer, and
    // wake any /signal/answer long-pollers. webrtc-rs handles the ICE
    // ufrag/pwd swap transparently when set_remote_description is called
    // with an offer that carries `a=ice-ufrag` / `a=ice-pwd` lines
    // different from the prior negotiation.
    if let Some(existing_pc) = s.peer.read().await.clone() {
        use webrtc::peer_connection::RTCSessionDescription;
        let new_offer = match RTCSessionDescription::offer(body.sdp.clone()) {
            Ok(o) => o,
            Err(e) => {
                tracing::warn!(session = %session, error = %e, "renegotiation: parse offer");
                return StatusCode::BAD_REQUEST;
            }
        };
        if let Err(e) = existing_pc.set_remote_description(new_offer).await {
            tracing::warn!(session = %session, error = %e, "renegotiation: set_remote_description(offer)");
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
        let answer = match existing_pc.create_answer(None).await {
            Ok(a) => a,
            Err(e) => {
                tracing::warn!(session = %session, error = %e, "renegotiation: create_answer");
                return StatusCode::INTERNAL_SERVER_ERROR;
            }
        };
        let answer_sdp = answer.sdp.clone();
        if let Err(e) = existing_pc.set_local_description(answer).await {
            tracing::warn!(session = %session, error = %e, "renegotiation: set_local_description(answer)");
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
        *s.offer.write().await = Some(body);
        *s.answer.write().await = Some(SdpDesc {
            sdp: answer_sdp,
            kind: "answer".to_string(),
        });
        s.answer_notify.notify_waiters();
        return StatusCode::NO_CONTENT;
    }

    // Stash the offer for diagnostic / replay purposes.
    *s.offer.write().await = Some(body.clone());

    // Build the answerer peer. We deliberately *don't* await ICE-gathering
    // completion before responding — candidates trickle to the PWA via
    // `/signal/ice/{session}` instead.
    let peer = match build_answerer().await {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(session = %session, error = %e, "build_answerer failed");
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
    };

    // Subscribe to the peer's event broadcaster BEFORE applying the offer so
    // we don't miss any candidates that surface synchronously inside
    // `set_local_description`. Two subscribers: one for ICE relay (PWA
    // direction), one for the data-channel watcher (a2a ping/pong loop).
    let ice_rx = peer.subscribe();
    let session_state = s.clone();
    tokio::spawn(relay_local_ice(ice_rx, session_state));

    // Stash the underlying PeerConnection on the session so `post_ice`
    // (PWA → home direction) can feed it remote candidates.
    *s.peer.write().await = Some(peer.pc.clone());

    // Drive the offer/answer dance. After this, ICE gathering is in flight
    // and host candidates are being broadcast on `peer`'s event bus.
    let answer_sdp = match apply_offer_and_create_answer(&peer, body.sdp).await {
        Ok(a) => a,
        Err(e) => {
            tracing::error!(session = %session, error = %e, "apply_offer_and_create_answer failed");
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
    };

    // Now move the HomePeer wrapper into a background task that waits for
    // the PWA-created data channel to arrive and runs the JSON-RPC ping/pong
    // loop. `s.peer` already holds an Arc on the underlying PeerConnection
    // (so the event handler — which owns the broadcast `tx` — stays alive
    // for the session lifetime), but the HomePeer wrapper itself is what
    // exposes `subscribe()`, so we keep ownership of it inside the task.
    let session_state = s.clone();
    let bridge_for_loop = state.bridge.clone();
    let sync_for_loop = state.sync_store.clone();
    let session_for_loop = s.clone();
    tokio::spawn(async move {
        match run_a2a_loop_with_session(
            &peer,
            bridge_for_loop,
            Some(session_for_loop),
            sync_for_loop,
        )
        .await
        {
            Ok(dc) => {
                *session_state.data_channel.write().await = Some(dc);
            }
            Err(e) => {
                tracing::warn!(error = %e, "a2a loop exited before data channel arrived");
            }
        }
        // Drop `peer` here; the underlying PeerConnection is still kept alive
        // by `session_state.peer` until the session is GC'd or DELETE'd.
        drop(peer);
    });

    // Publish the answer + wake any /signal/answer long-pollers.
    *s.answer.write().await = Some(SdpDesc {
        sdp: answer_sdp,
        kind: "answer".to_string(),
    });
    s.answer_notify.notify_waiters();
    StatusCode::NO_CONTENT
}

/// Background task: forward local ICE candidates into the session's outbound
/// buffer and wake any long-pollers.
async fn relay_local_ice(mut rx: broadcast::Receiver<PeerEvent>, s: Arc<SessionState>) {
    loop {
        match rx.recv().await {
            Ok(PeerEvent::LocalIceCandidate {
                candidate,
                sdp_mid,
                sdp_mline_index,
            }) => {
                s.ice_candidates.write().await.push(IceCandidate {
                    candidate: Some(candidate),
                    sdp_mid,
                    sdp_m_line_index: sdp_mline_index,
                });
                s.ice_notify.notify_waiters();
            }
            Ok(_) => continue,
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }
}

async fn get_answer(
    State(state): State<AppState>,
    Path(session): Path<String>,
) -> Result<axum::response::Response, StatusCode> {
    let Some(s) = state.sessions.get(&session).map(|e| e.value().clone()) else {
        return Err(StatusCode::NOT_FOUND);
    };

    // Fast path: answer is already there.
    if let Some(a) = s.answer.read().await.clone() {
        return Ok(Json(a).into_response());
    }

    // Long-poll: wait for a notify or the deadline.
    let notified = s.answer_notify.notified();
    tokio::pin!(notified);
    let outcome = tokio::time::timeout(state.long_poll_timeout, &mut notified).await;
    match outcome {
        Ok(()) => {
            if let Some(a) = s.answer.read().await.clone() {
                Ok(Json(a).into_response())
            } else {
                Ok(StatusCode::NO_CONTENT.into_response())
            }
        }
        Err(_) => Ok(StatusCode::NO_CONTENT.into_response()),
    }
}

async fn post_ice(
    State(state): State<AppState>,
    Path(session): Path<String>,
    Json(body): Json<IceCandidate>,
) -> StatusCode {
    let Some(s) = state.sessions.get(&session).map(|e| e.value().clone()) else {
        return StatusCode::NOT_FOUND;
    };

    // PWA → home direction: feed the candidate into the answerer peer if it
    // has been built (i.e. `post_offer` has already run). End-of-candidates
    // markers (candidate == null) are dropped here — webrtc-rs treats an
    // empty `RTCIceCandidateInit.candidate` string as malformed, and the
    // home side doesn't need a sentinel because trickle-ICE keeps gathering
    // until the connection is established or fails.
    if let Some(s_str) = body.candidate.as_deref()
        && !s_str.is_empty()
    {
        if let Some(pc) = s.peer.read().await.clone() {
            if let Err(e) = pc
                .add_ice_candidate(RTCIceCandidateInit {
                    candidate: s_str.to_string(),
                    sdp_mid: body.sdp_mid.clone(),
                    sdp_mline_index: body.sdp_m_line_index,
                    username_fragment: None,
                    url: None,
                })
                .await
            {
                tracing::warn!(session = %session, error = %e, "add_ice_candidate failed");
            } else {
                s.peer_input_count.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            tracing::debug!(
                session = %session,
                "received PWA ICE candidate before offer; buffering would race the answerer build"
            );
        }
    }

    StatusCode::NO_CONTENT
}

async fn get_ice(
    State(state): State<AppState>,
    Path(session): Path<String>,
    Query(q): Query<IceQuery>,
) -> Result<axum::response::Response, StatusCode> {
    let Some(s) = state.sessions.get(&session).map(|e| e.value().clone()) else {
        return Err(StatusCode::NOT_FOUND);
    };

    // Snapshot fast path.
    {
        let cands = s.ice_candidates.read().await;
        if cands.len() > q.since {
            let slice = cands[q.since..].to_vec();
            let cursor = cands.len();
            return Ok(Json(IcePollResponse {
                candidates: slice,
                cursor,
            })
            .into_response());
        }
    }

    // Long-poll for a new candidate or deadline.
    let notified = s.ice_notify.notified();
    tokio::pin!(notified);
    let _ = tokio::time::timeout(state.long_poll_timeout, &mut notified).await;

    let cands = s.ice_candidates.read().await;
    let from = q.since.min(cands.len());
    let slice = cands[from..].to_vec();
    let cursor = cands.len();
    Ok(Json(IcePollResponse {
        candidates: slice,
        cursor,
    })
    .into_response())
}

async fn delete_session(State(state): State<AppState>, Path(session): Path<String>) -> StatusCode {
    state.sessions.remove(&session);
    StatusCode::NO_CONTENT
}

async fn agent_card() -> impl IntoResponse {
    let card = AgentCard {
        name: "brainwires-home".to_string(),
        description: "Brainwires dial-home daemon: WebRTC peer + A2A JSON-RPC \
                      bridge into the user's local TaskAgent."
            .to_string(),
        version: HOME_VERSION.to_string(),
        // The PWA overrides this with the actual tunnel hostname when it
        // fetches the card; the daemon itself doesn't know its public URL.
        // Per A2A 0.3, `supportedInterfaces[].url` is the canonical service URL.
        supported_interfaces: vec![AgentInterface {
            url: "/".to_string(),
            protocol_binding: "JSONRPC".to_string(),
            tenant: None,
            protocol_version: A2A_PROTOCOL_VERSION.to_string(),
        }],
        capabilities: AgentCapabilities {
            streaming: Some(true),
            push_notifications: Some(false),
            extended_agent_card: Some(false),
            extensions: None,
        },
        skills: Vec::new(),
        default_input_modes: vec!["text".to_string()],
        default_output_modes: vec!["text".to_string()],
        provider: Some(AgentProvider {
            url: "https://brainwires.net".to_string(),
            organization: "Brainwires".to_string(),
        }),
        security_schemes: None,
        security_requirements: None,
        documentation_url: None,
        icon_url: None,
        signatures: None,
    };
    Json(card)
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::{Method, Request};
    use serde_json::Value;
    use std::time::Duration;
    use tower::ServiceExt;

    fn test_state() -> AppState {
        AppState::new(Duration::from_millis(200), DEFAULT_SESSION_TTL)
    }

    async fn body_json(resp: axum::response::Response) -> Value {
        let body = resp.into_body();
        let bytes = to_bytes(body, 1 << 20).await.expect("collect body");
        serde_json::from_slice(&bytes).expect("body is valid JSON")
    }

    async fn body_bytes(resp: axum::response::Response) -> Vec<u8> {
        let body = resp.into_body();
        to_bytes(body, 1 << 20)
            .await
            .expect("collect body")
            .to_vec()
    }

    fn empty_request(method: Method, uri: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    }

    /// Convenience: open a fresh session and return its id.
    async fn new_session(app: &Router) -> String {
        let resp = app
            .clone()
            .oneshot(empty_request(Method::POST, "/signal/session"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        v["session_id"].as_str().unwrap().to_string()
    }

    #[tokio::test]
    async fn test_session_create_returns_id_and_default_ice_servers() {
        let app = router(test_state());
        let resp = app
            .oneshot(empty_request(Method::POST, "/signal/session"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        let id = v["session_id"].as_str().expect("session_id is string");
        // UUID v4 simple form is 32 hex chars.
        assert_eq!(id.len(), 32, "session_id should be 32 hex chars: {id}");
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));

        // M7: STUN-only fallback when TURN isn't configured. The default
        // builder doesn't set Cloudflare creds, so we expect exactly the two
        // public STUN entries.
        let ice = v["ice_servers"].as_array().expect("ice_servers is array");
        assert_eq!(
            ice.len(),
            2,
            "STUN-only default should yield 2 entries: {ice:?}"
        );
        let urls: Vec<&str> = ice.iter().filter_map(|s| s["urls"][0].as_str()).collect();
        assert!(
            urls.iter().any(|u| u.contains("stun.cloudflare.com")),
            "expected cloudflare STUN entry: {urls:?}"
        );
        assert!(
            urls.iter().any(|u| u.contains("stun.l.google.com")),
            "expected google STUN entry: {urls:?}"
        );
        // STUN entries must NOT carry credentials.
        for s in ice {
            assert!(
                s.get("username").is_none(),
                "STUN entry must not carry username"
            );
            assert!(
                s.get("credential").is_none(),
                "STUN entry must not carry credential"
            );
        }
    }

    /// Repurposed from the M2 stub-SDP test. Now that `post_offer` actually
    /// parses the SDP and runs an answerer, we can't use a fake string —
    /// instead we test the answer route's fast-path by poking the session
    /// state directly. The full handshake through `post_offer` is exercised
    /// by `test_offer_produces_answer` and `test_full_handshake_in_process`
    /// in the `m3_handshake` module.
    #[tokio::test]
    async fn test_answer_fast_path_returns_stored_sdp() {
        let state = test_state();
        let app = router(state.clone());
        let id = new_session(&app).await;

        // Pretend the home side has filled in an answer.
        let answer = SdpDesc {
            sdp: "v=0\r\n...answer".to_string(),
            kind: "answer".to_string(),
        };
        {
            let s = state.sessions.get(&id).unwrap().value().clone();
            *s.answer.write().await = Some(answer.clone());
            s.answer_notify.notify_waiters();
        }

        // GET /signal/answer/{id} fast-paths since the answer is set.
        let resp = app
            .clone()
            .oneshot(empty_request(Method::GET, &format!("/signal/answer/{id}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["sdp"].as_str().unwrap(), "v=0\r\n...answer");
        assert_eq!(v["type"].as_str().unwrap(), "answer");
    }

    #[tokio::test]
    async fn test_answer_long_poll_times_out() {
        let app = router(test_state());
        let id = new_session(&app).await;

        let start = Instant::now();
        let resp = app
            .clone()
            .oneshot(empty_request(Method::GET, &format!("/signal/answer/{id}")))
            .await
            .unwrap();
        let elapsed = start.elapsed();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        assert!(
            elapsed >= Duration::from_millis(150),
            "long-poll returned too fast: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "long-poll took too long: {elapsed:?}"
        );

        // Unknown session returns 404.
        let resp = app
            .oneshot(empty_request(
                Method::GET,
                "/signal/answer/00000000000000000000000000000000",
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    /// Repurposed from the M2 test that POST'd into the ICE buffer directly.
    /// In M3 the buffer is the **home → PWA** direction; PWA-originated
    /// candidates go straight into `peer.add_ice_candidate`. So we now poke
    /// the buffer the way `relay_local_ice` does and assert the GET endpoint
    /// surfaces them with cursor semantics.
    #[tokio::test]
    async fn test_ice_buffer_get_with_cursor() {
        let state = test_state();
        let app = router(state.clone());
        let id = new_session(&app).await;

        let s = state.sessions.get(&id).unwrap().value().clone();

        // Push two candidates into the home → PWA buffer the same way
        // `relay_local_ice` does when the answerer's peer surfaces them.
        let cand_a = IceCandidate {
            candidate: Some("candidate:1 1 UDP 2122260223 192.168.1.10 51234 typ host".to_string()),
            sdp_mid: Some("0".to_string()),
            sdp_m_line_index: Some(0),
        };
        let cand_b = IceCandidate {
            candidate: Some("candidate:2 1 UDP 2122194687 192.168.1.10 51235 typ host".to_string()),
            sdp_mid: Some("0".to_string()),
            sdp_m_line_index: Some(0),
        };

        s.ice_candidates.write().await.push(cand_a.clone());
        s.ice_notify.notify_waiters();

        // GET ?since=0 returns it.
        let resp = app
            .clone()
            .oneshot(empty_request(
                Method::GET,
                &format!("/signal/ice/{id}?since=0"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["cursor"].as_u64().unwrap(), 1);
        let cands = v["candidates"].as_array().unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(
            cands[0]["candidate"].as_str().unwrap(),
            cand_a.candidate.as_deref().unwrap()
        );

        s.ice_candidates.write().await.push(cand_b.clone());
        s.ice_notify.notify_waiters();

        // GET ?since=1 returns just the second.
        let resp = app
            .clone()
            .oneshot(empty_request(
                Method::GET,
                &format!("/signal/ice/{id}?since=1"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["cursor"].as_u64().unwrap(), 2);
        let cands = v["candidates"].as_array().unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(
            cands[0]["candidate"].as_str().unwrap(),
            cand_b.candidate.as_deref().unwrap()
        );
    }

    #[tokio::test]
    async fn test_delete_session_idempotent() {
        let app = router(test_state());
        let id = new_session(&app).await;

        for _ in 0..2 {
            let resp = app
                .clone()
                .oneshot(empty_request(Method::DELETE, &format!("/signal/{id}")))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::NO_CONTENT);
        }
    }

    #[tokio::test]
    async fn test_agent_card_returns_valid_json() {
        let app = router(test_state());
        let resp = app
            .oneshot(empty_request(Method::GET, "/.well-known/agent-card.json"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = body_bytes(resp).await;
        let v: Value = serde_json::from_slice(&bytes).expect("agent-card is valid JSON");
        assert_eq!(v["name"].as_str().unwrap(), "brainwires-home");
        // protocolVersion is per-supportedInterface in the A2A AgentCard struct.
        let iface = v["supportedInterfaces"]
            .as_array()
            .expect("supportedInterfaces array")
            .first()
            .expect("at least one interface");
        assert_eq!(iface["protocolVersion"].as_str().unwrap(), "0.3");
        assert_eq!(
            v["capabilities"]["streaming"].as_bool().unwrap(),
            true,
            "streaming capability"
        );
        // version field reflects the crate version.
        assert_eq!(v["version"].as_str().unwrap(), HOME_VERSION);
    }

    #[tokio::test]
    async fn test_gc_expires_old_sessions() {
        let mut state = test_state();
        state.session_ttl = Duration::from_millis(50);
        let app = router(state.clone());
        let id = new_session(&app).await;
        assert!(state.sessions.contains_key(&id));
        tokio::time::sleep(Duration::from_millis(80)).await;
        state.gc_expired();
        assert!(!state.sessions.contains_key(&id), "session should be GC'd");
    }

    // ---------- CORS preflight tests ----------
    //
    // These exercise the `CorsLayer` wiring through the real router rather
    // than testing tower-http internals. The goal is "verify CORS is
    // configured correctly" — a preflight from an allowed origin gets
    // ACAO back; a disallowed one doesn't.

    fn preflight_request(origin: &str) -> Request<Body> {
        Request::builder()
            .method(Method::OPTIONS)
            .uri("/signal/session")
            .header("origin", origin)
            .header("access-control-request-method", "POST")
            .header(
                "access-control-request-headers",
                "content-type,authorization",
            )
            .body(Body::empty())
            .unwrap()
    }

    #[tokio::test]
    async fn test_cors_preflight_allowed_origin() {
        let cors = CorsConfig::default().allow_origin("http://localhost:8080");
        let app = router_with_cors(test_state(), cors);
        let resp = app
            .oneshot(preflight_request("http://localhost:8080"))
            .await
            .unwrap();
        // tower-http emits 200 OK for accepted preflight.
        assert!(
            resp.status().is_success(),
            "preflight from allowed origin should succeed, got {}",
            resp.status()
        );
        let acao = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("ACAO header present on accepted preflight")
            .to_str()
            .unwrap();
        assert_eq!(acao, "http://localhost:8080");
    }

    #[tokio::test]
    async fn test_cors_preflight_disallowed_origin() {
        let cors = CorsConfig::default().allow_origin("http://localhost:8080");
        let app = router_with_cors(test_state(), cors);
        let resp = app
            .oneshot(preflight_request("https://evil.example.com"))
            .await
            .unwrap();
        // tower-http silently drops the ACAO header when the origin isn't
        // matched; the browser treats absence as block. Either no header
        // or a non-matching one is acceptable — the critical thing is that
        // the evil origin is NOT echoed back and `*` is NOT returned.
        match resp.headers().get("access-control-allow-origin") {
            None => {}
            Some(v) => {
                let s = v.to_str().unwrap_or("");
                assert_ne!(s, "*", "must not return wildcard ACAO");
                assert_ne!(
                    s, "https://evil.example.com",
                    "must not echo disallowed origin"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_cors_permissive_allows_any() {
        let cors = CorsConfig::default().permissive();
        let app = router_with_cors(test_state(), cors);
        let resp = app
            .oneshot(preflight_request("https://random.example.com"))
            .await
            .unwrap();
        assert!(
            resp.status().is_success(),
            "permissive preflight should succeed, got {}",
            resp.status()
        );
        // Permissive mode emits `*` (no credentials, so `*` is legal).
        let acao = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("ACAO header present in permissive mode")
            .to_str()
            .unwrap()
            .to_string();
        assert_eq!(acao, "*", "permissive mode must return wildcard");
    }

    #[tokio::test]
    async fn test_cors_default_localhost_origins() {
        // No CLI flags → DEFAULT_DEV_ORIGINS apply.
        let app = router_with_cors(test_state(), CorsConfig::default());

        // Allowed: chat-PWA dev container origin.
        let resp = app
            .clone()
            .oneshot(preflight_request("http://localhost:8080"))
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let acao = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("ACAO present for default-allowed origin")
            .to_str()
            .unwrap();
        assert_eq!(acao, "http://localhost:8080");

        // Disallowed: arbitrary public origin.
        let resp = app
            .oneshot(preflight_request("https://random.example.com"))
            .await
            .unwrap();
        match resp.headers().get("access-control-allow-origin") {
            None => {}
            Some(v) => {
                let s = v.to_str().unwrap_or("");
                assert_ne!(s, "*");
                assert_ne!(s, "https://random.example.com");
            }
        }
    }

    // ---------- M10 outbox / resume tests ----------

    #[tokio::test]
    async fn outbox_pushes_drop_oldest_at_capacity() {
        let s = SessionState::new("test".to_string());
        // Push 70 entries with monotonic ids; cap is 64.
        for i in 1..=70 {
            s.push_outbox(i, format!("frame-{i}")).await;
        }
        let q = s.outbox.read().await;
        assert_eq!(q.len(), OUTBOX_CAPACITY);
        // Oldest retained should be id=7 (70 - 64 + 1). Newest = 70.
        assert_eq!(q.front().unwrap().id, 7);
        assert_eq!(q.back().unwrap().id, 70);
    }

    #[tokio::test]
    async fn system_resume_returns_subset() {
        let s = SessionState::new("test".to_string());
        for i in 1..=5 {
            s.push_outbox(i, format!("frame-{i}")).await;
        }
        let (replayed, dropped) = s.resume_from(2).await;
        assert!(!dropped, "no entries lost; dropped should be false");
        let ids: Vec<&str> = replayed.iter().map(|f| f.as_str()).collect();
        assert_eq!(ids, vec!["frame-3", "frame-4", "frame-5"]);
    }

    #[tokio::test]
    async fn system_resume_dropped_flag_set() {
        let s = SessionState::new("test".to_string());
        for i in 1..=70 {
            s.push_outbox(i, format!("frame-{i}")).await;
        }
        // Cursor is older than the retained window (oldest id is 7).
        let (replayed, dropped) = s.resume_from(1).await;
        assert!(
            dropped,
            "cursor predates the outbox tail; dropped must be true"
        );
        // Should still return the entire retained window.
        assert_eq!(replayed.len(), OUTBOX_CAPACITY);
        assert_eq!(replayed.first().unwrap(), "frame-7");
        assert_eq!(replayed.last().unwrap(), "frame-70");
    }

    #[tokio::test]
    async fn system_resume_caught_up_returns_empty() {
        let s = SessionState::new("test".to_string());
        for i in 1..=5 {
            s.push_outbox(i, format!("frame-{i}")).await;
        }
        let (replayed, dropped) = s.resume_from(5).await;
        assert!(replayed.is_empty(), "cursor at tip; nothing to replay");
        assert!(!dropped);
    }

    #[tokio::test]
    async fn test_cors_allow_origin_replaces_dev_defaults() {
        // Sanity: once the caller adds an explicit origin, the dev
        // defaults must drop out — production daemons shouldn't silently
        // accept `http://localhost:8080` next to their real origin.
        let cors = CorsConfig::default().allow_origin("https://chat.example.com");
        let app = router_with_cors(test_state(), cors);

        let resp = app
            .clone()
            .oneshot(preflight_request("https://chat.example.com"))
            .await
            .unwrap();
        assert!(resp.status().is_success());

        let resp = app
            .oneshot(preflight_request("http://localhost:8080"))
            .await
            .unwrap();
        match resp.headers().get("access-control-allow-origin") {
            None => {}
            Some(v) => {
                let s = v.to_str().unwrap_or("");
                assert_ne!(
                    s, "http://localhost:8080",
                    "dev defaults should be cleared once the caller adds an explicit origin"
                );
            }
        }
    }
}

// ---------- M3 integration tests: real WebRTC handshake through the router ----------

#[cfg(test)]
mod m3_handshake {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::{Method, Request};
    use serde_json::Value;
    use std::time::Duration;
    use tokio::sync::mpsc;
    use tower::ServiceExt;
    use webrtc::data_channel::DataChannelEvent;

    use crate::webrtc::{A2A_CHANNEL_LABEL, build_peer, open_a2a_channel};

    fn handshake_state() -> AppState {
        // Short long-poll so timeouts don't drag the test out, but long
        // enough that a real ICE candidate arrives within one wait window.
        AppState::new(Duration::from_secs(2), DEFAULT_SESSION_TTL)
    }

    async fn body_json(resp: axum::response::Response) -> Value {
        let body = resp.into_body();
        let bytes = to_bytes(body, 1 << 20).await.expect("collect body");
        serde_json::from_slice(&bytes).expect("body is valid JSON")
    }

    fn json_request(method: Method, uri: &str, body: Value) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap()
    }

    fn empty_request(method: Method, uri: &str) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    }

    async fn new_session(app: &Router) -> String {
        let resp = app
            .clone()
            .oneshot(empty_request(Method::POST, "/signal/session"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        v["session_id"].as_str().unwrap().to_string()
    }

    /// `post_offer` accepts a real PWA-side SDP and the answer endpoint
    /// returns a non-empty SDP within a couple of seconds.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_offer_produces_answer() -> anyhow::Result<()> {
        let state = handshake_state();
        let app = router(state.clone());
        let id = new_session(&app).await;

        // Build a real PWA-side offerer with an `"a2a"` data channel, just
        // so the offer SDP is well-formed for the home parser.
        let pwa = build_peer(vec![]).await?;
        let _dc = open_a2a_channel(&pwa).await?;
        let offer = pwa
            .pc
            .create_offer(None)
            .await
            .map_err(|e| anyhow::anyhow!("create_offer: {e}"))?;
        pwa.pc
            .set_local_description(offer.clone())
            .await
            .map_err(|e| anyhow::anyhow!("set_local_description: {e}"))?;

        // POST the offer.
        let resp = app
            .clone()
            .oneshot(json_request(
                Method::POST,
                &format!("/signal/offer/{id}"),
                serde_json::json!({ "sdp": offer.sdp, "type": "offer" }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        // GET the answer; should fast-path because post_offer fills it
        // synchronously before returning.
        let resp = tokio::time::timeout(
            Duration::from_secs(5),
            app.clone()
                .oneshot(empty_request(Method::GET, &format!("/signal/answer/{id}"))),
        )
        .await
        .map_err(|_| anyhow::anyhow!("answer GET timed out"))?
        .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(v["type"].as_str().unwrap(), "answer");
        let sdp = v["sdp"].as_str().unwrap();
        assert!(!sdp.is_empty(), "answer SDP should be non-empty");
        assert!(
            sdp.starts_with("v=0"),
            "answer SDP should start with v=0: {sdp}"
        );

        // Cleanup.
        let _ = pwa.pc.close().await;
        let _ = app
            .oneshot(empty_request(Method::DELETE, &format!("/signal/{id}")))
            .await;
        Ok(())
    }

    /// Full end-to-end handshake. PWA-side offerer drives the real axum
    /// router via `tower::ServiceExt::oneshot` for every signaling call.
    /// Once the data channel opens, sends a JSON-RPC `ping` and asserts the
    /// home daemon echoes back a `pong`-shaped reply on the same channel.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_full_handshake_in_process() -> anyhow::Result<()> {
        // Long-poll long enough for ICE on a cold-start webrtc-rs instance.
        let state = AppState::new(Duration::from_secs(3), DEFAULT_SESSION_TTL);
        let app = router(state.clone());

        // Step 1: create the session.
        let id = new_session(&app).await;

        // Step 2: PWA builds a peer + opens the canonical data channel.
        let pwa = build_peer(vec![]).await?;
        let dc = open_a2a_channel(&pwa).await?;
        assert_eq!(dc.label().await.unwrap_or_default(), A2A_CHANNEL_LABEL);

        // Step 3: forward PWA local ICE candidates to the home side via
        // `POST /signal/ice/{id}`.
        let mut pwa_events = pwa.subscribe();
        let app_for_ice = app.clone();
        let id_for_ice = id.clone();
        let pwa_ice_relay = tokio::spawn(async move {
            loop {
                match pwa_events.recv().await {
                    Ok(crate::webrtc::PeerEvent::LocalIceCandidate {
                        candidate,
                        sdp_mid,
                        sdp_mline_index,
                    }) => {
                        let body = serde_json::json!({
                            "candidate": candidate,
                            "sdpMid": sdp_mid,
                            "sdpMLineIndex": sdp_mline_index,
                        });
                        let _ = app_for_ice
                            .clone()
                            .oneshot(json_request(
                                Method::POST,
                                &format!("/signal/ice/{id_for_ice}"),
                                body,
                            ))
                            .await;
                    }
                    Ok(crate::webrtc::PeerEvent::ConnectionState(s))
                        if matches!(
                            s,
                            webrtc::peer_connection::RTCPeerConnectionState::Connected
                                | webrtc::peer_connection::RTCPeerConnectionState::Failed
                                | webrtc::peer_connection::RTCPeerConnectionState::Closed
                        ) =>
                    {
                        break;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    _ => continue,
                }
            }
        });

        // Step 4: pump home → PWA ICE candidates by long-polling
        // `/signal/ice/{id}` and feeding them into `pwa.pc.add_ice_candidate`.
        let app_for_pull = app.clone();
        let id_for_pull = id.clone();
        let pwa_pc_clone = pwa.pc.clone();
        let home_to_pwa_relay = tokio::spawn(async move {
            let mut cursor: usize = 0;
            for _ in 0..40 {
                let resp = app_for_pull
                    .clone()
                    .oneshot(empty_request(
                        Method::GET,
                        &format!("/signal/ice/{id_for_pull}?since={cursor}"),
                    ))
                    .await
                    .unwrap();
                if resp.status() != StatusCode::OK {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
                let v = body_json(resp).await;
                let new_cursor = v["cursor"].as_u64().unwrap_or(cursor as u64) as usize;
                if let Some(arr) = v["candidates"].as_array() {
                    for c in arr {
                        if let Some(cand_str) = c.get("candidate").and_then(|x| x.as_str()) {
                            if cand_str.is_empty() {
                                continue;
                            }
                            let init = webrtc::peer_connection::RTCIceCandidateInit {
                                candidate: cand_str.to_string(),
                                sdp_mid: c
                                    .get("sdpMid")
                                    .and_then(|x| x.as_str())
                                    .map(|s| s.to_string()),
                                sdp_mline_index: c
                                    .get("sdpMLineIndex")
                                    .and_then(|x| x.as_u64())
                                    .map(|x| x as u16),
                                username_fragment: None,
                                url: None,
                            };
                            let _ = pwa_pc_clone.add_ice_candidate(init).await;
                        }
                    }
                }
                cursor = new_cursor;
            }
        });

        // Step 5: PWA → home offer.
        let offer = pwa
            .pc
            .create_offer(None)
            .await
            .map_err(|e| anyhow::anyhow!("create_offer: {e}"))?;
        pwa.pc
            .set_local_description(offer.clone())
            .await
            .map_err(|e| anyhow::anyhow!("set_local_description: {e}"))?;

        let resp = app
            .clone()
            .oneshot(json_request(
                Method::POST,
                &format!("/signal/offer/{id}"),
                serde_json::json!({ "sdp": offer.sdp, "type": "offer" }),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NO_CONTENT);

        // Step 6: GET the answer + apply it.
        let resp = app
            .clone()
            .oneshot(empty_request(Method::GET, &format!("/signal/answer/{id}")))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        let answer_sdp = v["sdp"].as_str().unwrap().to_string();
        assert!(!answer_sdp.is_empty(), "answer SDP should be non-empty");
        let answer = webrtc::peer_connection::RTCSessionDescription::answer(answer_sdp)
            .map_err(|e| anyhow::anyhow!("RTCSessionDescription::answer: {e}"))?;
        pwa.pc
            .set_remote_description(answer)
            .await
            .map_err(|e| anyhow::anyhow!("set_remote_description(answer): {e}"))?;

        // Step 7: Wait for the data channel to open on the PWA side, send
        // a JSON-RPC ping, capture the reply.
        let (got_tx, mut got_rx) = mpsc::channel::<String>(1);
        let dc_for_reader = dc.clone();
        let reader = tokio::spawn(async move {
            let ping = serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "ping",
                "params": {}
            })
            .to_string();
            let mut sent = false;
            loop {
                match dc_for_reader.poll().await {
                    Some(DataChannelEvent::OnOpen) => {
                        if !sent {
                            let _ = dc_for_reader.send_text(&ping).await;
                            sent = true;
                        }
                    }
                    Some(DataChannelEvent::OnMessage(msg)) => {
                        let s = String::from_utf8_lossy(&msg.data).into_owned();
                        let _ = got_tx.send(s).await;
                        break;
                    }
                    Some(DataChannelEvent::OnClose) | None => break,
                    _ => continue,
                }
            }
        });

        let reply_text = tokio::time::timeout(Duration::from_secs(15), got_rx.recv())
            .await
            .map_err(|_| anyhow::anyhow!("timed out waiting for pong reply"))?
            .ok_or_else(|| anyhow::anyhow!("data channel closed before reply"))?;

        let reply: Value = serde_json::from_str(&reply_text)
            .map_err(|e| anyhow::anyhow!("reply not valid JSON ({e}): {reply_text}"))?;
        assert_eq!(reply["jsonrpc"], "2.0", "reply must be JSON-RPC 2.0");
        assert_eq!(
            reply["id"],
            serde_json::json!(1),
            "id must echo the request id"
        );
        assert_eq!(
            reply["result"]["ok"].as_bool(),
            Some(true),
            "result.ok must be true: {reply:?}"
        );
        assert!(
            reply["result"]["ts"].as_u64().is_some(),
            "result.ts must be a u64: {reply:?}"
        );

        // Cleanup.
        let _ = dc.close().await;
        let _ = pwa.pc.close().await;
        let _ = app
            .oneshot(empty_request(Method::DELETE, &format!("/signal/{id}")))
            .await;
        let _ = reader.await;
        let _ = pwa_ice_relay.await;
        let _ = home_to_pwa_relay.await;
        Ok(())
    }
}
