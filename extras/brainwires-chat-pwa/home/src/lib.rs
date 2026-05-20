//! `brainwires-home` — dial-home daemon for the Brainwires chat PWA.
//!
//! Runs on the user's home machine; the PWA reaches it via WebRTC behind a
//! Cloudflare Tunnel (or equivalent). See `README.md` for the full
//! architecture, endpoints, and pairing flow.
//!
//! This is the **library** surface. The binary in `src/main.rs` is a thin
//! shim that parses CLI flags and calls [`HomeServer::serve`]. Headless
//! integration tests can spin one up via [`HomeServer::builder`] without
//! touching the binary path — and via [`HomeServer::router`] without binding
//! a port at all.

pub mod a2a;
pub mod binary;
pub mod pairing;
pub mod signaling;
pub mod sync;
pub mod turn;
pub mod webrtc;

use anyhow::{Context, Result};
use axum::Router;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::a2a::A2aBridge;
use crate::pairing::{CfAccessConfig, PairingState};
use crate::signaling::{AppState, CorsConfig, DEFAULT_LONG_POLL, DEFAULT_SESSION_TTL};
use crate::sync::SyncStore;
use crate::turn::DEFAULT_TTL_SECS;

/// Server-side TURN credential minting config.
///
/// When [`TurnConfig::turn_key_id`] **and** [`TurnConfig::api_token`] are both
/// set, `POST /signal/session` mints a fresh short-lived ICE credential per
/// session via the Cloudflare Calls API. Otherwise the daemon answers with
/// STUN-only fallback (`stun.cloudflare.com:3478`, `stun.l.google.com:19302`).
///
/// The PWA never sees the API token. It only ever receives the minted
/// `iceServers` list — so token rotation is a pure home-side operation.
#[derive(Clone, Debug)]
pub struct TurnConfig {
    /// Cloudflare Calls TURN key id. Find it under
    /// `dashboard.cloudflare.com → Calls → TURN keys`.
    pub turn_key_id: Option<String>,
    /// Cloudflare Calls API token (NOT a Cloudflare Tunnel token — different
    /// product). Scoped to the Calls API only.
    pub api_token: Option<String>,
    /// Lifetime of minted credentials in seconds. Default 600 (10 minutes).
    /// Floored at 60 s before being sent to Cloudflare.
    pub credential_ttl_secs: u32,
}

impl Default for TurnConfig {
    fn default() -> Self {
        Self {
            turn_key_id: None,
            api_token: None,
            credential_ttl_secs: DEFAULT_TTL_SECS,
        }
    }
}

/// Default loopback bind. The daemon expects to live behind a Cloudflare
/// Tunnel (or equivalent reverse tunnel) — listening on a public interface
/// directly is supported for dev only.
pub const DEFAULT_BIND: &str = "127.0.0.1:7878";

/// Top-level handle to the running daemon.
pub struct HomeServer {
    bind: SocketAddr,
    state: AppState,
    cors: CorsConfig,
    pairing: Option<PairingState>,
}

/// Builder for [`HomeServer`].
///
/// Phase-2 milestones layer on further setters: `with_task_agent(...)` (M4),
/// `with_turn_minter(...)` (M7), `with_pairing_store(...)` (M8). M2 wires
/// [`HomeServerBuilder::bind`], [`HomeServerBuilder::long_poll_timeout`], and
/// [`HomeServerBuilder::session_ttl`].
pub struct HomeServerBuilder {
    bind: Option<SocketAddr>,
    long_poll_timeout: Duration,
    session_ttl: Duration,
    bridge: Option<Arc<A2aBridge>>,
    cors: CorsConfig,
    cors_explicit: bool,
    turn: TurnConfig,
    pairing: Option<PairingState>,
    sync_store: Option<Arc<SyncStore>>,
}

impl HomeServer {
    /// Start a new builder.
    pub fn builder() -> HomeServerBuilder {
        HomeServerBuilder {
            bind: None,
            long_poll_timeout: DEFAULT_LONG_POLL,
            session_ttl: DEFAULT_SESSION_TTL,
            bridge: None,
            cors: CorsConfig::default(),
            cors_explicit: false,
            turn: TurnConfig::default(),
            pairing: None,
            sync_store: None,
        }
    }

    /// The address the daemon will bind to.
    pub fn bind_addr(&self) -> SocketAddr {
        self.bind
    }

    /// Borrow the shared application state. Useful in tests that want to
    /// poke the in-memory session map directly while exercising the router.
    pub fn state(&self) -> &AppState {
        &self.state
    }

    /// Build the configured `axum::Router` without binding a port.
    ///
    /// Tests can drive this via `tower::ServiceExt::oneshot`. Production code
    /// goes through [`HomeServer::serve`], which binds and runs it.
    pub fn router(&self) -> Router {
        let mut router = signaling::router_with_cors(self.state.clone(), self.cors.clone());
        if let Some(p) = self.pairing.clone() {
            // The signaling router already attached its CORS layer. Apply
            // the same one to the pairing sub-router before merging — axum
            // doesn't propagate layers across `merge` calls automatically.
            let pair_router = pairing::router(p).layer(self.cors.clone().into_layer());
            router = router.merge(pair_router);
        }
        router
    }

    /// Borrow the pairing state, if configured. Tests use this to mint
    /// offers and inspect `devices.json` without going through the CLI.
    pub fn pairing(&self) -> Option<&PairingState> {
        self.pairing.as_ref()
    }

    /// Run the server until it errors or is dropped.
    ///
    /// Binds to [`HomeServer::bind_addr`], spawns the session GC task, and
    /// hands the listener to `axum::serve`. Returns when the server exits.
    pub async fn serve(self) -> Result<()> {
        let _gc = self.state.spawn_gc();
        let mut app = signaling::router_with_cors(self.state.clone(), self.cors.clone());
        if let Some(p) = self.pairing.clone() {
            let pair_router = pairing::router(p).layer(self.cors.clone().into_layer());
            app = app.merge(pair_router);
        }
        let listener = tokio::net::TcpListener::bind(self.bind)
            .await
            .with_context(|| format!("bind {}", self.bind))?;
        tracing::info!(
            addr = %self.bind,
            "brainwires-home: signaling server listening (M2)",
        );
        axum::serve(listener, app)
            .await
            .context("axum::serve exited with error")
    }
}

impl HomeServerBuilder {
    /// Override the bind address. Default: [`DEFAULT_BIND`].
    pub fn bind(mut self, addr: SocketAddr) -> Self {
        self.bind = Some(addr);
        self
    }

    /// Override the long-poll wait for `/signal/answer` and `/signal/ice`.
    /// Default: 25 s. Tests usually shrink this to ~200 ms.
    pub fn long_poll_timeout(mut self, d: Duration) -> Self {
        self.long_poll_timeout = d;
        self
    }

    /// Override the session TTL. Sessions older than this are GC'd. Default:
    /// 30 minutes.
    pub fn session_ttl(mut self, d: Duration) -> Self {
        self.session_ttl = d;
        self
    }

    /// Attach the [`A2aBridge`] that the WebRTC data-channel loop will
    /// route inbound JSON-RPC frames through (M4).
    ///
    /// Production wiring (real provider, API keys, system prompt, ...)
    /// happens before construction: callers build a [`brainwires_agent::ChatAgent`]
    /// however they like, wrap it in an [`A2aBridge`], then hand it here.
    /// Tests use [`crate::a2a::test_support::echo_chat_agent`] to skip the
    /// network entirely.
    ///
    /// If unset, the daemon still answers `ping` for the M3 smoke-test path
    /// but rejects every other inbound method.
    pub fn with_agent(mut self, bridge: Arc<A2aBridge>) -> Self {
        self.bridge = Some(bridge);
        self
    }

    /// Add an exact-match origin to the CORS allow-list. Repeatable.
    ///
    /// The first call clears the [`signaling::DEFAULT_DEV_ORIGINS`]
    /// defaults — once the caller names a real PWA origin, the daemon
    /// should not also silently accept localhost dev origins.
    pub fn cors_allow_origin(mut self, origin: impl Into<String>) -> Self {
        self.cors = self.cors.allow_origin(origin);
        self.cors_explicit = true;
        self
    }

    /// Wide-open CORS (`Access-Control-Allow-Origin: *`). **Dev only.**
    /// Disables the default dev-origin allow-list and accepts any origin.
    /// The CLI exposes this as `--cors-permissive`; the README flags it
    /// as a footgun for production.
    pub fn cors_permissive(mut self) -> Self {
        self.cors = CorsConfig::default().permissive();
        self.cors_explicit = true;
        self
    }

    /// Configure Cloudflare Calls TURN credential minting (M7).
    ///
    /// When set, `POST /signal/session` returns an `ice_servers` array that
    /// includes a freshly-minted, short-lived TURN credential alongside the
    /// public STUN fallbacks. Without this call, only STUN is returned —
    /// which works for the ~85% of NAT topologies STUN can punch but fails
    /// on cellular symmetric NAT.
    ///
    /// `turn_key_id` is the Cloudflare Calls TURN key id (visible in the
    /// dashboard); `api_token` is a Cloudflare Calls API token (NOT a
    /// Cloudflare Tunnel token — different product, different page).
    pub fn with_cloudflare_turn(
        mut self,
        turn_key_id: impl Into<String>,
        api_token: impl Into<String>,
    ) -> Self {
        self.turn.turn_key_id = Some(turn_key_id.into());
        self.turn.api_token = Some(api_token.into());
        self
    }

    /// Override the TURN credential TTL in seconds. Default 600 (10 min).
    /// Floored at 60 s; anything shorter is more likely to expire mid-
    /// handshake than to be useful.
    pub fn with_turn_ttl(mut self, seconds: u32) -> Self {
        self.turn.credential_ttl_secs = seconds;
        self
    }

    /// Attach a [`PairingState`] (M8). When set, the daemon serves
    /// `POST /pair/claim` + `POST /pair/confirm` and persists confirmed
    /// devices to the configured `devices.json` path.
    pub fn with_pairing(mut self, pairing: PairingState) -> Self {
        self.pairing = Some(pairing);
        self
    }

    /// Attach a [`SyncStore`] for cross-device data synchronization.
    /// The store is opened (or created) at the given path.
    pub fn with_sync(mut self, path: impl Into<PathBuf>) -> Result<Self> {
        let store = SyncStore::new(path.into())?;
        self.sync_store = Some(Arc::new(store));
        Ok(self)
    }

    /// Configure pre-provisioned Cloudflare Access service-token creds
    /// (M8). Returned to every PWA that successfully pairs so it can
    /// include them as `CF-Access-Client-Id` / `CF-Access-Client-Secret`
    /// on signaling requests through Cloudflare Access. Only takes effect
    /// when [`HomeServerBuilder::with_pairing`] has also been called.
    pub fn with_cf_access(mut self, client_id: String, client_secret: String) -> Self {
        if let Some(p) = self.pairing.take() {
            // Rebuild with the CF creds attached. The CF config lives on
            // the PairingState itself (it's read on every confirm).
            self.pairing = Some(PairingState::new(
                p.devices_path().to_path_buf(),
                Some(CfAccessConfig {
                    client_id,
                    client_secret,
                }),
                p.peer_pubkey().to_string(),
            ));
        } else {
            tracing::warn!(
                "with_cf_access called before with_pairing; ignoring CF creds — \
                 attach pairing first",
            );
        }
        self
    }

    /// Materialize the builder into a [`HomeServer`].
    pub fn build(self) -> Result<HomeServer> {
        let bind = self
            .bind
            .unwrap_or_else(|| DEFAULT_BIND.parse().expect("DEFAULT_BIND parses"));
        let mut state =
            AppState::new(self.long_poll_timeout, self.session_ttl).with_turn(self.turn);
        if let Some(bridge) = self.bridge {
            state = state.with_bridge(bridge);
        }
        if let Some(sync) = self.sync_store {
            state = state.with_sync(sync);
        }
        if !self.cors_explicit {
            tracing::debug!(
                "no --cors-origin / --cors-permissive flags; allowing chat-PWA dev origins only"
            );
        }
        Ok(HomeServer {
            bind,
            state,
            cors: self.cors,
            pairing: self.pairing,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults_to_loopback_7878() {
        let server = HomeServer::builder().build().expect("build");
        assert_eq!(server.bind_addr().to_string(), "127.0.0.1:7878");
    }

    #[test]
    fn builder_respects_explicit_bind() {
        let addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let server = HomeServer::builder().bind(addr).build().expect("build");
        assert_eq!(server.bind_addr(), addr);
    }

    #[test]
    fn builder_exposes_router_without_binding() {
        // Constructing the router should never touch the network.
        let server = HomeServer::builder()
            .long_poll_timeout(Duration::from_millis(50))
            .session_ttl(Duration::from_secs(5))
            .build()
            .expect("build");
        let _router: Router = server.router();
    }
}
