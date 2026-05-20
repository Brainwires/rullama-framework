//! HTTPS webhook receiver for Google Chat bot events.
//!
//! Google signs every push with an RS256 JWT in the `Authorization`
//! header. The JWT is verified against Google's published JWKs
//! (`https://www.googleapis.com/oauth2/v3/certs`), with audience matching
//! and expiry checks. Requests that fail verification get 401.
//!
//! The message body is parsed via [`crate::google_chat::parse_event`] and
//! interesting events are forwarded to the gateway via the supplied
//! `ChannelEvent` sender.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use axum::{
    Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use parking_lot::RwLock;
use serde::Deserialize;
use tokio::sync::mpsc;
use tracing::Instrument;

use brainwires_network::channels::ChannelEvent;

use crate::google_chat::{IngressEvent, parse_event};

/// Google's JWKs endpoint. Exposed so tests can override it.
pub const GOOGLE_JWKS_URL: &str = "https://www.googleapis.com/oauth2/v3/certs";

/// Google's OIDC issuer — identity tokens minted by Google come from here.
pub const GOOGLE_ISSUER: &str = "https://accounts.google.com";

const JWKS_CACHE_TTL: Duration = Duration::from_secs(3600);

/// JWKs cache — shared between requests on one server instance.
#[derive(Default)]
struct JwksCache {
    fetched_at: Option<std::time::Instant>,
    keys: Vec<JwkEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct JwkEntry {
    kid: String,
    #[serde(default)]
    alg: Option<String>,
    n: String,
    e: String,
}

#[derive(Debug, Deserialize)]
struct JwksResponse {
    keys: Vec<JwkEntry>,
}

/// State shared with the Axum handlers.
#[derive(Clone)]
pub struct WebhookState {
    audience: String,
    jwks_url: String,
    http: reqwest::Client,
    jwks: Arc<RwLock<JwksCache>>,
    event_tx: mpsc::Sender<ChannelEvent>,
}

impl WebhookState {
    /// Build a new state from the run-time config.
    pub fn new(audience: impl Into<String>, event_tx: mpsc::Sender<ChannelEvent>) -> Self {
        Self {
            audience: audience.into(),
            jwks_url: GOOGLE_JWKS_URL.to_string(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("build reqwest client"),
            jwks: Arc::new(RwLock::new(JwksCache::default())),
            event_tx,
        }
    }

    /// Override the JWKs URL — tests only.
    pub fn with_jwks_url(mut self, url: impl Into<String>) -> Self {
        self.jwks_url = url.into();
        self
    }

    /// Override the HTTP client — tests only.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    async fn jwks(&self) -> Result<Vec<JwkEntry>> {
        if let Some(t) = self.jwks.read().fetched_at
            && t.elapsed() < JWKS_CACHE_TTL
        {
            return Ok(self.jwks.read().keys.clone());
        }
        let resp = self
            .http
            .get(&self.jwks_url)
            .send()
            .await
            .context("fetch Google JWKs")?;
        if !resp.status().is_success() {
            bail!("JWKs endpoint returned {}", resp.status());
        }
        let parsed: JwksResponse = resp.json().await.context("parse JWKs JSON")?;
        let mut cache = self.jwks.write();
        cache.fetched_at = Some(std::time::Instant::now());
        cache.keys = parsed.keys.clone();
        Ok(parsed.keys)
    }

    /// Verify a bearer-style `Authorization` header value.
    pub async fn verify(&self, header: &str) -> Result<GoogleClaims> {
        let token = header
            .trim()
            .strip_prefix("Bearer ")
            .unwrap_or(header)
            .trim();
        if token.is_empty() {
            bail!("empty bearer token");
        }
        let hdr = decode_header(token).context("decode JWT header")?;
        if hdr.alg != Algorithm::RS256 {
            bail!("unexpected alg {:?}", hdr.alg);
        }
        let kid = hdr.kid.ok_or_else(|| anyhow!("JWT header missing kid"))?;
        let jwks = self.jwks().await?;
        let jwk = jwks
            .iter()
            .find(|k| k.kid == kid)
            .ok_or_else(|| anyhow!("no matching JWK for kid {}", kid))?;
        if let Some(alg) = &jwk.alg
            && alg != "RS256"
        {
            bail!("JWK alg {} is not RS256", alg);
        }
        let dk = DecodingKey::from_rsa_components(&jwk.n, &jwk.e).context("build decoding key")?;
        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_issuer(&[GOOGLE_ISSUER]);
        let audience = [self.audience.as_str()];
        validation.set_audience(&audience);
        validation.validate_exp = true;
        let data = decode::<GoogleClaims>(token, &dk, &validation).context("verify JWT")?;
        Ok(data.claims)
    }
}

/// Claims we care about from the Google-signed JWT.
#[derive(Debug, Clone, Deserialize)]
pub struct GoogleClaims {
    /// Audience — must match our configured audience.
    pub aud: String,
    /// Issuer — always Google.
    #[serde(default)]
    pub iss: String,
    /// Service-account email that signed the push.
    #[serde(default)]
    pub email: Option<String>,
    /// Subject claim.
    #[serde(default)]
    pub sub: String,
}

/// Start the Axum webhook server.
pub async fn serve(state: WebhookState, listen_addr: &str) -> Result<()> {
    let app = Router::new()
        .route("/events", post(handle_event))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(listen_addr)
        .await
        .with_context(|| format!("bind {listen_addr}"))?;
    tracing::info!(%listen_addr, "Google Chat webhook listening");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_event(
    State(state): State<WebhookState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let auth = match headers.get("authorization").and_then(|v| v.to_str().ok()) {
        Some(v) => v,
        None => {
            tracing::warn!("google-chat webhook: missing Authorization header");
            return (StatusCode::UNAUTHORIZED, "missing Authorization").into_response();
        }
    };

    if let Err(e) = state.verify(auth).await {
        tracing::warn!(error = %e, "google-chat webhook: JWT verification failed");
        return (StatusCode::UNAUTHORIZED, "jwt verification failed").into_response();
    }

    let payload: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "google-chat webhook: body is not JSON");
            return (StatusCode::BAD_REQUEST, "invalid json").into_response();
        }
    };

    // Forward (or drop) asynchronously so we keep the HTTP handler fast.
    let tx = state.event_tx.clone();
    let span = tracing::info_span!("google_chat_ingress");
    tokio::spawn(
        async move {
            if let Err(e) = dispatch(&tx, &payload).await {
                tracing::error!(error = %e, "failed to dispatch Chat event");
            }
        }
        .instrument(span),
    );

    (StatusCode::OK, "ok").into_response()
}

/// Parse + forward a single Chat event. Returns `Ok(())` on both
/// forwarded and intentionally-dropped events; only real errors bubble.
pub async fn dispatch(tx: &mpsc::Sender<ChannelEvent>, payload: &serde_json::Value) -> Result<()> {
    let event = parse_event(payload)?;
    match event {
        IngressEvent::Message(msg) | IngressEvent::CardClicked(msg) => {
            audit_log(&msg);
            tx.send(ChannelEvent::MessageReceived(msg))
                .await
                .context("dispatch Chat message to gateway")
        }
        IngressEvent::Lifecycle { space_id, added } => {
            tracing::info!(%space_id, %added, "google-chat lifecycle event (not forwarded)");
            Ok(())
        }
        IngressEvent::Other { event_type } => {
            tracing::debug!(%event_type, "google-chat event ignored");
            Ok(())
        }
    }
}

fn audit_log(msg: &brainwires_network::channels::ChannelMessage) {
    let user_digest = hashed_user_id(&msg.author);
    let len = match &msg.content {
        brainwires_network::channels::MessageContent::Text(t) => t.len(),
        brainwires_network::channels::MessageContent::RichText { markdown, .. } => markdown.len(),
        _ => 0,
    };
    tracing::info!(
        channel = "google_chat",
        user = %user_digest,
        message_len = len,
        "forwarded"
    );
}

fn hashed_user_id(author: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(author.as_bytes());
    let out = h.finalize();
    hex::encode(&out[..6])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn dispatch_forwards_message_event() {
        let (tx, mut rx) = mpsc::channel(4);
        let payload = json!({
            "type": "MESSAGE",
            "message": {
                "name": "spaces/A/messages/M",
                "sender": { "name": "users/1", "displayName": "Alice" },
                "space": { "name": "spaces/A" },
                "text": "hi",
                "argumentText": "hi",
                "createTime": "2025-01-01T00:00:00Z",
            }
        });
        dispatch(&tx, &payload).await.unwrap();
        let evt = rx.recv().await.expect("event");
        match evt {
            ChannelEvent::MessageReceived(m) => {
                assert_eq!(m.author, "Alice");
                assert_eq!(m.conversation.channel_id, "A");
            }
            _ => panic!("expected MessageReceived"),
        }
    }

    #[tokio::test]
    async fn dispatch_drops_lifecycle() {
        let (tx, mut rx) = mpsc::channel(4);
        let payload = json!({
            "type": "ADDED_TO_SPACE",
            "space": { "name": "spaces/X" }
        });
        dispatch(&tx, &payload).await.unwrap();
        assert!(rx.try_recv().is_err(), "lifecycle must not forward");
    }

    #[tokio::test]
    async fn dispatch_drops_unknown() {
        let (tx, mut rx) = mpsc::channel(4);
        let payload = json!({ "type": "WHATEVER" });
        dispatch(&tx, &payload).await.unwrap();
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn hashed_user_id_is_short_hex() {
        let h = hashed_user_id("alice@example.com");
        assert_eq!(h.len(), 12);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[tokio::test]
    async fn verify_rejects_empty_bearer() {
        let (tx, _rx) = mpsc::channel(1);
        let state = WebhookState::new("aud", tx);
        let err = state.verify("").await.unwrap_err();
        assert!(err.to_string().contains("empty bearer"));
    }

    #[tokio::test]
    async fn verify_rejects_malformed_token() {
        let (tx, _rx) = mpsc::channel(1);
        let state = WebhookState::new("aud", tx);
        let err = state.verify("Bearer garbage").await.unwrap_err();
        // Any decode failure is acceptable — just not a panic.
        assert!(!err.to_string().is_empty());
    }
}
