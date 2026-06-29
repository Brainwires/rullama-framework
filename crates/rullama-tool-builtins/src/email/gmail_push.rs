//! Gmail push ingestion via Google Cloud Pub/Sub.
//!
//! This module implements the "push" path for inbound Gmail:
//!
//! 1. The operator registers a `users.watch` on a mailbox, pointing at a
//!    Pub/Sub topic.
//! 2. Pub/Sub POSTs each notification to the BrainClaw gateway webhook
//!    `/webhooks/gmail-push`, carrying a Google-signed JWT in the
//!    `Authorization` header.
//! 3. The gateway calls [`GmailPushHandler::verify_push_jwt`] and
//!    [`GmailPushHandler::parse_envelope`] to authenticate the request and
//!    extract the watched mailbox plus history id.
//! 4. [`GmailPushHandler::fetch_new_messages`] then calls `users.history.list`
//!    + `users.messages.get` on the Gmail REST API to pull the actual
//!      messages, returning them as [`EmailMessage`] records suitable for
//!      dispatching to an agent.
//!
//! Watch registration is time-limited (7 days), so callers must call
//! [`GmailPushHandler::register_watch`] on a timer (see the daemon's
//! `app.rs` for the background task that owns this cadence).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use chrono::{DateTime, TimeZone, Utc};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Default Gmail REST API base URL. Tests override this to point at a
/// local mock server.
const DEFAULT_GMAIL_BASE: &str = "https://gmail.googleapis.com";

/// Google's OpenID Connect issuer used for Pub/Sub push tokens.
const GOOGLE_ISSUER: &str = "https://accounts.google.com";

/// Google's JWKs endpoint for RS256 push-token verification.
pub const GOOGLE_JWKS_URL: &str = "https://www.googleapis.com/oauth2/v3/certs";

/// How long a fetched JWKs set is cached before we refresh it.
const JWKS_CACHE_TTL: Duration = Duration::from_secs(3600);

/// Configuration for a [`GmailPushHandler`].
#[derive(Debug, Clone)]
pub struct GmailPushConfig {
    /// GCP project id that owns the Pub/Sub topic.
    pub project_id: String,
    /// Fully-qualified topic name, `projects/<proj>/topics/<topic>`.
    pub topic_name: String,
    /// Expected `aud` claim on the Google-signed push JWT. Configured on
    /// the Pub/Sub subscription.
    pub push_audience: String,
    /// Gmail labels to watch — typically `["INBOX"]`.
    pub watched_label_ids: Vec<String>,
    /// OAuth 2.0 access token (bearer) for the watched mailbox. This is
    /// the *user's* token — not a service-account one — and must have the
    /// `https://www.googleapis.com/auth/gmail.modify` scope at minimum.
    pub oauth_token: String,
    /// Override for the Gmail REST API base URL. Production code should
    /// leave this at the default; tests point it at a mock server.
    pub gmail_base_url: Option<String>,
}

impl GmailPushConfig {
    fn gmail_base(&self) -> &str {
        self.gmail_base_url.as_deref().unwrap_or(DEFAULT_GMAIL_BASE)
    }
}

/// Cached JWKs set with a fetched-at timestamp.
#[derive(Default)]
struct JwksCache {
    fetched_at: Option<std::time::Instant>,
    keys: Vec<JwkEntry>,
}

/// A single RSA JWK entry we care about.
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

/// Pub/Sub push handler for one watched Gmail account.
pub struct GmailPushHandler {
    config: GmailPushConfig,
    http: reqwest::Client,
    jwks: Arc<RwLock<JwksCache>>,
    jwks_url: String,
}

impl GmailPushHandler {
    /// Construct a new push handler.
    pub fn new(config: GmailPushConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest::Client default build");
        Self {
            config,
            http,
            jwks: Arc::new(RwLock::new(JwksCache::default())),
            jwks_url: GOOGLE_JWKS_URL.to_string(),
        }
    }

    /// Inject a custom HTTP client — for tests that need routing to a
    /// mock server.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// Override the JWKs endpoint. Production code should never call this;
    /// tests point it at a local mock server that serves RSA public keys
    /// matching a hand-signed test JWT.
    pub fn with_jwks_url(mut self, url: impl Into<String>) -> Self {
        self.jwks_url = url.into();
        self
    }

    /// Expose the configured push audience so webhook code can double-check
    /// claims without reaching into the private config.
    pub fn push_audience(&self) -> &str {
        &self.config.push_audience
    }

    /// Fetch Google's JWKs set, using the 1-hour cache.
    async fn jwks(&self) -> Result<Vec<JwkEntry>> {
        // Fast path: fresh cache.
        {
            let cache = self.jwks.read();
            if let Some(fetched) = cache.fetched_at
                && fetched.elapsed() < JWKS_CACHE_TTL
                && !cache.keys.is_empty()
            {
                return Ok(cache.keys.clone());
            }
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
        let body: JwksResponse = resp.json().await.context("parse JWKs JSON")?;
        let keys = body.keys;

        {
            let mut cache = self.jwks.write();
            cache.fetched_at = Some(std::time::Instant::now());
            cache.keys = keys.clone();
        }
        Ok(keys)
    }

    /// Verify the Google-signed JWT delivered in the `Authorization`
    /// header of a Pub/Sub push request.
    ///
    /// Returns the decoded claims on success. Fails on signature
    /// mismatch, wrong issuer, wrong audience, or expired token.
    pub async fn verify_push_jwt(&self, bearer_token: &str) -> Result<VerifiedPush> {
        let token = bearer_token.trim();
        let token = token.strip_prefix("Bearer ").unwrap_or(token).trim();
        if token.is_empty() {
            bail!("empty bearer token");
        }

        let header = decode_header(token).context("decode JWT header")?;
        if header.alg != Algorithm::RS256 {
            bail!("unexpected JWT alg {:?}; expected RS256", header.alg);
        }
        let kid = header
            .kid
            .as_ref()
            .ok_or_else(|| anyhow!("JWT header missing kid"))?;

        let jwks = self.jwks().await?;
        let jwk = jwks
            .iter()
            .find(|k| &k.kid == kid)
            .ok_or_else(|| anyhow!("no matching JWK for kid {}", kid))?;
        if let Some(alg) = &jwk.alg
            && alg != "RS256"
        {
            bail!("JWK alg {} is not RS256", alg);
        }

        let decoding_key = DecodingKey::from_rsa_components(&jwk.n, &jwk.e)
            .context("build decoding key from JWK")?;

        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_issuer(&[GOOGLE_ISSUER]);
        let audience = [self.config.push_audience.as_str()];
        validation.set_audience(&audience);
        validation.validate_exp = true;

        let data = decode::<GoogleJwtClaims>(token, &decoding_key, &validation)
            .context("verify Google push JWT")?;

        // `parse_envelope` operates on the body — the JWT itself only
        // authenticates the sender. We still return the claims so the
        // caller can cross-reference (e.g. log the service account email).
        Ok(VerifiedPush {
            aud: data.claims.aud,
            sub: data.claims.email.unwrap_or(data.claims.sub),
        })
    }

    /// Parse the Pub/Sub HTTP push envelope.
    ///
    /// Pub/Sub delivers JSON of the shape:
    ///
    /// ```json
    /// {
    ///   "message": { "data": "<base64>", "messageId": "...", "publishTime": "..." },
    ///   "subscription": "projects/.../subscriptions/..."
    /// }
    /// ```
    ///
    /// The inner `data` decodes to `{ "emailAddress": ..., "historyId": ... }`.
    pub fn parse_envelope(body: &[u8]) -> Result<PushEnvelope> {
        #[derive(Deserialize)]
        struct Outer {
            message: OuterMessage,
            #[serde(default)]
            subscription: Option<String>,
        }
        #[derive(Deserialize)]
        struct OuterMessage {
            data: String,
            #[serde(rename = "messageId", default)]
            message_id: Option<String>,
            #[serde(rename = "publishTime", default)]
            publish_time: Option<String>,
        }
        #[derive(Deserialize)]
        struct Inner {
            #[serde(rename = "emailAddress")]
            email_address: String,
            #[serde(rename = "historyId")]
            history_id: serde_json::Value,
        }

        let outer: Outer =
            serde_json::from_slice(body).context("parse Pub/Sub push envelope JSON")?;
        let decoded = STANDARD
            .decode(outer.message.data.as_bytes())
            .context("base64-decode Pub/Sub message data")?;
        let inner: Inner =
            serde_json::from_slice(&decoded).context("parse Gmail push inner JSON")?;

        let history_id = match &inner.history_id {
            serde_json::Value::Number(n) => n
                .as_u64()
                .ok_or_else(|| anyhow!("historyId is not u64: {n}"))?,
            serde_json::Value::String(s) => s
                .parse::<u64>()
                .with_context(|| format!("historyId string is not u64: {s}"))?,
            other => bail!("historyId has unexpected type: {other:?}"),
        };

        let publish_time = match &outer.message.publish_time {
            Some(ts) => DateTime::parse_from_rfc3339(ts)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            None => Utc::now(),
        };

        Ok(PushEnvelope {
            email_address: inner.email_address,
            history_id,
            publish_time,
            message_id: outer.message.message_id,
            subscription: outer.subscription,
        })
    }

    /// Fetch messages added since `since_history_id` for `verified.envelope.email_address`.
    ///
    /// Returns the fetched messages plus the new "latest" history id that
    /// the caller should persist. On a dry history window (no messages),
    /// the returned history id equals `since_history_id`.
    pub async fn fetch_new_messages(
        &self,
        envelope: &PushEnvelope,
        since_history_id: u64,
    ) -> Result<(Vec<EmailMessage>, u64)> {
        let base = self.config.gmail_base();
        let email = &envelope.email_address;

        // Step 1: GET /gmail/v1/users/{email}/history
        //   ?startHistoryId=<since>&historyTypes=messageAdded[&labelId=<label>]
        let mut url = format!(
            "{base}/gmail/v1/users/{email}/history?startHistoryId={since}&historyTypes=messageAdded",
            base = base,
            email = urlencoding::encode(email),
            since = since_history_id,
        );
        for label in &self.config.watched_label_ids {
            url.push_str("&labelId=");
            url.push_str(&urlencoding::encode(label));
        }

        let hist_resp = self
            .http
            .get(&url)
            .bearer_auth(&self.config.oauth_token)
            .send()
            .await
            .context("Gmail history.list request")?;

        if hist_resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            bail!("Gmail rate-limited history.list (429)");
        }
        if !hist_resp.status().is_success() {
            let status = hist_resp.status();
            let body = hist_resp.text().await.unwrap_or_default();
            bail!("Gmail history.list returned {status}: {body}");
        }
        let hist_json: serde_json::Value = hist_resp
            .json()
            .await
            .context("parse Gmail history.list JSON")?;

        let mut message_ids: Vec<String> = Vec::new();
        let mut new_history_id = since_history_id;

        if let Some(top_hid) = hist_json.get("historyId").and_then(|v| v.as_str())
            && let Ok(h) = top_hid.parse::<u64>()
        {
            new_history_id = new_history_id.max(h);
        }

        if let Some(entries) = hist_json.get("history").and_then(|v| v.as_array()) {
            for entry in entries {
                if let Some(hid) = entry.get("id").and_then(|v| v.as_str())
                    && let Ok(h) = hid.parse::<u64>()
                {
                    new_history_id = new_history_id.max(h);
                }
                if let Some(added) = entry.get("messagesAdded").and_then(|v| v.as_array()) {
                    for item in added {
                        if let Some(id) = item
                            .get("message")
                            .and_then(|m| m.get("id"))
                            .and_then(|v| v.as_str())
                            && !message_ids.iter().any(|x| x == id)
                        {
                            message_ids.push(id.to_string());
                        }
                    }
                }
            }
        }

        // Step 2: fetch each message via /gmail/v1/users/{email}/messages/{id}?format=full
        let mut out = Vec::with_capacity(message_ids.len());
        for id in &message_ids {
            let url = format!(
                "{base}/gmail/v1/users/{email}/messages/{id}?format=full",
                base = base,
                email = urlencoding::encode(email),
                id = urlencoding::encode(id),
            );
            let resp = self
                .http
                .get(&url)
                .bearer_auth(&self.config.oauth_token)
                .send()
                .await
                .context("Gmail messages.get request")?;
            if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                // Soft-fail: return what we have so far.
                tracing::warn!(
                    email = %email,
                    message_id = %id,
                    "Gmail rate-limited messages.get (429); returning partial batch"
                );
                break;
            }
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                tracing::warn!(
                    email = %email,
                    message_id = %id,
                    %status,
                    %body,
                    "Gmail messages.get failed"
                );
                continue;
            }
            let msg_json: serde_json::Value =
                resp.json().await.context("parse Gmail messages.get JSON")?;
            match parse_gmail_message(&msg_json) {
                Ok(msg) => out.push(msg),
                Err(e) => {
                    tracing::warn!(error = %e, message_id = %id, "parse Gmail message failed")
                }
            }
        }

        Ok((out, new_history_id))
    }

    /// Register (or renew) the Gmail watch. Returns the new starting history
    /// id and the watch's expiry.
    pub async fn register_watch(&self) -> Result<WatchResponse> {
        let base = self.config.gmail_base();
        let url = format!("{base}/gmail/v1/users/me/watch");

        let body = serde_json::json!({
            "topicName": self.config.topic_name,
            "labelIds": self.config.watched_label_ids,
        });

        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.config.oauth_token)
            .json(&body)
            .send()
            .await
            .context("Gmail users.watch request")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("Gmail users.watch returned {status}: {body}");
        }
        let resp_json: serde_json::Value = resp.json().await.context("parse users.watch JSON")?;
        let history_id = resp_json
            .get("historyId")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("users.watch response missing historyId"))?
            .parse::<u64>()
            .context("historyId is not u64")?;
        let expiration_ms = resp_json
            .get("expiration")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("users.watch response missing expiration"))?
            .parse::<i64>()
            .context("expiration is not i64")?;

        let expiration = Utc
            .timestamp_millis_opt(expiration_ms)
            .single()
            .ok_or_else(|| anyhow!("invalid expiration ms: {expiration_ms}"))?;

        Ok(WatchResponse {
            history_id,
            expiration,
        })
    }
}

/// Parsed and authenticated Pub/Sub push.
#[derive(Debug, Clone)]
pub struct VerifiedPush {
    /// `aud` claim from the JWT — must match the configured audience.
    pub aud: String,
    /// Service-account email (if present) or `sub` claim.
    pub sub: String,
}

/// Parsed Pub/Sub push envelope.
#[derive(Debug, Clone)]
pub struct PushEnvelope {
    /// The mailbox that received a new message.
    pub email_address: String,
    /// Opaque monotonic history cursor for this mailbox.
    pub history_id: u64,
    /// When Pub/Sub published the notification.
    pub publish_time: DateTime<Utc>,
    /// The Pub/Sub message id — used for best-effort de-dup by callers.
    pub message_id: Option<String>,
    /// Subscription the push came from, if reported by Pub/Sub.
    pub subscription: Option<String>,
}

/// Parsed Gmail message suitable for agent dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailMessage {
    /// Gmail message id.
    pub id: String,
    /// Gmail thread id.
    pub thread_id: String,
    /// `From:` header value.
    pub from: String,
    /// Parsed `To:` recipients.
    #[serde(default)]
    pub to: Vec<String>,
    /// Parsed `Cc:` recipients.
    #[serde(default)]
    pub cc: Vec<String>,
    /// `Subject:` header value.
    #[serde(default)]
    pub subject: String,
    /// Plain-text body (best-effort extraction).
    #[serde(default)]
    pub body_text: String,
    /// HTML body, when available.
    #[serde(default)]
    pub body_html: Option<String>,
    /// Arrival timestamp (`internalDate` from Gmail).
    pub received_at: DateTime<Utc>,
    /// Gmail labels on the message.
    #[serde(default)]
    pub labels: Vec<String>,
}

/// Response from `users.watch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchResponse {
    /// Starting history id — pass this as `since_history_id` on the next
    /// push notification.
    pub history_id: u64,
    /// When the watch expires. Call `register_watch` again before this.
    pub expiration: DateTime<Utc>,
}

// ── Internals ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct GoogleJwtClaims {
    aud: String,
    #[allow(dead_code)]
    iss: String,
    sub: String,
    #[serde(default)]
    email: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    exp: Option<i64>,
}

fn parse_gmail_message(json: &serde_json::Value) -> Result<EmailMessage> {
    let id = json
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("message missing id"))?
        .to_string();
    let thread_id = json
        .get("threadId")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let labels: Vec<String> = json
        .get("labelIds")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let payload = json
        .get("payload")
        .ok_or_else(|| anyhow!("message missing payload"))?;

    let mut from = String::new();
    let mut subject = String::new();
    let mut to: Vec<String> = Vec::new();
    let mut cc: Vec<String> = Vec::new();

    if let Some(headers) = payload.get("headers").and_then(|v| v.as_array()) {
        for h in headers {
            let name = h.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let value = h.get("value").and_then(|v| v.as_str()).unwrap_or("");
            match name.to_ascii_lowercase().as_str() {
                "from" => from = value.to_string(),
                "subject" => subject = value.to_string(),
                "to" => to = split_addresses(value),
                "cc" => cc = split_addresses(value),
                _ => {}
            }
        }
    }

    let (body_text, body_html) = extract_bodies(payload);

    let received_at = json
        .get("internalDate")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<i64>().ok())
        .and_then(|ms| Utc.timestamp_millis_opt(ms).single())
        .unwrap_or_else(Utc::now);

    Ok(EmailMessage {
        id,
        thread_id,
        from,
        to,
        cc,
        subject,
        body_text,
        body_html,
        received_at,
        labels,
    })
}

fn split_addresses(s: &str) -> Vec<String> {
    s.split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .collect()
}

/// Walk the Gmail payload tree and pick out the first `text/plain` and
/// `text/html` parts. Gmail base64url-encodes the body data.
fn extract_bodies(payload: &serde_json::Value) -> (String, Option<String>) {
    let mut text: Option<String> = None;
    let mut html: Option<String> = None;
    walk_parts(payload, &mut text, &mut html);
    (text.unwrap_or_default(), html)
}

fn walk_parts(part: &serde_json::Value, text: &mut Option<String>, html: &mut Option<String>) {
    let mime = part.get("mimeType").and_then(|v| v.as_str()).unwrap_or("");
    let body = part.get("body");
    let data_b64 = body
        .and_then(|b| b.get("data"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());

    if let Some(b64) = data_b64
        // Gmail uses URL-safe base64 without padding.
        && let Ok(bytes) = URL_SAFE_NO_PAD.decode(b64.trim_end_matches('='))
        && let Ok(s) = String::from_utf8(bytes)
    {
        match mime {
            "text/plain" if text.is_none() => *text = Some(s),
            "text/html" if html.is_none() => *html = Some(s),
            _ => {}
        }
    }

    if let Some(parts) = part.get("parts").and_then(|v| v.as_array()) {
        for p in parts {
            walk_parts(p, text, html);
            if text.is_some() && html.is_some() {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_envelope_roundtrip() {
        let inner = serde_json::json!({
            "emailAddress": "alice@example.com",
            "historyId": 12345u64,
        });
        let encoded = STANDARD.encode(serde_json::to_vec(&inner).unwrap());
        let outer = serde_json::json!({
            "message": {
                "data": encoded,
                "messageId": "pubsub-abc",
                "publishTime": "2025-01-01T00:00:00Z",
            },
            "subscription": "projects/p/subscriptions/s",
        });
        let body = serde_json::to_vec(&outer).unwrap();
        let env = GmailPushHandler::parse_envelope(&body).unwrap();
        assert_eq!(env.email_address, "alice@example.com");
        assert_eq!(env.history_id, 12345);
        assert_eq!(env.message_id.as_deref(), Some("pubsub-abc"));
        assert_eq!(
            env.subscription.as_deref(),
            Some("projects/p/subscriptions/s")
        );
    }

    #[test]
    fn parse_envelope_accepts_string_history_id() {
        // Google documents historyId as uint64 but the JSON representation
        // is often a decimal string because JavaScript can't round-trip
        // uint64 exactly. Accept both.
        let inner = serde_json::json!({
            "emailAddress": "bob@example.com",
            "historyId": "99999999999999",
        });
        let encoded = STANDARD.encode(serde_json::to_vec(&inner).unwrap());
        let outer = serde_json::json!({
            "message": { "data": encoded },
        });
        let body = serde_json::to_vec(&outer).unwrap();
        let env = GmailPushHandler::parse_envelope(&body).unwrap();
        assert_eq!(env.email_address, "bob@example.com");
        assert_eq!(env.history_id, 99_999_999_999_999u64);
    }

    #[test]
    fn parse_gmail_message_plain_text() {
        let body_data = URL_SAFE_NO_PAD.encode(b"hello world");
        let payload = serde_json::json!({
            "id": "m-1",
            "threadId": "t-1",
            "labelIds": ["INBOX", "UNREAD"],
            "internalDate": "1700000000000",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    { "name": "From", "value": "alice@example.com" },
                    { "name": "To",   "value": "bob@example.com, carol@example.com" },
                    { "name": "Subject", "value": "hi" },
                ],
                "body": { "data": body_data },
            }
        });
        let msg = parse_gmail_message(&payload).unwrap();
        assert_eq!(msg.id, "m-1");
        assert_eq!(msg.thread_id, "t-1");
        assert_eq!(msg.from, "alice@example.com");
        assert_eq!(msg.to.len(), 2);
        assert_eq!(msg.subject, "hi");
        assert_eq!(msg.body_text, "hello world");
        assert!(msg.body_html.is_none());
        assert_eq!(msg.labels, vec!["INBOX", "UNREAD"]);
    }

    #[test]
    fn parse_gmail_message_multipart() {
        let text_data = URL_SAFE_NO_PAD.encode(b"plain body");
        let html_data = URL_SAFE_NO_PAD.encode(b"<p>html body</p>");
        let payload = serde_json::json!({
            "id": "m-2",
            "threadId": "t-2",
            "internalDate": "1700000000000",
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [
                    { "name": "From",    "value": "dave@example.com" },
                    { "name": "Subject", "value": "multi" },
                ],
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": { "data": text_data },
                    },
                    {
                        "mimeType": "text/html",
                        "body": { "data": html_data },
                    }
                ]
            }
        });
        let msg = parse_gmail_message(&payload).unwrap();
        assert_eq!(msg.body_text, "plain body");
        assert_eq!(msg.body_html.as_deref(), Some("<p>html body</p>"));
    }

    #[test]
    fn extract_bodies_short_circuits_on_complete() {
        let text_data = URL_SAFE_NO_PAD.encode(b"T");
        let html_data = URL_SAFE_NO_PAD.encode(b"H");
        let payload = serde_json::json!({
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        { "mimeType": "text/plain", "body": { "data": text_data } },
                        { "mimeType": "text/html",  "body": { "data": html_data } },
                    ]
                }
            ]
        });
        let (t, h) = extract_bodies(&payload);
        assert_eq!(t, "T");
        assert_eq!(h.as_deref(), Some("H"));
    }
}
