//! Service-account OAuth bearer minter.
//!
//! Implements the self-signed JWT flow documented at
//! <https://developers.google.com/identity/protocols/oauth2/service-account#authorizingrequests>
//! — sign a JWT with the service-account private key, POST it to Google's
//! token endpoint, receive a short-lived bearer. Tokens are cached until
//! five minutes before expiry and refreshed on demand.
//!
//! We avoid `yup-oauth2` to keep the dependency footprint small and to
//! match the existing project pattern of using `jsonwebtoken` directly
//! (see `crates/brainwires-tools/src/email/gmail_push.rs`).

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// OAuth scopes required to send Chat messages as a bot.
pub const CHAT_BOT_SCOPE: &str = "https://www.googleapis.com/auth/chat.bot";

/// Minimum remaining TTL before a cached token is considered stale.
const REFRESH_MARGIN: Duration = Duration::from_secs(5 * 60);

/// Minimal view of the service-account JSON key file.
#[derive(Debug, Clone, Deserialize)]
pub struct ServiceAccountKey {
    /// `client_email` — used as the JWT `iss`/`sub`.
    pub client_email: String,
    /// PEM-encoded RSA private key (`BEGIN PRIVATE KEY` …).
    pub private_key: String,
    /// Google's token endpoint. Optional — we fall back to the canonical
    /// URL when the JSON omits it.
    #[serde(default)]
    pub token_uri: Option<String>,
    /// The service-account type marker (`service_account`). Validated
    /// loosely; wrong values produce a descriptive error, not a silent
    /// failure.
    #[serde(default, rename = "type")]
    pub account_type: Option<String>,
}

/// JWT assertion claims signed with the service-account key.
#[derive(Debug, Serialize)]
struct Claims<'a> {
    iss: &'a str,
    scope: &'a str,
    aud: &'a str,
    iat: i64,
    exp: i64,
}

/// Successful token response from Google.
#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
    #[serde(default)]
    token_type: Option<String>,
}

/// A cached bearer plus its expiry wall-clock.
#[derive(Clone)]
struct CachedToken {
    bearer: String,
    fetched_at: Instant,
    ttl: Duration,
}

impl CachedToken {
    fn is_fresh(&self) -> bool {
        self.fetched_at.elapsed() + REFRESH_MARGIN < self.ttl
    }
}

/// Canonical token endpoint — used when the key file doesn't override it.
const DEFAULT_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";

/// Thread-safe minter for service-account bearer tokens.
pub struct TokenMinter {
    key: ServiceAccountKey,
    scope: String,
    http: reqwest::Client,
    cache: Arc<RwLock<Option<CachedToken>>>,
    token_uri_override: Option<String>,
}

impl TokenMinter {
    /// Construct a minter from a path to a service-account JSON key.
    pub fn from_key_path(path: &str, scope: impl Into<String>) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("read service-account key at {path}"))?;
        let key: ServiceAccountKey =
            serde_json::from_slice(&bytes).context("parse service-account JSON key")?;
        Self::from_key(key, scope)
    }

    /// Construct a minter directly from a parsed key (tests mostly).
    pub fn from_key(key: ServiceAccountKey, scope: impl Into<String>) -> Result<Self> {
        if let Some(t) = &key.account_type
            && t != "service_account"
        {
            bail!("key file has type '{}' — expected 'service_account'", t);
        }
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("build reqwest client")?;
        Ok(Self {
            key,
            scope: scope.into(),
            http,
            cache: Arc::new(RwLock::new(None)),
            token_uri_override: None,
        })
    }

    /// Override the token URI (tests only — production uses the default).
    pub fn with_token_uri(mut self, uri: impl Into<String>) -> Self {
        self.token_uri_override = Some(uri.into());
        self
    }

    /// Inject a custom HTTP client — used by integration tests pointed at
    /// a mock token server.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    fn token_uri(&self) -> &str {
        self.token_uri_override
            .as_deref()
            .or(self.key.token_uri.as_deref())
            .unwrap_or(DEFAULT_TOKEN_URI)
    }

    /// Return a currently-valid bearer token, refreshing on demand.
    pub async fn bearer(&self) -> Result<String> {
        if let Some(t) = self.cache.read().as_ref()
            && t.is_fresh()
        {
            return Ok(t.bearer.clone());
        }
        let fresh = self.mint_new().await?;
        let bearer = fresh.bearer.clone();
        *self.cache.write() = Some(fresh);
        Ok(bearer)
    }

    async fn mint_new(&self) -> Result<CachedToken> {
        let now = chrono::Utc::now().timestamp();
        let claims = Claims {
            iss: &self.key.client_email,
            scope: &self.scope,
            aud: self.token_uri(),
            iat: now,
            exp: now + 3600,
        };

        let header = Header::new(Algorithm::RS256);
        let encoding_key = EncodingKey::from_rsa_pem(self.key.private_key.as_bytes())
            .context("parse service-account private_key PEM")?;
        let assertion = encode(&header, &claims, &encoding_key)
            .context("sign service-account JWT assertion")?;

        let form = [
            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
            ("assertion", assertion.as_str()),
        ];

        let resp = self
            .http
            .post(self.token_uri())
            .form(&form)
            .send()
            .await
            .context("POST service-account assertion to Google token endpoint")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            // Deliberately redact — the body can contain assertion echoes.
            bail!(
                "token endpoint returned {}: {} bytes of error body",
                status,
                body.len()
            );
        }

        let parsed: TokenResponse = resp
            .json()
            .await
            .context("parse token endpoint response body")?;

        if parsed.access_token.is_empty() {
            return Err(anyhow!("token endpoint returned empty access_token"));
        }
        if let Some(tt) = &parsed.token_type
            && !tt.eq_ignore_ascii_case("Bearer")
        {
            bail!("unexpected token_type '{tt}'; expected 'Bearer'");
        }

        Ok(CachedToken {
            bearer: parsed.access_token,
            fetched_at: Instant::now(),
            ttl: Duration::from_secs(parsed.expires_in),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A deliberately-invalid PEM so `from_key` still passes the structural
    // check (file is parseable JSON) but signing would fail. We never hit
    // the sign path from unit tests — integration tests mock the token
    // endpoint and skip real keys entirely.
    const FAKE_PEM: &str = "-----BEGIN PRIVATE KEY-----\nQUJDREVG\n-----END PRIVATE KEY-----\n";

    #[test]
    fn rejects_non_service_account_type() {
        let key = ServiceAccountKey {
            client_email: "x@y.iam.gserviceaccount.com".into(),
            private_key: FAKE_PEM.into(),
            token_uri: None,
            account_type: Some("user".into()),
        };
        let err = TokenMinter::from_key(key, "scope")
            .err()
            .expect("should reject non-service account");
        assert!(err.to_string().contains("expected 'service_account'"));
    }

    #[test]
    fn accepts_service_account_type() {
        let key = ServiceAccountKey {
            client_email: "x@y.iam.gserviceaccount.com".into(),
            private_key: FAKE_PEM.into(),
            token_uri: None,
            account_type: Some("service_account".into()),
        };
        assert!(TokenMinter::from_key(key, CHAT_BOT_SCOPE).is_ok());
    }

    #[test]
    fn token_uri_override_takes_precedence() {
        let key = ServiceAccountKey {
            client_email: "x@y.iam.gserviceaccount.com".into(),
            private_key: FAKE_PEM.into(),
            token_uri: Some("https://override.example/token".into()),
            account_type: None,
        };
        let minter = TokenMinter::from_key(key, CHAT_BOT_SCOPE)
            .unwrap()
            .with_token_uri("https://test.local/token");
        assert_eq!(minter.token_uri(), "https://test.local/token");
    }

    #[test]
    fn cached_token_fresh_then_stale() {
        // Fresh: ttl = 1h, just fetched.
        let t = CachedToken {
            bearer: "abc".into(),
            fetched_at: Instant::now(),
            ttl: Duration::from_secs(3600),
        };
        assert!(t.is_fresh());

        // Stale: "fetched" two hours ago.
        let past = Instant::now()
            .checked_sub(Duration::from_secs(7200))
            .unwrap_or_else(Instant::now);
        let t = CachedToken {
            bearer: "abc".into(),
            fetched_at: past,
            ttl: Duration::from_secs(3600),
        };
        assert!(!t.is_fresh());

        // Inside refresh margin: ttl 10 min, fetched 9 min ago.
        let past = Instant::now()
            .checked_sub(Duration::from_secs(540))
            .unwrap_or_else(Instant::now);
        let t = CachedToken {
            bearer: "abc".into(),
            fetched_at: past,
            ttl: Duration::from_secs(600),
        };
        assert!(!t.is_fresh(), "should refresh within 5 min margin");
    }
}
