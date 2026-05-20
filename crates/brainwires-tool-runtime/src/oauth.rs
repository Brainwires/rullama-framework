//! OAuth 2.0 middleware for tool integrations.
//!
//! Gives agents access to OAuth-protected APIs (Google, GitHub, Salesforce,
//! Slack, …) without hard-coding tokens. The framework handles:
//!
//! - Authorization Code + PKCE flow (user-delegated)
//! - Client Credentials flow (service-to-service)
//! - Automatic token refresh on expiry or `401 Unauthorized`
//! - Pluggable [`OAuthTokenStore`] for per-user token storage
//!
//! ## Example — client credentials
//!
//! ```rust,no_run
//! use brainwires_tool_runtime::oauth::{OAuthConfig, OAuthFlow, OAuthClient, InMemoryTokenStore};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = OAuthConfig::client_credentials(
//!     "https://provider.example.com/token",
//!     "my-client-id",
//!     "my-client-secret",
//!     &["read:data", "write:data"],
//! );
//!
//! let store = InMemoryTokenStore::new();
//! let client = OAuthClient::new(config, store)?;
//!
//! // Returns a valid Bearer token, refreshing if necessary.
//! let token = client.access_token("service-account").await?;
//! println!("Bearer {token}");
//! # Ok(())
//! # }
//! ```

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

// ── Token types ───────────────────────────────────────────────────────────────

/// An OAuth 2.0 token pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthToken {
    /// The bearer access token.
    pub access_token: String,
    /// Refresh token (absent for client-credentials flows that don't issue one).
    pub refresh_token: Option<String>,
    /// UTC Unix timestamp (seconds) when the access token expires.
    pub expires_at: Option<u64>,
    /// Granted scopes.
    pub scope: Option<String>,
    /// Token type (usually `"Bearer"`).
    pub token_type: String,
}

impl OAuthToken {
    /// Returns `true` if the token is known to have expired (with a 30 s buffer).
    pub fn is_expired(&self) -> bool {
        let Some(exp) = self.expires_at else {
            return false;
        };
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now + 30 >= exp
    }
}

// ── Token store ───────────────────────────────────────────────────────────────

/// Pluggable storage for OAuth tokens.
///
/// Keys are `(user_id, provider)` pairs. Implement this to persist tokens
/// across agent restarts (e.g. in a keyring, SQLite, or secrets manager).
#[async_trait]
pub trait OAuthTokenStore: Send + Sync + 'static {
    /// Retrieve a stored token.
    async fn get(&self, user_id: &str, provider: &str) -> Option<OAuthToken>;
    /// Store a token.
    async fn set(&self, user_id: &str, provider: &str, token: OAuthToken);
    /// Delete a token (e.g. on revocation).
    async fn delete(&self, user_id: &str, provider: &str);
}

/// In-memory token store — tokens are lost when the process exits.
#[derive(Clone, Default)]
pub struct InMemoryTokenStore {
    tokens: Arc<Mutex<HashMap<(String, String), OAuthToken>>>,
}

impl InMemoryTokenStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl OAuthTokenStore for InMemoryTokenStore {
    async fn get(&self, user_id: &str, provider: &str) -> Option<OAuthToken> {
        self.tokens
            .lock()
            .unwrap()
            .get(&(user_id.to_string(), provider.to_string()))
            .cloned()
    }

    async fn set(&self, user_id: &str, provider: &str, token: OAuthToken) {
        self.tokens
            .lock()
            .unwrap()
            .insert((user_id.to_string(), provider.to_string()), token);
    }

    async fn delete(&self, user_id: &str, provider: &str) {
        self.tokens
            .lock()
            .unwrap()
            .remove(&(user_id.to_string(), provider.to_string()));
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Which OAuth 2.0 grant type to use.
#[derive(Debug, Clone)]
pub enum OAuthFlow {
    /// Authorization Code + PKCE (RFC 7636) — user-delegated access.
    AuthorizationCodePkce {
        /// Authorization endpoint URL.
        auth_url: String,
        /// Token endpoint URL.
        token_url: String,
        /// Redirect URI registered with the OAuth provider.
        redirect_uri: String,
    },
    /// Client Credentials (RFC 6749 §4.4) — service-to-service.
    ClientCredentials {
        /// Token endpoint URL.
        token_url: String,
    },
    /// Refresh-only — no interactive flow; start with a pre-existing token.
    RefreshOnly {
        /// Token endpoint URL.
        token_url: String,
    },
}

/// OAuth 2.0 application configuration.
#[derive(Debug, Clone)]
pub struct OAuthConfig {
    /// Human-readable provider name (e.g. `"google"`, `"github"`).
    pub provider: String,
    /// OAuth client ID.
    pub client_id: String,
    /// OAuth client secret (omit for public clients).
    pub client_secret: Option<String>,
    /// Requested permission scopes.
    pub scopes: Vec<String>,
    /// Grant flow.
    pub flow: OAuthFlow,
    /// HTTP request timeout (default: 30 s).
    pub timeout: Duration,
}

impl OAuthConfig {
    /// Build a client-credentials config.
    pub fn client_credentials(
        token_url: impl Into<String>,
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        scopes: &[&str],
    ) -> Self {
        Self {
            provider: "custom".to_string(),
            client_id: client_id.into(),
            client_secret: Some(client_secret.into()),
            scopes: scopes.iter().map(|s| s.to_string()).collect(),
            flow: OAuthFlow::ClientCredentials {
                token_url: token_url.into(),
            },
            timeout: Duration::from_secs(30),
        }
    }

    /// Build an Authorization Code + PKCE config.
    pub fn authorization_code_pkce(
        provider: impl Into<String>,
        auth_url: impl Into<String>,
        token_url: impl Into<String>,
        redirect_uri: impl Into<String>,
        client_id: impl Into<String>,
        scopes: &[&str],
    ) -> Self {
        Self {
            provider: provider.into(),
            client_id: client_id.into(),
            client_secret: None,
            scopes: scopes.iter().map(|s| s.to_string()).collect(),
            flow: OAuthFlow::AuthorizationCodePkce {
                auth_url: auth_url.into(),
                token_url: token_url.into(),
                redirect_uri: redirect_uri.into(),
            },
            timeout: Duration::from_secs(30),
        }
    }

    /// Override the HTTP timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Override the provider name.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }
}

// ── PKCE helpers ──────────────────────────────────────────────────────────────

/// A PKCE challenge pair.
#[derive(Debug, Clone)]
pub struct PkceChallenge {
    /// The `code_verifier` to include in the token exchange request.
    pub verifier: String,
    /// The `code_challenge` to include in the authorization URL.
    pub challenge: String,
}

impl PkceChallenge {
    /// Generate a fresh PKCE challenge using SHA-256.
    pub fn new() -> Self {
        use sha2::{Digest, Sha256};

        // 32 cryptographically random bytes → base64url verifier
        let mut raw = [0u8; 32];
        getrandom::getrandom(&mut raw).expect("CSPRNG unavailable");
        let verifier = base64_url_encode(&raw);

        // SHA-256(verifier) → base64url challenge
        let mut hasher = Sha256::new();
        hasher.update(verifier.as_bytes());
        let digest = hasher.finalize();
        let challenge = base64_url_encode(&digest);

        Self {
            verifier,
            challenge,
        }
    }

    /// Build the authorization URL with PKCE parameters appended.
    pub fn authorization_url(
        &self,
        auth_url: &str,
        client_id: &str,
        redirect_uri: &str,
        scopes: &[String],
        state: &str,
    ) -> String {
        let scope = scopes.join(" ");
        format!(
            "{auth_url}?response_type=code\
             &client_id={client_id}\
             &redirect_uri={redirect_uri}\
             &scope={scope}\
             &state={state}\
             &code_challenge={}\
             &code_challenge_method=S256",
            self.challenge
        )
    }
}

impl Default for PkceChallenge {
    fn default() -> Self {
        Self::new()
    }
}

fn base64_url_encode(data: &[u8]) -> String {
    use std::fmt::Write;
    // RFC 4648 base64url without padding
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity((data.len() * 4).div_ceil(3));
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = if chunk.len() > 1 {
            chunk[1] as usize
        } else {
            0
        };
        let b2 = if chunk.len() > 2 {
            chunk[2] as usize
        } else {
            0
        };
        let _ = write!(out, "{}", CHARS[(b0 >> 2) & 63] as char);
        let _ = write!(out, "{}", CHARS[((b0 << 4) | (b1 >> 4)) & 63] as char);
        if chunk.len() > 1 {
            let _ = write!(out, "{}", CHARS[((b1 << 2) | (b2 >> 6)) & 63] as char);
        }
        if chunk.len() > 2 {
            let _ = write!(out, "{}", CHARS[b2 & 63] as char);
        }
    }
    out
}

// ── OAuthClient ───────────────────────────────────────────────────────────────

/// OAuth 2.0 client that manages tokens on behalf of users.
///
/// Wrap this inside a tool implementation to produce a fresh Bearer token
/// for every API call, automatically refreshing when the stored token expires.
pub struct OAuthClient<S: OAuthTokenStore> {
    config: OAuthConfig,
    store: S,
    http: Client,
}

impl<S: OAuthTokenStore> OAuthClient<S> {
    /// Build a client from config and token store.
    pub fn new(config: OAuthConfig, store: S) -> anyhow::Result<Self> {
        let http = Client::builder().timeout(config.timeout).build()?;
        Ok(Self {
            config,
            store,
            http,
        })
    }

    /// Return a valid access token for `user_id`, refreshing or fetching as needed.
    ///
    /// - If a non-expired token is in the store, it is returned immediately.
    /// - If the token is expired and a refresh token is available, it is refreshed.
    /// - If no token exists and the flow is `ClientCredentials`, a new token is fetched.
    /// - Otherwise returns `Err` — the caller must initiate an interactive auth flow.
    pub async fn access_token(&self, user_id: &str) -> anyhow::Result<String> {
        // 1. Check store
        if let Some(token) = self.store.get(user_id, &self.config.provider).await {
            if !token.is_expired() {
                return Ok(token.access_token.clone());
            }
            // Try to refresh
            if let Some(refresh_token) = &token.refresh_token
                && let Ok(refreshed) = self.refresh_token(refresh_token).await
            {
                self.store
                    .set(user_id, &self.config.provider, refreshed.clone())
                    .await;
                return Ok(refreshed.access_token);
                // Refresh failed — fall through to re-auth
            }
        }

        // 2. Client credentials can fetch without user interaction
        if let OAuthFlow::ClientCredentials { .. } = &self.config.flow {
            let token = self.fetch_client_credentials().await?;
            self.store
                .set(user_id, &self.config.provider, token.clone())
                .await;
            return Ok(token.access_token);
        }

        anyhow::bail!(
            "No valid token for user '{}' on provider '{}'. \
             Initiate an authorization flow first via OAuthClient::authorization_url().",
            user_id,
            self.config.provider
        )
    }

    /// Store a token that was obtained through an external interactive flow.
    pub async fn store_token(&self, user_id: &str, token: OAuthToken) {
        self.store.set(user_id, &self.config.provider, token).await;
    }

    /// Delete the stored token for a user (e.g. on sign-out or revocation).
    pub async fn revoke(&self, user_id: &str) {
        self.store.delete(user_id, &self.config.provider).await;
    }

    /// Exchange an authorization code for tokens (Authorization Code + PKCE).
    ///
    /// Call this after the user is redirected back to your `redirect_uri` with
    /// a `code` parameter.
    pub async fn exchange_code(&self, code: &str, verifier: &str) -> anyhow::Result<OAuthToken> {
        let token_url = match &self.config.flow {
            OAuthFlow::AuthorizationCodePkce {
                token_url,
                redirect_uri,
                ..
            } => (token_url.clone(), Some(redirect_uri.clone())),
            _ => anyhow::bail!("exchange_code requires AuthorizationCodePkce flow"),
        };

        let mut params = vec![
            ("grant_type", "authorization_code".to_string()),
            ("code", code.to_string()),
            ("client_id", self.config.client_id.clone()),
            ("code_verifier", verifier.to_string()),
        ];
        if let Some(uri) = token_url.1 {
            params.push(("redirect_uri", uri));
        }
        if let Some(secret) = &self.config.client_secret {
            params.push(("client_secret", secret.clone()));
        }

        self.post_token(&token_url.0, &params).await
    }

    /// Build a PKCE authorization URL for the user to visit.
    ///
    /// Returns `(url, pkce_challenge)` — store the `challenge.verifier` so you
    /// can pass it to [`exchange_code`] when the callback arrives.
    pub fn authorization_url(&self, state: &str) -> anyhow::Result<(String, PkceChallenge)> {
        match &self.config.flow {
            OAuthFlow::AuthorizationCodePkce {
                auth_url,
                redirect_uri,
                ..
            } => {
                let pkce = PkceChallenge::new();
                let url = pkce.authorization_url(
                    auth_url,
                    &self.config.client_id,
                    redirect_uri,
                    &self.config.scopes,
                    state,
                );
                Ok((url, pkce))
            }
            _ => anyhow::bail!("authorization_url requires AuthorizationCodePkce flow"),
        }
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    async fn fetch_client_credentials(&self) -> anyhow::Result<OAuthToken> {
        let token_url = match &self.config.flow {
            OAuthFlow::ClientCredentials { token_url } => token_url.clone(),
            _ => anyhow::bail!("fetch_client_credentials called on non-ClientCredentials flow"),
        };

        let mut params = vec![
            ("grant_type", "client_credentials".to_string()),
            ("client_id", self.config.client_id.clone()),
        ];
        if !self.config.scopes.is_empty() {
            params.push(("scope", self.config.scopes.join(" ")));
        }
        if let Some(secret) = &self.config.client_secret {
            params.push(("client_secret", secret.clone()));
        }

        self.post_token(&token_url, &params).await
    }

    async fn refresh_token(&self, refresh_token: &str) -> anyhow::Result<OAuthToken> {
        let token_url = match &self.config.flow {
            OAuthFlow::AuthorizationCodePkce { token_url, .. } => token_url.clone(),
            OAuthFlow::RefreshOnly { token_url } => token_url.clone(),
            OAuthFlow::ClientCredentials { token_url } => token_url.clone(),
        };

        let mut params = vec![
            ("grant_type", "refresh_token".to_string()),
            ("refresh_token", refresh_token.to_string()),
            ("client_id", self.config.client_id.clone()),
        ];
        if let Some(secret) = &self.config.client_secret {
            params.push(("client_secret", secret.clone()));
        }

        self.post_token(&token_url, &params).await
    }

    async fn post_token(&self, url: &str, params: &[(&str, String)]) -> anyhow::Result<OAuthToken> {
        let resp = self
            .http
            .post(url)
            .form(params)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Token request failed: {e}"))?;

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("Token endpoint returned {status}: {body}");
        }

        let raw: TokenResponse =
            serde_json::from_str(&body).map_err(|e| anyhow::anyhow!("Token parse error: {e}"))?;

        let expires_at = raw.expires_in.map(|secs| {
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                + secs
        });

        Ok(OAuthToken {
            access_token: raw.access_token,
            refresh_token: raw.refresh_token,
            expires_at,
            scope: raw.scope,
            token_type: raw.token_type.unwrap_or_else(|| "Bearer".to_string()),
        })
    }
}

/// Raw OAuth token endpoint response.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
    scope: Option<String>,
    token_type: Option<String>,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_challenge_base64url_no_padding() {
        let pkce = PkceChallenge::new();
        assert!(!pkce.verifier.contains('='));
        assert!(!pkce.challenge.contains('='));
        assert!(!pkce.verifier.contains('+'));
        assert!(!pkce.challenge.contains('+'));
        assert!(!pkce.verifier.contains('/'));
        assert!(!pkce.challenge.contains('/'));
    }

    #[test]
    fn pkce_authorization_url_contains_required_params() {
        let pkce = PkceChallenge::new();
        let url = pkce.authorization_url(
            "https://auth.example.com/authorize",
            "client-abc",
            "https://myapp.example.com/callback",
            &["openid".to_string(), "profile".to_string()],
            "random-state",
        );
        assert!(url.contains("response_type=code"));
        assert!(url.contains("client_id=client-abc"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains(&pkce.challenge));
        assert!(url.contains("state=random-state"));
    }

    #[test]
    fn token_not_expired_without_expiry() {
        let t = OAuthToken {
            access_token: "tok".to_string(),
            refresh_token: None,
            expires_at: None,
            scope: None,
            token_type: "Bearer".to_string(),
        };
        assert!(!t.is_expired());
    }

    #[test]
    fn token_expired_in_past() {
        let t = OAuthToken {
            access_token: "tok".to_string(),
            refresh_token: None,
            expires_at: Some(1), // way in the past
            scope: None,
            token_type: "Bearer".to_string(),
        };
        assert!(t.is_expired());
    }

    #[test]
    fn in_memory_store_operations() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        rt.block_on(async {
            let store = InMemoryTokenStore::new();
            let token = OAuthToken {
                access_token: "abc".to_string(),
                refresh_token: None,
                expires_at: None,
                scope: None,
                token_type: "Bearer".to_string(),
            };
            store.set("user1", "github", token.clone()).await;
            let fetched = store.get("user1", "github").await.unwrap();
            assert_eq!(fetched.access_token, "abc");

            store.delete("user1", "github").await;
            assert!(store.get("user1", "github").await.is_none());
        });
    }

    #[test]
    fn config_client_credentials_builder() {
        let cfg = OAuthConfig::client_credentials(
            "https://token.example.com",
            "id",
            "secret",
            &["read", "write"],
        );
        assert_eq!(cfg.scopes, vec!["read", "write"]);
        matches!(cfg.flow, OAuthFlow::ClientCredentials { .. });
    }
}
