//! Client-credentials OAuth minter for the Bot Framework.
//!
//! Azure AD issues access tokens scoped to
//! `https://api.botframework.com/.default` when the bot presents its
//! `client_id` + `client_secret`. Tokens typically live 1h; we cache
//! until 5 minutes before expiry.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use parking_lot::RwLock;
use serde::Deserialize;

const REFRESH_MARGIN: Duration = Duration::from_secs(5 * 60);

/// Default Microsoft login endpoint, templated with tenant id.
pub const MS_LOGIN_BASE: &str = "https://login.microsoftonline.com";

/// Bot Framework OAuth scope.
pub const BOT_SCOPE: &str = "https://api.botframework.com/.default";

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
    #[serde(default)]
    token_type: Option<String>,
}

#[derive(Clone)]
struct CachedToken {
    bearer: String,
    fetched_at: Instant,
    ttl: Duration,
}

impl CachedToken {
    fn fresh(&self) -> bool {
        self.fetched_at.elapsed() + REFRESH_MARGIN < self.ttl
    }
}

/// Bearer minter for the Bot Framework.
pub struct BotTokenMinter {
    app_id: String,
    app_password: String,
    tenant_id: String,
    http: reqwest::Client,
    cache: Arc<RwLock<Option<CachedToken>>>,
    login_base: String,
}

impl BotTokenMinter {
    /// Construct a minter scoped to one app registration.
    pub fn new(
        app_id: impl Into<String>,
        app_password: impl Into<String>,
        tenant_id: impl Into<String>,
    ) -> Self {
        Self {
            app_id: app_id.into(),
            app_password: app_password.into(),
            tenant_id: tenant_id.into(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client"),
            cache: Arc::new(RwLock::new(None)),
            login_base: MS_LOGIN_BASE.to_string(),
        }
    }

    /// Override the Microsoft login base URL — tests only.
    pub fn with_login_base(mut self, base: impl Into<String>) -> Self {
        self.login_base = base.into();
        self
    }

    /// Inject a custom HTTP client — tests only.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// Fetch (or reuse) a bearer token.
    pub async fn bearer(&self) -> Result<String> {
        if let Some(t) = self.cache.read().as_ref()
            && t.fresh()
        {
            return Ok(t.bearer.clone());
        }
        let fresh = self.mint_new().await?;
        let bearer = fresh.bearer.clone();
        *self.cache.write() = Some(fresh);
        Ok(bearer)
    }

    fn token_url(&self) -> String {
        format!(
            "{}/{}/oauth2/v2.0/token",
            self.login_base.trim_end_matches('/'),
            self.tenant_id
        )
    }

    async fn mint_new(&self) -> Result<CachedToken> {
        let form = [
            ("grant_type", "client_credentials"),
            ("client_id", self.app_id.as_str()),
            ("client_secret", self.app_password.as_str()),
            ("scope", BOT_SCOPE),
        ];
        let resp = self
            .http
            .post(self.token_url())
            .form(&form)
            .send()
            .await
            .context("POST Microsoft token endpoint")?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!(
                "Microsoft token endpoint returned {}: {} bytes",
                status,
                body.len()
            );
        }
        let parsed: TokenResponse = resp.json().await.context("parse token response")?;
        if parsed.access_token.is_empty() {
            return Err(anyhow!("empty access_token"));
        }
        if let Some(tt) = &parsed.token_type
            && !tt.eq_ignore_ascii_case("Bearer")
        {
            bail!("unexpected token_type '{tt}'");
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

    #[test]
    fn token_url_uses_tenant() {
        let m = BotTokenMinter::new("id", "secret", "abc-123");
        assert_eq!(
            m.token_url(),
            "https://login.microsoftonline.com/abc-123/oauth2/v2.0/token"
        );
    }

    #[test]
    fn token_url_uses_custom_base() {
        let m = BotTokenMinter::new("id", "secret", "common").with_login_base("https://local.test");
        assert_eq!(m.token_url(), "https://local.test/common/oauth2/v2.0/token");
    }

    #[test]
    fn cached_token_freshness() {
        let t = CachedToken {
            bearer: "b".into(),
            fetched_at: Instant::now(),
            ttl: Duration::from_secs(3600),
        };
        assert!(t.fresh());

        let past = Instant::now()
            .checked_sub(Duration::from_secs(3600))
            .unwrap_or_else(Instant::now);
        let t = CachedToken {
            bearer: "b".into(),
            fetched_at: past,
            ttl: Duration::from_secs(3600),
        };
        assert!(!t.fresh());
    }
}
