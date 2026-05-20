//! Tenant-access-token minter for Feishu / Lark.
//!
//! Feishu issues tenant access tokens valid for ~2 hours in exchange
//! for `(app_id, app_secret)`. We cache until 5 minutes before expiry.
//!
//! Endpoint: `POST /open-apis/auth/v3/tenant_access_token/internal`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use parking_lot::RwLock;
use serde::Deserialize;
use serde_json::json;

const REFRESH_MARGIN: Duration = Duration::from_secs(5 * 60);

/// Default Feishu open-platform base.
pub const FEISHU_BASE: &str = "https://open.feishu.cn";

#[derive(Debug, Deserialize)]
struct TokenResp {
    code: i64,
    #[serde(default)]
    msg: Option<String>,
    #[serde(default)]
    tenant_access_token: Option<String>,
    #[serde(default)]
    expire: Option<u64>,
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

/// Bearer minter for the Feishu Open Platform.
pub struct TenantTokenMinter {
    app_id: String,
    app_secret: String,
    http: reqwest::Client,
    cache: Arc<RwLock<Option<CachedToken>>>,
    base_url: String,
}

impl TenantTokenMinter {
    /// Construct a new minter for one app registration.
    pub fn new(app_id: impl Into<String>, app_secret: impl Into<String>) -> Self {
        Self {
            app_id: app_id.into(),
            app_secret: app_secret.into(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client"),
            cache: Arc::new(RwLock::new(None)),
            base_url: FEISHU_BASE.to_string(),
        }
    }

    /// Override the Feishu base URL — tests only.
    pub fn with_base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into();
        self
    }

    /// Inject a custom HTTP client — tests only.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// Return a fresh bearer, minting a new token if needed.
    pub async fn bearer(&self) -> Result<String> {
        if let Some(t) = self.cache.read().clone()
            && t.fresh()
        {
            return Ok(t.bearer);
        }
        self.refresh().await
    }

    async fn refresh(&self) -> Result<String> {
        let url = format!(
            "{}/open-apis/auth/v3/tenant_access_token/internal",
            self.base_url.trim_end_matches('/')
        );
        let body = json!({
            "app_id": self.app_id,
            "app_secret": self.app_secret,
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("mint Feishu tenant token")?;
        if !resp.status().is_success() {
            bail!("Feishu token endpoint returned {}", resp.status());
        }
        let parsed: TokenResp = resp.json().await.context("parse Feishu token response")?;
        if parsed.code != 0 {
            bail!(
                "Feishu token endpoint returned code={} msg={:?}",
                parsed.code,
                parsed.msg
            );
        }
        let bearer = parsed
            .tenant_access_token
            .ok_or_else(|| anyhow::anyhow!("Feishu response missing tenant_access_token"))?;
        let ttl = Duration::from_secs(parsed.expire.unwrap_or(7_200));
        *self.cache.write() = Some(CachedToken {
            bearer: bearer.clone(),
            fetched_at: Instant::now(),
            ttl,
        });
        Ok(bearer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_logic_excludes_near_expiry() {
        let t = CachedToken {
            bearer: "b".into(),
            fetched_at: Instant::now() - Duration::from_secs(7_100),
            ttl: Duration::from_secs(7_200),
        };
        // 7100s elapsed + 300s margin = 7400s > 7200s TTL → not fresh.
        assert!(!t.fresh());
    }

    #[test]
    fn minter_constructs_without_panicking() {
        let _ = TenantTokenMinter::new("cli_a", "sec").with_base_url("http://localhost:1");
    }
}
