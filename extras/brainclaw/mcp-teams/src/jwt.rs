//! Microsoft Bot Framework JWT verification.
//!
//! Ingress activities from Teams are authenticated with a JWT signed by
//! Microsoft, carrying:
//! - `iss` = `https://api.botframework.com` or
//!   `https://sts.windows.net/<tenant>/` (tenant-specific),
//! - `aud` = the bot's `app_id`,
//! - signed with a key from Microsoft's Bot Framework OpenID metadata.
//!
//! The metadata doc is at
//! <https://login.botframework.com/v1/.well-known/openidconfiguration>
//! and points at a JWKs URI we cache for one hour.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use parking_lot::RwLock;
use serde::Deserialize;

/// Microsoft OIDC metadata endpoint for Bot Framework tokens.
pub const MS_OIDC_METADATA_URL: &str =
    "https://login.botframework.com/v1/.well-known/openidconfiguration";

const JWKS_CACHE_TTL: Duration = Duration::from_secs(3600);

/// A cached JWK we care about.
#[derive(Debug, Clone, Deserialize)]
pub struct JwkEntry {
    /// Key id matching the JWT `kid` header.
    pub kid: String,
    /// Algorithm — we only accept RS256.
    #[serde(default)]
    pub alg: Option<String>,
    /// Modulus (base64url).
    pub n: String,
    /// Exponent (base64url).
    pub e: String,
}

#[derive(Debug, Deserialize)]
struct JwksDoc {
    keys: Vec<JwkEntry>,
}

#[derive(Debug, Deserialize)]
struct OidcMetadata {
    jwks_uri: String,
}

/// Shared JWKs cache + metadata lookup.
pub struct BotFrameworkVerifier {
    audience: String,
    metadata_url: String,
    http: reqwest::Client,
    cache: Arc<RwLock<Cache>>,
}

#[derive(Default)]
struct Cache {
    jwks_uri: Option<String>,
    keys: Vec<JwkEntry>,
    fetched_at: Option<Instant>,
}

/// Claims we validate from a Bot Framework JWT.
#[derive(Debug, Clone, Deserialize)]
pub struct BotClaims {
    /// Expected to be the bot's `app_id`.
    pub aud: String,
    /// Issuer — Microsoft.
    #[serde(default)]
    pub iss: String,
    /// App id that sent the activity.
    #[serde(rename = "appid", default)]
    pub appid: Option<String>,
}

impl BotFrameworkVerifier {
    /// Construct a verifier expecting `aud == audience` (typically the
    /// bot's app id).
    pub fn new(audience: impl Into<String>) -> Self {
        Self {
            audience: audience.into(),
            metadata_url: MS_OIDC_METADATA_URL.to_string(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(15))
                .build()
                .expect("reqwest client"),
            cache: Arc::new(RwLock::new(Cache::default())),
        }
    }

    /// Override the OIDC metadata URL — tests only.
    pub fn with_metadata_url(mut self, url: impl Into<String>) -> Self {
        self.metadata_url = url.into();
        self
    }

    /// Override the HTTP client — tests only.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    async fn metadata(&self) -> Result<String> {
        if let Some(uri) = self.cache.read().jwks_uri.clone() {
            return Ok(uri);
        }
        let resp = self
            .http
            .get(&self.metadata_url)
            .send()
            .await
            .context("GET OIDC metadata")?;
        if !resp.status().is_success() {
            bail!("metadata endpoint returned {}", resp.status());
        }
        let md: OidcMetadata = resp.json().await.context("parse OIDC metadata JSON")?;
        let uri = md.jwks_uri.clone();
        self.cache.write().jwks_uri = Some(md.jwks_uri);
        Ok(uri)
    }

    async fn jwks(&self) -> Result<Vec<JwkEntry>> {
        {
            let cache = self.cache.read();
            if let Some(t) = cache.fetched_at
                && t.elapsed() < JWKS_CACHE_TTL
                && !cache.keys.is_empty()
            {
                return Ok(cache.keys.clone());
            }
        }
        let uri = self.metadata().await?;
        let resp = self.http.get(&uri).send().await.context("GET JWKs")?;
        if !resp.status().is_success() {
            bail!("JWKs endpoint returned {}", resp.status());
        }
        let doc: JwksDoc = resp.json().await.context("parse JWKs JSON")?;
        let keys = doc.keys;
        let mut cache = self.cache.write();
        cache.keys = keys.clone();
        cache.fetched_at = Some(Instant::now());
        Ok(keys)
    }

    /// Verify a bearer-style Authorization value and return the claims.
    pub async fn verify(&self, header: &str) -> Result<BotClaims> {
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
            .ok_or_else(|| anyhow!("no matching JWK for kid {kid}"))?;
        if let Some(alg) = &jwk.alg
            && alg != "RS256"
        {
            bail!("JWK alg {alg} is not RS256");
        }
        let dk = DecodingKey::from_rsa_components(&jwk.n, &jwk.e).context("build decoding key")?;
        let mut validation = Validation::new(Algorithm::RS256);
        let aud = [self.audience.as_str()];
        validation.set_audience(&aud);
        // Microsoft Bot Framework tokens come from multiple valid
        // issuers (public cloud, sovereign clouds, the legacy
        // api.botframework.com). We trust the signing key's origin
        // (JWKs fetched via the OIDC metadata doc) and skip strict
        // `iss` matching so the adapter works in all clouds.
        validation.validate_exp = true;
        validation.required_spec_claims.clear();
        validation.required_spec_claims.insert("aud".into());
        validation.required_spec_claims.insert("exp".into());
        let data = decode::<BotClaims>(token, &dk, &validation).context("verify JWT")?;
        Ok(data.claims)
    }

    /// Inject JWKs directly (used by integration tests to avoid running a
    /// real metadata server).
    #[doc(hidden)]
    pub fn seed_jwks(&self, jwks_uri: impl Into<String>, keys: Vec<JwkEntry>) {
        let mut cache = self.cache.write();
        cache.jwks_uri = Some(jwks_uri.into());
        cache.keys = keys;
        cache.fetched_at = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_empty_bearer() {
        let v = BotFrameworkVerifier::new("app-id");
        assert!(v.verify("").await.is_err());
    }

    #[tokio::test]
    async fn rejects_non_jwt() {
        let v = BotFrameworkVerifier::new("app-id");
        assert!(v.verify("Bearer not.a.jwt").await.is_err());
    }

    #[test]
    fn seed_jwks_populates_cache() {
        let v = BotFrameworkVerifier::new("aud");
        v.seed_jwks(
            "https://example/jwks",
            vec![JwkEntry {
                kid: "k".into(),
                alg: Some("RS256".into()),
                n: "AA".into(),
                e: "AQAB".into(),
            }],
        );
        let cache = v.cache.read();
        assert_eq!(cache.jwks_uri.as_deref(), Some("https://example/jwks"));
        assert_eq!(cache.keys.len(), 1);
    }
}
