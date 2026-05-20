//! OAuth 2.1 JWT validation middleware for MCP servers.
//!
//! Validates `Authorization: Bearer <jwt>` tokens on incoming requests using
//! HMAC-SHA256 (HS256) shared-secret or RSA public-key (RS256) validation.
//!
//! # Token delivery
//! The MCP JSON-RPC layer does not carry HTTP headers, so this middleware
//! expects the raw JWT in `params._bearer_token`. Axum middleware should
//! copy the `Authorization: Bearer <token>` header value into that field
//! before handing the request to the MCP event loop.
//!
//! # Feature flag
//! Only compiled when the `oauth` feature is enabled.

use async_trait::async_trait;
use brainwires_mcp_client::{JsonRpcError, JsonRpcRequest};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode};

use super::{Middleware, MiddlewareResult};
use crate::connection::RequestContext;

/// OAuth 2.1 JWT validation middleware.
///
/// Validates `Authorization: Bearer <token>` JWTs using the provided decoding
/// key. Optionally enforces `iss` (issuer) and `aud` (audience) claims.
pub struct OAuthMiddleware {
    decoding_key: DecodingKey,
    validation: Validation,
}

impl OAuthMiddleware {
    /// Create an HS256 middleware from a shared HMAC secret.
    pub fn with_secret(secret: &[u8]) -> Self {
        Self {
            decoding_key: DecodingKey::from_secret(secret),
            validation: Validation::new(Algorithm::HS256),
        }
    }

    /// Create an RS256 middleware from an RSA public key PEM string.
    pub fn with_rsa_pem(pem: &str) -> Result<Self, jsonwebtoken::errors::Error> {
        Ok(Self {
            decoding_key: DecodingKey::from_rsa_pem(pem.as_bytes())?,
            validation: Validation::new(Algorithm::RS256),
        })
    }

    /// Require `iss` claim to equal the given issuer.
    pub fn require_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.validation.set_issuer(&[issuer.into()]);
        self
    }

    /// Require `aud` claim to contain the given audience.
    pub fn require_audience(mut self, audience: impl Into<String>) -> Self {
        self.validation.set_audience(&[audience.into()]);
        self
    }

    fn reject(msg: &str) -> MiddlewareResult {
        MiddlewareResult::Reject(JsonRpcError {
            code: -32003,
            message: format!("Unauthorized: {}", msg),
            data: None,
        })
    }
}

#[async_trait]
impl Middleware for OAuthMiddleware {
    async fn process_request(
        &self,
        request: &JsonRpcRequest,
        ctx: &mut RequestContext,
    ) -> MiddlewareResult {
        // `initialize` is unauthenticated per MCP spec
        if request.method == "initialize" {
            return MiddlewareResult::Continue;
        }

        // Fast path: token already validated this session
        if ctx.get_metadata("oauth_validated").is_some() {
            return MiddlewareResult::Continue;
        }

        // Extract bearer token from params._bearer_token
        let token = match request
            .params
            .as_ref()
            .and_then(|p| p.get("_bearer_token"))
            .and_then(|v| v.as_str())
        {
            Some(t) => t,
            None => return Self::reject("missing _bearer_token in params"),
        };

        // Validate signature, expiry, and any configured claims
        match decode::<serde_json::Value>(token, &self.decoding_key, &self.validation) {
            Ok(_) => {
                ctx.set_metadata("oauth_validated".to_string(), serde_json::Value::Bool(true));
                MiddlewareResult::Continue
            }
            Err(e) => Self::reject(&e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::{EncodingKey, Header, encode};
    use serde_json::json;

    fn make_token(secret: &[u8]) -> String {
        let claims = json!({ "sub": "test", "exp": 9999999999u64 });
        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(secret),
        )
        .unwrap()
    }

    fn make_request(method: &str, token: Option<&str>) -> JsonRpcRequest {
        let params = token.map(|t| json!({ "_bearer_token": t }));
        JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: json!(1),
            method: method.to_string(),
            params,
        }
    }

    #[tokio::test]
    async fn valid_jwt_passes() {
        let secret = b"supersecret";
        let mw = OAuthMiddleware::with_secret(secret);
        let req = make_request("tools/call", Some(&make_token(secret)));
        let mut ctx = RequestContext::new(json!(1));
        assert!(matches!(
            mw.process_request(&req, &mut ctx).await,
            MiddlewareResult::Continue
        ));
    }

    #[tokio::test]
    async fn missing_token_rejects() {
        let mw = OAuthMiddleware::with_secret(b"secret");
        let req = make_request("tools/call", None);
        let mut ctx = RequestContext::new(json!(1));
        assert!(matches!(
            mw.process_request(&req, &mut ctx).await,
            MiddlewareResult::Reject(_)
        ));
    }

    #[tokio::test]
    async fn wrong_secret_rejects() {
        let token = make_token(b"correct_secret");
        let mw = OAuthMiddleware::with_secret(b"wrong_secret");
        let req = make_request("tools/call", Some(&token));
        let mut ctx = RequestContext::new(json!(1));
        assert!(matches!(
            mw.process_request(&req, &mut ctx).await,
            MiddlewareResult::Reject(_)
        ));
    }

    #[tokio::test]
    async fn initialize_skips_auth() {
        let mw = OAuthMiddleware::with_secret(b"secret");
        let req = make_request("initialize", None);
        let mut ctx = RequestContext::new(json!(1));
        assert!(matches!(
            mw.process_request(&req, &mut ctx).await,
            MiddlewareResult::Continue
        ));
    }

    #[tokio::test]
    async fn validated_token_cached_in_context() {
        let secret = b"supersecret";
        let mw = OAuthMiddleware::with_secret(secret);
        let mut ctx = RequestContext::new(json!(1));

        // First call validates and caches
        mw.process_request(
            &make_request("tools/call", Some(&make_token(secret))),
            &mut ctx,
        )
        .await;

        // Second call uses cached result — no token required
        assert!(matches!(
            mw.process_request(&make_request("tools/list", None), &mut ctx)
                .await,
            MiddlewareResult::Continue
        ));
    }
}
