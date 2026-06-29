//! Auth token forwarding/validation middleware.

use crate::error::ProxyResult;
use crate::middleware::{LayerAction, ProxyLayer};
use crate::types::{ProxyRequest, ProxyResponse};
use http::StatusCode;
use http::header::{AUTHORIZATION, HeaderValue};

/// Strategy for handling authentication tokens.
pub enum AuthStrategy {
    /// Forward a static bearer token to upstream.
    StaticBearer(String),
    /// Pass through the client's Authorization header unchanged.
    Passthrough,
    /// Require a specific bearer token from the client; reject mismatches.
    Validate(String),
    /// Strip the Authorization header before forwarding.
    Strip,
}

/// Auth middleware that manages Authorization headers.
pub struct AuthLayer {
    strategy: AuthStrategy,
}

impl AuthLayer {
    pub fn new(strategy: AuthStrategy) -> Self {
        Self { strategy }
    }

    /// Create an auth layer that injects a static bearer token.
    pub fn static_bearer(token: impl Into<String>) -> Self {
        Self::new(AuthStrategy::StaticBearer(token.into()))
    }

    /// Create an auth layer that passes through client auth.
    pub fn passthrough() -> Self {
        Self::new(AuthStrategy::Passthrough)
    }

    /// Create an auth layer that validates a required token.
    pub fn validate(expected: impl Into<String>) -> Self {
        Self::new(AuthStrategy::Validate(expected.into()))
    }

    /// Create an auth layer that strips auth headers.
    pub fn strip() -> Self {
        Self::new(AuthStrategy::Strip)
    }
}

#[async_trait::async_trait]
impl ProxyLayer for AuthLayer {
    async fn on_request(&self, mut request: ProxyRequest) -> ProxyResult<LayerAction> {
        match &self.strategy {
            AuthStrategy::StaticBearer(token) => {
                let value = HeaderValue::from_str(&format!("Bearer {token}"))
                    .map_err(|e| crate::error::ProxyError::Config(e.to_string()))?;
                request.headers.insert(AUTHORIZATION, value);
                Ok(LayerAction::Forward(request))
            }
            AuthStrategy::Passthrough => Ok(LayerAction::Forward(request)),
            AuthStrategy::Validate(expected) => {
                let expected_val = format!("Bearer {expected}");
                match request.headers.get(AUTHORIZATION) {
                    Some(val) if val.as_bytes() == expected_val.as_bytes() => {
                        Ok(LayerAction::Forward(request))
                    }
                    _ => {
                        tracing::warn!(request_id = %request.id, "auth validation failed");
                        Ok(LayerAction::Respond(
                            ProxyResponse::for_request(request.id, StatusCode::UNAUTHORIZED)
                                .with_body("Unauthorized"),
                        ))
                    }
                }
            }
            AuthStrategy::Strip => {
                request.headers.remove(AUTHORIZATION);
                Ok(LayerAction::Forward(request))
            }
        }
    }

    fn name(&self) -> &str {
        "auth"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::Method;

    fn make_request() -> ProxyRequest {
        ProxyRequest::new(Method::GET, "/api".parse().unwrap())
    }

    fn make_request_with_auth(token: &str) -> ProxyRequest {
        let mut req = make_request();
        req.headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
        req
    }

    #[tokio::test]
    async fn static_bearer_injects_token() {
        let layer = AuthLayer::static_bearer("sk-test-123");
        let result = layer.on_request(make_request()).await.unwrap();
        match result {
            LayerAction::Forward(req) => {
                assert_eq!(
                    req.headers.get(AUTHORIZATION).unwrap(),
                    "Bearer sk-test-123"
                );
            }
            _ => panic!("expected forward"),
        }
    }

    #[tokio::test]
    async fn passthrough_preserves_header() {
        let layer = AuthLayer::passthrough();
        let req = make_request_with_auth("my-token");
        let result = layer.on_request(req).await.unwrap();
        match result {
            LayerAction::Forward(req) => {
                assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer my-token");
            }
            _ => panic!("expected forward"),
        }
    }

    #[tokio::test]
    async fn validate_accepts_correct_token() {
        let layer = AuthLayer::validate("valid-token");
        let req = make_request_with_auth("valid-token");
        let result = layer.on_request(req).await.unwrap();
        assert!(matches!(result, LayerAction::Forward(_)));
    }

    #[tokio::test]
    async fn validate_rejects_wrong_token() {
        let layer = AuthLayer::validate("valid-token");
        let req = make_request_with_auth("wrong-token");
        let result = layer.on_request(req).await.unwrap();
        match result {
            LayerAction::Respond(resp) => {
                assert_eq!(resp.status, StatusCode::UNAUTHORIZED);
            }
            _ => panic!("expected reject"),
        }
    }

    #[tokio::test]
    async fn validate_rejects_missing_header() {
        let layer = AuthLayer::validate("valid-token");
        let result = layer.on_request(make_request()).await.unwrap();
        match result {
            LayerAction::Respond(resp) => {
                assert_eq!(resp.status, StatusCode::UNAUTHORIZED);
            }
            _ => panic!("expected reject"),
        }
    }

    #[tokio::test]
    async fn strip_removes_auth_header() {
        let layer = AuthLayer::strip();
        let req = make_request_with_auth("remove-me");
        let result = layer.on_request(req).await.unwrap();
        match result {
            LayerAction::Forward(req) => {
                assert!(req.headers.get(AUTHORIZATION).is_none());
            }
            _ => panic!("expected forward"),
        }
    }
}
