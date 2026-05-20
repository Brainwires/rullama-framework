//! Webhook handler for HTTP-based channel integrations.

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use hmac::{Hmac, Mac};
use sha2::Sha256;

use brainwires_network::channels::events::ChannelEvent;

use crate::state::AppState;

type HmacSha256 = Hmac<Sha256>;

/// Verify the HMAC-SHA256 signature on a webhook request body.
///
/// If `webhook_secret` is `None`, all requests are allowed.
fn verify_webhook_signature(
    headers: &HeaderMap,
    body: &[u8],
    secret: &Option<String>,
) -> Result<(), StatusCode> {
    let secret = match secret {
        Some(s) => s,
        None => return Ok(()), // no secret configured — allow all
    };

    let signature = headers
        .get("x-webhook-signature")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let expected_bytes = match hex::decode(signature) {
        Ok(b) => b,
        Err(_) => return Err(StatusCode::UNAUTHORIZED),
    };

    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC can take key of any size");
    mac.update(body);

    mac.verify_slice(&expected_bytes)
        .map_err(|_| StatusCode::UNAUTHORIZED)
}

/// Handle an incoming webhook POST request.
///
/// If a `webhook_secret` is configured, the request must include an
/// `X-Webhook-Signature` header containing the hex-encoded HMAC-SHA256
/// of the request body.
///
/// Parses the JSON payload as a `ChannelEvent` and routes it through the
/// message router. Returns 200 OK on success or an appropriate error.
pub async fn handle_webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    if !state.config.webhook_enabled {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "Webhook endpoint is disabled" })),
        );
    }

    // Verify HMAC signature if configured
    if let Err(status) = verify_webhook_signature(&headers, &body, &state.config.webhook_secret) {
        tracing::warn!("Webhook signature verification failed");
        return (
            status,
            Json(serde_json::json!({ "error": "Invalid or missing webhook signature" })),
        );
    }

    // Parse payload
    let event: ChannelEvent = match serde_json::from_slice(&body) {
        Ok(event) => event,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to parse webhook payload as ChannelEvent");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "Invalid payload",
                    "details": e.to_string()
                })),
            );
        }
    };

    // Use a synthetic channel ID of all-zeros for webhook-sourced events
    let webhook_channel_id = uuid::Uuid::nil();

    if let Err(e) = state
        .router
        .handle_inbound(webhook_channel_id, &event)
        .await
    {
        tracing::error!(error = %e, "Failed to route webhook event");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "Failed to process event",
                "details": e.to_string()
            })),
        );
    }

    (StatusCode::OK, Json(serde_json::json!({ "status": "ok" })))
}
