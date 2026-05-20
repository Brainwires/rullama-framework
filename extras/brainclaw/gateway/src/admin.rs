//! Admin API handlers for gateway monitoring and control.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::channel_registry::ChannelInfo;
use crate::config::GatewayConfig;
use crate::cron::CronJob;
use crate::identity::PlatformIdentity;
use crate::state::AppState;

/// Verify the admin bearer token from the `Authorization` header.
///
/// If `admin_token` is `None` in the config, all requests are allowed (backward
/// compatible). Otherwise the request must carry `Authorization: Bearer <token>`
/// matching the configured value.
pub fn check_admin_auth(headers: &HeaderMap, config: &GatewayConfig) -> Result<(), StatusCode> {
    let expected = match &config.admin_token {
        Some(token) => token,
        None => return Ok(()), // no token configured — open access
    };

    let auth_header = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let provided = auth_header.strip_prefix("Bearer ").unwrap_or("");

    if provided == expected.as_str() {
        Ok(())
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Server status.
    pub status: String,
    /// Uptime in seconds.
    pub uptime_secs: i64,
    /// Number of connected channels.
    pub channels_connected: usize,
    /// Number of active sessions.
    pub active_sessions: usize,
}

/// Session info for the admin API (serializable summary).
#[derive(Debug, Serialize)]
pub struct SessionInfo {
    /// Session UUID.
    pub id: String,
    /// Platform name.
    pub platform: String,
    /// Platform user ID.
    pub platform_user_id: String,
    /// Display name.
    pub display_name: String,
    /// Agent session ID.
    pub agent_session_id: String,
    /// When the session was created (ISO 8601).
    pub created_at: String,
    /// When the session was last active (ISO 8601).
    pub last_activity: String,
}

/// Request body for the broadcast endpoint.
#[derive(Debug, Deserialize)]
pub struct BroadcastRequest {
    /// Message content to broadcast.
    pub message: String,
    /// Optional: limit to specific channel type (e.g., "discord").
    /// If None, broadcast to all channels.
    pub channel_type: Option<String>,
}

/// GET /admin/health — health check endpoint.
pub async fn health_check(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<Json<HealthResponse>, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let uptime = Utc::now() - state.start_time;
    Ok(Json(HealthResponse {
        status: "ok".to_string(),
        uptime_secs: uptime.num_seconds(),
        channels_connected: state.channels.count(),
        active_sessions: state.sessions.count(),
    }))
}

/// GET /admin/channels — list all connected channels.
pub async fn list_channels(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<Json<Vec<ChannelInfo>>, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    Ok(Json(state.channels.list()))
}

/// GET /admin/sessions — list all active sessions.
pub async fn list_sessions(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<Json<Vec<SessionInfo>>, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let sessions = state
        .sessions
        .list_sessions()
        .into_iter()
        .map(|s| SessionInfo {
            id: s.id.to_string(),
            platform: s.channel_user.platform,
            platform_user_id: s.channel_user.platform_user_id,
            display_name: s.channel_user.display_name,
            agent_session_id: s.agent_session_id,
            created_at: s.created_at.to_rfc3339(),
            last_activity: s.last_activity.to_rfc3339(),
        })
        .collect();

    Ok(Json(sessions))
}

/// POST /admin/broadcast — send a message to all (or filtered) channels.
pub async fn broadcast(
    State(state): State<AppState>,
    Json(payload): Json<BroadcastRequest>,
) -> impl IntoResponse {
    let channels = state.channels.list();
    let mut sent = 0usize;
    let mut failed = 0usize;

    for info in &channels {
        // Filter by channel type if specified
        if let Some(ref ct) = payload.channel_type
            && info.channel_type != *ct
        {
            continue;
        }

        if let Some(tx) = state.channels.get_sender(&info.id) {
            match tx.try_send(payload.message.clone()) {
                Ok(()) => sent += 1,
                Err(_) => failed += 1,
            }
        } else {
            failed += 1;
        }
    }

    (
        StatusCode::OK,
        Json(json!({
            "sent": sent,
            "failed": failed,
            "total_channels": channels.len()
        })),
    )
}

/// GET /admin/metrics — in-memory gateway and token usage metrics.
pub async fn get_metrics(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    Ok(Json(state.metrics.snapshot()))
}

/// GET /admin/slash/commands — list available in-chat slash commands.
pub async fn list_slash_commands(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let entries: Vec<serde_json::Value> = crate::slash::help_entries()
        .iter()
        .map(|(cmd, desc)| json!({ "command": cmd, "description": desc }))
        .collect();
    Ok(Json(entries))
}

// ---------------------------------------------------------------------------
// Cron admin API
// ---------------------------------------------------------------------------

/// Request body for creating or updating a cron job.
#[derive(Debug, Deserialize)]
pub struct CronJobRequest {
    pub name: String,
    pub schedule: String,
    pub prompt: String,
    pub target_platform: String,
    pub target_channel_id: String,
    pub target_user_id: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

/// GET /admin/cron — list all cron jobs.
pub async fn list_cron_jobs(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.cron_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(store.list().await))
}

/// POST /admin/cron — create a new cron job.
pub async fn create_cron_job(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(payload): Json<CronJobRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.cron_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;

    CronJob::validate_schedule(&payload.schedule).map_err(|_| StatusCode::UNPROCESSABLE_ENTITY)?;

    let job = CronJob {
        id: Uuid::new_v4(),
        name: payload.name,
        schedule: payload.schedule,
        prompt: payload.prompt,
        target_platform: payload.target_platform,
        target_channel_id: payload.target_channel_id,
        target_user_id: payload.target_user_id,
        enabled: payload.enabled,
        last_run: None,
    };

    store.upsert(job.clone()).await.map_err(|e| {
        tracing::error!(error = %e, "Failed to create cron job");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok((StatusCode::CREATED, Json(job)))
}

/// GET /admin/cron/:id — get a single cron job.
pub async fn get_cron_job(
    headers: HeaderMap,
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.cron_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    let job = store.get(id).await.ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(job))
}

/// PUT /admin/cron/:id — update an existing cron job.
pub async fn update_cron_job(
    headers: HeaderMap,
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(payload): Json<CronJobRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.cron_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;

    let mut job = store.get(id).await.ok_or(StatusCode::NOT_FOUND)?;

    CronJob::validate_schedule(&payload.schedule).map_err(|_| StatusCode::UNPROCESSABLE_ENTITY)?;

    job.name = payload.name;
    job.schedule = payload.schedule;
    job.prompt = payload.prompt;
    job.target_platform = payload.target_platform;
    job.target_channel_id = payload.target_channel_id;
    job.target_user_id = payload.target_user_id;
    job.enabled = payload.enabled;

    store.upsert(job.clone()).await.map_err(|e| {
        tracing::error!(error = %e, "Failed to update cron job");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(job))
}

/// DELETE /admin/cron/:id — delete a cron job.
pub async fn delete_cron_job(
    headers: HeaderMap,
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.cron_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    let removed = store.delete(id).await.map_err(|e| {
        tracing::error!(error = %e, "Failed to delete cron job");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if removed {
        Ok(StatusCode::NO_CONTENT.into_response())
    } else {
        Ok(StatusCode::NOT_FOUND.into_response())
    }
}

// ---------------------------------------------------------------------------
// Identity admin API
// ---------------------------------------------------------------------------

/// Request body for linking two platform identities.
#[derive(Debug, Deserialize)]
pub struct LinkIdentityRequest {
    /// The primary platform identity (keeps its canonical UUID).
    pub primary: PlatformIdentity,
    /// The secondary platform identity (merged under primary's UUID).
    pub secondary: PlatformIdentity,
}

/// GET /admin/identity — list all canonical identities and their linked platforms.
pub async fn list_identities(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.identity_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    let all = store.list_all().await;
    let serializable: Vec<serde_json::Value> = all
        .into_iter()
        .map(|(id, identities)| {
            json!({
                "canonical_id": id,
                "identities": identities
            })
        })
        .collect();
    Ok(Json(serializable))
}

/// POST /admin/identity/link — link two platform identities.
pub async fn link_identities(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(payload): Json<LinkIdentityRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.identity_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    let canonical_id = store
        .link(&payload.primary, &payload.secondary)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "Failed to link identities");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok(Json(json!({
        "canonical_id": canonical_id,
        "primary": payload.primary,
        "secondary": payload.secondary
    })))
}

/// DELETE /admin/identity/unlink — unlink a platform identity from its group.
pub async fn unlink_identity(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(payload): Json<PlatformIdentity>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.identity_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    let old_id = store.unlink(&payload).await.map_err(|e| {
        tracing::error!(error = %e, "Failed to unlink identity");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    match old_id {
        Some(id) => Ok(Json(json!({ "unlinked_from": id, "identity": payload })).into_response()),
        None => Ok(StatusCode::NOT_FOUND.into_response()),
    }
}

// ---------------------------------------------------------------------------
// Pairing admin API
// ---------------------------------------------------------------------------

/// Request body for approving / rejecting a pairing code.
#[derive(Debug, Deserialize)]
pub struct PairingCodeRequest {
    /// The 6-digit code.
    pub code: String,
}

/// Request body for revoking a previously-approved peer.
#[derive(Debug, Deserialize)]
pub struct PairingRevokeRequest {
    /// Channel name (e.g. `"discord"`).
    pub channel: String,
    /// Platform user id.
    pub user_id: String,
}

/// GET /admin/pairing/pending — list all currently-valid pending codes.
pub async fn list_pending_pairing(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.pairing_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(store.list_pending().await))
}

/// GET /admin/pairing/approved — list approved peers (`<channel>:<user_id>`).
pub async fn list_approved_pairing(
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.pairing_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(store.list_approved().await))
}

/// POST /admin/pairing/approve — approve a pending code.
pub async fn approve_pairing(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(payload): Json<PairingCodeRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.pairing_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    match store.approve_by_code(&payload.code).await.map_err(|e| {
        tracing::error!(error = %e, "Failed to approve pairing code");
        StatusCode::INTERNAL_SERVER_ERROR
    })? {
        Some((channel, user_id)) => Ok(Json(json!({
            "approved": true,
            "channel": channel,
            "user_id": user_id
        }))),
        None => Ok(Json(json!({
            "approved": false,
            "reason": "code not found or expired"
        }))),
    }
}

/// POST /admin/pairing/reject — reject (discard) a pending code.
pub async fn reject_pairing(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(payload): Json<PairingCodeRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.pairing_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    let rejected = store.reject_by_code(&payload.code).await.map_err(|e| {
        tracing::error!(error = %e, "Failed to reject pairing code");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok(Json(json!({ "rejected": rejected })))
}

/// POST /admin/pairing/revoke — revoke a previously-approved peer.
pub async fn revoke_pairing(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(payload): Json<PairingRevokeRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_admin_auth(&headers, &state.config)?;
    let store = state.pairing_store.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    store
        .revoke(&payload.channel, &payload.user_id)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "Failed to revoke pairing");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    Ok(Json(json!({
        "revoked": true,
        "channel": payload.channel,
        "user_id": payload.user_id
    })))
}
