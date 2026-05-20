//! Axum REST API handlers
//!
//! Provides the HTTP endpoints for the skill registry server.

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use brainwires_agent::skills::SkillPackage;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::storage::SkillStore;

/// Shared application state.
pub type AppState = Arc<Mutex<SkillStore>>;

/// Build the axum router with all skill registry routes.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/api/skills", post(publish_skill))
        .route("/api/skills/search", get(search_skills))
        .route("/api/skills/{name}", get(get_latest_manifest))
        .route("/api/skills/{name}/versions", get(list_versions))
        .route("/api/skills/{name}/{version}", get(get_versioned_manifest))
        .route(
            "/api/skills/{name}/{version}/download",
            get(download_package),
        )
        .with_state(state)
}

/// POST /api/skills — publish a skill package
async fn publish_skill(
    State(store): State<AppState>,
    Json(package): Json<SkillPackage>,
) -> impl IntoResponse {
    if !package.verify_checksum() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Checksum verification failed"})),
        );
    }

    let store = store.lock().await;
    match store.insert_skill(&package) {
        Ok(()) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "message": "Skill published",
                "name": package.manifest.name,
                "version": package.manifest.version.to_string()
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{:#}", e)})),
        ),
    }
}

/// Query parameters for the search endpoint.
#[derive(Deserialize)]
struct SearchParams {
    q: Option<String>,
    tags: Option<String>,
    limit: Option<u32>,
}

/// GET /api/skills/search?q=query&tags=tag1,tag2&limit=20
async fn search_skills(
    State(store): State<AppState>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    let query = params.q.unwrap_or_default();
    let limit = params.limit.unwrap_or(20).min(100);
    let tags: Option<Vec<String>> = params
        .tags
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect());

    let store = store.lock().await;
    match store.search(&query, tags.as_deref(), limit) {
        Ok(results) => (StatusCode::OK, Json(serde_json::to_value(results).unwrap())),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{:#}", e)})),
        ),
    }
}

/// GET /api/skills/:name — get latest manifest
async fn get_latest_manifest(
    State(store): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let store = store.lock().await;
    match store.get_latest_manifest(&name) {
        Ok(Some(manifest)) => (
            StatusCode::OK,
            Json(serde_json::to_value(manifest).unwrap()),
        ),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Skill not found"})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{:#}", e)})),
        ),
    }
}

/// GET /api/skills/:name/versions — list all versions
async fn list_versions(
    State(store): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let store = store.lock().await;
    match store.list_versions(&name) {
        Ok(versions) => {
            let v: Vec<String> = versions.iter().map(|v| v.to_string()).collect();
            (StatusCode::OK, Json(serde_json::to_value(v).unwrap()))
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{:#}", e)})),
        ),
    }
}

/// GET /api/skills/:name/:version — get specific version manifest
async fn get_versioned_manifest(
    State(store): State<AppState>,
    Path((name, version)): Path<(String, String)>,
) -> impl IntoResponse {
    let store = store.lock().await;
    match store.get_manifest(&name, &version) {
        Ok(Some(manifest)) => (
            StatusCode::OK,
            Json(serde_json::to_value(manifest).unwrap()),
        ),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Version not found"})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{:#}", e)})),
        ),
    }
}

/// GET /api/skills/:name/:version/download — download full package
async fn download_package(
    State(store): State<AppState>,
    Path((name, version)): Path<(String, String)>,
) -> impl IntoResponse {
    let store = store.lock().await;
    match store.get_package(&name, &version) {
        Ok(Some(package)) => (StatusCode::OK, Json(serde_json::to_value(package).unwrap())),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "Package not found"})),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{:#}", e)})),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_router_builds() {
        let store = SkillStore::open_in_memory().unwrap();
        let state = Arc::new(Mutex::new(store));
        let _app = router(state);
        // Router builds without panic
    }
}
