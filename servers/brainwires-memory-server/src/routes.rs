//! Axum route handlers for the Mem0-compatible memory REST API.
//!
//! All handlers are backed by
//! [`brainwires_knowledge::knowledge::brain_client::BrainClient`]. Tenant
//! isolation is enforced by requiring `user_id` on every request that touches
//! stored memories; the value is forwarded as `owner_id` to the knowledge
//! layer.

use std::collections::HashMap;

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use brainwires_knowledge::knowledge::types::{
    CaptureThoughtRequest, GetThoughtResponse, ListRecentRequest, SearchMemoryRequest,
    ThoughtSummary,
};
use chrono::{DateTime, TimeZone, Utc};
use uuid::Uuid;

use crate::{
    AppState,
    types::{
        AddMemoryRequest, AddMemoryResponse, ListMemoriesQuery, ListMemoriesResponse, Memory,
        MemoryResult, MessageResponse, SearchMemoriesRequest, SearchMemoriesResponse, SearchResult,
        UpdateMemoryRequest,
    },
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn internal_error(e: anyhow::Error) -> (StatusCode, Json<MessageResponse>) {
    tracing::error!("internal error: {e:#}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(MessageResponse {
            message: e.to_string(),
        }),
    )
}

fn bad_request(msg: impl Into<String>) -> (StatusCode, Json<MessageResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(MessageResponse {
            message: msg.into(),
        }),
    )
}

fn not_found_id(id: &Uuid) -> (StatusCode, Json<MessageResponse>) {
    (
        StatusCode::NOT_FOUND,
        Json(MessageResponse {
            message: format!("Memory {id} not found"),
        }),
    )
}

fn ts_to_utc(ts: i64) -> DateTime<Utc> {
    Utc.timestamp_opt(ts, 0).single().unwrap_or_else(Utc::now)
}

fn parse_memory_id(id: &Uuid) -> String {
    id.to_string()
}

fn get_response_to_memory(resp: GetThoughtResponse, user_id: &str) -> Memory {
    let id = Uuid::parse_str(&resp.id).unwrap_or_else(|_| Uuid::nil());
    Memory {
        id,
        memory: resp.content,
        user_id: user_id.to_string(),
        agent_id: None,
        session_id: None,
        metadata: serde_json::Value::Object(Default::default()),
        created_at: ts_to_utc(resp.created_at),
        updated_at: ts_to_utc(resp.updated_at),
        categories: if resp.category.is_empty() {
            None
        } else {
            Some(vec![resp.category])
        },
    }
}

fn summary_to_memory(summary: ThoughtSummary, user_id: &str) -> Memory {
    let id = Uuid::parse_str(&summary.id).unwrap_or_else(|_| Uuid::nil());
    let created = ts_to_utc(summary.created_at);
    Memory {
        id,
        memory: summary.content,
        user_id: user_id.to_string(),
        agent_id: None,
        session_id: None,
        metadata: serde_json::Value::Object(Default::default()),
        created_at: created,
        updated_at: created,
        categories: if summary.category.is_empty() {
            None
        } else {
            Some(vec![summary.category])
        },
    }
}

// ── Health ────────────────────────────────────────────────────────────────────

/// `GET /health`
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

// ── Add memory ────────────────────────────────────────────────────────────────

/// `POST /v1/memories`
pub async fn add_memory(
    State(state): State<AppState>,
    Json(req): Json<AddMemoryRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<MessageResponse>)> {
    if req.user_id.is_empty() {
        return Err(bad_request("user_id is required"));
    }

    // Prefer explicit `memory` field, otherwise use non-system messages.
    let contents: Vec<String> = if let Some(direct) = req.memory {
        vec![direct]
    } else {
        req.messages
            .iter()
            .filter(|m| m.role != "system")
            .map(|m| m.content.clone())
            .collect()
    };

    if contents.is_empty() {
        return Err(bad_request("No memory content provided"));
    }

    let mut results = Vec::with_capacity(contents.len());
    {
        let mut client = state.client.lock().await;
        for content in contents {
            let resp = client
                .capture_thought(CaptureThoughtRequest {
                    content: content.clone(),
                    category: None,
                    tags: None,
                    importance: None,
                    source: None,
                    owner_id: Some(req.user_id.clone()),
                })
                .await
                .map_err(internal_error)?;

            let id = Uuid::parse_str(&resp.id).unwrap_or_else(|_| Uuid::nil());
            results.push(MemoryResult {
                memory: content,
                event: "add".to_string(),
                id,
            });
        }
    }

    Ok((StatusCode::CREATED, Json(AddMemoryResponse { results })))
}

// ── List memories ─────────────────────────────────────────────────────────────

/// `GET /v1/memories`
pub async fn list_memories(
    State(state): State<AppState>,
    Query(query): Query<ListMemoriesQuery>,
) -> Result<impl IntoResponse, (StatusCode, Json<MessageResponse>)> {
    let user_id = query
        .user_id
        .clone()
        .ok_or_else(|| bad_request("user_id query parameter is required"))?;

    let page = query.page.max(1);
    let page_size = query.page_size.max(1);
    // Fetch enough to cover the requested page; knowledge list_recent returns
    // up to `limit` items sorted by created_at DESC.
    let fetch_limit = (page as usize).saturating_mul(page_size as usize).max(1);

    let list = {
        let client = state.client.lock().await;
        client
            .list_recent(ListRecentRequest {
                limit: fetch_limit,
                category: None,
                // Look back far enough to effectively be "all time".
                since: Some("1970-01-01T00:00:00Z".to_string()),
                owner_id: Some(user_id.clone()),
            })
            .await
            .map_err(internal_error)?
    };

    // Total across the current filtered window (knowledge layer doesn't expose
    // an exact count API, so this is the size of the fetched window).
    let total = list.total as u64;

    // Client-side pagination over the fetched window.
    let start = ((page - 1) as usize).saturating_mul(page_size as usize);
    let end = start
        .saturating_add(page_size as usize)
        .min(list.thoughts.len());
    let slice = if start >= list.thoughts.len() {
        Vec::new()
    } else {
        list.thoughts[start..end].to_vec()
    };

    let results: Vec<Memory> = slice
        .into_iter()
        .map(|s| summary_to_memory(s, &user_id))
        .collect();

    Ok(Json(ListMemoriesResponse {
        results,
        total,
        page,
        page_size,
    }))
}

// ── Get memory ────────────────────────────────────────────────────────────────

/// `GET /v1/memories/{id}?user_id=…`
pub async fn get_memory(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, (StatusCode, Json<MessageResponse>)> {
    let user_id = params
        .get("user_id")
        .cloned()
        .ok_or_else(|| bad_request("user_id query parameter is required"))?;

    let id_str = parse_memory_id(&id);
    let fetched = {
        let client = state.client.lock().await;
        client
            .get_thought(&id_str, Some(&user_id))
            .await
            .map_err(internal_error)?
    };

    match fetched {
        Some(resp) => Ok(Json(get_response_to_memory(resp, &user_id))),
        None => Err(not_found_id(&id)),
    }
}

// ── Update memory ─────────────────────────────────────────────────────────────

/// `PATCH /v1/memories/{id}`
pub async fn update_memory(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateMemoryRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<MessageResponse>)> {
    if req.user_id.is_empty() {
        return Err(bad_request("user_id is required"));
    }

    let id_str = parse_memory_id(&id);

    let (updated, fetched) = {
        let client = state.client.lock().await;
        let updated = client
            .update_thought(&id_str, req.memory.clone(), Some(req.user_id.clone()))
            .await
            .map_err(internal_error)?;
        if !updated {
            return Err(not_found_id(&id));
        }
        let fetched = client
            .get_thought(&id_str, Some(&req.user_id))
            .await
            .map_err(internal_error)?;
        (updated, fetched)
    };

    match fetched {
        Some(resp) if updated => Ok(Json(get_response_to_memory(resp, &req.user_id))),
        _ => Err(not_found_id(&id)),
    }
}

// ── Delete memory ─────────────────────────────────────────────────────────────

/// `DELETE /v1/memories/{id}?user_id=…`
pub async fn delete_memory(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, (StatusCode, Json<MessageResponse>)> {
    let user_id = params
        .get("user_id")
        .cloned()
        .ok_or_else(|| bad_request("user_id query parameter is required"))?;

    let id_str = parse_memory_id(&id);
    let resp = {
        let client = state.client.lock().await;
        client
            .delete_thought(&id_str, Some(&user_id))
            .await
            .map_err(internal_error)?
    };

    if resp.deleted {
        Ok(Json(MessageResponse {
            message: format!("Memory {id} deleted"),
        }))
    } else {
        Err(not_found_id(&id))
    }
}

// ── Delete all ────────────────────────────────────────────────────────────────

/// `DELETE /v1/memories?user_id={user_id}`
pub async fn delete_all_memories(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, (StatusCode, Json<MessageResponse>)> {
    let user_id = params
        .get("user_id")
        .cloned()
        .ok_or_else(|| bad_request("user_id query parameter is required"))?;

    let mut count: u64 = 0;
    let client = state.client.lock().await;

    loop {
        let list = client
            .list_recent(ListRecentRequest {
                limit: 1000,
                category: None,
                since: Some("1970-01-01T00:00:00Z".to_string()),
                owner_id: Some(user_id.clone()),
            })
            .await
            .map_err(internal_error)?;

        if list.thoughts.is_empty() {
            break;
        }

        let batch_len = list.thoughts.len();
        for t in list.thoughts {
            let resp = client
                .delete_thought(&t.id, Some(&user_id))
                .await
                .map_err(internal_error)?;
            if resp.deleted {
                count += 1;
            }
        }

        if batch_len < 1000 {
            break;
        }
    }

    Ok(Json(MessageResponse {
        message: format!("Deleted {count} memories for user {user_id}"),
    }))
}

// ── Search ────────────────────────────────────────────────────────────────────

/// `POST /v1/memories/search`
pub async fn search_memories(
    State(state): State<AppState>,
    Json(req): Json<SearchMemoriesRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<MessageResponse>)> {
    let user_id = req
        .user_id
        .clone()
        .ok_or_else(|| bad_request("user_id is required"))?;

    let resp = {
        let client = state.client.lock().await;
        client
            .search_memory(SearchMemoryRequest {
                query: req.query.clone(),
                limit: req.limit as usize,
                min_score: 0.0,
                category: None,
                sources: Some(vec!["thoughts".into()]),
                owner_id: Some(user_id.clone()),
            })
            .await
            .map_err(internal_error)?
    };

    let results: Vec<SearchResult> = resp
        .results
        .into_iter()
        .filter_map(|r| {
            let thought_id = r.thought_id.as_deref()?;
            let id = Uuid::parse_str(thought_id).ok()?;
            let created = r.created_at.map(ts_to_utc).unwrap_or_else(Utc::now);
            let categories = r.category.map(|c| vec![c]);
            Some(SearchResult {
                memory: Memory {
                    id,
                    memory: r.content,
                    user_id: user_id.clone(),
                    agent_id: None,
                    session_id: None,
                    metadata: serde_json::Value::Object(Default::default()),
                    created_at: created,
                    updated_at: created,
                    categories,
                },
                score: r.score as f64,
            })
        })
        .collect();

    Ok(Json(SearchMemoriesResponse { results }))
}
