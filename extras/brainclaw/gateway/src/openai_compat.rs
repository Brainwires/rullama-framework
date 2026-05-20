//! OpenAI-compatible API endpoint.
//!
//! Exposes `/v1/chat/completions`, `/v1/models`, and `/v1/embeddings` so that
//! any OpenAI SDK client (Python `openai`, `curl`, Open WebUI, etc.) can point
//! its `base_url` at BrainClaw and use the configured LLM provider.
//!
//! # Authentication
//! Bearer token from the `Authorization` header is checked against
//! `GatewayConfig::auth_tokens`.  If the token list is empty, the endpoints
//! are open (dev mode).

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::Event;
use axum::response::{IntoResponse, Sse};
use brainwires_core::{ChatOptions, Message, StreamChunk};
use chrono::Utc;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::state::AppState;

// ── Request / response types ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OaiMessage>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub stop: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OaiMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: i64,
    owned_by: &'static str,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn check_auth(headers: &HeaderMap, tokens: &[String]) -> bool {
    if tokens.is_empty() {
        return true;
    }
    let bearer = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    tokens.iter().any(|t| t == bearer)
}

fn oai_messages_to_core(messages: Vec<OaiMessage>) -> Vec<Message> {
    messages
        .into_iter()
        .map(|m| {
            let content = m.content;
            match m.role.as_str() {
                "system" => Message::system(&content),
                "assistant" => Message::assistant(&content),
                _ => Message::user(&content),
            }
        })
        .collect()
}

fn build_options(req: &ChatCompletionRequest) -> ChatOptions {
    ChatOptions {
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        top_p: req.top_p,
        stop: req.stop.as_ref().and_then(|v| match v {
            Value::String(s) => Some(vec![s.clone()]),
            Value::Array(arr) => Some(
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect(),
            ),
            _ => None,
        }),
        system: None,
        model: Some(req.model.clone()),
        ..Default::default()
    }
}

// ── Handlers ─────────────────────────────────────────────────────────────

/// `GET /v1/models` — list the provider's models.
pub async fn list_models(headers: HeaderMap, State(state): State<AppState>) -> impl IntoResponse {
    if !check_auth(&headers, &state.config.auth_tokens) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "Unauthorized"})),
        )
            .into_response();
    }

    let provider_name = state
        .openai_provider
        .as_ref()
        .map(|p| p.name().to_string())
        .unwrap_or_else(|| "brainclaw".to_string());

    let now = Utc::now().timestamp();
    let data = vec![ModelObject {
        id: provider_name,
        object: "model",
        created: now,
        owned_by: "brainclaw",
    }];

    Json(ModelsResponse {
        object: "list",
        data,
    })
    .into_response()
}

/// `POST /v1/chat/completions` — OpenAI-compatible chat completion.
pub async fn chat_completions(
    headers: HeaderMap,
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if !check_auth(&headers, &state.config.auth_tokens) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "Unauthorized"})),
        )
            .into_response();
    }

    let provider = match &state.openai_provider {
        Some(p) => Arc::clone(p),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "No LLM provider configured for OpenAI-compat endpoint"})),
            )
                .into_response();
        }
    };

    let messages = oai_messages_to_core(req.messages.clone());
    let options = build_options(&req);

    if req.stream {
        // Streaming response — SSE events in OpenAI format.
        // We spawn a task that consumes the borrowed stream and forwards
        // chunks over an mpsc channel, avoiding lifetime issues.
        let completion_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
        let model = req.model.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamChunk, String>>(64);

        tokio::spawn(async move {
            let mut stream = provider.stream_chat(&messages, None, &options);
            while let Some(chunk) = stream.next().await {
                let item = chunk.map_err(|e| e.to_string());
                if tx.send(item).await.is_err() {
                    break;
                }
            }
        });

        let sse_stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(move |chunk| {
            let cid = completion_id.clone();
            let model = model.clone();
            match chunk {
                Ok(StreamChunk::Text(text)) => {
                    let data = json!({
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": Utc::now().timestamp(),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": text},
                            "finish_reason": null
                        }]
                    });
                    Ok::<Event, anyhow::Error>(Event::default().data(data.to_string()))
                }
                Ok(_) => Ok(Event::default().comment("")),
                Err(e) => {
                    let data = json!({"error": e});
                    Ok(Event::default().data(data.to_string()))
                }
            }
        });

        // Append the [DONE] sentinel
        let done_event = futures::stream::once(async {
            Ok::<Event, anyhow::Error>(Event::default().data("[DONE]"))
        });
        let combined = sse_stream.chain(done_event);

        Sse::new(combined).into_response()
    } else {
        // Non-streaming: collect the full response
        match provider.chat(&messages, None, &options).await {
            Ok(response) => {
                let text = response.message.text_or_summary();
                let completion_id = format!("chatcmpl-{}", Uuid::new_v4().simple());
                let body = json!({
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": Utc::now().timestamp(),
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                });
                Json(body).into_response()
            }
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
                .into_response(),
        }
    }
}

/// `POST /v1/embeddings` — proxy to provider (returns error if unsupported).
pub async fn embeddings(headers: HeaderMap, State(state): State<AppState>) -> impl IntoResponse {
    if !check_auth(&headers, &state.config.auth_tokens) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "Unauthorized"})),
        )
            .into_response();
    }

    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": {
                "message": "Embeddings endpoint not yet implemented in BrainClaw",
                "type": "invalid_request_error",
                "code": "not_implemented"
            }
        })),
    )
        .into_response()
}
