use std::collections::HashMap;
use std::sync::Mutex;

use brainwires_proxy::error::ProxyResult;
use brainwires_proxy::middleware::{LayerAction, ProxyLayer};
use brainwires_proxy::request_id::RequestId;
use brainwires_proxy::types::{ProxyRequest, ProxyResponse};
use http::StatusCode;
use tracing::info;

use crate::config::AdapterConfig;
use crate::convert_request::convert_request;
use crate::convert_response::convert_response;
use crate::convert_sse::format_as_sse;
use crate::tool_name_mapper::ToolNameMapper;
use crate::types_anthropic::AnthropicRequest;
use crate::types_openai::OpenAIChatResponse;

/// Metadata stashed between on_request and on_response.
struct AdapterMeta {
    was_streaming: bool,
    original_model: String,
    mapper: ToolNameMapper,
}

/// The core middleware that converts Anthropic ↔ OpenAI on the fly.
pub struct AdapterLayer {
    config: AdapterConfig,
    /// Maps RequestId → per-request metadata. The HttpConnector creates a fresh
    /// ProxyResponse, so we can't pass data through extensions — we use the
    /// RequestId (which IS preserved) as the lookup key.
    pending: Mutex<HashMap<RequestId, AdapterMeta>>,
}

impl AdapterLayer {
    pub fn new(config: AdapterConfig) -> Self {
        Self {
            config,
            pending: Mutex::new(HashMap::new()),
        }
    }

    fn json_response(request_id: RequestId, status: StatusCode, body: &str) -> ProxyResponse {
        let mut resp = ProxyResponse::for_request(request_id, status).with_body(body.to_string());
        resp.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        resp
    }

    fn error_response(request_id: RequestId, status: StatusCode, message: &str) -> ProxyResponse {
        let body = serde_json::json!({
            "error": {
                "type": "adapter_error",
                "message": message,
            }
        });
        Self::json_response(request_id, status, &body.to_string())
    }
}

#[async_trait::async_trait]
impl ProxyLayer for AdapterLayer {
    async fn on_request(&self, mut request: ProxyRequest) -> ProxyResult<LayerAction> {
        let path = request.uri.path();

        // GET /health — short-circuit
        if request.method == http::Method::GET && path == "/health" {
            let body = serde_json::json!({ "status": "ok", "adapter": "anthropic-openai-proxy" });
            let resp = Self::json_response(request.id.clone(), StatusCode::OK, &body.to_string());
            return Ok(LayerAction::Respond(resp));
        }

        // Only accept POST /v1/messages
        if request.method != http::Method::POST || path != "/v1/messages" {
            let resp = Self::error_response(
                request.id.clone(),
                StatusCode::NOT_FOUND,
                &format!("unsupported endpoint: {} {}", request.method, path),
            );
            return Ok(LayerAction::Respond(resp));
        }

        // Parse Anthropic request body
        let body_bytes = request.body.as_bytes();
        let anthropic_req: AnthropicRequest = match serde_json::from_slice(body_bytes) {
            Ok(r) => r,
            Err(e) => {
                let resp = Self::error_response(
                    request.id.clone(),
                    StatusCode::BAD_REQUEST,
                    &format!("invalid request body: {}", e),
                );
                return Ok(LayerAction::Respond(resp));
            }
        };

        let was_streaming = anthropic_req.stream.unwrap_or(false);
        let original_model = anthropic_req.model.clone();

        info!(
            request_id = %request.id,
            model = %original_model,
            stream = was_streaming,
            "incoming Anthropic request"
        );

        // Resolve model → provider
        let resolved = match self.config.resolve_model(&original_model) {
            Ok(r) => r,
            Err(e) => {
                let resp = Self::error_response(
                    request.id.clone(),
                    StatusCode::BAD_REQUEST,
                    &format!("model resolution failed: {}", e),
                );
                return Ok(LayerAction::Respond(resp));
            }
        };

        // Convert Anthropic → OpenAI
        let converted = match convert_request(&anthropic_req, resolved.target_model) {
            Ok(c) => c,
            Err(e) => {
                let resp = Self::error_response(
                    request.id.clone(),
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("conversion failed: {}", e),
                );
                return Ok(LayerAction::Respond(resp));
            }
        };

        // Serialize OpenAI request as body
        let openai_body = serde_json::to_vec(&converted.request).unwrap_or_default();

        // Stash metadata for on_response
        {
            let meta = AdapterMeta {
                was_streaming,
                original_model,
                mapper: converted.mapper,
            };
            self.pending
                .lock()
                .expect("adapter pending requests lock poisoned")
                .insert(request.id.clone(), meta);
        }

        // Rewrite the request
        request.body = openai_body.into();

        // Change path: /v1/messages → /v1/chat/completions
        let new_uri = rebuild_uri(&request.uri, "/v1/chat/completions");
        request.uri = new_uri;

        // Swap auth: remove x-api-key, set Authorization: Bearer
        request.headers.remove("x-api-key");
        if let Ok(val) =
            http::HeaderValue::from_str(&format!("Bearer {}", resolved.provider.api_key))
        {
            request.headers.insert(http::header::AUTHORIZATION, val);
        }

        // Ensure Content-Type is application/json
        request.headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );

        Ok(LayerAction::Forward(request))
    }

    async fn on_response(&self, mut response: ProxyResponse) -> ProxyResult<ProxyResponse> {
        // Look up the per-request metadata
        let meta = self
            .pending
            .lock()
            .expect("adapter pending requests lock poisoned")
            .remove(&response.id);
        let Some(meta) = meta else {
            // No metadata → this wasn't our request (e.g. health check). Pass through.
            return Ok(response);
        };

        // Parse OpenAI response
        let body_bytes = response.body.as_bytes();
        let openai_resp: OpenAIChatResponse = match serde_json::from_slice(body_bytes) {
            Ok(r) => r,
            Err(e) => {
                // If we can't parse the upstream response, return the error to the client.
                // Include the raw body for debugging.
                let raw = String::from_utf8_lossy(body_bytes);
                let err_body = serde_json::json!({
                    "error": {
                        "type": "upstream_error",
                        "message": format!("failed to parse upstream response: {}", e),
                        "upstream_body": raw,
                    }
                });
                response.body = err_body.to_string().into();
                response.status = StatusCode::BAD_GATEWAY;
                response.headers.insert(
                    http::header::CONTENT_TYPE,
                    http::HeaderValue::from_static("application/json"),
                );
                return Ok(response);
            }
        };

        // Convert OpenAI → Anthropic
        let anthropic_resp =
            match convert_response(&openai_resp, &meta.original_model, &meta.mapper) {
                Ok(r) => r,
                Err(e) => {
                    let err_body = serde_json::json!({
                        "error": {
                            "type": "conversion_error",
                            "message": format!("response conversion failed: {}", e),
                        }
                    });
                    response.body = err_body.to_string().into();
                    response.status = StatusCode::INTERNAL_SERVER_ERROR;
                    response.headers.insert(
                        http::header::CONTENT_TYPE,
                        http::HeaderValue::from_static("application/json"),
                    );
                    return Ok(response);
                }
            };

        info!(
            request_id = %response.id,
            stop_reason = ?anthropic_resp.stop_reason,
            input_tokens = anthropic_resp.usage.input_tokens,
            output_tokens = anthropic_resp.usage.output_tokens,
            "returning Anthropic response"
        );

        // Format as SSE or JSON depending on original stream flag
        if meta.was_streaming {
            let sse_body = format_as_sse(&anthropic_resp);
            response.body = sse_body.into();
            response.headers.insert(
                http::header::CONTENT_TYPE,
                http::HeaderValue::from_static("text/event-stream"),
            );
            response.headers.insert(
                http::header::CACHE_CONTROL,
                http::HeaderValue::from_static("no-cache"),
            );
        } else {
            let json_body = serde_json::to_vec(&anthropic_resp).unwrap_or_default();
            response.body = json_body.into();
            response.headers.insert(
                http::header::CONTENT_TYPE,
                http::HeaderValue::from_static("application/json"),
            );
        }

        response.status = StatusCode::OK;
        Ok(response)
    }

    fn name(&self) -> &str {
        "anthropic-openai-proxy"
    }
}

/// Rebuild a URI with a new path, preserving query string.
fn rebuild_uri(original: &http::Uri, new_path: &str) -> http::Uri {
    let mut parts = original.clone().into_parts();
    let pq = if let Some(q) = original.query() {
        format!("{}?{}", new_path, q)
    } else {
        new_path.to_string()
    };
    parts.path_and_query = Some(pq.parse().unwrap());
    http::Uri::from_parts(parts).unwrap_or_else(|_| new_path.parse().unwrap())
}
