//! HTTP client for the OpenAI Responses API — all 6 endpoints + streaming.

use std::sync::Arc;

use anyhow::{Context, Result};
use futures::stream::{BoxStream, StreamExt};
use reqwest::Client;

use super::types::{
    CreateResponseRequest, DeleteResponse, InputItemsList, ResponseInput, ResponseObject,
    ResponseStreamEvent,
};
use crate::rate_limiter::RateLimiter;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1/responses";

/// HTTP client for the OpenAI Responses API.
pub struct ResponsesClient {
    api_key: String,
    base_url: String,
    organization: Option<String>,
    http_client: Client,
    rate_limiter: Option<Arc<RateLimiter>>,
}

impl ResponsesClient {
    /// Create a new Responses API client.
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            organization: None,
            http_client: Client::new(),
            rate_limiter: None,
        }
    }

    /// Set a custom base URL.
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    /// Set an organization header.
    pub fn with_organization(mut self, org: String) -> Self {
        self.organization = Some(org);
        self
    }

    /// Set rate limiting.
    pub fn with_rate_limit(mut self, rpm: u32) -> Self {
        self.rate_limiter = Some(Arc::new(RateLimiter::new(rpm)));
        self
    }

    async fn acquire_rate_limit(&self) {
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire().await;
        }
    }

    /// Build a request with common headers.
    fn request(&self, method: reqwest::Method, url: &str) -> reqwest::RequestBuilder {
        let mut req = self
            .http_client
            .request(method, url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");
        if let Some(ref org) = self.organization {
            req = req.header("OpenAI-Organization", org);
        }
        req
    }

    /// `POST /v1/responses` — create a response (non-streaming).
    pub async fn create(&self, req: &CreateResponseRequest) -> Result<ResponseObject> {
        self.acquire_rate_limit().await;
        let response = self
            .request(reqwest::Method::POST, &self.base_url)
            .json(req)
            .send()
            .await
            .context("Failed to send request to Responses API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Responses API error ({}): {}", status, error_text);
        }

        response
            .json()
            .await
            .context("Failed to parse Responses API response")
    }

    /// `POST /v1/responses` with `stream: true` — streaming response.
    pub fn create_stream<'a>(
        &'a self,
        req: &'a CreateResponseRequest,
    ) -> BoxStream<'a, Result<ResponseStreamEvent>> {
        Box::pin(async_stream::stream! {
            self.acquire_rate_limit().await;

            // Build a streaming version of the request
            let mut stream_req = req.clone();
            stream_req.stream = Some(true);

            let response = match self
                .request(reqwest::Method::POST, &self.base_url)
                .json(&stream_req)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield Err(e.into());
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                yield Err(anyhow::anyhow!("Responses API error ({}): {}", status, error_text));
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(e.into());
                        continue;
                    }
                };

                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find("\n\n") {
                    let event_block = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    // SSE may have "event: <type>\ndata: <json>" or just "data: <json>"
                    for line in event_block.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data == "[DONE]" {
                                return;
                            }
                            match serde_json::from_str::<ResponseStreamEvent>(data) {
                                Ok(event) => yield Ok(event),
                                Err(e) => {
                                    tracing::warn!("Failed to parse Responses stream event: {} — data: {}", e, data);
                                }
                            }
                        }
                    }
                }
            }
        })
    }

    /// `GET /v1/responses/{response_id}` — retrieve a response.
    pub async fn retrieve(&self, response_id: &str) -> Result<ResponseObject> {
        let url = format!("{}/{}", self.base_url, response_id);
        self.acquire_rate_limit().await;
        let response = self
            .request(reqwest::Method::GET, &url)
            .send()
            .await
            .context("Failed to retrieve response")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Responses API error ({}): {}", status, error_text);
        }

        response.json().await.context("Failed to parse response")
    }

    /// `DELETE /v1/responses/{response_id}` — delete a stored response.
    pub async fn delete(&self, response_id: &str) -> Result<DeleteResponse> {
        let url = format!("{}/{}", self.base_url, response_id);
        self.acquire_rate_limit().await;
        let response = self
            .request(reqwest::Method::DELETE, &url)
            .send()
            .await
            .context("Failed to delete response")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Responses API error ({}): {}", status, error_text);
        }

        response
            .json()
            .await
            .context("Failed to parse delete response")
    }

    /// `POST /v1/responses/{response_id}/cancel` — cancel an in-progress response.
    pub async fn cancel(&self, response_id: &str) -> Result<ResponseObject> {
        let url = format!("{}/{}/cancel", self.base_url, response_id);
        self.acquire_rate_limit().await;
        let response = self
            .request(reqwest::Method::POST, &url)
            .send()
            .await
            .context("Failed to cancel response")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Responses API error ({}): {}", status, error_text);
        }

        response
            .json()
            .await
            .context("Failed to parse cancel response")
    }

    /// `GET /v1/responses/{response_id}/input_items` — list input items.
    pub async fn list_input_items(&self, response_id: &str) -> Result<InputItemsList> {
        let url = format!("{}/{}/input_items", self.base_url, response_id);
        self.acquire_rate_limit().await;
        let response = self
            .request(reqwest::Method::GET, &url)
            .send()
            .await
            .context("Failed to list input items")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Responses API error ({}): {}", status, error_text);
        }

        response
            .json()
            .await
            .context("Failed to parse input items list")
    }

    /// `POST /v1/responses/compact` — compress conversation context.
    pub async fn compact(
        &self,
        model: &str,
        input: ResponseInput,
        previous_response_id: Option<&str>,
    ) -> Result<ResponseObject> {
        let url = format!("{}/compact", self.base_url);
        let mut body = serde_json::json!({
            "model": model,
            "input": input,
        });
        if let Some(prev_id) = previous_response_id {
            body["previous_response_id"] = serde_json::json!(prev_id);
        }

        self.acquire_rate_limit().await;
        let response = self
            .request(reqwest::Method::POST, &url)
            .json(&body)
            .send()
            .await
            .context("Failed to compact response")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Responses API error ({}): {}", status, error_text);
        }

        response
            .json()
            .await
            .context("Failed to parse compact response")
    }
}
