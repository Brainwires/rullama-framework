//! JSON-RPC over HTTP+SSE transport.

use std::pin::Pin;

use futures::Stream;
use url::Url;

use crate::error::A2aError;
use crate::jsonrpc::{JsonRpcRequest, JsonRpcResponse, RequestId};
use crate::streaming::StreamResponse;

/// JSON-RPC transport client.
pub struct JsonRpcTransport {
    base_url: Url,
    client: reqwest::Client,
    bearer_token: Option<String>,
    request_counter: std::sync::atomic::AtomicI64,
}

impl JsonRpcTransport {
    /// Create a new transport pointing at the given base URL.
    pub fn new(base_url: Url, client: reqwest::Client, bearer_token: Option<String>) -> Self {
        Self {
            base_url,
            client,
            bearer_token,
            request_counter: std::sync::atomic::AtomicI64::new(1),
        }
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// Get the HTTP client.
    pub fn http_client(&self) -> &reqwest::Client {
        &self.client
    }

    fn next_id(&self) -> RequestId {
        let id = self
            .request_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        RequestId::Number(id)
    }

    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref token) = self.bearer_token {
            builder.bearer_auth(token)
        } else {
            builder
        }
    }

    /// Send a JSON-RPC request and get the response.
    pub async fn call(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, A2aError> {
        let id = self.next_id();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params: Some(params),
            id: id.clone(),
        };

        let builder = self.client.post(self.base_url.as_str()).json(&request);
        let resp = self
            .apply_auth(builder)
            .send()
            .await
            .map_err(|e| A2aError::internal(format!("HTTP request failed: {e}")))?;

        let rpc_resp: JsonRpcResponse = resp
            .json()
            .await
            .map_err(|e| A2aError::internal(format!("Failed to parse response: {e}")))?;

        if let Some(err) = rpc_resp.error {
            return Err(err);
        }

        rpc_resp
            .result
            .ok_or_else(|| A2aError::internal("Empty result"))
    }

    /// Send a JSON-RPC request and stream SSE responses incrementally.
    pub fn call_stream(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>> {
        let id = self.next_id();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params: Some(params),
            id,
        };
        let client = self.client.clone();
        let url = self.base_url.clone();
        let token = self.bearer_token.clone();

        Box::pin(async_stream::stream! {
            let mut builder = client.post(url.as_str()).json(&request);
            if let Some(ref t) = token {
                builder = builder.bearer_auth(t);
            }
            let resp = match builder.send().await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(A2aError::internal(format!("HTTP request failed: {e}")));
                    return;
                }
            };

            use futures::StreamExt;
            let byte_stream = resp.bytes_stream();
            let mut sse_stream = std::pin::pin!(
                crate::client::sse::parse_sse_byte_stream(byte_stream)
            );
            while let Some(item) = sse_stream.next().await {
                yield item;
            }
        })
    }
}
