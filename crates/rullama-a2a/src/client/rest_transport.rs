//! HTTP/REST transport.

use std::pin::Pin;

use futures::Stream;
use url::Url;

use crate::error::A2aError;
use crate::streaming::StreamResponse;

/// REST transport client.
pub struct RestTransport {
    base_url: Url,
    client: reqwest::Client,
    bearer_token: Option<String>,
}

impl RestTransport {
    /// Create a new REST transport.
    pub fn new(base_url: Url, client: reqwest::Client, bearer_token: Option<String>) -> Self {
        Self {
            base_url,
            client,
            bearer_token,
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

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.as_str().trim_end_matches('/'), path)
    }

    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref token) = self.bearer_token {
            builder.bearer_auth(token)
        } else {
            builder
        }
    }

    /// POST with JSON body, return JSON.
    pub async fn post(
        &self,
        path: &str,
        body: &impl serde::Serialize,
    ) -> Result<serde_json::Value, A2aError> {
        let builder = self.client.post(self.url(path)).json(body);
        let resp = self
            .apply_auth(builder)
            .send()
            .await
            .map_err(|e| A2aError::internal(format!("REST request failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(A2aError::internal(format!("REST error: {}", resp.status())));
        }

        resp.json()
            .await
            .map_err(|e| A2aError::internal(format!("Failed to parse REST response: {e}")))
    }

    /// GET, return JSON.
    pub async fn get(&self, path: &str) -> Result<serde_json::Value, A2aError> {
        let builder = self.client.get(self.url(path));
        let resp = self
            .apply_auth(builder)
            .send()
            .await
            .map_err(|e| A2aError::internal(format!("REST request failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(A2aError::internal(format!("REST error: {}", resp.status())));
        }

        resp.json()
            .await
            .map_err(|e| A2aError::internal(format!("Failed to parse REST response: {e}")))
    }

    /// DELETE, return nothing.
    pub async fn delete(&self, path: &str) -> Result<(), A2aError> {
        let builder = self.client.delete(self.url(path));
        let resp = self
            .apply_auth(builder)
            .send()
            .await
            .map_err(|e| A2aError::internal(format!("REST DELETE failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(A2aError::internal(format!(
                "REST DELETE error: {}",
                resp.status()
            )));
        }

        Ok(())
    }

    /// POST returning SSE stream.
    pub fn post_stream(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>> {
        let url = self.url(path);
        let client = self.client.clone();
        let token = self.bearer_token.clone();

        Box::pin(async_stream::stream! {
            let mut builder = client.post(&url).json(&body);
            if let Some(ref t) = token {
                builder = builder.bearer_auth(t);
            }
            let resp = match builder.send().await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(A2aError::internal(format!("REST stream request failed: {e}")));
                    return;
                }
            };

            use futures::StreamExt;
            let byte_stream = resp.bytes_stream();
            let mut sse_stream = std::pin::pin!(
                crate::client::sse::parse_sse_rest_byte_stream(byte_stream)
            );
            while let Some(item) = sse_stream.next().await {
                yield item;
            }
        })
    }

    /// GET returning SSE stream.
    pub fn get_stream(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>> {
        let url = self.url(path);
        let client = self.client.clone();
        let token = self.bearer_token.clone();

        Box::pin(async_stream::stream! {
            let mut builder = client.get(&url);
            if let Some(ref t) = token {
                builder = builder.bearer_auth(t);
            }
            let resp = match builder.send().await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(A2aError::internal(format!("REST stream GET failed: {e}")));
                    return;
                }
            };

            use futures::StreamExt;
            let byte_stream = resp.bytes_stream();
            let mut sse_stream = std::pin::pin!(
                crate::client::sse::parse_sse_rest_byte_stream(byte_stream)
            );
            while let Some(item) = sse_stream.next().await {
                yield item;
            }
        })
    }
}
