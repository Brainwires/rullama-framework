//! Provider trait implementation for the OpenAI Responses API over WebSocket.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use tokio::sync::Mutex;

use brainwires_core::{ChatOptions, ChatResponse, Message, Provider, StreamChunk, Tool};

use super::convert;
use super::types::ToolChoice;
use super::websocket::ResponsesWebSocket;

/// Chat provider backed by the OpenAI Responses API over WebSocket.
///
/// Maintains a persistent WebSocket connection for lower-latency multi-turn
/// interactions. The server caches the most recent response in-memory per
/// connection, so `previous_response_id` chaining is especially fast.
///
/// Uses `store: false` by default to leverage connection-local caching without
/// persisting responses server-side.
pub struct OpenAiResponsesWsProvider {
    ws: Arc<ResponsesWebSocket>,
    model: String,
    provider_name: String,
    last_response_id: Arc<Mutex<Option<String>>>,
    store: bool,
}

impl OpenAiResponsesWsProvider {
    /// Create a new WebSocket-based Responses API provider.
    pub fn new(ws: Arc<ResponsesWebSocket>, model: String) -> Self {
        Self {
            ws,
            model,
            provider_name: "openai-responses-ws".to_string(),
            last_response_id: Arc::new(Mutex::new(None)),
            store: false,
        }
    }

    /// Set a custom provider name.
    pub fn with_provider_name(mut self, name: impl Into<String>) -> Self {
        self.provider_name = name.into();
        self
    }

    /// Set whether to persist responses server-side.
    ///
    /// Default is `false` — responses are only cached in the connection's memory.
    /// Set to `true` if you need responses to survive reconnection.
    pub fn with_store(mut self, store: bool) -> Self {
        self.store = store;
        self
    }

    /// Get the last response ID (for manual chaining).
    pub async fn last_response_id(&self) -> Option<String> {
        self.last_response_id.lock().await.clone()
    }

    /// Get a reference to the underlying WebSocket client.
    pub fn websocket(&self) -> &Arc<ResponsesWebSocket> {
        &self.ws
    }
}

#[async_trait]
impl Provider for OpenAiResponsesWsProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    #[tracing::instrument(name = "provider.ws_chat", skip_all, fields(provider = %self.provider_name, model = %self.model))]
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let (input, system) = convert::messages_to_input(messages);
        let response_tools = tools
            .map(convert::tools_to_response_tools)
            .unwrap_or_default();
        let instructions = system.as_deref().or(options.system.as_deref());

        let prev_id = self.last_response_id.lock().await.clone();

        let mut req = convert::build_request(
            &self.model,
            input,
            instructions,
            if response_tools.is_empty() {
                None
            } else {
                Some(&response_tools)
            },
            options,
            prev_id.as_deref(),
        );

        if !response_tools.is_empty() {
            req.tool_choice = Some(ToolChoice::Mode("auto".to_string()));
        }
        req.store = Some(self.store);

        // Collect the full response from the stream
        let mut stream = self.ws.create_stream(&req);
        let mut last_response = None;

        while let Some(event_result) = stream.next().await {
            match event_result {
                Ok(event) => {
                    if let super::types::ResponseStreamEvent::ResponseCompleted { ref response } =
                        event
                    {
                        last_response = Some(response.clone());
                    }
                }
                Err(e) => return Err(e),
            }
        }

        let resp = last_response.ok_or_else(|| {
            anyhow::anyhow!("WebSocket stream ended without a completed response")
        })?;

        // Store response ID for chaining
        *self.last_response_id.lock().await = Some(resp.id.clone());

        convert::response_to_chat_response(&resp)
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        tracing::info!(provider = %self.provider_name, model = %self.model, "provider.ws_stream started");

        let (input, system) = convert::messages_to_input(messages);
        let response_tools = tools
            .map(convert::tools_to_response_tools)
            .unwrap_or_default();

        Box::pin(async_stream::stream! {
            let instructions = system.as_deref().or(options.system.as_deref());
            let prev_id = self.last_response_id.lock().await.clone();

            let mut req = convert::build_request(
                &self.model,
                input,
                instructions,
                if response_tools.is_empty() { None } else { Some(&response_tools) },
                options,
                prev_id.as_deref(),
            );

            if !response_tools.is_empty() {
                req.tool_choice = Some(ToolChoice::Mode("auto".to_string()));
            }
            req.store = Some(self.store);

            let mut raw_stream = self.ws.create_stream(&req);

            while let Some(event_result) = raw_stream.next().await {
                match event_result {
                    Ok(event) => {
                        // Store response ID from completed events
                        if let super::types::ResponseStreamEvent::ResponseCompleted { ref response } = event {
                            *self.last_response_id.lock().await = Some(response.id.clone());
                        }

                        if let Some(chunks) = convert::stream_event_to_chunk(&event) {
                            for chunk in chunks {
                                yield Ok(chunk);
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(e);
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let ws = Arc::new(ResponsesWebSocket::new("test-key".to_string()));
        let provider = OpenAiResponsesWsProvider::new(ws, "gpt-4o".to_string());
        assert_eq!(provider.name(), "openai-responses-ws");
    }

    #[test]
    fn test_provider_custom_name() {
        let ws = Arc::new(ResponsesWebSocket::new("test-key".to_string()));
        let provider = OpenAiResponsesWsProvider::new(ws, "gpt-4o".to_string())
            .with_provider_name("custom-ws");
        assert_eq!(provider.name(), "custom-ws");
    }

    #[test]
    fn test_store_default_false() {
        let ws = Arc::new(ResponsesWebSocket::new("test-key".to_string()));
        let provider = OpenAiResponsesWsProvider::new(ws, "gpt-4o".to_string());
        assert!(!provider.store);
    }

    #[test]
    fn test_store_configurable() {
        let ws = Arc::new(ResponsesWebSocket::new("test-key".to_string()));
        let provider = OpenAiResponsesWsProvider::new(ws, "gpt-4o".to_string()).with_store(true);
        assert!(provider.store);
    }

    #[tokio::test]
    async fn test_last_response_id_initially_none() {
        let ws = Arc::new(ResponsesWebSocket::new("test-key".to_string()));
        let provider = OpenAiResponsesWsProvider::new(ws, "gpt-4o".to_string());
        assert!(provider.last_response_id().await.is_none());
    }
}
