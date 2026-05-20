//! Provider trait implementation for the OpenAI Responses API.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use tokio::sync::Mutex;

use brainwires_core::{ChatOptions, ChatResponse, Message, Provider, StreamChunk, Tool};

use super::client::ResponsesClient;
use super::convert;
use super::types::ToolChoice;

/// Chat provider backed by the OpenAI Responses API.
///
/// Tracks the last response ID for automatic conversation chaining
/// via `previous_response_id`.
pub struct OpenAiResponsesProvider {
    client: Arc<ResponsesClient>,
    model: String,
    provider_name: String,
    last_response_id: Arc<Mutex<Option<String>>>,
    #[cfg(feature = "telemetry")]
    analytics_collector: Option<std::sync::Arc<brainwires_telemetry::AnalyticsCollector>>,
}

impl OpenAiResponsesProvider {
    /// Create a new Responses API provider.
    pub fn new(client: Arc<ResponsesClient>, model: String) -> Self {
        Self {
            client,
            model,
            provider_name: "openai-responses".to_string(),
            last_response_id: Arc::new(Mutex::new(None)),
            #[cfg(feature = "telemetry")]
            analytics_collector: None,
        }
    }

    /// Set a custom provider name.
    pub fn with_provider_name(mut self, name: impl Into<String>) -> Self {
        self.provider_name = name.into();
        self
    }

    /// Attach an analytics collector to this provider.
    #[cfg(feature = "telemetry")]
    pub fn with_analytics(
        mut self,
        collector: std::sync::Arc<brainwires_telemetry::AnalyticsCollector>,
    ) -> Self {
        self.analytics_collector = Some(collector);
        self
    }

    /// Get the last response ID (for manual chaining).
    pub async fn last_response_id(&self) -> Option<String> {
        self.last_response_id.lock().await.clone()
    }

    /// Get the underlying client.
    pub fn client(&self) -> &Arc<ResponsesClient> {
        &self.client
    }
}

#[async_trait]
impl Provider for OpenAiResponsesProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    #[tracing::instrument(name = "provider.chat", skip_all, fields(provider = %self.provider_name, model = %self.model))]
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

        let effective_model = options.model.as_deref().unwrap_or(&self.model);
        let mut req = convert::build_request(
            effective_model,
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

        // If tools are provided, set tool_choice to auto
        if !response_tools.is_empty() {
            req.tool_choice = Some(ToolChoice::Mode("auto".to_string()));
        }

        #[cfg(feature = "telemetry")]
        let _started = std::time::Instant::now();
        let resp = self.client.create(&req).await?;

        // Store response ID for chaining
        *self.last_response_id.lock().await = Some(resp.id.clone());

        let chat_response = convert::response_to_chat_response(&resp)?;
        #[cfg(feature = "telemetry")]
        if let Some(ref collector) = self.analytics_collector {
            use brainwires_telemetry::AnalyticsEvent;
            collector.record(AnalyticsEvent::ProviderCall {
                session_id: None,
                provider: self.provider_name.clone(),
                model: self.model.clone(),
                prompt_tokens: chat_response.usage.prompt_tokens,
                completion_tokens: chat_response.usage.completion_tokens,
                duration_ms: _started.elapsed().as_millis() as u64,
                cost_usd: 0.0,
                success: true,
                timestamp: chrono::Utc::now(),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                compliance: None,
            });
        }
        Ok(chat_response)
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        tracing::info!(provider = %self.provider_name, model = %self.model, "provider.stream started");

        let (input, system) = convert::messages_to_input(messages);
        let response_tools = tools
            .map(convert::tools_to_response_tools)
            .unwrap_or_default();

        Box::pin(async_stream::stream! {
            let instructions = system.as_deref().or(options.system.as_deref());
            let prev_id = self.last_response_id.lock().await.clone();

            let effective_model = options.model.as_deref().unwrap_or(&self.model);
            let mut req = convert::build_request(
                effective_model,
                input,
                instructions,
                if response_tools.is_empty() { None } else { Some(&response_tools) },
                options,
                prev_id.as_deref(),
            );

            if !response_tools.is_empty() {
                req.tool_choice = Some(ToolChoice::Mode("auto".to_string()));
            }

            let mut raw_stream = self.client.create_stream(&req);

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
        let client = Arc::new(ResponsesClient::new("test-key".to_string()));
        let provider = OpenAiResponsesProvider::new(client, "gpt-4o".to_string());
        assert_eq!(provider.name(), "openai-responses");
    }

    #[test]
    fn test_provider_custom_name() {
        let client = Arc::new(ResponsesClient::new("test-key".to_string()));
        let provider = OpenAiResponsesProvider::new(client, "gpt-4o".to_string())
            .with_provider_name("custom-responses");
        assert_eq!(provider.name(), "custom-responses");
    }

    #[tokio::test]
    async fn test_last_response_id_initially_none() {
        let client = Arc::new(ResponsesClient::new("test-key".to_string()));
        let provider = OpenAiResponsesProvider::new(client, "gpt-4o".to_string());
        assert!(provider.last_response_id().await.is_none());
    }
}
