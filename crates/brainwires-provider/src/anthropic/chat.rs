use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;

use super::{
    AnthropicClient, AnthropicContentBlock, AnthropicMessage, AnthropicRequest, AnthropicResponse,
    AnthropicTool,
};
use brainwires_core::{
    ChatOptions, ChatResponse, ContentBlock, ImageSource, Message, MessageContent, Provider, Role,
    StreamChunk, Tool, Usage,
};

use super::AnthropicImageSource;

// ---------------------------------------------------------------------------
// Chat provider
// ---------------------------------------------------------------------------

/// High-level chat provider that wraps an [`AnthropicClient`] and implements
/// the `Provider` trait from `brainwires_core`.
pub struct AnthropicChatProvider {
    client: Arc<AnthropicClient>,
    model: String,
    provider_name: String,
    #[cfg(feature = "telemetry")]
    analytics_collector: Option<std::sync::Arc<brainwires_telemetry::AnalyticsCollector>>,
}

impl AnthropicChatProvider {
    /// Create a new chat provider backed by the given client.
    pub fn new(client: Arc<AnthropicClient>, model: String) -> Self {
        Self {
            client,
            model,
            provider_name: "anthropic".to_string(),
            #[cfg(feature = "telemetry")]
            analytics_collector: None,
        }
    }

    /// Override the provider name reported by [`Provider::name`].
    ///
    /// This is useful for Bedrock / Vertex AI variants that share the same
    /// chat logic but should identify themselves differently.
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

    // -----------------------------------------------------------------------
    // Conversion helpers
    // -----------------------------------------------------------------------

    /// Convert core `Message` values to Anthropic-native messages.
    fn convert_messages(messages: &[Message]) -> Vec<AnthropicMessage> {
        messages
            .iter()
            .filter(|m| m.role != Role::System) // System goes in separate field
            .map(|m| AnthropicMessage {
                role: match m.role {
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                    _ => "user".to_string(),
                },
                content: match &m.content {
                    MessageContent::Text(text) => {
                        vec![AnthropicContentBlock::Text { text: text.clone() }]
                    }
                    MessageContent::Blocks(blocks) => blocks
                        .iter()
                        .map(|b| match b {
                            ContentBlock::Text { text } => {
                                AnthropicContentBlock::Text { text: text.clone() }
                            }
                            ContentBlock::Image { source } => match source {
                                ImageSource::Base64 { media_type, data } => {
                                    AnthropicContentBlock::Image {
                                        source: AnthropicImageSource::Base64 {
                                            media_type: media_type.clone(),
                                            data: data.clone(),
                                        },
                                    }
                                }
                            },
                            ContentBlock::ToolUse { id, name, input } => {
                                AnthropicContentBlock::ToolUse {
                                    id: id.clone(),
                                    name: name.clone(),
                                    input: input.clone(),
                                }
                            }
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                ..
                            } => AnthropicContentBlock::ToolResult {
                                tool_use_id: tool_use_id.clone(),
                                content: content.clone(),
                            },
                        })
                        .collect(),
                },
            })
            .collect()
    }

    /// Convert core `Tool` values to Anthropic-native tool definitions.
    fn convert_tools(tools: &[Tool]) -> Vec<AnthropicTool> {
        tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.input_schema.properties.clone().unwrap_or_default(),
            })
            .collect()
    }

    /// Map a [`CacheStrategy`](brainwires_core::provider::CacheStrategy) to the Anthropic request's `cache_prompt` bool.
    ///
    /// Today the underlying builder at `anthropic/mod.rs::build_body` treats
    /// `cache_prompt` as a combined "system + tools" switch, so `Off` and
    /// `SystemOnly` both decay to "no cache" in practice until we refine the
    /// request path to emit breakpoints per-field. `SystemAndTools` and
    /// `SystemAndTailTurn` both enable it.
    fn cache_prompt_enabled(strategy: brainwires_core::CacheStrategy) -> bool {
        use brainwires_core::CacheStrategy::*;
        matches!(strategy, SystemAndTools | SystemAndTailTurn { .. })
    }

    /// Extract the first system message from the message list.
    fn get_system_message(messages: &[Message]) -> Option<String> {
        messages
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.text().map(|s| s.to_string()))
    }

    /// Parse an `AnthropicResponse` into a core `ChatResponse`.
    fn parse_response(response: AnthropicResponse) -> ChatResponse {
        if let Some(read) = response.usage.cache_read_input_tokens
            && read > 0
        {
            tracing::info!(
                cache_read_input_tokens = read,
                cache_creation_input_tokens =
                    response.usage.cache_creation_input_tokens.unwrap_or(0),
                "prompt cache hit",
            );
        } else if let Some(write) = response.usage.cache_creation_input_tokens
            && write > 0
        {
            tracing::debug!(cache_creation_input_tokens = write, "prompt cache write");
        }
        let content = if response.content.len() == 1 {
            match &response.content[0] {
                AnthropicContentBlock::Text { text } => MessageContent::Text(text.clone()),
                _ => MessageContent::Blocks(
                    response
                        .content
                        .into_iter()
                        .filter_map(|block| match block {
                            AnthropicContentBlock::Text { text } => {
                                Some(ContentBlock::Text { text })
                            }
                            AnthropicContentBlock::ToolUse { id, name, input } => {
                                Some(ContentBlock::ToolUse { id, name, input })
                            }
                            _ => None,
                        })
                        .collect(),
                ),
            }
        } else {
            MessageContent::Blocks(
                response
                    .content
                    .into_iter()
                    .filter_map(|block| match block {
                        AnthropicContentBlock::Text { text } => Some(ContentBlock::Text { text }),
                        AnthropicContentBlock::ToolUse { id, name, input } => {
                            Some(ContentBlock::ToolUse { id, name, input })
                        }
                        _ => None,
                    })
                    .collect(),
            )
        };

        ChatResponse {
            message: Message {
                role: Role::Assistant,
                content,
                name: None,
                metadata: None,
            },
            usage: Usage {
                prompt_tokens: response.usage.input_tokens,
                completion_tokens: response.usage.output_tokens,
                total_tokens: response.usage.input_tokens + response.usage.output_tokens,
                cache_creation_input_tokens: response
                    .usage
                    .cache_creation_input_tokens
                    .unwrap_or(0),
                cache_read_input_tokens: response.usage.cache_read_input_tokens.unwrap_or(0),
            },
            finish_reason: Some(response.stop_reason),
        }
    }
}

#[async_trait]
impl Provider for AnthropicChatProvider {
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
        let anthropic_messages = Self::convert_messages(messages);
        let system = options
            .system
            .clone()
            .or_else(|| Self::get_system_message(messages));

        let req = AnthropicRequest {
            model: options.model.clone().unwrap_or_else(|| self.model.clone()),
            messages: anthropic_messages,
            system,
            max_tokens: options.max_tokens.unwrap_or(4096),
            temperature: options.temperature,
            top_p: None,
            stop_sequences: None,
            tools: tools.map(Self::convert_tools),
            stream: false,
            cache_prompt: Self::cache_prompt_enabled(options.cache_strategy),
        };

        #[cfg(feature = "telemetry")]
        let _started = std::time::Instant::now();
        let anthropic_response = self.client.messages(&req).await?;
        let chat_response = Self::parse_response(anthropic_response);
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
                cache_creation_input_tokens: chat_response.usage.cache_creation_input_tokens,
                cache_read_input_tokens: chat_response.usage.cache_read_input_tokens,
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
        Box::pin(async_stream::stream! {
            let anthropic_messages = Self::convert_messages(messages);
            let system = options
                .system
                .clone()
                .or_else(|| Self::get_system_message(messages));

            let req = AnthropicRequest {
                model: options.model.clone().unwrap_or_else(|| self.model.clone()),
                messages: anthropic_messages,
                system,
                max_tokens: options.max_tokens.unwrap_or(4096),
                temperature: options.temperature,
                top_p: None,
                stop_sequences: None,
                tools: tools.map(Self::convert_tools),
                stream: true,
                cache_prompt: Self::cache_prompt_enabled(options.cache_strategy),
            };

            let mut stream = self.client.stream_messages(&req);

            use futures::StreamExt;
            while let Some(event_result) = stream.next().await {
                match event_result {
                    Ok(event) => {
                        match event.event_type.as_str() {
                            "content_block_delta" => {
                                if let Some(delta) = event.delta
                                    && let Some(text) = delta.text {
                                        yield Ok(StreamChunk::Text(text));
                                    }
                            }
                            "message_delta" => {
                                if let Some(usage) = event.usage {
                                    yield Ok(StreamChunk::Usage(Usage {
                                        prompt_tokens: 0,
                                        completion_tokens: usage.output_tokens,
                                        total_tokens: usage.output_tokens,
                                        cache_creation_input_tokens: usage
                                            .cache_creation_input_tokens
                                            .unwrap_or(0),
                                        cache_read_input_tokens: usage
                                            .cache_read_input_tokens
                                            .unwrap_or(0),
                                    }));
                                }
                            }
                            "message_stop" => {
                                yield Ok(StreamChunk::Done);
                            }
                            "context_window_management_event" => {
                                let summary = event.summary.unwrap_or_default();
                                let tokens_freed = event.tokens_freed;
                                yield Ok(StreamChunk::ContextCompacted { summary, tokens_freed });
                            }
                            _ => {}
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
    use brainwires_core::ToolInputSchema;
    use serde_json::json;
    use std::collections::HashMap;

    // Helper: build a throwaway client wrapped in Arc (never hits the network
    // in these unit tests).
    fn dummy_client() -> Arc<AnthropicClient> {
        Arc::new(AnthropicClient::new(
            "test-key".to_string(),
            "claude-sonnet-4-6".to_string(),
        ))
    }

    fn provider() -> AnthropicChatProvider {
        AnthropicChatProvider::new(dummy_client(), "claude-sonnet-4-6".to_string())
    }

    #[test]
    fn test_provider_name() {
        let p = provider();
        assert_eq!(p.name(), "anthropic");
    }

    #[test]
    fn test_provider_name_override() {
        let p = provider().with_provider_name("bedrock");
        assert_eq!(p.name(), "bedrock");
    }

    #[test]
    fn test_convert_messages_text() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            metadata: None,
        }];

        let converted = AnthropicChatProvider::convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
    }

    #[test]
    fn test_convert_messages_filters_system() {
        let messages = vec![
            Message {
                role: Role::System,
                content: MessageContent::Text("System prompt".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::Text("Hello".to_string()),
                name: None,
                metadata: None,
            },
        ];

        let converted = AnthropicChatProvider::convert_messages(&messages);
        // System message should be filtered out
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
    }

    #[test]
    fn test_convert_messages_with_blocks() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Response".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "tool-1".to_string(),
                    name: "test_tool".to_string(),
                    input: json!({"arg": "value"}),
                },
            ]),
            name: None,
            metadata: None,
        }];

        let converted = AnthropicChatProvider::convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "assistant");
        assert_eq!(converted[0].content.len(), 2);
    }

    #[test]
    fn test_convert_messages_image_block_roundtrips_as_anthropic_image() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "What's in this picture?".to_string(),
                },
                ContentBlock::Image {
                    source: ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: "iVBORw0KGgo=".to_string(),
                    },
                },
            ]),
            name: None,
            metadata: None,
        }];

        let converted = AnthropicChatProvider::convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].content.len(), 2);

        // Second block must serialise as Anthropic's native image envelope so
        // the provider crate stays the single source of truth for the wire
        // format.
        let json = serde_json::to_value(&converted[0]).unwrap();
        let blocks = json["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["source"]["type"], "base64");
        assert_eq!(blocks[1]["source"]["media_type"], "image/png");
        assert_eq!(blocks[1]["source"]["data"], "iVBORw0KGgo=");
    }

    #[test]
    fn test_convert_messages_with_tool_result() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                tool_use_id: "tool-1".to_string(),
                content: "Result".to_string(),
                is_error: Some(false),
            }]),
            name: None,
            metadata: None,
        }];

        let converted = AnthropicChatProvider::convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].content.len(), 1);
    }

    #[test]
    fn test_convert_tools() {
        let mut properties = HashMap::new();
        properties.insert(
            "arg1".to_string(),
            json!({
                "type": "string",
                "description": "First argument"
            }),
        );

        let tools = vec![Tool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: ToolInputSchema::object(properties.clone(), vec!["arg1".to_string()]),
            requires_approval: false,
            ..Default::default()
        }];

        let converted = AnthropicChatProvider::convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].name, "test_tool");
        assert_eq!(converted[0].description, "A test tool");
        assert!(converted[0].input_schema.contains_key("arg1"));
    }

    #[test]
    fn test_convert_tools_empty() {
        let tools: Vec<Tool> = vec![];

        let converted = AnthropicChatProvider::convert_tools(&tools);
        assert_eq!(converted.len(), 0);
    }

    #[test]
    fn test_get_system_message_found() {
        let messages = vec![
            Message {
                role: Role::System,
                content: MessageContent::Text("You are a helpful assistant".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::Text("Hello".to_string()),
                name: None,
                metadata: None,
            },
        ];

        let system = AnthropicChatProvider::get_system_message(&messages);
        assert!(system.is_some());
        assert_eq!(system.unwrap(), "You are a helpful assistant");
    }

    #[test]
    fn test_get_system_message_not_found() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            metadata: None,
        }];

        let system = AnthropicChatProvider::get_system_message(&messages);
        assert!(system.is_none());
    }

    #[test]
    fn test_convert_messages_multiple_roles() {
        let messages = vec![
            Message {
                role: Role::User,
                content: MessageContent::Text("Question".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("Answer".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::Text("Follow-up".to_string()),
                name: None,
                metadata: None,
            },
        ];

        let converted = AnthropicChatProvider::convert_messages(&messages);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].role, "user");
        assert_eq!(converted[1].role, "assistant");
        assert_eq!(converted[2].role, "user");
    }

    #[test]
    fn test_convert_tools_multiple() {
        let tools = vec![
            Tool {
                name: "tool1".to_string(),
                description: "First tool".to_string(),
                input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "tool2".to_string(),
                description: "Second tool".to_string(),
                input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
                requires_approval: true,
                ..Default::default()
            },
        ];

        let converted = AnthropicChatProvider::convert_tools(&tools);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].name, "tool1");
        assert_eq!(converted[1].name, "tool2");
    }

    #[test]
    fn test_convert_messages_empty_list() {
        let messages: Vec<Message> = vec![];

        let converted = AnthropicChatProvider::convert_messages(&messages);
        assert_eq!(converted.len(), 0);
    }

    #[test]
    fn test_cache_prompt_enabled_maps_strategy() {
        use brainwires_core::CacheStrategy;
        assert!(!AnthropicChatProvider::cache_prompt_enabled(
            CacheStrategy::Off
        ));
        assert!(!AnthropicChatProvider::cache_prompt_enabled(
            CacheStrategy::SystemOnly
        ));
        assert!(AnthropicChatProvider::cache_prompt_enabled(
            CacheStrategy::SystemAndTools
        ));
        assert!(AnthropicChatProvider::cache_prompt_enabled(
            CacheStrategy::SystemAndTailTurn {
                threshold_tokens: 2000
            }
        ));
    }

    #[test]
    fn test_parse_response_propagates_cache_tokens() {
        let response = crate::anthropic::AnthropicResponse {
            content: vec![AnthropicContentBlock::Text { text: "ok".into() }],
            stop_reason: "end_turn".into(),
            usage: crate::anthropic::AnthropicUsage {
                input_tokens: 100,
                output_tokens: 20,
                cache_creation_input_tokens: Some(50),
                cache_read_input_tokens: Some(800),
            },
        };
        let cr = AnthropicChatProvider::parse_response(response);
        assert_eq!(cr.usage.cache_creation_input_tokens, 50);
        assert_eq!(cr.usage.cache_read_input_tokens, 800);
    }

    #[test]
    fn test_parse_response_single_text() {
        let response = crate::anthropic::AnthropicResponse {
            content: vec![AnthropicContentBlock::Text {
                text: "Hello!".to_string(),
            }],
            stop_reason: "end_turn".to_string(),
            usage: crate::anthropic::AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let chat_response = AnthropicChatProvider::parse_response(response);
        assert_eq!(chat_response.message.role, Role::Assistant);
        assert_eq!(chat_response.usage.prompt_tokens, 10);
        assert_eq!(chat_response.usage.completion_tokens, 5);
        assert_eq!(chat_response.usage.total_tokens, 15);
        assert_eq!(chat_response.finish_reason, Some("end_turn".to_string()));

        if let MessageContent::Text(text) = &chat_response.message.content {
            assert_eq!(text, "Hello!");
        } else {
            panic!("Expected Text content for single text response");
        }
    }

    #[test]
    fn test_parse_response_multiple_blocks() {
        let response = crate::anthropic::AnthropicResponse {
            content: vec![
                AnthropicContentBlock::Text {
                    text: "Let me search".to_string(),
                },
                AnthropicContentBlock::ToolUse {
                    id: "call_1".to_string(),
                    name: "search".to_string(),
                    input: json!({"q": "test"}),
                },
            ],
            stop_reason: "tool_use".to_string(),
            usage: crate::anthropic::AnthropicUsage {
                input_tokens: 20,
                output_tokens: 15,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };

        let chat_response = AnthropicChatProvider::parse_response(response);
        if let MessageContent::Blocks(blocks) = &chat_response.message.content {
            assert_eq!(blocks.len(), 2);
        } else {
            panic!("Expected Blocks content for multi-block response");
        }
    }
}
