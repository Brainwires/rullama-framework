//! Chat-layer wrapper around [`GoogleClient`] that implements the
//! [`Provider`] trait from `brainwires_core`.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use serde_json::json;
use uuid::Uuid;

use super::{
    GeminiFunctionCall, GeminiFunctionDeclaration, GeminiFunctionResponse, GeminiGenerationConfig,
    GeminiInlineData, GeminiMessage, GeminiPart, GeminiRequest, GeminiSystemInstruction,
    GeminiToolSet, GoogleClient,
};
use brainwires_core::Provider;
use brainwires_core::{
    ChatOptions, ChatResponse, ContentBlock, ImageSource, Message, MessageContent, Role,
    StreamChunk, Tool, Usage,
};

/// High-level Google Gemini chat provider.
pub struct GoogleChatProvider {
    client: Arc<GoogleClient>,
    model: String,
    #[cfg(feature = "telemetry")]
    analytics_collector: Option<std::sync::Arc<brainwires_telemetry::AnalyticsCollector>>,
}

impl GoogleChatProvider {
    /// Create a new chat provider from an existing client and model name.
    pub fn new(client: Arc<GoogleClient>, model: String) -> Self {
        Self {
            client,
            model,
            #[cfg(feature = "telemetry")]
            analytics_collector: None,
        }
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

    fn convert_messages(messages: &[Message]) -> Vec<GeminiMessage> {
        messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "model",
                    _ => "user",
                };

                let parts = match &m.content {
                    MessageContent::Text(text) => vec![GeminiPart::Text { text: text.clone() }],
                    MessageContent::Blocks(blocks) => blocks
                        .iter()
                        .filter_map(Self::convert_content_block)
                        .collect(),
                };

                GeminiMessage {
                    role: role.to_string(),
                    parts,
                }
            })
            .collect()
    }

    fn convert_content_block(block: &ContentBlock) -> Option<GeminiPart> {
        match block {
            ContentBlock::Text { text } => Some(GeminiPart::Text { text: text.clone() }),
            ContentBlock::Image { source } => match source {
                ImageSource::Base64 { media_type, data } => Some(GeminiPart::InlineData {
                    inline_data: GeminiInlineData {
                        mime_type: media_type.clone(),
                        data: data.clone(),
                    },
                }),
            },
            ContentBlock::ToolUse {
                id: _id,
                name,
                input,
            } => Some(GeminiPart::FunctionCall {
                function_call: GeminiFunctionCall {
                    name: name.clone(),
                    args: input.clone(),
                },
            }),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => Some(GeminiPart::FunctionResponse {
                function_response: GeminiFunctionResponse {
                    name: tool_use_id.clone(),
                    response: json!({ "result": content }),
                },
            }),
        }
    }

    fn convert_tools(tools: &[Tool]) -> Vec<GeminiFunctionDeclaration> {
        tools
            .iter()
            .map(|t| GeminiFunctionDeclaration {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.properties.clone().unwrap_or_default(),
            })
            .collect()
    }

    fn get_system_instruction(messages: &[Message]) -> Option<String> {
        messages
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.text().map(|s| s.to_string()))
    }

    fn build_request(
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> GeminiRequest {
        let contents = Self::convert_messages(messages);

        let system_text = options
            .system
            .clone()
            .or_else(|| Self::get_system_instruction(messages));

        let system_instruction = system_text.map(|text| GeminiSystemInstruction {
            parts: vec![GeminiPart::Text { text }],
        });

        let generation_config = {
            let has_any = options.temperature.is_some()
                || options.max_tokens.is_some()
                || options.top_p.is_some();
            if has_any {
                Some(GeminiGenerationConfig {
                    temperature: options.temperature,
                    max_output_tokens: options.max_tokens,
                    top_p: options.top_p,
                })
            } else {
                None
            }
        };

        let gemini_tools = match tools {
            Some(tools_list) if !tools_list.is_empty() => Some(vec![GeminiToolSet {
                function_declarations: Self::convert_tools(tools_list),
            }]),
            _ => None,
        };

        GeminiRequest {
            contents,
            system_instruction,
            generation_config,
            tools: gemini_tools,
        }
    }

    fn convert_candidate_content(parts: Vec<GeminiPart>) -> MessageContent {
        if parts.len() == 1
            && let GeminiPart::Text { ref text } = parts[0]
        {
            return MessageContent::Text(text.clone());
        }

        MessageContent::Blocks(
            parts
                .into_iter()
                .filter_map(|part| match part {
                    GeminiPart::Text { text } => Some(ContentBlock::Text { text }),
                    GeminiPart::FunctionCall { function_call } => Some(ContentBlock::ToolUse {
                        id: Uuid::new_v4().to_string(),
                        name: function_call.name,
                        input: function_call.args,
                    }),
                    _ => None,
                })
                .collect(),
        )
    }
}

#[async_trait]
impl Provider for GoogleChatProvider {
    fn name(&self) -> &str {
        "google"
    }

    #[tracing::instrument(name = "provider.chat", skip_all, fields(provider = "google", model = %self.model))]
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let request = Self::build_request(messages, tools, options);
        #[cfg(feature = "telemetry")]
        let _started = std::time::Instant::now();
        let gemini_response = if let Some(ref override_model) = options.model {
            self.client
                .generate_content_for_model(override_model, &request)
                .await?
        } else {
            self.client.generate_content(&request).await?
        };

        let candidate = gemini_response
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No candidates in Gemini response"))?;

        let content = Self::convert_candidate_content(candidate.content.parts);

        let usage = gemini_response
            .usage_metadata
            .map(|u| Usage {
                prompt_tokens: u.prompt_token_count,
                completion_tokens: u.candidates_token_count,
                total_tokens: u.total_token_count,
                ..Default::default()
            })
            .unwrap_or_default();

        let chat_response = ChatResponse {
            message: Message {
                role: Role::Assistant,
                content,
                name: None,
                metadata: None,
            },
            usage,
            finish_reason: Some(candidate.finish_reason),
        };
        #[cfg(feature = "telemetry")]
        if let Some(ref collector) = self.analytics_collector {
            use brainwires_telemetry::AnalyticsEvent;
            collector.record(AnalyticsEvent::ProviderCall {
                session_id: None,
                provider: "google".to_string(),
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
        tracing::info!(provider = "google", model = %self.model, "provider.stream started");
        Box::pin(async_stream::stream! {
            let request = Self::build_request(messages, tools, options);
            let mut stream = if let Some(ref override_model) = options.model {
                self.client.stream_generate_content_for_model(override_model.clone(), &request)
            } else {
                self.client.stream_generate_content(&request)
            };

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        if let Some(candidate) = chunk.candidates.into_iter().next() {
                            for part in candidate.content.parts {
                                match part {
                                    GeminiPart::Text { text } => {
                                        yield Ok(StreamChunk::Text(text));
                                    }
                                    GeminiPart::FunctionCall { function_call } => {
                                        yield Ok(StreamChunk::ToolUse {
                                            id: Uuid::new_v4().to_string(),
                                            name: function_call.name,
                                        });
                                    }
                                    _ => {}
                                }
                            }

                            if candidate.finish_reason != "STOP"
                                && !candidate.finish_reason.is_empty()
                            {
                                yield Ok(StreamChunk::Done);
                            }
                        }

                        if let Some(usage) = chunk.usage_metadata {
                            yield Ok(StreamChunk::Usage(Usage {
                                prompt_tokens: usage.prompt_token_count,
                                completion_tokens: usage.candidates_token_count,
                                total_tokens: usage.total_token_count,
                                ..Default::default()
                            }));
                        }
                    }
                    Err(e) => {
                        yield Err(e);
                    }
                }
            }

            yield Ok(StreamChunk::Done);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::ToolInputSchema;
    use std::collections::HashMap;

    #[test]
    fn test_convert_messages_text() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            metadata: None,
        }];

        let converted = GoogleChatProvider::convert_messages(&messages);
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

        let converted = GoogleChatProvider::convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
    }

    #[test]
    fn test_get_system_instruction_found() {
        let messages = vec![
            Message {
                role: Role::System,
                content: MessageContent::Text("You are helpful".to_string()),
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

        let system = GoogleChatProvider::get_system_instruction(&messages);
        assert!(system.is_some());
        assert_eq!(system.unwrap(), "You are helpful");
    }

    #[test]
    fn test_get_system_instruction_not_found() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            metadata: None,
        }];

        let system = GoogleChatProvider::get_system_instruction(&messages);
        assert!(system.is_none());
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

        let converted = GoogleChatProvider::convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].name, "test_tool");
        assert_eq!(converted[0].description, "A test tool");
    }

    #[test]
    fn test_convert_tools_empty() {
        let tools: Vec<Tool> = vec![];

        let converted = GoogleChatProvider::convert_tools(&tools);
        assert_eq!(converted.len(), 0);
    }

    #[test]
    fn test_convert_messages_assistant_role() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: MessageContent::Text("I'm an assistant".to_string()),
            name: None,
            metadata: None,
        }];

        let converted = GoogleChatProvider::convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "model");
    }

    #[test]
    fn test_convert_content_block_text() {
        let block = ContentBlock::Text {
            text: "Test text".to_string(),
        };

        let converted = GoogleChatProvider::convert_content_block(&block);
        assert!(converted.is_some());
        match converted.unwrap() {
            GeminiPart::Text { text } => assert_eq!(text, "Test text"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_convert_content_block_image() {
        let block = ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type: "image/png".to_string(),
                data: "base64data".to_string(),
            },
        };

        let converted = GoogleChatProvider::convert_content_block(&block);
        assert!(converted.is_some());
        match converted.unwrap() {
            GeminiPart::InlineData { inline_data } => {
                assert_eq!(inline_data.mime_type, "image/png");
                assert_eq!(inline_data.data, "base64data");
            }
            _ => panic!("Expected InlineData variant"),
        }
    }

    #[test]
    fn test_convert_content_block_tool_use() {
        let block = ContentBlock::ToolUse {
            id: "tool-123".to_string(),
            name: "test_tool".to_string(),
            input: json!({"arg": "value"}),
        };

        let converted = GoogleChatProvider::convert_content_block(&block);
        assert!(converted.is_some());
        match converted.unwrap() {
            GeminiPart::FunctionCall { function_call } => {
                assert_eq!(function_call.name, "test_tool");
                assert_eq!(function_call.args, json!({"arg": "value"}));
            }
            _ => panic!("Expected FunctionCall variant"),
        }
    }

    #[test]
    fn test_convert_content_block_tool_result() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "tool-123".to_string(),
            content: "Result content".to_string(),
            is_error: Some(false),
        };

        let converted = GoogleChatProvider::convert_content_block(&block);
        assert!(converted.is_some());
        match converted.unwrap() {
            GeminiPart::FunctionResponse { function_response } => {
                assert_eq!(function_response.name, "tool-123");
                assert_eq!(
                    function_response.response,
                    json!({"result": "Result content"})
                );
            }
            _ => panic!("Expected FunctionResponse variant"),
        }
    }

    #[test]
    fn test_convert_messages_empty() {
        let messages: Vec<Message> = vec![];

        let converted = GoogleChatProvider::convert_messages(&messages);
        assert_eq!(converted.len(), 0);
    }

    #[test]
    fn test_build_request_minimal() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            metadata: None,
        }];
        let options = ChatOptions {
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop: None,
            system: None,
            model: None,
            cache_strategy: Default::default(),
        };

        let req = GoogleChatProvider::build_request(&messages, None, &options);
        assert_eq!(req.contents.len(), 1);
        assert!(req.system_instruction.is_none());
        assert!(req.generation_config.is_none());
        assert!(req.tools.is_none());
    }

    #[test]
    fn test_build_request_with_system() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            metadata: None,
        }];
        let options = ChatOptions {
            temperature: None,
            max_tokens: None,
            top_p: None,
            stop: None,
            system: Some("Be helpful".to_string()),
            model: None,
            cache_strategy: Default::default(),
        };

        let req = GoogleChatProvider::build_request(&messages, None, &options);
        assert!(req.system_instruction.is_some());
    }

    #[test]
    fn test_build_request_with_generation_config() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            metadata: None,
        }];
        let options = ChatOptions {
            temperature: Some(0.5),
            max_tokens: Some(1024),
            top_p: None,
            stop: None,
            system: None,
            model: None,
            cache_strategy: Default::default(),
        };

        let req = GoogleChatProvider::build_request(&messages, None, &options);
        assert!(req.generation_config.is_some());
        let gc = req.generation_config.unwrap();
        assert_eq!(gc.temperature, Some(0.5));
        assert_eq!(gc.max_output_tokens, Some(1024));
    }

    #[test]
    fn test_convert_candidate_content_single_text() {
        let parts = vec![GeminiPart::Text {
            text: "Hello world".to_string(),
        }];
        let content = GoogleChatProvider::convert_candidate_content(parts);
        match content {
            MessageContent::Text(t) => assert_eq!(t, "Hello world"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_convert_candidate_content_multiple_parts() {
        let parts = vec![
            GeminiPart::Text {
                text: "Part 1".to_string(),
            },
            GeminiPart::Text {
                text: "Part 2".to_string(),
            },
        ];
        let content = GoogleChatProvider::convert_candidate_content(parts);
        match content {
            MessageContent::Blocks(blocks) => assert_eq!(blocks.len(), 2),
            _ => panic!("Expected Blocks variant"),
        }
    }

    #[test]
    fn test_convert_candidate_content_with_function_call() {
        let parts = vec![GeminiPart::FunctionCall {
            function_call: GeminiFunctionCall {
                name: "do_thing".to_string(),
                args: json!({"a": 1}),
            },
        }];
        let content = GoogleChatProvider::convert_candidate_content(parts);
        match content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    ContentBlock::ToolUse { name, input, .. } => {
                        assert_eq!(name, "do_thing");
                        assert_eq!(*input, json!({"a": 1}));
                    }
                    _ => panic!("Expected ToolUse block"),
                }
            }
            _ => panic!("Expected Blocks variant"),
        }
    }
}
