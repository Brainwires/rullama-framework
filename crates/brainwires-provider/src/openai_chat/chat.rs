use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};

use super::{
    OpenAIContent, OpenAIContentPart, OpenAIFunction, OpenAIImageUrl, OpenAIMessage,
    OpenAIStreamChunk, OpenAITool, OpenAiClient, OpenAiRequestOptions,
};
use brainwires_core::{
    ChatOptions, ChatResponse, ContentBlock, ImageSource, Message, MessageContent, Provider, Role,
    StreamChunk, Tool, Usage,
};

// ---------------------------------------------------------------------------
// Chat wrapper
// ---------------------------------------------------------------------------

/// High-level chat provider that adapts `OpenAiClient` to the
/// `brainwires_core::Provider` trait.
///
/// All `brainwires_core` message / tool types are converted to and from the
/// OpenAI wire format automatically.
pub struct OpenAiChatProvider {
    client: Arc<OpenAiClient>,
    model: String,
    provider_name: String,
    #[cfg(feature = "telemetry")]
    analytics_collector: Option<std::sync::Arc<brainwires_telemetry::AnalyticsCollector>>,
}

impl OpenAiChatProvider {
    /// Create a new chat provider backed by the given client.
    pub fn new(client: Arc<OpenAiClient>, model: String) -> Self {
        Self {
            client,
            model,
            provider_name: "openai".to_string(),
            #[cfg(feature = "telemetry")]
            analytics_collector: None,
        }
    }

    /// Override the provider name reported by [`Provider::name`].
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
}

#[async_trait]
impl Provider for OpenAiChatProvider {
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
        let openai_messages = convert_messages(messages);
        let openai_tools: Vec<OpenAITool> = tools.map(convert_tools).unwrap_or_default();
        let tools_ref: Option<&[OpenAITool]> = if openai_tools.is_empty() {
            None
        } else {
            Some(&openai_tools)
        };

        let opts = chat_options_to_request_options(options);

        // O1 models don't support streaming, temperature, max_tokens, or
        // system messages - the client handles the option filtering.
        let effective_model = options.model.as_deref().unwrap_or(&self.model);
        #[cfg(feature = "telemetry")]
        let _started = std::time::Instant::now();
        let openai_response = self
            .client
            .chat_completions(&openai_messages, effective_model, tools_ref, &opts)
            .await?;

        let chat_response = parse_response(openai_response)?;
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

        // O1 models don't support streaming - fall back to non-streaming.
        let effective_model: &str = options.model.as_deref().unwrap_or(&self.model);
        if OpenAiClient::is_o1_model(effective_model) {
            return Box::pin(async_stream::stream! {
                match self.chat(messages, tools, options).await {
                    Ok(response) => {
                        if let Some(text) = response.message.text() {
                            yield Ok(StreamChunk::Text(text.to_string()));
                        }
                        yield Ok(StreamChunk::Usage(response.usage));
                        yield Ok(StreamChunk::Done);
                    }
                    Err(e) => {
                        yield Err(e);
                    }
                }
            });
        }

        let effective_model_owned = effective_model.to_string();
        let openai_messages = convert_messages(messages);
        let openai_tools: Vec<OpenAITool> = tools.map(convert_tools).unwrap_or_default();

        let opts = chat_options_to_request_options(options);

        Box::pin(async_stream::stream! {
            let tools_ref: Option<&[OpenAITool]> = if openai_tools.is_empty() {
                None
            } else {
                Some(&openai_tools)
            };

            let mut raw_stream = self
                .client
                .stream_chat_completions(&openai_messages, &effective_model_owned, tools_ref, &opts);

            while let Some(chunk_result) = raw_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        // Convert each raw OpenAI stream chunk into our
                        // StreamChunk variants.
                        for stream_chunk in convert_stream_chunk(chunk) {
                            yield Ok(stream_chunk);
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

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Map `ChatOptions` to the provider-specific `OpenAiRequestOptions`.
fn chat_options_to_request_options(options: &ChatOptions) -> OpenAiRequestOptions {
    OpenAiRequestOptions {
        temperature: options.temperature,
        max_tokens: options.max_tokens,
        top_p: options.top_p,
        stop: None,
        system: None,
    }
}

/// Convert brainwires-core `Message` values into OpenAI wire messages.
pub fn convert_messages(messages: &[Message]) -> Vec<OpenAIMessage> {
    messages
        .iter()
        .map(|m| {
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
                Role::Tool => "tool",
            };

            let content = match &m.content {
                MessageContent::Text(text) => OpenAIContent::Text(text.clone()),
                MessageContent::Blocks(blocks) => {
                    if blocks.len() == 1 {
                        match &blocks[0] {
                            ContentBlock::Text { text } => OpenAIContent::Text(text.clone()),
                            _ => OpenAIContent::Array(
                                blocks.iter().filter_map(convert_content_block).collect(),
                            ),
                        }
                    } else {
                        OpenAIContent::Array(
                            blocks.iter().filter_map(convert_content_block).collect(),
                        )
                    }
                }
            };

            OpenAIMessage {
                role: role.to_string(),
                content,
                name: m.name.clone(),
                tool_calls: None,
                tool_call_id: None,
            }
        })
        .collect()
}

/// Convert a single `ContentBlock` to the OpenAI content-part format.
fn convert_content_block(block: &ContentBlock) -> Option<OpenAIContentPart> {
    match block {
        ContentBlock::Text { text } => Some(OpenAIContentPart::Text { text: text.clone() }),
        ContentBlock::Image { source } => match source {
            ImageSource::Base64 { media_type, data } => Some(OpenAIContentPart::ImageUrl {
                image_url: OpenAIImageUrl {
                    url: format!("data:{};base64,{}", media_type, data),
                },
            }),
        },
        _ => None,
    }
}

/// Convert brainwires-core `Tool` values into OpenAI wire tools.
pub fn convert_tools(tools: &[Tool]) -> Vec<OpenAITool> {
    tools
        .iter()
        .map(|t| OpenAITool {
            r#type: "function".to_string(),
            function: OpenAIFunction {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.properties.clone().unwrap_or_default(),
            },
        })
        .collect()
}

/// Parse a raw `OpenAIResponse` into the brainwires-core `ChatResponse`.
pub fn parse_response(openai_response: super::OpenAIResponse) -> Result<ChatResponse> {
    let usage = Usage {
        prompt_tokens: openai_response.usage.prompt_tokens,
        completion_tokens: openai_response.usage.completion_tokens,
        total_tokens: openai_response.usage.total_tokens,
        ..Default::default()
    };

    let choice = openai_response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No choices in OpenAI response"))?;

    let content = match choice.message.content {
        OpenAIContent::Text(text) => MessageContent::Text(text),
        OpenAIContent::Array(parts) => MessageContent::Blocks(
            parts
                .into_iter()
                .filter_map(|part| match part {
                    OpenAIContentPart::Text { text, .. } => Some(ContentBlock::Text { text }),
                    _ => None,
                })
                .collect(),
        ),
    };

    Ok(ChatResponse {
        message: Message {
            role: Role::Assistant,
            content,
            name: None,
            metadata: None,
        },
        usage,
        finish_reason: Some(choice.finish_reason),
    })
}

/// Convert a raw `OpenAIStreamChunk` into zero or more `StreamChunk` values.
fn convert_stream_chunk(chunk: OpenAIStreamChunk) -> Vec<StreamChunk> {
    let mut out = Vec::new();

    for choice in chunk.choices {
        if let Some(delta) = choice.delta {
            if let Some(content) = delta.content {
                out.push(StreamChunk::Text(content));
            }
            if let Some(tool_calls) = delta.tool_calls {
                for tool_call in tool_calls {
                    out.push(StreamChunk::ToolUse {
                        id: tool_call.id.unwrap_or_default(),
                        name: tool_call.function.name.unwrap_or_default(),
                    });
                }
            }
        }
    }

    if let Some(usage) = chunk.usage {
        out.push(StreamChunk::Usage(Usage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            ..Default::default()
        }));
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::ToolInputSchema;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn test_convert_messages_text() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            metadata: None,
        }];

        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
    }

    #[test]
    fn test_convert_messages_system() {
        let messages = vec![Message {
            role: Role::System,
            content: MessageContent::Text("You are helpful".to_string()),
            name: None,
            metadata: None,
        }];

        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "system");
    }

    #[test]
    fn test_convert_messages_assistant_role() {
        let messages = vec![Message {
            role: Role::Assistant,
            content: MessageContent::Text("I can help with that".to_string()),
            name: None,
            metadata: None,
        }];

        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "assistant");
    }

    #[test]
    fn test_convert_messages_tool_role() {
        let messages = vec![Message {
            role: Role::Tool,
            content: MessageContent::Text("Tool response".to_string()),
            name: None,
            metadata: None,
        }];

        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "tool");
    }

    #[test]
    fn test_convert_messages_with_name() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text("Hello".to_string()),
            name: Some("user_1".to_string()),
            metadata: None,
        }];

        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].name, Some("user_1".to_string()));
    }

    #[test]
    fn test_convert_messages_with_text_block() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::Text {
                text: "Hello world".to_string(),
            }]),
            name: None,
            metadata: None,
        }];

        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        match &converted[0].content {
            OpenAIContent::Text(text) => assert_eq!(text, "Hello world"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_convert_messages_with_multiple_blocks() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "First block".to_string(),
                },
                ContentBlock::Text {
                    text: "Second block".to_string(),
                },
            ]),
            name: None,
            metadata: None,
        }];

        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        match &converted[0].content {
            OpenAIContent::Array(parts) => assert_eq!(parts.len(), 2),
            _ => panic!("Expected array content"),
        }
    }

    #[test]
    fn test_convert_messages_with_image_block() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: "image/png".to_string(),
                    data: "base64data".to_string(),
                },
            }]),
            name: None,
            metadata: None,
        }];

        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        match &converted[0].content {
            OpenAIContent::Array(parts) => {
                assert_eq!(parts.len(), 1);
                match &parts[0] {
                    OpenAIContentPart::ImageUrl { image_url } => {
                        assert!(image_url.url.starts_with("data:image/png;base64,"));
                    }
                    _ => panic!("Expected image url content"),
                }
            }
            _ => panic!("Expected array content"),
        }
    }

    #[test]
    fn test_convert_messages_mixed_text_and_image() {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Check this image".to_string(),
                },
                ContentBlock::Image {
                    source: ImageSource::Base64 {
                        media_type: "image/jpeg".to_string(),
                        data: "imagedata".to_string(),
                    },
                },
            ]),
            name: None,
            metadata: None,
        }];

        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        match &converted[0].content {
            OpenAIContent::Array(parts) => assert_eq!(parts.len(), 2),
            _ => panic!("Expected array content"),
        }
    }

    #[test]
    fn test_convert_content_block_text() {
        let block = ContentBlock::Text {
            text: "Test text".to_string(),
        };

        let converted = convert_content_block(&block);
        assert!(converted.is_some());
        match converted.unwrap() {
            OpenAIContentPart::Text { text } => assert_eq!(text, "Test text"),
            _ => panic!("Expected text part"),
        }
    }

    #[test]
    fn test_convert_content_block_image() {
        let block = ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type: "image/webp".to_string(),
                data: "webpdata".to_string(),
            },
        };

        let converted = convert_content_block(&block);
        assert!(converted.is_some());
        match converted.unwrap() {
            OpenAIContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "data:image/webp;base64,webpdata");
            }
            _ => panic!("Expected image url part"),
        }
    }

    #[test]
    fn test_convert_content_block_tool_use() {
        let block = ContentBlock::ToolUse {
            id: "tool-1".to_string(),
            name: "test_tool".to_string(),
            input: json!({"key": "value"}),
        };

        // Tool use blocks should return None for OpenAI
        let converted = convert_content_block(&block);
        assert!(converted.is_none());
    }

    #[test]
    fn test_convert_messages_preserves_order() {
        let messages = vec![
            Message {
                role: Role::System,
                content: MessageContent::Text("System message".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::Text("User message 1".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("Assistant message".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::Text("User message 2".to_string()),
                name: None,
                metadata: None,
            },
        ];

        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 4);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[1].role, "user");
        assert_eq!(converted[2].role, "assistant");
        assert_eq!(converted[3].role, "user");
    }

    #[test]
    fn test_convert_messages_empty() {
        let messages: Vec<Message> = vec![];

        let converted = convert_messages(&messages);
        assert_eq!(converted.len(), 0);
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

        let converted = convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].r#type, "function");
        assert_eq!(converted[0].function.name, "test_tool");
    }

    #[test]
    fn test_convert_tools_empty() {
        let tools: Vec<Tool> = vec![];

        let converted = convert_tools(&tools);
        assert_eq!(converted.len(), 0);
    }

    #[test]
    fn test_convert_tools_multiple() {
        let mut properties1 = HashMap::new();
        properties1.insert("arg1".to_string(), json!({"type": "string"}));

        let mut properties2 = HashMap::new();
        properties2.insert("arg2".to_string(), json!({"type": "number"}));

        let tools = vec![
            Tool {
                name: "tool1".to_string(),
                description: "First tool".to_string(),
                input_schema: ToolInputSchema::object(properties1, vec![]),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "tool2".to_string(),
                description: "Second tool".to_string(),
                input_schema: ToolInputSchema::object(properties2, vec![]),
                requires_approval: true,
                ..Default::default()
            },
        ];

        let converted = convert_tools(&tools);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].function.name, "tool1");
        assert_eq!(converted[1].function.name, "tool2");
    }

    #[test]
    fn test_convert_tools_without_properties() {
        let tools = vec![Tool {
            name: "simple_tool".to_string(),
            description: "A simple tool".to_string(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: None,
                required: None,
            },
            requires_approval: false,
            ..Default::default()
        }];

        let converted = convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].function.name, "simple_tool");
        assert!(converted[0].function.parameters.is_empty());
    }

    #[test]
    fn test_convert_tools_with_complex_parameters() {
        let mut properties = HashMap::new();
        properties.insert(
            "location".to_string(),
            json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"}
                },
                "required": ["city"]
            }),
        );
        properties.insert(
            "units".to_string(),
            json!({
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
            }),
        );

        let tools = vec![Tool {
            name: "get_weather".to_string(),
            description: "Get weather for a location".to_string(),
            input_schema: ToolInputSchema::object(properties.clone(), vec!["location".to_string()]),
            requires_approval: false,
            ..Default::default()
        }];

        let converted = convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].function.name, "get_weather");
        assert_eq!(converted[0].function.parameters.len(), 2);
        assert!(converted[0].function.parameters.contains_key("location"));
        assert!(converted[0].function.parameters.contains_key("units"));
    }

    #[test]
    fn test_different_image_media_types() {
        let media_types = vec!["image/png", "image/jpeg", "image/webp", "image/gif"];

        for media_type in media_types {
            let block = ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: media_type.to_string(),
                    data: "data123".to_string(),
                },
            };

            let converted = convert_content_block(&block);
            assert!(converted.is_some());
            match converted.unwrap() {
                OpenAIContentPart::ImageUrl { image_url } => {
                    assert!(
                        image_url
                            .url
                            .starts_with(&format!("data:{};base64,", media_type))
                    );
                }
                _ => panic!("Expected image url part"),
            }
        }
    }

    #[test]
    fn test_parse_response_basic() {
        use crate::openai_chat::{
            OpenAIChoice, OpenAIContent, OpenAIResponse, OpenAIResponseMessage, OpenAIUsage,
        };

        let response = OpenAIResponse {
            choices: vec![OpenAIChoice {
                message: OpenAIResponseMessage {
                    content: OpenAIContent::Text("Hello!".to_string()),
                    tool_calls: None,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };

        let chat_response = parse_response(response).unwrap();
        assert_eq!(chat_response.message.role, Role::Assistant);
        assert_eq!(chat_response.usage.prompt_tokens, 10);
        assert_eq!(chat_response.usage.completion_tokens, 5);
        assert_eq!(chat_response.usage.total_tokens, 15);
        assert_eq!(chat_response.finish_reason, Some("stop".to_string()));
    }

    #[test]
    fn test_parse_response_no_choices() {
        use crate::openai_chat::{OpenAIResponse, OpenAIUsage};

        let response = OpenAIResponse {
            choices: vec![],
            usage: OpenAIUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };

        let result = parse_response(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_stream_chunk_text() {
        use crate::openai_chat::{OpenAIStreamChoice, OpenAIStreamDelta};

        let chunk = OpenAIStreamChunk {
            choices: vec![OpenAIStreamChoice {
                delta: Some(OpenAIStreamDelta {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                }),
            }],
            usage: None,
        };

        let converted = convert_stream_chunk(chunk);
        assert_eq!(converted.len(), 1);
        match &converted[0] {
            StreamChunk::Text(text) => assert_eq!(text, "Hello"),
            _ => panic!("Expected text chunk"),
        }
    }

    #[test]
    fn test_convert_stream_chunk_usage() {
        use crate::openai_chat::OpenAIUsage;

        let chunk = OpenAIStreamChunk {
            choices: vec![],
            usage: Some(OpenAIUsage {
                prompt_tokens: 20,
                completion_tokens: 10,
                total_tokens: 30,
            }),
        };

        let converted = convert_stream_chunk(chunk);
        assert_eq!(converted.len(), 1);
        match &converted[0] {
            StreamChunk::Usage(u) => {
                assert_eq!(u.prompt_tokens, 20);
                assert_eq!(u.completion_tokens, 10);
                assert_eq!(u.total_tokens, 30);
            }
            _ => panic!("Expected usage chunk"),
        }
    }

    #[test]
    fn test_convert_stream_chunk_empty() {
        let chunk = OpenAIStreamChunk {
            choices: vec![],
            usage: None,
        };

        let converted = convert_stream_chunk(chunk);
        assert!(converted.is_empty());
    }

    #[test]
    fn test_chat_options_to_request_options() {
        let opts = ChatOptions {
            temperature: Some(0.5),
            max_tokens: Some(1024),
            top_p: Some(0.9),
            ..ChatOptions::default()
        };

        let req_opts = chat_options_to_request_options(&opts);
        assert_eq!(req_opts.temperature, Some(0.5));
        assert_eq!(req_opts.max_tokens, Some(1024));
        assert_eq!(req_opts.top_p, Some(0.9));
        assert!(req_opts.stop.is_none());
        assert!(req_opts.system.is_none());
    }

    #[test]
    fn test_chat_options_to_request_options_defaults() {
        let opts = ChatOptions::default();

        let req_opts = chat_options_to_request_options(&opts);
        // ChatOptions default may have a temperature set; just make sure
        // it maps without panicking.
        assert!(req_opts.stop.is_none());
        assert!(req_opts.system.is_none());
    }

    #[test]
    fn test_provider_name_default() {
        let client = Arc::new(OpenAiClient::new("key".to_string(), "gpt-4".to_string()));
        let provider = OpenAiChatProvider::new(client, "gpt-4".to_string());
        assert_eq!(provider.name(), "openai");
    }

    #[test]
    fn test_provider_name_custom() {
        let client = Arc::new(OpenAiClient::new("key".to_string(), "gpt-4".to_string()));
        let provider =
            OpenAiChatProvider::new(client, "gpt-4".to_string()).with_provider_name("groq");
        assert_eq!(provider.name(), "groq");
    }
}
