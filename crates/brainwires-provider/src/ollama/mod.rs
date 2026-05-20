use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use brainwires_core::Tool;
use brainwires_core::{ChatOptions, Provider};
use brainwires_core::{
    ChatResponse, ContentBlock, ImageSource, Message, MessageContent, Role, StreamChunk, Usage,
};

use crate::rate_limiter::RateLimiter;

pub mod chat;
pub use chat::*;

/// Ollama local model provider.
pub struct OllamaProvider {
    model: String,
    base_url: String,
    http_client: Client,
    rate_limiter: Option<std::sync::Arc<RateLimiter>>,
}

impl OllamaProvider {
    /// Create a new Ollama provider with the given model and optional base URL.
    pub fn new(model: String, base_url: Option<String>) -> Self {
        Self {
            model,
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
            http_client: Client::new(),
            rate_limiter: None,
        }
    }

    /// Create a provider with rate limiting (requests per minute).
    pub fn with_rate_limit(
        model: String,
        base_url: Option<String>,
        requests_per_minute: u32,
    ) -> Self {
        Self {
            model,
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
            http_client: Client::new(),
            rate_limiter: Some(std::sync::Arc::new(RateLimiter::new(requests_per_minute))),
        }
    }

    /// Wait for rate-limit clearance (no-op if not configured).
    async fn acquire_rate_limit(&self) {
        if let Some(ref limiter) = self.rate_limiter {
            limiter.acquire().await;
        }
    }

    /// Convert our Message format to Ollama's format
    fn convert_messages(&self, messages: &[Message]) -> Vec<OllamaMessage> {
        messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                    Role::Tool => "tool",
                };

                let (content, images) = match &m.content {
                    MessageContent::Text(text) => (text.clone(), None),
                    MessageContent::Blocks(blocks) => {
                        let mut text_parts = Vec::new();
                        let mut image_data = Vec::new();
                        for b in blocks {
                            match b {
                                ContentBlock::Text { text } => text_parts.push(text.as_str()),
                                ContentBlock::Image {
                                    source: ImageSource::Base64 { data, .. },
                                } => {
                                    image_data.push(data.clone());
                                }
                                _ => {}
                            }
                        }
                        let images = if image_data.is_empty() {
                            None
                        } else {
                            Some(image_data)
                        };
                        (text_parts.join("\n"), images)
                    }
                };

                OllamaMessage {
                    role: role.to_string(),
                    content,
                    images,
                }
            })
            .collect()
    }

    /// Convert our Tool format to Ollama's format
    fn convert_tools(&self, tools: &[Tool]) -> Vec<OllamaTool> {
        tools
            .iter()
            .map(|t| OllamaTool {
                r#type: "function".to_string(),
                function: OllamaFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: OllamaFunctionParameters {
                        r#type: "object".to_string(),
                        properties: t.input_schema.properties.clone().unwrap_or_default(),
                        required: t.input_schema.required.clone().unwrap_or_default(),
                    },
                },
            })
            .collect()
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    #[tracing::instrument(name = "provider.chat", skip_all, fields(provider = "ollama", model = %self.model))]
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let ollama_messages = self.convert_messages(messages);

        let effective_model = options.model.as_deref().unwrap_or(&self.model);
        let mut request_body = json!({
            "model": effective_model,
            "messages": ollama_messages,
            "stream": false,
        });

        // Options
        let mut opts = json!({});
        if let Some(temp) = options.temperature {
            opts["temperature"] = json!(temp);
        }
        if let Some(top_p) = options.top_p {
            opts["top_p"] = json!(top_p);
        }
        if !opts
            .as_object()
            .expect("opts is always a JSON object")
            .is_empty()
        {
            request_body["options"] = opts;
        }

        // Tools (Ollama has experimental tool support)
        if let Some(tools_list) = tools
            && !tools_list.is_empty()
        {
            request_body["tools"] = json!(self.convert_tools(tools_list));
        }

        let url = format!("{}/api/chat", self.base_url);

        self.acquire_rate_limit().await;
        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .context("Failed to send request to Ollama")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("Ollama API error ({}): {}", status, error_text);
        }

        let ollama_response: OllamaResponse = response
            .json()
            .await
            .context("Failed to parse Ollama response")?;

        // Convert response to our format
        let content = MessageContent::Text(ollama_response.message.content);

        Ok(ChatResponse {
            message: Message {
                role: Role::Assistant,
                content,
                name: None,
                metadata: None,
            },
            usage: Usage {
                prompt_tokens: ollama_response.prompt_eval_count.unwrap_or(0),
                completion_tokens: ollama_response.eval_count.unwrap_or(0),
                total_tokens: ollama_response.prompt_eval_count.unwrap_or(0)
                    + ollama_response.eval_count.unwrap_or(0),
                ..Default::default()
            },
            finish_reason: Some(
                ollama_response
                    .done_reason
                    .unwrap_or_else(|| "stop".to_string()),
            ),
        })
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        tracing::info!(provider = "ollama", model = %self.model, "provider.stream started");
        Box::pin(async_stream::stream! {
            let ollama_messages = self.convert_messages(messages);
            let effective_model = options.model.as_deref().unwrap_or(&self.model);

            let mut request_body = json!({
                "model": effective_model,
                "messages": ollama_messages,
                "stream": true,
            });

            // Options
            let mut opts = json!({});
            if let Some(temp) = options.temperature {
                opts["temperature"] = json!(temp);
            }
            if let Some(top_p) = options.top_p {
                opts["top_p"] = json!(top_p);
            }
            if !opts.as_object().expect("opts is always a JSON object").is_empty() {
                request_body["options"] = opts;
            }

            // Tools (Ollama has experimental tool support)
            if let Some(tools_list) = tools
                && !tools_list.is_empty() {
                    request_body["tools"] = json!(self.convert_tools(tools_list));
                }

            let url = format!("{}/api/chat", self.base_url);

            self.acquire_rate_limit().await;
            let response = match self
                .http_client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&request_body)
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
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                yield Err(anyhow::anyhow!("Ollama API error ({}): {}", status, error_text));
                return;
            }

            // Parse streaming response (newline-delimited JSON)
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

                // Process complete JSON objects (delimited by newlines)
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    // Parse the JSON line
                    match serde_json::from_str::<OllamaStreamChunk>(&line) {
                        Ok(chunk) => {
                            if let Some(message) = chunk.message
                                && !message.content.is_empty() {
                                    yield Ok(StreamChunk::Text(message.content));
                                }

                            if chunk.done {
                                // Emit usage if available
                                if let (Some(prompt_tokens), Some(completion_tokens)) =
                                    (chunk.prompt_eval_count, chunk.eval_count)
                                {
                                    yield Ok(StreamChunk::Usage(Usage {
                                        prompt_tokens,
                                        completion_tokens,
                                        total_tokens: prompt_tokens + completion_tokens,
                                        ..Default::default()
                                    }));
                                }
                                yield Ok(StreamChunk::Done);
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse Ollama stream chunk: {}", e);
                        }
                    }
                }
            }
        })
    }
}

// Ollama API types

#[derive(Debug, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct OllamaTool {
    r#type: String,
    function: OllamaFunction,
}

#[derive(Debug, Serialize)]
struct OllamaFunction {
    name: String,
    description: String,
    parameters: OllamaFunctionParameters,
}

#[derive(Debug, Serialize)]
struct OllamaFunctionParameters {
    r#type: String,
    properties: std::collections::HashMap<String, serde_json::Value>,
    required: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: OllamaResponseMessage,
    done_reason: Option<String>,
    #[serde(rename = "prompt_eval_count")]
    prompt_eval_count: Option<u32>,
    #[serde(rename = "eval_count")]
    eval_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponseMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    message: Option<OllamaResponseMessage>,
    done: bool,
    #[serde(rename = "prompt_eval_count")]
    prompt_eval_count: Option<u32>,
    #[serde(rename = "eval_count")]
    eval_count: Option<u32>,
}

// ---------------------------------------------------------------------------
// Model listing
// ---------------------------------------------------------------------------

use crate::model_listing::{AvailableModel, ModelCapability, ModelLister, OllamaTagsResponse};

/// Lists locally downloaded models from an Ollama instance.
pub struct OllamaModelLister {
    base_url: String,
    http_client: Client,
}

impl OllamaModelLister {
    /// Create a new model lister with an optional Ollama base URL.
    pub fn new(base_url: Option<String>) -> Self {
        Self {
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
            http_client: Client::new(),
        }
    }
}

#[async_trait]
impl ModelLister for OllamaModelLister {
    async fn list_models(&self) -> Result<Vec<AvailableModel>> {
        let url = format!("{}/api/tags", self.base_url);

        let resp = self
            .http_client
            .get(&url)
            .send()
            .await
            .context("Failed to connect to Ollama. Is it running?")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Ollama API returned {}: {}", status, body));
        }

        let tags: OllamaTagsResponse = resp
            .json()
            .await
            .context("Failed to parse Ollama tags response")?;

        let models = tags
            .models
            .into_iter()
            .map(|entry| AvailableModel {
                id: entry.name.clone(),
                display_name: Some(entry.name),
                provider: crate::ProviderType::Ollama,
                capabilities: vec![ModelCapability::Chat],
                owned_by: Some("local".to_string()),
                context_window: None,
                max_output_tokens: None,
                created_at: None,
            })
            .collect();

        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::ToolInputSchema;
    use std::collections::HashMap;

    #[test]
    fn test_ollama_provider_new_with_default_url() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        assert_eq!(provider.model, "llama2");
        assert_eq!(provider.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_ollama_provider_new_with_custom_url() {
        let provider =
            OllamaProvider::new("llama2".to_string(), Some("http://custom:8080".to_string()));
        assert_eq!(provider.model, "llama2");
        assert_eq!(provider.base_url, "http://custom:8080");
    }

    #[test]
    fn test_provider_name() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        assert_eq!(provider.name(), "ollama");
    }

    #[test]
    fn test_convert_messages_text() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text("Hello".to_string()),
            name: None,
            metadata: None,
        }];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "user");
        assert_eq!(converted[0].content, "Hello");
    }

    #[test]
    fn test_convert_messages_system_role() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages = vec![Message {
            role: Role::System,
            content: MessageContent::Text("You are helpful".to_string()),
            name: None,
            metadata: None,
        }];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "system");
    }

    #[test]
    fn test_convert_messages_with_blocks() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages = vec![Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "First".to_string(),
                },
                ContentBlock::Text {
                    text: "Second".to_string(),
                },
            ]),
            name: None,
            metadata: None,
        }];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "assistant");
        // Text blocks should be concatenated with newline
        assert!(converted[0].content.contains("First"));
        assert!(converted[0].content.contains("Second"));
    }

    #[test]
    fn test_convert_messages_multiple() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
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
        ];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].role, "user");
        assert_eq!(converted[1].role, "assistant");
    }

    #[test]
    fn test_convert_tools() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
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

        let converted = provider.convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].r#type, "function");
        assert_eq!(converted[0].function.name, "test_tool");
        assert_eq!(converted[0].function.description, "A test tool");
    }

    #[test]
    fn test_convert_tools_empty() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let tools: Vec<Tool> = vec![];

        let converted = provider.convert_tools(&tools);
        assert_eq!(converted.len(), 0);
    }

    #[test]
    fn test_convert_messages_filters_non_text_blocks() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Text content".to_string(),
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

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        // Only text blocks should be included
        assert_eq!(converted[0].content, "Text content");
    }

    #[test]
    fn test_convert_messages_with_image_blocks() {
        let provider = OllamaProvider::new("llava".to_string(), None);
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "What's in this image?".to_string(),
                },
                ContentBlock::Image {
                    source: brainwires_core::ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: "iVBORw0KGgo=".to_string(),
                    },
                },
            ]),
            name: None,
            metadata: None,
        }];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].content, "What's in this image?");
        assert!(converted[0].images.is_some());
        let images = converted[0].images.as_ref().unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0], "iVBORw0KGgo=");
    }

    #[test]
    fn test_convert_messages_with_multiple_images() {
        let provider = OllamaProvider::new("llava".to_string(), None);
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Compare these images".to_string(),
                },
                ContentBlock::Image {
                    source: brainwires_core::ImageSource::Base64 {
                        media_type: "image/png".to_string(),
                        data: "image1data".to_string(),
                    },
                },
                ContentBlock::Image {
                    source: brainwires_core::ImageSource::Base64 {
                        media_type: "image/jpeg".to_string(),
                        data: "image2data".to_string(),
                    },
                },
            ]),
            name: None,
            metadata: None,
        }];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        let images = converted[0].images.as_ref().unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(images[0], "image1data");
        assert_eq!(images[1], "image2data");
    }

    #[test]
    fn test_convert_messages_text_only_no_images() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::Text {
                text: "Just text".to_string(),
            }]),
            name: None,
            metadata: None,
        }];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert!(converted[0].images.is_none());
    }

    #[test]
    fn test_convert_messages_tool_role() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages = vec![Message {
            role: Role::Tool,
            content: MessageContent::Text("Tool result".to_string()),
            name: None,
            metadata: None,
        }];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].role, "tool");
        assert_eq!(converted[0].content, "Tool result");
    }

    #[test]
    fn test_convert_messages_empty_list() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages: Vec<Message> = vec![];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 0);
    }

    #[test]
    fn test_convert_messages_empty_content() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text("".to_string()),
            name: None,
            metadata: None,
        }];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].content, "");
    }

    #[test]
    fn test_convert_messages_blocks_with_only_non_text() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![
                ContentBlock::ToolUse {
                    id: "tool-1".to_string(),
                    name: "test_tool".to_string(),
                    input: json!({"arg": "value"}),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tool-1".to_string(),
                    content: "result".to_string(),
                    is_error: Some(false),
                },
            ]),
            name: None,
            metadata: None,
        }];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        // Should result in empty content since no text blocks
        assert_eq!(converted[0].content, "");
    }

    #[test]
    fn test_convert_messages_blocks_multiple_text() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Line 1".to_string(),
                },
                ContentBlock::Text {
                    text: "Line 2".to_string(),
                },
                ContentBlock::Text {
                    text: "Line 3".to_string(),
                },
            ]),
            name: None,
            metadata: None,
        }];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].content, "Line 1\nLine 2\nLine 3");
    }

    #[test]
    fn test_convert_messages_all_roles() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages = vec![
            Message {
                role: Role::System,
                content: MessageContent::Text("system".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::Text("user".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("assistant".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::Tool,
                content: MessageContent::Text("tool".to_string()),
                name: None,
                metadata: None,
            },
        ];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 4);
        assert_eq!(converted[0].role, "system");
        assert_eq!(converted[1].role, "user");
        assert_eq!(converted[2].role, "assistant");
        assert_eq!(converted[3].role, "tool");
    }

    #[test]
    fn test_convert_tools_multiple() {
        let provider = OllamaProvider::new("llama2".to_string(), None);

        let mut properties1 = HashMap::new();
        properties1.insert(
            "arg1".to_string(),
            json!({"type": "string", "description": "First argument"}),
        );

        let mut properties2 = HashMap::new();
        properties2.insert(
            "arg2".to_string(),
            json!({"type": "number", "description": "Second argument"}),
        );

        let tools = vec![
            Tool {
                name: "tool1".to_string(),
                description: "First tool".to_string(),
                input_schema: ToolInputSchema::object(properties1, vec!["arg1".to_string()]),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "tool2".to_string(),
                description: "Second tool".to_string(),
                input_schema: ToolInputSchema::object(properties2, vec!["arg2".to_string()]),
                requires_approval: false,
                ..Default::default()
            },
        ];

        let converted = provider.convert_tools(&tools);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].function.name, "tool1");
        assert_eq!(converted[1].function.name, "tool2");
    }

    #[test]
    fn test_convert_tools_with_complex_schema() {
        let provider = OllamaProvider::new("llama2".to_string(), None);

        let mut properties = HashMap::new();
        properties.insert(
            "name".to_string(),
            json!({"type": "string", "description": "Name field"}),
        );
        properties.insert(
            "age".to_string(),
            json!({"type": "integer", "description": "Age field"}),
        );
        properties.insert(
            "active".to_string(),
            json!({"type": "boolean", "description": "Active status"}),
        );

        let tools = vec![Tool {
            name: "complex_tool".to_string(),
            description: "A complex tool with multiple parameters".to_string(),
            input_schema: ToolInputSchema::object(
                properties.clone(),
                vec!["name".to_string(), "age".to_string()],
            ),
            requires_approval: true,
            ..Default::default()
        }];

        let converted = provider.convert_tools(&tools);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].function.name, "complex_tool");
        assert_eq!(
            converted[0].function.description,
            "A complex tool with multiple parameters"
        );
        assert_eq!(converted[0].function.parameters.r#type, "object");
        assert_eq!(converted[0].function.parameters.properties.len(), 3);
    }

    #[test]
    fn test_provider_new_with_different_models() {
        let provider1 = OllamaProvider::new("llama2".to_string(), None);
        assert_eq!(provider1.model, "llama2");

        let provider2 = OllamaProvider::new("mistral".to_string(), None);
        assert_eq!(provider2.model, "mistral");

        let provider3 = OllamaProvider::new("codellama".to_string(), None);
        assert_eq!(provider3.model, "codellama");
    }

    #[test]
    fn test_provider_new_with_custom_url_variations() {
        let provider1 = OllamaProvider::new(
            "llama2".to_string(),
            Some("http://192.168.1.100:11434".to_string()),
        );
        assert_eq!(provider1.base_url, "http://192.168.1.100:11434");

        let provider2 = OllamaProvider::new(
            "llama2".to_string(),
            Some("https://ollama.example.com".to_string()),
        );
        assert_eq!(provider2.base_url, "https://ollama.example.com");

        let provider3 = OllamaProvider::new(
            "llama2".to_string(),
            Some("http://localhost:8080".to_string()),
        );
        assert_eq!(provider3.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_ollama_message_serialization() {
        let message = OllamaMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
            images: None,
        };

        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("user"));
        assert!(json.contains("Hello"));
        assert!(!json.contains("images")); // Should be skipped when None
    }

    #[test]
    fn test_ollama_message_with_images() {
        let message = OllamaMessage {
            role: "user".to_string(),
            content: "What's in this image?".to_string(),
            images: Some(vec!["base64_encoded_image".to_string()]),
        };

        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("images"));
        assert!(json.contains("base64_encoded_image"));
    }

    #[test]
    fn test_ollama_response_deserialization() {
        let json = r#"{
            "message": {
                "content": "Hello, how can I help you?"
            },
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 20
        }"#;

        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.message.content, "Hello, how can I help you?");
        assert_eq!(response.done_reason, Some("stop".to_string()));
        assert_eq!(response.prompt_eval_count, Some(10));
        assert_eq!(response.eval_count, Some(20));
    }

    #[test]
    fn test_ollama_response_deserialization_missing_optional_fields() {
        let json = r#"{
            "message": {
                "content": "Response"
            }
        }"#;

        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.message.content, "Response");
        assert_eq!(response.done_reason, None);
        assert_eq!(response.prompt_eval_count, None);
        assert_eq!(response.eval_count, None);
    }

    #[test]
    fn test_ollama_stream_chunk_deserialization() {
        let json = r#"{
            "message": {
                "content": "chunk"
            },
            "done": false
        }"#;

        let chunk: OllamaStreamChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.message.is_some());
        assert_eq!(chunk.message.unwrap().content, "chunk");
        assert!(!chunk.done);
    }

    #[test]
    fn test_ollama_stream_chunk_done() {
        let json = r#"{
            "done": true,
            "prompt_eval_count": 15,
            "eval_count": 25
        }"#;

        let chunk: OllamaStreamChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.message.is_none());
        assert!(chunk.done);
        assert_eq!(chunk.prompt_eval_count, Some(15));
        assert_eq!(chunk.eval_count, Some(25));
    }

    #[test]
    fn test_ollama_tool_serialization() {
        let mut properties = HashMap::new();
        properties.insert(
            "query".to_string(),
            json!({"type": "string", "description": "Search query"}),
        );

        let tool = OllamaTool {
            r#type: "function".to_string(),
            function: OllamaFunction {
                name: "search".to_string(),
                description: "Search for information".to_string(),
                parameters: OllamaFunctionParameters {
                    r#type: "object".to_string(),
                    properties,
                    required: vec!["query".to_string()],
                },
            },
        };

        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "search");
        assert_eq!(json["function"]["parameters"]["type"], "object");
        assert!(json["function"]["parameters"]["required"].is_array());
    }

    #[test]
    fn test_convert_messages_preserves_order() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages = vec![
            Message {
                role: Role::System,
                content: MessageContent::Text("First".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::User,
                content: MessageContent::Text("Second".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("Third".to_string()),
                name: None,
                metadata: None,
            },
        ];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].content, "First");
        assert_eq!(converted[1].content, "Second");
        assert_eq!(converted[2].content, "Third");
    }

    #[test]
    fn test_convert_tools_preserves_order() {
        let provider = OllamaProvider::new("llama2".to_string(), None);

        let tools = vec![
            Tool {
                name: "first".to_string(),
                description: "First tool".to_string(),
                input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "second".to_string(),
                description: "Second tool".to_string(),
                input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
                requires_approval: false,
                ..Default::default()
            },
            Tool {
                name: "third".to_string(),
                description: "Third tool".to_string(),
                input_schema: ToolInputSchema::object(HashMap::new(), vec![]),
                requires_approval: false,
                ..Default::default()
            },
        ];

        let converted = provider.convert_tools(&tools);
        assert_eq!(converted.len(), 3);
        assert_eq!(converted[0].function.name, "first");
        assert_eq!(converted[1].function.name, "second");
        assert_eq!(converted[2].function.name, "third");
    }

    #[test]
    fn test_convert_messages_with_mixed_content_types() {
        let provider = OllamaProvider::new("llama2".to_string(), None);
        let messages = vec![
            Message {
                role: Role::User,
                content: MessageContent::Text("Plain text".to_string()),
                name: None,
                metadata: None,
            },
            Message {
                role: Role::Assistant,
                content: MessageContent::Blocks(vec![
                    ContentBlock::Text {
                        text: "Block 1".to_string(),
                    },
                    ContentBlock::Text {
                        text: "Block 2".to_string(),
                    },
                ]),
                name: None,
                metadata: None,
            },
        ];

        let converted = provider.convert_messages(&messages);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].content, "Plain text");
        assert_eq!(converted[1].content, "Block 1\nBlock 2");
    }

    #[test]
    fn test_ollama_function_parameters_structure() {
        let mut properties = HashMap::new();
        properties.insert("key1".to_string(), json!({"type": "string"}));
        properties.insert("key2".to_string(), json!({"type": "number"}));

        let params = OllamaFunctionParameters {
            r#type: "object".to_string(),
            properties: properties.clone(),
            required: vec!["key1".to_string()],
        };

        assert_eq!(params.r#type, "object");
        assert_eq!(params.properties.len(), 2);
        assert_eq!(params.required.len(), 1);
        assert_eq!(params.required[0], "key1");
    }
}
