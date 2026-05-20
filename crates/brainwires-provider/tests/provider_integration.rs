/// Integration tests for the brainwires-provider crate.
///
/// These tests validate the Provider trait contract using a MockProvider,
/// and exercise per-provider request/response serialization logic without
/// hitting real network endpoints.
use std::sync::Arc;

use async_trait::async_trait;
use brainwires_core::{
    message::{ChatResponse, Message, StreamChunk, Usage},
    provider::ChatOptions,
    tool::Tool,
};
use brainwires_provider::Provider;
use futures::stream::{self, BoxStream};

// ── MockProvider ──────────────────────────────────────────────────────────────

/// A mock Provider that returns a fixed response without any network I/O.
struct MockProvider {
    name: String,
    response_text: String,
}

impl MockProvider {
    fn new(name: &str, response_text: &str) -> Self {
        Self {
            name: name.to_string(),
            response_text: response_text.to_string(),
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&[Tool]>,
        _options: &ChatOptions,
    ) -> anyhow::Result<ChatResponse> {
        Ok(ChatResponse {
            message: Message::assistant(&self.response_text),
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            },
            finish_reason: None,
        })
    }

    fn stream_chat<'a>(
        &'a self,
        _messages: &'a [Message],
        _tools: Option<&'a [Tool]>,
        _options: &'a ChatOptions,
    ) -> BoxStream<'a, anyhow::Result<StreamChunk>> {
        Box::pin(stream::empty())
    }
}

// ── Trait contract tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn mock_provider_name_returns_expected() {
    let provider = MockProvider::new("test-provider", "hello");
    assert_eq!(provider.name(), "test-provider");
}

#[tokio::test]
async fn mock_provider_chat_returns_fixed_response() {
    let provider = MockProvider::new("mock", "The answer is 42");
    let messages = vec![Message::user("What is the answer?")];
    let options = ChatOptions::default();

    let response = provider.chat(&messages, None, &options).await.unwrap();
    assert_eq!(
        response.message.text().unwrap_or_default(),
        "The answer is 42"
    );
}

#[tokio::test]
async fn mock_provider_chat_usage_is_populated() {
    let provider = MockProvider::new("mock", "response");
    let messages = vec![Message::user("test")];
    let options = ChatOptions::default();

    let response = provider.chat(&messages, None, &options).await.unwrap();
    assert_eq!(response.usage.total_tokens, 15);
    assert_eq!(response.usage.prompt_tokens, 10);
    assert_eq!(response.usage.completion_tokens, 5);
}

#[tokio::test]
async fn mock_provider_works_behind_arc() {
    let provider: Arc<dyn Provider> = Arc::new(MockProvider::new("arc-mock", "works"));
    let messages = vec![Message::user("test")];
    let options = ChatOptions::default();

    let response = provider.chat(&messages, None, &options).await.unwrap();
    assert!(!response.message.text().unwrap_or_default().is_empty());
}

#[tokio::test]
async fn mock_provider_stream_chat_is_empty() {
    use futures::StreamExt;
    let provider = MockProvider::new("stream-mock", "irrelevant");
    let messages = vec![Message::user("test")];
    let options = ChatOptions::default();

    let mut stream = provider.stream_chat(&messages, None, &options);
    let next = stream.next().await;
    assert!(next.is_none(), "MockProvider stream should be empty");
}

#[tokio::test]
async fn mock_provider_max_output_tokens_defaults_to_none() {
    let provider = MockProvider::new("mock", "x");
    assert_eq!(provider.max_output_tokens(), None);
}

// ── ChatOptions tests ─────────────────────────────────────────────────────────

#[test]
fn chat_options_default_has_temperature_and_max_tokens() {
    let opts = ChatOptions::default();
    assert_eq!(opts.temperature, Some(0.7));
    assert_eq!(opts.max_tokens, Some(4096));
    assert!(opts.system.is_none());
}

#[test]
fn chat_options_builder_sets_temperature() {
    let opts = ChatOptions {
        temperature: Some(0.0),
        ..ChatOptions::default()
    };
    assert_eq!(opts.temperature, Some(0.0));
}

// ── Bedrock URL helpers ───────────────────────────────────────────────────────

#[cfg(feature = "bedrock")]
mod bedrock_tests {
    use brainwires_provider::anthropic::bedrock::{bedrock_invoke_url, bedrock_stream_url};

    #[test]
    fn invoke_url_format() {
        let url = bedrock_invoke_url("us-east-1", "anthropic.claude-3-sonnet");
        assert!(url.starts_with("https://bedrock-runtime.us-east-1.amazonaws.com"));
        assert!(url.ends_with("/invoke"));
    }

    #[test]
    fn stream_url_format() {
        let url = bedrock_stream_url("us-west-2", "anthropic.claude-instant");
        assert!(url.ends_with("/invoke-with-response-stream"));
    }
}

// ── Vertex AI URL helpers ─────────────────────────────────────────────────────

#[cfg(feature = "vertex-ai")]
mod vertex_tests {
    use brainwires_provider::anthropic::vertex::{vertex_raw_predict_url, vertex_stream_url};

    #[test]
    fn stream_url_format() {
        let url = vertex_stream_url("us-central1", "my-project", "claude-3-sonnet");
        assert!(url.contains("aiplatform.googleapis.com"));
        assert!(url.ends_with("streamRawPredict"));
    }

    #[test]
    fn raw_predict_url_format() {
        let url = vertex_raw_predict_url("europe-west4", "proj", "model");
        assert!(url.ends_with("rawPredict"));
    }
}
