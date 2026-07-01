/// Integration tests for the rullama-provider crate.
///
/// These tests validate the Provider trait contract using a ScriptedProvider,
/// and exercise per-provider request/response serialization logic without
/// hitting real network endpoints.
use std::sync::Arc;

use rullama_core::{
    message::{ChatResponse, Message, Usage},
    provider::ChatOptions,
};
use rullama_provider::Provider;
use rullama_test_fixtures::ScriptedProvider;

/// Build a provider with the integration-test usage shape (10/5/15 tokens),
/// so `mock_provider_chat_usage_is_populated` continues to assert
/// canned-Usage round-tripping through the framework.
fn make_provider(name: &str, response_text: &str) -> ScriptedProvider {
    ScriptedProvider::always_response(
        name,
        ChatResponse {
            message: Message::assistant(response_text),
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            },
            finish_reason: None,
        },
    )
}

// ── Trait contract tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn mock_provider_name_returns_expected() {
    let provider = make_provider("test-provider", "hello");
    assert_eq!(provider.name(), "test-provider");
}

#[tokio::test]
async fn mock_provider_chat_returns_fixed_response() {
    let provider = make_provider("mock", "The answer is 42");
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
    let provider = make_provider("mock", "response");
    let messages = vec![Message::user("test")];
    let options = ChatOptions::default();

    let response = provider.chat(&messages, None, &options).await.unwrap();
    assert_eq!(response.usage.total_tokens, 15);
    assert_eq!(response.usage.prompt_tokens, 10);
    assert_eq!(response.usage.completion_tokens, 5);
}

#[tokio::test]
async fn mock_provider_works_behind_arc() {
    let provider: Arc<dyn Provider> = Arc::new(make_provider("arc-mock", "works"));
    let messages = vec![Message::user("test")];
    let options = ChatOptions::default();

    let response = provider.chat(&messages, None, &options).await.unwrap();
    assert!(!response.message.text().unwrap_or_default().is_empty());
}

#[tokio::test]
async fn mock_provider_stream_chat_emits_text_then_done() {
    use futures::StreamExt;
    use rullama_core::StreamChunk;

    let provider = make_provider("stream-mock", "streamed");
    let messages = vec![Message::user("test")];
    let options = ChatOptions::default();

    let chunks: Vec<_> = provider
        .stream_chat(&messages, None, &options)
        .collect()
        .await;
    // ScriptedProvider with non-zero canned Usage emits Text + Usage + Done
    // so consumers tracking cumulative_usage through stream_chat see the
    // same totals they'd see through chat().
    assert_eq!(chunks.len(), 3);
    match &chunks[0] {
        Ok(StreamChunk::Text(t)) => assert_eq!(t, "streamed"),
        other => panic!("expected Text chunk, got {other:?}"),
    }
    assert!(matches!(chunks[1], Ok(StreamChunk::Usage(_))));
    assert!(matches!(chunks[2], Ok(StreamChunk::Done)));
}

#[tokio::test]
async fn mock_provider_max_output_tokens_defaults_to_none() {
    let provider = make_provider("mock", "x");
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
    use rullama_provider::anthropic::bedrock::{bedrock_invoke_url, bedrock_stream_url};

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
    use rullama_provider::anthropic::vertex::{vertex_raw_predict_url, vertex_stream_url};

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
