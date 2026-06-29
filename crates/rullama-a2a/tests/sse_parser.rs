//! Tests for the SSE parser — both batch and incremental byte-stream parsing.

use bytes::Bytes;
use futures::StreamExt;

use rullama_a2a::error::A2aError;
use rullama_a2a::jsonrpc::{JsonRpcResponse, RequestId};
use rullama_a2a::streaming::{StreamResponse, TaskStatusUpdateEvent};
use rullama_a2a::task::{TaskState, TaskStatus};

fn make_status_response(task_id: &str, state: TaskState) -> StreamResponse {
    StreamResponse {
        task: None,
        message: None,
        status_update: Some(TaskStatusUpdateEvent {
            task_id: task_id.into(),
            context_id: "ctx".into(),
            status: TaskStatus {
                state,
                message: None,
                timestamp: None,
            },
            trace_id: None,
            sequence: None,
            metadata: None,
        }),
        artifact_update: None,
    }
}

fn wrap_jsonrpc(id: i64, event: &StreamResponse) -> String {
    let resp =
        JsonRpcResponse::success(RequestId::Number(id), serde_json::to_value(event).unwrap());
    serde_json::to_string(&resp).unwrap()
}

// ---- Batch parser (parse_sse_stream) ----

#[tokio::test]
async fn test_parse_sse_stream_single_event() {
    let event = make_status_response("t-1", TaskState::Working);
    let body = format!("data: {}\n\n", wrap_jsonrpc(1, &event));

    let mut stream = std::pin::pin!(rullama_a2a::client::sse::parse_sse_stream(body));
    let item = stream.next().await.unwrap().unwrap();

    assert!(item.status_update.is_some());
    let su = item.status_update.unwrap();
    assert_eq!(su.task_id, "t-1");
    assert_eq!(su.status.state, TaskState::Working);

    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn test_parse_sse_stream_multiple_events() {
    let e1 = make_status_response("t-1", TaskState::Working);
    let e2 = make_status_response("t-1", TaskState::Completed);
    let body = format!(
        "data: {}\n\ndata: {}\n\n",
        wrap_jsonrpc(1, &e1),
        wrap_jsonrpc(1, &e2)
    );

    let stream = std::pin::pin!(rullama_a2a::client::sse::parse_sse_stream(body));
    let items: Vec<_> = stream.collect().await;
    assert_eq!(items.len(), 2);
    assert!(items[0].is_ok());
    assert!(items[1].is_ok());
}

#[tokio::test]
async fn test_parse_sse_stream_with_error_response() {
    let err = A2aError::task_not_found("t-42");
    let resp = JsonRpcResponse::error(RequestId::Number(1), err);
    let body = format!("data: {}\n\n", serde_json::to_string(&resp).unwrap());

    let mut stream = std::pin::pin!(rullama_a2a::client::sse::parse_sse_stream(body));
    let item = stream.next().await.unwrap();
    assert!(item.is_err());
    assert_eq!(
        item.unwrap_err().code,
        rullama_a2a::error::TASK_NOT_FOUND
    );
}

#[tokio::test]
async fn test_parse_sse_stream_ignores_non_data_lines() {
    let event = make_status_response("t-1", TaskState::Working);
    let body = format!(
        ": this is a comment\nevent: update\ndata: {}\n\n",
        wrap_jsonrpc(1, &event)
    );

    let stream = std::pin::pin!(rullama_a2a::client::sse::parse_sse_stream(body));
    let items: Vec<_> = stream.collect().await;
    assert_eq!(items.len(), 1);
}

#[tokio::test]
async fn test_parse_sse_stream_invalid_json() {
    let body = "data: {not valid json}\n\n".to_string();

    let mut stream = std::pin::pin!(rullama_a2a::client::sse::parse_sse_stream(body));
    let item = stream.next().await.unwrap();
    assert!(item.is_err());
    assert_eq!(
        item.unwrap_err().code,
        rullama_a2a::error::JSON_PARSE_ERROR
    );
}

#[tokio::test]
async fn test_parse_sse_stream_empty_body() {
    let body = String::new();

    let stream = std::pin::pin!(rullama_a2a::client::sse::parse_sse_stream(body));
    let items: Vec<_> = stream.collect().await;
    assert!(items.is_empty());
}

// ---- Incremental byte stream parser (JSON-RPC) ----

fn bytes_stream(
    chunks: Vec<&str>,
) -> impl futures::Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static {
    let owned: Vec<String> = chunks.into_iter().map(String::from).collect();
    futures::stream::iter(owned.into_iter().map(|s| Ok(Bytes::from(s))))
}

#[tokio::test]
async fn test_incremental_sse_single_chunk() {
    let event = make_status_response("t-1", TaskState::Completed);
    let data = format!("data: {}\n\n", wrap_jsonrpc(1, &event));

    let stream = rullama_a2a::client::sse::parse_sse_byte_stream(bytes_stream(vec![&data]));
    let items: Vec<_> = std::pin::pin!(stream).collect().await;
    assert_eq!(items.len(), 1);
    assert!(items[0].is_ok());
}

#[tokio::test]
async fn test_incremental_sse_split_across_chunks() {
    let event = make_status_response("t-1", TaskState::Working);
    let full = format!("data: {}\n\n", wrap_jsonrpc(1, &event));

    // Split the SSE frame in the middle
    let mid = full.len() / 2;
    let chunk1 = &full[..mid];
    let chunk2 = &full[mid..];

    let stream =
        rullama_a2a::client::sse::parse_sse_byte_stream(bytes_stream(vec![chunk1, chunk2]));
    let items: Vec<_> = std::pin::pin!(stream).collect().await;
    assert_eq!(items.len(), 1);
    assert!(items[0].is_ok());
}

#[tokio::test]
async fn test_incremental_sse_multiple_events_one_chunk() {
    let e1 = make_status_response("t-1", TaskState::Working);
    let e2 = make_status_response("t-1", TaskState::Completed);
    let data = format!(
        "data: {}\n\ndata: {}\n\n",
        wrap_jsonrpc(1, &e1),
        wrap_jsonrpc(1, &e2)
    );

    let stream = rullama_a2a::client::sse::parse_sse_byte_stream(bytes_stream(vec![&data]));
    let items: Vec<_> = std::pin::pin!(stream).collect().await;
    assert_eq!(items.len(), 2);
}

#[tokio::test]
async fn test_incremental_sse_multiline_data() {
    // SSE spec: multiple data: lines concatenated with \n
    let event = make_status_response("t-1", TaskState::Working);
    let json = wrap_jsonrpc(1, &event);

    // Split JSON across two data: lines (simulate a server that wraps lines)
    let mid = json.len() / 2;
    let part1 = &json[..mid];
    let part2 = &json[mid..];
    let frame = format!("data: {part1}\ndata: {part2}\n\n");

    let stream = rullama_a2a::client::sse::parse_sse_byte_stream(bytes_stream(vec![&frame]));
    let items: Vec<_> = std::pin::pin!(stream).collect().await;
    assert_eq!(items.len(), 1);
    // Multi-line data: gets concatenated with \n, which makes it invalid JSON
    // unless the JSON itself was split exactly at a valid boundary.
    // The key test is that the parser doesn't crash or hang.
}

#[tokio::test]
async fn test_incremental_sse_ignores_comments_and_event_fields() {
    let event = make_status_response("t-1", TaskState::Working);
    let frame = format!(
        ": keep-alive\nevent: status\nid: 42\nretry: 5000\ndata: {}\n\n",
        wrap_jsonrpc(1, &event)
    );

    let stream = rullama_a2a::client::sse::parse_sse_byte_stream(bytes_stream(vec![&frame]));
    let items: Vec<_> = std::pin::pin!(stream).collect().await;
    assert_eq!(items.len(), 1);
    assert!(items[0].is_ok());
}

#[tokio::test]
async fn test_incremental_sse_empty_stream() {
    let stream = rullama_a2a::client::sse::parse_sse_byte_stream(bytes_stream(vec![]));
    let items: Vec<_> = std::pin::pin!(stream).collect().await;
    assert!(items.is_empty());
}

// ---- Incremental byte stream parser (REST — no JSON-RPC envelope) ----

#[tokio::test]
async fn test_incremental_rest_sse_single_event() {
    let event = make_status_response("t-1", TaskState::Completed);
    let json = serde_json::to_string(&event).unwrap();
    let data = format!("data: {json}\n\n");

    let stream = rullama_a2a::client::sse::parse_sse_rest_byte_stream(bytes_stream(vec![&data]));
    let items: Vec<_> = std::pin::pin!(stream).collect().await;
    assert_eq!(items.len(), 1);
    let item = items[0].as_ref().unwrap();
    assert!(item.status_update.is_some());
    let su = item.status_update.as_ref().unwrap();
    assert_eq!(su.task_id, "t-1");
    assert_eq!(su.status.state, TaskState::Completed);
}

#[tokio::test]
async fn test_incremental_rest_sse_split_chunks() {
    let event = make_status_response("t-1", TaskState::Working);
    let json = serde_json::to_string(&event).unwrap();
    let full = format!("data: {json}\n\n");

    let mid = full.len() / 2;
    let stream = rullama_a2a::client::sse::parse_sse_rest_byte_stream(bytes_stream(vec![
        &full[..mid],
        &full[mid..],
    ]));
    let items: Vec<_> = std::pin::pin!(stream).collect().await;
    assert_eq!(items.len(), 1);
    assert!(items[0].is_ok());
}

#[tokio::test]
async fn test_incremental_rest_sse_invalid_json() {
    let data = "data: {broken}\n\n".to_string();

    let stream = rullama_a2a::client::sse::parse_sse_rest_byte_stream(bytes_stream(vec![&data]));
    let items: Vec<_> = std::pin::pin!(stream).collect().await;
    assert_eq!(items.len(), 1);
    assert!(items[0].is_err());
}

// ---- parse_sse_bytes ----

#[tokio::test]
async fn test_parse_sse_bytes() {
    let event = make_status_response("t-1", TaskState::Working);
    let body = format!("data: {}\n\n", wrap_jsonrpc(1, &event));
    let bytes = Bytes::from(body);

    let stream = std::pin::pin!(rullama_a2a::client::sse::parse_sse_bytes(bytes));
    let items: Vec<_> = stream.collect().await;
    assert_eq!(items.len(), 1);
    assert!(items[0].is_ok());
}
