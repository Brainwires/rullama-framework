//! Integration test for the stdio transport: spawns a trivial child process
//! that emits a canned JSON-RPC response per input line, and verifies that
//! `StdioTransport` ships the request bytes to the child's stdin and parses
//! the response off stdout.
//!
//! Linux/macOS only — relies on `bash`. CI on those platforms ships with it;
//! Windows builds skip the test rather than gate on PowerShell.

#![cfg(all(unix, feature = "native"))]

use brainwires_mcp_client::transport::StdioTransport;
use brainwires_mcp_client::types::{JsonRpcMessage, JsonRpcRequest};
use serde_json::{Value, json};

/// Bash one-liner that echoes a fixed JSON-RPC success envelope on every input
/// line, with the same `id` the request used. Reading via `IFS=` preserves the
/// raw line so the script can `jq` out the id; we keep it simpler by hard-coding
/// `id: 1` and only sending one request per test.
const ECHO_SERVER: &str = r#"
while IFS= read -r line; do
  echo '{"jsonrpc":"2.0","id":1,"result":{"echoed":true}}'
done
"#;

#[tokio::test]
async fn send_request_and_receive_response_round_trip() {
    let mut transport = StdioTransport::new("bash", &["-c".into(), ECHO_SERVER.into()])
        .await
        .expect("spawn echo server");

    let req = JsonRpcRequest::new::<Value>(json!(1), "ping".into(), None)
        .expect("serialize ping request");

    transport.send_request(&req).await.expect("send request");

    let resp = transport
        .receive_response()
        .await
        .expect("receive response");

    assert_eq!(resp.jsonrpc, "2.0");
    assert_eq!(resp.id, json!(1));
    assert_eq!(resp.result, Some(json!({"echoed": true})));
    assert!(resp.error.is_none());

    transport.close().await.expect("close transport");
}

#[tokio::test]
async fn receive_message_classifies_response() {
    let mut transport = StdioTransport::new("bash", &["-c".into(), ECHO_SERVER.into()])
        .await
        .expect("spawn echo server");

    let req = JsonRpcRequest::new::<Value>(json!(1), "ping".into(), None).unwrap();
    transport.send_request(&req).await.unwrap();

    let msg = transport.receive_message().await.expect("receive message");
    assert!(
        matches!(msg, JsonRpcMessage::Response(_)),
        "expected Response variant, got {msg:?}"
    );

    transport.close().await.unwrap();
}

#[tokio::test]
async fn receive_message_classifies_notification_when_id_absent() {
    // Server that emits a notification (no `id`) instead of a response.
    let server = r#"
while IFS= read -r _; do
  echo '{"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"t","progress":1.0}}'
done
"#;
    let mut transport = StdioTransport::new("bash", &["-c".into(), server.into()])
        .await
        .expect("spawn server");

    let req = JsonRpcRequest::new::<Value>(json!(7), "ignored".into(), None).unwrap();
    transport.send_request(&req).await.unwrap();

    let msg = transport.receive_message().await.expect("receive message");
    match msg {
        JsonRpcMessage::Notification(n) => assert_eq!(n.method, "notifications/progress"),
        other => panic!("expected Notification, got {other:?}"),
    }

    transport.close().await.unwrap();
}

#[tokio::test]
async fn server_eof_surfaces_as_error() {
    // /bin/true exits immediately, closing stdout.
    let mut transport = StdioTransport::new("/bin/true", &[])
        .await
        .expect("spawn /bin/true");

    let req = JsonRpcRequest::new::<Value>(json!(1), "ping".into(), None).unwrap();
    let _ = transport.send_request(&req).await; // may succeed before EOF reaches us

    let err = transport
        .receive_message()
        .await
        .expect_err("EOF must surface as error, not silently hang");
    let msg = err.to_string();
    assert!(
        msg.contains("EOF") || msg.contains("closed") || msg.contains("empty"),
        "expected an EOF/closed-style error, got: {msg}"
    );
}
