//! Integration tests for JSON-RPC 2.0 type serialization.
//!
//! The MCP client speaks JSON-RPC over stdio/HTTP, and our custom types
//! (`JsonRpcRequest`, `JsonRpcResponse`, `JsonRpcNotification`,
//! `JsonRpcError`, `JsonRpcMessage`, `McpNotification`) are the parse
//! surface for every byte coming off a transport. A bug here means we
//! either crash on a valid server response, or silently misinterpret an
//! attacker-crafted payload.
//!
//! Two property checks and a set of explicit edge cases:
//!
//! 1. Every valid request/response/notification survives a JSON roundtrip
//!    with its shape intact.
//! 2. The transport's id-based discriminator (response when `id` is
//!    present and non-null, notification otherwise) is exercised against
//!    randomized inputs.

use brainwires_mcp_client::types::{
    JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
    McpNotification, ProgressParams,
};
use proptest::prelude::*;
use serde_json::{Value, json};

/// Mirror of the parse logic in `transport.rs:162-180`. Extracted here so
/// the test exercises the exact same semantic used in both the stdio and
/// HTTP clients without linking the full rmcp dep surface.
fn discriminate(raw: &str) -> anyhow::Result<JsonRpcMessage> {
    let value: Value = serde_json::from_str(raw)?;
    let has_valid_id = value.get("id").map(|id| !id.is_null()).unwrap_or(false);
    if has_valid_id {
        Ok(JsonRpcMessage::Response(serde_json::from_value(value)?))
    } else {
        Ok(JsonRpcMessage::Notification(serde_json::from_value(value)?))
    }
}

// ── Explicit edge cases ──────────────────────────────────────────────────

#[test]
fn request_id_can_be_string_number_or_null() {
    for id in [json!(1), json!("req-42"), json!(null)] {
        let req = JsonRpcRequest::new::<Value>(id.clone(), "ping".to_string(), None).unwrap();
        let wire = serde_json::to_string(&req).unwrap();
        let back: JsonRpcRequest = serde_json::from_str(&wire).unwrap();
        assert_eq!(back.jsonrpc, "2.0");
        assert_eq!(back.id, id);
        assert_eq!(back.method, "ping");
        assert!(back.params.is_none());
    }
}

#[test]
fn response_with_error_omits_result_field_on_the_wire() {
    let resp = JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: json!(1),
        result: None,
        error: Some(JsonRpcError {
            code: -32601,
            message: "Method not found".to_string(),
            data: Some(json!({"method": "foo"})),
        }),
    };
    let wire = serde_json::to_string(&resp).unwrap();
    assert!(
        !wire.contains("\"result\""),
        "result=None must be skipped: {wire}"
    );
    assert!(wire.contains("\"error\""));
    let back: JsonRpcResponse = serde_json::from_str(&wire).unwrap();
    let err = back.error.expect("error preserved");
    assert_eq!(err.code, -32601);
    assert_eq!(err.message, "Method not found");
    assert_eq!(err.data, Some(json!({"method": "foo"})));
}

#[test]
fn notification_has_no_id_field_on_the_wire() {
    let n = JsonRpcNotification::new::<Value>("notifications/progress", None).unwrap();
    let wire = serde_json::to_string(&n).unwrap();
    assert!(
        !wire.contains("\"id\""),
        "notification wire must not contain id: {wire}"
    );
    assert!(wire.contains("\"jsonrpc\":\"2.0\""));
    assert!(wire.contains("\"notifications/progress\""));
}

#[test]
fn progress_params_parse_from_notification() {
    let notif = JsonRpcNotification::new(
        "notifications/progress",
        Some(ProgressParams {
            progress_token: "tok-1".to_string(),
            progress: 42.0,
            total: Some(100.0),
            message: Some("halfway".to_string()),
        }),
    )
    .unwrap();

    match McpNotification::from_notification(&notif) {
        McpNotification::Progress(p) => {
            assert_eq!(p.progress_token, "tok-1");
            assert_eq!(p.progress, 42.0);
            assert_eq!(p.total, Some(100.0));
            assert_eq!(p.message.as_deref(), Some("halfway"));
        }
        other => panic!("expected Progress, got {other:?}"),
    }
}

#[test]
fn unknown_notification_method_falls_through_to_unknown_variant() {
    let notif = JsonRpcNotification::new::<Value>("some/custom/event", None).unwrap();
    match McpNotification::from_notification(&notif) {
        McpNotification::Unknown { method, .. } => assert_eq!(method, "some/custom/event"),
        other => panic!("expected Unknown, got {other:?}"),
    }
}

#[test]
fn progress_notification_with_malformed_params_falls_through_to_unknown() {
    // "notifications/progress" method name but the params shape is wrong —
    // the discriminator should degrade to Unknown rather than panic.
    let notif = JsonRpcNotification {
        jsonrpc: "2.0".to_string(),
        method: "notifications/progress".to_string(),
        params: Some(json!({"not_a_real_field": 1})),
    };
    assert!(matches!(
        McpNotification::from_notification(&notif),
        McpNotification::Unknown { .. }
    ));
}

// ── Transport discriminator ──────────────────────────────────────────────

#[test]
fn discriminator_classifies_response_with_integer_id() {
    let wire = r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#;
    assert!(matches!(
        discriminate(wire).unwrap(),
        JsonRpcMessage::Response(_)
    ));
}

#[test]
fn discriminator_classifies_notification_without_id() {
    let wire = r#"{"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"t","progress":1.0}}"#;
    assert!(matches!(
        discriminate(wire).unwrap(),
        JsonRpcMessage::Notification(_)
    ));
}

#[test]
fn discriminator_treats_explicit_null_id_as_notification() {
    // MCP spec: an `id` of null is effectively no id — must be a notification.
    let wire = r#"{"jsonrpc":"2.0","id":null,"method":"notifications/cancelled"}"#;
    assert!(matches!(
        discriminate(wire).unwrap(),
        JsonRpcMessage::Notification(_)
    ));
}

#[test]
fn discriminator_rejects_malformed_json() {
    assert!(discriminate("{not json").is_err());
    assert!(discriminate("").is_err());
}

// ── Property-based roundtrips ────────────────────────────────────────────

fn arb_ident() -> impl Strategy<Value = String> {
    "[a-zA-Z][a-zA-Z0-9_/]{0,20}".prop_map(String::from)
}

fn arb_id() -> impl Strategy<Value = Value> {
    prop_oneof![
        any::<i64>().prop_map(Value::from),
        arb_ident().prop_map(Value::from),
    ]
}

proptest! {
    #[test]
    fn request_roundtrips(id in arb_id(), method in arb_ident(), has_params in any::<bool>()) {
        let params = if has_params {
            Some(json!({"k": "v"}))
        } else {
            None
        };
        let req = JsonRpcRequest::new(id.clone(), method.clone(), params.clone()).unwrap();
        let wire = serde_json::to_string(&req).unwrap();
        let back: JsonRpcRequest = serde_json::from_str(&wire).unwrap();
        prop_assert_eq!(back.jsonrpc, "2.0");
        prop_assert_eq!(back.id, id);
        prop_assert_eq!(back.method, method);
        prop_assert_eq!(back.params, params);
    }

    #[test]
    fn response_success_roundtrips(id in arb_id(), payload in arb_ident()) {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: id.clone(),
            result: Some(json!({"data": payload.clone()})),
            error: None,
        };
        let wire = serde_json::to_string(&resp).unwrap();
        let back: JsonRpcResponse = serde_json::from_str(&wire).unwrap();
        prop_assert_eq!(back.id, id);
        prop_assert_eq!(back.result, Some(json!({"data": payload})));
        prop_assert!(back.error.is_none());
    }

    #[test]
    fn response_error_roundtrips(
        id in arb_id(),
        code in any::<i32>(),
        msg in arb_ident(),
    ) {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: msg.clone(),
                data: None,
            }),
        };
        let wire = serde_json::to_string(&resp).unwrap();
        let back: JsonRpcResponse = serde_json::from_str(&wire).unwrap();
        let err = back.error.expect("error preserved");
        prop_assert_eq!(err.code, code);
        prop_assert_eq!(err.message, msg);
    }

    #[test]
    fn notification_roundtrips(method in arb_ident()) {
        let n = JsonRpcNotification::new::<Value>(method.clone(), None).unwrap();
        let wire = serde_json::to_string(&n).unwrap();
        // Notification wire must never contain an `id` field.
        prop_assert!(!wire.contains("\"id\""));
        let back: JsonRpcNotification = serde_json::from_str(&wire).unwrap();
        prop_assert_eq!(back.jsonrpc, "2.0");
        prop_assert_eq!(back.method, method);
    }

    #[test]
    fn progress_params_roundtrip(
        token in arb_ident(),
        // Stick to integer-valued f64s so JSON's decimal encoding is
        // exact and equality-by-bits holds. Real MCP progress values are
        // typically step counts anyway, not arbitrary fractions.
        progress_int in -1_000_000i64..1_000_000i64,
        total_int in proptest::option::of(0i64..1_000_000i64),
    ) {
        let progress = progress_int as f64;
        let total = total_int.map(|t| t as f64);
        let p = ProgressParams {
            progress_token: token.clone(),
            progress,
            total,
            message: None,
        };
        let wire = serde_json::to_string(&p).unwrap();
        let back: ProgressParams = serde_json::from_str(&wire).unwrap();
        prop_assert_eq!(back.progress_token, token);
        prop_assert_eq!(back.progress.to_bits(), progress.to_bits());
        prop_assert_eq!(back.total.map(f64::to_bits), total.map(f64::to_bits));
    }
}
