//! SSE (Server-Sent Events) response utilities for streaming.

use std::pin::Pin;

use bytes::Bytes;
use futures::Stream;
use tokio_stream::StreamExt;

use crate::error::A2aError;
use crate::jsonrpc::{JsonRpcResponse, RequestId};
use crate::streaming::StreamResponse;

/// Convert a stream of `StreamResponse` items into an SSE byte stream (JSON-RPC envelope).
///
/// Each event is serialized as a JSON-RPC response wrapped in an SSE `data:` line.
pub fn stream_to_sse(
    id: RequestId,
    stream: Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>>,
) -> Pin<Box<dyn Stream<Item = Result<http_body::Frame<Bytes>, std::io::Error>> + Send>> {
    let mapped = stream.map(move |item| {
        let response = match item {
            Ok(event) => match serde_json::to_value(&event) {
                Ok(val) => JsonRpcResponse::success(id.clone(), val),
                Err(e) => JsonRpcResponse::error(
                    id.clone(),
                    A2aError::internal(format!("Failed to serialize event: {e}")),
                ),
            },
            Err(e) => JsonRpcResponse::error(id.clone(), e),
        };

        // JsonRpcResponse is a simple struct with Serialize — serialization is infallible
        // in practice, but we handle it gracefully just in case.
        let json = serde_json::to_string(&response).unwrap_or_else(|e| {
            let fallback = JsonRpcResponse::error(
                id.clone(),
                A2aError::internal(format!("SSE serialization error: {e}")),
            );
            serde_json::to_string(&fallback).unwrap_or_default()
        });
        let sse_line = format!("data: {json}\n\n");
        Ok(http_body::Frame::data(Bytes::from(sse_line)))
    });

    Box::pin(mapped)
}

/// Convert a stream of `StreamResponse` items into an SSE byte stream (REST — no JSON-RPC envelope).
///
/// Each event is serialized directly as JSON in an SSE `data:` line.
pub fn stream_to_sse_rest(
    stream: Pin<Box<dyn Stream<Item = Result<StreamResponse, A2aError>> + Send>>,
) -> Pin<Box<dyn Stream<Item = Result<http_body::Frame<Bytes>, std::io::Error>> + Send>> {
    let mapped = stream.map(|item| {
        let json = match item {
            Ok(event) => serde_json::to_string(&event).unwrap_or_else(|e| {
                let err = A2aError::internal(format!("Failed to serialize event: {e}"));
                serde_json::to_string(&err).unwrap_or_default()
            }),
            Err(e) => serde_json::to_string(&e).unwrap_or_else(|e2| {
                format!("{{\"code\":-32603,\"message\":\"Serialization error: {e2}\"}}")
            }),
        };
        let sse_line = format!("data: {json}\n\n");
        Ok(http_body::Frame::data(Bytes::from(sse_line)))
    });

    Box::pin(mapped)
}
