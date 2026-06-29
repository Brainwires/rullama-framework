//! A2A server — serves JSON-RPC, REST, and optionally gRPC.

/// gRPC service implementation.
pub mod grpc_service;
/// Core handler trait.
pub mod handler;
/// JSON-RPC method dispatch.
pub mod jsonrpc_router;
/// HTTP/REST route handling.
pub mod rest_router;
/// SSE response construction.
pub mod sse_response;

pub use handler::A2aHandler;

#[cfg(feature = "grpc-server")]
pub use grpc_service::GrpcBridge;

use std::net::SocketAddr;
use std::sync::Arc;

use crate::error::A2aError;
use crate::jsonrpc::{JsonRpcRequest, METHOD_MESSAGE_STREAM, METHOD_TASKS_RESUBSCRIBE, RequestId};
use crate::params::{SendMessageRequest, SubscribeToTaskRequest};

/// Maximum request body size (10 MB).
const MAX_REQUEST_BODY_SIZE: usize = 10 * 1024 * 1024;

/// Unified A2A server serving JSON-RPC + REST (HTTP) and optionally gRPC.
pub struct A2aServer<H: A2aHandler> {
    handler: Arc<H>,
    addr: SocketAddr,
    #[cfg(feature = "grpc-server")]
    grpc_addr: Option<SocketAddr>,
    shutdown: Option<tokio::sync::watch::Receiver<()>>,
}

impl<H: A2aHandler> A2aServer<H> {
    /// Create a new server bound to `addr`.
    pub fn new(handler: H, addr: SocketAddr) -> Self {
        Self {
            handler: Arc::new(handler),
            addr,
            #[cfg(feature = "grpc-server")]
            grpc_addr: None,
            shutdown: None,
        }
    }

    /// Enable gRPC on a separate port.
    #[cfg(feature = "grpc-server")]
    pub fn with_grpc(mut self, grpc_addr: SocketAddr) -> Self {
        self.grpc_addr = Some(grpc_addr);
        self
    }

    /// Set a shutdown signal. When the sender is dropped or a value is sent,
    /// the server will stop accepting new connections.
    pub fn with_shutdown(mut self, rx: tokio::sync::watch::Receiver<()>) -> Self {
        self.shutdown = Some(rx);
        self
    }

    /// Run the server (blocks until shutdown signal or forever if none set).
    pub async fn run(self) -> Result<(), A2aError> {
        use hyper::body::Incoming;
        use hyper::service::service_fn;
        use hyper_util::rt::TokioIo;

        let handler = self.handler.clone();
        let listener = tokio::net::TcpListener::bind(self.addr)
            .await
            .map_err(|e| A2aError::internal(format!("Failed to bind: {e}")))?;

        tracing::info!("A2A server listening on {}", self.addr);

        // Optionally spawn gRPC server with bind error propagation
        #[cfg(feature = "grpc-server")]
        if let Some(grpc_addr) = self.grpc_addr {
            let grpc_handler = self.handler.clone();
            let shutdown_rx = self.shutdown.clone();
            let (bind_tx, mut bind_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
            tokio::spawn(async move {
                let bridge = GrpcBridge::new(grpc_handler);
                let svc =
                    crate::proto::lf_a2a_v1::a2a_service_server::A2aServiceServer::new(bridge);
                tracing::info!("A2A gRPC server listening on {grpc_addr}");
                let builder = tonic::transport::Server::builder().add_service(svc);
                let result = if let Some(mut rx) = shutdown_rx {
                    builder
                        .serve_with_shutdown(grpc_addr, async move {
                            let _ = rx.changed().await;
                        })
                        .await
                } else {
                    builder.serve(grpc_addr).await
                };
                match result {
                    Ok(()) => {
                        let _ = bind_tx.send(Ok(()));
                    }
                    Err(e) => {
                        let msg = format!("gRPC server error: {e}");
                        tracing::error!("{msg}");
                        let _ = bind_tx.send(Err(msg));
                    }
                }
            });
            // Give gRPC a moment to fail on immediate bind errors
            tokio::task::yield_now().await;
            if let Ok(Err(msg)) = bind_rx.try_recv() {
                return Err(A2aError::internal(msg));
            }
        }

        let mut shutdown = self.shutdown;

        loop {
            let accept_result = if let Some(ref mut rx) = shutdown {
                tokio::select! {
                    result = listener.accept() => Some(result),
                    _ = rx.changed() => None,
                }
            } else {
                Some(listener.accept().await)
            };

            let (stream, _peer) = match accept_result {
                None => {
                    tracing::info!("A2A server shutting down");
                    return Ok(());
                }
                Some(Ok(conn)) => conn,
                Some(Err(e)) => {
                    tracing::warn!("Accept error (continuing): {e}");
                    continue;
                }
            };

            let handler = handler.clone();
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let svc = service_fn(move |req: hyper::Request<Incoming>| {
                    let handler = handler.clone();
                    async move { handle_http_request(handler, req).await }
                });
                if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                    hyper_util::rt::TokioExecutor::new(),
                )
                .serve_connection(io, svc)
                .await
                {
                    tracing::debug!("Connection error: {e}");
                }
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Response body type — supports both buffered and streaming responses
// ---------------------------------------------------------------------------

#[cfg(feature = "server")]
type BoxBody = http_body_util::Either<
    http_body_util::Full<bytes::Bytes>,
    http_body_util::StreamBody<
        std::pin::Pin<
            Box<
                dyn futures::Stream<Item = Result<http_body::Frame<bytes::Bytes>, std::io::Error>>
                    + Send,
            >,
        >,
    >,
>;

#[cfg(feature = "server")]
fn json_response(status: u16, body: String) -> hyper::Response<BoxBody> {
    let mut resp = hyper::Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        .header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization",
        )
        .body(http_body_util::Either::Left(http_body_util::Full::new(
            bytes::Bytes::from(body),
        )))
        .expect("response builder with valid status and headers cannot fail");
    let _ = &mut resp;
    resp
}

#[cfg(feature = "server")]
fn sse_response(
    stream: std::pin::Pin<
        Box<
            dyn futures::Stream<Item = Result<http_body::Frame<bytes::Bytes>, std::io::Error>>
                + Send,
        >,
    >,
) -> hyper::Response<BoxBody> {
    hyper::Response::builder()
        .status(200)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        .header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization",
        )
        .body(http_body_util::Either::Right(
            http_body_util::StreamBody::new(stream),
        ))
        .expect("response builder with valid status and headers cannot fail")
}

// ---------------------------------------------------------------------------
// HTTP request handler
// ---------------------------------------------------------------------------

#[cfg(feature = "server")]
async fn handle_http_request<H: A2aHandler>(
    handler: Arc<H>,
    req: hyper::Request<hyper::body::Incoming>,
) -> Result<hyper::Response<BoxBody>, hyper::Error> {
    use http_body_util::BodyExt;

    let method = req.method().clone();
    let path = req.uri().path().to_string();

    // CORS preflight
    if method == hyper::Method::OPTIONS {
        return Ok(json_response(204, String::new()));
    }

    // Agent card discovery
    if method == hyper::Method::GET && path == "/.well-known/agent-card.json" {
        let card = handler.agent_card();
        let body = serde_json::to_string(card).unwrap_or_default();
        return Ok(json_response(200, body));
    }

    // Collect body with size limit
    let limited = http_body_util::Limited::new(req.into_body(), MAX_REQUEST_BODY_SIZE);
    let body_bytes = match limited.collect().await {
        Ok(c) => c.to_bytes(),
        Err(_) => {
            let err = A2aError::invalid_request("Request body too large");
            let body = serde_json::to_string(&err).unwrap_or_default();
            return Ok(json_response(413, body));
        }
    };

    // JSON-RPC: POST to /
    if method == hyper::Method::POST && path == "/" {
        return handle_jsonrpc(&handler, &body_bytes).await;
    }

    // REST routes
    let method_str = method.as_str();
    match rest_router::dispatch_rest(&handler, method_str, &path, &body_bytes).await {
        Ok(rest_router::RestResult::Json(val)) => {
            let body = serde_json::to_string(&val).unwrap_or_default();
            Ok(json_response(200, body))
        }
        Ok(rest_router::RestResult::Stream(stream)) => {
            let sse_stream = sse_response::stream_to_sse_rest(stream);
            Ok(sse_response(sse_stream))
        }
        Err(e) => {
            let body = serde_json::to_string(&e).unwrap_or_default();
            Ok(json_response(404, body))
        }
    }
}

#[cfg(feature = "server")]
async fn handle_jsonrpc<H: A2aHandler>(
    handler: &Arc<H>,
    body: &bytes::Bytes,
) -> Result<hyper::Response<BoxBody>, hyper::Error> {
    let request: JsonRpcRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            let resp = crate::jsonrpc::JsonRpcResponse::error(
                RequestId::Number(0),
                A2aError::parse_error(e.to_string()),
            );
            let body = serde_json::to_string(&resp).unwrap_or_default();
            return Ok(json_response(200, body));
        }
    };

    // Check for streaming methods
    if request.method == METHOD_MESSAGE_STREAM {
        let id = request.id.clone();
        let params = request.params.clone().unwrap_or(serde_json::Value::Null);
        let req: SendMessageRequest = match serde_json::from_value(params) {
            Ok(r) => r,
            Err(e) => {
                let resp = crate::jsonrpc::JsonRpcResponse::error(id, A2aError::from(e));
                let body = serde_json::to_string(&resp).unwrap_or_default();
                return Ok(json_response(200, body));
            }
        };
        match handler.on_send_streaming_message(req).await {
            Ok(stream) => {
                let sse_stream = sse_response::stream_to_sse(id, stream);
                return Ok(sse_response(sse_stream));
            }
            Err(e) => {
                let resp = crate::jsonrpc::JsonRpcResponse::error(id, e);
                let body = serde_json::to_string(&resp).unwrap_or_default();
                return Ok(json_response(200, body));
            }
        }
    }

    if request.method == METHOD_TASKS_RESUBSCRIBE {
        let id = request.id.clone();
        let params = request.params.clone().unwrap_or(serde_json::Value::Null);
        let req: SubscribeToTaskRequest = match serde_json::from_value(params) {
            Ok(r) => r,
            Err(e) => {
                let resp = crate::jsonrpc::JsonRpcResponse::error(id, A2aError::from(e));
                let body = serde_json::to_string(&resp).unwrap_or_default();
                return Ok(json_response(200, body));
            }
        };
        match handler.on_subscribe_to_task(req).await {
            Ok(stream) => {
                let sse_stream = sse_response::stream_to_sse(id, stream);
                return Ok(sse_response(sse_stream));
            }
            Err(e) => {
                let resp = crate::jsonrpc::JsonRpcResponse::error(id, e);
                let body = serde_json::to_string(&resp).unwrap_or_default();
                return Ok(json_response(200, body));
            }
        }
    }

    // Non-streaming JSON-RPC
    let response = match jsonrpc_router::dispatch(handler, &request).await {
        Ok(Some(resp)) => resp,
        Ok(None) => {
            // Should not happen — streaming methods handled above

            crate::jsonrpc::JsonRpcResponse::error(
                request.id,
                A2aError::internal("Unexpected routing state"),
            )
        }
        Err(resp) => resp,
    };

    let body = serde_json::to_string(&response).unwrap_or_default();
    Ok(json_response(200, body))
}
