//! HTTP transport — listener (hyper server) and connector (hyper client).

use crate::error::{ProxyError, ProxyResult};
use crate::request_id::RequestId;
use crate::transport::{InboundConnection, TransportConnector, TransportListener};
use crate::types::{ProxyBody, ProxyRequest, ProxyResponse, TransportKind};

use bytes::Bytes;
use http::{StatusCode, Uri};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use hyper_util::server::conn::auto::Builder as ServerBuilder;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot, watch};

/// HTTP listener using hyper.
pub struct HttpListener {
    addr: SocketAddr,
}

impl HttpListener {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }
}

#[async_trait::async_trait]
impl TransportListener for HttpListener {
    async fn listen(
        &self,
        tx: mpsc::Sender<InboundConnection>,
        mut shutdown: watch::Receiver<bool>,
    ) -> ProxyResult<()> {
        let listener = TcpListener::bind(self.addr).await?;
        tracing::info!(addr = %self.addr, "HTTP listener started");

        loop {
            tokio::select! {
                accept = listener.accept() => {
                    let (stream, peer) = accept?;
                    let tx = tx.clone();
                    let io = TokioIo::new(stream);

                    tokio::spawn(async move {
                        let service = service_fn(move |req: hyper::Request<Incoming>| {
                            let tx = tx.clone();
                            async move {
                                handle_http_request(req, peer, tx).await
                            }
                        });

                        if let Err(e) = ServerBuilder::new(hyper_util::rt::TokioExecutor::new())
                            .serve_connection(io, service)
                            .await
                        {
                            tracing::debug!(peer = %peer, error = %e, "connection error");
                        }
                    });
                }
                _ = shutdown.changed() => {
                    tracing::info!("HTTP listener shutting down");
                    break;
                }
            }
        }

        Ok(())
    }

    fn transport_name(&self) -> &str {
        "http"
    }
}

async fn handle_http_request(
    req: hyper::Request<Incoming>,
    _peer: SocketAddr,
    tx: mpsc::Sender<InboundConnection>,
) -> Result<hyper::Response<Full<Bytes>>, hyper::Error> {
    let (parts, body) = req.into_parts();
    let body_bytes = body
        .collect()
        .await
        .map(|b| b.to_bytes())
        .unwrap_or_default();

    let proxy_req = ProxyRequest {
        id: RequestId::new(),
        method: parts.method,
        uri: parts.uri,
        headers: parts.headers,
        body: ProxyBody::from(body_bytes),
        transport: TransportKind::Http,
        timestamp: chrono::Utc::now(),
        extensions: crate::types::Extensions::new(),
    };

    let (resp_tx, resp_rx) = oneshot::channel();

    if tx.send((proxy_req.clone(), resp_tx)).await.is_err() {
        let mut resp = hyper::Response::new(Full::new(Bytes::from("Proxy unavailable")));
        *resp.status_mut() = StatusCode::BAD_GATEWAY;
        return Ok(resp);
    }

    match resp_rx.await {
        Ok(proxy_resp) => Ok(proxy_response_to_hyper(proxy_resp)),
        Err(_) => {
            let mut resp = hyper::Response::new(Full::new(Bytes::from("Upstream timeout")));
            *resp.status_mut() = StatusCode::GATEWAY_TIMEOUT;
            Ok(resp)
        }
    }
}

fn proxy_response_to_hyper(resp: ProxyResponse) -> hyper::Response<Full<Bytes>> {
    let mut builder = hyper::Response::builder().status(resp.status);
    if let Some(headers) = builder.headers_mut() {
        *headers = resp.headers;
    }
    builder
        .body(Full::new(resp.body.into_bytes()))
        .unwrap_or_else(|_| hyper::Response::new(Full::new(Bytes::from("Internal proxy error"))))
}

/// HTTP connector — forwards requests to an upstream URL using hyper client.
pub struct HttpConnector {
    upstream_url: url::Url,
}

impl HttpConnector {
    pub fn new(upstream_url: url::Url) -> Self {
        Self { upstream_url }
    }
}

#[async_trait::async_trait]
impl TransportConnector for HttpConnector {
    async fn forward(&self, request: ProxyRequest) -> ProxyResult<ProxyResponse> {
        use hyper_util::client::legacy::Client;
        use hyper_util::rt::TokioExecutor;

        let client: Client<_, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();

        // Build upstream URI by combining upstream base URL with request path/query
        let mut upstream_uri = self.upstream_url.clone();
        upstream_uri.set_path(request.uri.path());
        upstream_uri.set_query(request.uri.query());

        let uri: Uri = upstream_uri
            .as_str()
            .parse()
            .map_err(|e: http::uri::InvalidUri| ProxyError::Connection(e.to_string()))?;

        let mut builder = hyper::Request::builder().method(request.method).uri(uri);

        if let Some(headers) = builder.headers_mut() {
            *headers = request.headers;
        }

        let hyper_req = builder
            .body(Full::new(request.body.into_bytes()))
            .map_err(|e| ProxyError::Connection(e.to_string()))?;

        let hyper_resp = client
            .request(hyper_req)
            .await
            .map_err(|e| ProxyError::UpstreamUnreachable(e.to_string()))?;

        let status = hyper_resp.status();
        let headers = hyper_resp.headers().clone();
        let body_bytes = hyper_resp
            .into_body()
            .collect()
            .await
            .map(|b| b.to_bytes())
            .map_err(|e| ProxyError::Transport(e.to_string()))?;

        Ok(ProxyResponse {
            id: request.id,
            status,
            headers,
            body: ProxyBody::from(body_bytes),
            timestamp: chrono::Utc::now(),
            extensions: crate::types::Extensions::new(),
        })
    }

    fn connector_name(&self) -> &str {
        "http"
    }
}
