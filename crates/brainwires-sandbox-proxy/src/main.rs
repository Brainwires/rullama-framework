//! Egress allowlist proxy for `brainwires-sandbox` `NetworkPolicy::Limited`.
//!
//! The sandbox container is placed on an `internal: true` docker network
//! with no default route. This proxy is the only host on that network with
//! external connectivity and its host-match rules are the sole egress
//! policy. Non-HTTP TCP is blocked by design — if you need raw TCP egress,
//! use `NetworkPolicy::Full` and accept the tradeoff.

use std::convert::Infallible;
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use bytes::Bytes;
use http::{HeaderValue, Method, Request, Response, StatusCode, Uri};
use http_body_util::{BodyExt, Empty, Full, combinators::BoxBody};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use hyper_util::rt::{TokioExecutor, TokioIo};
use tokio::io::{AsyncWriteExt, copy_bidirectional};
use tokio::net::{TcpListener, TcpStream};

#[derive(Debug, Clone)]
struct Allowlist {
    exact: Vec<String>,
    wildcard_suffix: Vec<String>,
}

impl Allowlist {
    fn parse(raw: &str) -> Self {
        let mut exact = Vec::new();
        let mut wildcard_suffix = Vec::new();
        for entry in raw.split(',') {
            let e = entry.trim().to_ascii_lowercase();
            if e.is_empty() {
                continue;
            }
            if let Some(rest) = e.strip_prefix("*.") {
                // Wildcard matches any subdomain AND the bare apex.
                wildcard_suffix.push(rest.to_string());
            } else {
                exact.push(e);
            }
        }
        Self {
            exact,
            wildcard_suffix,
        }
    }

    fn matches(&self, host: &str) -> bool {
        let h = host.trim().to_ascii_lowercase();
        if self.exact.iter().any(|e| e == &h) {
            return true;
        }
        for suffix in &self.wildcard_suffix {
            if h == *suffix || h.ends_with(&format!(".{suffix}")) {
                return true;
            }
        }
        false
    }
}

type ResponseBody = BoxBody<Bytes, Box<dyn std::error::Error + Send + Sync>>;

fn forbidden_body(msg: String) -> ResponseBody {
    Full::new(Bytes::from(msg))
        .map_err(|never: Infallible| match never {})
        .boxed()
}

fn empty_body() -> ResponseBody {
    Empty::<Bytes>::new()
        .map_err(|never: Infallible| match never {})
        .boxed()
}

fn strip_port(host_port: &str) -> &str {
    match host_port.rsplit_once(':') {
        Some((h, _)) => h,
        None => host_port,
    }
}

/// Extract the "host" for an HTTP proxy request. Preference order:
/// 1. Request URI authority (absolute-form, what proxies receive).
/// 2. `Host` header.
fn request_host(req: &Request<Incoming>) -> Option<String> {
    if let Some(auth) = req.uri().authority() {
        return Some(auth.host().to_string());
    }
    req.headers()
        .get(http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(|s| strip_port(s).to_string())
}

fn deny(host: &str) -> Response<ResponseBody> {
    let body = forbidden_body(format!("denied by sandbox proxy allowlist: {host}\n"));
    let mut resp = Response::new(body);
    *resp.status_mut() = StatusCode::FORBIDDEN;
    resp.headers_mut().insert(
        http::header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    resp
}

async fn tunnel_connect(
    req: Request<Incoming>,
    target: String,
) -> Result<Response<ResponseBody>, Infallible> {
    // Pre-connect to upstream so we can surface dial errors before hijacking.
    let upstream = match TcpStream::connect(&target).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(%target, error = %e, "CONNECT upstream dial failed");
            let body = forbidden_body(format!("upstream dial failed: {e}\n"));
            let mut resp = Response::new(body);
            *resp.status_mut() = StatusCode::BAD_GATEWAY;
            return Ok(resp);
        }
    };

    tokio::spawn(async move {
        match hyper::upgrade::on(req).await {
            Ok(upgraded) => {
                let mut upgraded = TokioIo::new(upgraded);
                let mut upstream = upstream;
                match copy_bidirectional(&mut upgraded, &mut upstream).await {
                    Ok((c2s, s2c)) => {
                        tracing::debug!(%target, %c2s, %s2c, "CONNECT tunnel closed");
                    }
                    Err(e) => tracing::debug!(%target, error = %e, "CONNECT tunnel error"),
                }
                let _ = upstream.shutdown().await;
            }
            Err(e) => tracing::warn!(error = %e, "upgrade failed"),
        }
    });

    let mut resp = Response::new(empty_body());
    *resp.status_mut() = StatusCode::OK;
    Ok(resp)
}

async fn forward_plain_http(
    client: Arc<Client<HttpConnector, Incoming>>,
    req: Request<Incoming>,
) -> Result<Response<ResponseBody>, Infallible> {
    // Reconstruct the upstream URI. Proxies receive absolute-form URIs for
    // plain HTTP; if the client instead sends origin-form with a Host
    // header, rebuild from scheme/host.
    let uri = if req.uri().scheme().is_some() && req.uri().authority().is_some() {
        req.uri().clone()
    } else {
        let host = match req
            .headers()
            .get(http::header::HOST)
            .and_then(|v| v.to_str().ok())
        {
            Some(h) => h.to_string(),
            None => {
                let body = forbidden_body("missing Host header\n".into());
                let mut resp = Response::new(body);
                *resp.status_mut() = StatusCode::BAD_REQUEST;
                return Ok(resp);
            }
        };
        let path = req
            .uri()
            .path_and_query()
            .map(|p| p.as_str())
            .unwrap_or("/");
        match format!("http://{host}{path}").parse::<Uri>() {
            Ok(u) => u,
            Err(_) => {
                let body = forbidden_body("bad request-target\n".into());
                let mut resp = Response::new(body);
                *resp.status_mut() = StatusCode::BAD_REQUEST;
                return Ok(resp);
            }
        }
    };

    let (mut parts, body) = req.into_parts();
    parts.uri = uri;
    // Strip hop-by-hop headers.
    for h in [
        "connection",
        "proxy-connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
    ] {
        parts.headers.remove(h);
    }
    let upstream_req = Request::from_parts(parts, body);

    match client.request(upstream_req).await {
        Ok(resp) => {
            let (parts, body) = resp.into_parts();
            let body = body
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
                .boxed();
            Ok(Response::from_parts(parts, body))
        }
        Err(e) => {
            tracing::warn!(error = %e, "upstream http forward failed");
            let body = forbidden_body(format!("upstream error: {e}\n"));
            let mut resp = Response::new(body);
            *resp.status_mut() = StatusCode::BAD_GATEWAY;
            Ok(resp)
        }
    }
}

async fn handle(
    req: Request<Incoming>,
    allow: Arc<Allowlist>,
    client: Arc<Client<HttpConnector, Incoming>>,
) -> Result<Response<ResponseBody>, Infallible> {
    if req.method() == Method::CONNECT {
        let target = match req.uri().authority() {
            Some(a) => a.as_str().to_string(),
            None => {
                let body = forbidden_body("CONNECT requires authority-form\n".into());
                let mut resp = Response::new(body);
                *resp.status_mut() = StatusCode::BAD_REQUEST;
                return Ok(resp);
            }
        };
        let host = strip_port(&target);
        if !allow.matches(host) {
            tracing::info!(%host, "CONNECT denied");
            return Ok(deny(host));
        }
        tracing::info!(%host, "CONNECT allowed");
        return tunnel_connect(req, target).await;
    }

    let host = match request_host(&req) {
        Some(h) => h,
        None => {
            let body = forbidden_body("no host in request\n".into());
            let mut resp = Response::new(body);
            *resp.status_mut() = StatusCode::BAD_REQUEST;
            return Ok(resp);
        }
    };

    if !allow.matches(&host) {
        tracing::info!(method = %req.method(), %host, "HTTP denied");
        return Ok(deny(&host));
    }
    tracing::info!(method = %req.method(), %host, "HTTP allowed");
    forward_plain_http(client, req).await
}

fn install_tracing(level: &str) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_new(level).unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let level = env::var("PROXY_LOG").unwrap_or_else(|_| "info".to_string());
    install_tracing(&level);

    let listen: SocketAddr = env::var("PROXY_LISTEN")
        .unwrap_or_else(|_| "0.0.0.0:3128".to_string())
        .parse()
        .context("PROXY_LISTEN must be a valid socket address")?;
    let raw_allow = env::var("PROXY_ALLOW_HOSTS").unwrap_or_default();
    let allow = Arc::new(Allowlist::parse(&raw_allow));

    tracing::info!(
        listen = %listen,
        allow_exact = ?allow.exact,
        allow_wildcard = ?allow.wildcard_suffix,
        "brainwires-sandbox-proxy starting"
    );
    if allow.exact.is_empty() && allow.wildcard_suffix.is_empty() {
        tracing::warn!("PROXY_ALLOW_HOSTS is empty — ALL requests will be denied with 403");
    }

    let listener = TcpListener::bind(listen)
        .await
        .with_context(|| format!("bind {listen}"))?;
    let client: Arc<Client<HttpConnector, Incoming>> =
        Arc::new(Client::builder(TokioExecutor::new()).build(HttpConnector::new()));

    let shutdown = async {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let mut term = match signal(SignalKind::terminate()) {
                Ok(s) => s,
                Err(_) => {
                    let _ = ctrl_c.await;
                    return;
                }
            };
            tokio::select! {
                _ = ctrl_c => {}
                _ = term.recv() => {}
            }
        }
        #[cfg(not(unix))]
        {
            let _ = ctrl_c.await;
        }
    };
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            _ = &mut shutdown => {
                tracing::info!("shutdown signal received, exiting accept loop");
                return Ok(());
            }
            accepted = listener.accept() => {
                let (stream, peer) = match accepted {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!(error = %e, "accept failed");
                        continue;
                    }
                };
                let allow = allow.clone();
                let client = client.clone();
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let svc = service_fn(move |req| handle(req, allow.clone(), client.clone()));
                    // Plain HTTP proxy — HTTP/1.1 only. `with_upgrades`
                    // is required for CONNECT tunnels.
                    let mut builder = http1::Builder::new();
                    builder
                        .preserve_header_case(true)
                        .title_case_headers(true);
                    let result = builder.serve_connection(io, svc).with_upgrades().await;
                    if let Err(e) = result {
                        tracing::debug!(%peer, error = %e, "connection serve error");
                    }
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_exact_match() {
        let a = Allowlist::parse("pypi.org,api.anthropic.com");
        assert!(a.matches("pypi.org"));
        assert!(a.matches("API.ANTHROPIC.COM"));
        assert!(!a.matches("evil.com"));
        assert!(!a.matches("pypi.org.evil.com"));
    }

    #[test]
    fn allowlist_wildcard_suffix_match() {
        let a = Allowlist::parse("*.anthropic.com");
        assert!(a.matches("api.anthropic.com"));
        assert!(a.matches("www.api.anthropic.com"));
        assert!(a.matches("anthropic.com"));
        assert!(!a.matches("notanthropic.com"));
        assert!(!a.matches("anthropic.com.evil.io"));
    }

    #[test]
    fn allowlist_empty_blocks_all() {
        let a = Allowlist::parse("");
        assert!(!a.matches("pypi.org"));
        let a2 = Allowlist::parse("   ,  ,");
        assert!(!a2.matches("pypi.org"));
    }

    #[test]
    fn strip_port_handles_no_port() {
        assert_eq!(strip_port("pypi.org"), "pypi.org");
        assert_eq!(strip_port("pypi.org:443"), "pypi.org");
    }
}
