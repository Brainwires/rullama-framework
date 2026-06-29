# brainwires-proxy

[![Crates.io](https://img.shields.io/crates/v/brainwires-proxy.svg)](https://crates.io/crates/brainwires-proxy)
[![Documentation](https://img.shields.io/docsrs/brainwires-proxy)](https://docs.rs/brainwires-proxy)
[![License](https://img.shields.io/crates/l/brainwires-proxy.svg)](LICENSE)

Protocol-agnostic proxy framework for debugging and transforming application traffic.

## Overview

`brainwires-proxy` is a composable, async-first proxy framework built on Tokio. It supports multiple transport protocols, pluggable middleware, format conversion, and live traffic inspection — all behind a fluent builder API.

**Design principles:**

- **Standalone** — no dependency on `brainwires-core` or the rest of the Brainwires ecosystem
- **Composable** — mix and match transports, middleware, and converters via traits
- **Async-native** — built entirely on `tokio`, `futures`, and `async-trait`

```text
                ┌───────────────────────────────────────────────┐
                │                ProxyService                   │
                │                                               │
 Client ──────►  Listener  ──►  Middleware Stack  ──►  Connector  ──────►  Upstream
                │                 (onion model)                 │
                │                                               │
                │          ┌──────────┐  ┌────────────┐         │
                │          │Inspector │  │ Conversion │         │
                │          │  Store   │  │  Registry  │         │
                │          └──────────┘  └────────────┘         │
                └───────────────────────────────────────────────┘
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
brainwires-proxy = "0.11"
```

Minimal HTTP reverse proxy:

```rust
use brainwires_proxy::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proxy = ProxyBuilder::new()
        .listen_on("127.0.0.1:8080")
        .upstream_url("http://localhost:3000")
        .with_logging()
        .build()?;

    proxy.run().await?;
    Ok(())
}
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `http` | Yes | HTTP/1.1 and HTTP/2 transport via Hyper |
| `websocket` | No | WebSocket transport via tokio-tungstenite |
| `tls` | No | TLS termination via rustls |
| `inspector-api` | No | HTTP API for querying captured traffic events |
| `full` | No | Enables all features |

TCP and Unix socket transports are always available regardless of feature flags.

Enable features in `Cargo.toml`:

```toml
# Pick what you need
brainwires-proxy = { version = "0.11", features = ["websocket", "inspector-api"] }

# Or enable everything
brainwires-proxy = { version = "0.11", features = ["full"] }
```

## Architecture

### Transport

Defines how the proxy accepts inbound connections and forwards them upstream.

**Traits:**

- `TransportListener` — accepts inbound connections, produces `(ProxyRequest, oneshot::Sender<ProxyResponse>)` pairs
- `TransportConnector` — forwards a `ProxyRequest` to the upstream and returns a `ProxyResponse`

**Built-in implementations:**

| Struct | Feature | Direction | Description |
|--------|---------|-----------|-------------|
| `HttpListener` | `http` | Inbound | HTTP/1.1 + HTTP/2 server via Hyper |
| `HttpConnector` | `http` | Outbound | HTTP client, combines upstream base URL with request path |
| `WebSocketListener` | `websocket` | Inbound | WebSocket server, each message becomes a `ProxyRequest` |
| `TcpRawListener` | — | Inbound | Raw TCP, reads full payload into a single request body |
| `UnixListener` | — | Inbound | Unix domain socket, auto-cleans stale socket files |

SSE utilities (`is_sse_response`, `parse_sse_chunk`, `serialize_sse_event`) are also provided for working with Server-Sent Events streams.

### Middleware

Middleware follows an **onion model**: requests flow forward through layers, responses flow back in reverse order.

**Trait:**

```rust
#[async_trait]
pub trait ProxyLayer: Send + Sync {
    async fn on_request(&self, request: ProxyRequest) -> ProxyResult<LayerAction>;
    async fn on_response(&self, response: ProxyResponse) -> ProxyResult<ProxyResponse> {
        Ok(response) // default: pass through
    }
    fn name(&self) -> &str;
}
```

`LayerAction::Forward(req)` continues to the next layer; `LayerAction::Respond(resp)` short-circuits the chain.

**Built-in layers:**

| Layer | Description |
|-------|-------------|
| `LoggingLayer` | Structured request/response logging via `tracing`, optional body capture |
| `InspectorLayer` | Captures traffic events into `EventStore` and broadcasts to subscribers |
| `RateLimitLayer` | Token-bucket rate limiter, returns 429 when exhausted |
| `AuthLayer` | Auth strategies: `StaticBearer`, `Passthrough`, `Validate`, `Strip` |
| `HeaderInjectLayer` | Set, append, or remove headers on requests and/or responses |

### Conversion

Transform request/response bodies between formats with auto-detection.

**Traits:**

- `Converter` — converts a complete body (`Bytes → Bytes`) between a source and target `FormatId`
- `StreamConverter` — converts chunk-by-chunk for streaming scenarios
- `FormatDetector` — inspects body bytes and/or `Content-Type` to determine the format

**`ConversionRegistry`** ties them together: register converters and detectors, then call `registry.convert(body, source, target, content_type)` — it auto-detects the source format if not provided.

**Built-in components:**

| Struct | Description |
|--------|-------------|
| `GenericJsonDetector` | Detects any valid JSON (by content-type or syntax) |
| `JsonFieldDetector` | Detects JSON containing specific fields |
| `JsonTransformer` | Applies `JsonRule` transforms: rename, remove, set, wrap, unwrap fields |

**`JsonRule` variants:**

| Rule | Description |
|------|-------------|
| `RenameField { from, to }` | Rename a top-level field |
| `RemoveField(path)` | Remove a field |
| `SetField { path, value }` | Set a field to a value (dot-separated paths) |
| `WrapIn(key)` | Wrap the entire body as a nested object |
| `Unwrap(key)` | Extract a nested value as the new root |

### Inspector

Captures and broadcasts traffic events for debugging and analysis.

- **`EventStore`** — ring-buffer storage with configurable capacity, auto-evicts oldest events
- **`EventBroadcaster`** — `tokio::sync::broadcast`-based live event fan-out to subscribers
- **`TrafficEvent`** — event record containing request ID, timestamp, direction, and kind
- **`EventFilter`** — query by direction, request ID, event kind, timestamp, or limit

**`TrafficEventKind` variants:** `Request`, `Response`, `SseEvent`, `WebSocketMessage`, `Error`, `Connection`, `Conversion`

## Usage Examples

### HTTP Reverse Proxy with Logging

```rust
use brainwires_proxy::prelude::*;

let proxy = ProxyBuilder::new()
    .listen_on("0.0.0.0:8080")
    .upstream_url("http://api.internal:3000")
    .with_body_logging()    // logs request + response bodies
    .build()?;

proxy.run().await?;
```

### Auth Token Injection

```rust
use brainwires_proxy::middleware::auth::{AuthLayer, AuthStrategy};

let proxy = ProxyBuilder::new()
    .listen_on("127.0.0.1:8080")
    .upstream_url("https://api.example.com")
    .layer(AuthLayer::static_bearer("my-secret-token"))
    .with_logging()
    .build()?;
```

### Rate Limiting

```rust
use brainwires_proxy::middleware::rate_limit::RateLimitLayer;

let proxy = ProxyBuilder::new()
    .listen_on("127.0.0.1:8080")
    .upstream_url("http://localhost:3000")
    .layer(RateLimitLayer::new(
        100.0,  // burst capacity
        10.0,   // sustained requests per second
    ))
    .build()?;
```

### Header Manipulation

```rust
use brainwires_proxy::middleware::header_inject::{HeaderInjectLayer, HeaderRule};
use http::header::{HeaderName, HeaderValue};

let proxy = ProxyBuilder::new()
    .listen_on("127.0.0.1:8080")
    .upstream_url("http://localhost:3000")
    .layer(
        HeaderInjectLayer::new()
            .set_request_header(
                HeaderName::from_static("x-forwarded-by"),
                HeaderValue::from_static("brainwires-proxy"),
            )
            .remove_request_header(HeaderName::from_static("x-internal-only"))
            .set_response_header(
                HeaderName::from_static("x-proxy-version"),
                HeaderValue::from_static("0.1"),
            ),
    )
    .build()?;
```

### Traffic Inspection with Live Broadcast

```rust
use brainwires_proxy::prelude::*;
use std::net::SocketAddr;

let proxy = ProxyBuilder::new()
    .listen_on("127.0.0.1:8080")
    .upstream_url("http://localhost:3000")
    .with_inspector()
    .with_inspector_api("127.0.0.1:9090".parse::<SocketAddr>().unwrap())
    .inspector_capacity(50_000)
    .build()?;

// Access captured events programmatically
let store = proxy.event_store().clone();
let broadcaster = proxy.event_broadcaster().clone();

// Subscribe to live events
let mut rx = broadcaster.subscribe();
tokio::spawn(async move {
    while let Ok(event) = rx.recv().await {
        println!("{:?}", event.kind);
    }
});

proxy.run().await?;
```

### Custom Middleware

```rust
use brainwires_proxy::prelude::*;
use async_trait::async_trait;

struct TimingLayer;

#[async_trait]
impl ProxyLayer for TimingLayer {
    async fn on_request(&self, mut request: ProxyRequest) -> ProxyResult<LayerAction> {
        request.extensions.insert(std::time::Instant::now());
        Ok(LayerAction::Forward(request))
    }

    async fn on_response(&self, response: ProxyResponse) -> ProxyResult<ProxyResponse> {
        // Response flows back through layers in reverse order
        Ok(response)
    }

    fn name(&self) -> &str {
        "timing"
    }
}

let proxy = ProxyBuilder::new()
    .listen_on("127.0.0.1:8080")
    .upstream_url("http://localhost:3000")
    .layer(TimingLayer)
    .build()?;
```

### Format Conversion with Auto-Detection

```rust
use brainwires_proxy::convert::{
    ConversionRegistry, FormatDetector,
    detect::JsonFieldDetector,
    json_transform::{JsonTransformer, JsonRule},
};
use brainwires_proxy::types::FormatId;

let mut registry = ConversionRegistry::new();

// Detect API responses by field presence
registry.register_detector(JsonFieldDetector::new(
    FormatId::new("api-v1"),
    vec!["data".into(), "meta".into()],
));

// Transform: unwrap nested "data" field to top level
let transformer = JsonTransformer::new(vec![
    JsonRule::Unwrap("data".into()),
]);
```

### Validate Inbound Auth

```rust
use brainwires_proxy::middleware::auth::AuthLayer;

// Reject requests without a valid token (returns 401)
let proxy = ProxyBuilder::new()
    .listen_on("127.0.0.1:8080")
    .upstream_url("http://localhost:3000")
    .layer(AuthLayer::validate("expected-secret-token"))
    .build()?;
```

## Configuration

`ProxyConfig` can be built programmatically or deserialized from JSON/TOML:

```rust
pub struct ProxyConfig {
    pub listener: ListenerConfig,      // Where to bind
    pub upstream: UpstreamConfig,      // Where to forward
    pub max_body_size: usize,          // Default: 10 MiB
    pub timeout: Duration,             // Default: 30s
    pub inspector: InspectorConfig,    // Traffic capture settings
    pub metadata: HashMap<String, String>,
}
```

**Listener variants:**

| Variant | Fields | Default |
|---------|--------|---------|
| `Tcp` | `addr: SocketAddr` | `127.0.0.1:8080` |
| `Unix` | `path: PathBuf` | — |

**Upstream variants:**

| Variant | Fields | Default |
|---------|--------|---------|
| `Url` | `url: String` | `http://localhost:3000` |
| `Tcp` | `host: String, port: u16` | — |
| `Unix` | `path: PathBuf` | — |

**JSON example:**

```json
{
  "listener": { "type": "tcp", "addr": "127.0.0.1:8080" },
  "upstream": { "type": "url", "url": "http://localhost:3000" },
  "max_body_size": 10485760,
  "timeout": { "secs": 30, "nanos": 0 },
  "inspector": {
    "enabled": true,
    "event_capacity": 10000,
    "broadcast_capacity": 256,
    "api_addr": "127.0.0.1:9090"
  },
  "metadata": {}
}
```

## Inspector API

When `inspector-api` is enabled and an API address is configured, two HTTP endpoints are exposed:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/events` | GET | Query captured traffic events |
| `/stats` | GET | Get event store statistics |

### `GET /events`

| Parameter | Type | Description |
|-----------|------|-------------|
| `direction` | `inbound` / `outbound` | Filter by traffic direction |
| `kind` | `request`, `response`, `error`, ... | Filter by event kind |
| `request_id` | string | Filter by request ID |
| `limit` | number | Max events to return |

```bash
# All events
curl http://127.0.0.1:9090/events

# Only inbound requests, last 50
curl 'http://127.0.0.1:9090/events?direction=inbound&kind=request&limit=50'

# Events for a specific request
curl 'http://127.0.0.1:9090/events?request_id=42-a1b2c3d4'
```

### `GET /stats`

```bash
curl http://127.0.0.1:9090/stats
```

```json
{
  "stored": 1523,
  "capacity": 10000,
  "total_pushed": 1523,
  "evicted": 0
}
```

## Custom Implementations

### Custom Transport Listener

```rust
use brainwires_proxy::prelude::*;
use brainwires_proxy::transport::InboundConnection;
use tokio::sync::{mpsc, watch};
use async_trait::async_trait;

struct MyListener { /* ... */ }

#[async_trait]
impl TransportListener for MyListener {
    async fn listen(
        &self,
        tx: mpsc::Sender<InboundConnection>,
        shutdown: watch::Receiver<bool>,
    ) -> ProxyResult<()> {
        // Accept connections, create ProxyRequest, send via tx
        // Watch shutdown for graceful termination
        Ok(())
    }

    fn transport_name(&self) -> &str { "my-transport" }
}

let proxy = ProxyBuilder::new()
    .listener(MyListener { /* ... */ })
    .upstream_url("http://localhost:3000")
    .build()?;
```

### Custom Converter + Format Detector

```rust
use brainwires_proxy::prelude::*;
use async_trait::async_trait;
use bytes::Bytes;

struct MsgPackToJson;

#[async_trait]
impl Converter for MsgPackToJson {
    fn source(&self) -> &FormatId { &FormatId::new("msgpack") }
    fn target(&self) -> &FormatId { &FormatId::new("json") }

    async fn convert(&self, body: Bytes) -> ProxyResult<Bytes> {
        // Decode msgpack → serialize as JSON
        todo!()
    }
}

struct MsgPackDetector;

impl FormatDetector for MsgPackDetector {
    fn detect(&self, body: &[u8], content_type: Option<&str>) -> Option<FormatId> {
        if content_type == Some("application/msgpack") {
            Some(FormatId::new("msgpack"))
        } else {
            None
        }
    }
    fn name(&self) -> &str { "msgpack" }
}
```

## Integration with Brainwires

Use via the `brainwires` facade crate:

```toml
[dependencies]
brainwires = { version = "0.11", features = ["proxy"] }
```

Or use standalone — `brainwires-proxy` has no dependency on any other Brainwires crate.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
