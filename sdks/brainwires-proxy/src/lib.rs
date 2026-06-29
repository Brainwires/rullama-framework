//! # brainwires-proxy
//!
//! Protocol-agnostic proxy framework for debugging app traffic.
//!
//! Compose transports, middleware, converters, and inspectors to build
//! custom debugging proxies for any protocol.
//!
//! ## Features
//!
//! - **`http`** (default) — HTTP/HTTPS transport via hyper
//! - **`websocket`** — WebSocket transport via tokio-tungstenite
//! - **`tls`** — TLS termination via tokio-rustls
//! - **`inspector-api`** — HTTP query API for captured traffic
//! - **`full`** — All features enabled
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use brainwires_proxy::builder::ProxyBuilder;
//!
//! # async fn example() -> brainwires_proxy::error::ProxyResult<()> {
//! let proxy = ProxyBuilder::new()
//!     .listen_on("127.0.0.1:8080")
//!     .upstream_url("http://localhost:3000")
//!     .build()?;
//!
//! proxy.run().await
//! # }
//! ```

pub mod config;
pub mod error;
pub mod request_id;
pub mod types;

pub mod convert;
pub mod inspector;
pub mod middleware;
pub mod transport;

pub mod builder;
pub mod proxy;

/// Convenience re-exports.
pub mod prelude {
    pub use crate::builder::ProxyBuilder;
    pub use crate::config::ProxyConfig;
    pub use crate::convert::{ConversionRegistry, Converter, FormatDetector, StreamConverter};
    pub use crate::error::{ProxyError, ProxyResult};
    pub use crate::middleware::{LayerAction, MiddlewareStack, ProxyLayer};
    pub use crate::proxy::ProxyService;
    pub use crate::request_id::RequestId;
    pub use crate::transport::{TransportConnector, TransportListener};
    pub use crate::types::{
        Extensions, FormatId, ProxyBody, ProxyRequest, ProxyResponse, TransportKind,
    };
}
