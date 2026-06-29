use bytes::Bytes;
use http::{HeaderMap, Method, StatusCode, Uri};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use crate::request_id::RequestId;

/// The transport protocol over which a request arrived.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TransportKind {
    Http,
    WebSocket,
    Tcp,
    Unix,
    Sse,
}

/// Body payload for proxied messages.
#[derive(Debug, Clone)]
pub enum ProxyBody {
    /// Complete body available in memory.
    Full(Bytes),
    /// Empty body.
    Empty,
}

impl ProxyBody {
    pub fn is_empty(&self) -> bool {
        match self {
            ProxyBody::Full(b) => b.is_empty(),
            ProxyBody::Empty => true,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ProxyBody::Full(b) => b.len(),
            ProxyBody::Empty => 0,
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            ProxyBody::Full(b) => b,
            ProxyBody::Empty => &[],
        }
    }

    pub fn into_bytes(self) -> Bytes {
        match self {
            ProxyBody::Full(b) => b,
            ProxyBody::Empty => Bytes::new(),
        }
    }
}

impl From<Bytes> for ProxyBody {
    fn from(b: Bytes) -> Self {
        if b.is_empty() {
            ProxyBody::Empty
        } else {
            ProxyBody::Full(b)
        }
    }
}

impl From<Vec<u8>> for ProxyBody {
    fn from(v: Vec<u8>) -> Self {
        Bytes::from(v).into()
    }
}

impl From<String> for ProxyBody {
    fn from(s: String) -> Self {
        Bytes::from(s).into()
    }
}

impl From<&str> for ProxyBody {
    fn from(s: &str) -> Self {
        Bytes::copy_from_slice(s.as_bytes()).into()
    }
}

/// Type-safe extension map for attaching arbitrary metadata to requests/responses.
#[derive(Default, Clone)]
pub struct Extensions {
    map: HashMap<std::any::TypeId, Arc<dyn Any + Send + Sync>>,
}

impl Extensions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<T: Send + Sync + 'static>(&mut self, val: T) {
        self.map.insert(std::any::TypeId::of::<T>(), Arc::new(val));
    }

    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.map
            .get(&std::any::TypeId::of::<T>())
            .and_then(|v| v.downcast_ref::<T>())
    }
}

impl std::fmt::Debug for Extensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extensions")
            .field("count", &self.map.len())
            .finish()
    }
}

/// A request flowing through the proxy.
#[derive(Debug, Clone)]
pub struct ProxyRequest {
    pub id: RequestId,
    pub method: Method,
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: ProxyBody,
    pub transport: TransportKind,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub extensions: Extensions,
}

impl ProxyRequest {
    pub fn new(method: Method, uri: Uri) -> Self {
        Self {
            id: RequestId::new(),
            method,
            uri,
            headers: HeaderMap::new(),
            body: ProxyBody::Empty,
            transport: TransportKind::Http,
            timestamp: chrono::Utc::now(),
            extensions: Extensions::new(),
        }
    }

    pub fn with_body(mut self, body: impl Into<ProxyBody>) -> Self {
        self.body = body.into();
        self
    }

    pub fn with_headers(mut self, headers: HeaderMap) -> Self {
        self.headers = headers;
        self
    }

    pub fn with_transport(mut self, transport: TransportKind) -> Self {
        self.transport = transport;
        self
    }
}

/// A response flowing back through the proxy.
#[derive(Debug, Clone)]
pub struct ProxyResponse {
    pub id: RequestId,
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: ProxyBody,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub extensions: Extensions,
}

impl ProxyResponse {
    pub fn new(status: StatusCode) -> Self {
        Self {
            id: RequestId::new(),
            status,
            headers: HeaderMap::new(),
            body: ProxyBody::Empty,
            timestamp: chrono::Utc::now(),
            extensions: Extensions::new(),
        }
    }

    pub fn for_request(request_id: RequestId, status: StatusCode) -> Self {
        Self {
            id: request_id,
            status,
            headers: HeaderMap::new(),
            body: ProxyBody::Empty,
            timestamp: chrono::Utc::now(),
            extensions: Extensions::new(),
        }
    }

    pub fn with_body(mut self, body: impl Into<ProxyBody>) -> Self {
        self.body = body.into();
        self
    }

    pub fn with_headers(mut self, headers: HeaderMap) -> Self {
        self.headers = headers;
        self
    }
}

/// Identifier for a body format used by the conversion system.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct FormatId(pub String);

impl FormatId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for FormatId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_body_empty() {
        let body = ProxyBody::Empty;
        assert!(body.is_empty());
        assert_eq!(body.len(), 0);
        assert_eq!(body.as_bytes(), &[] as &[u8]);
        assert!(body.into_bytes().is_empty());
    }

    #[test]
    fn proxy_body_from_str() {
        let body = ProxyBody::from("hello");
        assert!(!body.is_empty());
        assert_eq!(body.len(), 5);
        assert_eq!(body.as_bytes(), b"hello");
    }

    #[test]
    fn proxy_body_from_string() {
        let body = ProxyBody::from("world".to_string());
        assert_eq!(body.as_bytes(), b"world");
    }

    #[test]
    fn proxy_body_from_vec() {
        let body = ProxyBody::from(vec![1, 2, 3]);
        assert_eq!(body.len(), 3);
        assert_eq!(body.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn proxy_body_from_empty_bytes_is_empty_variant() {
        let body = ProxyBody::from(Bytes::new());
        assert!(body.is_empty());
        assert!(matches!(body, ProxyBody::Empty));
    }

    #[test]
    fn extensions_insert_and_get() {
        let mut ext = Extensions::new();
        ext.insert(42u32);
        ext.insert("hello".to_string());

        assert_eq!(ext.get::<u32>(), Some(&42));
        assert_eq!(ext.get::<String>(), Some(&"hello".to_string()));
        assert_eq!(ext.get::<bool>(), None);
    }

    #[test]
    fn proxy_request_builder() {
        let req = ProxyRequest::new(Method::GET, "/api/test".parse().unwrap())
            .with_body("request body")
            .with_transport(TransportKind::Http);

        assert_eq!(req.method, Method::GET);
        assert_eq!(req.uri, "/api/test");
        assert_eq!(req.body.as_bytes(), b"request body");
        assert_eq!(req.transport, TransportKind::Http);
    }

    #[test]
    fn proxy_response_for_request() {
        let req = ProxyRequest::new(Method::POST, "/submit".parse().unwrap());
        let req_id = req.id.clone();
        let resp = ProxyResponse::for_request(req.id, StatusCode::OK).with_body("ok");

        assert_eq!(resp.id, req_id);
        assert_eq!(resp.status, StatusCode::OK);
        assert_eq!(resp.body.as_bytes(), b"ok");
    }

    #[test]
    fn format_id_display() {
        let fmt = FormatId::new("application/json");
        assert_eq!(fmt.to_string(), "application/json");
    }

    #[test]
    fn transport_kind_serde() {
        let kind = TransportKind::WebSocket;
        let json = serde_json::to_string(&kind).unwrap();
        let deserialized: TransportKind = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, TransportKind::WebSocket);
    }
}
