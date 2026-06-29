//! Middleware pipeline with onion-model request/response processing.

pub mod auth;
pub mod header_inject;
pub mod inspector;
pub mod logging;
pub mod rate_limit;

use crate::error::ProxyResult;
use crate::types::{ProxyRequest, ProxyResponse};

/// Action a middleware layer can take on a request.
pub enum LayerAction {
    /// Forward the (possibly modified) request to the next layer.
    Forward(ProxyRequest),
    /// Short-circuit and return this response immediately.
    Respond(ProxyResponse),
}

/// A single middleware layer in the proxy pipeline.
///
/// Layers form an onion: requests flow inward through `on_request()`,
/// responses flow outward through `on_response()` in reverse order.
#[async_trait::async_trait]
pub trait ProxyLayer: Send + Sync {
    /// Process an incoming request. Return `Forward` to pass it on,
    /// or `Respond` to short-circuit with an immediate response.
    async fn on_request(&self, request: ProxyRequest) -> ProxyResult<LayerAction>;

    /// Process a response before it's sent back to the client.
    /// Called in reverse layer order.
    async fn on_response(&self, response: ProxyResponse) -> ProxyResult<ProxyResponse> {
        Ok(response)
    }

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}

/// Ordered stack of middleware layers implementing the onion model.
pub struct MiddlewareStack {
    layers: Vec<Box<dyn ProxyLayer>>,
}

impl MiddlewareStack {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Push a layer onto the stack. Layers are processed in insertion order
    /// for requests and reverse order for responses.
    pub fn push(&mut self, layer: impl ProxyLayer + 'static) {
        self.layers.push(Box::new(layer));
    }

    /// Process a request through all layers.
    /// Returns the (possibly modified) request and the index of the deepest
    /// layer reached, or a short-circuit response.
    pub async fn process_request(
        &self,
        mut request: ProxyRequest,
    ) -> ProxyResult<Result<(ProxyRequest, usize), ProxyResponse>> {
        for layer in self.layers.iter() {
            match layer.on_request(request).await? {
                LayerAction::Forward(req) => request = req,
                LayerAction::Respond(resp) => return Ok(Err(resp)),
            }
        }
        Ok(Ok((request, self.layers.len())))
    }

    /// Process a response back through layers in reverse order.
    /// `depth` is the number of layers the request passed through.
    pub async fn process_response(
        &self,
        mut response: ProxyResponse,
        depth: usize,
    ) -> ProxyResult<ProxyResponse> {
        for layer in self.layers[..depth].iter().rev() {
            response = layer.on_response(response).await?;
        }
        Ok(response)
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }
}

impl Default for MiddlewareStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::{Method, StatusCode};
    use std::sync::Arc;

    fn make_request() -> ProxyRequest {
        ProxyRequest::new(Method::GET, "/test".parse().unwrap()).with_body("hello")
    }

    /// A layer that appends a marker header and tracks call order.
    struct MarkerLayer {
        name: String,
        order: Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl ProxyLayer for MarkerLayer {
        async fn on_request(&self, mut request: ProxyRequest) -> ProxyResult<LayerAction> {
            self.order
                .lock()
                .unwrap()
                .push(format!("{}-req", self.name));
            request.headers.insert(
                http::header::HeaderName::from_bytes(self.name.as_bytes()).unwrap(),
                http::header::HeaderValue::from_static("true"),
            );
            Ok(LayerAction::Forward(request))
        }

        async fn on_response(&self, response: ProxyResponse) -> ProxyResult<ProxyResponse> {
            self.order
                .lock()
                .unwrap()
                .push(format!("{}-resp", self.name));
            Ok(response)
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    /// A layer that short-circuits with a 403 response.
    struct BlockingLayer;

    #[async_trait::async_trait]
    impl ProxyLayer for BlockingLayer {
        async fn on_request(&self, request: ProxyRequest) -> ProxyResult<LayerAction> {
            Ok(LayerAction::Respond(
                ProxyResponse::for_request(request.id, StatusCode::FORBIDDEN).with_body("blocked"),
            ))
        }
        fn name(&self) -> &str {
            "blocker"
        }
    }

    #[tokio::test]
    async fn empty_stack_passes_through() {
        let stack = MiddlewareStack::new();
        let req = make_request();
        let result = stack.process_request(req).await.unwrap();
        assert!(result.is_ok());
        let (req, depth) = result.unwrap();
        assert_eq!(depth, 0);
        assert_eq!(req.body.as_bytes(), b"hello");
    }

    #[tokio::test]
    async fn onion_model_order() {
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut stack = MiddlewareStack::new();

        stack.push(MarkerLayer {
            name: "a".into(),
            order: order.clone(),
        });
        stack.push(MarkerLayer {
            name: "b".into(),
            order: order.clone(),
        });
        stack.push(MarkerLayer {
            name: "c".into(),
            order: order.clone(),
        });

        let req = make_request();
        let result = stack.process_request(req).await.unwrap().unwrap();
        let (_, depth) = result;
        assert_eq!(depth, 3);

        let resp = ProxyResponse::new(StatusCode::OK);
        stack.process_response(resp, depth).await.unwrap();

        let log = order.lock().unwrap();
        // Request order: a, b, c; Response order: c, b, a
        assert_eq!(
            *log,
            vec!["a-req", "b-req", "c-req", "c-resp", "b-resp", "a-resp"]
        );
    }

    #[tokio::test]
    async fn short_circuit_stops_processing() {
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut stack = MiddlewareStack::new();

        stack.push(MarkerLayer {
            name: "a".into(),
            order: order.clone(),
        });
        stack.push(BlockingLayer);
        stack.push(MarkerLayer {
            name: "c".into(),
            order: order.clone(),
        });

        let req = make_request();
        let result = stack.process_request(req).await.unwrap();
        assert!(result.is_err()); // short-circuited
        let resp = result.unwrap_err();
        assert_eq!(resp.status, StatusCode::FORBIDDEN);

        let log = order.lock().unwrap();
        // Only 'a' was called, 'c' was never reached
        assert_eq!(*log, vec!["a-req"]);
    }

    #[tokio::test]
    async fn response_depth_limits_reverse_traversal() {
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut stack = MiddlewareStack::new();

        stack.push(MarkerLayer {
            name: "a".into(),
            order: order.clone(),
        });
        stack.push(MarkerLayer {
            name: "b".into(),
            order: order.clone(),
        });
        stack.push(MarkerLayer {
            name: "c".into(),
            order: order.clone(),
        });

        // Process response with depth=2 (only a,b should run on_response)
        let resp = ProxyResponse::new(StatusCode::OK);
        stack.process_response(resp, 2).await.unwrap();

        let log = order.lock().unwrap();
        assert_eq!(*log, vec!["b-resp", "a-resp"]);
    }
}
