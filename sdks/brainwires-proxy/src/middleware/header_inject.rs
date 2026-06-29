//! Header add/remove/replace middleware.

use crate::error::ProxyResult;
use crate::middleware::{LayerAction, ProxyLayer};
use crate::types::{ProxyRequest, ProxyResponse};
use http::HeaderValue;
use http::header::HeaderName;

/// Rule for modifying headers.
#[derive(Clone)]
pub enum HeaderRule {
    /// Set a header, replacing any existing value.
    Set(HeaderName, HeaderValue),
    /// Append a header value (allows duplicates).
    Append(HeaderName, HeaderValue),
    /// Remove a header.
    Remove(HeaderName),
}

/// Applies header rules to requests and/or responses.
pub struct HeaderInjectLayer {
    request_rules: Vec<HeaderRule>,
    response_rules: Vec<HeaderRule>,
}

impl HeaderInjectLayer {
    pub fn new() -> Self {
        Self {
            request_rules: Vec::new(),
            response_rules: Vec::new(),
        }
    }

    /// Add a rule applied to requests.
    pub fn request_rule(mut self, rule: HeaderRule) -> Self {
        self.request_rules.push(rule);
        self
    }

    /// Add a rule applied to responses.
    pub fn response_rule(mut self, rule: HeaderRule) -> Self {
        self.response_rules.push(rule);
        self
    }

    /// Set a request header.
    pub fn set_request_header(self, name: HeaderName, value: HeaderValue) -> Self {
        self.request_rule(HeaderRule::Set(name, value))
    }

    /// Remove a request header.
    pub fn remove_request_header(self, name: HeaderName) -> Self {
        self.request_rule(HeaderRule::Remove(name))
    }

    /// Set a response header.
    pub fn set_response_header(self, name: HeaderName, value: HeaderValue) -> Self {
        self.response_rule(HeaderRule::Set(name, value))
    }
}

impl Default for HeaderInjectLayer {
    fn default() -> Self {
        Self::new()
    }
}

fn apply_rules(headers: &mut http::HeaderMap, rules: &[HeaderRule]) {
    for rule in rules {
        match rule {
            HeaderRule::Set(name, value) => {
                headers.insert(name.clone(), value.clone());
            }
            HeaderRule::Append(name, value) => {
                headers.append(name.clone(), value.clone());
            }
            HeaderRule::Remove(name) => {
                headers.remove(name);
            }
        }
    }
}

#[async_trait::async_trait]
impl ProxyLayer for HeaderInjectLayer {
    async fn on_request(&self, mut request: ProxyRequest) -> ProxyResult<LayerAction> {
        apply_rules(&mut request.headers, &self.request_rules);
        Ok(LayerAction::Forward(request))
    }

    async fn on_response(&self, mut response: ProxyResponse) -> ProxyResult<ProxyResponse> {
        apply_rules(&mut response.headers, &self.response_rules);
        Ok(response)
    }

    fn name(&self) -> &str {
        "header_inject"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProxyRequest;
    use http::{Method, StatusCode, header};

    fn make_request() -> ProxyRequest {
        ProxyRequest::new(Method::GET, "/test".parse().unwrap())
    }

    #[tokio::test]
    async fn set_request_header() {
        let layer = HeaderInjectLayer::new().set_request_header(
            header::HeaderName::from_static("x-custom"),
            HeaderValue::from_static("value"),
        );

        let result = layer.on_request(make_request()).await.unwrap();
        match result {
            LayerAction::Forward(req) => {
                assert_eq!(req.headers.get("x-custom").unwrap(), "value");
            }
            _ => panic!("expected forward"),
        }
    }

    #[tokio::test]
    async fn remove_request_header() {
        let mut req = make_request();
        req.headers.insert(
            header::HeaderName::from_static("x-remove-me"),
            HeaderValue::from_static("bye"),
        );

        let layer = HeaderInjectLayer::new()
            .remove_request_header(header::HeaderName::from_static("x-remove-me"));

        let result = layer.on_request(req).await.unwrap();
        match result {
            LayerAction::Forward(req) => {
                assert!(req.headers.get("x-remove-me").is_none());
            }
            _ => panic!("expected forward"),
        }
    }

    #[tokio::test]
    async fn set_response_header() {
        let layer = HeaderInjectLayer::new().set_response_header(
            header::HeaderName::from_static("x-proxy"),
            HeaderValue::from_static("brainwires"),
        );

        let resp = crate::types::ProxyResponse::new(StatusCode::OK);
        let resp = layer.on_response(resp).await.unwrap();
        assert_eq!(resp.headers.get("x-proxy").unwrap(), "brainwires");
    }

    #[tokio::test]
    async fn append_creates_multiple_values() {
        let layer = HeaderInjectLayer::new()
            .request_rule(HeaderRule::Append(
                header::HeaderName::from_static("x-tag"),
                HeaderValue::from_static("a"),
            ))
            .request_rule(HeaderRule::Append(
                header::HeaderName::from_static("x-tag"),
                HeaderValue::from_static("b"),
            ));

        let result = layer.on_request(make_request()).await.unwrap();
        match result {
            LayerAction::Forward(req) => {
                let values: Vec<_> = req.headers.get_all("x-tag").iter().collect();
                assert_eq!(values.len(), 2);
            }
            _ => panic!("expected forward"),
        }
    }
}
