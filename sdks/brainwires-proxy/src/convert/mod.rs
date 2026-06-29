//! Body conversion system — registry-based format conversion for request/response bodies.

pub mod detect;
pub mod json_transform;

use crate::error::{ProxyError, ProxyResult};
use crate::types::FormatId;
use bytes::Bytes;
use std::collections::HashMap;

/// Converts a complete body between two formats atomically.
#[async_trait::async_trait]
pub trait Converter: Send + Sync {
    /// Source format this converter reads.
    fn source(&self) -> &FormatId;
    /// Target format this converter produces.
    fn target(&self) -> &FormatId;
    /// Convert the body bytes.
    async fn convert(&self, body: Bytes) -> ProxyResult<Bytes>;
}

/// Converts streaming data chunk-by-chunk (for SSE, WebSocket, etc.).
#[async_trait::async_trait]
pub trait StreamConverter: Send + Sync {
    fn source(&self) -> &FormatId;
    fn target(&self) -> &FormatId;
    /// Process one chunk, returning zero or more output chunks.
    async fn convert_chunk(&self, chunk: Bytes) -> ProxyResult<Vec<Bytes>>;
    /// Flush any buffered data at end of stream.
    async fn flush(&self) -> ProxyResult<Vec<Bytes>>;
}

/// Detects the format of a body payload.
pub trait FormatDetector: Send + Sync {
    /// Inspect bytes and return the detected format, or `None` if unknown.
    fn detect(&self, body: &[u8], content_type: Option<&str>) -> Option<FormatId>;
    /// Human-readable detector name.
    fn name(&self) -> &str;
}

/// Registry mapping `(source, target)` format pairs to converters.
pub struct ConversionRegistry {
    converters: HashMap<(FormatId, FormatId), Box<dyn Converter>>,
    stream_converters: HashMap<(FormatId, FormatId), Box<dyn StreamConverter>>,
    detectors: Vec<Box<dyn FormatDetector>>,
}

impl ConversionRegistry {
    pub fn new() -> Self {
        Self {
            converters: HashMap::new(),
            stream_converters: HashMap::new(),
            detectors: Vec::new(),
        }
    }

    /// Register an atomic converter for a format pair.
    pub fn register_converter(&mut self, converter: impl Converter + 'static) {
        let key = (converter.source().clone(), converter.target().clone());
        self.converters.insert(key, Box::new(converter));
    }

    /// Register a streaming converter for a format pair.
    pub fn register_stream_converter(&mut self, converter: impl StreamConverter + 'static) {
        let key = (converter.source().clone(), converter.target().clone());
        self.stream_converters.insert(key, Box::new(converter));
    }

    /// Register a format detector.
    pub fn register_detector(&mut self, detector: impl FormatDetector + 'static) {
        self.detectors.push(Box::new(detector));
    }

    /// Look up an atomic converter.
    pub fn get_converter(&self, source: &FormatId, target: &FormatId) -> Option<&dyn Converter> {
        self.converters
            .get(&(source.clone(), target.clone()))
            .map(|c| c.as_ref())
    }

    /// Look up a streaming converter.
    pub fn get_stream_converter(
        &self,
        source: &FormatId,
        target: &FormatId,
    ) -> Option<&dyn StreamConverter> {
        self.stream_converters
            .get(&(source.clone(), target.clone()))
            .map(|c| c.as_ref())
    }

    /// Detect the format of a body payload using registered detectors.
    pub fn detect_format(&self, body: &[u8], content_type: Option<&str>) -> Option<FormatId> {
        for detector in &self.detectors {
            if let Some(fmt) = detector.detect(body, content_type) {
                return Some(fmt);
            }
        }
        None
    }

    /// Convert body between formats, auto-detecting source if not provided.
    pub async fn convert(
        &self,
        body: Bytes,
        source: Option<&FormatId>,
        target: &FormatId,
        content_type: Option<&str>,
    ) -> ProxyResult<Bytes> {
        let detected;
        let source = match source {
            Some(s) => s,
            None => {
                detected = self
                    .detect_format(&body, content_type)
                    .ok_or(ProxyError::FormatDetectionFailed)?;
                &detected
            }
        };

        let converter = self.get_converter(source, target).ok_or_else(|| {
            ProxyError::UnsupportedConversion {
                src: source.to_string(),
                dst: target.to_string(),
            }
        })?;

        converter.convert(body).await
    }
}

impl Default for ConversionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple test converter that upper-cases body text.
    struct UpperCaseConverter {
        source: FormatId,
        target: FormatId,
    }

    impl UpperCaseConverter {
        fn new() -> Self {
            Self {
                source: FormatId::new("text"),
                target: FormatId::new("upper"),
            }
        }
    }

    #[async_trait::async_trait]
    impl Converter for UpperCaseConverter {
        fn source(&self) -> &FormatId {
            &self.source
        }
        fn target(&self) -> &FormatId {
            &self.target
        }
        async fn convert(&self, body: Bytes) -> ProxyResult<Bytes> {
            let text = String::from_utf8_lossy(&body).to_uppercase();
            Ok(Bytes::from(text))
        }
    }

    /// A detector that identifies "text" format.
    struct TextDetector;

    impl FormatDetector for TextDetector {
        fn detect(&self, _body: &[u8], content_type: Option<&str>) -> Option<FormatId> {
            if content_type?.contains("text/plain") {
                Some(FormatId::new("text"))
            } else {
                None
            }
        }
        fn name(&self) -> &str {
            "text_detector"
        }
    }

    #[tokio::test]
    async fn register_and_lookup_converter() {
        let mut registry = ConversionRegistry::new();
        registry.register_converter(UpperCaseConverter::new());

        let source = FormatId::new("text");
        let target = FormatId::new("upper");
        assert!(registry.get_converter(&source, &target).is_some());
        assert!(registry.get_converter(&target, &source).is_none());
    }

    #[tokio::test]
    async fn convert_body() {
        let mut registry = ConversionRegistry::new();
        registry.register_converter(UpperCaseConverter::new());

        let source = FormatId::new("text");
        let target = FormatId::new("upper");
        let result = registry
            .convert(Bytes::from("hello"), Some(&source), &target, None)
            .await
            .unwrap();
        assert_eq!(result.as_ref(), b"HELLO");
    }

    #[tokio::test]
    async fn auto_detect_source_format() {
        let mut registry = ConversionRegistry::new();
        registry.register_converter(UpperCaseConverter::new());
        registry.register_detector(TextDetector);

        let target = FormatId::new("upper");
        let result = registry
            .convert(Bytes::from("world"), None, &target, Some("text/plain"))
            .await
            .unwrap();
        assert_eq!(result.as_ref(), b"WORLD");
    }

    #[tokio::test]
    async fn detection_failure_returns_error() {
        let registry = ConversionRegistry::new(); // no detectors
        let target = FormatId::new("upper");
        let result = registry
            .convert(Bytes::from("data"), None, &target, None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn unsupported_conversion_returns_error() {
        let mut registry = ConversionRegistry::new();
        registry.register_converter(UpperCaseConverter::new());

        let source = FormatId::new("text");
        let target = FormatId::new("nonexistent");
        let result = registry
            .convert(Bytes::from("data"), Some(&source), &target, None)
            .await;
        assert!(result.is_err());
    }
}
