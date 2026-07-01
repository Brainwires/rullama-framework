//! Format detection implementations.

use crate::convert::FormatDetector;
use crate::types::FormatId;

/// Detects JSON format by checking for known field names in the body.
pub struct JsonFieldDetector {
    /// Format ID to return when detection succeeds.
    format_id: FormatId,
    /// Field names to look for in the JSON. If any are found, detection succeeds.
    fields: Vec<String>,
}

impl JsonFieldDetector {
    pub fn new(format_id: FormatId, fields: Vec<String>) -> Self {
        Self { format_id, fields }
    }
}

impl FormatDetector for JsonFieldDetector {
    fn detect(&self, body: &[u8], content_type: Option<&str>) -> Option<FormatId> {
        // Quick content-type check
        if let Some(ct) = content_type
            && !ct.contains("json")
            && !ct.contains("text")
        {
            return None;
        }

        // Try to parse as JSON
        let value: serde_json::Value = serde_json::from_slice(body).ok()?;
        let obj = value.as_object()?;

        for field in &self.fields {
            if obj.contains_key(field) {
                return Some(self.format_id.clone());
            }
        }

        None
    }

    fn name(&self) -> &str {
        "json_field_detector"
    }
}

/// Generic JSON detector — matches any valid JSON body.
#[derive(Default)]
pub struct GenericJsonDetector;

impl FormatDetector for GenericJsonDetector {
    fn detect(&self, body: &[u8], content_type: Option<&str>) -> Option<FormatId> {
        if let Some(ct) = content_type
            && ct.contains("json")
        {
            return Some(FormatId::new("json"));
        }

        // Try parsing
        if serde_json::from_slice::<serde_json::Value>(body).is_ok() {
            Some(FormatId::new("json"))
        } else {
            None
        }
    }

    fn name(&self) -> &str {
        "generic_json"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_field_detector_matches() {
        let detector = JsonFieldDetector::new(
            FormatId::new("openai"),
            vec!["model".into(), "messages".into()],
        );

        let body = br#"{"model": "gpt-4", "messages": []}"#;
        let result = detector.detect(body, Some("application/json"));
        assert_eq!(result, Some(FormatId::new("openai")));
    }

    #[test]
    fn json_field_detector_no_match() {
        let detector = JsonFieldDetector::new(FormatId::new("openai"), vec!["model".into()]);

        let body = br#"{"data": "something else"}"#;
        let result = detector.detect(body, Some("application/json"));
        assert!(result.is_none());
    }

    #[test]
    fn json_field_detector_rejects_non_json_content_type() {
        let detector = JsonFieldDetector::new(FormatId::new("openai"), vec!["model".into()]);

        let body = br#"{"model": "gpt-4"}"#;
        let result = detector.detect(body, Some("application/xml"));
        assert!(result.is_none());
    }

    #[test]
    fn json_field_detector_invalid_json() {
        let detector = JsonFieldDetector::new(FormatId::new("test"), vec!["key".into()]);

        let body = b"not json at all";
        let result = detector.detect(body, Some("application/json"));
        assert!(result.is_none());
    }

    #[test]
    fn generic_json_detector_by_content_type() {
        let detector = GenericJsonDetector;
        let body = b"not valid json";
        let result = detector.detect(body, Some("application/json"));
        assert_eq!(result, Some(FormatId::new("json")));
    }

    #[test]
    fn generic_json_detector_by_parsing() {
        let detector = GenericJsonDetector;
        let body = br#"{"key": "value"}"#;
        let result = detector.detect(body, None);
        assert_eq!(result, Some(FormatId::new("json")));
    }

    #[test]
    fn generic_json_detector_no_match() {
        let detector = GenericJsonDetector;
        let body = b"plain text";
        let result = detector.detect(body, Some("text/plain"));
        assert!(result.is_none());
    }
}
