//! Helper functions for NornicDB query construction and result mapping.

use rullama_core::SearchResult;
use serde_json::{Value, json};

// ── Helper functions ────────────────────────────────────────────────────

/// Extract the hostname from a URL string, stripping scheme and port.
#[allow(dead_code)]
pub(super) fn extract_host(url: &str) -> String {
    let host = url
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .split(':')
        .next()
        .unwrap_or("localhost")
        .split('/')
        .next()
        .unwrap_or("localhost");
    if host.is_empty() {
        "localhost".to_string()
    } else {
        host.to_string()
    }
}

/// Build a JSON filter object for search endpoints.
pub(super) fn build_filters(
    project: Option<&str>,
    root_path: Option<&str>,
    extensions: &[String],
    languages: &[String],
) -> Value {
    let mut filters = serde_json::Map::new();
    if let Some(p) = project {
        filters.insert("project".into(), json!(p));
    }
    if let Some(rp) = root_path {
        filters.insert("root_path".into(), json!(rp));
    }
    if !extensions.is_empty() {
        filters.insert("extension".into(), json!(extensions));
    }
    if !languages.is_empty() {
        filters.insert("language".into(), json!(languages));
    }
    Value::Object(filters)
}

/// Map a raw search-result JSON value into a [`SearchResult`].
pub(super) fn map_to_search_result(v: &Value) -> Option<SearchResult> {
    Some(SearchResult {
        file_path: v.get("file_path")?.as_str()?.to_string(),
        root_path: v
            .get("root_path")
            .and_then(|v| v.as_str())
            .map(String::from),
        content: v.get("content")?.as_str()?.to_string(),
        score: v.get("score")?.as_f64()? as f32,
        vector_score: v
            .get("vector_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32,
        keyword_score: v
            .get("keyword_score")
            .and_then(|v| v.as_f64())
            .map(|s| s as f32),
        start_line: v.get("start_line").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        end_line: v.get("end_line").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        language: v
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown")
            .to_string(),
        project: v.get("project").and_then(|v| v.as_str()).map(String::from),
        indexed_at: v.get("indexed_at").and_then(|v| v.as_i64()).unwrap_or(0),
    })
}

/// Map a Cypher node result into a [`SearchResult`].
///
/// Cypher nodes may have properties nested or flat depending on the
/// transport — this delegates to [`map_to_search_result`] which handles
/// both shapes.
pub(super) fn map_node_to_search_result(node: &Value) -> Option<SearchResult> {
    map_to_search_result(node)
}
