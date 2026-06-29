//! JSON structural transformation utilities.

use crate::error::{ProxyError, ProxyResult};
use bytes::Bytes;
use serde_json::Value;

/// A rule for transforming a JSON structure.
#[derive(Debug, Clone)]
pub enum JsonRule {
    /// Rename a top-level field.
    RenameField { from: String, to: String },
    /// Remove a field by path (dot-separated).
    RemoveField(String),
    /// Set a field to a constant value.
    SetField { path: String, value: Value },
    /// Wrap the entire body inside a new object field.
    WrapIn(String),
    /// Unwrap: extract the value at a given path and use it as the new root.
    Unwrap(String),
}

/// Applies a sequence of JSON transformation rules.
pub struct JsonTransformer {
    rules: Vec<JsonRule>,
}

impl JsonTransformer {
    pub fn new(rules: Vec<JsonRule>) -> Self {
        Self { rules }
    }

    pub fn transform(&self, input: &[u8]) -> ProxyResult<Bytes> {
        let mut value: Value =
            serde_json::from_slice(input).map_err(|e| ProxyError::Conversion(e.to_string()))?;

        for rule in &self.rules {
            value = apply_rule(value, rule)?;
        }

        let out = serde_json::to_vec(&value).map_err(|e| ProxyError::Conversion(e.to_string()))?;
        Ok(Bytes::from(out))
    }
}

fn apply_rule(mut value: Value, rule: &JsonRule) -> ProxyResult<Value> {
    match rule {
        JsonRule::RenameField { from, to } => {
            if let Value::Object(ref mut map) = value
                && let Some(v) = map.remove(from)
            {
                map.insert(to.clone(), v);
            }
            Ok(value)
        }
        JsonRule::RemoveField(path) => {
            remove_at_path(&mut value, path);
            Ok(value)
        }
        JsonRule::SetField { path, value: val } => {
            set_at_path(&mut value, path, val.clone());
            Ok(value)
        }
        JsonRule::WrapIn(key) => {
            let mut map = serde_json::Map::new();
            map.insert(key.clone(), value);
            Ok(Value::Object(map))
        }
        JsonRule::Unwrap(path) => get_at_path(&value, path)
            .cloned()
            .ok_or_else(|| ProxyError::Conversion(format!("path not found: {path}"))),
    }
}

fn get_at_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for segment in path.split('.') {
        current = current.get(segment)?;
    }
    Some(current)
}

fn set_at_path(value: &mut Value, path: &str, new_val: Value) {
    let segments: Vec<&str> = path.split('.').collect();
    if segments.is_empty() {
        return;
    }

    let mut current = value;
    for segment in &segments[..segments.len() - 1] {
        current = match current {
            Value::Object(map) => map
                .entry(*segment)
                .or_insert_with(|| Value::Object(serde_json::Map::new())),
            _ => return,
        };
    }

    if let Value::Object(map) = current {
        map.insert(segments[segments.len() - 1].to_string(), new_val);
    }
}

fn remove_at_path(value: &mut Value, path: &str) {
    let segments: Vec<&str> = path.split('.').collect();
    if segments.is_empty() {
        return;
    }

    let mut current = value;
    for segment in &segments[..segments.len() - 1] {
        current = match current {
            Value::Object(map) => match map.get_mut(*segment) {
                Some(v) => v,
                None => return,
            },
            _ => return,
        };
    }

    if let Value::Object(map) = current {
        map.remove(segments[segments.len() - 1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rename_field() {
        let input = br#"{"old_name": 42, "keep": true}"#;
        let t = JsonTransformer::new(vec![JsonRule::RenameField {
            from: "old_name".into(),
            to: "new_name".into(),
        }]);
        let out = t.transform(input).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["new_name"], 42);
        assert!(v.get("old_name").is_none());
    }

    #[test]
    fn test_wrap_and_unwrap() {
        let input = br#"{"data": [1,2,3]}"#;
        let t = JsonTransformer::new(vec![JsonRule::WrapIn("wrapper".into())]);
        let out = t.transform(input).unwrap();
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert!(v["wrapper"]["data"].is_array());

        let t2 = JsonTransformer::new(vec![JsonRule::Unwrap("wrapper".into())]);
        let out2 = t2.transform(&out).unwrap();
        let v2: Value = serde_json::from_slice(&out2).unwrap();
        assert_eq!(v2["data"], serde_json::json!([1, 2, 3]));
    }
}
