//! Rhai executor - Native Rust scripting language
//!
//! Rhai is the fastest option as it's a pure Rust scripting engine.
//! It has excellent integration with Rust types and functions.
//!
//! ## Features
//! - Native Rust execution (no interpreter overhead)
//! - Built-in safety limits (max operations, memory)
//! - Good for configuration scripts and simple logic
//!
//! ## Limitations
//! - Smaller standard library than Python/JS
//! - Less familiar syntax for most users

use rhai::{Dynamic, Engine, EvalAltResult, Scope};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::super::types::{ExecutionLimits, ExecutionRequest, ExecutionResult};
use super::{LanguageExecutor, get_limits, truncate_output};

/// Rhai code executor
pub struct RhaiExecutor {
    _limits: ExecutionLimits,
}

impl RhaiExecutor {
    /// Create a new Rhai executor with default limits
    pub fn new() -> Self {
        Self {
            _limits: ExecutionLimits::default(),
        }
    }

    /// Create a new Rhai executor with custom limits
    pub fn with_limits(limits: ExecutionLimits) -> Self {
        Self { _limits: limits }
    }

    /// Configure a Rhai engine with safety limits
    fn configure_engine(&self, limits: &ExecutionLimits) -> Engine {
        let mut engine = Engine::new();

        // Apply safety limits
        engine.set_max_operations(limits.max_operations);
        engine.set_max_string_size(limits.max_string_length);
        engine.set_max_array_size(limits.max_array_length);
        engine.set_max_map_size(limits.max_map_size);
        engine.set_max_expr_depths(
            limits.max_call_depth as usize,
            limits.max_call_depth as usize,
        );

        // Disable potentially dangerous features for sandboxing
        engine.set_allow_looping(true); // Allow loops (controlled by max_operations)
        engine.set_strict_variables(true); // Require variable declaration

        engine
    }

    /// Inject context variables into the scope
    fn inject_context(&self, scope: &mut Scope, context: &serde_json::Value) {
        if let serde_json::Value::Object(map) = context {
            for (key, value) in map {
                let dynamic_value = json_to_dynamic(value);
                scope.push(key.clone(), dynamic_value);
            }
        }
    }

    /// Execute Rhai code
    pub fn execute_code(&self, request: &ExecutionRequest) -> ExecutionResult {
        let limits = get_limits(request);
        let engine = self.configure_engine(&limits);

        let start = Instant::now();

        // Create scope and inject context
        let mut scope = Scope::new();
        if let Some(context) = &request.context {
            self.inject_context(&mut scope, context);
        }

        // Capture print output
        let output = Arc::new(Mutex::new(Vec::<String>::new()));
        let output_clone = output.clone();

        // Capture print and debug output via engine callbacks
        let mut engine = engine;
        engine.on_print(move |s| {
            if let Ok(mut out) = output_clone.lock() {
                out.push(s.to_string());
            }
        });

        let output_clone2 = output.clone();
        engine.on_debug(move |s, _src, _pos| {
            if let Ok(mut out) = output_clone2.lock() {
                out.push(format!("[DEBUG] {}", s));
            }
        });

        // Execute the script
        let result: Result<Dynamic, Box<EvalAltResult>> =
            engine.eval_with_scope(&mut scope, &request.code);
        let timing_ms = start.elapsed().as_millis() as u64;

        // Get captured output
        let stdout = output.lock().map(|out| out.join("\n")).unwrap_or_default();
        let stdout = truncate_output(&stdout, limits.max_output_bytes);

        match result {
            Ok(value) => {
                let result_value = dynamic_to_json(&value);
                let mut stdout_with_result = stdout;

                // If there's a non-unit result, append it to stdout
                if !value.is_unit() {
                    if !stdout_with_result.is_empty() {
                        stdout_with_result.push('\n');
                    }
                    stdout_with_result.push_str(&format!("{}", value));
                }

                ExecutionResult {
                    success: true,
                    stdout: stdout_with_result,
                    stderr: String::new(),
                    result: result_value,
                    error: None,
                    timing_ms,
                    memory_used_bytes: None,
                    operations_count: None,
                }
            }
            Err(e) => {
                let error_message = format_rhai_error(&e);
                ExecutionResult {
                    success: false,
                    stdout,
                    stderr: error_message.clone(),
                    result: None,
                    error: Some(error_message),
                    timing_ms,
                    memory_used_bytes: None,
                    operations_count: None,
                }
            }
        }
    }
}

impl Default for RhaiExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExecutor for RhaiExecutor {
    fn execute(&self, request: &ExecutionRequest) -> ExecutionResult {
        self.execute_code(request)
    }

    fn language_name(&self) -> &'static str {
        "rhai"
    }

    fn language_version(&self) -> String {
        // Rhai version is determined at compile time
        "1.20".to_string()
    }
}

/// Convert JSON value to Rhai Dynamic
fn json_to_dynamic(value: &serde_json::Value) -> Dynamic {
    match value {
        serde_json::Value::Null => Dynamic::UNIT,
        serde_json::Value::Bool(b) => Dynamic::from(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Dynamic::from(i)
            } else if let Some(f) = n.as_f64() {
                Dynamic::from(f)
            } else {
                Dynamic::UNIT
            }
        }
        serde_json::Value::String(s) => Dynamic::from(s.clone()),
        serde_json::Value::Array(arr) => {
            let vec: Vec<Dynamic> = arr.iter().map(json_to_dynamic).collect();
            Dynamic::from(vec)
        }
        serde_json::Value::Object(obj) => {
            let mut map = rhai::Map::new();
            for (k, v) in obj {
                map.insert(k.clone().into(), json_to_dynamic(v));
            }
            Dynamic::from(map)
        }
    }
}

/// Convert Rhai Dynamic to JSON value
fn dynamic_to_json(value: &Dynamic) -> Option<serde_json::Value> {
    if value.is_unit() {
        return None;
    }

    if value.is_bool() {
        return Some(serde_json::Value::Bool(value.as_bool().unwrap_or(false)));
    }

    if value.is_int() {
        return Some(serde_json::Value::Number(serde_json::Number::from(
            value.as_int().unwrap_or(0),
        )));
    }

    if value.is_float() {
        if let Ok(f) = value.as_float()
            && let Some(n) = serde_json::Number::from_f64(f)
        {
            return Some(serde_json::Value::Number(n));
        }
        return None;
    }

    if value.is_string() {
        return Some(serde_json::Value::String(
            value.clone().into_string().unwrap_or_default(),
        ));
    }

    if value.is_array()
        && let Ok(arr) = value.clone().into_array()
    {
        let json_arr: Vec<serde_json::Value> = arr
            .into_iter()
            .filter_map(|v| dynamic_to_json(&v))
            .collect();
        return Some(serde_json::Value::Array(json_arr));
    }

    if value.is_map()
        && let Some(map) = value.clone().try_cast::<rhai::Map>()
    {
        let mut json_map = serde_json::Map::new();
        for (k, v) in map {
            if let Some(json_v) = dynamic_to_json(&v) {
                json_map.insert(k.to_string(), json_v);
            }
        }
        return Some(serde_json::Value::Object(json_map));
    }

    // Default: convert to string representation
    Some(serde_json::Value::String(format!("{}", value)))
}

/// Format Rhai error for user display
fn format_rhai_error(error: &EvalAltResult) -> String {
    match error {
        EvalAltResult::ErrorTooManyOperations(_) => {
            "Operation limit exceeded - possible infinite loop".to_string()
        }
        EvalAltResult::ErrorDataTooLarge(msg, _) => {
            format!("Data too large: {}", msg)
        }
        EvalAltResult::ErrorStackOverflow(_) => {
            "Stack overflow - too many nested calls".to_string()
        }
        EvalAltResult::ErrorParsing(parse_error, _) => {
            format!("Syntax error: {}", parse_error)
        }
        EvalAltResult::ErrorVariableNotFound(name, _) => {
            format!("Variable not found: {}", name)
        }
        EvalAltResult::ErrorFunctionNotFound(name, _) => {
            format!("Function not found: {}", name)
        }
        _ => format!("{}", error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreters::Language;

    fn make_request(code: &str) -> ExecutionRequest {
        ExecutionRequest {
            language: Language::Rhai,
            code: code.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_simple_expression() {
        let executor = RhaiExecutor::new();
        let result = executor.execute(&make_request("1 + 2"));
        assert!(result.success);
        assert!(result.stdout.contains("3"));
    }

    #[test]
    fn test_string_expression() {
        let executor = RhaiExecutor::new();
        let result = executor.execute(&make_request(r#""Hello, World!""#));
        assert!(result.success);
        assert!(result.stdout.contains("Hello, World!"));
    }

    #[test]
    fn test_variable_declaration() {
        let executor = RhaiExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            let x = 10;
            let y = 20;
            x + y
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("30"));
    }

    #[test]
    fn test_loop() {
        let executor = RhaiExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            let sum = 0;
            for i in 0..10 {
                sum += i;
            }
            sum
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("45")); // Sum of 0..9
    }

    #[test]
    fn test_syntax_error() {
        let executor = RhaiExecutor::new();
        let result = executor.execute(&make_request("let x = "));
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_undefined_variable() {
        let executor = RhaiExecutor::new();
        let result = executor.execute(&make_request("undefined_var"));
        assert!(!result.success);
        let err = result.error.unwrap();
        assert!(
            err.contains("not found") || err.contains("Undefined"),
            "Unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_context_injection() {
        let executor = RhaiExecutor::new();
        let mut request = make_request("x + y");
        request.context = Some(serde_json::json!({
            "x": 10,
            "y": 20
        }));
        let result = executor.execute(&request);
        assert!(result.success);
        assert!(result.stdout.contains("30"));
    }

    #[test]
    fn test_array_operations() {
        let executor = RhaiExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            let arr = [1, 2, 3, 4, 5];
            arr.len()
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("5"));
    }

    #[test]
    fn test_map_operations() {
        let executor = RhaiExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            let map = #{
                name: "test",
                value: 42
            };
            map.value
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("42"));
    }
}
