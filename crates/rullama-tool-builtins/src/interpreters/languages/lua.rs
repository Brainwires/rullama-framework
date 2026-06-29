//! Lua executor - Small, fast scripting language
//!
//! Lua is lightweight and fast, making it ideal for embedded scripting.
//! Uses mlua which supports Lua 5.4 with vendored builds.
//!
//! ## Features
//! - Very small runtime footprint
//! - Fast execution
//! - Good for game scripting and configuration
//! - Memory limit support
//!
//! ## Limitations
//! - Smaller ecosystem than Python/JS
//! - 1-indexed arrays (can be confusing)

use mlua::{Lua, MultiValue, Result as LuaResult, Value};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::super::types::{ExecutionLimits, ExecutionRequest, ExecutionResult};
use super::{LanguageExecutor, get_limits, truncate_output};

/// Lua code executor
pub struct LuaExecutor {
    _limits: ExecutionLimits,
}

impl LuaExecutor {
    /// Create a new Lua executor with default limits
    pub fn new() -> Self {
        Self {
            _limits: ExecutionLimits::default(),
        }
    }

    /// Create a new Lua executor with custom limits
    pub fn with_limits(limits: ExecutionLimits) -> Self {
        Self { _limits: limits }
    }

    /// Execute Lua code
    pub fn execute_code(&self, request: &ExecutionRequest) -> ExecutionResult {
        let limits = get_limits(request);
        let start = Instant::now();

        // Create Lua instance
        let lua = Lua::new();

        // Set memory limit
        let _ = lua.set_memory_limit(limits.max_memory_mb as usize * 1024 * 1024);

        // Capture print output
        let output = Arc::new(Mutex::new(Vec::<String>::new()));

        // Override print function to capture output
        if let Err(e) = self.setup_print(&lua, output.clone()) {
            return ExecutionResult::error(
                format!("Failed to setup print: {}", e),
                start.elapsed().as_millis() as u64,
            );
        }

        // Inject context variables
        if let Some(context) = &request.context
            && let Err(e) = self.inject_context(&lua, context)
        {
            return ExecutionResult::error(
                format!("Failed to inject context: {}", e),
                start.elapsed().as_millis() as u64,
            );
        }

        // Execute the code
        let result = lua.load(&request.code).eval::<Value>();
        let timing_ms = start.elapsed().as_millis() as u64;

        // Get captured output
        let stdout = output.lock().map(|out| out.join("\n")).unwrap_or_default();
        let stdout = truncate_output(&stdout, limits.max_output_bytes);

        // Get memory usage
        let memory_used = lua.used_memory() as u64;

        match result {
            Ok(value) => {
                let result_value = lua_to_json(&value);
                let mut stdout_with_result = stdout;

                // If there's a non-nil result, append it to stdout
                if !matches!(value, Value::Nil) {
                    if !stdout_with_result.is_empty() {
                        stdout_with_result.push('\n');
                    }
                    stdout_with_result.push_str(&format_lua_value(&value));
                }

                ExecutionResult {
                    success: true,
                    stdout: stdout_with_result,
                    stderr: String::new(),
                    result: result_value,
                    error: None,
                    timing_ms,
                    memory_used_bytes: Some(memory_used),
                    operations_count: None,
                }
            }
            Err(e) => {
                let error_message = format_lua_error(&e);
                ExecutionResult {
                    success: false,
                    stdout,
                    stderr: error_message.clone(),
                    result: None,
                    error: Some(error_message),
                    timing_ms,
                    memory_used_bytes: Some(memory_used),
                    operations_count: None,
                }
            }
        }
    }

    /// Setup print function to capture output
    fn setup_print(&self, lua: &Lua, output: Arc<Mutex<Vec<String>>>) -> LuaResult<()> {
        let print_fn = lua.create_function(move |_, args: MultiValue| {
            let parts: Vec<String> = args.into_iter().map(|v| format_lua_value(&v)).collect();
            let line = parts.join("\t");

            if let Ok(mut out) = output.lock() {
                out.push(line);
            }
            Ok(())
        })?;

        lua.globals().set("print", print_fn)?;
        Ok(())
    }

    /// Inject context variables into Lua globals
    fn inject_context(&self, lua: &Lua, context: &serde_json::Value) -> LuaResult<()> {
        if let serde_json::Value::Object(map) = context {
            let globals = lua.globals();
            for (key, value) in map {
                let lua_value = json_to_lua(lua, value)?;
                globals.set(key.as_str(), lua_value)?;
            }
        }
        Ok(())
    }
}

impl Default for LuaExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExecutor for LuaExecutor {
    fn execute(&self, request: &ExecutionRequest) -> ExecutionResult {
        self.execute_code(request)
    }

    fn language_name(&self) -> &'static str {
        "lua"
    }

    fn language_version(&self) -> String {
        "5.4".to_string()
    }
}

/// Convert JSON value to Lua value
fn json_to_lua(lua: &Lua, value: &serde_json::Value) -> LuaResult<Value> {
    match value {
        serde_json::Value::Null => Ok(Value::Nil),
        serde_json::Value::Bool(b) => Ok(Value::Boolean(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::Number(f))
            } else {
                Ok(Value::Nil)
            }
        }
        serde_json::Value::String(s) => {
            let lua_string = lua.create_string(s)?;
            Ok(Value::String(lua_string))
        }
        serde_json::Value::Array(arr) => {
            let table = lua.create_table()?;
            for (i, v) in arr.iter().enumerate() {
                let lua_value = json_to_lua(lua, v)?;
                table.set(i + 1, lua_value)?; // Lua is 1-indexed
            }
            Ok(Value::Table(table))
        }
        serde_json::Value::Object(obj) => {
            let table = lua.create_table()?;
            for (k, v) in obj {
                let lua_value = json_to_lua(lua, v)?;
                table.set(k.as_str(), lua_value)?;
            }
            Ok(Value::Table(table))
        }
    }
}

/// Convert Lua value to JSON
fn lua_to_json(value: &Value) -> Option<serde_json::Value> {
    match value {
        Value::Nil => None,
        Value::Boolean(b) => Some(serde_json::Value::Bool(*b)),
        Value::Integer(i) => Some(serde_json::Value::Number(serde_json::Number::from(*i))),
        Value::Number(f) => serde_json::Number::from_f64(*f).map(serde_json::Value::Number),
        Value::String(s) => s
            .to_str()
            .ok()
            .map(|s| serde_json::Value::String(s.to_string())),
        Value::Table(t) => {
            // Try to determine if it's an array or object
            let mut is_array = true;
            let mut max_index = 0i64;
            let mut has_string_keys = false;

            // Check keys
            if let Ok(pairs) = t
                .clone()
                .pairs::<Value, Value>()
                .collect::<LuaResult<Vec<_>>>()
            {
                for (k, _) in &pairs {
                    match k {
                        Value::Integer(i) if *i > 0 => {
                            max_index = max_index.max(*i);
                        }
                        Value::String(_) => {
                            has_string_keys = true;
                            is_array = false;
                        }
                        _ => {
                            is_array = false;
                        }
                    }
                }

                if is_array && !has_string_keys && max_index > 0 {
                    // It's an array
                    let mut arr = Vec::new();
                    for i in 1..=max_index {
                        if let Ok(v) = t.get::<Value>(i) {
                            arr.push(lua_to_json(&v).unwrap_or(serde_json::Value::Null));
                        }
                    }
                    Some(serde_json::Value::Array(arr))
                } else {
                    // It's an object
                    let mut map = serde_json::Map::new();
                    for (k, v) in pairs {
                        let key = format_lua_value(&k);
                        if let Some(json_v) = lua_to_json(&v) {
                            map.insert(key, json_v);
                        }
                    }
                    Some(serde_json::Value::Object(map))
                }
            } else {
                None
            }
        }
        Value::Function(_) => Some(serde_json::Value::String("[function]".to_string())),
        Value::Thread(_) => Some(serde_json::Value::String("[thread]".to_string())),
        Value::UserData(_) => Some(serde_json::Value::String("[userdata]".to_string())),
        Value::LightUserData(_) => Some(serde_json::Value::String("[lightuserdata]".to_string())),
        Value::Error(e) => Some(serde_json::Value::String(format!("[error: {}]", e))),
        _ => Some(serde_json::Value::String("[unknown]".to_string())),
    }
}

/// Format Lua value for display
fn format_lua_value(value: &Value) -> String {
    match value {
        Value::Nil => "nil".to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Number(f) => f.to_string(),
        Value::String(s) => s
            .to_str()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "[invalid utf8]".to_string()),
        Value::Table(t) => {
            // Simple table representation
            let mut parts = Vec::new();
            if let Ok(pairs) = t
                .clone()
                .pairs::<Value, Value>()
                .collect::<LuaResult<Vec<_>>>()
            {
                for (k, v) in pairs.iter().take(10) {
                    parts.push(format!("{}={}", format_lua_value(k), format_lua_value(v)));
                }
                if pairs.len() > 10 {
                    parts.push("...".to_string());
                }
            }
            format!("{{{}}}", parts.join(", "))
        }
        Value::Function(_) => "[function]".to_string(),
        Value::Thread(_) => "[thread]".to_string(),
        Value::UserData(_) => "[userdata]".to_string(),
        Value::LightUserData(_) => "[lightuserdata]".to_string(),
        Value::Error(e) => format!("[error: {}]", e),
        _ => "[unknown]".to_string(),
    }
}

/// Format Lua error for user display
fn format_lua_error(error: &mlua::Error) -> String {
    match error {
        mlua::Error::SyntaxError { message, .. } => {
            format!("Syntax error: {}", message)
        }
        mlua::Error::RuntimeError(msg) => {
            format!("Runtime error: {}", msg)
        }
        mlua::Error::MemoryError(msg) => {
            format!("Memory error: {}", msg)
        }
        mlua::Error::CallbackError { traceback, cause } => {
            format!("Callback error: {}\nTraceback: {}", cause, traceback)
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
            language: Language::Lua,
            code: code.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_simple_expression() {
        let executor = LuaExecutor::new();
        let result = executor.execute(&make_request("return 1 + 2"));
        assert!(result.success);
        assert!(result.stdout.contains("3"));
    }

    #[test]
    fn test_print() {
        let executor = LuaExecutor::new();
        let result = executor.execute(&make_request(r#"print("Hello, World!")"#));
        assert!(result.success);
        assert!(result.stdout.contains("Hello, World!"));
    }

    #[test]
    fn test_variables() {
        let executor = LuaExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            local x = 10
            local y = 20
            return x + y
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("30"));
    }

    #[test]
    fn test_loop() {
        let executor = LuaExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            local sum = 0
            for i = 0, 9 do
                sum = sum + i
            end
            return sum
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("45")); // Sum of 0..9
    }

    #[test]
    fn test_table() {
        let executor = LuaExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            local t = {1, 2, 3, 4, 5}
            return #t
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("5"));
    }

    #[test]
    fn test_syntax_error() {
        let executor = LuaExecutor::new();
        let result = executor.execute(&make_request("local x = "));
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_context_injection() {
        let executor = LuaExecutor::new();
        let mut request = make_request("return x + y");
        request.context = Some(serde_json::json!({
            "x": 10,
            "y": 20
        }));
        let result = executor.execute(&request);
        assert!(result.success);
        assert!(result.stdout.contains("30"));
    }

    #[test]
    fn test_string_operations() {
        let executor = LuaExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            local s = "hello"
            return string.upper(s)
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("HELLO"));
    }

    #[test]
    fn test_function_definition() {
        let executor = LuaExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            local function add(a, b)
                return a + b
            end
            return add(3, 4)
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("7"));
    }
}
