//! JavaScript executor - ECMAScript via Boa engine
//!
//! Boa is a JavaScript engine written in Rust that aims for ECMAScript compliance.
//! It supports most ES2022+ features and is actively developed.
//!
//! ## Features
//! - High ECMAScript conformance (~94%)
//! - Full async/await support
//! - Modern JS features (classes, modules, etc.)
//! - Good for web-like scripting
//!
//! ## Limitations
//! - Slower than V8/SpiderMonkey
//! - No DOM APIs (pure JS only)
//! - Some edge cases may differ from browsers

use boa_engine::{
    Context, JsError, JsResult, JsValue, Source, js_string, native_function::NativeFunction,
    object::builtins::JsArray, property::Attribute, value::JsVariant,
};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use super::super::types::{ExecutionLimits, ExecutionRequest, ExecutionResult};
use super::{LanguageExecutor, get_limits, truncate_output};

/// JavaScript code executor using Boa engine
pub struct JavaScriptExecutor {
    _limits: ExecutionLimits,
}

impl JavaScriptExecutor {
    /// Create a new JavaScript executor with default limits
    pub fn new() -> Self {
        Self {
            _limits: ExecutionLimits::default(),
        }
    }

    /// Create a new JavaScript executor with custom limits
    pub fn with_limits(limits: ExecutionLimits) -> Self {
        Self { _limits: limits }
    }

    /// Execute JavaScript code
    pub fn execute_code(&self, request: &ExecutionRequest) -> ExecutionResult {
        let limits = get_limits(request);
        let start = Instant::now();

        // Create Boa context
        let mut context = Context::default();

        // Capture console output
        let output = Rc::new(RefCell::new(Vec::<String>::new()));
        let errors = Rc::new(RefCell::new(Vec::<String>::new()));

        // Setup console object
        if let Err(e) = self.setup_console(&mut context, output.clone(), errors.clone()) {
            return ExecutionResult::error(
                format!("Failed to setup console: {:?}", e),
                start.elapsed().as_millis() as u64,
            );
        }

        // Inject context variables
        if let Some(ctx) = &request.context
            && let Err(e) = self.inject_context(&mut context, ctx)
        {
            return ExecutionResult::error(
                format!("Failed to inject context: {:?}", e),
                start.elapsed().as_millis() as u64,
            );
        }

        // Execute the code
        let result = context.eval(Source::from_bytes(&request.code));
        let timing_ms = start.elapsed().as_millis() as u64;

        // Get captured output
        let stdout = output.borrow().join("\n");
        let stdout = truncate_output(&stdout, limits.max_output_bytes);

        let stderr = errors.borrow().join("\n");

        match result {
            Ok(value) => {
                let result_value = js_to_json(&value, &mut context);
                let mut stdout_with_result = stdout;

                // If there's a non-undefined result, append it to stdout
                if !value.is_undefined() {
                    if !stdout_with_result.is_empty() {
                        stdout_with_result.push('\n');
                    }
                    stdout_with_result.push_str(&format_js_value(&value, &mut context));
                }

                ExecutionResult {
                    success: true,
                    stdout: stdout_with_result,
                    stderr,
                    result: result_value,
                    error: None,
                    timing_ms,
                    memory_used_bytes: None,
                    operations_count: None,
                }
            }
            Err(e) => {
                let error_message = format_js_error(&e, &mut context);
                ExecutionResult {
                    success: false,
                    stdout,
                    stderr: if stderr.is_empty() {
                        error_message.clone()
                    } else {
                        format!("{}\n{}", stderr, error_message)
                    },
                    result: None,
                    error: Some(error_message),
                    timing_ms,
                    memory_used_bytes: None,
                    operations_count: None,
                }
            }
        }
    }

    /// Setup console object with log, error, warn methods
    fn setup_console(
        &self,
        context: &mut Context,
        output: Rc<RefCell<Vec<String>>>,
        errors: Rc<RefCell<Vec<String>>>,
    ) -> JsResult<()> {
        // Create console object
        let console = boa_engine::JsObject::with_null_proto();

        // console.log
        let output_log = output.clone();
        // SAFETY: The closure only captures Rc<RefCell<Vec<String>>> which is safe
        // as long as we don't hold borrows across await points (we don't use async here)
        let log_fn = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let parts: Vec<String> = args.iter().map(|v| format_js_value(v, ctx)).collect();
                let line = parts.join(" ");

                output_log.borrow_mut().push(line);
                Ok(JsValue::undefined())
            })
        };
        console.define_property_or_throw(
            js_string!("log"),
            boa_engine::property::PropertyDescriptor::builder()
                .value(log_fn.to_js_function(context.realm()))
                .writable(true)
                .enumerable(false)
                .configurable(true)
                .build(),
            context,
        )?;

        // console.error
        let errors_err = errors.clone();
        // SAFETY: Same reasoning as console.log
        let error_fn = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let parts: Vec<String> = args.iter().map(|v| format_js_value(v, ctx)).collect();
                let line = parts.join(" ");

                errors_err.borrow_mut().push(line);
                Ok(JsValue::undefined())
            })
        };
        console.define_property_or_throw(
            js_string!("error"),
            boa_engine::property::PropertyDescriptor::builder()
                .value(error_fn.to_js_function(context.realm()))
                .writable(true)
                .enumerable(false)
                .configurable(true)
                .build(),
            context,
        )?;

        // console.warn (alias to log for simplicity)
        let output_warn = output.clone();
        // SAFETY: Same reasoning as console.log
        let warn_fn = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let parts: Vec<String> = args
                    .iter()
                    .map(|v| format!("[WARN] {}", format_js_value(v, ctx)))
                    .collect();
                let line = parts.join(" ");

                output_warn.borrow_mut().push(line);
                Ok(JsValue::undefined())
            })
        };
        console.define_property_or_throw(
            js_string!("warn"),
            boa_engine::property::PropertyDescriptor::builder()
                .value(warn_fn.to_js_function(context.realm()))
                .writable(true)
                .enumerable(false)
                .configurable(true)
                .build(),
            context,
        )?;

        // console.info (alias to log)
        let output_info = output.clone();
        // SAFETY: Same reasoning as console.log
        let info_fn = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let parts: Vec<String> = args.iter().map(|v| format_js_value(v, ctx)).collect();
                let line = parts.join(" ");

                output_info.borrow_mut().push(line);
                Ok(JsValue::undefined())
            })
        };
        console.define_property_or_throw(
            js_string!("info"),
            boa_engine::property::PropertyDescriptor::builder()
                .value(info_fn.to_js_function(context.realm()))
                .writable(true)
                .enumerable(false)
                .configurable(true)
                .build(),
            context,
        )?;

        // Register console globally
        context.register_global_property(
            js_string!("console"),
            console,
            Attribute::WRITABLE | Attribute::CONFIGURABLE,
        )?;

        Ok(())
    }

    /// Inject context variables as global variables
    fn inject_context(&self, context: &mut Context, ctx_value: &serde_json::Value) -> JsResult<()> {
        if let serde_json::Value::Object(map) = ctx_value {
            for (key, value) in map {
                let js_value = json_to_js(value, context)?;
                context.register_global_property(
                    js_string!(key.clone()),
                    js_value,
                    Attribute::WRITABLE | Attribute::CONFIGURABLE,
                )?;
            }
        }
        Ok(())
    }
}

impl Default for JavaScriptExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExecutor for JavaScriptExecutor {
    fn execute(&self, request: &ExecutionRequest) -> ExecutionResult {
        self.execute_code(request)
    }

    fn language_name(&self) -> &'static str {
        "javascript"
    }

    fn language_version(&self) -> String {
        "ES2022+ (Boa 0.21)".to_string()
    }
}

/// Convert JSON to JavaScript value
fn json_to_js(value: &serde_json::Value, context: &mut Context) -> JsResult<JsValue> {
    match value {
        serde_json::Value::Null => Ok(JsValue::null()),
        serde_json::Value::Bool(b) => Ok(JsValue::from(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(JsValue::from(i as i32))
            } else if let Some(f) = n.as_f64() {
                Ok(JsValue::from(f))
            } else {
                Ok(JsValue::undefined())
            }
        }
        serde_json::Value::String(s) => Ok(JsValue::from(js_string!(s.clone()))),
        serde_json::Value::Array(arr) => {
            let js_array = JsArray::new(context);
            for (i, v) in arr.iter().enumerate() {
                let js_value = json_to_js(v, context)?;
                js_array.set(i as u32, js_value, false, context)?;
            }
            Ok(js_array.into())
        }
        serde_json::Value::Object(obj) => {
            let js_obj = boa_engine::JsObject::with_null_proto();
            for (k, v) in obj {
                let js_value = json_to_js(v, context)?;
                js_obj.set(js_string!(k.clone()), js_value, false, context)?;
            }
            Ok(js_obj.into())
        }
    }
}

/// Convert JavaScript value to JSON
fn js_to_json(value: &JsValue, context: &mut Context) -> Option<serde_json::Value> {
    match value.variant() {
        JsVariant::Undefined | JsVariant::Null => None,
        JsVariant::Boolean(b) => Some(serde_json::Value::Bool(b)),
        JsVariant::Integer32(i) => Some(serde_json::Value::Number(serde_json::Number::from(i))),
        JsVariant::Float64(f) => {
            if f.is_nan() || f.is_infinite() {
                Some(serde_json::Value::Null)
            } else {
                serde_json::Number::from_f64(f).map(serde_json::Value::Number)
            }
        }
        JsVariant::String(s) => Some(serde_json::Value::String(s.to_std_string_escaped())),
        JsVariant::BigInt(bi) => Some(serde_json::Value::String(bi.to_string())),
        JsVariant::Object(obj) => {
            // Check if it's an array
            if obj.is_array()
                && let Ok(length) = obj.get(js_string!("length"), context)
                && let Some(len) = length.as_number()
            {
                let mut arr = Vec::new();
                for i in 0..(len as u32) {
                    if let Ok(v) = obj.get(i, context) {
                        arr.push(js_to_json(&v, context).unwrap_or(serde_json::Value::Null));
                    }
                }
                return Some(serde_json::Value::Array(arr));
            }

            // Regular object
            let mut map = serde_json::Map::new();
            if let Ok(keys) = obj.own_property_keys(context) {
                for key in keys {
                    let key_str = key.to_string();
                    if let Ok(v) = obj.get(key, context)
                        && let Some(json_v) = js_to_json(&v, context)
                    {
                        map.insert(key_str, json_v);
                    }
                }
            }
            Some(serde_json::Value::Object(map))
        }
        JsVariant::Symbol(_) => Some(serde_json::Value::String("[Symbol]".to_string())),
    }
}

/// Format JavaScript value for display
fn format_js_value(value: &JsValue, context: &mut Context) -> String {
    match value.variant() {
        JsVariant::Undefined => "undefined".to_string(),
        JsVariant::Null => "null".to_string(),
        JsVariant::Boolean(b) => b.to_string(),
        JsVariant::Integer32(i) => i.to_string(),
        JsVariant::Float64(f) => {
            if f.is_nan() {
                "NaN".to_string()
            } else if f.is_infinite() {
                if f > 0.0 {
                    "Infinity".to_string()
                } else {
                    "-Infinity".to_string()
                }
            } else {
                f.to_string()
            }
        }
        JsVariant::String(s) => s.to_std_string_escaped(),
        JsVariant::BigInt(bi) => format!("{}n", bi),
        JsVariant::Object(obj) => {
            // Check if it's an array
            if obj.is_array()
                && let Ok(length) = obj.get(js_string!("length"), context)
                && let Some(len) = length.as_number()
            {
                let mut parts = Vec::new();
                let max_items = (len as usize).min(10);
                for i in 0..max_items {
                    if let Ok(v) = obj.get(i as u32, context) {
                        parts.push(format_js_value(&v, context));
                    }
                }
                if len as usize > 10 {
                    parts.push("...".to_string());
                }
                return format!("[{}]", parts.join(", "));
            }

            // Check if it's a function
            if obj.is_callable() {
                return "[Function]".to_string();
            }

            // Regular object - simple representation
            let mut parts = Vec::new();
            if let Ok(keys) = obj.own_property_keys(context) {
                for key in keys.iter().take(5) {
                    let key_str = key.to_string();
                    if let Ok(v) = obj.get(key.clone(), context) {
                        parts.push(format!("{}: {}", key_str, format_js_value(&v, context)));
                    }
                }
                if keys.len() > 5 {
                    parts.push("...".to_string());
                }
            }
            format!("{{{}}}", parts.join(", "))
        }
        JsVariant::Symbol(s) => format!(
            "Symbol({})",
            s.description()
                .map(|d| d.to_std_string_escaped())
                .unwrap_or_default()
        ),
    }
}

/// Format JavaScript error for display
fn format_js_error(error: &JsError, context: &mut Context) -> String {
    // Try to get a meaningful error message
    let js_value = error.to_opaque(context);

    if let Some(obj) = js_value.as_object() {
        // Try to get 'message' property
        if let Ok(msg) = obj.get(js_string!("message"), context)
            && let Some(s) = msg.as_string()
        {
            let msg_str = s.to_std_string_escaped();

            // Try to get 'name' property
            if let Ok(name) = obj.get(js_string!("name"), context)
                && let Some(n) = name.as_string()
            {
                return format!("{}: {}", n.to_std_string_escaped(), msg_str);
            }
            return msg_str;
        }
    }

    // Fallback to default formatting
    format!("{}", error)
}

#[cfg(test)]
mod tests {
    use super::super::super::types::Language;
    use super::*;

    fn make_request(code: &str) -> ExecutionRequest {
        ExecutionRequest {
            language: Language::JavaScript,
            code: code.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_simple_expression() {
        let executor = JavaScriptExecutor::new();
        let result = executor.execute(&make_request("1 + 2"));
        assert!(result.success);
        assert!(result.stdout.contains("3"));
    }

    #[test]
    fn test_console_log() {
        let executor = JavaScriptExecutor::new();
        let result = executor.execute(&make_request(r#"console.log("Hello, World!")"#));
        assert!(result.success);
        assert!(result.stdout.contains("Hello, World!"));
    }

    #[test]
    fn test_variables() {
        let executor = JavaScriptExecutor::new();
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
        let executor = JavaScriptExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            let sum = 0;
            for (let i = 0; i < 10; i++) {
                sum += i;
            }
            sum
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("45")); // Sum of 0..9
    }

    #[test]
    fn test_array() {
        let executor = JavaScriptExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            const arr = [1, 2, 3, 4, 5];
            arr.length
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("5"));
    }

    #[test]
    fn test_object() {
        let executor = JavaScriptExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            const obj = { name: "test", value: 42 };
            obj.value
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("42"));
    }

    #[test]
    fn test_function() {
        let executor = JavaScriptExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            function add(a, b) {
                return a + b;
            }
            add(3, 4)
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("7"));
    }

    #[test]
    fn test_arrow_function() {
        let executor = JavaScriptExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            const multiply = (a, b) => a * b;
            multiply(3, 4)
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("12"));
    }

    #[test]
    fn test_syntax_error() {
        let executor = JavaScriptExecutor::new();
        let result = executor.execute(&make_request("let x = "));
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_runtime_error() {
        let executor = JavaScriptExecutor::new();
        let result = executor.execute(&make_request("undefined_variable"));
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_context_injection() {
        let executor = JavaScriptExecutor::new();
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
    fn test_array_methods() {
        let executor = JavaScriptExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            const arr = [1, 2, 3, 4, 5];
            arr.map(x => x * 2).reduce((a, b) => a + b, 0)
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("30")); // 2+4+6+8+10
    }

    #[test]
    fn test_string_methods() {
        let executor = JavaScriptExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            const s = "hello world";
            s.toUpperCase()
            "#,
        ));
        assert!(result.success);
        assert!(result.stdout.contains("HELLO WORLD"));
    }

    #[test]
    fn test_json_operations() {
        let executor = JavaScriptExecutor::new();
        let result = executor.execute(&make_request(
            r#"
            const obj = { a: 1, b: 2 };
            JSON.stringify(obj)
            "#,
        ));
        assert!(result.success);
        // Should contain the JSON string
        assert!(result.stdout.contains("a") && result.stdout.contains("b"));
    }
}
