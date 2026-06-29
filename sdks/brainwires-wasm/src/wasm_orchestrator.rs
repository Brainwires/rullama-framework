//! WASM bindings for the tool orchestrator.
//!
//! This module provides JavaScript-compatible bindings for the Brainwires tool orchestrator,
//! allowing AI models to execute [Rhai](https://rhai.rs/) scripts that call registered
//! JavaScript tool functions from the browser.
//!
//! ## Overview
//!
//! The orchestrator lets you:
//! 1. Register JavaScript callback functions as named tools.
//! 2. Execute Rhai scripts that can call those tools by name.
//! 3. Enforce resource limits (operations, time, memory) for safe sandboxed execution.
//!
//! ## JS Usage
//!
//! ```js
//! import { WasmOrchestrator, ExecutionLimits } from 'brainwires-wasm';
//!
//! const orchestrator = new WasmOrchestrator();
//! orchestrator.register_tool("greet", (inputJson) => {
//!     const input = JSON.parse(inputJson);
//!     return `Hello, ${input}!`;
//! });
//!
//! const limits = new ExecutionLimits();
//! const result = orchestrator.execute('greet("World")', limits);
//! console.log(result); // { success: true, output: "Hello, World!", ... }
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use brainwires_tool_runtime::orchestrator::ExecutionLimits as CoreExecutionLimits;
use brainwires_tool_runtime::orchestrator::dynamic_to_json;
use brainwires_tool_runtime::orchestrator::{
    OrchestratorResult as CoreOrchestratorResult, ToolCall as CoreToolCall,
};

// ============================================================================
// Engine Configuration Constants
// ============================================================================

/// Maximum expression nesting depth enforced by the Rhai engine.
///
/// Prevents stack overflow from deeply nested expressions (e.g., `((((a + b) + c) + d) ...)`).
/// Set to 64 levels, which is generous for typical orchestration scripts.
const MAX_EXPR_DEPTH: usize = 64;

/// Maximum function call nesting depth enforced by the Rhai engine.
///
/// Prevents stack overflow from deep recursion (e.g., `fn f() { f() }`).
/// Set to 64 levels to allow reasonable recursion while preventing runaway stacks.
const MAX_CALL_DEPTH: usize = 64;

// ============================================================================
// WASM-compatible ExecutionLimits wrapper
// ============================================================================

/// Resource limits for safe, sandboxed Rhai script execution in WASM.
///
/// This is a WASM-compatible wrapper around the core `ExecutionLimits` type, exposing
/// getters and setters as `wasm_bindgen` properties so they can be read and written
/// naturally from JavaScript.
///
/// All limits have sensible defaults. Use the [`quick()`](ExecutionLimits::quick) or
/// [`extended()`](ExecutionLimits::extended) constructors for common presets, or create
/// with [`new()`](ExecutionLimits::new) and customize individual properties.
///
/// ## Default values
///
/// | Property          | Default   | Quick    | Extended  |
/// |-------------------|-----------|----------|-----------|
/// | `max_operations`  | 100,000   | 10,000   | 500,000   |
/// | `max_tool_calls`  | 50        | 10       | 100       |
/// | `timeout_ms`      | 30,000    | 5,000    | 120,000   |
/// | `max_string_size` | 10,000,000| 10,000,000| 10,000,000|
/// | `max_array_size`  | 10,000    | 10,000   | 10,000    |
///
/// ## JS Example
///
/// ```js
/// const limits = new ExecutionLimits();
/// limits.max_operations = 50_000;
/// limits.timeout_ms = 10_000;
/// ```
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct ExecutionLimits {
    /// The wrapped core execution limits (not exposed to JS).
    inner: CoreExecutionLimits,
}

#[wasm_bindgen]
impl ExecutionLimits {
    /// Creates a new `ExecutionLimits` with default values.
    ///
    /// Defaults: 100,000 max operations, 50 max tool calls, 30s timeout,
    /// 10MB max string size, 10,000 max array size.
    ///
    /// ```js
    /// const limits = new ExecutionLimits();
    /// ```
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: CoreExecutionLimits::default(),
        }
    }

    /// Creates execution limits tuned for simple, short-running scripts.
    ///
    /// Uses 10,000 max operations, 10 max tool calls, and a 5-second timeout.
    /// Ideal for lightweight validation or single-tool invocations.
    ///
    /// ```js
    /// const limits = ExecutionLimits.quick();
    /// ```
    #[wasm_bindgen]
    #[must_use]
    pub fn quick() -> Self {
        Self {
            inner: CoreExecutionLimits::quick(),
        }
    }

    /// Creates execution limits tuned for complex, long-running orchestration scripts.
    ///
    /// Uses 500,000 max operations, 100 max tool calls, and a 120-second timeout.
    /// Suitable for multi-step agent workflows that invoke many tools.
    ///
    /// ```js
    /// const limits = ExecutionLimits.extended();
    /// ```
    #[wasm_bindgen]
    #[must_use]
    pub fn extended() -> Self {
        Self {
            inner: CoreExecutionLimits::extended(),
        }
    }

    /// The maximum number of Rhai operations (statements, expressions) allowed before
    /// the script is terminated. Prevents infinite loops and runaway computation.
    ///
    /// Default: 100,000. Set to 0 for unlimited (not recommended).
    #[wasm_bindgen(getter)]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn max_operations(&self) -> u64 {
        self.inner.max_operations
    }

    /// Sets the maximum number of Rhai operations allowed.
    #[wasm_bindgen(setter)]
    #[allow(clippy::missing_const_for_fn)]
    pub fn set_max_operations(&mut self, value: u64) {
        self.inner.max_operations = value;
    }

    /// The maximum number of tool calls a script may make. Once exceeded, further
    /// tool calls return an error string instead of invoking the callback.
    ///
    /// Default: 50.
    #[wasm_bindgen(getter)]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn max_tool_calls(&self) -> usize {
        self.inner.max_tool_calls
    }

    /// Sets the maximum number of tool calls allowed per script execution.
    #[wasm_bindgen(setter)]
    #[allow(clippy::missing_const_for_fn)]
    pub fn set_max_tool_calls(&mut self, value: usize) {
        self.inner.max_tool_calls = value;
    }

    /// The wall-clock timeout in milliseconds. If the script runs longer than this,
    /// it is terminated with a timeout error.
    ///
    /// Default: 30,000 (30 seconds).
    #[wasm_bindgen(getter)]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn timeout_ms(&self) -> u64 {
        self.inner.timeout_ms
    }

    /// Sets the wall-clock timeout in milliseconds.
    #[wasm_bindgen(setter)]
    #[allow(clippy::missing_const_for_fn)]
    pub fn set_timeout_ms(&mut self, value: u64) {
        self.inner.timeout_ms = value;
    }

    /// The maximum size (in bytes) of any single string value within the Rhai engine.
    /// Prevents memory exhaustion from unbounded string concatenation.
    ///
    /// Default: 10,000,000 (10 MB).
    #[wasm_bindgen(getter)]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn max_string_size(&self) -> usize {
        self.inner.max_string_size
    }

    /// Sets the maximum string size in bytes.
    #[wasm_bindgen(setter)]
    #[allow(clippy::missing_const_for_fn)]
    pub fn set_max_string_size(&mut self, value: usize) {
        self.inner.max_string_size = value;
    }

    /// The maximum number of elements in any single array within the Rhai engine.
    /// Prevents memory exhaustion from unbounded array growth.
    ///
    /// Default: 10,000.
    #[wasm_bindgen(getter)]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn max_array_size(&self) -> usize {
        self.inner.max_array_size
    }

    /// Sets the maximum array size (number of elements).
    #[wasm_bindgen(setter)]
    #[allow(clippy::missing_const_for_fn)]
    pub fn set_max_array_size(&mut self, value: usize) {
        self.inner.max_array_size = value;
    }
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WASM Orchestrator
// ============================================================================

/// Internal type alias for a reference-counted, interior-mutable JavaScript callback function.
///
/// Each registered tool maps to one of these so it can be shared with the Rhai engine
/// closures during script execution.
type JsToolExecutor = Rc<RefCell<js_sys::Function>>;

/// A WASM-compatible tool orchestrator that executes Rhai scripts with registered
/// JavaScript tool callbacks.
///
/// `WasmOrchestrator` is the main entry point for browser-based tool orchestration.
/// You register JavaScript functions as named tools, then execute Rhai scripts that
/// can call those tools by name. The orchestrator enforces resource limits via
/// [`ExecutionLimits`] to prevent runaway scripts.
///
/// ## Lifecycle
///
/// 1. Create an orchestrator with [`WasmOrchestrator::new()`].
/// 2. Register one or more tools with [`register_tool()`](WasmOrchestrator::register_tool).
/// 3. Execute scripts with [`execute()`](WasmOrchestrator::execute).
///
/// ## JS Example
///
/// ```js
/// const orchestrator = new WasmOrchestrator();
///
/// // Register a tool that reads a file (synchronous callback)
/// orchestrator.register_tool("read_file", (inputJson) => {
///     const path = JSON.parse(inputJson);
///     return localStorage.getItem(path) || "File not found";
/// });
///
/// const limits = ExecutionLimits.quick();
/// const result = orchestrator.execute('read_file("config.json")', limits);
/// console.log(result.output);
/// ```
#[wasm_bindgen]
pub struct WasmOrchestrator {
    /// Map of tool name to JavaScript callback function.
    /// Populated by [`register_tool()`](WasmOrchestrator::register_tool).
    js_executors: HashMap<String, JsToolExecutor>,
}

#[wasm_bindgen]
impl WasmOrchestrator {
    /// Creates a new `WasmOrchestrator` with no registered tools.
    ///
    /// Also installs `console_error_panic_hook` for better panic messages in
    /// the browser console.
    ///
    /// ```js
    /// const orchestrator = new WasmOrchestrator();
    /// ```
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        // Set up panic hook for better error messages
        console_error_panic_hook::set_once();

        Self {
            js_executors: HashMap::new(),
        }
    }

    /// Registers a JavaScript function as a named tool.
    ///
    /// The callback will be invoked when a Rhai script calls a function with the
    /// matching `name`. It receives a single argument: a JSON string containing the
    /// serialized input value from the Rhai call. It must return a string result.
    ///
    /// If a tool with the same name is already registered, it is replaced.
    ///
    /// # Parameters
    ///
    /// - `name` — The tool name, which becomes a callable function in Rhai scripts.
    /// - `callback` — A JavaScript function with signature `(inputJson: string) => string`.
    ///
    /// # JS Example
    ///
    /// ```js
    /// orchestrator.register_tool("add", (inputJson) => {
    ///     const nums = JSON.parse(inputJson);
    ///     return String(nums[0] + nums[1]);
    /// });
    /// ```
    #[wasm_bindgen]
    pub fn register_tool(&mut self, name: &str, callback: js_sys::Function) {
        self.js_executors
            .insert(name.to_string(), Rc::new(RefCell::new(callback)));
    }

    /// Returns the names of all currently registered tools as an array of strings.
    ///
    /// Useful for introspection or displaying available tools to users.
    ///
    /// ```js
    /// const names = orchestrator.registered_tools();
    /// console.log("Available tools:", names); // e.g. ["read_file", "write_file"]
    /// ```
    #[wasm_bindgen]
    #[must_use]
    pub fn registered_tools(&self) -> Vec<String> {
        self.js_executors.keys().cloned().collect()
    }

    /// Executes a Rhai script with the registered tools and returns the result.
    ///
    /// The script can call any registered tool by name as if it were a built-in
    /// Rhai function. Resource limits are enforced throughout execution: if the
    /// script exceeds operations, tool calls, or timeout, it is terminated and
    /// an error result is returned (not thrown).
    ///
    /// # Parameters
    ///
    /// - `script` — A Rhai script string to execute. Tool names registered via
    ///   [`register_tool()`](WasmOrchestrator::register_tool) are available as
    ///   callable functions.
    /// - `limits` — An [`ExecutionLimits`] instance controlling resource bounds.
    ///
    /// # Returns
    ///
    /// A `JsValue` containing an `OrchestratorResult` object with these fields:
    /// - `success` (`boolean`) — Whether the script completed without error.
    /// - `output` (`string`) — The script's return value (or error message).
    /// - `tool_calls` (`Array`) — Log of all tool invocations with inputs, outputs,
    ///   success status, and duration.
    /// - `execution_time_ms` (`number`) — Total wall-clock execution time.
    ///
    /// # Errors
    ///
    /// Returns a `JsValue` error only if the result cannot be serialized to JS.
    /// Script compilation errors and runtime errors are returned as non-success
    /// `OrchestratorResult` values, not thrown exceptions.
    ///
    /// # JS Example
    ///
    /// ```js
    /// const result = orchestrator.execute(`
    ///     let data = read_file("input.json");
    ///     let processed = transform(data);
    ///     write_file("output.json", processed);
    ///     "done"
    /// `, limits);
    ///
    /// if (result.success) {
    ///     console.log("Output:", result.output);
    /// } else {
    ///     console.error("Script failed:", result.output);
    /// }
    /// console.log(`Took ${result.execution_time_ms}ms, ${result.tool_calls.length} tool calls`);
    /// ```
    #[wasm_bindgen]
    #[allow(clippy::too_many_lines)]
    pub fn execute(&self, script: &str, limits: &ExecutionLimits) -> Result<JsValue, JsValue> {
        use web_time::Instant;

        let start_time = Instant::now();
        let tool_calls: Rc<RefCell<Vec<CoreToolCall>>> = Rc::new(RefCell::new(Vec::new()));
        let call_count: Rc<RefCell<usize>> = Rc::new(RefCell::new(0));

        // Create a new Rhai engine with limits
        let mut engine = rhai::Engine::new();

        // Apply resource limits from ExecutionLimits
        engine.set_max_operations(limits.inner.max_operations);
        engine.set_max_string_size(limits.inner.max_string_size);
        engine.set_max_array_size(limits.inner.max_array_size);
        engine.set_max_map_size(limits.inner.max_map_size);
        engine.set_max_expr_depths(MAX_EXPR_DEPTH, MAX_CALL_DEPTH);

        // Set up real-time timeout via on_progress callback
        let timeout_ms = limits.inner.timeout_ms;
        let progress_start = Instant::now();
        engine.on_progress(move |_ops| {
            let elapsed = u64::try_from(progress_start.elapsed().as_millis()).unwrap_or(u64::MAX);
            if elapsed > timeout_ms {
                Some(rhai::Dynamic::from("timeout"))
            } else {
                None
            }
        });

        // Register each JS tool as a Rhai function
        for (name, executor) in &self.js_executors {
            let exec = Rc::clone(executor);
            let calls = Rc::clone(&tool_calls);
            let count = Rc::clone(&call_count);
            let max_calls = limits.inner.max_tool_calls;
            let tool_name = name.clone();

            engine.register_fn(name.as_str(), move |input: rhai::Dynamic| -> String {
                let call_start = Instant::now();

                // Check call limit
                {
                    let mut c = count.borrow_mut();
                    if *c >= max_calls {
                        return format!("ERROR: Maximum tool calls ({max_calls}) exceeded");
                    }
                    *c += 1;
                }

                // Convert Dynamic to JSON
                let json_input = dynamic_to_json(&input);
                let json_str = serde_json::to_string(&json_input).unwrap_or_default();

                // Call the JavaScript function
                let callback = exec.borrow();
                let js_input = JsValue::from_str(&json_str);

                let (output, success) = match callback.call1(&JsValue::NULL, &js_input) {
                    Ok(result) => result.as_string().map_or_else(
                        || ("Tool returned non-string result".to_string(), false),
                        |s| (s, true),
                    ),
                    Err(e) => {
                        let err_msg = e.as_string().map_or_else(
                            || "Tool execution failed".to_string(),
                            |s| format!("Tool error: {s}"),
                        );
                        (err_msg, false)
                    }
                };

                // Record the call
                {
                    let duration_ms =
                        u64::try_from(call_start.elapsed().as_millis()).unwrap_or(u64::MAX);
                    let call = CoreToolCall::new(
                        tool_name.clone(),
                        json_input,
                        output.clone(),
                        success,
                        duration_ms,
                    );
                    calls.borrow_mut().push(call);
                }

                output
            });
        }

        // Compile the script
        let ast = match engine.compile(script) {
            Ok(ast) => ast,
            Err(e) => {
                let result = CoreOrchestratorResult::error(
                    format!("Compilation error: {e}"),
                    tool_calls.borrow().clone(),
                    u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX),
                );
                return serde_wasm_bindgen::to_value(&result)
                    .map_err(|e| JsValue::from_str(&e.to_string()));
            }
        };

        // Execute the script
        let mut scope = rhai::Scope::new();
        let eval_result = engine.eval_ast_with_scope::<rhai::Dynamic>(&mut scope, &ast);

        let execution_time_ms = u64::try_from(start_time.elapsed().as_millis()).unwrap_or(u64::MAX);
        let calls = tool_calls.borrow().clone();

        match eval_result {
            Ok(result) => {
                let output = if result.is_string() {
                    result.into_string().unwrap_or_default()
                } else if result.is_unit() {
                    String::new()
                } else {
                    format!("{result:?}")
                };

                let result = CoreOrchestratorResult::success(output, calls, execution_time_ms);
                serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
            }
            Err(e) => {
                let error_msg = match *e {
                    rhai::EvalAltResult::ErrorTooManyOperations(_) => {
                        format!(
                            "Script exceeded maximum operations ({})",
                            limits.inner.max_operations
                        )
                    }
                    rhai::EvalAltResult::ErrorTerminated(_, _) => {
                        format!(
                            "Script execution timed out after {}ms",
                            limits.inner.timeout_ms
                        )
                    }
                    _ => format!("Execution error: {e}"),
                };

                let result = CoreOrchestratorResult::error(error_msg, calls, execution_time_ms);
                serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
            }
        }
    }
}

impl Default for WasmOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_limits_default() {
        let limits = ExecutionLimits::default();
        assert_eq!(limits.max_operations(), 100_000);
        assert_eq!(limits.max_tool_calls(), 50);
    }

    #[test]
    fn test_execution_limits_quick() {
        let limits = ExecutionLimits::quick();
        assert_eq!(limits.max_operations(), 10_000);
        assert_eq!(limits.max_tool_calls(), 10);
    }

    #[test]
    fn test_wasm_orchestrator_creation() {
        let orchestrator = WasmOrchestrator::new();
        assert!(orchestrator.registered_tools().is_empty());
    }
}
