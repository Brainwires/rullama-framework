//! WASM bindings for code interpreters
//!
//! Provides JavaScript/TypeScript bindings for executing code in various languages
//! directly in the browser via WebAssembly.

use super::{ExecutionLimits, ExecutionRequest, Executor, Language};
use wasm_bindgen::prelude::*;

/// WASM-compatible code executor
#[wasm_bindgen]
pub struct WasmExecutor {
    executor: Executor,
}

#[wasm_bindgen]
impl WasmExecutor {
    /// Create a new executor with default limits
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            executor: Executor::new(),
        }
    }

    /// Create an executor with strict limits for untrusted code
    #[wasm_bindgen]
    pub fn with_strict_limits() -> Self {
        console_error_panic_hook::set_once();
        Self {
            executor: Executor::with_limits(ExecutionLimits::strict()),
        }
    }

    /// Create an executor with relaxed limits for trusted code
    #[wasm_bindgen]
    pub fn with_relaxed_limits() -> Self {
        console_error_panic_hook::set_once();
        Self {
            executor: Executor::with_limits(ExecutionLimits::relaxed()),
        }
    }

    /// Execute code in the specified language
    /// Returns a JsValue containing the ExecutionResult
    #[wasm_bindgen]
    pub fn execute(&self, language: &str, code: &str) -> Result<JsValue, JsValue> {
        let result = self.executor.execute_str(language, code);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Execute code with a full request object (JSON string)
    #[wasm_bindgen]
    pub fn execute_request(&self, request_json: &str) -> Result<JsValue, JsValue> {
        let request: ExecutionRequest = serde_json::from_str(request_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid request: {}", e)))?;

        let result = self.executor.execute(request);
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get list of supported languages
    #[wasm_bindgen]
    pub fn supported_languages(&self) -> Vec<String> {
        self.executor
            .supported_languages()
            .iter()
            .map(|l| l.as_str().to_string())
            .collect()
    }

    /// Check if a language is supported
    #[wasm_bindgen]
    pub fn is_supported(&self, language: &str) -> bool {
        Language::parse(language)
            .map(|l| self.executor.is_supported(l))
            .unwrap_or(false)
    }
}

impl Default for WasmExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute code directly without creating an executor instance
/// Convenience function for simple one-off executions
#[wasm_bindgen]
pub fn execute_code(language: &str, code: &str) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let executor = Executor::new();
    let result = executor.execute_str(language, code);
    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Get list of all supported languages
#[wasm_bindgen]
pub fn get_supported_languages() -> Vec<String> {
    super::supported_languages()
        .iter()
        .map(|l| l.as_str().to_string())
        .collect()
}

// Panic hook for better error messages in WASM
mod console_error_panic_hook {
    use std::sync::Once;
    static SET_HOOK: Once = Once::new();

    pub fn set_once() {
        SET_HOOK.call_once(|| {
            std::panic::set_hook(Box::new(|panic_info| {
                let msg = panic_info.to_string();
                web_sys::console::error_1(&::wasm_bindgen::JsValue::from_str(&msg));
            }));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_executor_creation() {
        let executor = WasmExecutor::new();
        assert!(!executor.supported_languages().is_empty());
    }
}
