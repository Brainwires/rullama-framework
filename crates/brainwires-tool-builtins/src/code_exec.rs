//! Code Execution Tool - Sandboxed code execution via embedded interpreters
//!
//! Provides a unified `execute_code` tool that supports multiple languages:
//!
//! ## Default (Native Interpreters via brainwires-tools interpreters module crate)
//! - **Rhai**: Lightweight Rust scripting (always available)
//! - **Lua**: Lua 5.4 via mlua (always available)
//! - **JavaScript**: ES2022+ via Boa engine (with feature flag)
//!
//! Requires the `interpreters` feature flag.

use crate::interpreters::{ExecutionLimits, ExecutionRequest, Executor, Language};
use anyhow::Result;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;

use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult};

/// Default execution timeout in milliseconds
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// Maximum allowed timeout in milliseconds
const MAX_TIMEOUT_MS: u64 = 60_000;

/// Code execution tool with native interpreters
pub struct CodeExecTool;

impl CodeExecTool {
    /// Get all code execution tool definitions
    pub fn get_tools() -> Vec<Tool> {
        vec![Self::execute_code_tool()]
    }

    /// Execute code tool definition
    fn execute_code_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "language".to_string(),
            json!({
                "type": "string",
                "description": Self::language_description()
            }),
        );
        properties.insert(
            "code".to_string(),
            json!({
                "type": "string",
                "description": "Source code to execute"
            }),
        );
        properties.insert(
            "timeout_ms".to_string(),
            json!({
                "type": "integer",
                "description": "Execution timeout in milliseconds (default: 10000, max: 60000)",
                "default": 10000
            }),
        );
        properties.insert(
            "context".to_string(),
            json!({
                "type": "object",
                "description": "Context variables to inject into the script (as global variables)",
                "default": {}
            }),
        );

        Tool {
            name: "execute_code".to_string(),
            description: Self::tool_description(),
            input_schema: ToolInputSchema::object(
                properties,
                vec!["language".to_string(), "code".to_string()],
            ),
            requires_approval: true,
            defer_loading: true,
            ..Default::default()
        }
    }

    /// Generate language description based on available features
    fn language_description() -> String {
        let langs = ["'rhai'", "'lua'", "'javascript'"];
        format!(
            "Programming language identifier: {}. Native interpreters run in-process.",
            langs.join(", ")
        )
    }

    /// Generate tool description
    fn tool_description() -> String {
        String::from(
            r#"Execute code in a sandboxed environment.

Native interpreters (no Docker required):
- rhai: Lightweight Rust scripting
- lua: Lua 5.4
- javascript: ES2022+ via Boa engine

Examples:
- Rhai: language="rhai", code="let x = 1 + 2; x"
- Lua: language="lua", code="return 1 + 2"
- Use 'context' parameter to inject variables into scripts."#,
        )
    }

    /// Execute a code execution tool
    pub async fn execute(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        _context: &ToolContext,
    ) -> ToolResult {
        let result = match tool_name {
            "execute_code" => Self::execute_code(input).await,
            _ => Err(anyhow::anyhow!(
                "Unknown code execution tool: {}",
                tool_name
            )),
        };

        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("Code execution failed: {}", e),
            ),
        }
    }

    /// Execute code implementation - routes to appropriate backend
    async fn execute_code(input: &Value) -> Result<String> {
        #[derive(Deserialize)]
        struct ExecuteCodeInput {
            language: String,
            code: String,
            #[serde(default = "default_timeout")]
            timeout_ms: u64,
            #[serde(default)]
            context: Option<serde_json::Value>,
        }

        fn default_timeout() -> u64 {
            DEFAULT_TIMEOUT_MS
        }

        let params: ExecuteCodeInput = serde_json::from_value(input.clone())?;
        let timeout_ms = params.timeout_ms.min(MAX_TIMEOUT_MS);

        let language_lower = params.language.to_lowercase();
        if let Some(lang) = Self::parse_native_language(&language_lower) {
            return Self::execute_native(lang, &params.code, timeout_ms, params.context.as_ref());
        }

        let supported = Self::supported_languages();
        Err(anyhow::anyhow!(
            "Language '{}' is not supported. Supported languages: {}.",
            params.language,
            supported.join(", ")
        ))
    }

    /// Parse language string to native Language enum
    fn parse_native_language(lang: &str) -> Option<Language> {
        match lang {
            "rhai" => Some(Language::Rhai),
            "lua" => Some(Language::Lua),
            "javascript" | "js" => Some(Language::JavaScript),
            _ => None,
        }
    }

    /// Get list of supported native languages
    fn supported_languages() -> Vec<&'static str> {
        vec!["rhai", "lua", "javascript"]
    }

    /// Execute code using native interpreter
    fn execute_native(
        language: Language,
        code: &str,
        timeout_ms: u64,
        context: Option<&serde_json::Value>,
    ) -> Result<String> {
        let limits = ExecutionLimits {
            max_timeout_ms: timeout_ms,
            max_memory_mb: 256,
            max_output_bytes: 1_048_576,
            max_operations: 100_000,
            max_call_depth: 64,
            max_string_length: 1_000_000,
            max_array_length: 10_000,
            max_map_size: 10_000,
        };

        let executor = Executor::with_limits(limits.clone());

        let request = ExecutionRequest {
            language,
            code: code.to_string(),
            stdin: None,
            timeout_ms,
            memory_limit_mb: 256,
            context: context.cloned(),
            limits: Some(limits),
        };

        let result = executor.execute(request);

        let lang_name = match language {
            Language::Rhai => "rhai",
            Language::Lua => "lua",
            Language::JavaScript => "javascript",
        };

        let mut output = format!(
            "Language: {} (native)\nSuccess: {}\nDuration: {}ms\n",
            lang_name, result.success, result.timing_ms
        );

        if let Some(ops) = result.operations_count {
            output.push_str(&format!("Operations: {}\n", ops));
        }

        output.push_str("\n--- stdout ---\n");
        if result.stdout.is_empty() {
            output.push_str("(empty)\n");
        } else {
            output.push_str(&result.stdout);
            if !result.stdout.ends_with('\n') {
                output.push('\n');
            }
        }

        if !result.stderr.is_empty() {
            output.push_str("\n--- stderr ---\n");
            output.push_str(&result.stderr);
            if !result.stderr.ends_with('\n') {
                output.push('\n');
            }
        }

        if let Some(json_result) = &result.result {
            output.push_str(&format!("\n--- result ---\n{}\n", json_result));
        }

        if let Some(error) = &result.error {
            output.push_str(&format!("\n--- error ---\n{}\n", error));
        }

        Ok(output)
    }

    /// Execute code (for orchestrator) - simplified interface
    pub async fn execute_code_helper(language: &str, code: &str) -> Result<String, String> {
        let input = json!({
            "language": language,
            "code": code
        });
        Self::execute_code(&input).await.map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tools() {
        let tools = CodeExecTool::get_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "execute_code");
        assert!(tools[0].requires_approval);
        assert!(tools[0].defer_loading);
    }

    #[test]
    fn test_execute_code_tool_definition() {
        let tool = CodeExecTool::execute_code_tool();
        assert_eq!(tool.name, "execute_code");
        assert!(tool.description.contains("rhai"));
        assert!(tool.description.contains("lua"));
    }

    #[test]
    fn test_parse_native_language() {
        assert_eq!(
            CodeExecTool::parse_native_language("rhai"),
            Some(Language::Rhai)
        );
        assert_eq!(
            CodeExecTool::parse_native_language("lua"),
            Some(Language::Lua)
        );
        assert_eq!(CodeExecTool::parse_native_language("RHAI"), None);
    }

    #[test]
    fn test_supported_languages() {
        let langs = CodeExecTool::supported_languages();
        assert!(langs.contains(&"rhai"));
        assert!(langs.contains(&"lua"));
    }

    #[test]
    fn test_execute_native_rhai() {
        let result = CodeExecTool::execute_native(Language::Rhai, "let x = 1 + 2; x", 10000, None);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("Language: rhai (native)"));
        assert!(output.contains("Success: true"));
    }

    #[test]
    fn test_execute_native_lua() {
        let result = CodeExecTool::execute_native(Language::Lua, "return 1 + 2", 10000, None);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("Language: lua (native)"));
        assert!(output.contains("Success: true"));
    }

    #[test]
    fn test_execute_native_with_context() {
        let context = json!({
            "x": 10,
            "y": 20
        });
        let result = CodeExecTool::execute_native(Language::Rhai, "x + y", 10000, Some(&context));
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("30"));
    }

    #[tokio::test]
    async fn test_execute_unknown_tool() {
        let context = ToolContext::default();
        let result = CodeExecTool::execute("test-id", "unknown_tool", &json!({}), &context).await;
        assert!(result.is_error);
        assert!(result.content.contains("Unknown code execution tool"));
    }

    #[tokio::test]
    async fn test_execute_code_routes_to_rhai() {
        let context = ToolContext::default();
        let input = json!({
            "language": "rhai",
            "code": "42"
        });

        let result = CodeExecTool::execute("test-id", "execute_code", &input, &context).await;
        assert!(!result.is_error);
        assert!(result.content.contains("Language: rhai"));
    }

    #[tokio::test]
    async fn test_execute_unsupported_language() {
        let context = ToolContext::default();
        let input = json!({
            "language": "cobol",
            "code": "DISPLAY 'HELLO'"
        });

        let result = CodeExecTool::execute("test-id", "execute_code", &input, &context).await;
        assert!(result.is_error);
    }
}
