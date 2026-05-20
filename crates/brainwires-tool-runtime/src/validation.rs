//! Validation tools for agents to verify their work

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

use brainwires_core::{Tool, ToolInputSchema, ToolResult};

const BUILD_TIMEOUT: Duration = Duration::from_secs(600);
const SYNTAX_CHECK_TIMEOUT: Duration = Duration::from_secs(30);

fn validate_file_path(file_path: &str) -> Result<PathBuf> {
    if file_path.is_empty() {
        return Err(anyhow!("File path cannot be empty"));
    }
    if file_path.contains('\0') {
        return Err(anyhow!("File path contains null byte"));
    }
    let path = PathBuf::from(file_path);
    let canonical = path
        .canonicalize()
        .with_context(|| format!("Failed to resolve path: {}", file_path))?;
    if !canonical.exists() {
        return Err(anyhow!("File does not exist: {}", file_path));
    }
    if !canonical.is_file() {
        return Err(anyhow!("Path is not a file: {}", file_path));
    }
    Ok(canonical)
}

fn validate_directory_path(dir_path: &str) -> Result<PathBuf> {
    if dir_path.is_empty() {
        return Err(anyhow!("Directory path cannot be empty"));
    }
    if dir_path.contains('\0') {
        return Err(anyhow!("Directory path contains null byte"));
    }
    let path = PathBuf::from(dir_path);
    let canonical = path
        .canonicalize()
        .with_context(|| format!("Failed to resolve directory: {}", dir_path))?;
    if !canonical.exists() {
        return Err(anyhow!("Directory does not exist: {}", dir_path));
    }
    if !canonical.is_dir() {
        return Err(anyhow!("Path is not a directory: {}", dir_path));
    }
    Ok(canonical)
}

/// Check for duplicate exports/constants in a file
#[tracing::instrument(name = "tool.validate.duplicates")]
pub async fn check_duplicates(file_path: &str) -> Result<ToolResult> {
    let validated_path = validate_file_path(file_path)?;
    let content = tokio::fs::read_to_string(&validated_path)
        .await
        .with_context(|| format!("Failed to read file: {}", file_path))?;
    let lines: Vec<&str> = content.lines().collect();
    let mut exports = HashMap::new();
    let mut duplicates = Vec::new();
    for (line_num, line) in lines.iter().enumerate() {
        if is_export_line(line)
            && let Some(name) = extract_export_name(line)
        {
            if let Some(first) = exports.get(&name) {
                duplicates.push(json!({"name": name, "first_line": first, "duplicate_line": line_num + 1, "code": line.trim()}));
            } else {
                exports.insert(name, line_num + 1);
            }
        }
    }
    let result = json!({"file": validated_path.display().to_string(), "has_duplicates": !duplicates.is_empty(), "duplicate_count": duplicates.len(), "duplicates": duplicates, "total_exports": exports.len()});
    Ok(ToolResult {
        tool_use_id: String::new(),
        content: serde_json::to_string_pretty(&result)?,
        is_error: false,
    })
}

fn is_export_line(line: &str) -> bool {
    let t = line.trim();
    t.starts_with("export const ")
        || t.starts_with("export let ")
        || t.starts_with("export var ")
        || t.starts_with("export function ")
        || t.starts_with("export async function ")
        || t.starts_with("export class ")
        || t.starts_with("export interface ")
        || t.starts_with("export type ")
        || t.starts_with("export enum ")
        || t.starts_with("export namespace ")
        || t.starts_with("export default class ")
        || t.starts_with("export default function ")
        || t.starts_with("export default async function ")
}

fn extract_export_name(line: &str) -> Option<String> {
    let t = line.trim();
    for prefix in &["export const ", "export let ", "export var "] {
        if let Some(after) = t.strip_prefix(prefix) {
            return after
                .split_whitespace()
                .next()
                .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '$'))
                .map(String::from);
        }
    }
    if let Some(after) = t.strip_prefix("export async function ") {
        return after.split('(').next().map(|s| s.trim().to_string());
    }
    if let Some(after) = t.strip_prefix("export function ") {
        return after.split('(').next().map(|s| s.trim().to_string());
    }
    if let Some(after) = t.strip_prefix("export default async function ") {
        let name = after.split('(').next().map(|s| s.trim().to_string())?;
        return Some(if name.is_empty() {
            "default".to_string()
        } else {
            name
        });
    }
    if let Some(after) = t.strip_prefix("export default function ") {
        let name = after.split('(').next().map(|s| s.trim().to_string())?;
        return Some(if name.is_empty() {
            "default".to_string()
        } else {
            name
        });
    }
    if let Some(after) = t.strip_prefix("export default class ") {
        let name = after.split_whitespace().next().map(|s| s.to_string())?;
        return Some(if name.is_empty() || name == "{" {
            "default".to_string()
        } else {
            name
        });
    }
    if let Some(after) = t.strip_prefix("export class ") {
        return after.split_whitespace().next().map(|s| s.to_string());
    }
    if let Some(after) = t.strip_prefix("export interface ") {
        return after.split_whitespace().next().map(|s| s.to_string());
    }
    if let Some(after) = t.strip_prefix("export type ") {
        return after
            .split(|c: char| c.is_whitespace() || c == '=' || c == '<')
            .next()
            .map(|s| s.trim().to_string());
    }
    if let Some(after) = t.strip_prefix("export enum ") {
        return after.split_whitespace().next().map(|s| s.to_string());
    }
    if let Some(after) = t.strip_prefix("export namespace ") {
        return after.split_whitespace().next().map(|s| s.to_string());
    }
    None
}

/// Verify build by running the appropriate build command
#[tracing::instrument(name = "tool.validate.build")]
pub async fn verify_build(working_directory: &str, build_type: &str) -> Result<ToolResult> {
    let validated_dir = validate_directory_path(working_directory)?;
    let (command, args) = match build_type {
        "npm" => ("npm", vec!["run", "build"]),
        "yarn" => ("yarn", vec!["build"]),
        "pnpm" => ("pnpm", vec!["build"]),
        "bun" => ("bun", vec!["run", "build"]),
        "cargo" => ("cargo", vec!["build"]),
        "typescript" => ("npx", vec!["tsc", "--noEmit"]),
        "go" => ("go", vec!["build", "./..."]),
        "python" => ("python", vec!["-m", "py_compile"]),
        "gradle" => ("gradle", vec!["build"]),
        "maven" => ("mvn", vec!["compile"]),
        "make" => ("make", vec![]),
        _ => {
            return Ok(ToolResult {
                tool_use_id: String::new(),
                content: format!("Unknown build type: {}", build_type),
                is_error: true,
            });
        }
    };

    let output_result = timeout(
        BUILD_TIMEOUT,
        Command::new(command)
            .args(&args)
            .current_dir(&validated_dir)
            .output(),
    )
    .await;
    let output = match output_result {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            return Ok(ToolResult {
                tool_use_id: String::new(),
                content: json!({"success": false, "error": format!("Failed to execute: {}", e)})
                    .to_string(),
                is_error: true,
            });
        }
        Err(_) => {
            return Ok(ToolResult {
                tool_use_id: String::new(),
                content: json!({"success": false, "error": "Build timed out", "timed_out": true})
                    .to_string(),
                is_error: true,
            });
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let success = output.status.success();
    let errors = parse_build_errors(&stderr, &stdout, build_type);
    let result = json!({"success": success, "exit_code": output.status.code(), "error_count": errors.len(), "errors": errors, "working_directory": validated_dir.display().to_string()});
    Ok(ToolResult {
        tool_use_id: String::new(),
        content: serde_json::to_string_pretty(&result)?,
        is_error: !success,
    })
}

fn parse_build_errors(stderr: &str, stdout: &str, build_type: &str) -> Vec<Value> {
    let mut errors = Vec::new();
    let mut seen = HashSet::new();
    let combined = format!("{}\n{}", stderr, stdout);
    for line in combined.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_lowercase();
        if lower.starts_with("warning:")
            || lower.starts_with("note:")
            || lower.starts_with("help:")
            || lower.starts_with("-->")
        {
            continue;
        }
        let error = match build_type {
            "typescript" | "npm" | "yarn" | "pnpm" | "bun" => {
                parse_typescript_error(line).or_else(|| parse_javascript_error(line))
            }
            "cargo" => parse_rust_error(line),
            "go" => parse_go_error(line),
            "python" => parse_python_error(line),
            "gradle" | "maven" => parse_java_error(line),
            _ => None,
        };
        if let Some(mut error) = error {
            let key = error["message"].as_str().unwrap_or("").to_string();
            if !seen.contains(&key) {
                seen.insert(key);
                error["build_type"] = json!(build_type);
                errors.push(error);
            }
            continue;
        }
        if lower.contains("error") && !lower.contains("0 error") {
            let key = trimmed.to_string();
            if !seen.contains(&key) {
                seen.insert(key);
                errors
                    .push(json!({"message": trimmed, "type": "generic", "build_type": build_type}));
            }
        }
    }
    errors.truncate(25);
    errors
}

fn parse_typescript_error(line: &str) -> Option<Value> {
    let parts: Vec<&str> = line.splitn(2, " - error ").collect();
    if parts.len() == 2 {
        Some(json!({"location": parts[0].trim(), "message": parts[1].trim(), "type": "typescript"}))
    } else {
        None
    }
}

fn parse_rust_error(line: &str) -> Option<Value> {
    if line.contains("error[E") || line.trim().starts_with("error:") {
        Some(json!({"message": line.trim(), "type": "rust", "severity": "error"}))
    } else {
        None
    }
}

fn parse_javascript_error(line: &str) -> Option<Value> {
    let t = line.trim();
    if t.contains("Error:")
        && (t.contains("SyntaxError")
            || t.contains("ReferenceError")
            || t.contains("TypeError")
            || t.contains("RangeError"))
    {
        Some(json!({"message": t, "type": "javascript", "severity": "error"}))
    } else {
        None
    }
}

fn parse_go_error(line: &str) -> Option<Value> {
    let t = line.trim();
    if t.contains(".go:") && t.contains(": ") {
        let parts: Vec<&str> = t.splitn(2, ": ").collect();
        if parts.len() == 2 {
            return Some(
                json!({"location": parts[0].trim(), "message": parts[1].trim(), "type": "go"}),
            );
        }
    }
    if t.starts_with("can't load package:") || t.starts_with("package") {
        return Some(json!({"message": t, "type": "go", "severity": "error"}));
    }
    None
}

fn parse_python_error(line: &str) -> Option<Value> {
    let t = line.trim();
    if t.starts_with("File \"") && t.contains("line ") {
        return Some(json!({"location": t, "type": "python"}));
    }
    if (t.ends_with("Error:") || t.contains("Error: "))
        && (t.contains("SyntaxError")
            || t.contains("IndentationError")
            || t.contains("NameError")
            || t.contains("ImportError")
            || t.contains("ModuleNotFoundError"))
    {
        return Some(json!({"message": t, "type": "python", "severity": "error"}));
    }
    None
}

fn parse_java_error(line: &str) -> Option<Value> {
    let t = line.trim();
    if t.contains(".java:") && t.contains("error:") {
        let parts: Vec<&str> = t.splitn(2, "error:").collect();
        if parts.len() == 2 {
            return Some(
                json!({"location": parts[0].trim(), "message": parts[1].trim(), "type": "java"}),
            );
        }
    }
    if t.starts_with("[ERROR]") {
        return Some(json!({"message": t.trim_start_matches("[ERROR]").trim(), "type": "java"}));
    }
    if t.contains("COMPILATION ERROR") || t.contains("BUILD FAILURE") {
        return Some(json!({"message": t, "type": "java"}));
    }
    None
}

/// Check syntax without full build
pub async fn check_syntax(file_path: &str) -> Result<ToolResult> {
    let validated_path = validate_file_path(file_path)?;
    let extension = validated_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    if matches!(extension, "ts" | "tsx") {
        let content = tokio::fs::read_to_string(&validated_path)
            .await
            .with_context(|| format!("Failed to read: {}", file_path))?;
        let mut errors = Vec::new();
        if content.contains("export export") {
            errors.push(json!({"message": "Duplicate 'export' keyword", "type": "syntax_error"}));
        }
        if content.contains("import import") {
            errors.push(json!({"message": "Duplicate 'import' keyword", "type": "syntax_error"}));
        }
        let open = content.matches('{').count();
        let close = content.matches('}').count();
        if open != close {
            errors.push(json!({"message": format!("Unmatched braces: {} open, {} close", open, close), "type": "syntax_error"}));
        }
        if !errors.is_empty() {
            return Ok(ToolResult { tool_use_id: String::new(), content: json!({"file": validated_path.display().to_string(), "valid_syntax": false, "errors": errors}).to_string(), is_error: true });
        }
        return Ok(ToolResult { tool_use_id: String::new(), content: json!({"file": validated_path.display().to_string(), "valid_syntax": true, "skipped": true}).to_string(), is_error: false });
    }

    let file_path_str = validated_path.display().to_string();
    let (command, args) = match extension {
        "js" | "jsx" => (
            "npx",
            vec![
                "eslint",
                "--no-eslintrc",
                "--parser",
                "@babel/eslint-parser",
                &file_path_str,
            ],
        ),
        "rs" => (
            "rustc",
            vec![
                "--crate-type",
                "lib",
                "--error-format",
                "json",
                &file_path_str,
            ],
        ),
        "py" => ("python", vec!["-m", "py_compile", &file_path_str]),
        "go" => ("gofmt", vec!["-e", &file_path_str]),
        _ => {
            return Ok(ToolResult {
                tool_use_id: String::new(),
                content: format!("Unsupported file type: {}", extension),
                is_error: true,
            });
        }
    };

    let working_dir = validated_path.parent().unwrap_or_else(|| Path::new("."));
    let output_result = timeout(
        SYNTAX_CHECK_TIMEOUT,
        Command::new(command)
            .args(&args)
            .current_dir(working_dir)
            .output(),
    )
    .await;
    let output = match output_result {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            return Ok(ToolResult {
                tool_use_id: String::new(),
                content:
                    json!({"file": file_path, "valid_syntax": false, "error": format!("{}", e)})
                        .to_string(),
                is_error: true,
            });
        }
        Err(_) => {
            return Ok(ToolResult {
                tool_use_id: String::new(),
                content: json!({"file": file_path, "valid_syntax": false, "timed_out": true})
                    .to_string(),
                is_error: true,
            });
        }
    };

    let success = output.status.success();
    let result = json!({"file": validated_path.display().to_string(), "valid_syntax": success});
    Ok(ToolResult {
        tool_use_id: String::new(),
        content: serde_json::to_string_pretty(&result)?,
        is_error: !success,
    })
}

/// Validation tool dispatcher
pub struct ValidationTool;

impl ValidationTool {
    /// Return validation tool definitions.
    pub fn get_tools() -> Vec<Tool> {
        get_validation_tools()
    }

    /// Execute a validation tool by name
    pub async fn execute(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        _context: &brainwires_core::ToolContext,
    ) -> ToolResult {
        let result = match tool_name {
            "check_duplicates" => {
                let file_path = input["file_path"].as_str().unwrap_or("");
                check_duplicates(file_path).await
            }
            "verify_build" => {
                let dir = input["working_directory"].as_str().unwrap_or(".");
                let build_type = input["build_type"].as_str().unwrap_or("cargo");
                verify_build(dir, build_type).await
            }
            "check_syntax" => {
                let file_path = input["file_path"].as_str().unwrap_or("");
                check_syntax(file_path).await
            }
            _ => Err(anyhow!("Unknown validation tool: {}", tool_name)),
        };

        match result {
            Ok(tool_result) => tool_result,
            Err(e) => {
                ToolResult::error(tool_use_id.to_string(), format!("Validation failed: {}", e))
            }
        }
    }
}

/// Get validation tool definitions
pub fn get_validation_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "check_duplicates".to_string(),
            description: "Check a file for duplicate exports, constants, or function definitions.".to_string(),
            input_schema: ToolInputSchema::object({ let mut p = HashMap::new(); p.insert("file_path".to_string(), json!({"type": "string", "description": "Path to file"})); p }, vec!["file_path".to_string()]),
            requires_approval: false, ..Default::default()
        },
        Tool {
            name: "verify_build".to_string(),
            description: "Run a build command and verify it succeeds. Supports: npm, yarn, pnpm, bun, cargo, typescript, go, python, gradle, maven, make.".to_string(),
            input_schema: ToolInputSchema::object({ let mut p = HashMap::new(); p.insert("working_directory".to_string(), json!({"type": "string"})); p.insert("build_type".to_string(), json!({"type": "string", "enum": ["npm", "yarn", "pnpm", "bun", "cargo", "typescript", "go", "python", "gradle", "maven", "make"]})); p }, vec!["working_directory".to_string(), "build_type".to_string()]),
            requires_approval: false, ..Default::default()
        },
        Tool {
            name: "check_syntax".to_string(),
            description: "Check syntax of a single file without running a full build.".to_string(),
            input_schema: ToolInputSchema::object({ let mut p = HashMap::new(); p.insert("file_path".to_string(), json!({"type": "string"})); p }, vec!["file_path".to_string()]),
            requires_approval: false, ..Default::default()
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_extract_export_name() {
        assert_eq!(
            extract_export_name("export const FOO = 'bar'"),
            Some("FOO".to_string())
        );
        assert_eq!(
            extract_export_name("export function myFunc() {"),
            Some("myFunc".to_string())
        );
        assert_eq!(
            extract_export_name("export interface MyInterface {"),
            Some("MyInterface".to_string())
        );
        assert_eq!(
            extract_export_name("export type MyType = string"),
            Some("MyType".to_string())
        );
    }

    #[test]
    fn test_is_export_line() {
        assert!(is_export_line("export const FOO = 'bar'"));
        assert!(is_export_line("export interface MyInterface {"));
        assert!(!is_export_line("const FOO = 'bar'"));
    }

    #[test]
    fn test_parse_rust_error() {
        let result = parse_rust_error("error[E0425]: cannot find value `foo`");
        assert!(result.is_some());
    }

    #[tokio::test]
    async fn test_check_duplicates() {
        let mut temp = NamedTempFile::new().unwrap();
        writeln!(temp, "export const FOO = 'bar'").unwrap();
        writeln!(temp, "export const BAZ = 'qux'").unwrap();
        writeln!(temp, "export const FOO = 'dup'").unwrap();
        temp.flush().unwrap();
        let result = check_duplicates(temp.path().to_str().unwrap())
            .await
            .unwrap();
        let parsed: Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["has_duplicates"], true);
    }
}
