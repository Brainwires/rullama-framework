use anyhow::{Context, Result};
use diffy::{Patch, apply};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use brainwires_core::{StagedWrite, Tool, ToolContext, ToolInputSchema, ToolResult};

/// File operations tool implementation
pub struct FileOpsTool;

impl FileOpsTool {
    /// Get all file operation tool definitions
    pub fn get_tools() -> Vec<Tool> {
        vec![
            Self::read_file_tool(),
            Self::write_file_tool(),
            Self::edit_file_tool(),
            Self::patch_file_tool(),
            Self::list_directory_tool(),
            Self::search_files_tool(),
            Self::delete_file_tool(),
            Self::create_directory_tool(),
        ]
    }

    fn read_file_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            json!({"type": "string", "description": "Path to the file to read (relative or absolute)"}),
        );
        properties.insert(
            "offset".to_string(),
            json!({
                "type": "number",
                "description": "Line number to start reading from (1-based, default 1)",
                "default": 1
            }),
        );
        properties.insert(
            "limit".to_string(),
            json!({
                "type": "number",
                "description": "Maximum lines to read (default 2000). Output truncation marker is appended if the file is larger.",
                "default": 2000
            }),
        );
        Tool {
            name: "read_file".to_string(),
            description: "Read the contents of a local file. Defaults to the first 2000 lines; use offset+limit for paged reads of large files.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["path".to_string()]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn write_file_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            json!({"type": "string", "description": "Path to the file to write"}),
        );
        properties.insert(
            "content".to_string(),
            json!({"type": "string", "description": "Content to write to the file"}),
        );
        Tool {
            name: "write_file".to_string(),
            description: "Create or overwrite a file with the given content.".to_string(),
            input_schema: ToolInputSchema::object(
                properties,
                vec!["path".to_string(), "content".to_string()],
            ),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    fn edit_file_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            json!({"type": "string", "description": "Path to the file to edit"}),
        );
        properties.insert(
            "old_text".to_string(),
            json!({"type": "string", "description": "Exact text to find in the file"}),
        );
        properties.insert(
            "new_text".to_string(),
            json!({"type": "string", "description": "Text to replace old_text with"}),
        );
        Tool {
            name: "edit_file".to_string(),
            description: "Replace the first occurrence of old_text with new_text in a file."
                .to_string(),
            input_schema: ToolInputSchema::object(
                properties,
                vec![
                    "path".to_string(),
                    "old_text".to_string(),
                    "new_text".to_string(),
                ],
            ),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    fn patch_file_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            json!({"type": "string", "description": "Path to the file to patch"}),
        );
        properties.insert(
            "patch".to_string(),
            json!({"type": "string", "description": "Unified diff patch to apply"}),
        );
        Tool {
            name: "patch_file".to_string(),
            description: "Apply a unified diff patch to a file.".to_string(),
            input_schema: ToolInputSchema::object(
                properties,
                vec!["path".to_string(), "patch".to_string()],
            ),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    fn list_directory_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            json!({"type": "string", "description": "Path to the directory to list"}),
        );
        properties.insert("recursive".to_string(), json!({"type": "boolean", "description": "Whether to list recursively", "default": false}));
        Tool {
            name: "list_directory".to_string(),
            description: "List files and directories in a local path.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["path".to_string()]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn search_files_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            json!({"type": "string", "description": "Directory to search in"}),
        );
        properties.insert(
            "pattern".to_string(),
            json!({"type": "string", "description": "File name pattern to match (glob pattern)"}),
        );
        Tool {
            name: "search_files".to_string(),
            description: "Search for files matching a glob pattern.".to_string(),
            input_schema: ToolInputSchema::object(
                properties,
                vec!["path".to_string(), "pattern".to_string()],
            ),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn delete_file_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            json!({"type": "string", "description": "Path to the file or directory to delete"}),
        );
        Tool {
            name: "delete_file".to_string(),
            description: "Delete a file or directory.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["path".to_string()]),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    fn create_directory_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "path".to_string(),
            json!({"type": "string", "description": "Path to the directory to create"}),
        );
        Tool {
            name: "create_directory".to_string(),
            description: "Create a new directory (including parent directories).".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["path".to_string()]),
            requires_approval: true,
            serialize: true,
            ..Default::default()
        }
    }

    /// Execute a file operation tool
    #[tracing::instrument(name = "tool.execute", skip(input, context), fields(tool_name))]
    pub fn execute(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        context: &ToolContext,
    ) -> ToolResult {
        let result = match tool_name {
            "read_file" => Self::read_file(input, context),
            "write_file" => Self::write_file(input, context),
            "edit_file" => Self::edit_file(input, context),
            "patch_file" => Self::patch_file(input, context),
            "list_directory" => Self::list_directory(input, context),
            "search_files" => Self::search_files(input, context),
            "delete_file" => Self::delete_file(input, context),
            "create_directory" => Self::create_directory(input, context),
            _ => Err(anyhow::anyhow!(
                "Unknown file operation tool: {}",
                tool_name
            )),
        };
        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("File operation failed: {}", e),
            ),
        }
    }

    fn read_file(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            path: String,
            #[serde(default = "default_read_offset")]
            offset: u32,
            #[serde(default = "default_read_limit")]
            limit: u32,
        }
        fn default_read_offset() -> u32 {
            1
        }
        fn default_read_limit() -> u32 {
            2000
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let full_path = Self::resolve_path(&params.path, context)?;
        let content = fs::read_to_string(&full_path)
            .with_context(|| format!("Failed to read file: {}", full_path.display()))?;
        let total_bytes = content.len();
        let total_lines = content.lines().count();

        let start = params.offset.saturating_sub(1) as usize;
        let limit = params.limit.max(1) as usize;
        let sliced: String = content
            .lines()
            .skip(start)
            .take(limit)
            .collect::<Vec<_>>()
            .join("\n");

        let end = (start + limit).min(total_lines);
        let truncated = end < total_lines;
        let header = if truncated {
            format!(
                "File: {}\nSize: {} bytes, {} lines total\nShowing lines {}-{} of {} (... truncated: call again with offset={} to continue)\n\n",
                full_path.display(),
                total_bytes,
                total_lines,
                start + 1,
                end,
                total_lines,
                end + 1,
            )
        } else {
            format!(
                "File: {}\nSize: {} bytes, {} lines total\nShowing lines {}-{}\n\n",
                full_path.display(),
                total_bytes,
                total_lines,
                start + 1,
                end.max(start + 1),
            )
        };
        Ok(format!("{}{}", header, sliced))
    }

    fn write_file(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            path: String,
            content: String,
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let full_path = Self::resolve_path(&params.path, context)?;

        // 1. Idempotency check — return cached result on exact retry
        let content_hash = Sha256::digest(params.content.as_bytes());
        let key = Self::derive_idempotency_key("write_file", &full_path, &content_hash);
        if let Some(ref registry) = context.idempotency_registry
            && let Some(record) = registry.get(&key)
        {
            tracing::debug!(path = %full_path.display(), "write_file: idempotent retry, returning cached result");
            return Ok(record.cached_result);
        }

        // 2. Staging check — stage write for two-phase commit when backend present
        if let Some(ref backend) = context.staging_backend {
            let staged = StagedWrite {
                key,
                target_path: full_path.clone(),
                content: params.content.clone(),
            };
            backend.stage(staged);
            return Ok(format!(
                "Staged write of {} bytes to {} (pending commit)",
                params.content.len(),
                full_path.display()
            ));
        }

        // 3. Direct write
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create parent directory: {}", parent.display())
            })?;
        }
        fs::write(&full_path, &params.content)
            .with_context(|| format!("Failed to write file: {}", full_path.display()))?;

        // 4. Read-back verification — detects concurrent clobber.
        //
        // The in-process file-lock manager serializes individual write syscalls,
        // but it does NOT prevent two agents from independently overwriting each
        // other's content: from the lock manager's perspective, those are valid
        // sequential writes. Without this check, the losing writer would report
        // `Success: true` while its content is silently gone.
        //
        // We re-read the file and compare bytes. A mismatch means another
        // process modified the file between our write and our read-back, so we
        // surface a tool-level error that the agent's LLM can see and handle
        // (retry with a unique filename, coordinate, or abort — its choice).
        let readback = fs::read(&full_path)
            .with_context(|| format!("post-write readback failed for {}", full_path.display()))?;
        if readback.as_slice() != params.content.as_bytes() {
            return Err(anyhow::anyhow!(
                "Write to {} succeeded but immediate read-back returned {} bytes (wrote {} bytes). \
                 This indicates concurrent modification by another process. \
                 Use a unique filename or coordinate with the other writer.",
                full_path.display(),
                readback.len(),
                params.content.len()
            ));
        }

        // 5. Record intended content hash for post-validation clobber detection.
        //
        // The read-back above proves our bytes were on disk at time T; it
        // cannot prove they remain on disk when the agent later finalises
        // `Success: true` at some T' > T.  We record the hash so the agent's
        // validation loop can re-read at finalisation and detect a clobber.
        //
        // No-op when no registry is attached (CLI-driven invocations).
        let content_hash_bytes: [u8; 32] = content_hash.into();
        context.record_write(full_path.clone(), content_hash_bytes);

        let msg = format!(
            "Successfully wrote {} bytes to {}",
            params.content.len(),
            full_path.display()
        );
        if let Some(ref registry) = context.idempotency_registry {
            registry.record(
                Self::derive_idempotency_key("write_file", &full_path, &content_hash),
                msg.clone(),
            );
        }
        Ok(msg)
    }

    fn edit_file(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            path: String,
            old_text: String,
            new_text: String,
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let full_path = Self::resolve_path(&params.path, context)?;

        // Idempotency key = tool + path + sha256(old_text '\0' new_text)
        let mut hasher = Sha256::new();
        hasher.update(params.old_text.as_bytes());
        hasher.update(b"\0");
        hasher.update(params.new_text.as_bytes());
        let content_hash = hasher.finalize();
        let key = Self::derive_idempotency_key("edit_file", &full_path, &content_hash);

        // 1. Idempotency check
        if let Some(ref registry) = context.idempotency_registry
            && let Some(record) = registry.get(&key)
        {
            tracing::debug!(path = %full_path.display(), "edit_file: idempotent retry, returning cached result");
            return Ok(record.cached_result);
        }

        // Compute new content (needed for both staging and direct write)
        let current = fs::read_to_string(&full_path)
            .with_context(|| format!("Failed to read file: {}", full_path.display()))?;
        if !current.contains(&params.old_text) {
            return Err(anyhow::anyhow!(
                "Text not found in file: '{}'",
                params.old_text
            ));
        }
        let new_content = current.replacen(&params.old_text, &params.new_text, 1);

        // 2. Staging check — stage the fully-computed new content
        if let Some(ref backend) = context.staging_backend {
            backend.stage(StagedWrite {
                key,
                target_path: full_path.clone(),
                content: new_content,
            });
            return Ok(format!(
                "Staged edit (1 replacement) in {} (pending commit)",
                full_path.display()
            ));
        }

        // 3. Direct write
        fs::write(&full_path, &new_content)
            .with_context(|| format!("Failed to write file: {}", full_path.display()))?;
        let msg = format!(
            "Successfully replaced 1 occurrence(s) in {}",
            full_path.display()
        );
        if let Some(ref registry) = context.idempotency_registry {
            registry.record(
                Self::derive_idempotency_key("edit_file", &full_path, &content_hash),
                msg.clone(),
            );
        }
        Ok(msg)
    }

    fn patch_file(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            path: String,
            patch: String,
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let full_path = Self::resolve_path(&params.path, context)?;

        // Idempotency key = tool + path + sha256(patch)
        let patch_hash = Sha256::digest(params.patch.as_bytes());
        let key = Self::derive_idempotency_key("patch_file", &full_path, &patch_hash);

        // 1. Idempotency check
        if let Some(ref registry) = context.idempotency_registry
            && let Some(record) = registry.get(&key)
        {
            tracing::debug!(path = %full_path.display(), "patch_file: idempotent retry, returning cached result");
            return Ok(record.cached_result);
        }

        // Compute patched content (needed for both staging and direct write)
        let content = fs::read_to_string(&full_path)
            .with_context(|| format!("Failed to read file: {}", full_path.display()))?;
        let patch: Patch<'_, str> = Patch::from_str(&params.patch)
            .map_err(|e| anyhow::anyhow!("Failed to parse patch: {}", e))?;
        let hunk_count = patch.hunks().len();
        let new_content =
            apply(&content, &patch).map_err(|e| anyhow::anyhow!("Failed to apply patch: {}", e))?;

        // 2. Staging check — stage the fully-patched content
        if let Some(ref backend) = context.staging_backend {
            backend.stage(StagedWrite {
                key,
                target_path: full_path.clone(),
                content: new_content.to_string(),
            });
            return Ok(format!(
                "Staged patch of {} hunk(s) to {} (pending commit)",
                hunk_count,
                full_path.display()
            ));
        }

        // 3. Direct write
        fs::write(&full_path, new_content.as_str())
            .with_context(|| format!("Failed to write file: {}", full_path.display()))?;
        let msg = format!(
            "Successfully applied patch with {} hunk(s) to {}",
            hunk_count,
            full_path.display()
        );
        if let Some(ref registry) = context.idempotency_registry {
            registry.record(
                Self::derive_idempotency_key("patch_file", &full_path, &patch_hash),
                msg.clone(),
            );
        }
        Ok(msg)
    }

    fn list_directory(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            path: String,
            #[serde(default)]
            recursive: bool,
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let full_path = Self::resolve_path(&params.path, context)?;
        if !full_path.is_dir() {
            return Err(anyhow::anyhow!("Not a directory: {}", full_path.display()));
        }

        let mut entries = Vec::new();
        if params.recursive {
            for entry in WalkDir::new(&full_path).min_depth(1) {
                let entry = entry?;
                let path = entry.path();
                let relative = path.strip_prefix(&full_path).unwrap_or(path);
                let type_str = if path.is_dir() { "dir" } else { "file" };
                entries.push(format!("{} - {}", type_str, relative.display()));
            }
        } else {
            for entry in fs::read_dir(&full_path)? {
                let entry = entry?;
                let path = entry.path();
                let name = entry.file_name();
                let type_str = if path.is_dir() { "dir" } else { "file" };
                entries.push(format!("{} - {}", type_str, name.to_string_lossy()));
            }
        }
        entries.sort();
        Ok(format!(
            "Directory: {}\nEntries: {}\n\n{}",
            full_path.display(),
            entries.len(),
            entries.join("\n")
        ))
    }

    fn search_files(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            path: String,
            pattern: String,
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let full_path = Self::resolve_path(&params.path, context)?;
        let glob_pattern = full_path.join(&params.pattern);
        let pattern_str = glob_pattern.to_string_lossy().to_string();
        let mut matches = Vec::new();
        for entry in glob::glob(&pattern_str)? {
            match entry {
                Ok(path) => {
                    let relative = path.strip_prefix(&full_path).unwrap_or(&path);
                    matches.push(relative.display().to_string());
                }
                Err(e) => tracing::warn!("Error reading glob entry: {}", e),
            }
        }
        matches.sort();
        Ok(format!(
            "Search pattern: {}\nMatches: {}\n\n{}",
            params.pattern,
            matches.len(),
            matches.join("\n")
        ))
    }

    fn delete_file(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            path: String,
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let full_path = Self::resolve_path(&params.path, context)?;

        // Idempotency: key = tool + path (no content factor; deleting same path twice is safe to deduplicate)
        if let Some(ref registry) = context.idempotency_registry {
            let key = Self::derive_idempotency_key("delete_file", &full_path, b"");
            if let Some(record) = registry.get(&key) {
                tracing::debug!(path = %full_path.display(), "delete_file: idempotent retry, returning cached result");
                return Ok(record.cached_result);
            }
            let msg = if full_path.is_dir() {
                fs::remove_dir_all(&full_path).with_context(|| {
                    format!("Failed to delete directory: {}", full_path.display())
                })?;
                format!("Successfully deleted directory: {}", full_path.display())
            } else {
                fs::remove_file(&full_path)
                    .with_context(|| format!("Failed to delete file: {}", full_path.display()))?;
                format!("Successfully deleted file: {}", full_path.display())
            };
            registry.record(key, msg.clone());
            return Ok(msg);
        }

        if full_path.is_dir() {
            fs::remove_dir_all(&full_path)
                .with_context(|| format!("Failed to delete directory: {}", full_path.display()))?;
            Ok(format!(
                "Successfully deleted directory: {}",
                full_path.display()
            ))
        } else {
            fs::remove_file(&full_path)
                .with_context(|| format!("Failed to delete file: {}", full_path.display()))?;
            Ok(format!(
                "Successfully deleted file: {}",
                full_path.display()
            ))
        }
    }

    fn create_directory(input: &Value, context: &ToolContext) -> Result<String> {
        #[derive(Deserialize)]
        struct Input {
            path: String,
        }
        let params: Input = serde_json::from_value(input.clone())?;
        let full_path = Self::resolve_path(&params.path, context)?;

        // Idempotency: key = tool + path
        if let Some(ref registry) = context.idempotency_registry {
            let key = Self::derive_idempotency_key("create_directory", &full_path, b"");
            if let Some(record) = registry.get(&key) {
                tracing::debug!(path = %full_path.display(), "create_directory: idempotent retry, returning cached result");
                return Ok(record.cached_result);
            }
            fs::create_dir_all(&full_path)
                .with_context(|| format!("Failed to create directory: {}", full_path.display()))?;
            let msg = format!("Successfully created directory: {}", full_path.display());
            registry.record(key, msg.clone());
            return Ok(msg);
        }

        fs::create_dir_all(&full_path)
            .with_context(|| format!("Failed to create directory: {}", full_path.display()))?;
        Ok(format!(
            "Successfully created directory: {}",
            full_path.display()
        ))
    }

    /// Resolve a path relative to the working directory
    pub fn resolve_path(path: &str, context: &ToolContext) -> Result<PathBuf> {
        let path = Path::new(path);
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            Path::new(&context.working_directory).join(path)
        };
        Ok(resolved.canonicalize().unwrap_or(resolved))
    }

    /// Derive an idempotency key for a mutating operation.
    ///
    /// The key is a hex-encoded SHA-256 hash of
    /// `tool_name '\0' canonical_path '\0' content_factor`.
    ///
    /// `content_factor` encodes the operation payload so that:
    /// - retries with identical content reuse the cached result
    /// - genuinely different writes to the same path produce a new key
    fn derive_idempotency_key(tool_name: &str, path: &Path, content_factor: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(tool_name.as_bytes());
        hasher.update(b"\0");
        hasher.update(path.to_string_lossy().as_bytes());
        hasher.update(b"\0");
        hasher.update(content_factor);
        hex::encode(hasher.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_context(working_dir: &str) -> ToolContext {
        ToolContext {
            working_directory: working_dir.to_string(),
            ..Default::default()
        }
    }

    fn create_test_context_with_registry(working_dir: &str) -> ToolContext {
        ToolContext {
            working_directory: working_dir.to_string(),
            idempotency_registry: Some(brainwires_core::IdempotencyRegistry::new()),
            ..Default::default()
        }
    }

    #[test]
    fn test_get_tools() {
        let tools = FileOpsTool::get_tools();
        assert_eq!(tools.len(), 8);
        let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_file"));
        assert!(names.contains(&"edit_file"));
        assert!(names.contains(&"patch_file"));
    }

    #[test]
    fn test_read_file() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "Hello, World!").unwrap();
        let context = create_test_context(temp_dir.path().to_str().unwrap());
        let input = json!({"path": "test.txt"});
        let result = FileOpsTool::execute("1", "read_file", &input, &context);
        assert!(!result.is_error);
        assert!(result.content.contains("Hello, World!"));
    }

    #[test]
    fn test_read_file_truncates_large_file_and_emits_marker() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("big.txt");
        let body = (1..=3000)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&test_file, &body).unwrap();
        let context = create_test_context(temp_dir.path().to_str().unwrap());
        // Default limit is 2000 lines
        let input = json!({"path": "big.txt"});
        let result = FileOpsTool::execute("1", "read_file", &input, &context);
        assert!(!result.is_error);
        assert!(result.content.contains("truncated"));
        assert!(result.content.contains("line 1\n"));
        assert!(result.content.contains("line 2000"));
        assert!(!result.content.contains("line 2001"));
    }

    #[test]
    fn test_read_file_respects_offset_and_limit() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("paged.txt");
        let body = (1..=100)
            .map(|i| format!("row {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&test_file, &body).unwrap();
        let context = create_test_context(temp_dir.path().to_str().unwrap());
        // Read lines 10..=14 (offset=10, limit=5)
        let input = json!({"path": "paged.txt", "offset": 10, "limit": 5});
        let result = FileOpsTool::execute("1", "read_file", &input, &context);
        assert!(!result.is_error);
        assert!(result.content.contains("row 10"));
        assert!(result.content.contains("row 14"));
        assert!(!result.content.contains("row 15"));
        assert!(!result.content.contains("row 9\n"));
    }

    #[test]
    fn test_write_file() {
        let temp_dir = TempDir::new().unwrap();
        let context = create_test_context(temp_dir.path().to_str().unwrap());
        let input = json!({"path": "new.txt", "content": "Test"});
        let result = FileOpsTool::execute("2", "write_file", &input, &context);
        assert!(!result.is_error);
        assert!(temp_dir.path().join("new.txt").exists());
    }

    #[test]
    fn test_edit_file() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(
            temp_dir.path().join("edit.txt"),
            "Hello World! Hello World!",
        )
        .unwrap();
        let context = create_test_context(temp_dir.path().to_str().unwrap());
        let input = json!({"path": "edit.txt", "old_text": "World", "new_text": "Rust"});
        let result = FileOpsTool::execute("3", "edit_file", &input, &context);
        assert!(!result.is_error);
        let content = fs::read_to_string(temp_dir.path().join("edit.txt")).unwrap();
        assert_eq!(content, "Hello Rust! Hello World!");
    }

    #[test]
    fn test_list_directory() {
        let temp_dir = TempDir::new().unwrap();
        fs::write(temp_dir.path().join("a.txt"), "").unwrap();
        fs::write(temp_dir.path().join("b.txt"), "").unwrap();
        let context = create_test_context(temp_dir.path().to_str().unwrap());
        let input = json!({"path": ".", "recursive": false});
        let result = FileOpsTool::execute("4", "list_directory", &input, &context);
        assert!(!result.is_error);
        assert!(result.content.contains("a.txt"));
        assert!(result.content.contains("b.txt"));
    }

    #[test]
    fn test_delete_file() {
        let temp_dir = TempDir::new().unwrap();
        let file = temp_dir.path().join("del.txt");
        fs::write(&file, "").unwrap();
        let context = create_test_context(temp_dir.path().to_str().unwrap());
        let input = json!({"path": "del.txt"});
        let result = FileOpsTool::execute("5", "delete_file", &input, &context);
        assert!(!result.is_error);
        assert!(!file.exists());
    }

    // ── Idempotency tests ─────────────────────────────────────────────────────

    #[test]
    fn test_write_file_idempotent_same_content() {
        let temp_dir = TempDir::new().unwrap();
        let ctx = create_test_context_with_registry(temp_dir.path().to_str().unwrap());
        let input = json!({"path": "idem.txt", "content": "Hello"});

        let r1 = FileOpsTool::execute("1", "write_file", &input, &ctx);
        assert!(!r1.is_error);
        assert!(temp_dir.path().join("idem.txt").exists());

        // Overwrite the file on disk to simulate a crash-then-retry scenario
        fs::write(temp_dir.path().join("idem.txt"), "CORRUPTED").unwrap();

        // Retry with identical inputs → cached result returned, file NOT re-written
        let r2 = FileOpsTool::execute("2", "write_file", &input, &ctx);
        assert!(!r2.is_error);
        let on_disk = fs::read_to_string(temp_dir.path().join("idem.txt")).unwrap();
        assert_eq!(
            on_disk, "CORRUPTED",
            "Idempotent retry must not overwrite the file"
        );
    }

    #[test]
    fn test_write_file_different_content_not_idempotent() {
        let temp_dir = TempDir::new().unwrap();
        let ctx = create_test_context_with_registry(temp_dir.path().to_str().unwrap());

        FileOpsTool::execute(
            "1",
            "write_file",
            &json!({"path": "f.txt", "content": "v1"}),
            &ctx,
        );
        FileOpsTool::execute(
            "2",
            "write_file",
            &json!({"path": "f.txt", "content": "v2"}),
            &ctx,
        );

        let on_disk = fs::read_to_string(temp_dir.path().join("f.txt")).unwrap();
        assert_eq!(on_disk, "v2", "Different content must produce a new write");
    }

    #[test]
    fn test_write_file_no_registry_always_writes() {
        let temp_dir = TempDir::new().unwrap();
        let ctx = create_test_context(temp_dir.path().to_str().unwrap()); // no registry
        let input = json!({"path": "f.txt", "content": "v1"});

        FileOpsTool::execute("1", "write_file", &input, &ctx);
        fs::write(temp_dir.path().join("f.txt"), "v_corrupted").unwrap();
        FileOpsTool::execute("2", "write_file", &input, &ctx);

        let on_disk = fs::read_to_string(temp_dir.path().join("f.txt")).unwrap();
        assert_eq!(on_disk, "v1", "Without registry every call must go through");
    }

    #[test]
    fn test_delete_file_idempotent() {
        let temp_dir = TempDir::new().unwrap();
        let ctx = create_test_context_with_registry(temp_dir.path().to_str().unwrap());
        let file = temp_dir.path().join("del.txt");
        fs::write(&file, "").unwrap();

        let r1 = FileOpsTool::execute("1", "delete_file", &json!({"path": "del.txt"}), &ctx);
        assert!(!r1.is_error);
        assert!(!file.exists());

        // File is gone; second call must return cached result without error
        let r2 = FileOpsTool::execute("2", "delete_file", &json!({"path": "del.txt"}), &ctx);
        assert!(
            !r2.is_error,
            "Idempotent delete must not fail on missing file"
        );
    }

    #[test]
    fn test_create_directory_idempotent() {
        let temp_dir = TempDir::new().unwrap();
        let ctx = create_test_context_with_registry(temp_dir.path().to_str().unwrap());

        let r1 = FileOpsTool::execute("1", "create_directory", &json!({"path": "sub/dir"}), &ctx);
        assert!(!r1.is_error);
        assert!(temp_dir.path().join("sub/dir").is_dir());

        let r2 = FileOpsTool::execute("2", "create_directory", &json!({"path": "sub/dir"}), &ctx);
        assert!(
            !r2.is_error,
            "Second create_directory must return cached success"
        );
    }

    #[test]
    fn test_idempotency_registry_cloned_context_shares_state() {
        let temp_dir = TempDir::new().unwrap();
        let ctx = create_test_context_with_registry(temp_dir.path().to_str().unwrap());
        let ctx2 = ctx.clone(); // cloned context shares the same registry

        FileOpsTool::execute(
            "1",
            "write_file",
            &json!({"path": "shared.txt", "content": "x"}),
            &ctx,
        );
        fs::write(temp_dir.path().join("shared.txt"), "CORRUPTED").unwrap();

        // Execute via the cloned context — same registry, so idempotent
        FileOpsTool::execute(
            "2",
            "write_file",
            &json!({"path": "shared.txt", "content": "x"}),
            &ctx2,
        );
        let on_disk = fs::read_to_string(temp_dir.path().join("shared.txt")).unwrap();
        assert_eq!(
            on_disk, "CORRUPTED",
            "Cloned context must share idempotency state"
        );
    }

    // ── Staging backend (two-phase commit) tests ──────────────────────────────

    #[test]
    fn test_write_file_staged_commit() {
        use brainwires_core::StagingBackend;
        use brainwires_tool_runtime::TransactionManager;
        use std::sync::Arc;

        let temp_dir = TempDir::new().unwrap();
        let target = temp_dir.path().join("staged.txt");
        let mgr = Arc::new(TransactionManager::new().unwrap());
        let ctx = ToolContext {
            working_directory: temp_dir.path().to_str().unwrap().to_string(),
            staging_backend: Some(mgr.clone()),
            ..Default::default()
        };

        let result = FileOpsTool::execute(
            "1",
            "write_file",
            &json!({"path": "staged.txt", "content": "staged content"}),
            &ctx,
        );
        assert!(!result.is_error);
        assert!(
            result.content.contains("Staged"),
            "Result must indicate staging"
        );
        assert!(!target.exists(), "File must not exist before commit");

        mgr.commit().unwrap();
        assert!(target.exists());
        assert_eq!(fs::read_to_string(&target).unwrap(), "staged content");
    }

    #[test]
    fn test_write_file_staged_rollback() {
        use brainwires_core::StagingBackend;
        use brainwires_tool_runtime::TransactionManager;
        use std::sync::Arc;

        let temp_dir = TempDir::new().unwrap();
        let target = temp_dir.path().join("rollback.txt");
        let mgr = Arc::new(TransactionManager::new().unwrap());
        let ctx = ToolContext {
            working_directory: temp_dir.path().to_str().unwrap().to_string(),
            staging_backend: Some(mgr.clone()),
            ..Default::default()
        };

        FileOpsTool::execute(
            "1",
            "write_file",
            &json!({"path": "rollback.txt", "content": "data"}),
            &ctx,
        );
        mgr.rollback();
        assert!(!target.exists(), "File must not exist after rollback");
    }

    // ── Concurrent clobber detection ──────────────────────────────────────────
    //
    // Regression test for the bug where two concurrent agents both reported
    // `Success: true` after writing conflicting content to the same file.
    //
    // Honest scope: the in-process FileLockManager serializes individual write
    // syscalls but does NOT prevent two writers from each issuing a full
    // overwrite. The fix — an immediate read-back after write — closes the
    // common race window where writer X's read-back lands after writer Y's
    // write. It does NOT catch the case where X's full write+readback
    // completes before Y's write begins (those are logically sequential and
    // indistinguishable from the non-concurrent case).
    //
    // So this test runs many iterations of tight concurrent writes and asserts
    // that across the batch, at least one writer observes the clobber and
    // returns an error. With the fix this is reliable (the race fires
    // overwhelmingly often); without the fix, both writers always return
    // success regardless of outcome — exactly the bug we're preventing.
    #[test]
    fn write_file_detects_concurrent_clobber() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        const ITERATIONS: usize = 128;
        let mut errors_observed = 0usize;

        for _ in 0..ITERATIONS {
            let temp_dir = TempDir::new().unwrap();
            let working_dir = temp_dir.path().to_str().unwrap().to_string();
            // Larger content widens the write+read window, making the race
            // fire more consistently.
            let content_a = "A".repeat(16 * 1024);
            let content_b = "B".repeat(16 * 1024);
            let barrier = Arc::new(Barrier::new(2));

            let b1 = barrier.clone();
            let wd1 = working_dir.clone();
            let ca = content_a.clone();
            let t1 = thread::spawn(move || {
                let ctx = ToolContext {
                    working_directory: wd1,
                    ..Default::default()
                };
                b1.wait();
                FileOpsTool::execute(
                    "a",
                    "write_file",
                    &json!({"path": "conflict.txt", "content": ca}),
                    &ctx,
                )
            });

            let b2 = barrier.clone();
            let wd2 = working_dir.clone();
            let cb = content_b.clone();
            let t2 = thread::spawn(move || {
                let ctx = ToolContext {
                    working_directory: wd2,
                    ..Default::default()
                };
                b2.wait();
                FileOpsTool::execute(
                    "b",
                    "write_file",
                    &json!({"path": "conflict.txt", "content": cb}),
                    &ctx,
                )
            });

            let r1 = t1.join().unwrap();
            let r2 = t2.join().unwrap();

            // Final file must be exactly one of the two inputs — write_file
            // does not produce interleaved bytes.
            let on_disk = fs::read_to_string(temp_dir.path().join("conflict.txt")).unwrap();
            assert!(on_disk == content_a || on_disk == content_b);

            if r1.is_error || r2.is_error {
                errors_observed += 1;
            }
        }

        // With tight concurrent writes across 128 iterations and 16 KiB
        // contents, the read-back race should fire on a substantial fraction
        // of runs. A single detection is enough to prove the mechanism works;
        // we require at least one to avoid confidence-free green tests.
        assert!(
            errors_observed >= 1,
            "Expected at least one concurrent writer to observe the clobber \
             across {} iterations; saw {}. This suggests the read-back check \
             is not engaging.",
            ITERATIONS,
            errors_observed
        );
    }

    #[test]
    fn test_edit_file_staged_commit() {
        use brainwires_core::StagingBackend;
        use brainwires_tool_runtime::TransactionManager;
        use std::sync::Arc;

        let temp_dir = TempDir::new().unwrap();
        let target = temp_dir.path().join("edit.txt");
        fs::write(&target, "Hello World").unwrap();

        let mgr = Arc::new(TransactionManager::new().unwrap());
        let ctx = ToolContext {
            working_directory: temp_dir.path().to_str().unwrap().to_string(),
            staging_backend: Some(mgr.clone()),
            ..Default::default()
        };

        let result = FileOpsTool::execute(
            "1",
            "edit_file",
            &json!({"path": "edit.txt", "old_text": "World", "new_text": "Rust"}),
            &ctx,
        );
        assert!(!result.is_error);
        assert!(
            result.content.contains("Staged"),
            "Result must indicate staging"
        );

        // Original content unchanged until commit
        assert_eq!(fs::read_to_string(&target).unwrap(), "Hello World");

        mgr.commit().unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "Hello Rust");
    }
}
