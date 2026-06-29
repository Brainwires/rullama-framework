//! Agent Discovery and Metadata Management
//!
//! Provides functions for discovering, listing, and managing agent sessions
//! via their IPC socket files and metadata JSON files.
//!
//! All functions take `sessions_dir: &Path` as a parameter instead of
//! depending on CLI-specific path resolution. The CLI binds
//! `PlatformPaths::sessions_dir()` at call sites.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::net::UnixStream;

use super::protocol::AgentMetadata;

// ============================================================================
// Agent Metadata File I/O
// ============================================================================

/// Get the metadata file path for an agent session
pub fn get_agent_metadata_path(sessions_dir: &Path, session_id: &str) -> PathBuf {
    sessions_dir.join(format!("{}.meta.json", session_id))
}

/// Write agent metadata to disk
pub fn write_agent_metadata(sessions_dir: &Path, metadata: &AgentMetadata) -> Result<()> {
    let meta_path = get_agent_metadata_path(sessions_dir, &metadata.session_id);

    // Ensure parent directory exists
    if let Some(parent) = meta_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let json =
        serde_json::to_string_pretty(metadata).context("Failed to serialize agent metadata")?;

    std::fs::write(&meta_path, json)
        .with_context(|| format!("Failed to write metadata to {}", meta_path.display()))?;

    tracing::debug!("Wrote agent metadata: {}", meta_path.display());
    Ok(())
}

/// Read agent metadata from disk
pub fn read_agent_metadata(sessions_dir: &Path, session_id: &str) -> Result<Option<AgentMetadata>> {
    let meta_path = get_agent_metadata_path(sessions_dir, session_id);

    if !meta_path.exists() {
        return Ok(None);
    }

    let json = std::fs::read_to_string(&meta_path)
        .with_context(|| format!("Failed to read metadata from {}", meta_path.display()))?;

    let metadata: AgentMetadata = serde_json::from_str(&json)
        .with_context(|| format!("Failed to parse metadata from {}", meta_path.display()))?;

    Ok(Some(metadata))
}

/// Update agent metadata (read-modify-write pattern)
pub fn update_agent_metadata<F>(sessions_dir: &Path, session_id: &str, updater: F) -> Result<()>
where
    F: FnOnce(&mut AgentMetadata),
{
    let mut metadata = read_agent_metadata(sessions_dir, session_id)?
        .ok_or_else(|| anyhow::anyhow!("No metadata found for session {}", session_id))?;

    updater(&mut metadata);
    write_agent_metadata(sessions_dir, &metadata)?;

    Ok(())
}

/// Delete agent metadata file
pub fn delete_agent_metadata(sessions_dir: &Path, session_id: &str) -> Result<()> {
    let meta_path = get_agent_metadata_path(sessions_dir, session_id);

    if meta_path.exists() {
        std::fs::remove_file(&meta_path)
            .with_context(|| format!("Failed to delete metadata: {}", meta_path.display()))?;
        tracing::debug!("Deleted agent metadata: {}", meta_path.display());
    }

    Ok(())
}

// ============================================================================
// Session Listing
// ============================================================================

/// List all available agent sessions
///
/// Lists agent IPC sessions (.sock files), excluding PTY sessions (.pty.sock)
pub fn list_agent_sessions(sessions_dir: &Path) -> Result<Vec<String>> {
    if !sessions_dir.exists() {
        return Ok(Vec::new());
    }

    let mut sessions = Vec::new();

    for entry in std::fs::read_dir(sessions_dir)? {
        let entry = entry?;
        let path = entry.path();

        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            // Skip PTY sockets (.pty.sock)
            if filename.ends_with(".pty.sock") {
                continue;
            }

            // Only process agent IPC sockets (.sock)
            if let Some(session_id) = filename.strip_suffix(".sock") {
                // Remove ".sock"
                sessions.push(session_id.to_string());
            }
        }
    }

    Ok(sessions)
}

/// Check if an agent session exists and is alive
///
/// Uses socket connection test to verify the agent is actually accepting connections.
/// PID checks alone are insufficient because the socket listener may have crashed
/// while the process is still running.
pub async fn is_agent_alive(sessions_dir: &Path, session_id: &str) -> bool {
    let socket_path = super::socket::get_agent_socket_path(sessions_dir, session_id);

    if !socket_path.exists() {
        return false;
    }

    // Try to connect with a reasonable timeout
    // This is the only reliable way to know if the socket is accepting connections
    matches!(
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            UnixStream::connect(&socket_path),
        )
        .await,
        Ok(Ok(_))
    )
}

/// Clean up stale socket files
pub async fn cleanup_stale_sockets(sessions_dir: &Path) -> Result<()> {
    let sessions = list_agent_sessions(sessions_dir)?;

    for session_id in sessions {
        if !is_agent_alive(sessions_dir, &session_id).await {
            cleanup_session(sessions_dir, &session_id)?;
        }
    }

    Ok(())
}

/// Clean up all files for a specific session
///
/// Removes .sock, .pty.sock, .token, .meta.json, .log, .stdout.log, .stderr.log files
pub fn cleanup_session(sessions_dir: &Path, session_id: &str) -> Result<()> {
    let extensions = [
        "sock",
        "pty.sock",
        "token",
        "meta.json",
        "log",
        "stdout.log",
        "stderr.log",
    ];

    for ext in extensions {
        let file_path = sessions_dir.join(format!("{}.{}", session_id, ext));
        if file_path.exists() {
            if let Err(e) = std::fs::remove_file(&file_path) {
                tracing::warn!("Failed to remove {}: {}", file_path.display(), e);
            } else {
                tracing::debug!("Cleaned up: {}", file_path.display());
            }
        }
    }

    Ok(())
}

// ============================================================================
// Agent Discovery with Metadata
// ============================================================================

/// List all agent sessions with their metadata
///
/// Returns metadata for all agents that have both a socket and metadata file.
/// Agents without metadata are included with basic info derived from the socket.
pub fn list_agent_sessions_with_metadata(sessions_dir: &Path) -> Result<Vec<AgentMetadata>> {
    if !sessions_dir.exists() {
        return Ok(Vec::new());
    }

    let mut agents = Vec::new();

    for entry in std::fs::read_dir(sessions_dir)? {
        let entry = entry?;
        let path = entry.path();

        // Look for .sock files but NOT .pty.sock files
        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            // Skip PTY sockets
            if filename.ends_with(".pty.sock") {
                continue;
            }

            // Only process agent IPC sockets (.sock)
            if let Some(session_id) = filename.strip_suffix(".sock") {
                // Remove ".sock"

                // Try to read metadata
                match read_agent_metadata(sessions_dir, session_id) {
                    Ok(Some(metadata)) => {
                        agents.push(metadata);
                    }
                    Ok(None) => {
                        // No metadata file - create basic metadata
                        let metadata = AgentMetadata::new(
                            session_id.to_string(),
                            "unknown".to_string(),
                            std::env::current_dir()
                                .map(|p| p.to_string_lossy().to_string())
                                .unwrap_or_default(),
                        );
                        agents.push(metadata);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to read metadata for session {}: {}", session_id, e);
                    }
                }
            }
        }
    }

    // Sort by created_at (oldest first for tree building)
    agents.sort_by_key(|a| a.created_at);

    Ok(agents)
}

/// Get children of a given agent
pub fn get_child_agents(
    sessions_dir: &Path,
    parent_session_id: &str,
) -> Result<Vec<AgentMetadata>> {
    let all_agents = list_agent_sessions_with_metadata(sessions_dir)?;

    Ok(all_agents
        .into_iter()
        .filter(|a| a.parent_agent_id.as_deref() == Some(parent_session_id))
        .collect())
}

/// Get the root agents (those without a parent)
pub fn get_root_agents(sessions_dir: &Path) -> Result<Vec<AgentMetadata>> {
    let all_agents = list_agent_sessions_with_metadata(sessions_dir)?;

    Ok(all_agents
        .into_iter()
        .filter(|a| a.parent_agent_id.is_none())
        .collect())
}

/// Get the depth of an agent in the tree (root = 0)
///
/// This walks up the parent chain to calculate the depth.
/// Returns 0 for root agents (no parent).
pub fn get_agent_depth(sessions_dir: &Path, session_id: &str) -> Result<u32> {
    let mut depth = 0;
    let mut current_id = session_id.to_string();

    loop {
        match read_agent_metadata(sessions_dir, &current_id) {
            Ok(Some(metadata)) => match metadata.parent_agent_id {
                Some(parent_id) => {
                    depth += 1;
                    current_id = parent_id;
                }
                None => break,
            },
            Ok(None) => break, // No metadata = assume root
            Err(_) => break,   // Error reading = stop traversal
        }
    }

    Ok(depth)
}

/// Build a tree structure of agents for display
///
/// Returns a formatted string showing the agent hierarchy.
pub fn format_agent_tree(sessions_dir: &Path, current_session_id: Option<&str>) -> Result<String> {
    let all_agents = list_agent_sessions_with_metadata(sessions_dir)?;

    if all_agents.is_empty() {
        return Ok("No active agents".to_string());
    }

    let mut output = String::new();

    // Helper function to render a subtree
    fn render_subtree(
        agents: &[AgentMetadata],
        parent_id: Option<&str>,
        prefix: &str,
        is_last: bool,
        current_session_id: Option<&str>,
        output: &mut String,
    ) {
        let children: Vec<_> = agents
            .iter()
            .filter(|a| a.parent_agent_id.as_deref() == parent_id)
            .collect();

        for (i, agent) in children.iter().enumerate() {
            let is_last_child = i == children.len() - 1;
            let connector = if is_last { "└── " } else { "├── " };
            let child_prefix = if is_last { "    " } else { "│   " };

            // Mark current agent
            let marker = if current_session_id == Some(agent.session_id.as_str()) {
                " ← current"
            } else {
                ""
            };

            // Format status
            let status = if agent.is_busy { "busy" } else { "idle" };

            // Format the agent line
            let reason = agent.spawn_reason.as_deref().unwrap_or("");
            let reason_str = if reason.is_empty() {
                String::new()
            } else {
                format!(" ({})", reason)
            };

            output.push_str(&format!(
                "{}{}{} [{}] {}{}{}\n",
                prefix, connector, agent.session_id, agent.model, status, reason_str, marker,
            ));

            // Render children
            render_subtree(
                agents,
                Some(&agent.session_id),
                &format!("{}{}", prefix, child_prefix),
                is_last_child,
                current_session_id,
                output,
            );
        }
    }

    output.push_str("Agents:\n");
    render_subtree(&all_agents, None, "", true, current_session_id, &mut output);

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_metadata_serialization() {
        let metadata = AgentMetadata::new(
            "test-session-123".to_string(),
            "gpt-4".to_string(),
            "/home/user/project".to_string(),
        );

        let json = serde_json::to_string(&metadata).unwrap();
        let parsed: AgentMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.session_id, "test-session-123");
        assert_eq!(parsed.model, "gpt-4");
        assert_eq!(parsed.working_directory, "/home/user/project");
        assert!(parsed.parent_agent_id.is_none());
        assert!(!parsed.is_busy);
    }

    #[test]
    fn test_agent_metadata_with_parent() {
        let metadata = AgentMetadata::new(
            "child-session".to_string(),
            "gpt-3.5".to_string(),
            "/home/user/project".to_string(),
        )
        .with_parent(
            "parent-session".to_string(),
            Some("investigate bug".to_string()),
        );

        assert_eq!(metadata.parent_agent_id, Some("parent-session".to_string()));
        assert_eq!(metadata.spawn_reason, Some("investigate bug".to_string()));
    }

    #[test]
    fn test_metadata_file_io() {
        let temp_dir = tempfile::tempdir().unwrap();
        let sessions_dir = temp_dir.path();

        let metadata = AgentMetadata::new(
            "test-io-session".to_string(),
            "claude-2".to_string(),
            "/home/user/project".to_string(),
        );

        // Write and read back
        write_agent_metadata(sessions_dir, &metadata).unwrap();
        let read_back = read_agent_metadata(sessions_dir, "test-io-session").unwrap();
        assert!(read_back.is_some());
        let read_back = read_back.unwrap();
        assert_eq!(read_back.session_id, "test-io-session");
        assert_eq!(read_back.model, "claude-2");

        // Update metadata
        update_agent_metadata(sessions_dir, "test-io-session", |m| {
            m.set_busy(true);
        })
        .unwrap();
        let updated = read_agent_metadata(sessions_dir, "test-io-session")
            .unwrap()
            .unwrap();
        assert!(updated.is_busy);

        // Delete metadata
        delete_agent_metadata(sessions_dir, "test-io-session").unwrap();
        let deleted = read_agent_metadata(sessions_dir, "test-io-session").unwrap();
        assert!(deleted.is_none());
    }

    #[test]
    fn test_agent_tree_relationships() {
        let agents = [
            AgentMetadata::new(
                "parent-1".to_string(),
                "gpt-4".to_string(),
                "/home".to_string(),
            ),
            AgentMetadata::new(
                "child-1".to_string(),
                "gpt-3.5".to_string(),
                "/home".to_string(),
            )
            .with_parent("parent-1".to_string(), Some("investigate".to_string())),
            AgentMetadata::new(
                "child-2".to_string(),
                "claude".to_string(),
                "/home".to_string(),
            )
            .with_parent("parent-1".to_string(), Some("code review".to_string())),
        ];

        // Verify parent-child relationships
        let children: Vec<_> = agents
            .iter()
            .filter(|a| a.parent_agent_id.as_deref() == Some("parent-1"))
            .collect();

        assert_eq!(children.len(), 2);
        assert!(children.iter().any(|a| a.session_id == "child-1"));
        assert!(children.iter().any(|a| a.session_id == "child-2"));
    }
}
