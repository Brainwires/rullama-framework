//! Skill handler — detects /commands and dispatches to skill system.

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Result;
use brainwires_agent::skills::{RegistryClient, SkillRegistry, SkillSource};
use semver::VersionReq;

/// Handles skill-based /commands from user messages.
pub struct SkillHandler {
    registry: Mutex<SkillRegistry>,
    /// Optional remote registry URL for skill fallback lookup.
    registry_url: Option<String>,
}

impl SkillHandler {
    /// Create a new skill handler that discovers skills from the given directories.
    pub fn new(skill_dirs: &[PathBuf]) -> Result<Self> {
        let mut registry = SkillRegistry::new();

        if !skill_dirs.is_empty() {
            let paths: Vec<(PathBuf, SkillSource)> = skill_dirs
                .iter()
                .map(|dir| (dir.clone(), SkillSource::Project))
                .collect();
            registry.discover_from(&paths)?;
        }

        Ok(Self {
            registry: Mutex::new(registry),
            registry_url: None,
        })
    }

    /// Attach a remote registry URL for skill fallback lookups.
    pub fn with_registry_url(mut self, url: String) -> Self {
        self.registry_url = Some(url);
        self
    }

    /// Create an empty skill handler with no skills loaded.
    pub fn empty() -> Self {
        Self {
            registry: Mutex::new(SkillRegistry::new()),
            registry_url: None,
        }
    }

    /// Parse a /command from the beginning of a text message.
    ///
    /// Returns `Some((command, args))` if the text starts with `/`,
    /// or `None` if it does not.
    pub fn parse_command(text: &str) -> Option<(&str, &str)> {
        let text = text.trim();
        if !text.starts_with('/') {
            return None;
        }

        // Split on first whitespace
        let without_slash = &text[1..];
        if without_slash.is_empty() {
            return None;
        }

        match without_slash.find(char::is_whitespace) {
            Some(pos) => {
                let command = &without_slash[..pos];
                let args = without_slash[pos..].trim_start();
                Some((command, args))
            }
            None => Some((without_slash, "")),
        }
    }

    /// Resolve a /command to its skill instructions string.
    ///
    /// Returns `Some(instructions)` if the skill exists and its content can be
    /// loaded, `None` if no skill matches the command name.
    ///
    /// Resolution order:
    /// 1. Local filesystem registry (always checked first, synchronously).
    /// 2. Remote registry server (if `registry_url` is configured and no local
    ///    skill was found).  Uses `block_in_place` to call the async client.
    ///
    /// The returned `instructions` string is intended to be prepended to the
    /// user's message so that the agent executes the skill inline.
    pub fn resolve_command(&self, command: &str, _args: &str) -> Result<Option<String>> {
        // 1. Local registry check (fast, no network)
        {
            let mut registry = self
                .registry
                .lock()
                .map_err(|_| anyhow::anyhow!("SkillRegistry lock poisoned"))?;

            if let Ok(skill) = registry.get_skill(command) {
                return Ok(Some(skill.instructions.clone()));
            }
        }

        // 2. Remote registry fallback
        if let Some(ref url) = self.registry_url {
            let url = url.clone();
            let cmd = command.to_string();
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    let client = RegistryClient::new(&url, None);
                    // Search for an exact match on the skill name
                    let matches = client.search(&cmd, None, Some(5)).await?;
                    let matched = matches
                        .into_iter()
                        .find(|m| m.name.eq_ignore_ascii_case(&cmd));

                    if let Some(manifest) = matched {
                        // Download latest version and extract instructions from skill_content
                        let req = VersionReq::STAR; // any version
                        let pkg = client.download(&manifest.name, &req).await?;
                        // Strip YAML frontmatter to extract the Markdown instructions
                        let instructions = strip_frontmatter(&pkg.skill_content);
                        return Ok::<Option<String>, anyhow::Error>(Some(instructions));
                    }
                    Ok(None)
                })
            });

            match result {
                Ok(Some(instructions)) => return Ok(Some(instructions)),
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(
                        command = %command,
                        error = %e,
                        "Registry fallback failed; skill not available"
                    );
                }
            }
        }

        Ok(None)
    }

    /// Return the number of loaded skills.
    pub fn skill_count(&self) -> usize {
        self.registry.lock().map(|r| r.len()).unwrap_or(0)
    }
}

/// Remove YAML frontmatter (everything between the first two `---` delimiters)
/// from a SKILL.md string and return the Markdown body.
fn strip_frontmatter(content: &str) -> String {
    let content = content.trim_start();
    if !content.starts_with("---") {
        return content.to_string();
    }
    // Skip the opening `---`
    let after_open = &content[3..];
    // Find the closing `---`
    if let Some(close_pos) = after_open.find("\n---") {
        let body = &after_open[close_pos + 4..]; // skip `\n---`
        body.trim_start_matches('\n').to_string()
    } else {
        content.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_command_simple() {
        let result = SkillHandler::parse_command("/help");
        assert_eq!(result, Some(("help", "")));
    }

    #[test]
    fn test_parse_command_with_args() {
        let result = SkillHandler::parse_command("/review-pr 123");
        assert_eq!(result, Some(("review-pr", "123")));
    }

    #[test]
    fn test_parse_command_with_multi_args() {
        let result = SkillHandler::parse_command("/search code patterns");
        assert_eq!(result, Some(("search", "code patterns")));
    }

    #[test]
    fn test_parse_command_with_leading_whitespace() {
        let result = SkillHandler::parse_command("  /help ");
        assert_eq!(result, Some(("help", "")));
    }

    #[test]
    fn test_parse_command_not_a_command() {
        assert!(SkillHandler::parse_command("hello world").is_none());
        assert!(SkillHandler::parse_command("").is_none());
        assert!(SkillHandler::parse_command("no slash").is_none());
    }

    #[test]
    fn test_parse_command_bare_slash() {
        assert!(SkillHandler::parse_command("/").is_none());
    }

    #[test]
    fn test_empty_handler() {
        let handler = SkillHandler::empty();
        assert_eq!(handler.skill_count(), 0);
    }

    #[test]
    fn test_resolve_unknown_command() {
        let handler = SkillHandler::empty();
        let result = handler.resolve_command("nonexistent", "").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_new_with_nonexistent_dir() {
        // Should succeed even with non-existent directories (they are skipped)
        let handler = SkillHandler::new(&[PathBuf::from("/nonexistent/skills/dir")]);
        assert!(handler.is_ok());
        assert_eq!(handler.unwrap().skill_count(), 0);
    }
}
