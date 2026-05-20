//! Skill Metadata and Core Types
//!
//! Defines the data structures for Agent Skills with progressive disclosure:
//! - `SkillMetadata`: Lightweight metadata loaded at startup
//! - `Skill`: Full skill content loaded on-demand
//! - Supporting enums for source and execution modes

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Level 3 resources for a skill — discovered and loaded on-demand only when referenced.
///
/// Per the Agent Skills specification, a skill directory may contain:
/// - `scripts/`    — executable files (Python, Bash, JS, etc.)
/// - `references/` — detailed documentation or reference material
/// - `assets/`     — templates, images, data files, or other supporting content
///
/// These are never loaded at startup (Level 1) or on skill activation (Level 2).
/// They are listed here so agents can discover what's available and request specific
/// files as needed, keeping context usage minimal.
#[derive(Debug, Clone, Default)]
pub struct SkillResources {
    /// Files in `<skill-dir>/scripts/`
    pub scripts: Vec<PathBuf>,
    /// Files in `<skill-dir>/references/`
    pub references: Vec<PathBuf>,
    /// Files in `<skill-dir>/assets/`
    pub assets: Vec<PathBuf>,
}

impl SkillResources {
    /// Returns true if no resource directories contain any files
    pub fn is_empty(&self) -> bool {
        self.scripts.is_empty() && self.references.is_empty() && self.assets.is_empty()
    }

    /// Total number of resource files across all directories
    pub fn total_count(&self) -> usize {
        self.scripts.len() + self.references.len() + self.assets.len()
    }
}

/// Source location of a skill
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SkillSource {
    /// Personal skills from ~/.brainwires/skills/
    #[default]
    Personal,
    /// Project skills from .brainwires/skills/
    Project,
    /// Built-in skills bundled with the application
    Builtin,
}

impl std::fmt::Display for SkillSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkillSource::Personal => write!(f, "personal"),
            SkillSource::Project => write!(f, "project"),
            SkillSource::Builtin => write!(f, "builtin"),
        }
    }
}

/// Execution mode for a skill
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SkillExecutionMode {
    /// Execute inline in current conversation context
    /// Instructions are injected into the conversation
    #[default]
    Inline,
    /// Spawn a dedicated subagent via AgentPool
    /// Skill runs in background with its own context
    Subagent,
    /// Execute as Rhai script via OrchestratorTool
    /// For programmatic tool orchestration
    Script,
}

impl SkillExecutionMode {
    /// Parse execution mode from string
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "subagent" => SkillExecutionMode::Subagent,
            "script" => SkillExecutionMode::Script,
            _ => SkillExecutionMode::Inline,
        }
    }
}

impl std::fmt::Display for SkillExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkillExecutionMode::Inline => write!(f, "inline"),
            SkillExecutionMode::Subagent => write!(f, "subagent"),
            SkillExecutionMode::Script => write!(f, "script"),
        }
    }
}

/// Lightweight skill metadata loaded at startup
///
/// Only contains the information needed for:
/// - Displaying skill listings
/// - Semantic matching against descriptions
/// - Determining if full content should be loaded
///
/// The actual instructions are lazily loaded when the skill is activated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMetadata {
    /// Skill name (lowercase, hyphens only, max 64 chars)
    /// Used as the identifier and for `/skill-name` invocation
    pub name: String,

    /// Description (max 1024 chars)
    /// Used for semantic matching - should include keywords users would naturally say
    pub description: String,

    /// Optional: Restrict available tools during execution
    /// If None, all tools are available
    /// Tool names should match those in ToolRegistry
    #[serde(rename = "allowed-tools")]
    pub allowed_tools: Option<Vec<String>>,

    /// Optional: Software license for the skill
    pub license: Option<String>,

    /// Optional: Environment requirements (max 500 chars)
    /// Indicates intended product, required system packages, network access, etc.
    /// Part of the Agent Skills specification.
    pub compatibility: Option<String>,

    /// Optional: Specific model to use for this skill
    /// Overrides the default model when executing
    /// (Brainwires extension, not part of the Agent Skills specification)
    pub model: Option<String>,

    /// Optional: Custom key-value metadata
    /// Common keys: "category", "execution", "author", "version"
    pub metadata: Option<HashMap<String, String>>,

    /// Optional: lifecycle hook event types this skill subscribes to.
    ///
    /// When set, the skill executor will register hooks that fire the skill
    /// on matching lifecycle events (e.g., `["agent_started", "tool_after_execute"]`).
    /// See `brainwires_core::lifecycle::LifecycleEvent` for valid event types.
    /// (Brainwires extension, not part of the Agent Skills specification)
    #[serde(default)]
    pub hooks: Option<Vec<String>>,

    /// Source location (Personal, Project, or Builtin)
    #[serde(skip)]
    pub source: SkillSource,

    /// File path for lazy loading the full content
    #[serde(skip)]
    pub source_path: PathBuf,

    /// Parent directory of the skill (set only for subdirectory layout: `skill-name/SKILL.md`).
    ///
    /// Used by [`SkillRegistry::get_resources`](crate::registry::SkillRegistry::get_resources) to discover Level 3 resource files
    /// (`scripts/`, `references/`, `assets/`) without scanning at startup.
    /// None for flat file layout (`skill-name.md`).
    #[serde(skip)]
    pub resources_dir: Option<PathBuf>,
}

impl SkillMetadata {
    /// Create new skill metadata
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description,
            allowed_tools: None,
            license: None,
            compatibility: None,
            model: None,
            metadata: None,
            hooks: None,
            source: SkillSource::Personal,
            source_path: PathBuf::new(),
            resources_dir: None,
        }
    }

    /// Set the source location
    pub fn with_source(mut self, source: SkillSource) -> Self {
        self.source = source;
        self
    }

    /// Set the source path
    pub fn with_source_path(mut self, path: PathBuf) -> Self {
        self.source_path = path;
        self
    }

    /// Get the execution mode from metadata
    pub fn execution_mode(&self) -> SkillExecutionMode {
        self.metadata
            .as_ref()
            .and_then(|m| m.get("execution"))
            .map(|e| SkillExecutionMode::parse(e))
            .unwrap_or_default()
    }

    /// Get a custom metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.as_ref().and_then(|m| m.get(key))
    }

    /// Check if skill has tool restrictions
    pub fn has_tool_restrictions(&self) -> bool {
        self.allowed_tools
            .as_ref()
            .map(|t| !t.is_empty())
            .unwrap_or(false)
    }

    /// Check if a tool is allowed for this skill
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        match &self.allowed_tools {
            Some(allowed) => allowed.iter().any(|t| t == tool_name),
            None => true, // No restrictions = all tools allowed
        }
    }
}

/// Full skill content loaded on-demand
///
/// Contains both the metadata and the instruction content.
/// Created by parsing the full SKILL.md file when the skill is activated.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Lightweight metadata
    pub metadata: SkillMetadata,

    /// Full instruction content (markdown body after frontmatter)
    pub instructions: String,

    /// Execution mode (derived from metadata or defaults to Inline)
    pub execution_mode: SkillExecutionMode,
}

impl Skill {
    /// Create a new skill from metadata and instructions
    pub fn new(metadata: SkillMetadata, instructions: String) -> Self {
        let execution_mode = metadata.execution_mode();
        Self {
            metadata,
            instructions,
            execution_mode,
        }
    }

    /// Get the skill name
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Get the skill description
    pub fn description(&self) -> &str {
        &self.metadata.description
    }

    /// Get the allowed tools (if any)
    pub fn allowed_tools(&self) -> Option<&Vec<String>> {
        self.metadata.allowed_tools.as_ref()
    }

    /// Get the model override (if any)
    pub fn model(&self) -> Option<&String> {
        self.metadata.model.as_ref()
    }

    /// Check if this skill should run as a subagent
    pub fn runs_as_subagent(&self) -> bool {
        matches!(self.execution_mode, SkillExecutionMode::Subagent)
    }

    /// Check if this skill is a Rhai script
    pub fn is_script(&self) -> bool {
        matches!(self.execution_mode, SkillExecutionMode::Script)
    }
}

/// Result of skill execution
#[derive(Debug, Clone)]
pub enum SkillResult {
    /// Instructions to inject into conversation (inline execution)
    Inline {
        /// The rendered instructions
        instructions: String,
        /// Optional model override
        model_override: Option<String>,
    },
    /// Subagent spawned (background execution)
    Subagent {
        /// The spawned agent's ID
        agent_id: String,
    },
    /// Script executed via OrchestratorTool
    Script {
        /// Script output
        output: String,
        /// Whether execution resulted in an error
        is_error: bool,
    },
}

impl SkillResult {
    /// Create an inline result
    pub fn inline(instructions: String, model_override: Option<String>) -> Self {
        SkillResult::Inline {
            instructions,
            model_override,
        }
    }

    /// Create a subagent result
    pub fn subagent(agent_id: String) -> Self {
        SkillResult::Subagent { agent_id }
    }

    /// Create a script result
    pub fn script(output: String, is_error: bool) -> Self {
        SkillResult::Script { output, is_error }
    }

    /// Check if this result is an error
    pub fn is_error(&self) -> bool {
        matches!(self, SkillResult::Script { is_error: true, .. })
    }
}

/// Match result from skill router
#[derive(Debug, Clone)]
pub struct SkillMatch {
    /// Name of the matched skill
    pub skill_name: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// How the match was determined
    pub source: MatchSource,
}

impl SkillMatch {
    /// Create a new skill match
    pub fn new(skill_name: String, confidence: f32, source: MatchSource) -> Self {
        Self {
            skill_name,
            confidence,
            source,
        }
    }

    /// Create a semantic match
    pub fn semantic(skill_name: String, confidence: f32) -> Self {
        Self::new(skill_name, confidence, MatchSource::Semantic)
    }

    /// Create a keyword match
    pub fn keyword(skill_name: String, confidence: f32) -> Self {
        Self::new(skill_name, confidence, MatchSource::Keyword)
    }

    /// Create an explicit match (user invoked directly)
    pub fn explicit(skill_name: String) -> Self {
        Self::new(skill_name, 1.0, MatchSource::Explicit)
    }
}

/// How a skill match was determined
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchSource {
    /// Matched via semantic similarity (LocalRouter)
    Semantic,
    /// Matched via keyword patterns
    Keyword,
    /// User explicitly invoked the skill
    Explicit,
}

impl std::fmt::Display for MatchSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchSource::Semantic => write!(f, "semantic"),
            MatchSource::Keyword => write!(f, "keyword"),
            MatchSource::Explicit => write!(f, "explicit"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_source_display() {
        assert_eq!(SkillSource::Personal.to_string(), "personal");
        assert_eq!(SkillSource::Project.to_string(), "project");
        assert_eq!(SkillSource::Builtin.to_string(), "builtin");
    }

    #[test]
    fn test_execution_mode_from_str() {
        assert_eq!(
            SkillExecutionMode::parse("inline"),
            SkillExecutionMode::Inline
        );
        assert_eq!(
            SkillExecutionMode::parse("subagent"),
            SkillExecutionMode::Subagent
        );
        assert_eq!(
            SkillExecutionMode::parse("script"),
            SkillExecutionMode::Script
        );
        assert_eq!(
            SkillExecutionMode::parse("SUBAGENT"),
            SkillExecutionMode::Subagent
        );
        assert_eq!(
            SkillExecutionMode::parse("unknown"),
            SkillExecutionMode::Inline
        );
    }

    #[test]
    fn test_skill_metadata_creation() {
        let metadata = SkillMetadata::new(
            "test-skill".to_string(),
            "A test skill for unit testing".to_string(),
        );

        assert_eq!(metadata.name, "test-skill");
        assert_eq!(metadata.description, "A test skill for unit testing");
        assert!(metadata.allowed_tools.is_none());
        assert!(metadata.license.is_none());
        assert!(metadata.model.is_none());
        assert_eq!(metadata.source, SkillSource::Personal);
    }

    #[test]
    fn test_skill_metadata_with_source() {
        let metadata = SkillMetadata::new("test".to_string(), "desc".to_string())
            .with_source(SkillSource::Project);

        assert_eq!(metadata.source, SkillSource::Project);
    }

    #[test]
    fn test_skill_metadata_tool_restrictions() {
        let mut metadata = SkillMetadata::new("test".to_string(), "desc".to_string());

        // No restrictions
        assert!(!metadata.has_tool_restrictions());
        assert!(metadata.is_tool_allowed("any_tool"));

        // With restrictions
        metadata.allowed_tools = Some(vec!["Read".to_string(), "Grep".to_string()]);
        assert!(metadata.has_tool_restrictions());
        assert!(metadata.is_tool_allowed("Read"));
        assert!(metadata.is_tool_allowed("Grep"));
        assert!(!metadata.is_tool_allowed("Write"));
    }

    #[test]
    fn test_skill_metadata_execution_mode() {
        let mut metadata = SkillMetadata::new("test".to_string(), "desc".to_string());

        // Default (no metadata)
        assert_eq!(metadata.execution_mode(), SkillExecutionMode::Inline);

        // With execution metadata
        let mut custom_metadata = HashMap::new();
        custom_metadata.insert("execution".to_string(), "subagent".to_string());
        metadata.metadata = Some(custom_metadata);

        assert_eq!(metadata.execution_mode(), SkillExecutionMode::Subagent);
    }

    #[test]
    fn test_skill_creation() {
        let metadata =
            SkillMetadata::new("review-pr".to_string(), "Reviews pull requests".to_string());
        let skill = Skill::new(
            metadata,
            "# Review Instructions\n\nDo the review.".to_string(),
        );

        assert_eq!(skill.name(), "review-pr");
        assert_eq!(skill.description(), "Reviews pull requests");
        assert!(skill.instructions.contains("Review Instructions"));
        assert_eq!(skill.execution_mode, SkillExecutionMode::Inline);
    }

    #[test]
    fn test_skill_result_types() {
        let inline = SkillResult::inline("instructions".to_string(), None);
        assert!(!inline.is_error());

        let subagent = SkillResult::subagent("agent-123".to_string());
        assert!(!subagent.is_error());

        let script_ok = SkillResult::script("output".to_string(), false);
        assert!(!script_ok.is_error());

        let script_err = SkillResult::script("error".to_string(), true);
        assert!(script_err.is_error());
    }

    #[test]
    fn test_skill_match() {
        let semantic = SkillMatch::semantic("review-pr".to_string(), 0.85);
        assert_eq!(semantic.source, MatchSource::Semantic);
        assert_eq!(semantic.confidence, 0.85);

        let keyword = SkillMatch::keyword("commit".to_string(), 0.6);
        assert_eq!(keyword.source, MatchSource::Keyword);

        let explicit = SkillMatch::explicit("explain-code".to_string());
        assert_eq!(explicit.source, MatchSource::Explicit);
        assert_eq!(explicit.confidence, 1.0);
    }
}
