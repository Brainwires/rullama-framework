//! Bridge skills into the `ToolExecutor` tool-calling surface.
//!
//! A [`SkillToolExecutor`] decorates any base [`ToolExecutor`] with one extra
//! tool per registered skill. The model sees a flat list of tools — regular
//! tools plus `skill_<name>` entries — and can call them uniformly. Behind
//! the scenes each skill tool call routes through [`SkillExecutor`] and
//! returns the skill's rendered instructions (or agent id / script output,
//! depending on execution mode) as the tool result.
//!
//! This closes the skill-wiring divergence in the extras/ apps: with this
//! adapter `agent-chat`, `brainwires-cli`, `brainclaw`, and `voice-assistant`
//! all enable the same skill with one `.with_skills(...)` call on the
//! ChatAgent builder.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::RwLock;

use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult, ToolUse};
use brainwires_tool_runtime::ToolExecutor;

use super::executor::SkillExecutor;
use super::metadata::SkillResult;
use super::registry::SkillRegistry;

/// Prefix used for skill-tool names so they don't collide with regular tools.
const SKILL_TOOL_PREFIX: &str = "skill_";

/// Decorates a base [`ToolExecutor`] with one extra tool per skill in the
/// attached [`SkillRegistry`].
pub struct SkillToolExecutor {
    base: Arc<dyn ToolExecutor>,
    registry: Arc<RwLock<SkillRegistry>>,
    skill_executor: SkillExecutor,
}

impl SkillToolExecutor {
    /// Wrap `base` with skill dispatch.
    pub fn new(base: Arc<dyn ToolExecutor>, registry: Arc<RwLock<SkillRegistry>>) -> Self {
        let skill_executor = SkillExecutor::new(registry.clone());
        Self {
            base,
            registry,
            skill_executor,
        }
    }

    /// Normalise a skill name into a valid tool identifier.
    fn tool_name_for(skill_name: &str) -> String {
        format!("{SKILL_TOOL_PREFIX}{}", skill_name.replace('-', "_"))
    }

    /// Resolve a tool name back to the underlying skill name, if any.
    fn skill_name_for(tool_name: &str) -> Option<String> {
        tool_name
            .strip_prefix(SKILL_TOOL_PREFIX)
            .map(|s| s.replace('_', "-"))
    }

    /// Build the JSON-Schema for a skill's argument bag. Skills accept a
    /// generic `args: object` map — the template engine substitutes
    /// `{{name}}` placeholders in the skill body from this map.
    fn skill_input_schema() -> ToolInputSchema {
        let mut properties = HashMap::new();
        properties.insert(
            "args".to_string(),
            json!({
                "type": "object",
                "description": "Optional key-value arguments the skill template references via {{name}} placeholders.",
                "additionalProperties": { "type": "string" }
            }),
        );
        ToolInputSchema::object(properties, vec![])
    }
}

#[async_trait]
impl ToolExecutor for SkillToolExecutor {
    async fn execute(&self, tool_use: &ToolUse, context: &ToolContext) -> Result<ToolResult> {
        // Route skill_* tools to SkillExecutor; everything else falls through.
        let Some(skill_name) = Self::skill_name_for(&tool_use.name) else {
            return self.base.execute(tool_use, context).await;
        };

        let args: HashMap<String, String> = tool_use
            .input
            .get("args")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        match self.skill_executor.execute_by_name(&skill_name, args).await {
            Ok(SkillResult::Inline { instructions, .. }) => {
                Ok(ToolResult::success(tool_use.id.clone(), instructions))
            }
            Ok(SkillResult::Subagent { agent_id }) => Ok(ToolResult::success(
                tool_use.id.clone(),
                format!(
                    "[skill `{skill_name}` prepared as subagent {agent_id}; caller should spawn it]"
                ),
            )),
            Ok(SkillResult::Script { output, is_error }) => {
                if is_error {
                    Ok(ToolResult::error(tool_use.id.clone(), output))
                } else {
                    Ok(ToolResult::success(tool_use.id.clone(), output))
                }
            }
            Err(e) => Ok(ToolResult::error(
                tool_use.id.clone(),
                format!("skill `{skill_name}` execution failed: {e:#}"),
            )),
        }
    }

    fn available_tools(&self) -> Vec<Tool> {
        let mut tools = self.base.available_tools();
        // best-effort snapshot of the registry — lock is cheap & fast.
        if let Ok(registry) = self.registry.try_read() {
            for meta in registry.all_metadata() {
                tools.push(Tool {
                    name: Self::tool_name_for(&meta.name),
                    description: meta.description.clone(),
                    input_schema: Self::skill_input_schema(),
                    requires_approval: false,
                    defer_loading: false,
                    allowed_callers: vec![],
                    input_examples: vec![],
                    serialize: false,
                });
            }
        }
        tools
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::SkillSource;
    use brainwires_tool_builtins::BuiltinToolExecutor;
    use brainwires_tool_runtime::ToolRegistry;

    async fn build_registry_with(name: &str, description: &str) -> Arc<RwLock<SkillRegistry>> {
        // Use a temp SKILL.md so parser can actually load the full content
        // on-demand when the executor calls execute_by_name.
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        let body = format!(
            "---\nname: {name}\ndescription: {description}\n---\n\nHello from skill {name}.\n"
        );
        std::fs::write(skill_dir.join("SKILL.md"), body).unwrap();

        let mut reg = SkillRegistry::new();
        reg.discover_from(&[(dir.path().to_path_buf(), SkillSource::Personal)])
            .unwrap();
        // Intentionally leak the tempdir so the SKILL.md stays on disk for
        // the lifetime of the test — tempfile drop runs after the registry
        // would otherwise try to lazy-load.
        std::mem::forget(dir);
        Arc::new(RwLock::new(reg))
    }

    #[tokio::test]
    async fn tool_name_roundtrip() {
        let t = SkillToolExecutor::tool_name_for("code-review");
        assert_eq!(t, "skill_code_review");
        assert_eq!(
            SkillToolExecutor::skill_name_for(&t),
            Some("code-review".to_string())
        );
        assert_eq!(SkillToolExecutor::skill_name_for("read_file"), None);
    }

    #[tokio::test]
    async fn available_tools_includes_skill_entries() {
        let base = Arc::new(BuiltinToolExecutor::new(
            ToolRegistry::new(),
            ToolContext::default(),
        )) as Arc<dyn ToolExecutor>;
        let registry = build_registry_with("my-skill", "does useful things").await;
        let exec = SkillToolExecutor::new(base, registry);

        let tools = exec.available_tools();
        let names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
        assert!(
            names.contains(&"skill_my_skill".to_string()),
            "names were {names:?}"
        );
    }

    #[tokio::test]
    async fn invoke_skill_returns_rendered_instructions() {
        let base = Arc::new(BuiltinToolExecutor::new(
            ToolRegistry::new(),
            ToolContext::default(),
        )) as Arc<dyn ToolExecutor>;
        let registry = build_registry_with("greeter", "say hi").await;
        let exec = SkillToolExecutor::new(base, registry);

        let tu = ToolUse {
            id: "t-1".into(),
            name: "skill_greeter".into(),
            input: json!({ "args": {} }),
        };
        let result = exec.execute(&tu, &ToolContext::default()).await.unwrap();
        assert!(!result.is_error);
        assert!(
            result.content.contains("Hello from skill greeter"),
            "content was: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn non_skill_tool_passes_through() {
        // Base executor that refuses every call — proves we're not
        // accidentally routing non-skill tools to SkillExecutor.
        struct RejectExecutor;
        #[async_trait]
        impl ToolExecutor for RejectExecutor {
            async fn execute(&self, tu: &ToolUse, _: &ToolContext) -> Result<ToolResult> {
                Ok(ToolResult::error(
                    tu.id.clone(),
                    format!("rejected: {}", tu.name),
                ))
            }
            fn available_tools(&self) -> Vec<Tool> {
                vec![]
            }
        }

        let base = Arc::new(RejectExecutor) as Arc<dyn ToolExecutor>;
        let registry = Arc::new(RwLock::new(SkillRegistry::new()));
        let exec = SkillToolExecutor::new(base, registry);

        let tu = ToolUse {
            id: "t-1".into(),
            name: "execute_command".into(),
            input: json!({}),
        };
        let result = exec.execute(&tu, &ToolContext::default()).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("rejected: execute_command"));
    }

    #[tokio::test]
    async fn unknown_skill_returns_error_result() {
        let base = Arc::new(BuiltinToolExecutor::new(
            ToolRegistry::new(),
            ToolContext::default(),
        )) as Arc<dyn ToolExecutor>;
        let registry = Arc::new(RwLock::new(SkillRegistry::new()));
        let exec = SkillToolExecutor::new(base, registry);

        let tu = ToolUse {
            id: "t-1".into(),
            name: "skill_does_not_exist".into(),
            input: json!({ "args": {} }),
        };
        let result = exec.execute(&tu, &ToolContext::default()).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("execution failed"));
    }
}
