//! Skill slash commands — /skill, /skills, /skill:show, /skill:reload, /skill:create

use super::super::super::state::{App, LogLevel, TuiMessage};
impl App {
    /// Handle /skill <name> [args...] - invoke a skill
    ///
    /// Dispatches on the skill's declared `execution_mode`:
    ///
    /// - **Inline** — skill body injected as a system message; `allowed_tools`
    ///   restricts the next AI turn via `pending_skill_tool_scope`.
    /// - **Subagent** — body formatted as the agent's system prompt (via
    ///   `SkillExecutor::prepare_subagent`); runs inside the current agent
    ///   context with the declared tool allowlist. Functionally equivalent
    ///   to Inline + scoping for this TUI — spawning a separate TaskAgent
    ///   requires `AgentPool` wiring that is a future pass. Users who want
    ///   true isolation invoke `/spawn` explicitly.
    /// - **Script** — rendered body is framed as an instruction for the AI
    ///   to run via the `execute_script` tool. Tools are scoped to the
    ///   skill's allowlist so the script runs in a constrained context.
    pub(super) async fn handle_invoke_skill(&mut self, name: &str, args: Vec<String>) {
        use brainwires_skills::{SkillExecutionMode, SkillSource};

        if self.skill_registry.is_none() {
            self.add_console_message("Skill registry not initialized".to_string());
            return;
        }

        // Render positional args as key=value when they look like assignments,
        // else keep the bare tokens under `args` so the template can reference
        // them positionally if needed. SkillExecutor::execute renders
        // `{{key}}` substitutions against this map.
        let mut arg_map: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        let mut positional: Vec<String> = Vec::new();
        for a in &args {
            if let Some((k, v)) = a.split_once('=') {
                arg_map.insert(k.trim().to_string(), v.trim().to_string());
            } else {
                positional.push(a.clone());
            }
        }
        if !positional.is_empty() {
            arg_map.insert("args".to_string(), positional.join(" "));
        }

        // Load skill + mode + allowed_tools + description. Clone off what we
        // need before dropping the registry borrow.
        let (rendered_body, description, source_str, mode, allowed_tools, skill_name_copy) = {
            let registry = self.skill_registry.as_mut().expect("checked above");
            match registry.get_skill(name) {
                Ok(skill) => {
                    let src = match skill.metadata.source {
                        SkillSource::Personal => "personal",
                        SkillSource::Project => "project",
                        SkillSource::Builtin => "builtin",
                    };
                    let mode = skill.execution_mode;
                    let allowed = skill.allowed_tools().cloned();
                    let name_s = skill.name().to_string();
                    let desc = skill.description().to_string();
                    let body = brainwires_skills::render_template(&skill.instructions, &arg_map);
                    (body, desc, src, mode, allowed, name_s)
                }
                Err(e) => {
                    self.add_console_message(format!("Failed to load skill '{}': {}", name, e));
                    return;
                }
            }
        };

        // Mode-specific message body.
        let skill_msg = match mode {
            SkillExecutionMode::Inline => format!(
                "## Skill: {} ({})\n\n{}",
                skill_name_copy, source_str, rendered_body
            ),
            SkillExecutionMode::Subagent => {
                // Use the framework's prepared system prompt so subagent
                // semantics are consistent with the framework tests.
                format!(
                    "## Skill (subagent-prepared): {} ({})\n\n\
                     You are executing the '{}' skill.\n\n\
                     **Description**: {}\n\n\
                     **Instructions**:\n{}",
                    skill_name_copy, source_str, skill_name_copy, description, rendered_body
                )
            }
            SkillExecutionMode::Script => {
                // Frame the body as a direct instruction to run it. This is
                // deliberately emphatic — the model complies with explicit
                // "execute this script" framing.
                format!(
                    "## Skill (script): {} ({})\n\n\
                     You MUST execute the following Rhai script via the \
                     `execute_script` tool. Do not paraphrase or plan — \
                     call the tool directly with this script as its body:\n\n\
                     ```rhai\n{}\n```",
                    skill_name_copy, source_str, rendered_body
                )
            }
        };

        if !positional.is_empty() {
            self.add_console_message(format!(
                "Skill '{}' received unresolved positional args: {}",
                skill_name_copy,
                positional.join(" ")
            ));
        }

        // User-visible transcript line (system role, so the UI paints it
        // distinct from user messages).
        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: skill_msg.clone(),
            created_at: chrono::Utc::now().timestamp(),
        });

        use crate::types::message::{Message, MessageContent, Role};
        self.conversation_history.push(Message {
            role: Role::System,
            content: MessageContent::Text(skill_msg),
            name: None,
            metadata: None,
        });

        // Scope the next AI turn's tool set if the skill declared a list.
        // For Script mode, ensure `execute_script` stays available — the
        // whole point of that mode is to run a script.
        if let Some(mut tools) = allowed_tools {
            if matches!(mode, SkillExecutionMode::Script)
                && !tools.iter().any(|t| t == "execute_script")
            {
                tools.push("execute_script".to_string());
            }
            if !tools.is_empty() {
                self.pending_skill_tool_scope = Some(tools);
            }
        }

        let label = match mode {
            SkillExecutionMode::Inline => "inline",
            SkillExecutionMode::Subagent => "subagent",
            SkillExecutionMode::Script => "script",
        };
        self.set_status(
            LogLevel::Info,
            format!("Skill '{}' invoked ({})", skill_name_copy, label),
        );
        self.clear_input();
    }

    /// Handle /skills - list all available skills
    pub(super) async fn handle_list_skills(&mut self) {
        use brainwires_skills::SkillSource;

        let result = if let Some(ref registry) = self.skill_registry {
            let skills = registry.list_skills();

            if skills.is_empty() {
                "No skills found.\n\n\
                Skills can be placed in:\n\
                - Personal: ~/.brainwires/skills/\n\
                - Project: .brainwires/skills/\n\n\
                Each skill is a SKILL.md file with YAML frontmatter."
                    .to_string()
            } else {
                let mut output = format!("Available Skills ({} total)\n\n", skills.len());

                // Group by source
                let mut personal: Vec<_> = Vec::new();
                let mut project: Vec<_> = Vec::new();
                let mut builtin: Vec<_> = Vec::new();

                for name in skills {
                    if let Some(meta) = registry.get_metadata(name) {
                        match meta.source {
                            SkillSource::Personal => personal.push(meta),
                            SkillSource::Project => project.push(meta),
                            SkillSource::Builtin => builtin.push(meta),
                        }
                    }
                }

                if !project.is_empty() {
                    output.push_str("**Project Skills:**\n");
                    for meta in &project {
                        output.push_str(&format!(
                            "  - /{} - {}\n",
                            meta.name,
                            truncate_description(&meta.description, 60)
                        ));
                    }
                    output.push('\n');
                }

                if !personal.is_empty() {
                    output.push_str("**Personal Skills:**\n");
                    for meta in &personal {
                        output.push_str(&format!(
                            "  - /{} - {}\n",
                            meta.name,
                            truncate_description(&meta.description, 60)
                        ));
                    }
                    output.push('\n');
                }

                if !builtin.is_empty() {
                    output.push_str("**Builtin Skills:**\n");
                    for meta in &builtin {
                        output.push_str(&format!(
                            "  - /{} - {}\n",
                            meta.name,
                            truncate_description(&meta.description, 60)
                        ));
                    }
                    output.push('\n');
                }

                output
                    .push_str("\nUse /skill:show <name> for details, or /<skill-name> to invoke.");
                output
            }
        } else {
            "Skill registry not initialized".to_string()
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: result,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle /skill:show <name> - show skill details
    pub(super) async fn handle_show_skill(&mut self, name: &str) {
        use brainwires_skills::SkillSource;

        let result = if let Some(ref mut registry) = self.skill_registry {
            // Collect the resource listing first (shared borrow scope) so it
            // doesn't fight with the later `&mut get_skill` borrow.
            let resources_listing = registry
                .get_resources(name)
                .ok()
                .map(|r| {
                    let mut lines = Vec::new();
                    let fmt = |label: &str, paths: &[std::path::PathBuf]| {
                        if paths.is_empty() {
                            None
                        } else {
                            let names: Vec<&str> = paths
                                .iter()
                                .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
                                .collect();
                            Some(format!("  {}/ ({})", label, names.join(", ")))
                        }
                    };
                    if let Some(l) = fmt("scripts", &r.scripts) {
                        lines.push(l);
                    }
                    if let Some(l) = fmt("references", &r.references) {
                        lines.push(l);
                    }
                    if let Some(l) = fmt("assets", &r.assets) {
                        lines.push(l);
                    }
                    if lines.is_empty() {
                        "(none)".to_string()
                    } else {
                        lines.join("\n")
                    }
                })
                .unwrap_or_else(|| "(none)".to_string());

            match registry.get_skill(name) {
                Ok(skill) => {
                    let source_str = match skill.metadata.source {
                        SkillSource::Personal => "Personal (~/.brainwires/skills/)",
                        SkillSource::Project => "Project (.brainwires/skills/)",
                        SkillSource::Builtin => "Builtin",
                    };

                    let allowed_tools = skill
                        .metadata
                        .allowed_tools
                        .as_ref()
                        .map(|tools| tools.join(", "))
                        .unwrap_or_else(|| "all".to_string());

                    let model = skill.metadata.model.as_deref().unwrap_or("default");
                    let license = skill.metadata.license.as_deref().unwrap_or("unspecified");

                    let instructions_preview = if skill.instructions.len() > 500 {
                        format!(
                            "{}...\n\n(truncated, {} chars total)",
                            &skill.instructions[..500],
                            skill.instructions.len()
                        )
                    } else {
                        skill.instructions.clone()
                    };

                    format!(
                        "**Skill: {}**\n\n\
                        **Description:**\n{}\n\n\
                        **Source:** {}\n\
                        **Model:** {}\n\
                        **Allowed Tools:** {}\n\
                        **License:** {}\n\n\
                        **Resources:**\n{}\n\n\
                        **Instructions:**\n{}",
                        name,
                        skill.metadata.description,
                        source_str,
                        model,
                        allowed_tools,
                        license,
                        resources_listing,
                        instructions_preview
                    )
                }
                Err(e) => format!("Failed to load skill '{}': {}", name, e),
            }
        } else {
            "Skill registry not initialized".to_string()
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content: result,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle /skill:reload - reload skills from disk
    pub(super) async fn handle_reload_skills(&mut self) {
        if let Some(ref mut registry) = self.skill_registry {
            match registry.reload() {
                Ok(()) => {
                    let count = registry.list_skills().len();
                    self.add_console_message(format!("Reloaded {} skill(s)", count));
                }
                Err(e) => {
                    self.add_console_message(format!("Failed to reload skills: {}", e));
                }
            }
        } else {
            self.add_console_message("Skill registry not initialized".to_string());
        }

        self.clear_input();
    }

    /// Handle /skill:create <name> [location] - create a new skill
    pub(super) async fn handle_create_skill(&mut self, name: &str, location: Option<&str>) {
        use crate::utils::paths::PlatformPaths;

        // Validate name
        if name.len() > 64 || !name.chars().all(|c| c.is_ascii_lowercase() || c == '-') {
            self.add_console_message(
                "Invalid skill name. Use lowercase letters and hyphens only, max 64 chars."
                    .to_string(),
            );
            return;
        }

        // Determine target directory
        let skills_dir = match location {
            Some("project") | None => {
                // Default to project
                std::env::current_dir()
                    .map(|cwd| cwd.join(".brainwires/skills"))
                    .unwrap_or_else(|_| std::path::PathBuf::from(".brainwires/skills"))
            }
            Some("personal") => match PlatformPaths::personal_skills_dir() {
                Ok(dir) => dir,
                Err(e) => {
                    self.add_console_message(format!("Failed to get personal skills dir: {}", e));
                    return;
                }
            },
            Some(other) => {
                self.add_console_message(format!(
                    "Invalid location: {}. Use 'personal' or 'project'.",
                    other
                ));
                return;
            }
        };

        // Ensure directory exists
        if let Err(e) = std::fs::create_dir_all(&skills_dir) {
            self.add_console_message(format!("Failed to create skills directory: {}", e));
            return;
        }

        // Create the skill file
        let skill_path = skills_dir.join(format!("{}.md", name));

        if skill_path.exists() {
            self.add_console_message(format!(
                "Skill '{}' already exists at: {}",
                name,
                skill_path.display()
            ));
            return;
        }

        let template = format!(
            r#"---
name: {name}
description: |
  Brief description of what this skill does.
  This is used for semantic matching when suggesting skills.
allowed-tools:
  - Read
  - Grep
  - Glob
# model: claude-sonnet  # Optional: override model
# license: MIT
metadata:
  category: utility
  execution: inline  # inline | subagent | script
---

# {name} Skill Instructions

Write your skill instructions here. These will be injected into the
conversation when the skill is invoked.

## Example Usage

Describe how to use this skill and what it does.
"#,
            name = name
        );

        match std::fs::write(&skill_path, template) {
            Ok(()) => {
                self.add_console_message(format!(
                    "Created new skill at:\n{}\n\nEdit the file to customize your skill, then use /skill:reload to load it.",
                    skill_path.display()
                ));

                // Reload to pick up the new skill
                if let Some(ref mut registry) = self.skill_registry {
                    let _ = registry.reload();
                }
            }
            Err(e) => {
                self.add_console_message(format!("Failed to create skill file: {}", e));
            }
        }

        self.clear_input();
    }
}

/// Truncate a description to max length, adding ellipsis if needed.
fn truncate_description(s: &str, max_len: usize) -> String {
    let first_line = s.lines().next().unwrap_or(s);
    if first_line.len() > max_len {
        format!("{}...", &first_line[..max_len.saturating_sub(3)])
    } else {
        first_line.to_string()
    }
}
