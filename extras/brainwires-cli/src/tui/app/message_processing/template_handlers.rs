//! Template Handlers
//!
//! Handles template management command operations.

use super::super::state::{App, LogLevel, TuiMessage};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

impl App {
    /// Handle list templates command
    pub(super) async fn handle_list_templates(&mut self) {
        use crate::storage::TemplateStore;

        let content = match TemplateStore::with_default_dir() {
            Ok(store) => match store.list() {
                Ok(templates) => {
                    if templates.is_empty() {
                        "No templates saved.\n\n\
                             Save a template with: /template:save <name> [description]\n\
                             (Requires an active plan)"
                            .to_string()
                    } else {
                        let mut lines = vec!["Templates:".to_string(), "".to_string()];
                        for template in &templates {
                            let vars_info = if template.variables.is_empty() {
                                String::new()
                            } else {
                                format!(" [vars: {}]", template.variables.join(", "))
                            };
                            let usage_info = if template.usage_count > 0 {
                                format!(
                                    " (used {} time{})",
                                    template.usage_count,
                                    if template.usage_count == 1 { "" } else { "s" }
                                )
                            } else {
                                String::new()
                            };
                            lines.push(format!(
                                "  {} - {}{}{}",
                                template.name, template.description, vars_info, usage_info
                            ));
                            lines.push(format!("    ID: {}", &template.template_id[..8]));
                        }
                        lines.push("".to_string());
                        lines
                            .push("Use: /template:show <name> or /template:use <name>".to_string());
                        lines.join("\n")
                    }
                }
                Err(e) => format!("Failed to list templates: {}", e),
            },
            Err(e) => format!("Failed to access template store: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle save template command
    pub(super) async fn handle_save_template(
        &mut self,
        name: String,
        description: Option<String>,
    ) -> Result<()> {
        use crate::storage::{PlanTemplate, TemplateStore};

        let content = if let Some(ref plan) = self.active_plan {
            match TemplateStore::with_default_dir() {
                Ok(store) => {
                    let desc = description.unwrap_or_else(|| {
                        format!(
                            "Template from plan: {}",
                            &plan.task_description[..50.min(plan.task_description.len())]
                        )
                    });
                    let template = PlanTemplate::from_plan(
                        name.clone(),
                        desc,
                        plan.plan_content.clone(),
                        plan.plan_id.clone(),
                    );

                    match store.save(&template) {
                        Ok(()) => {
                            let vars_info = if template.variables.is_empty() {
                                "No variables detected.".to_string()
                            } else {
                                format!("Variables: {}", template.variables.join(", "))
                            };
                            format!(
                                "Template saved: {}\n\
                                 ID: {}\n\
                                 {}\n\n\
                                 Use it with: /template:use {}",
                                name,
                                &template.template_id[..8],
                                vars_info,
                                name
                            )
                        }
                        Err(e) => format!("Failed to save template: {}", e),
                    }
                }
                Err(e) => format!("Failed to access template store: {}", e),
            }
        } else {
            "No active plan. Activate a plan first with /plan:activate <id>".to_string()
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
        Ok(())
    }

    /// Handle show template command
    pub(super) async fn handle_show_template(&mut self, name: String) {
        use crate::storage::TemplateStore;

        let content = match TemplateStore::with_default_dir() {
            Ok(store) => match store.get_by_name(&name) {
                Ok(Some(template)) => {
                    let vars_info = if template.variables.is_empty() {
                        "None".to_string()
                    } else {
                        template.variables.join(", ")
                    };
                    let category_info = template.category.as_deref().unwrap_or("None");
                    let tags_info = if template.tags.is_empty() {
                        "None".to_string()
                    } else {
                        template.tags.join(", ")
                    };

                    format!(
                        "Template: {}\n\
                             ID: {}\n\
                             Description: {}\n\
                             Category: {}\n\
                             Tags: {}\n\
                             Variables: {}\n\
                             Used: {} time{}\n\n\
                             ---\n\n\
                             {}",
                        template.name,
                        template.template_id,
                        template.description,
                        category_info,
                        tags_info,
                        vars_info,
                        template.usage_count,
                        if template.usage_count == 1 { "" } else { "s" },
                        template.content
                    )
                }
                Ok(None) => format!("Template not found: {}", name),
                Err(e) => format!("Failed to load template: {}", e),
            },
            Err(e) => format!("Failed to access template store: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle use template command - instantiate and create a new plan
    pub(super) async fn handle_use_template(
        &mut self,
        name: String,
        vars: Vec<String>,
    ) -> Result<()> {
        use crate::config::PlatformPaths;
        use crate::storage::{
            CachedEmbeddingProvider, LanceDatabase, PlanStore, TemplateStore, VectorDatabase,
        };
        use crate::types::plan::{PlanMetadata, PlanStatus};
        use crate::utils::plan_parser::{parse_plan_steps, steps_to_tasks};

        let content = match TemplateStore::with_default_dir() {
            Ok(store) => {
                match store.get_by_name(&name) {
                    Ok(Some(mut template)) => {
                        // Parse variable substitutions
                        let mut substitutions = HashMap::new();
                        for var_str in vars {
                            if let Some((key, value)) = var_str.split_once('=') {
                                substitutions.insert(key.to_string(), value.to_string());
                            }
                        }

                        // Check for missing variables
                        let missing: Vec<_> = template
                            .variables
                            .iter()
                            .filter(|v| !substitutions.contains_key(*v))
                            .collect();

                        if !missing.is_empty() {
                            format!(
                                "Missing variables: {}\n\n\
                                 Usage: /template:use {} {}\n\n\
                                 Required: {}",
                                missing
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", "),
                                name,
                                template
                                    .variables
                                    .iter()
                                    .map(|v| format!("{}=value", v))
                                    .collect::<Vec<_>>()
                                    .join(" "),
                                template.variables.join(", ")
                            )
                        } else {
                            // Instantiate template
                            let plan_content = template.instantiate(&substitutions);

                            // Mark template as used
                            template.mark_used();
                            let _ = store.save(&template);

                            // Create a new plan from the template
                            let db_path = PlatformPaths::conversations_db_path()?;
                            let client = Arc::new(
                                LanceDatabase::new(
                                    db_path
                                        .to_str()
                                        .ok_or_else(|| anyhow::anyhow!("Invalid DB path"))?,
                                )
                                .await?,
                            );
                            let embeddings = Arc::new(CachedEmbeddingProvider::new()?);
                            client.initialize(embeddings.dimension()).await?;
                            let plan_store = PlanStore::new(client.clone(), embeddings);

                            // Create plan metadata
                            let plan = PlanMetadata::new(
                                format!("From template: {}", template.name),
                                plan_content.clone(),
                                self.session_id.clone(),
                            );

                            // Save to store
                            plan_store.save(&plan).await?;

                            // Parse and load tasks
                            let steps = parse_plan_steps(&plan_content);
                            let tasks = steps_to_tasks(&steps, &plan.plan_id);

                            {
                                let task_mgr = self.task_manager.write().await;
                                task_mgr.clear().await;
                                task_mgr.load_tasks(tasks.clone()).await;
                            }

                            for task in &tasks {
                                let _ = self.task_store.save(task, &self.session_id).await;
                            }

                            self.update_task_cache().await;

                            // Activate the plan
                            let mut active_plan = plan.clone();
                            active_plan.set_status(PlanStatus::Active);

                            self.active_plan = Some(active_plan);
                            self.completed_plan_steps.clear();
                            self.set_status(LogLevel::Info, "Plan active");

                            format!(
                                "Plan created from template '{}':\n\n\
                                 Plan ID: {}\n\
                                 Tasks: {} created\n\n\
                                 The plan is now active. Use /tasks to see the task list.",
                                template.name,
                                &plan.plan_id[..8],
                                tasks.len()
                            )
                        }
                    }
                    Ok(None) => format!("Template not found: {}", name),
                    Err(e) => format!("Failed to load template: {}", e),
                }
            }
            Err(e) => format!("Failed to access template store: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
        Ok(())
    }

    /// Handle delete template command
    pub(super) async fn handle_delete_template(&mut self, name: String) {
        use crate::storage::TemplateStore;

        let content = match TemplateStore::with_default_dir() {
            Ok(store) => {
                // First find the template to get its ID
                match store.get_by_name(&name) {
                    Ok(Some(template)) => match store.delete(&template.template_id) {
                        Ok(true) => format!("Template '{}' deleted.", template.name),
                        Ok(false) => format!("Template '{}' not found.", name),
                        Err(e) => format!("Failed to delete template: {}", e),
                    },
                    Ok(None) => format!("Template not found: {}", name),
                    Err(e) => format!("Failed to find template: {}", e),
                }
            }
            Err(e) => format!("Failed to access template store: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }
}
