//! Plan Hierarchy Management
//!
//! Handles plan hierarchy operations: search, branch, merge, tree.

use super::super::state::{App, TuiMessage};
use crate::types::plan::PlanStatus;
use anyhow::Result;
use std::sync::Arc;

impl App {
    /// Handle search plans command
    pub(super) async fn handle_search_plans(&mut self, query: String) {
        use crate::config::PlatformPaths;
        use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};

        let content = match PlatformPaths::conversations_db_path() {
            Ok(db_path) => {
                if let Some(db_str) = db_path.to_str() {
                    match LanceDatabase::new(db_str).await {
                        Ok(client) => {
                            let client = Arc::new(client);
                            let embeddings = match CachedEmbeddingProvider::new() {
                                Ok(e) => Arc::new(e),
                                Err(_e) => {
                                    return;
                                }
                            };
                            let _ = client.initialize(embeddings.dimension()).await;
                            let plan_store = PlanStore::new(client, embeddings);

                            match plan_store.search(&query, 20).await {
                                Ok(plans) => {
                                    if plans.is_empty() {
                                        format!("No plans found matching '{}'", query)
                                    } else {
                                        let mut lines = vec![
                                            format!("Search results for '{}':", query),
                                            "".to_string(),
                                        ];
                                        for plan in &plans {
                                            let created = chrono::DateTime::from_timestamp(
                                                plan.created_at,
                                                0,
                                            )
                                            .map(|dt| dt.format("%Y-%m-%d").to_string())
                                            .unwrap_or_else(|| "Unknown".to_string());

                                            lines.push(format!(
                                                "  {} - {} [{}]",
                                                &plan.plan_id[..8],
                                                if plan.title.len() > 40 {
                                                    format!("{}...", &plan.title[..37])
                                                } else {
                                                    plan.title.clone()
                                                },
                                                plan.status
                                            ));
                                            lines.push(format!(
                                                "    Created: {} | Task: {}...",
                                                created,
                                                &plan.task_description
                                                    [..50.min(plan.task_description.len())]
                                            ));
                                        }
                                        lines.push("".to_string());
                                        lines.push(
                                            "Use /plan:show <id> to view details".to_string(),
                                        );
                                        lines.join("\n")
                                    }
                                }
                                Err(e) => format!("Search failed: {}", e),
                            }
                        }
                        Err(e) => format!("Failed to access database: {}", e),
                    }
                } else {
                    "Invalid database path".to_string()
                }
            }
            Err(e) => format!("Failed to get database path: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }

    /// Handle branch plan command
    pub(super) async fn handle_branch_plan(
        &mut self,
        branch_name: String,
        task_description: String,
    ) -> Result<()> {
        use crate::config::PlatformPaths;
        use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};

        // Clone active plan info upfront to avoid borrow issues
        let parent_info = self.active_plan.as_ref().map(|p| {
            let branch = p.create_branch(
                branch_name.clone(),
                task_description.clone(),
                format!(
                    "Sub-plan for: {}\n\n[Plan content to be generated]",
                    task_description
                ),
            );
            let mut updated_parent = p.clone();
            updated_parent.add_child(branch.plan_id.clone());
            let parent_id_short = p.plan_id[..8].to_string();
            (branch, updated_parent, parent_id_short)
        });

        let (content, new_active_plan) =
            if let Some((branch, updated_parent, parent_id_short)) = parent_info {
                // Save the branch

                match PlatformPaths::conversations_db_path() {
                    Ok(db_path) => {
                        if let Some(db_str) = db_path.to_str() {
                            match LanceDatabase::new(db_str).await {
                                Ok(client) => {
                                    let client = Arc::new(client);
                                    let embeddings = match CachedEmbeddingProvider::new() {
                                        Ok(e) => Arc::new(e),
                                        Err(e) => {
                                            return Err(anyhow::anyhow!(
                                                "Failed to create embedding provider: {}",
                                                e
                                            ));
                                        }
                                    };
                                    let _ = client.initialize(embeddings.dimension()).await;
                                    let plan_store = PlanStore::new(client, embeddings);

                                    // Save both
                                    if let Err(e) = plan_store.save(&branch).await {
                                        (format!("Failed to save branch: {}", e), None)
                                    } else if let Err(e) = plan_store.save(&updated_parent).await {
                                        (format!("Failed to update parent plan: {}", e), None)
                                    } else {
                                        let msg = format!(
                                            "Branch created: {}\n\n\
                                         Branch ID: {}\n\
                                         Parent: {}\n\
                                         Depth: {}\n\n\
                                         Use /plan:activate {} to work on the branch.\n\
                                         Use /plan:merge to merge back when done.",
                                            branch_name,
                                            &branch.plan_id[..8],
                                            parent_id_short,
                                            branch.depth,
                                            &branch.plan_id[..8]
                                        );
                                        (msg, Some(updated_parent))
                                    }
                                }
                                Err(e) => (format!("Failed to access database: {}", e), None),
                            }
                        } else {
                            ("Invalid database path".to_string(), None)
                        }
                    }
                    Err(e) => (format!("Failed to get database path: {}", e), None),
                }
            } else {
                (
                    "No active plan. Use /plan:activate <id> to activate a plan first.".to_string(),
                    None,
                )
            };

        // Update active plan after the borrow ends
        if let Some(plan) = new_active_plan {
            self.active_plan = Some(plan);
        }

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
        Ok(())
    }

    /// Handle merge plan command
    pub(super) async fn handle_merge_plan(&mut self, plan_id: Option<String>) -> Result<()> {
        use crate::config::PlatformPaths;
        use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};

        let content = match PlatformPaths::conversations_db_path() {
            Ok(db_path) => {
                if let Some(db_str) = db_path.to_str() {
                    match LanceDatabase::new(db_str).await {
                        Ok(client) => {
                            let client = Arc::new(client);
                            let embeddings = match CachedEmbeddingProvider::new() {
                                Ok(e) => Arc::new(e),
                                Err(e) => {
                                    return Err(anyhow::anyhow!(
                                        "Failed to create embedding provider: {}",
                                        e
                                    ));
                                }
                            };
                            let _ = client.initialize(embeddings.dimension()).await;
                            let plan_store = PlanStore::new(client, embeddings);

                            // Get the plan to merge (active plan or specified ID)
                            let plan_to_merge = if let Some(id) = plan_id {
                                plan_store.get(&id).await.ok().flatten()
                            } else {
                                self.active_plan.clone()
                            };

                            if let Some(mut plan) = plan_to_merge {
                                if plan.parent_plan_id.is_none() {
                                    "Cannot merge a root plan. Only branch plans can be merged."
                                        .to_string()
                                } else if plan.merged {
                                    "Plan is already merged.".to_string()
                                } else {
                                    plan.mark_merged();
                                    if let Err(e) = plan_store.save(&plan).await {
                                        format!("Failed to save merged plan: {}", e)
                                    } else {
                                        format!(
                                            "Branch '{}' merged.\n\n\
                                             Plan ID: {}\n\
                                             Status: completed\n\n\
                                             The branch is now marked as merged back to parent.",
                                            plan.branch_name.as_deref().unwrap_or("unnamed"),
                                            &plan.plan_id[..8]
                                        )
                                    }
                                }
                            } else {
                                "No plan to merge. Specify a plan ID or activate a branch plan."
                                    .to_string()
                            }
                        }
                        Err(e) => format!("Failed to access database: {}", e),
                    }
                } else {
                    "Invalid database path".to_string()
                }
            }
            Err(e) => format!("Failed to get database path: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
        Ok(())
    }

    /// Handle plan tree command
    pub(super) async fn handle_plan_tree(&mut self, plan_id: Option<String>) {
        use crate::config::PlatformPaths;
        use crate::storage::{CachedEmbeddingProvider, LanceDatabase, PlanStore, VectorDatabase};

        let content = match PlatformPaths::conversations_db_path() {
            Ok(db_path) => {
                if let Some(db_str) = db_path.to_str() {
                    match LanceDatabase::new(db_str).await {
                        Ok(client) => {
                            let client = Arc::new(client);
                            let embeddings = match CachedEmbeddingProvider::new() {
                                Ok(e) => Arc::new(e),
                                Err(_e) => {
                                    return;
                                }
                            };
                            let _ = client.initialize(embeddings.dimension()).await;
                            let plan_store = PlanStore::new(client, embeddings);

                            // Get the root plan ID
                            let root_id = if let Some(id) = plan_id {
                                id
                            } else if let Some(ref plan) = self.active_plan {
                                // Walk up to find root
                                let mut current = plan.clone();
                                while let Some(parent_id) = &current.parent_plan_id {
                                    if let Ok(Some(parent)) = plan_store.get(parent_id).await {
                                        current = parent;
                                    } else {
                                        break;
                                    }
                                }
                                current.plan_id.clone()
                            } else {
                                return {
                                    self.messages.push(TuiMessage {
                                        role: "system".to_string(),
                                        content: "No plan specified and no active plan."
                                            .to_string(),
                                        created_at: chrono::Utc::now().timestamp(),
                                    });
                                    self.clear_input();
                                };
                            };

                            // Get hierarchy
                            match plan_store.get_hierarchy(&root_id).await {
                                Ok(plans) => {
                                    if plans.is_empty() {
                                        "Plan not found.".to_string()
                                    } else {
                                        let mut lines =
                                            vec!["Plan Hierarchy:".to_string(), "".to_string()];

                                        for plan in &plans {
                                            let indent = "  ".repeat(plan.depth as usize);
                                            let branch_label =
                                                plan.branch_name.as_deref().unwrap_or("");
                                            let status_icon = if plan.merged {
                                                "✓"
                                            } else if plan.status == PlanStatus::Active {
                                                "▶"
                                            } else {
                                                "○"
                                            };

                                            let title_preview = if plan.title.len() > 30 {
                                                format!("{}...", &plan.title[..27])
                                            } else {
                                                plan.title.clone()
                                            };

                                            if plan.depth == 0 {
                                                lines.push(format!(
                                                    "{}{} {} [{}]",
                                                    indent, status_icon, title_preview, plan.status
                                                ));
                                            } else {
                                                lines.push(format!(
                                                    "{}└─ {} {} ({}) [{}]",
                                                    indent,
                                                    status_icon,
                                                    branch_label,
                                                    &plan.plan_id[..6],
                                                    plan.status
                                                ));
                                            }
                                        }
                                        lines.push("".to_string());
                                        lines.push(
                                            "Legend: ▶ active  ✓ merged  ○ other".to_string(),
                                        );
                                        lines.join("\n")
                                    }
                                }
                                Err(e) => format!("Failed to get hierarchy: {}", e),
                            }
                        }
                        Err(e) => format!("Failed to access database: {}", e),
                    }
                } else {
                    "Invalid database path".to_string()
                }
            }
            Err(e) => format!("Failed to get database path: {}", e),
        };

        self.messages.push(TuiMessage {
            role: "system".to_string(),
            content,
            created_at: chrono::Utc::now().timestamp(),
        });
        self.clear_input();
    }
}
