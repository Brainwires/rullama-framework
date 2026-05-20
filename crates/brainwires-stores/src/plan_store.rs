//! Plan Store - Persists execution plans with conversation association
//!
//! Plans are stored via a [`StorageBackend`](brainwires_storage::databases::StorageBackend) for querying and linked to conversations.
//! They can also be exported as Markdown files for human readability.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;

use brainwires_core::{PlanMetadata, PlanStatus};
use brainwires_storage::databases::{
    FieldDef, FieldType, FieldValue, Filter, Record, StorageBackend, record_get,
};
use brainwires_storage::embeddings::CachedEmbeddingProvider;

const TABLE_NAME: &str = "plans";

// ── Schema ──────────────────────────────────────────────────────────────

/// Return the backend-agnostic field definitions for the plans table.
pub fn plans_field_defs(embedding_dim: usize) -> Vec<FieldDef> {
    vec![
        FieldDef::required("plan_id", FieldType::Utf8),
        FieldDef::required("conversation_id", FieldType::Utf8),
        FieldDef::required("title", FieldType::Utf8),
        FieldDef::required("task_description", FieldType::Utf8),
        FieldDef::required("plan_content", FieldType::Utf8),
        FieldDef::optional("model_id", FieldType::Utf8),
        FieldDef::required("status", FieldType::Utf8),
        FieldDef::required("executed", FieldType::Boolean),
        FieldDef::required("iterations_used", FieldType::Int32),
        FieldDef::required("created_at", FieldType::Int64),
        FieldDef::required("updated_at", FieldType::Int64),
        FieldDef::optional("file_path", FieldType::Utf8),
        // Branching fields
        FieldDef::optional("parent_plan_id", FieldType::Utf8),
        FieldDef::optional("child_plan_ids", FieldType::Utf8), // JSON array
        FieldDef::optional("branch_name", FieldType::Utf8),
        FieldDef::required("merged", FieldType::Boolean),
        FieldDef::required("depth", FieldType::Int32),
        // Embedding vector
        FieldDef::optional("embedding", FieldType::Vector(embedding_dim)),
    ]
}

/// Arrow schema for the plans table, used by `LanceDatabase` table creation.
pub fn plans_schema() -> std::sync::Arc<arrow_schema::Schema> {
    use arrow_schema::{DataType, Field};

    std::sync::Arc::new(arrow_schema::Schema::new(vec![
        Field::new("plan_id", DataType::Utf8, false),
        Field::new("conversation_id", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("task_description", DataType::Utf8, false),
        Field::new("plan_content", DataType::Utf8, false),
        Field::new("model_id", DataType::Utf8, true),
        Field::new("status", DataType::Utf8, false),
        Field::new("executed", DataType::Boolean, false),
        Field::new("iterations_used", DataType::Int32, false),
        Field::new("created_at", DataType::Int64, false),
        Field::new("updated_at", DataType::Int64, false),
        Field::new("file_path", DataType::Utf8, true),
        // Branching fields
        Field::new("parent_plan_id", DataType::Utf8, true),
        Field::new("child_plan_ids", DataType::Utf8, true),
        Field::new("branch_name", DataType::Utf8, true),
        Field::new("merged", DataType::Boolean, false),
        Field::new("depth", DataType::Int32, false),
    ]))
}

// ── Record conversion helpers ───────────────────────────────────────────

fn to_record(plan: &PlanMetadata) -> Record {
    let child_plan_ids_json = serde_json::to_string(&plan.child_plan_ids).unwrap_or_default();

    vec![
        (
            "plan_id".into(),
            FieldValue::Utf8(Some(plan.plan_id.clone())),
        ),
        (
            "conversation_id".into(),
            FieldValue::Utf8(Some(plan.conversation_id.clone())),
        ),
        ("title".into(), FieldValue::Utf8(Some(plan.title.clone()))),
        (
            "task_description".into(),
            FieldValue::Utf8(Some(plan.task_description.clone())),
        ),
        (
            "plan_content".into(),
            FieldValue::Utf8(Some(plan.plan_content.clone())),
        ),
        ("model_id".into(), FieldValue::Utf8(plan.model_id.clone())),
        (
            "status".into(),
            FieldValue::Utf8(Some(plan.status.to_string())),
        ),
        ("executed".into(), FieldValue::Boolean(Some(plan.executed))),
        (
            "iterations_used".into(),
            FieldValue::Int32(Some(plan.iterations_used as i32)),
        ),
        (
            "created_at".into(),
            FieldValue::Int64(Some(plan.created_at)),
        ),
        (
            "updated_at".into(),
            FieldValue::Int64(Some(plan.updated_at)),
        ),
        ("file_path".into(), FieldValue::Utf8(plan.file_path.clone())),
        (
            "parent_plan_id".into(),
            FieldValue::Utf8(plan.parent_plan_id.clone()),
        ),
        (
            "child_plan_ids".into(),
            FieldValue::Utf8(Some(child_plan_ids_json)),
        ),
        (
            "branch_name".into(),
            FieldValue::Utf8(plan.branch_name.clone()),
        ),
        ("merged".into(), FieldValue::Boolean(Some(plan.merged))),
        ("depth".into(), FieldValue::Int32(Some(plan.depth as i32))),
        (
            "embedding".into(),
            FieldValue::Vector(plan.embedding.clone().unwrap_or_default()),
        ),
    ]
}

fn from_record(r: &Record) -> Result<PlanMetadata> {
    let status_str = record_get(r, "status")
        .and_then(|v| v.as_str())
        .unwrap_or("draft");
    let status = status_str.parse::<PlanStatus>().unwrap_or_default();

    let child_plan_ids: Vec<String> = record_get(r, "child_plan_ids")
        .and_then(|v| v.as_str())
        .and_then(|json| serde_json::from_str(json).ok())
        .unwrap_or_default();

    let embedding = record_get(r, "embedding")
        .and_then(|v| v.as_vector())
        .map(|v| v.to_vec());

    Ok(PlanMetadata {
        plan_id: record_get(r, "plan_id")
            .and_then(|v| v.as_str())
            .context("missing plan_id")?
            .to_string(),
        conversation_id: record_get(r, "conversation_id")
            .and_then(|v| v.as_str())
            .context("missing conversation_id")?
            .to_string(),
        title: record_get(r, "title")
            .and_then(|v| v.as_str())
            .context("missing title")?
            .to_string(),
        task_description: record_get(r, "task_description")
            .and_then(|v| v.as_str())
            .context("missing task_description")?
            .to_string(),
        plan_content: record_get(r, "plan_content")
            .and_then(|v| v.as_str())
            .context("missing plan_content")?
            .to_string(),
        model_id: record_get(r, "model_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        status,
        executed: record_get(r, "executed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        iterations_used: record_get(r, "iterations_used")
            .and_then(|v| v.as_i32())
            .unwrap_or(0) as u32,
        created_at: record_get(r, "created_at")
            .and_then(|v| v.as_i64())
            .context("missing created_at")?,
        updated_at: record_get(r, "updated_at")
            .and_then(|v| v.as_i64())
            .context("missing updated_at")?,
        file_path: record_get(r, "file_path")
            .and_then(|v| v.as_str())
            .map(String::from),
        embedding,
        parent_plan_id: record_get(r, "parent_plan_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        child_plan_ids,
        branch_name: record_get(r, "branch_name")
            .and_then(|v| v.as_str())
            .map(String::from),
        merged: record_get(r, "merged")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        depth: record_get(r, "depth").and_then(|v| v.as_i32()).unwrap_or(0) as u32,
    })
}

// ── PlanStore ───────────────────────────────────────────────────────────

/// Store for managing execution plans
pub struct PlanStore<B: StorageBackend = brainwires_storage::databases::lance::LanceDatabase> {
    backend: Arc<B>,
    embeddings: Arc<CachedEmbeddingProvider>,
    /// Directory for plan markdown exports
    plans_dir: Option<PathBuf>,
}

impl<B: StorageBackend> PlanStore<B> {
    /// Create a new plan store
    pub fn new(backend: Arc<B>, embeddings: Arc<CachedEmbeddingProvider>) -> Self {
        Self {
            backend,
            embeddings,
            plans_dir: None,
        }
    }

    /// Create a plan store with a plans directory for markdown exports
    pub fn with_plans_dir(
        backend: Arc<B>,
        embeddings: Arc<CachedEmbeddingProvider>,
        plans_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            backend,
            embeddings,
            plans_dir: Some(plans_dir.into()),
        }
    }

    /// Ensure the underlying table exists.
    pub async fn ensure_table(&self) -> Result<()> {
        let dim = self.embeddings.dimension();
        self.backend
            .ensure_table(TABLE_NAME, &plans_field_defs(dim))
            .await
    }

    /// Save a plan (create or update)
    pub async fn save(&self, plan: &PlanMetadata) -> Result<()> {
        // Delete existing plan with same ID (if any)
        let _ = self.delete(&plan.plan_id).await;

        // Generate embedding from the plan content if not already present
        let mut plan = plan.clone();
        if plan.embedding.is_none() {
            let text = format!("{} {}", plan.title, plan.task_description);
            plan.embedding = Some(self.embeddings.embed(&text)?);
        }

        self.backend
            .insert(TABLE_NAME, vec![to_record(&plan)])
            .await
            .context("Failed to save plan")?;

        Ok(())
    }

    /// Get a plan by ID
    pub async fn get(&self, plan_id: &str) -> Result<Option<PlanMetadata>> {
        let filter = Filter::Eq(
            "plan_id".into(),
            FieldValue::Utf8(Some(plan_id.to_string())),
        );
        let records = self
            .backend
            .query(TABLE_NAME, Some(&filter), Some(1))
            .await?;

        match records.first() {
            Some(r) => Ok(Some(from_record(r)?)),
            None => Ok(None),
        }
    }

    /// Get all plans for a conversation
    pub async fn get_by_conversation(&self, conversation_id: &str) -> Result<Vec<PlanMetadata>> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        let records = self.backend.query(TABLE_NAME, Some(&filter), None).await?;

        let mut plans: Vec<PlanMetadata> =
            records.iter().filter_map(|r| from_record(r).ok()).collect();

        // Sort by created_at descending (newest first)
        plans.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(plans)
    }

    /// List recent plans across all conversations
    pub async fn list_recent(&self, limit: usize) -> Result<Vec<PlanMetadata>> {
        // Fetch more than needed so we can sort and truncate
        let records = self
            .backend
            .query(TABLE_NAME, None, Some(limit * 2))
            .await?;

        let mut plans: Vec<PlanMetadata> =
            records.iter().filter_map(|r| from_record(r).ok()).collect();

        // Sort by created_at descending and take limit
        plans.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        plans.truncate(limit);

        Ok(plans)
    }

    /// Update an existing plan
    pub async fn update(&self, plan: &PlanMetadata) -> Result<()> {
        self.save(plan).await
    }

    /// Delete a plan by ID
    pub async fn delete(&self, plan_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "plan_id".into(),
            FieldValue::Utf8(Some(plan_id.to_string())),
        );
        self.backend
            .delete(TABLE_NAME, &filter)
            .await
            .context("Failed to delete plan")?;
        Ok(())
    }

    /// Delete all plans for a conversation
    pub async fn delete_by_conversation(&self, conversation_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        self.backend
            .delete(TABLE_NAME, &filter)
            .await
            .context("Failed to delete plans for conversation")?;
        Ok(())
    }

    /// Search plans by semantic similarity to a query string
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<PlanMetadata>> {
        let query_embedding = self.embeddings.embed(query)?;

        let scored = self
            .backend
            .vector_search(TABLE_NAME, "embedding", query_embedding, limit, None)
            .await?;

        let plans: Vec<PlanMetadata> = scored
            .iter()
            .filter_map(|sr| from_record(&sr.record).ok())
            .collect();

        Ok(plans)
    }

    /// Export a plan to a markdown file
    ///
    /// Requires `plans_dir` to be set via `with_plans_dir()`.
    /// Returns the path to the created file.
    pub async fn export_to_markdown(&self, plan_id: &str) -> Result<PathBuf> {
        let plans_dir = self.plans_dir.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Plans directory not configured; use with_plans_dir()")
        })?;

        let plan = self
            .get(plan_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Plan not found: {}", plan_id))?;

        // Ensure plans directory exists
        std::fs::create_dir_all(plans_dir)?;

        // Get file path
        let file_path = plans_dir.join(format!("{}.md", plan_id));

        // Generate markdown and write to file
        let markdown = plan.to_markdown();
        std::fs::write(&file_path, markdown)
            .with_context(|| format!("Failed to write plan to {}", file_path.display()))?;

        Ok(file_path)
    }

    /// Save a plan and export to markdown in one operation
    pub async fn save_and_export(&self, plan: &mut PlanMetadata) -> Result<PathBuf> {
        let plans_dir = self.plans_dir.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Plans directory not configured; use with_plans_dir()")
        })?;

        // Export to markdown first
        std::fs::create_dir_all(plans_dir)?;
        let file_path = plans_dir.join(format!("{}.md", &plan.plan_id));
        let markdown = plan.to_markdown();
        std::fs::write(&file_path, &markdown)
            .with_context(|| format!("Failed to write plan to {}", file_path.display()))?;

        // Update file_path in plan
        plan.set_file_path(file_path.to_string_lossy().to_string());

        // Save to database
        self.save(plan).await?;

        Ok(file_path)
    }

    /// Load a plan from its markdown file (useful for editing)
    pub fn load_from_markdown(file_path: &std::path::Path) -> Result<String> {
        std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read plan from {}", file_path.display()))
    }

    /// Get all child plans (sub-plans/branches) of a plan
    pub async fn get_children(&self, plan_id: &str) -> Result<Vec<PlanMetadata>> {
        let filter = Filter::Eq(
            "parent_plan_id".into(),
            FieldValue::Utf8(Some(plan_id.to_string())),
        );
        let records = self.backend.query(TABLE_NAME, Some(&filter), None).await?;

        let mut plans: Vec<PlanMetadata> =
            records.iter().filter_map(|r| from_record(r).ok()).collect();

        // Sort by created_at ascending
        plans.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        Ok(plans)
    }

    /// Get the full plan hierarchy (parent and all descendants)
    pub async fn get_hierarchy(&self, plan_id: &str) -> Result<Vec<PlanMetadata>> {
        let mut hierarchy = Vec::new();

        // Get the root plan
        if let Some(root) = self.get(plan_id).await? {
            hierarchy.push(root.clone());

            // Recursively get children
            self.collect_descendants(plan_id, &mut hierarchy).await?;
        }

        Ok(hierarchy)
    }

    /// Recursively collect all descendants
    async fn collect_descendants(
        &self,
        plan_id: &str,
        hierarchy: &mut Vec<PlanMetadata>,
    ) -> Result<()> {
        let children = self.get_children(plan_id).await?;
        for child in children {
            let child_id = child.plan_id.clone();
            hierarchy.push(child);
            // Recursively get children of this child (with depth limit)
            if hierarchy.len() < 100 {
                Box::pin(self.collect_descendants(&child_id, hierarchy)).await?;
            }
        }
        Ok(())
    }
}
