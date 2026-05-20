//! Task Store - Persists tasks via a backend-agnostic storage layer.
//!
//! Also includes agent state persistence for background task agents.

use anyhow::{Context, Result};
use std::sync::Arc;

use brainwires_core::{Task, TaskPriority, TaskStatus};
use brainwires_storage::databases::{
    FieldDef, FieldType, FieldValue, Filter, Record, StorageBackend, record_get,
};

const TASK_TABLE: &str = "tasks";
const AGENT_STATE_TABLE: &str = "agent_states";

// ── Schema helpers ──────────────────────────────────────────────────────

fn tasks_field_defs() -> Vec<FieldDef> {
    vec![
        FieldDef::required("task_id", FieldType::Utf8),
        FieldDef::required("conversation_id", FieldType::Utf8),
        FieldDef::optional("plan_id", FieldType::Utf8),
        FieldDef::required("description", FieldType::Utf8),
        FieldDef::required("status", FieldType::Utf8),
        FieldDef::optional("parent_id", FieldType::Utf8),
        FieldDef::required("children", FieldType::Utf8), // JSON array
        FieldDef::required("depends_on", FieldType::Utf8), // JSON array
        FieldDef::required("priority", FieldType::Utf8),
        FieldDef::optional("assigned_to", FieldType::Utf8),
        FieldDef::required("iterations", FieldType::Int32),
        FieldDef::optional("summary", FieldType::Utf8),
        FieldDef::required("created_at", FieldType::Int64),
        FieldDef::required("updated_at", FieldType::Int64),
        FieldDef::optional("started_at", FieldType::Int64),
        FieldDef::optional("completed_at", FieldType::Int64),
    ]
}

fn agent_states_field_defs() -> Vec<FieldDef> {
    vec![
        FieldDef::required("agent_id", FieldType::Utf8),
        FieldDef::required("task_id", FieldType::Utf8),
        FieldDef::required("conversation_id", FieldType::Utf8),
        FieldDef::required("status", FieldType::Utf8),
        FieldDef::required("iteration", FieldType::Int32),
        FieldDef::required("context_json", FieldType::Utf8),
        FieldDef::required("created_at", FieldType::Int64),
        FieldDef::required("updated_at", FieldType::Int64),
    ]
}

// ── Record conversion helpers ───────────────────────────────────────────

fn task_to_record(m: &TaskMetadata) -> Record {
    vec![
        ("task_id".into(), FieldValue::Utf8(Some(m.task_id.clone()))),
        (
            "conversation_id".into(),
            FieldValue::Utf8(Some(m.conversation_id.clone())),
        ),
        ("plan_id".into(), FieldValue::Utf8(m.plan_id.clone())),
        (
            "description".into(),
            FieldValue::Utf8(Some(m.description.clone())),
        ),
        ("status".into(), FieldValue::Utf8(Some(m.status.clone()))),
        ("parent_id".into(), FieldValue::Utf8(m.parent_id.clone())),
        (
            "children".into(),
            FieldValue::Utf8(Some(m.children.clone())),
        ),
        (
            "depends_on".into(),
            FieldValue::Utf8(Some(m.depends_on.clone())),
        ),
        (
            "priority".into(),
            FieldValue::Utf8(Some(m.priority.clone())),
        ),
        (
            "assigned_to".into(),
            FieldValue::Utf8(m.assigned_to.clone()),
        ),
        ("iterations".into(), FieldValue::Int32(Some(m.iterations))),
        ("summary".into(), FieldValue::Utf8(m.summary.clone())),
        ("created_at".into(), FieldValue::Int64(Some(m.created_at))),
        ("updated_at".into(), FieldValue::Int64(Some(m.updated_at))),
        ("started_at".into(), FieldValue::Int64(m.started_at)),
        ("completed_at".into(), FieldValue::Int64(m.completed_at)),
    ]
}

fn task_from_record(r: &Record) -> Result<TaskMetadata> {
    Ok(TaskMetadata {
        task_id: record_get(r, "task_id")
            .and_then(|v| v.as_str())
            .context("missing task_id")?
            .to_string(),
        conversation_id: record_get(r, "conversation_id")
            .and_then(|v| v.as_str())
            .context("missing conversation_id")?
            .to_string(),
        plan_id: record_get(r, "plan_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        description: record_get(r, "description")
            .and_then(|v| v.as_str())
            .context("missing description")?
            .to_string(),
        status: record_get(r, "status")
            .and_then(|v| v.as_str())
            .context("missing status")?
            .to_string(),
        parent_id: record_get(r, "parent_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        children: record_get(r, "children")
            .and_then(|v| v.as_str())
            .context("missing children")?
            .to_string(),
        depends_on: record_get(r, "depends_on")
            .and_then(|v| v.as_str())
            .context("missing depends_on")?
            .to_string(),
        priority: record_get(r, "priority")
            .and_then(|v| v.as_str())
            .context("missing priority")?
            .to_string(),
        assigned_to: record_get(r, "assigned_to")
            .and_then(|v| v.as_str())
            .map(String::from),
        iterations: record_get(r, "iterations")
            .and_then(|v| v.as_i32())
            .context("missing iterations")?,
        summary: record_get(r, "summary")
            .and_then(|v| v.as_str())
            .map(String::from),
        created_at: record_get(r, "created_at")
            .and_then(|v| v.as_i64())
            .context("missing created_at")?,
        updated_at: record_get(r, "updated_at")
            .and_then(|v| v.as_i64())
            .context("missing updated_at")?,
        started_at: record_get(r, "started_at").and_then(|v| v.as_i64()),
        completed_at: record_get(r, "completed_at").and_then(|v| v.as_i64()),
    })
}

fn state_to_record(s: &AgentStateMetadata) -> Record {
    vec![
        (
            "agent_id".into(),
            FieldValue::Utf8(Some(s.agent_id.clone())),
        ),
        ("task_id".into(), FieldValue::Utf8(Some(s.task_id.clone()))),
        (
            "conversation_id".into(),
            FieldValue::Utf8(Some(s.conversation_id.clone())),
        ),
        ("status".into(), FieldValue::Utf8(Some(s.status.clone()))),
        ("iteration".into(), FieldValue::Int32(Some(s.iteration))),
        (
            "context_json".into(),
            FieldValue::Utf8(Some(s.context_json.clone())),
        ),
        ("created_at".into(), FieldValue::Int64(Some(s.created_at))),
        ("updated_at".into(), FieldValue::Int64(Some(s.updated_at))),
    ]
}

fn state_from_record(r: &Record) -> Result<AgentStateMetadata> {
    Ok(AgentStateMetadata {
        agent_id: record_get(r, "agent_id")
            .and_then(|v| v.as_str())
            .context("missing agent_id")?
            .to_string(),
        task_id: record_get(r, "task_id")
            .and_then(|v| v.as_str())
            .context("missing task_id")?
            .to_string(),
        conversation_id: record_get(r, "conversation_id")
            .and_then(|v| v.as_str())
            .context("missing conversation_id")?
            .to_string(),
        status: record_get(r, "status")
            .and_then(|v| v.as_str())
            .context("missing status")?
            .to_string(),
        iteration: record_get(r, "iteration")
            .and_then(|v| v.as_i32())
            .context("missing iteration")?,
        context_json: record_get(r, "context_json")
            .and_then(|v| v.as_str())
            .context("missing context_json")?
            .to_string(),
        created_at: record_get(r, "created_at")
            .and_then(|v| v.as_i64())
            .context("missing created_at")?,
        updated_at: record_get(r, "updated_at")
            .and_then(|v| v.as_i64())
            .context("missing updated_at")?,
    })
}

// ── TaskMetadata ────────────────────────────────────────────────────────

/// Metadata for storing tasks
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskMetadata {
    /// Unique task identifier.
    pub task_id: String,
    /// Conversation this task belongs to.
    pub conversation_id: String,
    /// Plan this task belongs to.
    pub plan_id: Option<String>,
    /// Task description.
    pub description: String,
    /// Current task status.
    pub status: String,
    /// Parent task identifier.
    pub parent_id: Option<String>,
    /// Child task IDs (JSON array).
    pub children: String, // JSON array
    /// Task dependency IDs (JSON array).
    pub depends_on: String, // JSON array
    /// Task priority level.
    pub priority: String,
    /// Agent assigned to this task.
    pub assigned_to: Option<String>,
    /// Number of iterations completed.
    pub iterations: i32,
    /// Task completion summary.
    pub summary: Option<String>,
    /// Creation timestamp (Unix seconds).
    pub created_at: i64,
    /// Last update timestamp (Unix seconds).
    pub updated_at: i64,
    /// Start timestamp (Unix seconds).
    pub started_at: Option<i64>,
    /// Completion timestamp (Unix seconds).
    pub completed_at: Option<i64>,
}

impl TaskMetadata {
    /// Convert from Task
    pub fn from_task(task: &Task, conversation_id: &str) -> Self {
        Self {
            task_id: task.id.clone(),
            conversation_id: conversation_id.to_string(),
            plan_id: task.plan_id.clone(),
            description: task.description.clone(),
            status: format!("{:?}", task.status).to_lowercase(),
            parent_id: task.parent_id.clone(),
            children: serde_json::to_string(&task.children).unwrap_or_default(),
            depends_on: serde_json::to_string(&task.depends_on).unwrap_or_default(),
            priority: format!("{:?}", task.priority).to_lowercase(),
            assigned_to: task.assigned_to.clone(),
            iterations: task.iterations as i32,
            summary: task.summary.clone(),
            created_at: task.created_at,
            updated_at: task.updated_at,
            started_at: task.started_at,
            completed_at: task.completed_at,
        }
    }

    /// Convert to Task
    pub fn to_task(&self) -> Task {
        let status = match self.status.as_str() {
            "pending" => TaskStatus::Pending,
            "inprogress" => TaskStatus::InProgress,
            "completed" => TaskStatus::Completed,
            "failed" => TaskStatus::Failed,
            "blocked" => TaskStatus::Blocked,
            _ => TaskStatus::Pending,
        };

        let priority = match self.priority.as_str() {
            "low" => TaskPriority::Low,
            "normal" => TaskPriority::Normal,
            "high" => TaskPriority::High,
            "urgent" => TaskPriority::Urgent,
            _ => TaskPriority::Normal,
        };

        let children: Vec<String> = serde_json::from_str(&self.children).unwrap_or_default();
        let depends_on: Vec<String> = serde_json::from_str(&self.depends_on).unwrap_or_default();

        Task {
            id: self.task_id.clone(),
            description: self.description.clone(),
            status,
            plan_id: self.plan_id.clone(),
            parent_id: self.parent_id.clone(),
            children,
            depends_on,
            priority,
            assigned_to: self.assigned_to.clone(),
            iterations: self.iterations as u32,
            summary: self.summary.clone(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            started_at: self.started_at,
            completed_at: self.completed_at,
        }
    }
}

// ── TaskStore ───────────────────────────────────────────────────────────

/// Store for managing tasks
pub struct TaskStore<
    B: StorageBackend + 'static = brainwires_storage::databases::lance::LanceDatabase,
> {
    backend: Arc<B>,
}

// Manual Clone impl: Arc<B> is always Clone regardless of B
impl<B: StorageBackend + 'static> Clone for TaskStore<B> {
    fn clone(&self) -> Self {
        Self {
            backend: Arc::clone(&self.backend),
        }
    }
}

impl<B: StorageBackend + 'static> TaskStore<B> {
    /// Create a new task store
    pub fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }

    /// Ensure the underlying table exists.
    pub async fn ensure_table(&self) -> Result<()> {
        self.backend
            .ensure_table(TASK_TABLE, &tasks_field_defs())
            .await
    }

    /// Save a task
    pub async fn save(&self, task: &Task, conversation_id: &str) -> Result<()> {
        let metadata = TaskMetadata::from_task(task, conversation_id);

        // First try to delete existing task with same ID
        let _ = self.delete(&task.id).await;

        self.backend
            .insert(TASK_TABLE, vec![task_to_record(&metadata)])
            .await
            .context("Failed to save task")?;

        Ok(())
    }

    /// Get a task by ID
    pub async fn get(&self, task_id: &str) -> Result<Option<Task>> {
        let filter = Filter::Eq(
            "task_id".into(),
            FieldValue::Utf8(Some(task_id.to_string())),
        );
        let records = self
            .backend
            .query(TASK_TABLE, Some(&filter), Some(1))
            .await?;

        match records.first() {
            Some(r) => Ok(Some(task_from_record(r)?.to_task())),
            None => Ok(None),
        }
    }

    /// Get all tasks for a conversation
    pub async fn get_by_conversation(&self, conversation_id: &str) -> Result<Vec<Task>> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        let records = self.backend.query(TASK_TABLE, Some(&filter), None).await?;

        records
            .iter()
            .map(|r| task_from_record(r).map(|m| m.to_task()))
            .collect()
    }

    /// Get all tasks for a plan
    pub async fn get_by_plan(&self, plan_id: &str) -> Result<Vec<Task>> {
        let filter = Filter::Eq(
            "plan_id".into(),
            FieldValue::Utf8(Some(plan_id.to_string())),
        );
        let records = self.backend.query(TASK_TABLE, Some(&filter), None).await?;

        records
            .iter()
            .map(|r| task_from_record(r).map(|m| m.to_task()))
            .collect()
    }

    /// Delete a task
    pub async fn delete(&self, task_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "task_id".into(),
            FieldValue::Utf8(Some(task_id.to_string())),
        );
        self.backend
            .delete(TASK_TABLE, &filter)
            .await
            .context("Failed to delete task")?;
        Ok(())
    }

    /// Delete all tasks for a conversation
    pub async fn delete_by_conversation(&self, conversation_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        self.backend
            .delete(TASK_TABLE, &filter)
            .await
            .context("Failed to delete tasks for conversation")?;
        Ok(())
    }

    /// Delete all tasks for a plan
    pub async fn delete_by_plan(&self, plan_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "plan_id".into(),
            FieldValue::Utf8(Some(plan_id.to_string())),
        );
        self.backend
            .delete(TASK_TABLE, &filter)
            .await
            .context("Failed to delete tasks for plan")?;
        Ok(())
    }

    /// Schema for the tasks table as backend-agnostic field definitions.
    pub fn tasks_schema() -> Vec<FieldDef> {
        tasks_field_defs()
    }

    /// Arrow schema for the tasks table, used by `LanceDatabase` table creation.
    pub fn tasks_arrow_schema() -> Arc<arrow_schema::Schema> {
        use arrow_schema::{DataType, Field, Schema};
        Arc::new(Schema::new(vec![
            Field::new("task_id", DataType::Utf8, false),
            Field::new("conversation_id", DataType::Utf8, false),
            Field::new("plan_id", DataType::Utf8, true),
            Field::new("description", DataType::Utf8, false),
            Field::new("status", DataType::Utf8, false),
            Field::new("parent_id", DataType::Utf8, true),
            Field::new("children", DataType::Utf8, false),
            Field::new("depends_on", DataType::Utf8, false),
            Field::new("priority", DataType::Utf8, false),
            Field::new("assigned_to", DataType::Utf8, true),
            Field::new("iterations", DataType::Int32, false),
            Field::new("summary", DataType::Utf8, true),
            Field::new("created_at", DataType::Int64, false),
            Field::new("updated_at", DataType::Int64, false),
            Field::new("started_at", DataType::Int64, true),
            Field::new("completed_at", DataType::Int64, true),
        ]))
    }
}

// ── AgentStateMetadata ──────────────────────────────────────────────────

/// Metadata for storing agent state
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentStateMetadata {
    /// Unique agent identifier.
    pub agent_id: String,
    /// Task the agent is working on.
    pub task_id: String,
    /// Conversation context.
    pub conversation_id: String,
    /// Current agent status.
    pub status: String,
    /// Current iteration number.
    pub iteration: i32,
    /// Serialized agent context (JSON).
    pub context_json: String, // Serialized AgentContext
    /// Creation timestamp (Unix seconds).
    pub created_at: i64,
    /// Last update timestamp (Unix seconds).
    pub updated_at: i64,
}

// ── AgentStateStore ─────────────────────────────────────────────────────

/// Store for managing agent state persistence
pub struct AgentStateStore<
    B: StorageBackend + 'static = brainwires_storage::databases::lance::LanceDatabase,
> {
    backend: Arc<B>,
}

impl<B: StorageBackend + 'static> AgentStateStore<B> {
    /// Create a new agent state store
    pub fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }

    /// Ensure the underlying table exists.
    pub async fn ensure_table(&self) -> Result<()> {
        self.backend
            .ensure_table(AGENT_STATE_TABLE, &agent_states_field_defs())
            .await
    }

    /// Save agent state
    pub async fn save(&self, state: &AgentStateMetadata) -> Result<()> {
        // First try to delete existing state with same agent ID
        let _ = self.delete(&state.agent_id).await;

        self.backend
            .insert(AGENT_STATE_TABLE, vec![state_to_record(state)])
            .await
            .context("Failed to save agent state")?;

        Ok(())
    }

    /// Get agent state by ID
    pub async fn get(&self, agent_id: &str) -> Result<Option<AgentStateMetadata>> {
        let filter = Filter::Eq(
            "agent_id".into(),
            FieldValue::Utf8(Some(agent_id.to_string())),
        );
        let records = self
            .backend
            .query(AGENT_STATE_TABLE, Some(&filter), Some(1))
            .await?;

        match records.first() {
            Some(r) => Ok(Some(state_from_record(r)?)),
            None => Ok(None),
        }
    }

    /// Get all agent states for a conversation
    pub async fn get_by_conversation(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<AgentStateMetadata>> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        let records = self
            .backend
            .query(AGENT_STATE_TABLE, Some(&filter), None)
            .await?;

        records.iter().map(state_from_record).collect()
    }

    /// Get agent state by task ID
    pub async fn get_by_task(&self, task_id: &str) -> Result<Option<AgentStateMetadata>> {
        let filter = Filter::Eq(
            "task_id".into(),
            FieldValue::Utf8(Some(task_id.to_string())),
        );
        let records = self
            .backend
            .query(AGENT_STATE_TABLE, Some(&filter), Some(1))
            .await?;

        match records.first() {
            Some(r) => Ok(Some(state_from_record(r)?)),
            None => Ok(None),
        }
    }

    /// Delete agent state
    pub async fn delete(&self, agent_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "agent_id".into(),
            FieldValue::Utf8(Some(agent_id.to_string())),
        );
        self.backend
            .delete(AGENT_STATE_TABLE, &filter)
            .await
            .context("Failed to delete agent state")?;
        Ok(())
    }

    /// Delete all agent states for a conversation
    pub async fn delete_by_conversation(&self, conversation_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        self.backend
            .delete(AGENT_STATE_TABLE, &filter)
            .await
            .context("Failed to delete agent states for conversation")?;
        Ok(())
    }

    /// Schema for the agent_states table as backend-agnostic field definitions.
    pub fn agent_states_schema() -> Vec<FieldDef> {
        agent_states_field_defs()
    }

    /// Arrow schema for the agent_states table, used by `LanceDatabase` table creation.
    pub fn agent_states_arrow_schema() -> Arc<arrow_schema::Schema> {
        use arrow_schema::{DataType, Field, Schema};
        Arc::new(Schema::new(vec![
            Field::new("agent_id", DataType::Utf8, false),
            Field::new("task_id", DataType::Utf8, false),
            Field::new("conversation_id", DataType::Utf8, false),
            Field::new("status", DataType::Utf8, false),
            Field::new("iteration", DataType::Int32, false),
            Field::new("context_json", DataType::Utf8, false),
            Field::new("created_at", DataType::Int64, false),
            Field::new("updated_at", DataType::Int64, false),
        ]))
    }
}
