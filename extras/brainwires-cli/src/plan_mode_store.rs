//! Plan Mode Store - Persists plan mode sessions to LanceDB
//!
//! Plan mode sessions are stored separately from the main conversation,
//! allowing isolated planning context that persists across TUI attach/detach cycles.

use anyhow::{Context, Result};
use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::sync::Arc;

use crate::types::message::Message;
use crate::types::plan_mode::PlanModeState;
use brainwires::agent_network::ipc::DisplayMessage;
use brainwires_storage::LanceDatabase;

/// Store for managing plan mode sessions
pub struct PlanModeStore {
    client: Arc<LanceDatabase>,
}

impl PlanModeStore {
    /// Create a new plan mode store
    pub fn new(client: Arc<LanceDatabase>) -> Self {
        Self { client }
    }

    /// Save a plan mode state (create or update)
    pub async fn save(&self, state: &PlanModeState) -> Result<()> {
        // First try to delete existing state with same ID
        let _ = self.delete(&state.plan_session_id).await;

        // Create record batch
        let batch = self.state_to_batch(state)?;

        // Add to table
        let table = self
            .client
            .connection()
            .open_table("plan_mode_sessions")
            .execute()
            .await
            .context("Failed to open plan_mode_sessions table")?;

        let schema = batch.schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        table
            .add(Box::new(batches) as Box<dyn arrow_array::RecordBatchReader + Send>)
            .execute()
            .await
            .context("Failed to save plan mode state")?;

        Ok(())
    }

    /// Get plan mode state by plan session ID
    pub async fn get(&self, plan_session_id: &str) -> Result<Option<PlanModeState>> {
        let table = self
            .client
            .connection()
            .open_table("plan_mode_sessions")
            .execute()
            .await
            .context("Failed to open plan_mode_sessions table")?;

        let filter = format!("plan_session_id = '{}'", plan_session_id);
        let stream = table.query().only_if(filter).execute().await?;
        let results: Vec<RecordBatch> = stream.try_collect().await?;

        if results.is_empty() {
            return Ok(None);
        }

        let batch = &results[0];
        if batch.num_rows() == 0 {
            return Ok(None);
        }

        let states = self.batch_to_states(batch)?;
        Ok(states.into_iter().next())
    }

    /// Get plan mode state for a main session
    pub async fn get_by_main_session(
        &self,
        main_session_id: &str,
    ) -> Result<Option<PlanModeState>> {
        let table = self
            .client
            .connection()
            .open_table("plan_mode_sessions")
            .execute()
            .await
            .context("Failed to open plan_mode_sessions table")?;

        // Get the most recent active plan mode for this main session
        let filter = format!("main_session_id = '{}' AND active = true", main_session_id);
        let stream = table.query().only_if(filter).execute().await?;
        let results: Vec<RecordBatch> = stream.try_collect().await?;

        let mut states = Vec::new();
        for batch in results {
            states.extend(self.batch_to_states(&batch)?);
        }

        // Sort by updated_at descending (most recent first)
        states.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

        Ok(states.into_iter().next())
    }

    /// List all plan mode sessions (for debugging/admin)
    pub async fn list(&self, limit: Option<usize>) -> Result<Vec<PlanModeState>> {
        let table = self
            .client
            .connection()
            .open_table("plan_mode_sessions")
            .execute()
            .await
            .context("Failed to open plan_mode_sessions table")?;

        let limit = limit.unwrap_or(100);
        let stream = table.query().limit(limit).execute().await?;
        let results: Vec<RecordBatch> = stream.try_collect().await?;

        let mut states = Vec::new();
        for batch in results {
            states.extend(self.batch_to_states(&batch)?);
        }

        // Sort by updated_at descending
        states.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

        Ok(states)
    }

    /// Delete a plan mode state by ID
    pub async fn delete(&self, plan_session_id: &str) -> Result<()> {
        let table = self
            .client
            .connection()
            .open_table("plan_mode_sessions")
            .execute()
            .await
            .context("Failed to open plan_mode_sessions table")?;

        table
            .delete(&format!("plan_session_id = '{}'", plan_session_id))
            .await
            .context("Failed to delete plan mode state")?;

        Ok(())
    }

    /// Delete all plan mode states for a main session
    pub async fn delete_by_main_session(&self, main_session_id: &str) -> Result<()> {
        let table = self
            .client
            .connection()
            .open_table("plan_mode_sessions")
            .execute()
            .await
            .context("Failed to open plan_mode_sessions table")?;

        table
            .delete(&format!("main_session_id = '{}'", main_session_id))
            .await
            .context("Failed to delete plan mode states for main session")?;

        Ok(())
    }

    /// Convert plan mode state to record batch
    fn state_to_batch(&self, state: &PlanModeState) -> Result<RecordBatch> {
        let schema = Self::plan_mode_schema();

        // Serialize messages and conversation history to JSON
        let messages_json =
            serde_json::to_string(&state.messages).context("Failed to serialize messages")?;
        let history_json = serde_json::to_string(&state.conversation_history)
            .context("Failed to serialize conversation history")?;

        let plan_session_ids = StringArray::from(vec![state.plan_session_id.as_str()]);
        let main_session_ids = StringArray::from(vec![state.main_session_id.as_str()]);
        let messages = StringArray::from(vec![messages_json.as_str()]);
        let conversation_histories = StringArray::from(vec![history_json.as_str()]);
        let started_ats = Int64Array::from(vec![state.started_at]);
        let updated_ats = Int64Array::from(vec![state.updated_at]);
        let focuses = StringArray::from(vec![state.focus.as_deref()]);
        let actives = BooleanArray::from(vec![state.active]);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(plan_session_ids),
                Arc::new(main_session_ids),
                Arc::new(messages),
                Arc::new(conversation_histories),
                Arc::new(started_ats),
                Arc::new(updated_ats),
                Arc::new(focuses),
                Arc::new(actives),
            ],
        )
        .context("Failed to create plan mode record batch")
    }

    /// Convert record batch to plan mode states
    fn batch_to_states(&self, batch: &RecordBatch) -> Result<Vec<PlanModeState>> {
        let plan_session_ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let main_session_ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let messages_json = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let history_json = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let started_ats = batch
            .column(4)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let updated_ats = batch
            .column(5)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let focuses = batch
            .column(6)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let actives = batch
            .column(7)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();

        let mut states = Vec::new();
        for i in 0..batch.num_rows() {
            // Deserialize messages
            let messages: Vec<DisplayMessage> = if messages_json.is_null(i) {
                Vec::new()
            } else {
                serde_json::from_str(messages_json.value(i)).unwrap_or_default()
            };

            // Deserialize conversation history
            let conversation_history: Vec<Message> = if history_json.is_null(i) {
                Vec::new()
            } else {
                serde_json::from_str(history_json.value(i)).unwrap_or_default()
            };

            states.push(PlanModeState {
                plan_session_id: plan_session_ids.value(i).to_string(),
                main_session_id: main_session_ids.value(i).to_string(),
                messages,
                conversation_history,
                started_at: started_ats.value(i),
                updated_at: updated_ats.value(i),
                focus: if focuses.is_null(i) {
                    None
                } else {
                    Some(focuses.value(i).to_string())
                },
                active: actives.value(i),
            });
        }

        Ok(states)
    }

    /// Schema for plan_mode_sessions table
    pub fn plan_mode_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("plan_session_id", DataType::Utf8, false),
            Field::new("main_session_id", DataType::Utf8, false),
            Field::new("messages", DataType::Utf8, false), // JSON
            Field::new("conversation_history", DataType::Utf8, false), // JSON
            Field::new("started_at", DataType::Int64, false),
            Field::new("updated_at", DataType::Int64, false),
            Field::new("focus", DataType::Utf8, true),
            Field::new("active", DataType::Boolean, false),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_mode_schema() {
        let schema = PlanModeStore::plan_mode_schema();
        assert_eq!(schema.fields().len(), 8);
        assert_eq!(schema.field(0).name(), "plan_session_id");
        assert_eq!(schema.field(1).name(), "main_session_id");
        assert_eq!(schema.field(2).name(), "messages");
        assert_eq!(schema.field(3).name(), "conversation_history");
        assert_eq!(schema.field(4).name(), "started_at");
        assert_eq!(schema.field(5).name(), "updated_at");
        assert_eq!(schema.field(6).name(), "focus");
        assert_eq!(schema.field(7).name(), "active");
    }
}
