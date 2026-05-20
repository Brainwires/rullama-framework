use anyhow::{Context, Result};
use chrono::Utc;
use std::sync::Arc;

use brainwires_storage::databases::{
    FieldDef, FieldType, FieldValue, Filter, Record, StorageBackend, record_get,
};

const TABLE_NAME: &str = "conversations";

/// Metadata for a conversation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConversationMetadata {
    /// Unique conversation identifier.
    pub conversation_id: String,
    /// Optional conversation title.
    pub title: Option<String>,
    /// Model used in this conversation.
    pub model_id: Option<String>,
    /// Creation timestamp (Unix seconds).
    pub created_at: i64,
    /// Last update timestamp (Unix seconds).
    pub updated_at: i64,
    /// Number of messages in this conversation.
    pub message_count: i32,
}

fn table_schema() -> Vec<FieldDef> {
    vec![
        FieldDef::required("conversation_id", FieldType::Utf8),
        FieldDef::optional("title", FieldType::Utf8),
        FieldDef::optional("model_id", FieldType::Utf8),
        FieldDef::required("created_at", FieldType::Int64),
        FieldDef::required("updated_at", FieldType::Int64),
        FieldDef::required("message_count", FieldType::Int32),
    ]
}

fn to_record(m: &ConversationMetadata) -> Record {
    vec![
        (
            "conversation_id".into(),
            FieldValue::Utf8(Some(m.conversation_id.clone())),
        ),
        ("title".into(), FieldValue::Utf8(m.title.clone())),
        ("model_id".into(), FieldValue::Utf8(m.model_id.clone())),
        ("created_at".into(), FieldValue::Int64(Some(m.created_at))),
        ("updated_at".into(), FieldValue::Int64(Some(m.updated_at))),
        (
            "message_count".into(),
            FieldValue::Int32(Some(m.message_count)),
        ),
    ]
}

fn from_record(r: &Record) -> Result<ConversationMetadata> {
    Ok(ConversationMetadata {
        conversation_id: record_get(r, "conversation_id")
            .and_then(|v| v.as_str())
            .context("missing conversation_id")?
            .to_string(),
        title: record_get(r, "title")
            .and_then(|v| v.as_str())
            .map(String::from),
        model_id: record_get(r, "model_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        created_at: record_get(r, "created_at")
            .and_then(|v| v.as_i64())
            .context("missing created_at")?,
        updated_at: record_get(r, "updated_at")
            .and_then(|v| v.as_i64())
            .context("missing updated_at")?,
        message_count: record_get(r, "message_count")
            .and_then(|v| v.as_i32())
            .context("missing message_count")?,
    })
}

/// Store for managing conversations
pub struct ConversationStore<
    B: StorageBackend = brainwires_storage::databases::lance::LanceDatabase,
> {
    backend: Arc<B>,
}

impl<B: StorageBackend> ConversationStore<B> {
    /// Create a new conversation store
    pub fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }

    /// Ensure the underlying table exists.
    pub async fn ensure_table(&self) -> Result<()> {
        self.backend.ensure_table(TABLE_NAME, &table_schema()).await
    }

    /// Create a new conversation (or update if it already exists)
    pub async fn create(
        &self,
        conversation_id: String,
        title: Option<String>,
        model_id: Option<String>,
        message_count: Option<i32>,
    ) -> Result<ConversationMetadata> {
        // Check if conversation already exists - if so, just update timestamp
        if let Ok(Some(existing)) = self.get(&conversation_id).await {
            self.update(
                &conversation_id,
                title.or(existing.title.clone()),
                message_count,
            )
            .await?;
            return self
                .get(&conversation_id)
                .await?
                .context("Conversation should exist after update");
        }

        let now = Utc::now().timestamp();

        let metadata = ConversationMetadata {
            conversation_id,
            title,
            model_id,
            created_at: now,
            updated_at: now,
            message_count: message_count.unwrap_or(0),
        };

        self.backend
            .insert(TABLE_NAME, vec![to_record(&metadata)])
            .await
            .context("Failed to create conversation")?;

        Ok(metadata)
    }

    /// Get a conversation by ID
    pub async fn get(&self, conversation_id: &str) -> Result<Option<ConversationMetadata>> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
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

    /// List all conversations, sorted by most recently updated first
    pub async fn list(&self, limit: Option<usize>) -> Result<Vec<ConversationMetadata>> {
        let records = self.backend.query(TABLE_NAME, None, None).await?;

        let mut conversations: Vec<ConversationMetadata> =
            records.iter().filter_map(|r| from_record(r).ok()).collect();

        // Sort by updated_at descending (most recent first)
        conversations.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

        if let Some(limit) = limit {
            conversations.truncate(limit);
        }

        Ok(conversations)
    }

    /// Update conversation metadata
    pub async fn update(
        &self,
        conversation_id: &str,
        title: Option<String>,
        message_count: Option<i32>,
    ) -> Result<()> {
        let current = self
            .get(conversation_id)
            .await?
            .context("Conversation not found")?;

        // Delete current
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        self.backend.delete(TABLE_NAME, &filter).await?;

        // Re-insert with updated fields
        let updated = ConversationMetadata {
            conversation_id: conversation_id.to_string(),
            title: title.or(current.title),
            model_id: current.model_id,
            created_at: current.created_at,
            updated_at: Utc::now().timestamp(),
            message_count: message_count.unwrap_or(current.message_count),
        };

        self.backend
            .insert(TABLE_NAME, vec![to_record(&updated)])
            .await?;
        Ok(())
    }

    /// Delete a conversation
    pub async fn delete(&self, conversation_id: &str) -> Result<()> {
        let filter = Filter::Eq(
            "conversation_id".into(),
            FieldValue::Utf8(Some(conversation_id.to_string())),
        );
        self.backend.delete(TABLE_NAME, &filter).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn setup() -> (
        TempDir,
        ConversationStore<brainwires_storage::databases::lance::LanceDatabase>,
    ) {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("test.lance");

        let backend = Arc::new(
            brainwires_storage::databases::lance::LanceDatabase::new(db_path.to_str().unwrap())
                .await
                .unwrap(),
        );
        let store = ConversationStore::new(Arc::clone(&backend));
        store.ensure_table().await.unwrap();

        (temp, store)
    }

    #[tokio::test]
    async fn test_create_conversation() {
        let (_temp, store) = setup().await;

        let conv = store
            .create(
                "test-conv-1".to_string(),
                Some("Test Conversation".to_string()),
                Some("gpt-4".to_string()),
                None,
            )
            .await
            .unwrap();

        assert_eq!(conv.conversation_id, "test-conv-1");
        assert_eq!(conv.title, Some("Test Conversation".to_string()));
        assert_eq!(conv.model_id, Some("gpt-4".to_string()));
        assert_eq!(conv.message_count, 0);
    }

    #[tokio::test]
    async fn test_create_conversation_with_message_count() {
        let (_temp, store) = setup().await;

        let conv = store
            .create(
                "test-conv-1".to_string(),
                Some("Test Conversation".to_string()),
                Some("gpt-4".to_string()),
                Some(5),
            )
            .await
            .unwrap();

        assert_eq!(conv.conversation_id, "test-conv-1");
        assert_eq!(conv.message_count, 5);
    }

    #[tokio::test]
    async fn test_get_conversation() {
        let (_temp, store) = setup().await;

        store
            .create(
                "test-conv-2".to_string(),
                Some("Test".to_string()),
                None,
                None,
            )
            .await
            .unwrap();

        let conv = store.get("test-conv-2").await.unwrap();
        assert!(conv.is_some());
        assert_eq!(conv.unwrap().conversation_id, "test-conv-2");
    }

    #[tokio::test]
    async fn test_get_nonexistent() {
        let (_temp, store) = setup().await;

        let conv = store.get("nonexistent").await.unwrap();
        assert!(conv.is_none());
    }

    #[tokio::test]
    async fn test_list_conversations() {
        let (_temp, store) = setup().await;

        store
            .create("conv-1".to_string(), Some("Conv 1".to_string()), None, None)
            .await
            .unwrap();
        store
            .create("conv-2".to_string(), Some("Conv 2".to_string()), None, None)
            .await
            .unwrap();
        store
            .create("conv-3".to_string(), Some("Conv 3".to_string()), None, None)
            .await
            .unwrap();

        let convs = store.list(None).await.unwrap();
        assert_eq!(convs.len(), 3);
    }

    #[tokio::test]
    async fn test_update_conversation() {
        let (_temp, store) = setup().await;

        store
            .create(
                "conv-update".to_string(),
                Some("Original".to_string()),
                None,
                None,
            )
            .await
            .unwrap();

        store
            .update("conv-update", Some("Updated".to_string()), Some(5))
            .await
            .unwrap();

        let conv = store.get("conv-update").await.unwrap().unwrap();
        assert_eq!(conv.title, Some("Updated".to_string()));
        assert_eq!(conv.message_count, 5);
    }

    #[tokio::test]
    async fn test_delete_conversation() {
        let (_temp, store) = setup().await;

        store
            .create("conv-delete".to_string(), None, None, None)
            .await
            .unwrap();

        let conv = store.get("conv-delete").await.unwrap();
        assert!(conv.is_some());

        store.delete("conv-delete").await.unwrap();

        let conv = store.get("conv-delete").await.unwrap();
        assert!(conv.is_none());
    }
}
