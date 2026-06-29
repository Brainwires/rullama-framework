//! Bridge between BrainClient thought storage and DreamSessionStore trait.

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::Mutex;

use brainwires_core::Message;
use brainwires_knowledge::knowledge::brain_client::BrainClient;
use brainwires_knowledge::knowledge::types::*;
use brainwires_memory::dream::consolidator::DreamSessionStore;

/// Adapts BrainClient's thought storage to the DreamSessionStore trait
/// required by the DreamConsolidator.
pub struct BrainSessionAdapter {
    client: Arc<Mutex<BrainClient>>,
}

impl BrainSessionAdapter {
    pub fn new(client: Arc<Mutex<BrainClient>>) -> Self {
        Self { client }
    }
}

#[async_trait::async_trait]
impl DreamSessionStore for BrainSessionAdapter {
    async fn list_sessions(&self) -> Result<Vec<String>> {
        // List recent thoughts and extract unique session-like groups.
        // For now, return a single "default" session containing all thoughts.
        // A more sophisticated implementation would group by date or conversation_id tags.
        let client = self.client.lock().await;
        let recent = client
            .list_recent(ListRecentRequest {
                limit: 1000,
                category: None,
                since: None,
                owner_id: None,
            })
            .await?;

        if recent.thoughts.is_empty() {
            return Ok(Vec::new());
        }

        // Group by the "session:" tag prefix if present, otherwise "default"
        let mut sessions: Vec<String> = recent
            .thoughts
            .iter()
            .flat_map(|t| {
                t.tags
                    .iter()
                    .filter(|tag| tag.starts_with("session:"))
                    .map(|tag| tag.strip_prefix("session:").unwrap_or(tag).to_string())
            })
            .collect();

        sessions.sort();
        sessions.dedup();

        if sessions.is_empty() {
            sessions.push("default".to_string());
        }

        Ok(sessions)
    }

    async fn load(&self, session_key: &str) -> Result<Option<Vec<Message>>> {
        use brainwires_storage::{FieldValue, Filter};

        let client = self.client.lock().await;
        let safe_key = crate::sanitize_tag_value(session_key);

        // Filter query — exact tag match, no semantic search ambiguity
        let filter = Filter::And(vec![
            Filter::Eq("deleted".into(), FieldValue::Boolean(Some(false))),
            Filter::Raw(format!("tags LIKE '%session:{}%'", safe_key)),
            Filter::Raw("tags LIKE '%auto-capture%'".to_string()),
        ]);
        let contents = client
            .query_thought_contents(&filter, 500)
            .await
            .unwrap_or_default();

        if contents.is_empty() {
            return Ok(None);
        }

        // Convert to Messages based on [role] prefix
        let messages: Vec<Message> = contents
            .iter()
            .map(|c| {
                let content = c
                    .strip_prefix("[assistant] ")
                    .or_else(|| c.strip_prefix("[user] "))
                    .unwrap_or(c);
                if c.starts_with("[assistant]") {
                    Message::assistant(content)
                } else {
                    Message::user(content)
                }
            })
            .collect();

        Ok(Some(messages))
    }

    async fn save(&self, session_key: &str, messages: &[Message]) -> Result<()> {
        use brainwires_storage::{FieldValue, Filter};

        let mut client = self.client.lock().await;

        // Store consolidated summary as a new high-importance thought
        let summary_content = messages
            .iter()
            .filter_map(|m| match &m.content {
                brainwires_core::MessageContent::Text(s) => Some(s.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if !summary_content.is_empty() {
            client
                .capture_thought(CaptureThoughtRequest {
                    content: summary_content,
                    category: Some("insight".to_string()),
                    tags: Some(vec![
                        "consolidated".to_string(),
                        format!("session:{session_key}"),
                        "claude-code".to_string(),
                    ]),
                    importance: Some(0.85),
                    source: Some("dream-consolidation".to_string()),
                    owner_id: None,
                })
                .await?;
        }

        // Delete original session thoughts
        let filter = Filter::And(vec![
            Filter::Eq("deleted".into(), FieldValue::Boolean(Some(false))),
            Filter::Raw(format!(
                "tags LIKE '%session:{}%'",
                crate::sanitize_tag_value(session_key)
            )),
            Filter::Raw("tags LIKE '%auto-capture%'".to_string()),
        ]);
        let deleted = client.delete_by_filter(&filter).await?;
        tracing::info!(
            "Consolidated session {session_key}: stored summary, deleted {deleted} originals"
        );

        Ok(())
    }
}
