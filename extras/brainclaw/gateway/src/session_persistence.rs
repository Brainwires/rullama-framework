//! Session persistence — save and restore conversation history across restarts.
//!
//! Provides a [`SessionStore`] trait with a default JSON-file implementation
//! ([`JsonFileStore`]) so that conversation messages survive gateway restarts.
//! The trait is designed to be swapped out for a full [`TieredMemory`] backend
//! later without changing the handler code.

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_core::Message;

// ── Trait ────────────────────────────────────────────────────────────────────

/// Backend-agnostic session persistence.
///
/// Implementations must be `Send + Sync` so they can be shared across async
/// tasks behind an `Arc`.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Persist all messages for `session_key`, replacing any previous data.
    async fn save(&self, session_key: &str, messages: &[Message]) -> Result<()>;

    /// Load previously persisted messages, or `None` if no data exists.
    async fn load(&self, session_key: &str) -> Result<Option<Vec<Message>>>;

    /// Delete persisted data for `session_key`.
    async fn delete(&self, session_key: &str) -> Result<()>;

    /// List all session keys that have persisted data.
    async fn list_sessions(&self) -> Result<Vec<String>>;
}

// ── JSON file store ──────────────────────────────────────────────────────────

/// Simple JSON-file–backed [`SessionStore`].
///
/// Each session is stored as `{storage_dir}/{session_key}.json` containing a
/// JSON array of [`Message`] objects.
pub struct JsonFileStore {
    storage_dir: PathBuf,
}

impl JsonFileStore {
    /// Create a new `JsonFileStore` rooted at `storage_dir`.
    ///
    /// The directory is created (recursively) if it does not exist.
    pub fn new(storage_dir: impl Into<PathBuf>) -> Result<Self> {
        let storage_dir = storage_dir.into();
        std::fs::create_dir_all(&storage_dir)?;
        Ok(Self { storage_dir })
    }

    /// Return the file path for a given session key.
    fn session_path(&self, session_key: &str) -> PathBuf {
        // Sanitise the key so it is safe as a filename.
        let safe_key = sanitize_session_key(session_key);
        self.storage_dir.join(format!("{safe_key}.json"))
    }

    /// Append a single message to an existing session file (or create it).
    ///
    /// This is a convenience helper that avoids rewriting the entire file on
    /// every message — it loads, appends, then saves.
    pub async fn save_on_message(&self, session_key: &str, message: &Message) -> Result<()> {
        let mut messages = self.load(session_key).await?.unwrap_or_default();
        messages.push(message.clone());
        self.save(session_key, &messages).await
    }
}

#[async_trait]
impl SessionStore for JsonFileStore {
    async fn save(&self, session_key: &str, messages: &[Message]) -> Result<()> {
        let path = self.session_path(session_key);
        let json = serde_json::to_string_pretty(messages)?;
        tokio::fs::write(&path, json).await?;
        Ok(())
    }

    async fn load(&self, session_key: &str) -> Result<Option<Vec<Message>>> {
        let path = self.session_path(session_key);
        if !path.exists() {
            return Ok(None);
        }
        let data = tokio::fs::read_to_string(&path).await?;
        let messages: Vec<Message> = serde_json::from_str(&data)?;
        Ok(Some(messages))
    }

    async fn delete(&self, session_key: &str) -> Result<()> {
        let path = self.session_path(session_key);
        if path.exists() {
            tokio::fs::remove_file(&path).await?;
        }
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        let mut sessions = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.storage_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            {
                sessions.push(stem.to_string());
            }
        }
        Ok(sessions)
    }
}

/// Build a session key from a platform name and user identifier.
pub fn session_key(platform: &str, user_id: &str) -> String {
    format!("{platform}_{user_id}")
}

/// Replace characters that are unsafe in filenames with underscores.
fn sanitize_session_key(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

// ── Expand tilde helper ──────────────────────────────────────────────────────

/// Expand a leading `~` in a path to the user's home directory.
pub fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn temp_store() -> (TempDir, JsonFileStore) {
        let dir = TempDir::new().unwrap();
        let store = JsonFileStore::new(dir.path()).unwrap();
        (dir, store)
    }

    #[tokio::test]
    async fn test_save_and_load() {
        let (_dir, store) = temp_store();
        let messages = vec![Message::user("Hello"), Message::assistant("Hi there!")];
        store.save("test_session", &messages).await.unwrap();

        let loaded = store.load("test_session").await.unwrap().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text(), Some("Hello"));
        assert_eq!(loaded[1].text(), Some("Hi there!"));
    }

    #[tokio::test]
    async fn test_load_nonexistent_returns_none() {
        let (_dir, store) = temp_store();
        let result = store.load("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_delete() {
        let (_dir, store) = temp_store();
        let messages = vec![Message::user("temp")];
        store.save("to_delete", &messages).await.unwrap();
        assert!(store.load("to_delete").await.unwrap().is_some());

        store.delete("to_delete").await.unwrap();
        assert!(store.load("to_delete").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_delete_nonexistent_is_ok() {
        let (_dir, store) = temp_store();
        // Should not error
        store.delete("ghost").await.unwrap();
    }

    #[tokio::test]
    async fn test_list_sessions() {
        let (_dir, store) = temp_store();
        store
            .save("discord_user1", &[Message::user("a")])
            .await
            .unwrap();
        store
            .save("telegram_user2", &[Message::user("b")])
            .await
            .unwrap();

        let mut sessions = store.list_sessions().await.unwrap();
        sessions.sort();
        assert_eq!(sessions, vec!["discord_user1", "telegram_user2"]);
    }

    #[tokio::test]
    async fn test_save_on_message_appends() {
        let (_dir, store) = temp_store();
        store
            .save_on_message("sess", &Message::user("first"))
            .await
            .unwrap();
        store
            .save_on_message("sess", &Message::assistant("second"))
            .await
            .unwrap();

        let loaded = store.load("sess").await.unwrap().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text(), Some("first"));
        assert_eq!(loaded[1].text(), Some("second"));
    }

    #[test]
    fn test_sanitize_session_key() {
        assert_eq!(sanitize_session_key("discord_user-1"), "discord_user-1");
        assert_eq!(sanitize_session_key("a/b\\c:d"), "a_b_c_d");
    }

    #[test]
    fn test_session_key_builder() {
        assert_eq!(session_key("discord", "user123"), "discord_user123");
    }
}
