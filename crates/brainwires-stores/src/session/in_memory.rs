//! In-memory [`SessionStore`] implementation backed by a mutex-guarded map.
//!
//! Intended for tests, ephemeral sessions, and embedding use-cases. Nothing
//! persists across process restarts.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use tokio::sync::Mutex;

use brainwires_core::Message;

use super::{SessionId, SessionRecord, SessionStore};

#[derive(Debug)]
struct Entry {
    messages: Vec<Message>,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

/// In-memory session store. Cheap to `Arc`-clone — all clones share state.
#[derive(Clone, Default)]
pub struct InMemorySessionStore {
    inner: Arc<Mutex<HashMap<SessionId, Entry>>>,
}

impl InMemorySessionStore {
    /// Build an empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn load(&self, id: &SessionId) -> Result<Option<Vec<Message>>> {
        let map = self.inner.lock().await;
        Ok(map.get(id).map(|e| e.messages.clone()))
    }

    async fn save(&self, id: &SessionId, messages: &[Message]) -> Result<()> {
        let mut map = self.inner.lock().await;
        let now = Utc::now();
        match map.get_mut(id) {
            Some(entry) => {
                entry.messages = messages.to_vec();
                entry.updated_at = now;
            }
            None => {
                map.insert(
                    id.clone(),
                    Entry {
                        messages: messages.to_vec(),
                        created_at: now,
                        updated_at: now,
                    },
                );
            }
        }
        Ok(())
    }

    async fn list(&self) -> Result<Vec<SessionRecord>> {
        let map = self.inner.lock().await;
        let mut out: Vec<SessionRecord> = map
            .iter()
            .map(|(id, e)| SessionRecord {
                id: id.clone(),
                message_count: e.messages.len(),
                created_at: e.created_at,
                updated_at: e.updated_at,
            })
            .collect();
        out.sort_by_key(|r| r.updated_at);
        Ok(out)
    }

    async fn delete(&self, id: &SessionId) -> Result<()> {
        self.inner.lock().await.remove(id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brainwires_core::Message;

    #[tokio::test]
    async fn roundtrip_save_load_delete() {
        let store = InMemorySessionStore::new();
        let id = SessionId::new("alice");

        assert!(store.load(&id).await.unwrap().is_none());

        let msgs = vec![Message::user("hi"), Message::assistant("hello")];
        store.save(&id, &msgs).await.unwrap();

        let loaded = store.load(&id).await.unwrap().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text(), Some("hi"));

        store.delete(&id).await.unwrap();
        assert!(store.load(&id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn save_overwrites_atomically() {
        let store = InMemorySessionStore::new();
        let id = SessionId::new("bob");
        store.save(&id, &[Message::user("one")]).await.unwrap();
        store
            .save(&id, &[Message::user("two"), Message::user("three")])
            .await
            .unwrap();
        let loaded = store.load(&id).await.unwrap().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text(), Some("two"));
    }

    #[tokio::test]
    async fn list_returns_known_sessions() {
        let store = InMemorySessionStore::new();
        store
            .save(&SessionId::new("a"), &[Message::user("x")])
            .await
            .unwrap();
        store
            .save(&SessionId::new("b"), &[Message::user("y")])
            .await
            .unwrap();
        let list = store.list().await.unwrap();
        assert_eq!(list.len(), 2);
        let ids: Vec<&str> = list.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a") && ids.contains(&"b"));
    }

    #[tokio::test]
    async fn delete_unknown_is_noop() {
        let store = InMemorySessionStore::new();
        store
            .delete(&SessionId::new("never-existed"))
            .await
            .unwrap();
    }
}
