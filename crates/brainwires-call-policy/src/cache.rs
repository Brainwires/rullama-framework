//! Response caching decorator.
//!
//! Wraps a [`Provider`] in a content-addressed cache so deterministic eval
//! runs are byte-reproducible and local development stops burning real
//! tokens. The cache key is `SHA-256(messages_json || tools_json ||
//! options_json)` — any change to inputs produces a miss.
//!
//! Only the non-streaming [`Provider::chat`] path is cached. Streaming
//! passes through unchanged (reconstructing a replayable stream from a
//! recorded response is out of scope for this decorator).
//!
//! The in-memory [`MemoryCache`] is the default backend. A SQLite-backed
//! `SqliteCache` lives behind the `cache` feature flag for runs that need
//! persistence across process restarts.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_core::message::{ChatResponse, Message, MessageContent, Role, StreamChunk, Usage};
use brainwires_core::provider::{ChatOptions, Provider};
use brainwires_core::tool::Tool;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

/// Key used to address a cached response.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey(pub String);

/// Wire representation of a cached response, suitable for JSON storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    /// Serialised [`Role`] of the produced message.
    pub role: Role,
    /// Message payload as plain text (blocks are rendered to a single string
    /// since cache hits are only useful for deterministic evals where tool
    /// calls produce the same output each time anyway).
    pub text: String,
    /// Usage counters at record time.
    pub usage: Usage,
    /// Original `finish_reason`, when the provider supplied one.
    pub finish_reason: Option<String>,
}

impl CachedResponse {
    fn from_chat(resp: &ChatResponse) -> Self {
        let text = match &resp.message.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Blocks(_) => resp
                .message
                .text()
                .map(|s| s.to_string())
                .unwrap_or_default(),
        };
        Self {
            role: resp.message.role.clone(),
            text,
            usage: resp.usage.clone(),
            finish_reason: resp.finish_reason.clone(),
        }
    }

    fn into_chat(self) -> ChatResponse {
        let msg = match self.role {
            Role::Assistant => Message::assistant(self.text.clone()),
            Role::System => Message::system(self.text.clone()),
            _ => Message::user(self.text.clone()),
        };
        ChatResponse {
            message: msg,
            usage: self.usage,
            finish_reason: self.finish_reason,
        }
    }
}

/// Pluggable storage backend behind [`CachedProvider`].
#[async_trait]
pub trait CacheBackend: Send + Sync {
    /// Return the cached response for `key`, if any.
    async fn get(&self, key: &CacheKey) -> Result<Option<CachedResponse>>;
    /// Persist `resp` under `key`. Overwrites any previous value.
    async fn put(&self, key: &CacheKey, resp: CachedResponse) -> Result<()>;
}

/// In-memory cache — the default backend. Cheap to `Arc`-clone.
#[derive(Clone, Default)]
pub struct MemoryCache {
    inner: Arc<Mutex<HashMap<CacheKey, CachedResponse>>>,
}

impl MemoryCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cached responses.
    pub async fn len(&self) -> usize {
        self.inner.lock().await.len()
    }

    /// `true` if no responses are cached yet.
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

#[async_trait]
impl CacheBackend for MemoryCache {
    async fn get(&self, key: &CacheKey) -> Result<Option<CachedResponse>> {
        Ok(self.inner.lock().await.get(key).cloned())
    }
    async fn put(&self, key: &CacheKey, resp: CachedResponse) -> Result<()> {
        self.inner.lock().await.insert(key.clone(), resp);
        Ok(())
    }
}

/// Compute a stable cache key from the inputs to a `chat()` call.
///
/// Tools are name-sorted before hashing so reordering them doesn't break
/// cache hits. Options and messages are serialised verbatim.
pub fn cache_key_for(
    messages: &[Message],
    tools: Option<&[Tool]>,
    options: &ChatOptions,
) -> CacheKey {
    let mut hasher = Sha256::new();
    // Serialising with serde_json gives us a canonical representation.
    let msgs = serde_json::to_vec(messages).unwrap_or_default();
    hasher.update(&msgs);

    if let Some(ts) = tools {
        let mut names: Vec<&str> = ts.iter().map(|t| t.name.as_str()).collect();
        names.sort();
        for n in names {
            hasher.update(b"\x00tool:");
            hasher.update(n.as_bytes());
        }
    }

    let opts = serde_json::to_vec(options).unwrap_or_default();
    hasher.update(b"\x00opts:");
    hasher.update(&opts);

    let digest = hasher.finalize();
    CacheKey(hex_encode(&digest))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// A [`Provider`] decorator that deduplicates identical `chat()` calls.
pub struct CachedProvider<P: Provider + ?Sized> {
    inner: Arc<P>,
    backend: Arc<dyn CacheBackend>,
}

impl<P: Provider + ?Sized> CachedProvider<P> {
    /// Wrap `inner` with the given cache backend.
    pub fn new(inner: Arc<P>, backend: Arc<dyn CacheBackend>) -> Self {
        Self { inner, backend }
    }

    /// Convenience constructor using the in-memory backend.
    pub fn with_memory_cache(inner: Arc<P>) -> (Self, MemoryCache) {
        let cache = MemoryCache::new();
        let me = Self::new(inner, Arc::new(cache.clone()));
        (me, cache)
    }
}

#[async_trait]
impl<P: Provider + ?Sized + 'static> Provider for CachedProvider<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn max_output_tokens(&self) -> Option<u32> {
        self.inner.max_output_tokens()
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        let key = cache_key_for(messages, tools, options);
        if let Some(cached) = self.backend.get(&key).await? {
            tracing::debug!(provider = self.inner.name(), key = %key.0, "cache hit");
            return Ok(cached.into_chat());
        }
        let resp = self.inner.chat(messages, tools, options).await?;
        self.backend
            .put(&key, CachedResponse::from_chat(&resp))
            .await
            .ok(); // caching failures are non-fatal
        Ok(resp)
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        // Streaming bypasses the cache — reconstructing a replayable event
        // stream from a single recorded response would fabricate data a
        // caller can't distinguish from real model output.
        self.inner.stream_chat(messages, tools, options)
    }
}

#[cfg(feature = "cache")]
mod sqlite_backend {
    use super::{CacheBackend, CacheKey, CachedResponse};
    use anyhow::{Context, Result};
    use async_trait::async_trait;
    use rusqlite::{Connection, OptionalExtension, params};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// Disk-backed cache. Uses a single shared connection guarded by a mutex.
    pub struct SqliteCache {
        conn: Arc<Mutex<Connection>>,
        path: PathBuf,
    }

    impl SqliteCache {
        /// Open (or create) the cache at `path`.
        pub fn open(path: impl AsRef<Path>) -> Result<Self> {
            let path = path.as_ref().to_path_buf();
            let conn = Connection::open(&path)
                .with_context(|| format!("opening cache at {}", path.display()))?;
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS responses (
                    key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                );",
            )?;
            Ok(Self {
                conn: Arc::new(Mutex::new(conn)),
                path,
            })
        }

        /// Path this cache writes to.
        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    #[async_trait]
    impl CacheBackend for SqliteCache {
        async fn get(&self, key: &CacheKey) -> Result<Option<CachedResponse>> {
            let conn = self.conn.lock().await;
            let raw: Option<String> = conn
                .query_row(
                    "SELECT payload FROM responses WHERE key = ?1",
                    params![&key.0],
                    |r| r.get(0),
                )
                .optional()?;
            Ok(match raw {
                Some(s) => Some(serde_json::from_str(&s)?),
                None => None,
            })
        }
        async fn put(&self, key: &CacheKey, resp: CachedResponse) -> Result<()> {
            let payload = serde_json::to_string(&resp)?;
            let conn = self.conn.lock().await;
            conn.execute(
                "INSERT INTO responses (key, payload) VALUES (?1, ?2)
                 ON CONFLICT(key) DO UPDATE SET payload = excluded.payload",
                params![&key.0, payload],
            )?;
            Ok(())
        }
    }
}

#[cfg(feature = "cache")]
pub use sqlite_backend::SqliteCache;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests_util::EchoProvider;

    #[tokio::test]
    async fn miss_populates_cache_then_hits_match() {
        let inner = Arc::new(EchoProvider::ok("p"));
        let (cached, mem) = CachedProvider::with_memory_cache(inner.clone());

        let msgs = vec![Message::user("hello")];
        let opts = ChatOptions::default();

        let r1 = cached.chat(&msgs, None, &opts).await.unwrap();
        assert_eq!(inner.calls(), 1);
        assert_eq!(mem.len().await, 1);

        let r2 = cached.chat(&msgs, None, &opts).await.unwrap();
        assert_eq!(
            inner.calls(),
            1,
            "cache hit must not call the inner provider again"
        );
        assert_eq!(r1.message.text(), r2.message.text());
    }

    #[tokio::test]
    async fn different_messages_miss() {
        let inner = Arc::new(EchoProvider::ok("p"));
        let (cached, _mem) = CachedProvider::with_memory_cache(inner.clone());
        let opts = ChatOptions::default();

        cached
            .chat(&[Message::user("a")], None, &opts)
            .await
            .unwrap();
        cached
            .chat(&[Message::user("b")], None, &opts)
            .await
            .unwrap();
        assert_eq!(inner.calls(), 2);
    }

    #[test]
    fn key_stable_across_reorderings() {
        let opts = ChatOptions::default();
        let msgs = vec![Message::user("x")];
        let tool_a = Tool {
            name: "alpha".into(),
            ..Default::default()
        };
        let tool_b = Tool {
            name: "beta".into(),
            ..Default::default()
        };

        let k1 = cache_key_for(&msgs, Some(&[tool_a.clone(), tool_b.clone()]), &opts);
        let k2 = cache_key_for(&msgs, Some(&[tool_b, tool_a]), &opts);
        assert_eq!(k1, k2, "tool order must not affect the key");
    }

    #[cfg(feature = "cache")]
    #[tokio::test]
    async fn sqlite_cache_persists() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cache.db");

        let inner = Arc::new(EchoProvider::ok("p"));
        {
            let backend = Arc::new(SqliteCache::open(&path).unwrap()) as Arc<dyn CacheBackend>;
            let cached = CachedProvider::new(inner.clone(), backend);
            cached
                .chat(&[Message::user("persist")], None, &ChatOptions::default())
                .await
                .unwrap();
        }
        // Reopen — fresh EchoProvider — hit the cache.
        let inner2 = Arc::new(EchoProvider::ok("p"));
        let backend = Arc::new(SqliteCache::open(&path).unwrap()) as Arc<dyn CacheBackend>;
        let cached = CachedProvider::new(inner2.clone(), backend);
        let r = cached
            .chat(&[Message::user("persist")], None, &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(r.message.text(), Some("ok"));
        assert_eq!(
            inner2.calls(),
            0,
            "cached response must come from the sqlite store"
        );
    }
}
