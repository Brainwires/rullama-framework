//! Persistent state backing the pairing flow.
//!
//! [`PairingStore`] is a JSON-file-backed store that tracks:
//!
//! - **approved peers** (`<channel>:<user_id>` strings) who are allowed to
//!   DM the bot;
//! - **pending codes** — 6-digit codes issued to unknown peers, waiting for
//!   operator approval or rejection;
//! - a static **allowlist** loaded from config that ORs with the dynamic
//!   approvals above.
//!
//! Writes go through a tokio task so the hot path (`issue_code`, `approve_*`)
//! is not blocked on file I/O. Reads at startup use `std::fs` since they
//! happen once.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, mpsc};

/// Key helper: build the canonical `"<channel>:<user_id>"` string used
/// throughout the pairing subsystem.
pub fn peer_key(channel: &str, user_id: &str) -> String {
    format!("{channel}:{user_id}")
}

/// A pending approval code issued to a peer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingCode {
    /// The 6-digit code.
    pub code: String,
    /// Channel name (e.g. `"discord"`).
    pub channel: String,
    /// Platform user id this code was issued for.
    pub user_id: String,
    /// Display name the channel uses for the peer (best-effort).
    pub peer_display: String,
    /// When the code was issued.
    pub created_at: DateTime<Utc>,
    /// When the code expires.
    pub expires_at: DateTime<Utc>,
}

impl PendingCode {
    /// Whether `now` is at-or-past `expires_at`.
    pub fn is_expired(&self, now: DateTime<Utc>) -> bool {
        now >= self.expires_at
    }
}

/// Serialisable state persisted to disk.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PairingState {
    /// Approved peers keyed by `<channel>:<user_id>`.
    #[serde(default)]
    approved: HashSet<String>,
    /// Pending codes keyed by `code` string.
    #[serde(default)]
    pending: HashMap<String, PendingCode>,
}

/// JSON-backed pairing state store. Cheap to clone — the state is kept
/// behind an [`Arc`].
#[derive(Clone)]
pub struct PairingStore {
    path: PathBuf,
    state: Arc<RwLock<PairingState>>,
    /// Static allow-list from config, keyed by `<channel>:<user_id>`.
    static_allow: Arc<RwLock<HashSet<String>>>,
    /// Channel used to ask the background writer to persist state.
    save_tx: mpsc::UnboundedSender<PairingState>,
}

impl std::fmt::Debug for PairingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairingStore")
            .field("path", &self.path)
            .finish()
    }
}

impl PairingStore {
    /// Load a store from `path`, creating parent directories as needed.
    ///
    /// If the file does not exist, starts with empty state. Errors reading
    /// a corrupt file are surfaced to the caller.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("creating pairing store parent dir {}", parent.display())
            })?;
        }

        let state: PairingState = if path.exists() {
            let data = std::fs::read_to_string(&path)
                .with_context(|| format!("reading pairing store {}", path.display()))?;
            if data.trim().is_empty() {
                PairingState::default()
            } else {
                serde_json::from_str(&data)
                    .with_context(|| format!("parsing pairing store {}", path.display()))?
            }
        } else {
            PairingState::default()
        };

        let state = Arc::new(RwLock::new(state));
        let (save_tx, mut save_rx) = mpsc::unbounded_channel::<PairingState>();
        let write_path = path.clone();

        // Background writer: serialises save requests, coalescing bursts
        // by always using the most recent state available in the queue.
        tokio::spawn(async move {
            while let Some(mut snapshot) = save_rx.recv().await {
                // Drain any further pending snapshots so we only write the
                // latest state once per burst.
                while let Ok(next) = save_rx.try_recv() {
                    snapshot = next;
                }
                if let Err(e) = write_atomic(&write_path, &snapshot).await {
                    tracing::warn!(path = %write_path.display(), error = %e, "pairing store: write failed");
                }
            }
        });

        Ok(Self {
            path,
            state,
            static_allow: Arc::new(RwLock::new(HashSet::new())),
            save_tx,
        })
    }

    /// The path the store writes to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Replace the static allow-list loaded from config.
    ///
    /// The static list ORs with the dynamic approved set — listed peers
    /// are pre-approved without needing a pairing code.
    pub async fn set_allowlist(&self, entries: Vec<String>) {
        let mut guard = self.static_allow.write().await;
        *guard = entries.into_iter().collect();
    }

    /// Is this peer approved to DM the bot?
    ///
    /// Checks both the dynamic approved set and the static allow-list.
    pub async fn is_approved(&self, channel: &str, user_id: &str) -> bool {
        let key = peer_key(channel, user_id);
        if self.static_allow.read().await.contains(&key) {
            return true;
        }
        self.state.read().await.approved.contains(&key)
    }

    /// Explicitly add a peer to the approved set and persist.
    pub async fn approve(&self, channel: &str, user_id: &str) -> Result<()> {
        let key = peer_key(channel, user_id);
        let snapshot = {
            let mut guard = self.state.write().await;
            guard.approved.insert(key);
            guard.clone()
        };
        self.request_save(snapshot);
        Ok(())
    }

    /// Remove a peer from the approved set and persist.
    pub async fn revoke(&self, channel: &str, user_id: &str) -> Result<()> {
        let key = peer_key(channel, user_id);
        let snapshot = {
            let mut guard = self.state.write().await;
            guard.approved.remove(&key);
            guard.clone()
        };
        self.request_save(snapshot);
        Ok(())
    }

    /// Find the pending code currently issued for this peer, if any.
    ///
    /// Expired codes are pruned as a side-effect.
    pub async fn pending_for_peer(&self, channel: &str, user_id: &str) -> Option<PendingCode> {
        let now = Utc::now();
        let mut snapshot_to_save: Option<PairingState> = None;
        let out = {
            let mut guard = self.state.write().await;
            let before = guard.pending.len();
            guard.pending.retain(|_, pc| !pc.is_expired(now));
            let pruned = before != guard.pending.len();
            if pruned {
                snapshot_to_save = Some(guard.clone());
            }
            guard
                .pending
                .values()
                .find(|pc| pc.channel == channel && pc.user_id == user_id)
                .cloned()
        };
        if let Some(s) = snapshot_to_save {
            self.request_save(s);
        }
        out
    }

    /// Issue a fresh 6-digit code for `channel:user_id`, persist, return it.
    ///
    /// Collision-retries so the new code is distinct from every currently
    /// active pending code. Any existing pending codes for this peer are
    /// removed first — callers that want "reuse if pending" should call
    /// [`Self::pending_for_peer`] first.
    pub async fn issue_code(
        &self,
        channel: &str,
        user_id: &str,
        peer_display: &str,
        ttl: Duration,
    ) -> Result<PendingCode> {
        let now = Utc::now();
        let expires_at = now
            + chrono::Duration::from_std(ttl).unwrap_or_else(|_| chrono::Duration::seconds(900));

        let snapshot = {
            let mut guard = self.state.write().await;

            // Prune expired + any existing entries for this peer.
            guard.pending.retain(|_, pc| {
                !(pc.is_expired(now) || pc.channel == channel && pc.user_id == user_id)
            });

            // Generate a fresh code that doesn't collide with an active
            // pending one.
            let code = loop {
                let candidate = generate_code();
                if !guard.pending.contains_key(&candidate) {
                    break candidate;
                }
            };

            let pc = PendingCode {
                code: code.clone(),
                channel: channel.to_string(),
                user_id: user_id.to_string(),
                peer_display: peer_display.to_string(),
                created_at: now,
                expires_at,
            };
            guard.pending.insert(code.clone(), pc);
            guard.clone()
        };

        // Locate the code we just inserted to return it.
        let pc = snapshot
            .pending
            .values()
            .find(|p| p.channel == channel && p.user_id == user_id)
            .cloned()
            .expect("issued code just inserted");

        self.request_save(snapshot);
        Ok(pc)
    }

    /// Approve a peer by code, atomically.
    ///
    /// Returns `Some((channel, user_id))` on success. Returns `None` if the
    /// code does not exist or has expired. A successful call moves the
    /// peer into the approved set and removes the pending entry.
    pub async fn approve_by_code(&self, code: &str) -> Result<Option<(String, String)>> {
        let now = Utc::now();
        let mut save: Option<PairingState> = None;
        let out = {
            let mut guard = self.state.write().await;
            // Prune expired
            guard.pending.retain(|_, pc| !pc.is_expired(now));
            match guard.pending.remove(code) {
                Some(pc) => {
                    let key = peer_key(&pc.channel, &pc.user_id);
                    guard.approved.insert(key);
                    save = Some(guard.clone());
                    Some((pc.channel, pc.user_id))
                }
                None => None,
            }
        };
        if let Some(s) = save {
            self.request_save(s);
        }
        Ok(out)
    }

    /// Reject a pending code — remove it without approving. Returns `true`
    /// if a matching entry was present.
    pub async fn reject_by_code(&self, code: &str) -> Result<bool> {
        let now = Utc::now();
        let mut save: Option<PairingState> = None;
        let removed = {
            let mut guard = self.state.write().await;
            guard.pending.retain(|_, pc| !pc.is_expired(now));
            let existed = guard.pending.remove(code).is_some();
            if existed {
                save = Some(guard.clone());
            }
            existed
        };
        if let Some(s) = save {
            self.request_save(s);
        }
        Ok(removed)
    }

    /// List all currently-valid pending codes.
    pub async fn list_pending(&self) -> Vec<PendingCode> {
        let now = Utc::now();
        let mut save: Option<PairingState> = None;
        let out = {
            let mut guard = self.state.write().await;
            let before = guard.pending.len();
            guard.pending.retain(|_, pc| !pc.is_expired(now));
            if before != guard.pending.len() {
                save = Some(guard.clone());
            }
            guard.pending.values().cloned().collect::<Vec<_>>()
        };
        if let Some(s) = save {
            self.request_save(s);
        }
        out
    }

    /// List all approved peers (`<channel>:<user_id>` keys), including
    /// entries from the static allowlist.
    pub async fn list_approved(&self) -> Vec<String> {
        let mut out: Vec<String> = self.state.read().await.approved.iter().cloned().collect();
        for entry in self.static_allow.read().await.iter() {
            if !out.contains(entry) {
                out.push(entry.clone());
            }
        }
        out.sort();
        out
    }

    /// Drop any expired pending codes and persist if anything changed.
    pub async fn prune_expired(&self) {
        let now = Utc::now();
        let mut save: Option<PairingState> = None;
        {
            let mut guard = self.state.write().await;
            let before = guard.pending.len();
            guard.pending.retain(|_, pc| !pc.is_expired(now));
            if before != guard.pending.len() {
                save = Some(guard.clone());
            }
        }
        if let Some(s) = save {
            self.request_save(s);
        }
    }

    /// Force a synchronous save to disk (used by tests).
    #[cfg(test)]
    async fn save_now(&self) -> Result<()> {
        let snapshot = self.state.read().await.clone();
        write_atomic(&self.path, &snapshot).await
    }

    fn request_save(&self, snapshot: PairingState) {
        if self.save_tx.send(snapshot).is_err() {
            tracing::warn!(
                path = %self.path.display(),
                "pairing store: background writer gone; state not persisted"
            );
        }
    }
}

/// Generate a 6-digit zero-padded numeric code derived from a fresh UUID's
/// random bits. Not cryptographically secure — sufficient for an
/// operator-gated approval flow.
fn generate_code() -> String {
    let uuid = uuid::Uuid::new_v4();
    let bytes = uuid.as_bytes();
    let n = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) % 1_000_000;
    format!("{n:06}")
}

/// Atomic write via temp file + rename.
async fn write_atomic(path: &Path, state: &PairingState) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("creating pairing store parent dir {}", parent.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_string_pretty(state).context("serializing pairing state")?;
    tokio::fs::write(&tmp, &json)
        .await
        .with_context(|| format!("writing pairing tmp file {}", tmp.display()))?;
    tokio::fs::rename(&tmp, path)
        .await
        .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::time::Duration;

    async fn mkstore() -> (tempfile::TempDir, PairingStore) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pairing.json");
        let store = PairingStore::load(&path).unwrap();
        (dir, store)
    }

    #[tokio::test]
    async fn issue_code_unique() {
        let (_dir, store) = mkstore().await;
        let mut seen: HashSet<String> = HashSet::new();
        for i in 0..100 {
            let user = format!("u{i}");
            let pc = store
                .issue_code("discord", &user, "Alice", Duration::from_secs(900))
                .await
                .unwrap();
            assert!(seen.insert(pc.code.clone()), "duplicate code: {}", pc.code);
            assert_eq!(pc.code.len(), 6);
            assert!(pc.code.chars().all(|c| c.is_ascii_digit()));
        }
    }

    #[tokio::test]
    async fn approve_by_code_atomic() {
        let (_dir, store) = mkstore().await;
        let pc = store
            .issue_code("discord", "alice", "Alice", Duration::from_secs(900))
            .await
            .unwrap();
        let first = store.approve_by_code(&pc.code).await.unwrap();
        assert_eq!(first, Some(("discord".to_string(), "alice".to_string())));
        let second = store.approve_by_code(&pc.code).await.unwrap();
        assert!(second.is_none());
        assert!(store.is_approved("discord", "alice").await);
    }

    #[tokio::test]
    async fn prune_expired_drops_old_entries() {
        let (_dir, store) = mkstore().await;
        // Inject a manually-expired entry directly into state.
        {
            let expired = PendingCode {
                code: "000000".to_string(),
                channel: "discord".to_string(),
                user_id: "ghost".to_string(),
                peer_display: "Ghost".to_string(),
                created_at: Utc::now() - chrono::Duration::seconds(3600),
                expires_at: Utc::now() - chrono::Duration::seconds(60),
            };
            store
                .state
                .write()
                .await
                .pending
                .insert(expired.code.clone(), expired);
        }
        assert_eq!(store.state.read().await.pending.len(), 1);
        store.prune_expired().await;
        assert_eq!(store.state.read().await.pending.len(), 0);
    }

    #[tokio::test]
    async fn allowlist_pre_approves_without_code() {
        let (_dir, store) = mkstore().await;
        store.set_allowlist(vec!["discord:admin".to_string()]).await;
        assert!(store.is_approved("discord", "admin").await);
        assert!(!store.is_approved("discord", "someone-else").await);
    }

    #[tokio::test]
    async fn save_and_reload_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pairing.json");
        {
            let store = PairingStore::load(&path).unwrap();
            let pc = store
                .issue_code("telegram", "bob", "Bob", Duration::from_secs(900))
                .await
                .unwrap();
            store.approve("discord", "alice").await.unwrap();
            // Keep a separate pending around (not approved).
            let _ = store
                .issue_code("discord", "carol", "Carol", Duration::from_secs(900))
                .await
                .unwrap();
            // Ensure we've written synchronously at least once so the
            // background writer has nothing left to flush.
            store.save_now().await.unwrap();
            // Keep pc alive until save.
            let _ = pc;
        }
        let reloaded = PairingStore::load(&path).unwrap();
        assert!(reloaded.is_approved("discord", "alice").await);
        let pending = reloaded.list_pending().await;
        assert_eq!(pending.len(), 2);
    }

    #[tokio::test]
    async fn reject_by_code_removes_without_approving() {
        let (_dir, store) = mkstore().await;
        let pc = store
            .issue_code("discord", "spammer", "Spammer", Duration::from_secs(900))
            .await
            .unwrap();
        assert!(store.reject_by_code(&pc.code).await.unwrap());
        assert!(!store.is_approved("discord", "spammer").await);
        assert!(store.list_pending().await.is_empty());
        assert!(!store.reject_by_code(&pc.code).await.unwrap());
    }

    #[tokio::test]
    async fn revoke_removes_approval() {
        let (_dir, store) = mkstore().await;
        store.approve("discord", "alice").await.unwrap();
        assert!(store.is_approved("discord", "alice").await);
        store.revoke("discord", "alice").await.unwrap();
        assert!(!store.is_approved("discord", "alice").await);
    }
}
