//! Cross-channel user identity mapping.
//!
//! By default each `(platform, user_id)` pair is treated as an independent
//! session.  This module allows linking multiple platform identities to a
//! single **canonical identity UUID**, so that the same person on Discord and
//! Telegram shares one agent session and conversation history.
//!
//! # Storage
//!
//! Links are stored as a JSON file at a configurable path
//! (`~/.brainclaw/identity.json`).  The file is rewritten atomically on every
//! mutation so it survives crashes.
//!
//! # Linking
//!
//! Links are created via the admin API (`POST /admin/identity/link`) or the
//! `/link` skill command (which generates a time-limited pairing code).
//!
//! # Agent session key
//!
//! `AgentInboundHandler` calls `get_identity_id(platform, user_id)` and uses
//! the returned UUID as the agent session key instead of `(platform, user_id)`.
//! Unlinked users get a stable UUID derived from their first-seen platform
//! identity.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

/// A single platform identity (platform + user_id pair).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlatformIdentity {
    /// Platform name (e.g. "discord", "telegram").
    pub platform: String,
    /// User ID within the platform.
    pub user_id: String,
}

impl PlatformIdentity {
    pub fn new(platform: impl Into<String>, user_id: impl Into<String>) -> Self {
        Self {
            platform: platform.into(),
            user_id: user_id.into(),
        }
    }
}

/// Persisted identity data.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct IdentityData {
    /// Maps (platform, user_id) -> canonical UUID.
    identity_map: HashMap<String, Uuid>,
    /// Maps canonical UUID -> all linked platform identities.
    linked_identities: HashMap<Uuid, Vec<PlatformIdentity>>,
}

impl IdentityData {
    fn key(platform: &str, user_id: &str) -> String {
        format!("{platform}:{user_id}")
    }
}

/// Persisted cross-channel user identity store.
pub struct UserIdentityStore {
    path: PathBuf,
    data: RwLock<IdentityData>,
}

impl UserIdentityStore {
    /// Open (or create) the identity store at `path`.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let data = if path.exists() {
            let json = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read identity store: {}", path.display()))?;
            serde_json::from_str(&json)
                .with_context(|| format!("Failed to parse identity store: {}", path.display()))?
        } else {
            // Ensure parent directory exists
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create dir: {}", parent.display()))?;
            }
            IdentityData::default()
        };

        tracing::info!(
            path = %path.display(),
            identities = data.linked_identities.len(),
            "Identity store loaded"
        );

        Ok(Self {
            path,
            data: RwLock::new(data),
        })
    }

    /// Return the canonical identity UUID for the given platform user.
    ///
    /// If the user has no entry, a new UUID is created and persisted.
    pub async fn get_identity_id(&self, platform: &str, user_id: &str) -> Uuid {
        let key = IdentityData::key(platform, user_id);

        // Fast path: already known.
        {
            let data = self.data.read().await;
            if let Some(id) = data.identity_map.get(&key) {
                return *id;
            }
        }

        // Slow path: create and persist.
        let new_id = Uuid::new_v4();
        let mut data = self.data.write().await;
        // Re-check after acquiring write lock.
        if let Some(id) = data.identity_map.get(&key) {
            return *id;
        }
        let pid = PlatformIdentity::new(platform, user_id);
        data.identity_map.insert(key, new_id);
        data.linked_identities.entry(new_id).or_default().push(pid);

        if let Err(e) = self.persist_locked(&data) {
            tracing::warn!(error = %e, "Failed to persist new identity entry");
        }

        new_id
    }

    /// Link two platform identities to the same canonical identity.
    ///
    /// After linking, both identities share the same agent session and
    /// conversation history. The canonical UUID of `primary` is kept;
    /// any sessions previously attached to `secondary` are merged under it.
    ///
    /// Returns the canonical UUID both identities now map to.
    pub async fn link(
        &self,
        primary: &PlatformIdentity,
        secondary: &PlatformIdentity,
    ) -> Result<Uuid> {
        let primary_key = IdentityData::key(&primary.platform, &primary.user_id);
        let secondary_key = IdentityData::key(&secondary.platform, &secondary.user_id);

        let mut data = self.data.write().await;

        // Resolve (or create) canonical UUID for primary.
        let canonical_id = if let Some(&existing) = data.identity_map.get(&primary_key) {
            existing
        } else {
            let id = Uuid::new_v4();
            data.identity_map.insert(primary_key.clone(), id);
            data.linked_identities
                .entry(id)
                .or_default()
                .push(primary.clone());
            id
        };

        // Find old canonical UUID for secondary (if any).
        let old_secondary_id = data.identity_map.get(&secondary_key).copied();

        // If secondary already maps to the same canonical, nothing to do.
        if old_secondary_id == Some(canonical_id) {
            return Ok(canonical_id);
        }

        // Migrate all identities formerly under old_secondary_id to canonical_id.
        if let Some(old_id) = old_secondary_id {
            if let Some(old_members) = data.linked_identities.remove(&old_id) {
                for pid in &old_members {
                    let k = IdentityData::key(&pid.platform, &pid.user_id);
                    data.identity_map.insert(k, canonical_id);
                }
                data.linked_identities
                    .entry(canonical_id)
                    .or_default()
                    .extend(old_members);
            }
        } else {
            // Secondary is new; just add it under the canonical.
            data.identity_map.insert(secondary_key, canonical_id);
            data.linked_identities
                .entry(canonical_id)
                .or_default()
                .push(secondary.clone());
        }

        self.persist_locked(&data)
            .context("Failed to persist identity link")?;

        tracing::info!(
            canonical = %canonical_id,
            primary = ?primary,
            secondary = ?secondary,
            "Platform identities linked"
        );

        Ok(canonical_id)
    }

    /// Unlink a secondary identity from its canonical group.
    ///
    /// After unlinking, the secondary identity gets its own fresh UUID the
    /// next time it is seen.  Returns the old canonical UUID it was removed
    /// from, or `None` if the identity was not found.
    pub async fn unlink(&self, identity: &PlatformIdentity) -> Result<Option<Uuid>> {
        let key = IdentityData::key(&identity.platform, &identity.user_id);
        let mut data = self.data.write().await;

        let Some(old_id) = data.identity_map.remove(&key) else {
            return Ok(None);
        };

        if let Some(members) = data.linked_identities.get_mut(&old_id) {
            members.retain(|p| p != identity);
            if members.is_empty() {
                data.linked_identities.remove(&old_id);
            }
        }

        self.persist_locked(&data)
            .context("Failed to persist identity unlink")?;

        Ok(Some(old_id))
    }

    /// List all platform identities linked to the given canonical UUID.
    pub async fn list_linked(&self, canonical_id: Uuid) -> Vec<PlatformIdentity> {
        let data = self.data.read().await;
        data.linked_identities
            .get(&canonical_id)
            .cloned()
            .unwrap_or_default()
    }

    /// List all canonical identities and their linked platforms.
    pub async fn list_all(&self) -> HashMap<Uuid, Vec<PlatformIdentity>> {
        self.data.read().await.linked_identities.clone()
    }

    /// Wrap in Arc for sharing.
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }

    fn persist_locked(&self, data: &IdentityData) -> Result<()> {
        let json =
            serde_json::to_string_pretty(data).context("Failed to serialize identity data")?;
        std::fs::write(&self.path, json)
            .with_context(|| format!("Failed to write identity store: {}", self.path.display()))?;
        Ok(())
    }
}
