//! Persistent cursor for the polling loop — one `last_guid` per chat.
//!
//! The store is a single JSON file under `state_dir`. Writes are
//! atomic-rename: we serialize to a sibling `.tmp` file and then rename it
//! over the canonical file, ensuring readers never see a half-written
//! cursor after a crash.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Persisted cursor state — maps `chat_guid` → last-seen message guid.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CursorState {
    /// Map of chat guid → last-seen message guid we already forwarded.
    pub last_guid: BTreeMap<String, String>,
}

impl CursorState {
    /// Load state from `path`, returning an empty state if the file does
    /// not exist.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read cursor {}", path.display()))?;
        if raw.trim().is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str(&raw).context("parse cursor json")
    }

    /// Persist state to `path` with atomic rename.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create state dir {}", parent.display()))?;
        }
        let tmp: PathBuf = tmp_sibling(path);
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, json).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
        Ok(())
    }
}

fn tmp_sibling(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".tmp");
    PathBuf::from(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_via_tempdir() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("cursor.json");
        let mut s = CursorState::default();
        s.last_guid.insert("chat-1".into(), "guid-a".into());
        s.save(&p).unwrap();
        let loaded = CursorState::load(&p).unwrap();
        assert_eq!(
            loaded.last_guid.get("chat-1").map(|s| s.as_str()),
            Some("guid-a")
        );
    }

    #[test]
    fn load_missing_is_empty() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("nope.json");
        let loaded = CursorState::load(&p).unwrap();
        assert!(loaded.last_guid.is_empty());
    }

    #[test]
    fn save_overwrites_atomically() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("c.json");
        let mut s = CursorState::default();
        s.last_guid.insert("chat".into(), "v1".into());
        s.save(&p).unwrap();
        s.last_guid.insert("chat".into(), "v2".into());
        s.save(&p).unwrap();
        let loaded = CursorState::load(&p).unwrap();
        assert_eq!(loaded.last_guid.get("chat").map(|s| s.as_str()), Some("v2"));
    }
}
