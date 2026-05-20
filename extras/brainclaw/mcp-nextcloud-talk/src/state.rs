//! Per-room `last_message_id` cursor — persisted as JSON with
//! atomic-rename.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Cursor state — map of room token → last-seen message id.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CursorState {
    /// Map of room token → last-seen message id.
    pub last_message_id: BTreeMap<String, u64>,
}

impl CursorState {
    /// Load cursor state from disk, empty-on-missing.
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

    /// Persist with atomic rename.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create state dir {}", parent.display()))?;
        }
        let tmp: PathBuf = {
            let mut p = path.as_os_str().to_owned();
            p.push(".tmp");
            PathBuf::from(p)
        };
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&tmp, json).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let td = tempfile::tempdir().unwrap();
        let p = td.path().join("c.json");
        let mut s = CursorState::default();
        s.last_message_id.insert("room-a".into(), 42);
        s.save(&p).unwrap();
        let loaded = CursorState::load(&p).unwrap();
        assert_eq!(loaded.last_message_id.get("room-a").copied(), Some(42));
    }
}
