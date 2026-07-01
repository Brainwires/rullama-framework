use anyhow::Result;
use rullama_homeauto::MatterDevice;
/// Fabric directory resolution and device listing helpers.
use std::path::{Path, PathBuf};

/// Resolve the fabric storage directory:
/// 1. Explicit `--fabric-dir` arg (if provided)
/// 2. `~/.local/share/matter-tool/` on Linux, `~/Library/Application Support/matter-tool/` on macOS
/// 3. `./.matter-tool/` as a final fallback
pub fn resolve_fabric_dir(override_path: Option<&PathBuf>) -> PathBuf {
    if let Some(p) = override_path {
        return p.clone();
    }
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("matter-tool")
}

/// Read commissioned devices from fabric storage.
///
/// The `MatterController` persists devices as JSON under `<fabric_dir>/devices.json`.
/// This function loads them without opening a full controller session.
pub async fn load_devices(fabric_dir: &Path) -> Result<Vec<MatterDevice>> {
    let path = fabric_dir.join("devices.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let raw = tokio::fs::read_to_string(&path).await?;
    let devices: Vec<MatterDevice> = serde_json::from_str(&raw)?;
    Ok(devices)
}

/// Interactive "are you sure?" prompt for destructive operations.
/// Returns `true` if the user typed exactly `yes`.
#[allow(dead_code)]
pub fn confirm_destructive(prompt: &str) -> bool {
    use std::io::{self, Write};
    print!("{prompt} [type 'yes' to confirm]: ");
    io::stdout().flush().ok();
    let mut input = String::new();
    io::stdin().read_line(&mut input).ok();
    input.trim() == "yes"
}

#[cfg(test)]
mod tests {
    use super::*;
    use rullama_homeauto::MatterDevice;

    #[test]
    fn resolve_fabric_dir_uses_override() {
        let custom = PathBuf::from("/tmp/my-fabric");
        let result = resolve_fabric_dir(Some(&custom));
        assert_eq!(result, custom);
    }

    #[test]
    fn resolve_fabric_dir_default_ends_with_matter_tool() {
        let result = resolve_fabric_dir(None);
        assert!(
            result.ends_with("matter-tool"),
            "default path should end with 'matter-tool', got: {}",
            result.display()
        );
    }

    #[tokio::test]
    async fn load_devices_missing_file_returns_empty() {
        let dir = PathBuf::from("/tmp/no-such-matter-tool-dir-xyz123");
        let devices = load_devices(&dir).await.unwrap();
        assert!(devices.is_empty());
    }

    #[tokio::test]
    async fn load_devices_reads_json() {
        let dir = tempfile::tempdir().unwrap();
        let device = MatterDevice::new(42);
        let json = serde_json::to_string(&vec![device]).unwrap();
        tokio::fs::write(dir.path().join("devices.json"), json)
            .await
            .unwrap();

        let devices = load_devices(dir.path()).await.unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].node_id, 42);
    }

    #[tokio::test]
    async fn load_devices_rejects_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(dir.path().join("devices.json"), b"not json")
            .await
            .unwrap();

        let result = load_devices(dir.path()).await;
        assert!(result.is_err());
    }
}
