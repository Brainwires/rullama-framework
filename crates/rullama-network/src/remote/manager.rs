//! Remote Bridge Manager
//!
//! Manages the lifecycle of the remote control bridge.
//!
//! Unlike the CLI version, this manager uses trait objects for configuration
//! and agent spawning, making it reusable outside the CLI.
//!
//! CLI-specific functions (`should_auto_start()`, `try_auto_start()`) remain
//! in the CLI's adapter module.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::{RwLock, broadcast};
use tokio::task::JoinHandle;

use super::bridge::{BridgeConfig, RemoteBridge};
use crate::traits::{AgentSpawner, BridgeConfigProvider};

/// Internal manager state
#[derive(Default)]
struct ManagerState {
    /// Handle to the bridge task
    task_handle: Option<JoinHandle<()>>,
    /// Shutdown signal sender
    shutdown_tx: Option<broadcast::Sender<()>>,
    /// Whether the manager is currently running
    running: bool,
}

/// Remote Bridge Manager
///
/// Provides a high-level interface for managing the remote control bridge.
/// Uses `BridgeConfigProvider` for configuration and `AgentSpawner` for
/// process creation, decoupled from CLI-specific types.
pub struct RemoteBridgeManager {
    state: Arc<RwLock<ManagerState>>,
    /// Configuration provider (replaces ConfigManager + SessionManager)
    config_provider: Box<dyn BridgeConfigProvider>,
    /// Agent spawner (for remote agent creation)
    agent_spawner: Arc<dyn AgentSpawner>,
    /// Sessions directory (injected platform path)
    sessions_dir: PathBuf,
    /// CLI version string (injected)
    version: String,
    /// Attachment storage directory (injected platform path)
    attachment_dir: PathBuf,
}

impl RemoteBridgeManager {
    /// Create a new remote bridge manager
    ///
    /// # Arguments
    /// * `config_provider` - Provides remote bridge configuration and API keys
    /// * `agent_spawner` - Creates new agent processes on demand
    /// * `sessions_dir` - Directory containing agent session files
    /// * `version` - CLI version string
    /// * `attachment_dir` - Directory for storing received attachments
    pub fn new(
        config_provider: Box<dyn BridgeConfigProvider>,
        agent_spawner: Arc<dyn AgentSpawner>,
        sessions_dir: PathBuf,
        version: String,
        attachment_dir: PathBuf,
    ) -> Self {
        Self {
            state: Arc::new(RwLock::new(ManagerState::default())),
            config_provider,
            agent_spawner,
            sessions_dir,
            version,
            attachment_dir,
        }
    }

    /// Check if remote control is enabled
    pub fn is_enabled(&self) -> Result<bool> {
        match self.config_provider.get_remote_config()? {
            Some(_) => Ok(true),
            None => Ok(false),
        }
    }

    /// Build a BridgeConfig from the provider's settings
    fn build_bridge_config(&self) -> Result<Option<BridgeConfig>> {
        let remote_config = match self.config_provider.get_remote_config()? {
            Some(c) => c,
            None => return Ok(None),
        };

        // Get API key: prefer the one from config, fall back to key store
        let api_key = if !remote_config.api_key.is_empty() {
            remote_config.api_key
        } else {
            match self.config_provider.get_api_key()? {
                Some(key) => key.to_string(),
                None => {
                    tracing::warn!("Remote control enabled but no API key available");
                    return Ok(None);
                }
            }
        };

        Ok(Some(BridgeConfig {
            backend_url: remote_config.backend_url,
            api_key,
            heartbeat_interval_secs: remote_config.heartbeat_interval_secs,
            reconnect_delay_secs: remote_config.reconnect_delay_secs,
            max_reconnect_attempts: remote_config.max_reconnect_attempts,
            version: self.version.clone(),
            sessions_dir: self.sessions_dir.clone(),
            attachment_dir: self.attachment_dir.clone(),
        }))
    }

    /// Start the remote bridge with an explicit BridgeConfig
    ///
    /// Returns `Ok(Some(handle))` if started, `Ok(None)` if already running.
    pub async fn start_with_config(&self, config: BridgeConfig) -> Result<Option<JoinHandle<()>>> {
        let mut state = self.state.write().await;

        if state.running {
            tracing::debug!("Remote bridge already running");
            return Ok(None);
        }

        if config.api_key.is_empty() {
            anyhow::bail!("No API key configured");
        }

        tracing::info!("Starting remote control bridge to {}", config.backend_url);

        let agent_spawner = Arc::clone(&self.agent_spawner);
        let mut bridge = RemoteBridge::new(config, Some(agent_spawner));

        // Create shutdown channel for graceful shutdown
        let (shutdown_tx, _) = broadcast::channel(1);
        bridge.set_shutdown_tx(shutdown_tx.clone());
        state.shutdown_tx = Some(shutdown_tx);

        // Spawn the bridge task
        let handle = tokio::spawn(async move {
            if let Err(e) = bridge.run().await {
                tracing::error!("Remote bridge error: {}", e);
            }
        });

        // Wait briefly for connection
        tokio::time::sleep(Duration::from_millis(100)).await;

        state.running = true;

        Ok(Some(handle))
    }

    /// Start the remote bridge using configuration from the provider
    pub async fn start_from_config(&self) -> Result<Option<JoinHandle<()>>> {
        let config = match self.build_bridge_config()? {
            Some(c) => c,
            None => {
                tracing::debug!("Remote control not enabled or not configured");
                return Ok(None);
            }
        };

        self.start_with_config(config).await
    }

    /// Stop the remote bridge gracefully
    pub async fn stop(&self) {
        let mut state = self.state.write().await;

        if !state.running {
            return;
        }

        tracing::info!("Stopping remote control bridge");

        // Send graceful shutdown signal
        if let Some(tx) = &state.shutdown_tx {
            let _ = tx.send(());
            tracing::info!("Sent graceful shutdown signal to bridge");

            // Wait briefly for the disconnect message to be sent
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        // Abort if still running
        if let Some(handle) = state.task_handle.take() {
            handle.abort();
        }

        state.shutdown_tx = None;
        state.running = false;
    }

    /// Check if the bridge is running
    pub async fn is_running(&self) -> bool {
        self.state.read().await.running
    }

    /// Get bridge status for display (sync version)
    pub fn status(&self) -> RemoteBridgeStatus {
        match self.state.try_read() {
            Ok(state) => {
                if !state.running {
                    RemoteBridgeStatus::Disconnected
                } else {
                    RemoteBridgeStatus::Connected
                }
            }
            Err(_) => RemoteBridgeStatus::Connecting,
        }
    }

    /// Get bridge status for display (async version)
    pub async fn status_async(&self) -> RemoteBridgeStatus {
        let state = self.state.read().await;

        if !state.running {
            RemoteBridgeStatus::Disconnected
        } else {
            RemoteBridgeStatus::Connected
        }
    }
}

/// Bridge status for display
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteBridgeStatus {
    /// Bridge is disconnected
    Disconnected,
    /// Bridge is connecting
    Connecting,
    /// Bridge is connected
    Connected,
    /// Bridge is authenticated and ready
    Authenticated,
    /// Bridge encountered an error
    Error(String),
}

impl std::fmt::Display for RemoteBridgeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected => write!(f, "Disconnected"),
            Self::Connecting => write!(f, "Connecting"),
            Self::Connected => write!(f, "Connected"),
            Self::Authenticated => write!(f, "Authenticated"),
            Self::Error(e) => write!(f, "Error: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{BridgeConfigProvider, RemoteBridgeConfig};
    use zeroize::Zeroizing;

    struct MockConfigProvider {
        config: Option<RemoteBridgeConfig>,
    }

    impl MockConfigProvider {
        fn disabled() -> Self {
            Self { config: None }
        }

        fn enabled() -> Self {
            Self {
                config: Some(RemoteBridgeConfig {
                    backend_url: "https://test.example.com".to_string(),
                    api_key: "test-key".to_string(),
                    heartbeat_interval_secs: 5,
                    reconnect_delay_secs: 5,
                    max_reconnect_attempts: 3,
                }),
            }
        }
    }

    impl BridgeConfigProvider for MockConfigProvider {
        fn get_remote_config(&self) -> Result<Option<RemoteBridgeConfig>> {
            Ok(self.config.clone())
        }

        fn get_api_key(&self) -> Result<Option<Zeroizing<String>>> {
            Ok(Some(Zeroizing::new("test-api-key".to_string())))
        }
    }

    struct MockSpawner;

    #[async_trait::async_trait]
    impl AgentSpawner for MockSpawner {
        async fn spawn_agent(
            &self,
            _session_id: &str,
            _model: Option<String>,
            _working_directory: Option<PathBuf>,
        ) -> Result<PathBuf> {
            Ok(PathBuf::from("/tmp/test.sock"))
        }
    }

    fn make_manager(config_provider: Box<dyn BridgeConfigProvider>) -> RemoteBridgeManager {
        RemoteBridgeManager::new(
            config_provider,
            Arc::new(MockSpawner),
            PathBuf::from("/tmp/test-sessions"),
            "0.1.0-test".to_string(),
            PathBuf::from("/tmp/test-attachments"),
        )
    }

    #[test]
    fn test_remote_bridge_status_display() {
        assert_eq!(
            format!("{}", RemoteBridgeStatus::Disconnected),
            "Disconnected"
        );
        assert_eq!(format!("{}", RemoteBridgeStatus::Connected), "Connected");
        assert_eq!(
            format!("{}", RemoteBridgeStatus::Authenticated),
            "Authenticated"
        );
    }

    #[tokio::test]
    async fn test_manager_not_running_by_default() {
        let manager = make_manager(Box::new(MockConfigProvider::disabled()));
        assert!(!manager.is_running().await);
    }

    #[test]
    fn test_is_enabled_disabled() {
        let manager = make_manager(Box::new(MockConfigProvider::disabled()));
        assert!(!manager.is_enabled().unwrap());
    }

    #[test]
    fn test_is_enabled_enabled() {
        let manager = make_manager(Box::new(MockConfigProvider::enabled()));
        assert!(manager.is_enabled().unwrap());
    }

    #[test]
    fn test_build_bridge_config_disabled() {
        let manager = make_manager(Box::new(MockConfigProvider::disabled()));
        assert!(manager.build_bridge_config().unwrap().is_none());
    }

    #[test]
    fn test_build_bridge_config_enabled() {
        let manager = make_manager(Box::new(MockConfigProvider::enabled()));
        let config = manager.build_bridge_config().unwrap();
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.backend_url, "https://test.example.com");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.version, "0.1.0-test");
    }
}
