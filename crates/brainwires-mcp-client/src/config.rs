use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
#[cfg(feature = "native")]
use std::fs;
#[cfg(feature = "native")]
use std::path::PathBuf;

/// Configuration for a single MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Unique name for this server.
    pub name: String,
    /// Command to launch the server.
    pub command: String,
    /// Arguments to pass to the command.
    pub args: Vec<String>,
    /// Optional environment variables.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<std::collections::HashMap<String, String>>,
}

#[cfg(feature = "native")]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct McpConfigFile {
    servers: Vec<McpServerConfig>,
}

/// Manages MCP server configurations on disk.
#[cfg(feature = "native")]
pub struct McpConfigManager {
    config_path: PathBuf,
    servers: Vec<McpServerConfig>,
}

#[cfg(feature = "native")]
impl McpConfigManager {
    /// Create a new config manager with empty servers list
    pub fn new() -> Result<Self> {
        let config_path = Self::get_config_path()?;
        Ok(Self {
            config_path,
            servers: vec![],
        })
    }

    /// Load config from file, create default if doesn't exist
    pub fn load() -> Result<Self> {
        let config_path = Self::get_config_path()?;

        // Ensure config directory exists
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent).context("Failed to create config directory")?;
        }

        // Load config file if it exists
        let servers = if config_path.exists() {
            let contents =
                fs::read_to_string(&config_path).context("Failed to read MCP config file")?;

            let config: McpConfigFile =
                serde_json::from_str(&contents).context("Failed to parse MCP config file")?;

            config.servers
        } else {
            // Create default config file
            let default_config = McpConfigFile { servers: vec![] };
            let json = serde_json::to_string_pretty(&default_config)?;
            fs::write(&config_path, json)?;
            vec![]
        };

        Ok(Self {
            config_path,
            servers,
        })
    }

    /// Save config to file
    pub fn save(&self) -> Result<()> {
        let config = McpConfigFile {
            servers: self.servers.clone(),
        };

        let json =
            serde_json::to_string_pretty(&config).context("Failed to serialize MCP config")?;

        fs::write(&self.config_path, json).context("Failed to write MCP config file")?;

        Ok(())
    }

    /// Add a new server configuration
    pub fn add_server(&mut self, config: McpServerConfig) -> Result<()> {
        // Check for duplicate names
        if self.servers.iter().any(|s| s.name == config.name) {
            anyhow::bail!("Server with name '{}' already exists", config.name);
        }

        self.servers.push(config);
        self.save()?;
        Ok(())
    }

    /// Remove a server configuration
    pub fn remove_server(&mut self, name: &str) -> Result<()> {
        let initial_len = self.servers.len();
        self.servers.retain(|s| s.name != name);

        if self.servers.len() == initial_len {
            anyhow::bail!("Server '{}' not found", name);
        }

        self.save()?;
        Ok(())
    }

    /// Get all server configurations
    pub fn get_servers(&self) -> &[McpServerConfig] {
        &self.servers
    }

    /// Get a specific server configuration
    pub fn get_server(&self, name: &str) -> Option<&McpServerConfig> {
        self.servers.iter().find(|s| s.name == name)
    }

    /// Get config file path
    fn get_config_path() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Failed to get home directory")?;
        Ok(home.join(".brainwires").join("mcp-config.json"))
    }
}

#[cfg(feature = "native")]
impl Default for McpConfigManager {
    fn default() -> Self {
        Self::new().expect("Failed to create MCP config manager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "native")]
    #[test]
    fn test_config_manager_creation() {
        let manager = McpConfigManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_server_config_serialization() {
        let config = McpServerConfig {
            name: "test-server".to_string(),
            command: "npx".to_string(),
            args: vec!["-y".to_string(), "test-mcp-server".to_string()],
            env: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: McpServerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, "test-server");
        assert_eq!(deserialized.command, "npx");
        assert_eq!(deserialized.args.len(), 2);
    }

    #[test]
    fn test_server_config_with_env() {
        let mut env = std::collections::HashMap::new();
        env.insert("API_KEY".to_string(), "test-key".to_string());

        let config = McpServerConfig {
            name: "test".to_string(),
            command: "node".to_string(),
            args: vec!["server.js".to_string()],
            env: Some(env),
        };

        assert!(config.env.is_some());
        assert_eq!(
            config.env.as_ref().unwrap().get("API_KEY").unwrap(),
            "test-key"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_get_servers_empty() {
        let manager = McpConfigManager::new().unwrap();
        assert_eq!(manager.get_servers().len(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_get_server_not_found() {
        let manager = McpConfigManager::new().unwrap();
        assert!(manager.get_server("nonexistent").is_none());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_default_manager() {
        let manager = McpConfigManager::default();
        assert_eq!(manager.get_servers().len(), 0);
    }
}
