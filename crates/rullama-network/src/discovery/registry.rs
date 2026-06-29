use anyhow::{Context, Result};
use async_trait::async_trait;
use uuid::Uuid;

use super::traits::{Discovery, DiscoveryProtocol};
use crate::identity::AgentIdentity;

/// HTTP-backed central agent registry discovery.
///
/// Agents register their identity with a remote registry service and
/// can discover other registered agents. The registry is accessed via
/// a REST API.
///
/// # Endpoints
///
/// - `POST /agents` — register an agent
/// - `DELETE /agents/{id}` — deregister an agent
/// - `GET /agents` — list all registered agents
/// - `GET /agents/{id}` — look up a specific agent
pub struct RegistryDiscovery {
    /// Base URL of the registry service.
    registry_url: String,
    /// Optional bearer token for authentication.
    api_key: Option<String>,
    /// HTTP client.
    client: reqwest::Client,
}

impl RegistryDiscovery {
    /// Create a new registry discovery client.
    pub fn new(registry_url: impl Into<String>) -> Self {
        Self {
            registry_url: registry_url.into(),
            api_key: None,
            client: reqwest::Client::new(),
        }
    }

    /// Set an API key for authenticated registry access.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Build an authenticated request.
    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{path}", self.registry_url);
        let mut req = self.client.request(method, &url);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        req
    }
}

impl std::fmt::Debug for RegistryDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryDiscovery")
            .field("registry_url", &self.registry_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

#[async_trait]
impl Discovery for RegistryDiscovery {
    async fn register(&self, identity: &AgentIdentity) -> Result<()> {
        let response = self
            .request(reqwest::Method::POST, "/agents")
            .json(identity)
            .send()
            .await
            .context("Failed to register with registry")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Registry registration failed ({status}): {body}");
        }

        Ok(())
    }

    async fn deregister(&self, id: &Uuid) -> Result<()> {
        let response = self
            .request(reqwest::Method::DELETE, &format!("/agents/{id}"))
            .send()
            .await
            .context("Failed to deregister from registry")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Registry deregistration failed ({status}): {body}");
        }

        Ok(())
    }

    async fn discover(&self) -> Result<Vec<AgentIdentity>> {
        let response = self
            .request(reqwest::Method::GET, "/agents")
            .send()
            .await
            .context("Failed to discover agents from registry")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Registry discovery failed ({status}): {body}");
        }

        let agents: Vec<AgentIdentity> = response
            .json()
            .await
            .context("Failed to parse registry response")?;

        Ok(agents)
    }

    async fn lookup(&self, id: &Uuid) -> Result<Option<AgentIdentity>> {
        let response = self
            .request(reqwest::Method::GET, &format!("/agents/{id}"))
            .send()
            .await
            .context("Failed to look up agent in registry")?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Registry lookup failed ({status}): {body}");
        }

        let agent: AgentIdentity = response
            .json()
            .await
            .context("Failed to parse registry response")?;

        Ok(Some(agent))
    }

    fn protocol(&self) -> DiscoveryProtocol {
        DiscoveryProtocol::Registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_debug_redacts_key() {
        let d = RegistryDiscovery::new("https://registry.example.com").with_api_key("secret-key");
        let debug = format!("{d:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("secret-key"));
    }

    #[test]
    fn registry_protocol() {
        let d = RegistryDiscovery::new("https://example.com");
        assert_eq!(d.protocol(), DiscoveryProtocol::Registry);
    }
}
