//! Registry Client
//!
//! HTTP client for interacting with a remote skill registry server.
//! Feature-gated behind the `registry` feature flag.

use anyhow::{Context, Result};

use super::manifest::SkillManifest;
use super::package::SkillPackage;

/// HTTP client for a remote skill registry.
pub struct RegistryClient {
    /// Base URL of the registry (e.g. `http://localhost:3000`)
    base_url: String,
    /// Optional API key for authenticated operations (publish)
    api_key: Option<String>,
    /// Underlying HTTP client
    http: reqwest::Client,
}

impl RegistryClient {
    /// Create a new registry client.
    pub fn new(base_url: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key,
            http: reqwest::Client::new(),
        }
    }

    /// Search for skills matching a query, optional tags, and result limit.
    pub async fn search(
        &self,
        query: &str,
        tags: Option<&[String]>,
        limit: Option<u32>,
    ) -> Result<Vec<SkillManifest>> {
        let mut url = format!("{}/api/skills/search?q={}", self.base_url, query);
        if let Some(tags) = tags {
            url.push_str(&format!("&tags={}", tags.join(",")));
        }
        if let Some(limit) = limit {
            url.push_str(&format!("&limit={}", limit));
        }

        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .context("Failed to search registry")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Registry search failed ({}): {}", status, body);
        }

        resp.json().await.context("Failed to parse search response")
    }

    /// Publish a skill package to the registry.
    pub async fn publish(&self, package: &SkillPackage) -> Result<()> {
        let mut req = self
            .http
            .post(format!("{}/api/skills", self.base_url))
            .json(package);

        if let Some(ref key) = self.api_key {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let resp = req.send().await.context("Failed to publish to registry")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Registry publish failed ({}): {}", status, body);
        }

        Ok(())
    }

    /// Download a skill package by name and version requirement.
    pub async fn download(
        &self,
        name: &str,
        version_req: &semver::VersionReq,
    ) -> Result<SkillPackage> {
        // First resolve the best matching version
        let versions = self.list_versions(name).await?;
        let matched = versions
            .iter()
            .filter(|v| version_req.matches(v))
            .max()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No version of '{}' matches requirement '{}'",
                    name,
                    version_req
                )
            })?;

        let url = format!("{}/api/skills/{}/{}/download", self.base_url, name, matched);

        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .context("Failed to download skill")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Registry download failed ({}): {}", status, body);
        }

        resp.json()
            .await
            .context("Failed to parse downloaded package")
    }

    /// List all available versions of a skill.
    pub async fn list_versions(&self, name: &str) -> Result<Vec<semver::Version>> {
        let url = format!("{}/api/skills/{}/versions", self.base_url, name);

        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .context("Failed to list versions")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Registry list_versions failed ({}): {}", status, body);
        }

        resp.json().await.context("Failed to parse version list")
    }

    /// Get the manifest for a specific skill version.
    pub async fn get_manifest(
        &self,
        name: &str,
        version: &semver::Version,
    ) -> Result<SkillManifest> {
        let url = format!("{}/api/skills/{}/{}", self.base_url, name, version);

        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .context("Failed to get manifest")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Registry get_manifest failed ({}): {}", status, body);
        }

        resp.json()
            .await
            .context("Failed to parse manifest response")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_construction() {
        let client = RegistryClient::new("http://localhost:3000", None);
        assert_eq!(client.base_url, "http://localhost:3000");
        assert!(client.api_key.is_none());
    }

    #[test]
    fn test_client_with_api_key() {
        let client =
            RegistryClient::new("https://registry.example.com", Some("secret".to_string()));
        assert_eq!(client.base_url, "https://registry.example.com");
        assert_eq!(client.api_key.as_deref(), Some("secret"));
    }
}
