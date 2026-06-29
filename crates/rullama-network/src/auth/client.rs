//! Authentication Client
//!
//! HTTP client for authenticating with the Brainwires Studio backend.
//! Uses injected endpoint configuration instead of CLI-specific constants.

use anyhow::{Context, Result, anyhow};
use regex::Regex;
use reqwest::Client;

use super::types::{AuthRequest, AuthResponse};

/// Authentication client for interacting with Brainwires Studio backend
///
/// Unlike the CLI version, this client does NOT auto-save sessions or manage
/// keyring storage. The caller is responsible for persisting the auth response.
pub struct AuthClient {
    http_client: Client,
    /// Full backend URL (e.g., `https://brainwires.studio`)
    backend_url: String,
    /// Auth endpoint path (e.g., "/api/cli/auth")
    auth_endpoint: String,
    /// Compiled API key validation pattern
    api_key_pattern: Regex,
}

impl AuthClient {
    /// Create a new authentication client
    ///
    /// # Arguments
    /// * `backend_url` - Base URL (e.g., `https://brainwires.studio`)
    /// * `auth_endpoint` - Auth endpoint path (e.g., "/api/cli/auth")
    /// * `api_key_pattern` - Regex pattern for API key validation (e.g., r"^bw_(prod|dev|test)_[a-z0-9]{32}$")
    pub fn new(backend_url: String, auth_endpoint: String, api_key_pattern: &str) -> Self {
        Self {
            http_client: Client::new(),
            backend_url,
            auth_endpoint,
            api_key_pattern: Regex::new(api_key_pattern).expect("Invalid API key regex pattern"),
        }
    }

    /// Create from an `AuthEndpoints` trait implementation
    pub fn from_endpoints(endpoints: &dyn crate::traits::AuthEndpoints) -> Self {
        Self::new(
            endpoints.backend_url().to_string(),
            endpoints.auth_endpoint(),
            endpoints.api_key_pattern(),
        )
    }

    /// Validate API key format against the configured pattern
    pub fn validate_api_key_format(&self, api_key: &str) -> Result<()> {
        if !self.api_key_pattern.is_match(api_key) {
            return Err(anyhow!(
                "Invalid API key format. Expected: bw_[env]_[32chars]"
            ));
        }
        Ok(())
    }

    /// Authenticate with API key
    ///
    /// Returns the raw `AuthResponse` from the backend. The caller is responsible
    /// for creating a session and storing the API key.
    pub async fn authenticate(&self, api_key: &str) -> Result<AuthResponse> {
        // Validate format
        self.validate_api_key_format(api_key)?;

        // Make API request
        let url = format!("{}{}", self.backend_url, self.auth_endpoint);
        let request = AuthRequest {
            api_key: api_key.to_string(),
        };

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send authentication request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            return Err(anyhow!(
                "Authentication failed (status {}): {}",
                status,
                error_text
            ));
        }

        let auth_response: AuthResponse = response
            .json()
            .await
            .context("Failed to parse authentication response")?;

        Ok(auth_response)
    }

    /// Get the configured backend URL
    pub fn backend_url(&self) -> &str {
        &self.backend_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_client() -> AuthClient {
        AuthClient::new(
            "https://test.example.com".to_string(),
            "/api/cli/auth".to_string(),
            r"^bw_(prod|dev|test)_[a-z0-9]{32}$",
        )
    }

    #[test]
    fn test_validate_api_key_format() {
        let client = make_client();

        // Valid keys
        assert!(
            client
                .validate_api_key_format("bw_dev_12345678901234567890123456789012")
                .is_ok()
        );
        assert!(
            client
                .validate_api_key_format("bw_prod_abcdefghijklmnopqrstuvwxyz123456")
                .is_ok()
        );
        assert!(
            client
                .validate_api_key_format("bw_test_00000000000000000000000000000000")
                .is_ok()
        );

        // Invalid keys
        assert!(client.validate_api_key_format("invalid").is_err());
        assert!(
            client
                .validate_api_key_format("bw_invalid_12345678901234567890123456789012")
                .is_err()
        );
        assert!(client.validate_api_key_format("bw_dev_short").is_err());
        assert!(
            client
                .validate_api_key_format("bw_dev_UPPERCASE0000000000000000000000")
                .is_err()
        );
    }

    #[test]
    fn test_auth_client_new() {
        let client = make_client();
        assert_eq!(client.backend_url(), "https://test.example.com");
    }

    #[test]
    fn test_validate_api_key_error_message() {
        let client = make_client();
        let result = client.validate_api_key_format("invalid_key");
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Invalid API key format"));
    }

    #[test]
    fn test_validate_api_key_edge_cases() {
        let client = make_client();

        assert!(client.validate_api_key_format("").is_err());
        assert!(client.validate_api_key_format("   ").is_err());
        assert!(client.validate_api_key_format("bw_dev_123").is_err()); // too short
        assert!(
            client
                .validate_api_key_format("bw_dev_123456789012345678901234567890123")
                .is_err()
        ); // too long
        assert!(
            client
                .validate_api_key_format("dev_12345678901234567890123456789012")
                .is_err()
        ); // missing prefix
        assert!(
            client
                .validate_api_key_format("bw__12345678901234567890123456789012")
                .is_err()
        ); // missing env
        assert!(
            client
                .validate_api_key_format("bw_Dev_12345678901234567890123456789012")
                .is_err()
        ); // mixed case env
    }
}
