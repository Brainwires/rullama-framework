/// OpenThread Border Router (OTBR) REST API client.
///
/// Targets Thread 1.3.0 and the OTBR REST API as shipped with OpenThread r/2024-01+.
/// Default base URL: `http://<host>:8081`.
///
/// Reference: <https://openthread.io/reference/group/api-rest>
use reqwest::Client;
use tracing::debug;

use super::types::{ThreadNeighbor, ThreadNetworkDataset, ThreadNodeInfo};
use crate::error::{HomeAutoError, HomeAutoResult};

/// HTTP client for the OpenThread Border Router REST API.
///
/// All requests are unauthenticated (OTBR does not implement auth by default on LAN).
pub struct ThreadBorderRouter {
    base_url: String,
    client: Client,
}

impl ThreadBorderRouter {
    /// Create a new client for the OTBR at `otbr_url` (e.g. `"http://192.168.1.100:8081"`).
    pub fn new(otbr_url: impl Into<String>) -> HomeAutoResult<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| HomeAutoError::ThreadHttp(e.to_string()))?;
        Ok(Self {
            base_url: otbr_url.into().trim_end_matches('/').to_string(),
            client,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    /// Get information about this Thread node (`GET /node`).
    pub async fn node_info(&self) -> HomeAutoResult<ThreadNodeInfo> {
        let url = self.url("/node");
        debug!("OTBR GET {url}");
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| HomeAutoError::ThreadHttp(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(HomeAutoError::ThreadHttp(format!(
                "GET /node returned {}",
                resp.status()
            )));
        }
        resp.json::<ThreadNodeInfo>()
            .await
            .map_err(|e| HomeAutoError::ThreadParse(e.to_string()))
    }

    /// Get the neighbor table (`GET /node/neighbors`).
    pub async fn neighbors(&self) -> HomeAutoResult<Vec<ThreadNeighbor>> {
        let url = self.url("/node/neighbors");
        debug!("OTBR GET {url}");
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| HomeAutoError::ThreadHttp(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(HomeAutoError::ThreadHttp(format!(
                "GET /node/neighbors returned {}",
                resp.status()
            )));
        }
        resp.json::<Vec<ThreadNeighbor>>()
            .await
            .map_err(|e| HomeAutoError::ThreadParse(e.to_string()))
    }

    /// Add a joiner to the commissioner (`POST /node/commissioner/joiner`).
    ///
    /// - `eui64`: device EUI-64, e.g. `"0000000000000001"` or `"*"` (wildcard).
    /// - `credential`: PSKd (pre-shared key for device), 6–32 ASCII uppercase characters.
    pub async fn add_joiner(&self, eui64: &str, credential: &str) -> HomeAutoResult<()> {
        let url = self.url("/node/commissioner/joiner");
        debug!("OTBR POST {url} eui64={eui64}");
        let body = serde_json::json!({
            "eui64": eui64,
            "pskd": credential,
        });
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| HomeAutoError::ThreadHttp(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(HomeAutoError::ThreadHttp(format!(
                "POST /node/commissioner/joiner returned {}",
                resp.status()
            )));
        }
        Ok(())
    }

    /// Get the active operational dataset (`GET /node/dataset/active`).
    /// Returns the dataset as a hex-encoded TLV string.
    pub async fn active_dataset(&self) -> HomeAutoResult<ThreadNetworkDataset> {
        let url = self.url("/node/dataset/active");
        debug!("OTBR GET {url}");
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| HomeAutoError::ThreadHttp(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(HomeAutoError::ThreadHttp(format!(
                "GET /node/dataset/active returned {}",
                resp.status()
            )));
        }
        // OTBR returns the dataset as a plain hex string
        let hex = resp
            .text()
            .await
            .map_err(|e| HomeAutoError::ThreadParse(e.to_string()))?;
        Ok(ThreadNetworkDataset {
            active_dataset: hex.trim().to_string(),
        })
    }

    /// Get pending operational dataset (`GET /node/dataset/pending`).
    pub async fn pending_dataset(&self) -> HomeAutoResult<Option<ThreadNetworkDataset>> {
        let url = self.url("/node/dataset/pending");
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| HomeAutoError::ThreadHttp(e.to_string()))?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !resp.status().is_success() {
            return Err(HomeAutoError::ThreadHttp(format!(
                "GET /node/dataset/pending returned {}",
                resp.status()
            )));
        }
        let hex = resp
            .text()
            .await
            .map_err(|e| HomeAutoError::ThreadParse(e.to_string()))?;
        Ok(Some(ThreadNetworkDataset {
            active_dataset: hex.trim().to_string(),
        }))
    }

    /// Set the active dataset (`PUT /node/dataset/active`). Supply a hex TLV string.
    pub async fn set_active_dataset(&self, hex_tlv: &str) -> HomeAutoResult<()> {
        let url = self.url("/node/dataset/active");
        let resp = self
            .client
            .put(&url)
            .body(hex_tlv.to_string())
            .header("Content-Type", "text/plain")
            .send()
            .await
            .map_err(|e| HomeAutoError::ThreadHttp(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(HomeAutoError::ThreadHttp(format!(
                "PUT /node/dataset/active returned {}",
                resp.status()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    #[tokio::test]
    async fn border_router_node_info_parses_correctly() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/node"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "rloc16": "0x0400",
                "extAddress": "aabbccddeeff0011",
                "networkName": "TestNet",
                "role": "leader",
            })))
            .mount(&server)
            .await;

        let client = ThreadBorderRouter::new(server.uri()).unwrap();
        let info = client.node_info().await.unwrap();
        assert_eq!(info.rloc16.as_deref(), Some("0x0400"));
        assert_eq!(info.network_name.as_deref(), Some("TestNet"));
    }

    #[tokio::test]
    async fn border_router_neighbors_empty_list() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/node/neighbors"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
            .mount(&server)
            .await;

        let client = ThreadBorderRouter::new(server.uri()).unwrap();
        let neighbors = client.neighbors().await.unwrap();
        assert!(neighbors.is_empty());
    }

    #[tokio::test]
    async fn border_router_add_joiner_sends_correct_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/node/commissioner/joiner"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;

        let client = ThreadBorderRouter::new(server.uri()).unwrap();
        client.add_joiner("*", "J01NME").await.unwrap();
        // Verify the request was received (wiremock counts it)
        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(body["eui64"].as_str(), Some("*"));
        assert_eq!(body["pskd"].as_str(), Some("J01NME"));
    }

    #[tokio::test]
    async fn border_router_error_propagation_on_404() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/node"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let client = ThreadBorderRouter::new(server.uri()).unwrap();
        let result = client.node_info().await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HomeAutoError::ThreadHttp(_)));
    }
}
