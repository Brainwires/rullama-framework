//! Backend registration and protocol/policy negotiation.

use anyhow::{Context, Result, bail};

use super::core::RemoteBridge;
use super::types::RealtimeCredentials;
use crate::remote::protocol::{NegotiatedProtocol, ProtocolHello};

impl RemoteBridge {
    /// Register with the backend via HTTP POST
    pub(super) async fn register_with_backend(&mut self) -> Result<()> {
        let url = format!("{}/api/remote/connect", self.config.backend_url);
        tracing::info!("Registering with backend: {}", url);

        let protocol_hello = ProtocolHello::default();
        let device_fingerprint = crate::remote::protocol::compute_device_fingerprint();
        let register_body = serde_json::json!({
            "hostname": gethostname::gethostname().to_string_lossy().to_string(),
            "os": std::env::consts::OS.to_string(),
            "version": self.config.version.clone(),
            "protocol": protocol_hello,
            "device_fingerprint": device_fingerprint,
        });

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&register_body)
            .send()
            .await
            .context("Failed to connect to backend")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("Registration failed: {} - {}", status, body);
        }

        let auth_response: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse registration response")?;

        if let Some(error) = auth_response.get("error") {
            bail!("Authentication failed: {}", error);
        }

        let session_token = auth_response
            .get("session_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing session_token in response"))?;

        let user_id = auth_response
            .get("user_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing user_id in response"))?;

        tracing::info!("Authenticated as user: {}", user_id);
        *self.session_token.write().await = Some(session_token.to_string());
        *self.user_id.write().await = Some(user_id.to_string());

        // Handle protocol negotiation response
        if let Some(protocol_value) = auth_response.get("protocol") {
            match serde_json::from_value::<crate::remote::protocol::ProtocolAccept>(
                protocol_value.clone(),
            ) {
                Ok(accept) => {
                    tracing::info!(
                        "Protocol negotiated: version={}, capabilities={:?}",
                        accept.selected_version,
                        accept.enabled_capabilities
                    );
                    *self.negotiated_protocol.write().await =
                        NegotiatedProtocol::from_accept(accept);
                }
                Err(e) => {
                    tracing::warn!("Failed to parse protocol accept: {}, using defaults", e);
                    *self.negotiated_protocol.write().await = NegotiatedProtocol::default();
                }
            }
        } else {
            tracing::debug!("Backend did not return protocol, using defaults");
            *self.negotiated_protocol.write().await = NegotiatedProtocol::default();
        }

        // Handle device allowlist status
        if let Some(ds) = auth_response.get("device_status") {
            match serde_json::from_value::<crate::remote::protocol::DeviceStatus>(ds.clone()) {
                Ok(status) => {
                    tracing::info!("Device status: {:?}", status);
                    if matches!(status, crate::remote::protocol::DeviceStatus::Blocked) {
                        bail!("Device is blocked by the user's device allowlist");
                    }
                    *self.device_status.write().await = Some(status);
                }
                Err(e) => tracing::warn!("Failed to parse device_status: {}", e),
            }
        }

        // Handle organization policies
        if let Some(op) = auth_response.get("org_policies") {
            match serde_json::from_value::<crate::remote::protocol::OrgPolicies>(op.clone()) {
                Ok(policies) => {
                    tracing::info!(
                        "Org policies: blocked_tools={:?}, permission_relay_required={}",
                        policies.blocked_tools,
                        policies.permission_relay_required
                    );
                    *self.org_policies.write().await = Some(policies);
                }
                Err(e) => tracing::warn!("Failed to parse org_policies: {}", e),
            }
        }

        // Check for Realtime credentials
        let use_realtime = auth_response
            .get("use_realtime")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if use_realtime {
            let realtime_token = auth_response.get("realtime_token").and_then(|v| v.as_str());
            let realtime_url = auth_response.get("realtime_url").and_then(|v| v.as_str());
            let channel_name = auth_response.get("channel_name").and_then(|v| v.as_str());
            let supabase_anon_key = auth_response
                .get("supabase_anon_key")
                .and_then(|v| v.as_str());

            if let (Some(token), Some(url), Some(channel), Some(anon_key)) = (
                realtime_token,
                realtime_url,
                channel_name,
                supabase_anon_key,
            ) {
                tracing::info!("Realtime credentials received, channel: {}", channel);
                *self.realtime_credentials.write().await = Some(RealtimeCredentials {
                    realtime_token: token.to_string(),
                    realtime_url: url.to_string(),
                    channel_name: channel.to_string(),
                    supabase_anon_key: anon_key.to_string(),
                });
            } else {
                tracing::warn!(
                    "use_realtime=true but missing Realtime credentials (token={}, url={}, channel={}, anon_key={})",
                    realtime_token.is_some(),
                    realtime_url.is_some(),
                    channel_name.is_some(),
                    supabase_anon_key.is_some()
                );
            }
        }

        Ok(())
    }
}
