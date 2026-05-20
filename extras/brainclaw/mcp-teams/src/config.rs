//! Teams adapter runtime configuration.

use serde::{Deserialize, Serialize};

/// Teams adapter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamsConfig {
    /// Bot Framework application id (aka client id of the Azure AD app).
    pub app_id: String,
    /// Client secret for the Azure AD app. Used to mint OAuth bearers.
    pub app_password: String,
    /// Tenant id — `"common"` for multi-tenant bots, a specific GUID for
    /// single-tenant bots.
    pub tenant_id: String,
    /// Gateway WebSocket URL.
    pub gateway_url: String,
    /// Optional gateway handshake token.
    pub gateway_token: Option<String>,
    /// HTTP listen address for the Bot Framework webhook.
    pub listen_addr: String,
}

impl Default for TeamsConfig {
    fn default() -> Self {
        Self {
            app_id: String::new(),
            app_password: String::new(),
            tenant_id: "common".to_string(),
            gateway_url: "ws://127.0.0.1:18789/ws".to_string(),
            gateway_token: None,
            listen_addr: "0.0.0.0:9102".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_listen_addr_is_9102() {
        assert_eq!(TeamsConfig::default().listen_addr, "0.0.0.0:9102");
    }

    #[test]
    fn default_tenant_is_common() {
        assert_eq!(TeamsConfig::default().tenant_id, "common");
    }

    #[test]
    fn serde_roundtrip() {
        let cfg = TeamsConfig {
            app_id: "abc".into(),
            app_password: "secret".into(),
            tenant_id: "tid".into(),
            gateway_url: "ws://x/y".into(),
            gateway_token: Some("gt".into()),
            listen_addr: "127.0.0.1:9102".into(),
        };
        let s = serde_json::to_string(&cfg).unwrap();
        let parsed: TeamsConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.app_id, "abc");
        assert_eq!(parsed.tenant_id, "tid");
    }
}
