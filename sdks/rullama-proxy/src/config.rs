use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// Top-level proxy configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProxyConfig {
    /// Where the proxy listens for incoming connections.
    pub listener: ListenerConfig,
    /// Where the proxy forwards traffic.
    pub upstream: UpstreamConfig,
    /// Maximum request body size in bytes (0 = unlimited).
    #[serde(default = "default_max_body_size")]
    pub max_body_size: usize,
    /// Request timeout.
    #[serde(with = "humantime_serde", default = "default_timeout")]
    pub timeout: Duration,
    /// Enable the traffic inspector.
    #[serde(default)]
    pub inspector: InspectorConfig,
    /// Extra key-value metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

fn default_max_body_size() -> usize {
    10 * 1024 * 1024 // 10 MiB
}

fn default_timeout() -> Duration {
    Duration::from_secs(30)
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            listener: ListenerConfig::default(),
            upstream: UpstreamConfig::default(),
            max_body_size: default_max_body_size(),
            timeout: default_timeout(),
            inspector: InspectorConfig::default(),
            metadata: HashMap::new(),
        }
    }
}

/// Configuration for the proxy listener (inbound side).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ListenerConfig {
    /// Listen on a TCP socket.
    Tcp { addr: SocketAddr },
    /// Listen on a Unix domain socket.
    Unix { path: PathBuf },
}

impl Default for ListenerConfig {
    fn default() -> Self {
        Self::Tcp {
            addr: SocketAddr::from(([127, 0, 0, 1], 8080)),
        }
    }
}

/// Configuration for the upstream target (outbound side).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UpstreamConfig {
    /// Connect to a URL (HTTP/HTTPS/WS/WSS).
    Url { url: String },
    /// Connect to a TCP host:port.
    Tcp { host: String, port: u16 },
    /// Connect to a Unix domain socket.
    Unix { path: PathBuf },
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        Self::Url {
            url: "http://localhost:3000".to_string(),
        }
    }
}

/// Inspector subsystem configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InspectorConfig {
    /// Enable traffic capture.
    #[serde(default)]
    pub enabled: bool,
    /// Maximum events stored in the ring buffer.
    #[serde(default = "default_event_capacity")]
    pub event_capacity: usize,
    /// Broadcast channel capacity.
    #[serde(default = "default_broadcast_capacity")]
    pub broadcast_capacity: usize,
    /// If set, bind the inspector HTTP API on this address.
    pub api_addr: Option<SocketAddr>,
}

fn default_event_capacity() -> usize {
    10_000
}

fn default_broadcast_capacity() -> usize {
    256
}

impl Default for InspectorConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            event_capacity: default_event_capacity(),
            broadcast_capacity: default_broadcast_capacity(),
            api_addr: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = ProxyConfig::default();
        assert_eq!(config.max_body_size, 10 * 1024 * 1024);
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(!config.inspector.enabled);
        assert!(config.metadata.is_empty());
    }

    #[test]
    fn config_serde_roundtrip() {
        let config = ProxyConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ProxyConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.max_body_size, config.max_body_size);
        assert_eq!(deserialized.timeout, config.timeout);
    }

    #[test]
    fn listener_config_tcp() {
        let listener = ListenerConfig::Tcp {
            addr: "0.0.0.0:9090".parse().unwrap(),
        };
        let json = serde_json::to_string(&listener).unwrap();
        assert!(json.contains("tcp"));
        assert!(json.contains("9090"));
    }

    #[test]
    fn upstream_config_url() {
        let upstream = UpstreamConfig::Url {
            url: "https://api.example.com".to_string(),
        };
        let json = serde_json::to_string(&upstream).unwrap();
        assert!(json.contains("api.example.com"));
    }

    #[test]
    fn inspector_config_defaults() {
        let config = InspectorConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.event_capacity, 10_000);
        assert_eq!(config.broadcast_capacity, 256);
        assert!(config.api_addr.is_none());
    }
}

/// Serde helper for `Duration` via humantime strings.
mod humantime_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(Duration::from_secs(secs))
    }
}
