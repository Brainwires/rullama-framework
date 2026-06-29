use serde::{Deserialize, Serialize};

/// Network capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCapabilities {
    /// Allowed domains (supports wildcards like *.github.com)
    #[serde(default)]
    pub allowed_domains: Vec<String>,

    /// Denied domains (override allows)
    #[serde(default)]
    pub denied_domains: Vec<String>,

    /// Allow all domains (use with caution)
    #[serde(default)]
    pub allow_all: bool,

    /// Rate limit (requests per minute)
    #[serde(default)]
    pub rate_limit: Option<u32>,

    /// Can make external API calls
    #[serde(default)]
    pub allow_api_calls: bool,

    /// Maximum response size to process (bytes)
    #[serde(default)]
    pub max_response_size: Option<u64>,
}

impl Default for NetworkCapabilities {
    fn default() -> Self {
        Self {
            allowed_domains: Vec::new(),
            denied_domains: Vec::new(),
            allow_all: false,
            rate_limit: Some(60),
            allow_api_calls: false,
            max_response_size: Some(10 * 1024 * 1024), // 10MB
        }
    }
}

impl NetworkCapabilities {
    /// Create disabled network capabilities
    pub fn disabled() -> Self {
        Self {
            allowed_domains: Vec::new(),
            denied_domains: Vec::new(),
            allow_all: false,
            rate_limit: Some(0),
            allow_api_calls: false,
            max_response_size: None,
        }
    }

    /// Create full network capabilities
    pub fn full() -> Self {
        Self {
            allowed_domains: Vec::new(),
            denied_domains: Vec::new(),
            allow_all: true,
            rate_limit: None,
            allow_api_calls: true,
            max_response_size: None,
        }
    }
}
