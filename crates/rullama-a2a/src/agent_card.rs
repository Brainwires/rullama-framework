//! Agent card types: AgentCard, AgentCapabilities, AgentSkill, security schemes, OAuth flows.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Self-describing manifest for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCard {
    /// Human-readable agent name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Agent version string.
    pub version: String,
    /// Ordered list of supported interfaces (first is preferred).
    #[serde(rename = "supportedInterfaces")]
    pub supported_interfaces: Vec<AgentInterface>,
    /// Agent capabilities.
    pub capabilities: AgentCapabilities,
    /// Agent skills.
    pub skills: Vec<AgentSkill>,
    /// Default input media types.
    #[serde(rename = "defaultInputModes")]
    pub default_input_modes: Vec<String>,
    /// Default output media types.
    #[serde(rename = "defaultOutputModes")]
    pub default_output_modes: Vec<String>,
    /// Service provider information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<AgentProvider>,
    /// Security scheme definitions.
    #[serde(rename = "securitySchemes", skip_serializing_if = "Option::is_none")]
    pub security_schemes: Option<HashMap<String, SecurityScheme>>,
    /// Security requirements.
    #[serde(
        rename = "securityRequirements",
        skip_serializing_if = "Option::is_none"
    )]
    pub security_requirements: Option<Vec<SecurityRequirement>>,
    /// URL to additional documentation.
    #[serde(rename = "documentationUrl", skip_serializing_if = "Option::is_none")]
    pub documentation_url: Option<String>,
    /// Icon URL.
    #[serde(rename = "iconUrl", skip_serializing_if = "Option::is_none")]
    pub icon_url: Option<String>,
    /// JWS signatures for the agent card.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signatures: Option<Vec<AgentCardSignature>>,
}

/// Declares a protocol binding interface for the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInterface {
    /// URL where this interface is available.
    pub url: String,
    /// Protocol binding: `JSONRPC`, `GRPC`, `HTTP+JSON`.
    #[serde(rename = "protocolBinding")]
    pub protocol_binding: String,
    /// Optional tenant identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    /// A2A protocol version. Use [`crate::A2A_PROTOCOL_VERSION`] for the canonical value.
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
}

/// Agent capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentCapabilities {
    /// Supports streaming responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub streaming: Option<bool>,
    /// Supports push notifications.
    #[serde(rename = "pushNotifications", skip_serializing_if = "Option::is_none")]
    pub push_notifications: Option<bool>,
    /// Supports extended agent card.
    #[serde(rename = "extendedAgentCard", skip_serializing_if = "Option::is_none")]
    pub extended_agent_card: Option<bool>,
    /// Protocol extensions supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Vec<AgentExtension>>,
}

/// A protocol extension declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentExtension {
    /// Unique URI identifying the extension.
    pub uri: String,
    /// Human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Whether the client must comply.
    #[serde(default)]
    pub required: bool,
    /// Extension-specific parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<HashMap<String, serde_json::Value>>,
}

/// An agent's specific capability or function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSkill {
    /// Unique skill identifier.
    pub id: String,
    /// Human-readable skill name.
    pub name: String,
    /// Detailed description.
    pub description: String,
    /// Keywords describing capabilities.
    pub tags: Vec<String>,
    /// Example prompts/scenarios.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub examples: Option<Vec<String>>,
    /// Override input modes for this skill.
    #[serde(rename = "inputModes", skip_serializing_if = "Option::is_none")]
    pub input_modes: Option<Vec<String>>,
    /// Override output modes for this skill.
    #[serde(rename = "outputModes", skip_serializing_if = "Option::is_none")]
    pub output_modes: Option<Vec<String>>,
    /// Security requirements for this skill.
    #[serde(
        rename = "securityRequirements",
        skip_serializing_if = "Option::is_none"
    )]
    pub security_requirements: Option<Vec<SecurityRequirement>>,
}

/// Agent service provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProvider {
    /// Provider website or documentation URL.
    pub url: String,
    /// Organization name.
    pub organization: String,
}

/// JWS signature for an AgentCard (RFC 7515).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCardSignature {
    /// Base64url-encoded protected JWS header.
    pub protected: String,
    /// Base64url-encoded computed signature.
    pub signature: String,
    /// Unprotected header values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<HashMap<String, serde_json::Value>>,
}

/// Security requirements map: scheme name -> required scopes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityRequirement {
    /// Map of security scheme names to their required scopes.
    pub schemes: HashMap<String, Vec<String>>,
}

/// Security scheme (wrapper-based oneOf).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SecurityScheme {
    /// API key authentication.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "apiKeySecurityScheme"
    )]
    pub api_key: Option<ApiKeySecurityScheme>,
    /// HTTP authentication (Bearer, Basic, etc).
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "httpAuthSecurityScheme"
    )]
    pub http_auth: Option<HttpAuthSecurityScheme>,
    /// OAuth 2.0 authentication.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "oauth2SecurityScheme"
    )]
    pub oauth2: Option<OAuth2SecurityScheme>,
    /// OpenID Connect authentication.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "openIdConnectSecurityScheme"
    )]
    pub open_id_connect: Option<OpenIdConnectSecurityScheme>,
    /// Mutual TLS authentication.
    #[serde(skip_serializing_if = "Option::is_none", rename = "mtlsSecurityScheme")]
    pub mtls: Option<MutualTlsSecurityScheme>,
}

/// API key security scheme details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiKeySecurityScheme {
    /// Parameter name.
    pub name: String,
    /// Location: `query`, `header`, or `cookie`.
    #[serde(rename = "in")]
    pub location: String,
    /// Description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// HTTP authentication security scheme details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HttpAuthSecurityScheme {
    /// Auth scheme name (e.g. `Bearer`).
    pub scheme: String,
    /// Format hint (e.g. `JWT`).
    #[serde(rename = "bearerFormat", skip_serializing_if = "Option::is_none")]
    pub bearer_format: Option<String>,
    /// Description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// OAuth 2.0 security scheme details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OAuth2SecurityScheme {
    /// OAuth2 flow configuration.
    pub flows: OAuthFlows,
    /// Description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// OAuth2 metadata URL (RFC 8414).
    #[serde(rename = "oauth2MetadataUrl", skip_serializing_if = "Option::is_none")]
    pub oauth2_metadata_url: Option<String>,
}

/// OpenID Connect security scheme details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenIdConnectSecurityScheme {
    /// OIDC discovery URL.
    #[serde(rename = "openIdConnectUrl")]
    pub open_id_connect_url: String,
    /// Description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Mutual TLS security scheme details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MutualTlsSecurityScheme {
    /// Description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// OAuth 2.0 flow configuration (wrapper-based oneOf).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OAuthFlows {
    /// Authorization Code flow.
    #[serde(skip_serializing_if = "Option::is_none", rename = "authorizationCode")]
    pub authorization_code: Option<AuthorizationCodeOAuthFlow>,
    /// Client Credentials flow.
    #[serde(skip_serializing_if = "Option::is_none", rename = "clientCredentials")]
    pub client_credentials: Option<ClientCredentialsOAuthFlow>,
    /// Implicit flow (deprecated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub implicit: Option<ImplicitOAuthFlow>,
    /// Password flow (deprecated).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<PasswordOAuthFlow>,
    /// Device Code flow (RFC 8628).
    #[serde(skip_serializing_if = "Option::is_none", rename = "deviceCode")]
    pub device_code: Option<DeviceCodeOAuthFlow>,
}

/// Authorization Code OAuth flow.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuthorizationCodeOAuthFlow {
    /// Authorization URL.
    #[serde(rename = "authorizationUrl")]
    pub authorization_url: String,
    /// Token URL.
    #[serde(rename = "tokenUrl")]
    pub token_url: String,
    /// Refresh URL.
    #[serde(rename = "refreshUrl", skip_serializing_if = "Option::is_none")]
    pub refresh_url: Option<String>,
    /// Available scopes.
    pub scopes: HashMap<String, String>,
    /// Whether PKCE is required.
    #[serde(rename = "pkceRequired", skip_serializing_if = "Option::is_none")]
    pub pkce_required: Option<bool>,
}

/// Client Credentials OAuth flow.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClientCredentialsOAuthFlow {
    /// Token URL.
    #[serde(rename = "tokenUrl")]
    pub token_url: String,
    /// Refresh URL.
    #[serde(rename = "refreshUrl", skip_serializing_if = "Option::is_none")]
    pub refresh_url: Option<String>,
    /// Available scopes.
    pub scopes: HashMap<String, String>,
}

/// Implicit OAuth flow (deprecated).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImplicitOAuthFlow {
    /// Authorization URL.
    #[serde(rename = "authorizationUrl", skip_serializing_if = "Option::is_none")]
    pub authorization_url: Option<String>,
    /// Refresh URL.
    #[serde(rename = "refreshUrl", skip_serializing_if = "Option::is_none")]
    pub refresh_url: Option<String>,
    /// Available scopes.
    #[serde(default)]
    pub scopes: HashMap<String, String>,
}

/// Password OAuth flow (deprecated).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PasswordOAuthFlow {
    /// Token URL.
    #[serde(rename = "tokenUrl", skip_serializing_if = "Option::is_none")]
    pub token_url: Option<String>,
    /// Refresh URL.
    #[serde(rename = "refreshUrl", skip_serializing_if = "Option::is_none")]
    pub refresh_url: Option<String>,
    /// Available scopes.
    #[serde(default)]
    pub scopes: HashMap<String, String>,
}

/// Device Code OAuth flow (RFC 8628).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceCodeOAuthFlow {
    /// Device authorization URL.
    #[serde(rename = "deviceAuthorizationUrl")]
    pub device_authorization_url: String,
    /// Token URL.
    #[serde(rename = "tokenUrl")]
    pub token_url: String,
    /// Refresh URL.
    #[serde(rename = "refreshUrl", skip_serializing_if = "Option::is_none")]
    pub refresh_url: Option<String>,
    /// Available scopes.
    pub scopes: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_card() -> AgentCard {
        AgentCard {
            name: "Test Agent".to_string(),
            description: "A test agent".to_string(),
            version: "1.0.0".to_string(),
            supported_interfaces: vec![AgentInterface {
                url: "https://example.com/a2a".to_string(),
                protocol_binding: "JSONRPC".to_string(),
                tenant: None,
                protocol_version: "0.3".to_string(),
            }],
            capabilities: AgentCapabilities::default(),
            skills: vec![],
            default_input_modes: vec!["text/plain".to_string()],
            default_output_modes: vec!["text/plain".to_string()],
            provider: None,
            security_schemes: None,
            security_requirements: None,
            documentation_url: None,
            icon_url: None,
            signatures: None,
        }
    }

    // --- AgentCard ---

    #[test]
    fn agent_card_roundtrip() {
        let card = minimal_card();
        let json = serde_json::to_string(&card).unwrap();
        let back: AgentCard = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, card.name);
        assert_eq!(back.version, card.version);
    }

    #[test]
    fn agent_card_json_uses_camel_case() {
        let card = minimal_card();
        let json = serde_json::to_string(&card).unwrap();
        assert!(json.contains("supportedInterfaces"));
        assert!(json.contains("defaultInputModes"));
        assert!(json.contains("defaultOutputModes"));
    }

    #[test]
    fn optional_fields_omitted_when_none() {
        let card = minimal_card();
        let json = serde_json::to_string(&card).unwrap();
        assert!(!json.contains("provider"));
        assert!(!json.contains("securitySchemes"));
        assert!(!json.contains("documentationUrl"));
        assert!(!json.contains("iconUrl"));
        assert!(!json.contains("signatures"));
    }

    #[test]
    fn agent_card_with_skill_roundtrip() {
        let mut card = minimal_card();
        card.skills = vec![AgentSkill {
            id: "skill-1".to_string(),
            name: "My Skill".to_string(),
            description: "Does something".to_string(),
            tags: vec!["tag1".to_string()],
            examples: Some(vec!["example prompt".to_string()]),
            input_modes: None,
            output_modes: None,
            security_requirements: None,
        }];
        let json = serde_json::to_string(&card).unwrap();
        let back: AgentCard = serde_json::from_str(&json).unwrap();
        assert_eq!(back.skills.len(), 1);
        assert_eq!(back.skills[0].id, "skill-1");
    }

    // --- AgentCapabilities ---

    #[test]
    fn capabilities_default_all_none() {
        let cap = AgentCapabilities::default();
        let json = serde_json::to_string(&cap).unwrap();
        assert!(!json.contains("streaming"));
        assert!(!json.contains("pushNotifications"));
        assert!(!json.contains("extendedAgentCard"));
    }

    #[test]
    fn capabilities_with_streaming_roundtrip() {
        let cap = AgentCapabilities {
            streaming: Some(true),
            push_notifications: Some(false),
            extended_agent_card: None,
            extensions: None,
        };
        let json = serde_json::to_string(&cap).unwrap();
        let back: AgentCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(back.streaming, Some(true));
        assert_eq!(back.push_notifications, Some(false));
    }

    // --- SecurityScheme ---

    #[test]
    fn api_key_security_scheme_roundtrip() {
        let scheme = SecurityScheme {
            api_key: Some(ApiKeySecurityScheme {
                name: "X-Api-Key".to_string(),
                location: "header".to_string(),
                description: None,
            }),
            http_auth: None,
            oauth2: None,
            open_id_connect: None,
            mtls: None,
        };
        let json = serde_json::to_string(&scheme).unwrap();
        let back: SecurityScheme = serde_json::from_str(&json).unwrap();
        assert_eq!(back, scheme);
        assert!(json.contains("apiKeySecurityScheme"));
    }

    #[test]
    fn oauth2_authorization_code_flow_roundtrip() {
        let scheme = SecurityScheme {
            api_key: None,
            http_auth: None,
            oauth2: Some(OAuth2SecurityScheme {
                flows: OAuthFlows {
                    authorization_code: Some(AuthorizationCodeOAuthFlow {
                        authorization_url: "https://auth.example.com/authorize".to_string(),
                        token_url: "https://auth.example.com/token".to_string(),
                        refresh_url: None,
                        scopes: [("read".to_string(), "Read access".to_string())]
                            .into_iter()
                            .collect(),
                        pkce_required: Some(true),
                    }),
                    client_credentials: None,
                    implicit: None,
                    password: None,
                    device_code: None,
                },
                description: None,
                oauth2_metadata_url: None,
            }),
            open_id_connect: None,
            mtls: None,
        };
        let json = serde_json::to_string(&scheme).unwrap();
        let back: SecurityScheme = serde_json::from_str(&json).unwrap();
        assert_eq!(back, scheme);
    }

    // --- AgentInterface ---

    #[test]
    fn agent_interface_json_uses_camel_case() {
        let iface = AgentInterface {
            url: "https://example.com".to_string(),
            protocol_binding: "JSONRPC".to_string(),
            tenant: None,
            protocol_version: "0.3".to_string(),
        };
        let json = serde_json::to_string(&iface).unwrap();
        assert!(json.contains("protocolBinding"));
        assert!(json.contains("protocolVersion"));
        assert!(!json.contains("protocol_binding"));
    }

    // --- AgentProvider ---

    #[test]
    fn agent_provider_roundtrip() {
        let p = AgentProvider {
            url: "https://acme.io".to_string(),
            organization: "ACME Corp".to_string(),
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: AgentProvider = serde_json::from_str(&json).unwrap();
        assert_eq!(back.organization, "ACME Corp");
    }
}
