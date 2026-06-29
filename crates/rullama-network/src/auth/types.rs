use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// User profile information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// Unique user identifier.
    #[serde(alias = "id")]
    pub user_id: String,
    /// Login username.
    pub username: String,
    /// Human-readable display name.
    pub display_name: String,
    /// User role (e.g. "user", "admin").
    pub role: String,
}

// Provider API keys are NEVER sent to the client - they stay on the server
// The CLI only talks to Brainwires backend, which handles provider API calls server-side

/// Supabase configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupabaseConfig {
    /// Supabase project URL.
    pub url: String,
    /// Supabase anonymous key.
    #[serde(alias = "anonKey")]
    pub anon_key: String,
}

/// Authentication session (stored in session file)
///
/// Note: API keys are stored separately in the system keyring for security.
/// The `api_key` field is kept for backwards compatibility with existing sessions
/// but new sessions will have it set to empty string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthSession {
    /// User profile
    pub user: UserProfile,

    /// Supabase configuration
    pub supabase: SupabaseConfig,

    /// API key name
    pub key_name: String,

    /// Brainwires API key (DEPRECATED - now stored in system keyring)
    /// Kept for backwards compatibility; new sessions have this as empty string
    #[serde(default)]
    pub api_key: String,

    /// Backend URL
    pub backend: String,

    /// When the session was authenticated
    pub authenticated_at: DateTime<Utc>,
}

impl AuthSession {
    /// Check if the session is expired
    ///
    /// Sessions no longer expire automatically - API keys persist until
    /// explicitly logged out or deleted.
    pub fn is_expired(&self) -> bool {
        false
    }
}

/// Authentication request payload
#[derive(Debug, Serialize)]
pub struct AuthRequest {
    /// The API key to authenticate with.
    #[serde(rename = "apiKey")]
    pub api_key: String,
}

/// Authentication response from backend
#[derive(Debug, Deserialize)]
pub struct AuthResponse {
    /// Authenticated user profile.
    pub user: UserProfile,
    /// Supabase configuration.
    pub supabase: SupabaseConfig,
    /// Name of the API key used.
    #[serde(rename = "keyName")]
    pub key_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_session() -> AuthSession {
        AuthSession {
            user: UserProfile {
                user_id: "user123".to_string(),
                username: "testuser".to_string(),
                display_name: "Test User".to_string(),
                role: "user".to_string(),
            },
            supabase: SupabaseConfig {
                url: "https://test.supabase.co".to_string(),
                anon_key: "test-anon-key".to_string(),
            },
            key_name: "test_key".to_string(),
            api_key: "bw_dev_12345678901234567890123456789012".to_string(),
            backend: "https://backend.test".to_string(),
            authenticated_at: Utc::now(),
        }
    }

    #[test]
    fn test_user_profile_creation() {
        let profile = UserProfile {
            user_id: "123".to_string(),
            username: "user".to_string(),
            display_name: "User Name".to_string(),
            role: "admin".to_string(),
        };

        assert_eq!(profile.user_id, "123");
        assert_eq!(profile.username, "user");
        assert_eq!(profile.display_name, "User Name");
        assert_eq!(profile.role, "admin");
    }

    #[test]
    fn test_supabase_config() {
        let config = SupabaseConfig {
            url: "https://example.supabase.co".to_string(),
            anon_key: "anon-key".to_string(),
        };

        assert_eq!(config.url, "https://example.supabase.co");
        assert_eq!(config.anon_key, "anon-key");
    }

    #[test]
    fn test_auth_session_never_expires() {
        let session = create_test_session();
        assert!(!session.is_expired());

        // Even old sessions should not expire
        let mut old_session = create_test_session();
        old_session.authenticated_at = Utc::now() - chrono::Duration::days(365);
        assert!(!old_session.is_expired());
    }

    #[test]
    fn test_auth_request_serialization() {
        let request = AuthRequest {
            api_key: "test-api-key".to_string(),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("apiKey"));
        assert!(json.contains("test-api-key"));
    }

    #[test]
    fn test_auth_response_deserialization() {
        let json = r#"{
            "user": {
                "user_id": "123",
                "username": "testuser",
                "display_name": "Test User",
                "role": "user"
            },
            "providers": {
                "openai_api_key": "key1"
            },
            "supabase": {
                "url": "https://test.supabase.co",
                "anon_key": "anon"
            },
            "keyName": "test_key"
        }"#;

        let response: AuthResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.user.user_id, "123");
        assert_eq!(response.key_name, "test_key");
    }

    #[test]
    fn test_session_serialization_roundtrip() {
        let session = create_test_session();
        let json = serde_json::to_string(&session).unwrap();
        let deserialized: AuthSession = serde_json::from_str(&json).unwrap();

        assert_eq!(session.user.user_id, deserialized.user.user_id);
        assert_eq!(session.key_name, deserialized.key_name);
    }
}
