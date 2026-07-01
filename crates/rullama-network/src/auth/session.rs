//! Session Management
//!
//! Manages authentication sessions with file-based storage and optional
//! keyring integration for secure API key storage.
//!
//! Unlike the CLI version, this module takes explicit paths and a `KeyStore`
//! trait object instead of depending on `PlatformPaths` and the `keyring` module.

use anyhow::{Context, Result};
use chrono::Utc;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, warn};
use zeroize::Zeroizing;

use super::types::{AuthResponse, AuthSession};
use crate::traits::KeyStore;

/// Session manager for handling authentication sessions
///
/// API keys are stored securely via the `KeyStore` trait (system keyring, encrypted file, etc.).
/// The session file only contains non-sensitive metadata.
pub struct SessionManager {
    /// Path to the session JSON file
    session_file: PathBuf,
    /// Optional secure key store for API keys
    key_store: Option<Box<dyn KeyStore>>,
}

impl SessionManager {
    /// Create a new session manager
    ///
    /// # Arguments
    /// * `session_file` - Path to the session JSON file
    /// * `key_store` - Optional secure key store for API keys. If None, API keys
    ///   are stored in the session file as a fallback.
    pub fn new(session_file: PathBuf, key_store: Option<Box<dyn KeyStore>>) -> Self {
        Self {
            session_file,
            key_store,
        }
    }

    /// Load session from disk
    pub fn load(&self) -> Result<Option<AuthSession>> {
        if !self.session_file.exists() {
            return Ok(None);
        }

        let contents =
            fs::read_to_string(&self.session_file).context("Failed to read session file")?;

        let session: AuthSession =
            serde_json::from_str(&contents).context("Failed to parse session file")?;

        Ok(Some(session))
    }

    /// Save session to disk and optionally store API key in key store
    ///
    /// The API key is stored in the key store if available, not in the session file.
    pub fn save(&self, session: &AuthSession, api_key: Option<&str>) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.session_file.parent() {
            fs::create_dir_all(parent)?;
        }

        // Store API key in key store if provided and available
        if let Some(key) = api_key {
            if let Some(ref key_store) = self.key_store {
                if let Err(e) = key_store.store_key(&session.user.user_id, key) {
                    warn!(
                        "Failed to store API key in key store: {}. Using fallback.",
                        e
                    );
                    // Fall back to storing in session file (less secure)
                    let mut session_with_key = session.clone();
                    session_with_key.api_key = key.to_string();
                    return self.save_session_file(&session_with_key);
                }
            } else {
                // No key store available - store in session file
                let mut session_with_key = session.clone();
                session_with_key.api_key = key.to_string();
                return self.save_session_file(&session_with_key);
            }
        }

        // Save session without API key
        let mut session_no_key = session.clone();
        session_no_key.api_key = String::new();
        self.save_session_file(&session_no_key)
    }

    /// Internal: Save session struct to file
    fn save_session_file(&self, session: &AuthSession) -> Result<()> {
        let contents =
            serde_json::to_string_pretty(session).context("Failed to serialize session")?;

        fs::write(&self.session_file, &contents).with_context(|| {
            format!(
                "Failed to write session file: {}",
                self.session_file.display()
            )
        })?;

        // Set file permissions to 0600 (owner read/write only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&self.session_file, fs::Permissions::from_mode(0o600))
                .context("Failed to set session file permissions")?;
        }

        Ok(())
    }

    /// Delete session from disk and key store
    pub fn delete(&self) -> Result<()> {
        // First, try to get the user_id to delete the key store entry
        if let Ok(Some(session)) = self.load()
            && let Some(ref key_store) = self.key_store
            && let Err(e) = key_store.delete_key(&session.user.user_id)
        {
            debug!("Failed to delete API key from key store: {}", e);
            // Continue anyway - deleting session file is more important
        }

        if self.session_file.exists() {
            fs::remove_file(&self.session_file).with_context(|| {
                format!(
                    "Failed to delete session file: {}",
                    self.session_file.display()
                )
            })?;
        }

        Ok(())
    }

    /// Check if user is authenticated (has valid session)
    pub fn is_authenticated(&self) -> Result<bool> {
        match self.load()? {
            Some(session) => Ok(!session.is_expired()),
            None => Ok(false),
        }
    }

    /// Get the current session if valid
    pub fn get_session(&self) -> Result<Option<AuthSession>> {
        match self.load()? {
            Some(session) if !session.is_expired() => Ok(Some(session)),
            _ => Ok(None),
        }
    }

    /// Get the API key for the current session
    ///
    /// Tries key store first, falls back to session file for backwards compatibility.
    /// Returns Zeroizing to ensure key is cleared from memory when dropped.
    pub fn get_api_key(&self) -> Result<Option<Zeroizing<String>>> {
        let session = match self.load()? {
            Some(s) => s,
            None => return Ok(None),
        };

        // Try key store first
        if let Some(ref key_store) = self.key_store {
            match key_store.get_key(&session.user.user_id) {
                Ok(Some(key)) => return Ok(Some(key)),
                Ok(None) => {
                    debug!("No key in key store, checking session file fallback");
                }
                Err(e) => {
                    debug!("Key store error: {}, checking session file fallback", e);
                }
            }
        }

        // Fall back to session file (for backwards compatibility)
        if !session.api_key.is_empty() {
            debug!("Using API key from session file (legacy)");
            return Ok(Some(Zeroizing::new(session.api_key)));
        }

        Ok(None)
    }

    /// Create session from authentication response
    pub fn create_session(
        response: AuthResponse,
        backend: String,
        _api_key: String,
    ) -> AuthSession {
        // Note: api_key is passed but stored in key store, not in session
        AuthSession {
            user: response.user,
            supabase: response.supabase,
            key_name: response.key_name,
            api_key: String::new(), // Stored in key store, not here
            backend,
            authenticated_at: Utc::now(),
        }
    }

    /// Migrate legacy session (with api_key in file) to key store
    ///
    /// Call this during login to migrate old sessions to secure storage.
    pub fn migrate_to_key_store(&self) -> Result<bool> {
        let key_store = match &self.key_store {
            Some(ks) => ks,
            None => return Ok(false), // No key store to migrate to
        };

        let session = match self.load()? {
            Some(s) => s,
            None => return Ok(false),
        };

        // If there's an API key in the session file, migrate it
        if !session.api_key.is_empty() {
            debug!("Migrating legacy API key to key store");

            // Store in key store
            key_store.store_key(&session.user.user_id, &session.api_key)?;

            // Clear from session file
            let mut updated = session;
            updated.api_key = String::new();
            self.save_session_file(&updated)?;

            return Ok(true);
        }

        Ok(false)
    }

    /// Get the session file path
    pub fn session_file(&self) -> &Path {
        &self.session_file
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::types::{SupabaseConfig, UserProfile};

    fn create_test_session() -> AuthSession {
        AuthSession {
            user: UserProfile {
                user_id: "test-user-id".to_string(),
                username: "testuser".to_string(),
                display_name: "Test User".to_string(),
                role: "basic".to_string(),
            },
            supabase: SupabaseConfig {
                url: "https://test.supabase.co".to_string(),
                anon_key: "test-anon-key".to_string(),
            },
            key_name: "test-key".to_string(),
            api_key: "bw_dev_12345678901234567890123456789012".to_string(),
            backend: "https://brainwires.studio".to_string(),
            authenticated_at: Utc::now(),
        }
    }

    fn make_manager() -> (tempfile::TempDir, SessionManager) {
        let temp_dir = tempfile::tempdir().unwrap();
        let session_file = temp_dir.path().join("session.json");
        let mgr = SessionManager::new(session_file, None);
        (temp_dir, mgr)
    }

    #[test]
    fn test_session_never_expires() {
        let session = create_test_session();
        assert!(!session.is_expired());
    }

    #[test]
    fn test_create_session() {
        let auth_response = AuthResponse {
            user: UserProfile {
                user_id: "user123".to_string(),
                username: "john".to_string(),
                display_name: "John Doe".to_string(),
                role: "admin".to_string(),
            },
            supabase: SupabaseConfig {
                url: "https://test.supabase.co".to_string(),
                anon_key: "anon-test".to_string(),
            },
            key_name: "my_key".to_string(),
        };

        let session = SessionManager::create_session(
            auth_response,
            "https://brainwires.studio".to_string(),
            "bw_dev_12345678901234567890123456789012".to_string(),
        );

        assert_eq!(session.user.user_id, "user123");
        assert_eq!(session.key_name, "my_key");
        assert_eq!(session.backend, "https://brainwires.studio");
        assert!(session.api_key.is_empty()); // Stored in key store, not session
        assert!(!session.is_expired());
    }

    #[test]
    fn test_save_and_load_session() {
        let (_dir, mgr) = make_manager();
        let session = create_test_session();

        mgr.save(&session, None).unwrap();
        let loaded = mgr.load().unwrap();
        assert!(loaded.is_some());
    }

    #[test]
    fn test_load_nonexistent_session() {
        let (_dir, mgr) = make_manager();
        let result = mgr.load().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_delete_session() {
        let (_dir, mgr) = make_manager();
        let session = create_test_session();

        mgr.save(&session, None).unwrap();
        mgr.delete().unwrap();

        let loaded = mgr.load().unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_delete_nonexistent_session() {
        let (_dir, mgr) = make_manager();
        let result = mgr.delete();
        assert!(result.is_ok());
    }

    #[test]
    fn test_is_authenticated_with_valid_session() {
        let (_dir, mgr) = make_manager();
        let session = create_test_session();
        mgr.save(&session, None).unwrap();
        assert!(mgr.is_authenticated().unwrap());
    }

    #[test]
    fn test_is_authenticated_without_session() {
        let (_dir, mgr) = make_manager();
        assert!(!mgr.is_authenticated().unwrap());
    }

    #[test]
    fn test_get_session_valid() {
        let (_dir, mgr) = make_manager();
        let session = create_test_session();
        mgr.save(&session, None).unwrap();

        let result = mgr.get_session().unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().user.user_id, "test-user-id");
    }

    #[test]
    fn test_get_session_none() {
        let (_dir, mgr) = make_manager();
        let result = mgr.get_session().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_save_with_api_key_no_keystore() {
        let (_dir, mgr) = make_manager();
        let session = create_test_session();

        // Without a key store, API key should be stored in the session file
        mgr.save(&session, Some("bw_test_00000000000000000000000000000000"))
            .unwrap();

        let loaded = mgr.load().unwrap().unwrap();
        assert_eq!(loaded.api_key, "bw_test_00000000000000000000000000000000");
    }

    #[test]
    fn test_get_api_key_from_session_file() {
        let (_dir, mgr) = make_manager();
        let session = create_test_session();

        // Save with API key in file (no key store)
        mgr.save(&session, Some("bw_test_00000000000000000000000000000000"))
            .unwrap();

        let key = mgr.get_api_key().unwrap();
        assert!(key.is_some());
        assert_eq!(
            key.unwrap().as_str(),
            "bw_test_00000000000000000000000000000000"
        );
    }

    #[test]
    fn test_session_serialization() {
        let session = create_test_session();
        let json = serde_json::to_string(&session).unwrap();
        let deserialized: AuthSession = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.user.user_id, session.user.user_id);
        assert_eq!(deserialized.key_name, session.key_name);
        assert_eq!(deserialized.backend, session.backend);
    }

    #[test]
    fn test_old_session_does_not_expire() {
        let mut session = create_test_session();
        session.authenticated_at = Utc::now() - chrono::Duration::days(365);
        assert!(!session.is_expired(), "Sessions should never expire");
    }
}
