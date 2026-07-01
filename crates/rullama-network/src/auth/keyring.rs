//! Secure API key storage using system keyring
//!
//! This module provides secure storage for API keys using the platform's native
//! credential storage:
//! - Linux: secret-service (GNOME Keyring / KWallet)
//! - macOS: Keychain
//! - Windows: Credential Manager
//!
//! Implements the `KeyStore` trait for use as a pluggable credential backend.
//!
//! This module requires the `auth-keyring` feature flag.

use anyhow::{Context, Result};
use keyring::Entry;
use tracing::{debug, warn};
use zeroize::Zeroizing;

use crate::traits::KeyStore;

const SERVICE_NAME: &str = "rullama-cli";
const API_KEY_ACCOUNT: &str = "api_key";

/// Keyring-backed implementation of `KeyStore`
///
/// Uses the system's native credential storage (GNOME Keyring, macOS Keychain,
/// Windows Credential Manager).
pub struct KeyringKeyStore;

impl KeyringKeyStore {
    /// Create a new keyring key store
    pub fn new() -> Self {
        Self
    }
}

impl Default for KeyringKeyStore {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyStore for KeyringKeyStore {
    fn store_key(&self, user_id: &str, key: &str) -> Result<()> {
        store_api_key(user_id, key)
    }

    fn get_key(&self, user_id: &str) -> Result<Option<Zeroizing<String>>> {
        get_api_key(user_id)
    }

    fn delete_key(&self, user_id: &str) -> Result<()> {
        delete_api_key(user_id)
    }

    fn is_available(&self) -> bool {
        is_keyring_available()
    }
}

/// Store an API key in the system keyring
///
/// The key is associated with the user_id for multi-account support.
pub fn store_api_key(user_id: &str, api_key: &str) -> Result<()> {
    let account = format!("{}:{}", API_KEY_ACCOUNT, user_id);
    let entry = Entry::new(SERVICE_NAME, &account).context("Failed to create keyring entry")?;

    entry
        .set_password(api_key)
        .context("Failed to store API key in keyring")?;

    debug!("API key stored in system keyring for user {}", user_id);
    Ok(())
}

/// Retrieve an API key from the system keyring
///
/// Returns the API key wrapped in Zeroizing to ensure it's cleared from memory
/// when dropped.
pub fn get_api_key(user_id: &str) -> Result<Option<Zeroizing<String>>> {
    let account = format!("{}:{}", API_KEY_ACCOUNT, user_id);
    let entry = Entry::new(SERVICE_NAME, &account).context("Failed to create keyring entry")?;

    match entry.get_password() {
        Ok(key) => {
            debug!("API key retrieved from system keyring for user {}", user_id);
            Ok(Some(Zeroizing::new(key)))
        }
        Err(keyring::Error::NoEntry) => {
            debug!("No API key found in keyring for user {}", user_id);
            Ok(None)
        }
        Err(e) => {
            warn!("Failed to retrieve API key from keyring: {}", e);
            Err(e).context("Failed to retrieve API key from keyring")
        }
    }
}

/// Delete an API key from the system keyring
pub fn delete_api_key(user_id: &str) -> Result<()> {
    let account = format!("{}:{}", API_KEY_ACCOUNT, user_id);
    let entry = Entry::new(SERVICE_NAME, &account).context("Failed to create keyring entry")?;

    match entry.delete_credential() {
        Ok(()) => {
            debug!("API key deleted from system keyring for user {}", user_id);
            Ok(())
        }
        Err(keyring::Error::NoEntry) => {
            // Already deleted or never existed - that's fine
            Ok(())
        }
        Err(e) => {
            warn!("Failed to delete API key from keyring: {}", e);
            Err(e).context("Failed to delete API key from keyring")
        }
    }
}

/// Check if keyring is available on this system
pub fn is_keyring_available() -> bool {
    // Try to create a test entry
    match Entry::new(SERVICE_NAME, "test_availability") {
        Ok(_) => true,
        Err(e) => {
            debug!("Keyring not available: {}", e);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a working keyring on the system
    // They are marked as ignored by default since CI environments
    // typically don't have keyring access

    #[test]
    #[ignore = "requires system keyring"]
    fn test_store_and_retrieve_api_key() {
        let test_user = "test-user-keyring-bridge-1";
        let test_key = "bw_test_12345678901234567890123456789012";

        // Clean up any existing key
        let _ = delete_api_key(test_user);

        // Store the key
        store_api_key(test_user, test_key).expect("Failed to store key");

        // Retrieve the key
        let retrieved = get_api_key(test_user)
            .expect("Failed to retrieve key")
            .expect("Key not found");

        assert_eq!(retrieved.as_str(), test_key);

        // Clean up
        delete_api_key(test_user).expect("Failed to delete key");
    }

    #[test]
    #[ignore = "requires system keyring"]
    fn test_keystore_trait() {
        let store = KeyringKeyStore::new();
        let test_user = "test-user-keyring-bridge-trait";
        let test_key = "bw_test_00000000000000000000000000000000";

        let _ = store.delete_key(test_user);

        store.store_key(test_user, test_key).unwrap();
        let retrieved = store.get_key(test_user).unwrap().unwrap();
        assert_eq!(retrieved.as_str(), test_key);

        store.delete_key(test_user).unwrap();
        let gone = store.get_key(test_user).unwrap();
        assert!(gone.is_none());
    }

    #[test]
    fn test_is_keyring_available() {
        // This test just verifies the function doesn't panic
        let _available = is_keyring_available();
    }
}
