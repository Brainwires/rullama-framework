/// Authentication client for the Brainwires backend.
pub mod client;
/// Session persistence and management.
pub mod session;
/// Authentication types (session, profile, config).
pub mod types;

#[cfg(feature = "auth-keyring")]
pub mod keyring;

pub use client::AuthClient;
pub use session::SessionManager;
pub use types::*;
