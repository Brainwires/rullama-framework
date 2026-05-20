//! Shared application state for the gateway.

use std::sync::Arc;

use chrono::{DateTime, Utc};

use brainwires_core::Provider;

use crate::audit::AuditLogger;
use crate::channel_registry::ChannelRegistry;
use crate::config::GatewayConfig;
use crate::cron::CronStore;
use crate::identity::UserIdentityStore;
use crate::metrics::MetricsCollector;
use crate::middleware::rate_limit::RateLimiter;
use crate::middleware::sanitizer::MessageSanitizer;
use crate::pairing::PairingStore;
use crate::router::InboundHandler;
use crate::session::SessionManager;
use crate::webchat::{SharedVerifier, WebChatHistory};

#[cfg(feature = "email-push")]
use crate::gmail_push::GmailPushRegistry;

/// Shared application state, passed to all axum handlers via Extension.
#[derive(Clone)]
pub struct AppState {
    /// Gateway configuration.
    pub config: Arc<GatewayConfig>,
    /// Session manager for user-to-agent mapping.
    pub sessions: Arc<SessionManager>,
    /// Registry of connected channel adapters.
    pub channels: Arc<ChannelRegistry>,
    /// Inbound event handler (trait object for extensibility).
    pub router: Arc<dyn InboundHandler>,
    /// Message sanitizer for inbound/outbound security.
    pub sanitizer: Arc<MessageSanitizer>,
    /// Per-user rate limiter.
    pub rate_limiter: Arc<RateLimiter>,
    /// Audit logger for security events.
    pub audit: Arc<AuditLogger>,
    /// In-memory metrics collector.
    pub metrics: Arc<MetricsCollector>,
    /// When the gateway was started.
    pub start_time: DateTime<Utc>,
    /// Optional LLM provider for the OpenAI-compatible API endpoint.
    pub openai_provider: Option<Arc<dyn Provider>>,
    /// Optional directory for serving TTS audio files at `/audio/<filename>`.
    pub audio_dir: Option<std::path::PathBuf>,
    /// Optional cron job store for the admin cron API.
    pub cron_store: Option<Arc<CronStore>>,
    /// Optional cross-channel user identity store.
    pub identity_store: Option<Arc<UserIdentityStore>>,
    /// Optional pairing store exposed to the admin pairing endpoints.
    pub pairing_store: Option<Arc<PairingStore>>,
    /// Optional bearer-token verifier for the JWT-gated `/webchat/ws`
    /// endpoint. When `None`, `/webchat/ws` rejects every upgrade.
    pub webchat_verifier: Option<SharedVerifier>,
    /// Optional per-session history store for the webchat channel.
    /// When `None`, history is not retained and `/webchat/history/:id`
    /// returns an empty list.
    pub webchat_history: Option<Arc<WebChatHistory>>,
    /// Optional Gmail push registry. When `Some`, the gateway exposes
    /// `POST /webhooks/gmail-push` and authenticates requests against the
    /// per-account [`GmailPushHandler`]s kept inside.
    #[cfg(feature = "email-push")]
    pub gmail_push_registry: Option<Arc<GmailPushRegistry>>,
}
