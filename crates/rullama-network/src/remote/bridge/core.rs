//! Core RemoteBridge struct definition, constructor, accessors, and top-level run loop.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use tokio::sync::RwLock;

use super::types::{
    AgentSubscription, BridgeConfig, BridgeState, ConnectionMode, RealtimeCredentials,
};
use crate::remote::attachments::AttachmentReceiver;
use crate::remote::heartbeat::HeartbeatCollector;
use crate::remote::protocol::{
    NegotiatedProtocol, ProtocolCapability, RemoteMessage, StreamChunkType,
};
use crate::traits::AgentSpawner;

const REMOTE_BRIDGE_TIMEOUT_SECS: u64 = 30;

/// Remote control bridge
///
/// Maintains communication with the backend using either Supabase Realtime
/// (preferred) or HTTP polling (fallback).
#[derive(Clone)]
pub struct RemoteBridge {
    pub(super) config: BridgeConfig,
    pub(super) http_client: reqwest::Client,
    /// Current bridge connection state.
    pub state: Arc<RwLock<BridgeState>>,
    pub(super) connection_mode: Arc<RwLock<ConnectionMode>>,
    pub(super) session_token: Arc<RwLock<Option<String>>>,
    pub(super) user_id: Arc<RwLock<Option<String>>>,
    pub(super) realtime_credentials: Arc<RwLock<Option<RealtimeCredentials>>>,
    pub(super) subscriptions: Arc<RwLock<HashSet<String>>>,
    pub(super) subscription_tasks: Arc<RwLock<HashMap<String, AgentSubscription>>>,
    pub(super) heartbeat_collector: Arc<RwLock<HeartbeatCollector>>,
    pub(super) command_result_queue: Arc<RwLock<Vec<RemoteMessage>>>,
    #[allow(clippy::type_complexity)]
    pub(super) stream_tx:
        Arc<RwLock<Option<tokio::sync::mpsc::Sender<(String, StreamChunkType, String)>>>>,
    pub(super) sync_trigger_tx: Arc<RwLock<Option<tokio::sync::mpsc::Sender<()>>>>,
    pub(super) shutdown_tx: Option<tokio::sync::broadcast::Sender<()>>,
    pub(super) negotiated_protocol: Arc<RwLock<NegotiatedProtocol>>,
    pub(super) attachment_receiver: AttachmentReceiver,
    /// Agent spawner for creating new agent processes (injected trait)
    pub(super) agent_spawner: Option<Arc<dyn AgentSpawner>>,
    /// Device allowlist status from last authentication.
    pub device_status: Arc<RwLock<Option<crate::remote::protocol::DeviceStatus>>>,
    /// Organization policies from last authentication.
    pub org_policies: Arc<RwLock<Option<crate::remote::protocol::OrgPolicies>>>,
    /// Permission relay for remote tool-approval prompts.
    pub permission_relay: crate::remote::permission_relay::PermissionRelay,
    /// Analytics collector for NetworkMessage events.
    #[cfg(feature = "telemetry")]
    pub(super) analytics_collector:
        Option<std::sync::Arc<rullama_telemetry::AnalyticsCollector>>,
}

impl RemoteBridge {
    /// Create a new remote bridge
    ///
    /// # Arguments
    /// * `config` - Bridge configuration with all injected platform values
    /// * `agent_spawner` - Optional agent spawner for remote agent creation
    pub fn new(config: BridgeConfig, agent_spawner: Option<Arc<dyn AgentSpawner>>) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(REMOTE_BRIDGE_TIMEOUT_SECS))
            .build()
            .expect("Failed to create HTTP client");

        let heartbeat_collector =
            HeartbeatCollector::new(config.sessions_dir.clone(), config.version.clone());

        let attachment_receiver = AttachmentReceiver::new(config.attachment_dir.clone());

        Self {
            config,
            http_client,
            state: Arc::new(RwLock::new(BridgeState::Disconnected)),
            connection_mode: Arc::new(RwLock::new(ConnectionMode::Polling)),
            session_token: Arc::new(RwLock::new(None)),
            user_id: Arc::new(RwLock::new(None)),
            realtime_credentials: Arc::new(RwLock::new(None)),
            subscriptions: Arc::new(RwLock::new(HashSet::new())),
            subscription_tasks: Arc::new(RwLock::new(HashMap::new())),
            heartbeat_collector: Arc::new(RwLock::new(heartbeat_collector)),
            command_result_queue: Arc::new(RwLock::new(Vec::new())),
            stream_tx: Arc::new(RwLock::new(None)),
            sync_trigger_tx: Arc::new(RwLock::new(None)),
            shutdown_tx: None,
            negotiated_protocol: Arc::new(RwLock::new(NegotiatedProtocol::default())),
            attachment_receiver,
            agent_spawner,
            device_status: Arc::new(RwLock::new(None)),
            org_policies: Arc::new(RwLock::new(None)),
            permission_relay: crate::remote::permission_relay::PermissionRelay::new(),
            #[cfg(feature = "telemetry")]
            analytics_collector: None,
        }
    }

    /// Attach an analytics collector to record NetworkMessage events.
    #[cfg(feature = "telemetry")]
    pub fn with_analytics(
        mut self,
        collector: std::sync::Arc<rullama_telemetry::AnalyticsCollector>,
    ) -> Self {
        self.analytics_collector = Some(collector);
        self
    }

    /// Get current connection mode
    pub async fn connection_mode(&self) -> ConnectionMode {
        *self.connection_mode.read().await
    }

    /// Get current bridge state
    pub async fn state(&self) -> BridgeState {
        *self.state.read().await
    }

    /// Check if bridge is connected and authenticated
    pub async fn is_ready(&self) -> bool {
        *self.state.read().await == BridgeState::Authenticated
    }

    /// Get the user ID (if authenticated)
    pub async fn user_id(&self) -> Option<String> {
        self.user_id.read().await.clone()
    }

    /// Get the negotiated protocol version
    pub async fn protocol_version(&self) -> String {
        self.negotiated_protocol.read().await.version.clone()
    }

    /// Check if a capability is enabled in the negotiated protocol
    pub async fn has_capability(&self, cap: ProtocolCapability) -> bool {
        self.negotiated_protocol.read().await.has_capability(cap)
    }

    /// Get all enabled capabilities
    pub async fn enabled_capabilities(&self) -> Vec<ProtocolCapability> {
        self.negotiated_protocol.read().await.capabilities.clone()
    }

    /// Send a permission request to the remote user and wait for their decision.
    ///
    /// Returns `Ok(decision)` if the user responds within the timeout,
    /// or `Ok(PermissionDecision { approved: false, .. })` on timeout.
    pub async fn send_permission_request(
        &self,
        agent_id: &str,
        tool_name: &str,
        action: &str,
        details: serde_json::Value,
    ) -> Result<crate::remote::permission_relay::PermissionDecision> {
        use crate::remote::permission_relay::PermissionDecision;

        // Check session-allowed list first
        if self.permission_relay.is_session_allowed(tool_name).await {
            return Ok(PermissionDecision {
                approved: true,
                remember_for_session: true,
                always_allow: true,
            });
        }

        let request_id = uuid::Uuid::new_v4().to_string();
        let timeout = self.permission_relay.default_timeout().await;

        // Register the pending request
        let rx = self
            .permission_relay
            .register_request(request_id.clone())
            .await;

        // Send the request message to the backend
        let msg = RemoteMessage::PermissionRequest {
            request_id: request_id.clone(),
            agent_id: agent_id.to_string(),
            tool_name: tool_name.to_string(),
            action: action.to_string(),
            details,
            timeout_secs: timeout.as_secs() as u32,
        };

        // Queue the message for sending
        self.command_result_queue.write().await.push(msg);

        // Wait for response with timeout
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(decision)) => Ok(decision),
            Ok(Err(_)) => {
                // Sender was dropped (request cancelled)
                Ok(PermissionDecision {
                    approved: false,
                    remember_for_session: false,
                    always_allow: false,
                })
            }
            Err(_) => {
                // Timeout — auto-deny
                self.permission_relay.cancel(&request_id).await;
                Ok(PermissionDecision {
                    approved: false,
                    remember_for_session: false,
                    always_allow: false,
                })
            }
        }
    }

    /// Set the shutdown signal sender (for external shutdown control)
    pub fn set_shutdown_tx(&mut self, tx: tokio::sync::broadcast::Sender<()>) {
        self.shutdown_tx = Some(tx);
    }

    /// Connect to the backend and run the main communication loop
    pub async fn run(&mut self) -> Result<()> {
        let shutdown_tx = self.shutdown_tx.clone().unwrap_or_else(|| {
            let (tx, _) = tokio::sync::broadcast::channel(1);
            self.shutdown_tx = Some(tx.clone());
            tx
        });

        let mut reconnect_attempts = 0;

        loop {
            if *self.state.read().await == BridgeState::ShuttingDown {
                tracing::info!("Remote bridge shutting down");
                break;
            }

            *self.state.write().await = BridgeState::Connecting;

            match self.register_with_backend().await {
                Ok(()) => {
                    reconnect_attempts = 0;
                    *self.state.write().await = BridgeState::Authenticated;

                    let realtime_creds = self.realtime_credentials.read().await.clone();

                    if let Some(creds) = realtime_creds {
                        *self.connection_mode.write().await = ConnectionMode::Realtime;
                        tracing::info!("Using Supabase Realtime for communication");

                        if let Err(e) = self.run_realtime_loop(shutdown_tx.subscribe(), creds).await
                        {
                            tracing::error!("Remote bridge Realtime error: {:?}", e);
                        }
                    } else {
                        *self.connection_mode.write().await = ConnectionMode::Polling;
                        tracing::info!(
                            "Using HTTP polling for communication (Realtime not available)"
                        );

                        if let Err(e) = self.run_polling_loop(shutdown_tx.subscribe()).await {
                            tracing::error!("Remote bridge polling error: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to register with backend: {}", e);
                    reconnect_attempts += 1;

                    if self.config.max_reconnect_attempts > 0
                        && reconnect_attempts >= self.config.max_reconnect_attempts
                    {
                        bail!(
                            "Max reconnect attempts ({}) reached",
                            self.config.max_reconnect_attempts
                        );
                    }
                }
            }

            // Clean up state
            *self.state.write().await = BridgeState::Disconnected;
            *self.connection_mode.write().await = ConnectionMode::Polling;
            *self.session_token.write().await = None;
            *self.realtime_credentials.write().await = None;
            self.subscriptions.write().await.clear();
            self.command_result_queue.write().await.clear();

            // Wait before reconnecting
            if *self.state.read().await != BridgeState::ShuttingDown {
                tracing::info!(
                    "Reconnecting in {} seconds...",
                    self.config.reconnect_delay_secs
                );
                tokio::time::sleep(Duration::from_secs(self.config.reconnect_delay_secs as u64))
                    .await;
            }
        }

        Ok(())
    }

    /// Shutdown the bridge
    pub async fn shutdown(&mut self) {
        *self.state.write().await = BridgeState::ShuttingDown;

        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(());
        }
    }

    /// Queue a command result to send with the next heartbeat
    pub(super) async fn queue_command_result_msg(&self, msg: RemoteMessage) -> Result<()> {
        self.command_result_queue.write().await.push(msg);
        Ok(())
    }
}
