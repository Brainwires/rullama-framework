//! Unit tests for the remote bridge.

use super::core::RemoteBridge;
use super::types::{BridgeConfig, BridgeState};
use crate::remote::protocol::RemoteMessage;

#[test]
fn test_bridge_config_default() {
    let config = BridgeConfig::default();
    assert!(config.backend_url.starts_with("https://"));
    assert_eq!(config.heartbeat_interval_secs, 5);
    assert_eq!(config.version, "unknown");
}

#[tokio::test]
async fn test_bridge_state() {
    let config = BridgeConfig::default();
    let bridge = RemoteBridge::new(config, None);

    assert_eq!(bridge.state().await, BridgeState::Disconnected);
    assert!(!bridge.is_ready().await);
    assert!(bridge.user_id().await.is_none());
}

#[tokio::test]
async fn test_bridge_command_result_queue() {
    let config = BridgeConfig::default();
    let bridge = RemoteBridge::new(config, None);

    // Queue should start empty
    assert!(bridge.command_result_queue.read().await.is_empty());

    // Queue a command result message
    bridge
        .queue_command_result_msg(RemoteMessage::Pong { timestamp: 12345 })
        .await
        .unwrap();

    // Queue should have one message
    assert_eq!(bridge.command_result_queue.read().await.len(), 1);
}
