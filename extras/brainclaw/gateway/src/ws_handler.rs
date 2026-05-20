//! WebSocket connection handler for channel adapters.

use axum::extract::ws::{Message, WebSocket};
use chrono::Utc;
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use uuid::Uuid;

use brainwires_network::channels::{ChannelEvent, ChannelHandshake, ChannelHandshakeResponse};

use crate::channel_registry::ConnectedChannel;
use crate::state::AppState;

/// Handle a new WebSocket connection from a channel adapter.
///
/// Protocol:
/// 1. Receive first message as `ChannelHandshake`
/// 2. Validate auth token against config
/// 3. Send `ChannelHandshakeResponse`
/// 4. If accepted, register channel and spawn read/write loops
/// 5. On disconnect, unregister channel
pub async fn handle_ws_connection(mut ws: WebSocket, state: AppState) {
    // Step 1: Receive handshake
    let handshake = match receive_handshake(&mut ws).await {
        Some(hs) => hs,
        None => {
            tracing::warn!("Channel connection closed before handshake");
            return;
        }
    };

    tracing::info!(
        channel_type = %handshake.channel_type,
        version = %handshake.channel_version,
        "Channel handshake received"
    );

    // Step 2a: Check master channel switch
    if !state.config.channels_enabled {
        let response = ChannelHandshakeResponse {
            accepted: false,
            channel_id: None,
            error: Some("Channel connections are disabled".to_string()),
        };
        let _ = send_json(&mut ws, &response).await;
        tracing::warn!("Handshake rejected: channels_enabled=false");
        return;
    }

    // Step 2b: Check channel type allowlist
    if !state.config.allowed_channel_types.is_empty()
        && !state
            .config
            .allowed_channel_types
            .iter()
            .any(|t| t.eq_ignore_ascii_case(&handshake.channel_type))
    {
        let response = ChannelHandshakeResponse {
            accepted: false,
            channel_id: None,
            error: Some(format!(
                "Channel type '{}' is not in the allowed list",
                handshake.channel_type
            )),
        };
        let _ = send_json(&mut ws, &response).await;
        tracing::warn!(
            channel_type = %handshake.channel_type,
            "Handshake rejected: channel type not allowed"
        );
        return;
    }

    // Step 2c: Validate auth token
    if !state.config.validate_token(&handshake.auth_token) {
        let response = ChannelHandshakeResponse {
            accepted: false,
            channel_id: None,
            error: Some("Invalid authentication token".to_string()),
        };
        let _ = send_json(&mut ws, &response).await;
        tracing::warn!(
            channel_type = %handshake.channel_type,
            "Handshake rejected: invalid auth token"
        );
        return;
    }

    // Check max connections
    if state.channels.count() >= state.config.max_connections {
        let response = ChannelHandshakeResponse {
            accepted: false,
            channel_id: None,
            error: Some("Maximum connections reached".to_string()),
        };
        let _ = send_json(&mut ws, &response).await;
        tracing::warn!(
            channel_type = %handshake.channel_type,
            max = state.config.max_connections,
            "Handshake rejected: max connections"
        );
        return;
    }

    // Step 3: Accept and send response
    let channel_id = Uuid::new_v4();
    let response = ChannelHandshakeResponse {
        accepted: true,
        channel_id: Some(channel_id),
        error: None,
    };
    if send_json(&mut ws, &response).await.is_err() {
        tracing::error!("Failed to send handshake response");
        return;
    }

    tracing::info!(
        channel_id = %channel_id,
        channel_type = %handshake.channel_type,
        "Channel accepted"
    );

    // Step 4: Register channel and spawn read/write loops
    let (outbound_tx, mut outbound_rx) = mpsc::channel::<String>(256);

    let connected = ConnectedChannel {
        id: channel_id,
        channel_type: handshake.channel_type.clone(),
        capabilities: handshake.capabilities,
        connected_at: Utc::now(),
        last_heartbeat: Utc::now(),
        message_tx: outbound_tx,
    };
    state.channels.register(connected);

    // Split WebSocket into sender and receiver
    let (mut ws_sender, mut ws_receiver) = ws.split();

    // Spawn writer task: forward outbound messages to WebSocket
    let writer_handle = tokio::spawn(async move {
        while let Some(msg) = outbound_rx.recv().await {
            if ws_sender.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    // Read loop: process inbound messages from the channel
    while let Some(result) = ws_receiver.next().await {
        match result {
            Ok(Message::Text(text)) => match serde_json::from_str::<ChannelEvent>(&text) {
                Ok(event) => {
                    let router = state.router.clone();
                    let cid = channel_id;
                    tokio::spawn(async move {
                        if let Err(e) = router.handle_inbound(cid, &event).await {
                            tracing::error!(
                                channel_id = %cid,
                                error = %e,
                                "Failed to handle inbound event"
                            );
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        channel_id = %channel_id,
                        error = %e,
                        "Failed to deserialize channel event"
                    );
                }
            },
            Ok(Message::Close(_)) => {
                tracing::info!(channel_id = %channel_id, "Channel sent close frame");
                break;
            }
            Ok(Message::Ping(_)) => {
                // Axum handles pong automatically
                state.channels.touch_heartbeat(&channel_id);
            }
            Ok(_) => {
                // Binary or Pong frames -- ignore
            }
            Err(e) => {
                tracing::warn!(
                    channel_id = %channel_id,
                    error = %e,
                    "WebSocket read error"
                );
                break;
            }
        }
    }

    // Step 5: Cleanup on disconnect
    writer_handle.abort();
    state.channels.unregister(&channel_id);

    tracing::info!(
        channel_id = %channel_id,
        channel_type = %handshake.channel_type,
        "Channel disconnected"
    );
}

/// Receive the first WebSocket message and parse it as a `ChannelHandshake`.
async fn receive_handshake(ws: &mut WebSocket) -> Option<ChannelHandshake> {
    // Wait for the first text message
    while let Some(result) = ws.next().await {
        match result {
            Ok(Message::Text(text)) => {
                return serde_json::from_str::<ChannelHandshake>(&text).ok();
            }
            Ok(Message::Close(_)) => return None,
            Ok(_) => continue, // skip binary/ping/pong
            Err(_) => return None,
        }
    }
    None
}

/// Serialize a value to JSON and send it as a WebSocket text message.
async fn send_json<T: serde::Serialize>(ws: &mut WebSocket, value: &T) -> Result<(), axum::Error> {
    let json = serde_json::to_string(value).map_err(axum::Error::new)?;
    ws.send(Message::Text(json.into()))
        .await
        .map_err(axum::Error::new)
}

#[cfg(test)]
mod tests {
    use brainwires_network::channels::{
        ChannelCapabilities, ChannelHandshake, ChannelHandshakeResponse,
    };

    #[test]
    fn handshake_validation_accepted() {
        let config = crate::config::GatewayConfig {
            auth_tokens: vec!["valid-token".to_string()],
            ..Default::default()
        };

        let handshake = ChannelHandshake {
            channel_type: "discord".to_string(),
            channel_version: "1.0.0".to_string(),
            capabilities: ChannelCapabilities::RICH_TEXT,
            auth_token: "valid-token".to_string(),
        };

        assert!(config.validate_token(&handshake.auth_token));
    }

    #[test]
    fn handshake_validation_rejected() {
        let config = crate::config::GatewayConfig {
            auth_tokens: vec!["valid-token".to_string()],
            ..Default::default()
        };

        let handshake = ChannelHandshake {
            channel_type: "discord".to_string(),
            channel_version: "1.0.0".to_string(),
            capabilities: ChannelCapabilities::RICH_TEXT,
            auth_token: "wrong-token".to_string(),
        };

        assert!(!config.validate_token(&handshake.auth_token));
    }

    #[test]
    fn handshake_response_serialization() {
        let response = ChannelHandshakeResponse {
            accepted: true,
            channel_id: Some(uuid::Uuid::new_v4()),
            error: None,
        };
        let json = serde_json::to_string(&response).unwrap();
        let parsed: ChannelHandshakeResponse = serde_json::from_str(&json).unwrap();
        assert!(parsed.accepted);
        assert!(parsed.channel_id.is_some());
    }
}
