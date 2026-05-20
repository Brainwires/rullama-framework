//! Integration tests for the CommunicationHub.
//!
//! Tests multi-agent messaging, broadcast delivery, and agent lifecycle
//! interactions across the communication module.

use brainwires_agent::communication::{AgentMessage, CommunicationHub};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Multi-agent registration and messaging
// ---------------------------------------------------------------------------

#[tokio::test]
async fn register_multiple_agents_and_send_directed_messages() {
    let hub = CommunicationHub::new();

    hub.register_agent("agent-a".into()).await.unwrap();
    hub.register_agent("agent-b".into()).await.unwrap();
    hub.register_agent("agent-c".into()).await.unwrap();

    assert_eq!(hub.agent_count().await, 3);

    // Agent A sends a task request to Agent B
    hub.send_message(
        "agent-a".into(),
        "agent-b".into(),
        AgentMessage::TaskRequest {
            task_id: "task-42".into(),
            description: "Implement caching".into(),
            priority: 3,
        },
    )
    .await
    .unwrap();

    // Agent B should receive the message
    let envelope = hub.try_receive_message("agent-b").await.unwrap();
    assert_eq!(envelope.from, "agent-a");
    assert_eq!(envelope.to, "agent-b");

    // Agent C should have no messages
    assert!(hub.try_receive_message("agent-c").await.is_none());
}

#[tokio::test]
async fn send_to_unregistered_agent_fails() {
    let hub = CommunicationHub::new();
    hub.register_agent("agent-a".into()).await.unwrap();

    let result = hub
        .send_message(
            "agent-a".into(),
            "ghost".into(),
            AgentMessage::Broadcast {
                sender: "agent-a".into(),
                message: "hello?".into(),
            },
        )
        .await;

    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Broadcast delivery
// ---------------------------------------------------------------------------

#[tokio::test]
async fn broadcast_delivers_to_all_registered_agents() {
    let hub = CommunicationHub::new();

    let agents = ["alpha", "beta", "gamma", "delta"];
    for a in &agents {
        hub.register_agent(a.to_string()).await.unwrap();
    }

    hub.broadcast(
        "orchestrator".into(),
        AgentMessage::Broadcast {
            sender: "orchestrator".into(),
            message: "System reset in 5 minutes".into(),
        },
    )
    .await
    .unwrap();

    // Every registered agent should receive exactly one message
    for a in &agents {
        let env = hub.try_receive_message(a).await;
        assert!(env.is_some(), "Agent {} should have received broadcast", a);
        // Second receive should be empty
        assert!(hub.try_receive_message(a).await.is_none());
    }
}

#[tokio::test]
async fn broadcast_status_updates_across_agents() {
    let hub = CommunicationHub::new();

    hub.register_agent("worker-1".into()).await.unwrap();
    hub.register_agent("worker-2".into()).await.unwrap();
    hub.register_agent("monitor".into()).await.unwrap();

    // Worker-1 broadcasts a status update
    hub.broadcast(
        "worker-1".into(),
        AgentMessage::StatusUpdate {
            agent_id: "worker-1".into(),
            status: "compiling".into(),
            details: Some("Building module X".into()),
        },
    )
    .await
    .unwrap();

    // All agents (including worker-1 itself) should receive
    for agent_id in &["worker-1", "worker-2", "monitor"] {
        let env = hub.try_receive_message(agent_id).await.unwrap();
        match env.message {
            AgentMessage::StatusUpdate {
                ref agent_id,
                ref status,
                ..
            } => {
                assert_eq!(agent_id, "worker-1");
                assert_eq!(status, "compiling");
            }
            _ => panic!("Expected StatusUpdate message"),
        }
    }
}

// ---------------------------------------------------------------------------
// Agent lifecycle: register -> communicate -> unregister
// ---------------------------------------------------------------------------

#[tokio::test]
async fn agent_lifecycle_register_communicate_unregister() {
    let hub = CommunicationHub::new();

    // Register two agents
    hub.register_agent("ephemeral".into()).await.unwrap();
    hub.register_agent("persistent".into()).await.unwrap();

    // Communicate
    hub.send_message(
        "persistent".into(),
        "ephemeral".into(),
        AgentMessage::HelpRequest {
            request_id: "help-1".into(),
            topic: "syntax".into(),
            details: "Need help parsing".into(),
        },
    )
    .await
    .unwrap();

    // Ephemeral receives and responds
    let received = hub.try_receive_message("ephemeral").await.unwrap();
    match received.message {
        AgentMessage::HelpRequest { ref request_id, .. } => {
            // Send response back
            hub.send_message(
                "ephemeral".into(),
                "persistent".into(),
                AgentMessage::HelpResponse {
                    request_id: request_id.clone(),
                    response: "Use regex".into(),
                },
            )
            .await
            .unwrap();
        }
        _ => panic!("Expected HelpRequest"),
    }

    // Unregister ephemeral
    hub.unregister_agent("ephemeral").await.unwrap();
    assert!(!hub.is_registered("ephemeral").await);
    assert_eq!(hub.agent_count().await, 1);

    // Can't send to unregistered agent
    let result = hub
        .send_message(
            "persistent".into(),
            "ephemeral".into(),
            AgentMessage::Broadcast {
                sender: "persistent".into(),
                message: "Are you there?".into(),
            },
        )
        .await;
    assert!(result.is_err());

    // Persistent should still have the help response
    let env = hub.try_receive_message("persistent").await.unwrap();
    match env.message {
        AgentMessage::HelpResponse { ref response, .. } => {
            assert_eq!(response, "Use regex");
        }
        _ => panic!("Expected HelpResponse"),
    }
}

#[tokio::test]
async fn duplicate_registration_fails() {
    let hub = CommunicationHub::new();
    hub.register_agent("agent-1".into()).await.unwrap();
    let result = hub.register_agent("agent-1".into()).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn unregister_unknown_agent_fails() {
    let hub = CommunicationHub::new();
    let result = hub.unregister_agent("nobody").await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Concurrent message sending from multiple agents
// ---------------------------------------------------------------------------

#[tokio::test]
async fn concurrent_sends_to_single_agent() {
    let hub = Arc::new(CommunicationHub::new());

    hub.register_agent("receiver".into()).await.unwrap();

    let sender_count = 10;
    for i in 0..sender_count {
        hub.register_agent(format!("sender-{}", i)).await.unwrap();
    }

    // All senders send concurrently
    let mut handles = Vec::new();
    for i in 0..sender_count {
        let hub_clone = Arc::clone(&hub);
        handles.push(tokio::spawn(async move {
            hub_clone
                .send_message(
                    format!("sender-{}", i),
                    "receiver".into(),
                    AgentMessage::AgentProgress {
                        agent_id: format!("sender-{}", i),
                        progress_percent: (i * 10) as u8,
                        message: format!("Progress from sender {}", i),
                    },
                )
                .await
                .unwrap();
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    // Receiver should have exactly sender_count messages
    let mut received = 0;
    while hub.try_receive_message("receiver").await.is_some() {
        received += 1;
    }
    assert_eq!(received, sender_count);
}

// ---------------------------------------------------------------------------
// list_agents
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_agents_returns_all_registered() {
    let hub = CommunicationHub::new();
    hub.register_agent("x".into()).await.unwrap();
    hub.register_agent("y".into()).await.unwrap();
    hub.register_agent("z".into()).await.unwrap();

    let mut agents = hub.list_agents().await;
    agents.sort();
    assert_eq!(agents, vec!["x", "y", "z"]);
}
