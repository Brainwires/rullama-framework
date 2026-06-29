//! Example: Agent system quickstart
//!
//! Shows how to set up the core agent infrastructure (CommunicationHub,
//! FileLockManager) and coordinate multiple agents via file locks and messaging.
//!
//! Run: cargo run -p rullama --example agent_quickstart --features agents

use rullama::agents::{AgentMessage, CommunicationHub, FileLockManager, LockType};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // 1. Create shared infrastructure
    let hub = CommunicationHub::new();
    let lock_manager = Arc::new(FileLockManager::new());

    // Register two agents with the communication hub
    hub.register_agent("agent-1".to_string()).await?;
    hub.register_agent("agent-2".to_string()).await?;

    // 2. Demonstrate file locking coordination
    // Multiple agents can hold read locks concurrently...
    {
        let _read1 = lock_manager
            .acquire_lock("agent-1", "src/auth.rs", LockType::Read)
            .await?;
        let _read2 = lock_manager
            .acquire_lock("agent-2", "src/auth.rs", LockType::Read)
            .await?;
        println!("Two agents reading src/auth.rs concurrently");
    }
    // Locks are released when dropped

    // ...but write locks are exclusive
    {
        let _write = lock_manager
            .acquire_lock("agent-1", "src/auth.rs", LockType::Write)
            .await?;
        println!("One agent has exclusive write access to src/auth.rs");
    }

    // 3. Use the communication hub for agent coordination
    hub.broadcast(
        "agent-1".to_string(),
        AgentMessage::StatusUpdate {
            agent_id: "agent-1".to_string(),
            status: "working".to_string(),
            details: Some("Analyzing auth module".to_string()),
        },
    )
    .await?;

    // Receive messages (agent-2 will get the broadcast from agent-1)
    match hub.receive_message("agent-2").await {
        Some(envelope) => println!("agent-2 received: {:?}", envelope.message),
        None => println!("No messages in queue"),
    }

    println!("\nAgent infrastructure ready!");
    println!("  - CommunicationHub: broadcasting messages between agents");
    println!("  - FileLockManager: coordinating file access with read/write locks");

    // To build a full autonomous agent, implement the AgentRuntime trait
    // (rullama::agents::AgentRuntime) which defines methods for the
    // agent loop: call_provider, execute_tool, on_completion, etc.
    // Then call run_agent_loop(runtime, provider, hub, lock_manager) to execute it.

    Ok(())
}
