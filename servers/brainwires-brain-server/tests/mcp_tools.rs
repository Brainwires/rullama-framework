//! Integration tests for all seven MCP tools exposed by `brainwires-brain-server`.
//!
//! Each test spins up an in-process `BrainMcpServer` over a `tokio::io::duplex`
//! pair, connects an rmcp client to it, and exercises the tool through the MCP
//! protocol exactly the way an external AI tool would.
//!
//! Storage is backed by a per-test `TempDir`, so tests run isolated and in
//! parallel safely.

use anyhow::{Context, Result};
use brainwires_brain_server::mcp_server::BrainMcpServer;
use brainwires_knowledge::knowledge::brain_client::BrainClient;
use rmcp::{ServiceExt, model::CallToolRequestParams, service::RunningService};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::DuplexStream;

/// Spin up a fresh `BrainMcpServer` backed by `temp` over an in-memory duplex
/// and return a connected rmcp client. The server task is spawned on the
/// current runtime; the returned `TempDir` and client must both stay alive
/// for the duration of the test.
async fn start_server(temp: &TempDir) -> Result<RunningService<rmcp::RoleClient, ()>> {
    let lance_path = temp.path().join("brain.lance");
    let pks_path = temp.path().join("pks.db");
    let bks_path = temp.path().join("bks.db");

    let brain_client = BrainClient::with_paths(
        lance_path.to_str().context("lance path not UTF-8")?,
        pks_path.to_str().context("pks path not UTF-8")?,
        bks_path.to_str().context("bks path not UTF-8")?,
    )
    .await
    .context("failed to build BrainClient")?;

    let server = BrainMcpServer::with_client(brain_client).context("failed to build server")?;

    let (server_transport, client_transport): (DuplexStream, DuplexStream) =
        tokio::io::duplex(8 * 1024);

    // Drive the server in a background task. It runs until the client drops
    // its half of the duplex at the end of the test.
    tokio::spawn(async move {
        if let Ok(running) = server.serve(server_transport).await {
            let _ = running.waiting().await;
        }
    });

    // `()` implements `ClientHandler`, which is all we need for driving tools.
    let client = ().serve(client_transport).await.context("client failed to init")?;

    Ok(client)
}

/// Call a tool and deserialize its first text-content chunk as JSON.
async fn call_tool_json(
    client: &RunningService<rmcp::RoleClient, ()>,
    name: &'static str,
    args: Value,
) -> Result<Value> {
    let params = CallToolRequestParams::new(name).with_arguments(
        args.as_object()
            .cloned()
            .context("arguments must be a JSON object")?,
    );
    let result = client.call_tool(params).await.context("call_tool failed")?;

    assert!(
        result.is_error != Some(true),
        "tool {name} returned is_error=true: {:?}",
        result.content
    );

    let text = result
        .content
        .first()
        .and_then(|c| c.raw.as_text())
        .map(|t| t.text.as_str())
        .with_context(|| format!("tool {name} returned no text content"))?;

    serde_json::from_str::<Value>(text)
        .with_context(|| format!("tool {name} response was not JSON: {text}"))
}

// ── 1. capture_thought → get_thought round-trip ──────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capture_thought_round_trip() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    let capture_resp = call_tool_json(
        &client,
        "capture_thought",
        json!({
            "content": "Round-trip integration test thought",
            "category": "insight",
            "tags": ["integration-test"],
        }),
    )
    .await?;

    let id = capture_resp["id"]
        .as_str()
        .context("capture response missing id")?
        .to_string();
    assert!(
        !id.is_empty(),
        "captured thought should have a non-empty id"
    );

    let got = call_tool_json(&client, "get_thought", json!({ "id": id })).await?;

    assert_eq!(got["id"].as_str(), Some(id.as_str()));
    assert_eq!(
        got["content"].as_str(),
        Some("Round-trip integration test thought")
    );

    Ok(())
}

// ── 2. search_memory finds a captured thought ────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_memory_finds_captured_thought() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    let capture_resp = call_tool_json(
        &client,
        "capture_thought",
        json!({
            "content": "brainwires-test-magic-word appears exactly here",
        }),
    )
    .await?;
    let id = capture_resp["id"]
        .as_str()
        .context("capture id missing")?
        .to_string();

    // min_score=0.0 so we don't depend on an embedding similarity cutoff.
    let resp = call_tool_json(
        &client,
        "search_memory",
        json!({
            "query": "brainwires-test-magic-word",
            "limit": 10,
            "min_score": 0.0,
            "sources": ["thoughts"],
        }),
    )
    .await?;

    let results = resp["results"].as_array().context("results not an array")?;
    assert!(
        !results.is_empty(),
        "search should return at least one match"
    );

    let matched = results
        .iter()
        .any(|r| r["thought_id"].as_str() == Some(id.as_str()));
    assert!(matched, "captured thought id {id} not in search results");

    Ok(())
}

// ── 3. list_recent returns captures newest-first ─────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_recent_returns_captures_in_order() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    let mut ids = Vec::new();
    for i in 0..3 {
        // Small sleep to ensure distinct `created_at` ordering (second
        // resolution). 1.1 s per capture keeps the test under ~4 s total.
        if i > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(1_100)).await;
        }
        let resp = call_tool_json(
            &client,
            "capture_thought",
            json!({ "content": format!("list-recent-test-{i}") }),
        )
        .await?;
        ids.push(
            resp["id"]
                .as_str()
                .context("capture id missing")?
                .to_string(),
        );
    }

    let resp = call_tool_json(&client, "list_recent", json!({ "limit": 10 })).await?;
    let thoughts = resp["thoughts"]
        .as_array()
        .context("thoughts not an array")?;

    assert_eq!(thoughts.len(), 3, "expected 3 recent thoughts");

    // Newest first: last captured id should be first in the response.
    let first_id = thoughts[0]["id"].as_str().unwrap_or("");
    assert_eq!(
        first_id,
        ids.last().unwrap(),
        "newest-first ordering violated: got {first_id}, expected {}",
        ids.last().unwrap()
    );

    Ok(())
}

// ── 4. get_thought with an unknown id returns a not-found shape ──────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_thought_unknown_id_returns_not_found() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    let bogus = uuid::Uuid::new_v4().to_string();
    let resp = call_tool_json(&client, "get_thought", json!({ "id": bogus })).await?;

    // `mcp_server.rs` emits `{"error": "Thought not found: <id>"}` for the
    // missing-id case — see BrainMcpServer::get_thought.
    let err = resp["error"]
        .as_str()
        .context("expected an 'error' field on not-found response")?;
    assert!(
        err.contains("not found") || err.contains("Not found") || err.contains("Thought not found"),
        "unexpected not-found error message: {err}"
    );

    Ok(())
}

// ── 5. delete_thought removes the row ────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_thought_removes_row() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    let capture_resp = call_tool_json(
        &client,
        "capture_thought",
        json!({ "content": "delete-me integration test" }),
    )
    .await?;
    let id = capture_resp["id"]
        .as_str()
        .context("capture id missing")?
        .to_string();

    let del_resp = call_tool_json(&client, "delete_thought", json!({ "id": id })).await?;
    assert_eq!(
        del_resp["deleted"].as_bool(),
        Some(true),
        "delete_thought should report deleted=true"
    );
    assert_eq!(del_resp["id"].as_str(), Some(id.as_str()));

    let after = call_tool_json(&client, "get_thought", json!({ "id": id })).await?;
    assert!(
        after.get("error").and_then(|e| e.as_str()).is_some(),
        "get_thought after delete should return an error shape, got: {after}"
    );

    Ok(())
}

// ── 6. memory_stats reflects captures ────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn memory_stats_reflects_captures() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    // Fresh store starts at zero thoughts.
    let stats0 = call_tool_json(&client, "memory_stats", json!({})).await?;
    assert_eq!(
        stats0["thoughts"]["total"].as_u64(),
        Some(0),
        "fresh store should report 0 thoughts"
    );

    const N: usize = 4;
    for i in 0..N {
        call_tool_json(
            &client,
            "capture_thought",
            json!({ "content": format!("stats-thought-{i}") }),
        )
        .await?;
    }

    let stats = call_tool_json(&client, "memory_stats", json!({})).await?;
    assert_eq!(
        stats["thoughts"]["total"].as_u64(),
        Some(N as u64),
        "thoughts.total should equal the number of captures"
    );
    // Sanity: recent-window counters should be populated too.
    assert_eq!(stats["thoughts"]["recent_24h"].as_u64(), Some(N as u64));

    Ok(())
}

// ── 7. search_knowledge smoke test on an empty store ─────────────────────
//
// `search_knowledge` queries PKS (personal facts) and BKS (behavioral truths).
// Neither store has seed data on a fresh TempDir, so this is a smoke test:
// the tool must respond successfully with an empty result set and not panic.
// Fuller PKS/BKS coverage belongs in `brainwires-knowledge`'s own test suite.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_knowledge_empty_store_returns_empty() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    let resp = call_tool_json(
        &client,
        "search_knowledge",
        json!({
            "query": "anything",
            "source": "all",
            "min_confidence": 0.0,
            "limit": 10,
        }),
    )
    .await?;

    let results = resp["results"].as_array().context("results not an array")?;
    assert!(
        results.is_empty(),
        "empty store should yield no knowledge results, got {} entries",
        results.len()
    );
    assert_eq!(resp["total"].as_u64(), Some(0));

    Ok(())
}
