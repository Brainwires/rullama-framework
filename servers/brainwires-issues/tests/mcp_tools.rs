//! Integration tests for the MCP tools exposed by `brainwires-issues`.
//!
//! Each test spins up an in-process `IssuesMcpServer` backed by a per-test
//! `TempDir` over a `tokio::io::duplex` pair, connects an rmcp client, and
//! drives tools through the MCP protocol exactly the way an external AI tool
//! would. Mirrors the pattern in `brainwires-brain-server/tests/mcp_tools.rs`.

use anyhow::{Context, Result};
use brainwires_issues::mcp_server::IssuesMcpServer;
use rmcp::{ServiceExt, model::CallToolRequestParams, service::RunningService};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::DuplexStream;

/// Spin up a fresh `IssuesMcpServer` backed by `temp` over an in-memory duplex
/// and return a connected rmcp client. The server task is spawned on the
/// current runtime; both the returned `TempDir` and client must stay alive
/// for the duration of the test.
async fn start_server(temp: &TempDir) -> Result<RunningService<rmcp::RoleClient, ()>> {
    let server = IssuesMcpServer::with_data_dir(temp.path())
        .await
        .context("failed to build IssuesMcpServer")?;

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

/// Call a tool and deserialize its first text-content chunk as JSON. Every
/// `brainwires-issues` tool returns a pretty-printed JSON string.
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

/// Call a tool that is expected to error. Returns the raw `CallToolResult` so
/// the test can inspect `is_error` and the error text. MCP tool errors from
/// rmcp surface as `Ok(CallToolResult { is_error: Some(true), ... })`.
async fn call_tool_raw(
    client: &RunningService<rmcp::RoleClient, ()>,
    name: &'static str,
    args: Value,
) -> Result<rmcp::model::CallToolResult> {
    let params = CallToolRequestParams::new(name).with_arguments(
        args.as_object()
            .cloned()
            .context("arguments must be a JSON object")?,
    );
    client.call_tool(params).await.context("call_tool failed")
}

// ── 1. create_issue → get_issue round-trip ───────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_and_get_roundtrip() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    let created = call_tool_json(
        &client,
        "create_issue",
        json!({
            "title": "Round-trip integration test",
            "description": "Ensure create -> get returns identical fields.",
        }),
    )
    .await?;

    let id = created["id"]
        .as_str()
        .context("create response missing id")?
        .to_string();
    assert!(!id.is_empty(), "created issue should have a non-empty id");
    assert_eq!(
        created["title"].as_str(),
        Some("Round-trip integration test")
    );
    assert_eq!(
        created["description"].as_str(),
        Some("Ensure create -> get returns identical fields.")
    );

    let got = call_tool_json(&client, "get_issue", json!({ "id": id })).await?;
    assert_eq!(got["id"].as_str(), Some(id.as_str()));
    assert_eq!(got["title"].as_str(), Some("Round-trip integration test"));
    assert_eq!(
        got["description"].as_str(),
        Some("Ensure create -> get returns identical fields.")
    );

    Ok(())
}

// ── 2. update_issue then close_issue flow ────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_and_close_flow() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    let created = call_tool_json(
        &client,
        "create_issue",
        json!({ "title": "Lifecycle test" }),
    )
    .await?;
    let id = created["id"].as_str().context("no id")?.to_string();

    // Move to in_progress via update_issue.
    let updated = call_tool_json(
        &client,
        "update_issue",
        json!({ "id": id, "status": "in_progress" }),
    )
    .await?;
    assert_eq!(
        updated["status"].as_str(),
        Some("in_progress"),
        "status should be in_progress after update"
    );

    // Close (default resolution = done).
    let closed = call_tool_json(&client, "close_issue", json!({ "id": id })).await?;
    assert_eq!(closed["status"].as_str(), Some("done"));
    assert!(
        closed["closed_at"].as_i64().is_some(),
        "closed_at should be set after close_issue"
    );

    // Confirm via get_issue.
    let got = call_tool_json(&client, "get_issue", json!({ "id": id })).await?;
    assert_eq!(got["status"].as_str(), Some("done"));

    Ok(())
}

// ── 3. add_comment / list_comments / delete_comment CRUD ─────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn comments_crud() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    let issue = call_tool_json(
        &client,
        "create_issue",
        json!({ "title": "Issue with comments" }),
    )
    .await?;
    let issue_id = issue["id"].as_str().context("no issue id")?.to_string();

    let c1 = call_tool_json(
        &client,
        "add_comment",
        json!({ "issue_id": issue_id, "body": "first comment" }),
    )
    .await?;
    let c1_id = c1["id"].as_str().context("no c1 id")?.to_string();

    // Sleep briefly so the second comment has a strictly-greater created_at
    // (seconds resolution) — list_for_issue sorts oldest-first by created_at.
    tokio::time::sleep(std::time::Duration::from_millis(1_100)).await;

    let c2 = call_tool_json(
        &client,
        "add_comment",
        json!({ "issue_id": issue_id, "body": "second comment" }),
    )
    .await?;
    let c2_id = c2["id"].as_str().context("no c2 id")?.to_string();

    let listed = call_tool_json(&client, "list_comments", json!({ "issue_id": issue_id })).await?;
    let comments = listed["comments"]
        .as_array()
        .context("comments not an array")?;
    assert_eq!(listed["count"].as_u64(), Some(2));
    assert_eq!(comments.len(), 2);
    // Oldest-first ordering: first added should be at index 0.
    assert_eq!(comments[0]["id"].as_str(), Some(c1_id.as_str()));
    assert_eq!(comments[1]["id"].as_str(), Some(c2_id.as_str()));

    // Delete the first comment.
    let deleted = call_tool_json(&client, "delete_comment", json!({ "id": c1_id })).await?;
    assert_eq!(deleted["deleted"].as_str(), Some(c1_id.as_str()));

    let after = call_tool_json(&client, "list_comments", json!({ "issue_id": issue_id })).await?;
    let remaining = after["comments"]
        .as_array()
        .context("remaining not an array")?;
    assert_eq!(after["count"].as_u64(), Some(1));
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0]["id"].as_str(), Some(c2_id.as_str()));

    Ok(())
}

// ── 4. list_issues offset-based pagination ───────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_pagination() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    // `list_issues` sorts by `updated_at` DESC (seconds resolution) in memory
    // AFTER pulling `offset+limit+1` records from the backend. If the backend
    // fetch cap is smaller than the total set, records with a higher
    // `updated_at` can be cut off before the sort, causing pages to overlap.
    //
    // To keep the test deterministic we:
    //   - Sleep ~1.1 s between creates so every issue has a distinct
    //     `updated_at` (the field is seconds-granularity).
    //   - Size N and LIMIT so `offset+LIMIT+1 >= N` on both pages (N=3 with
    //     LIMIT=2 means page 1 fetches 3 = all, page 2 fetches 5 ≥ all).
    const N: usize = 3;
    const LIMIT: u64 = 2;
    let mut created_ids = Vec::new();
    for i in 0..N {
        if i > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(1_100)).await;
        }
        let created = call_tool_json(
            &client,
            "create_issue",
            json!({ "title": format!("pager issue {i}") }),
        )
        .await?;
        created_ids.push(created["id"].as_str().context("no id")?.to_string());
    }

    // Page 1: limit = 2, no offset. Should contain the two newest issues
    // (last two in created_ids).
    let page1 = call_tool_json(&client, "list_issues", json!({ "limit": LIMIT })).await?;
    let p1_issues = page1["issues"]
        .as_array()
        .context("page1 issues not an array")?;
    assert_eq!(
        p1_issues.len(),
        LIMIT as usize,
        "page 1 should contain {LIMIT} issues"
    );
    assert_eq!(page1["count"].as_u64(), Some(LIMIT));
    let next_offset = page1["next_offset"]
        .as_u64()
        .context("next_offset should be present when more records exist")?;
    assert_eq!(next_offset, LIMIT);

    // Page 2: use the reported next_offset — should contain the remaining
    // (oldest) issue(s) and report no further pages.
    let page2 = call_tool_json(
        &client,
        "list_issues",
        json!({ "limit": LIMIT, "offset": next_offset }),
    )
    .await?;
    let p2_issues = page2["issues"]
        .as_array()
        .context("page2 issues not an array")?;
    let remaining = N - LIMIT as usize;
    assert_eq!(
        p2_issues.len(),
        remaining,
        "page 2 should contain the remaining {remaining} issue(s)"
    );
    assert_eq!(page2["count"].as_u64(), Some(remaining as u64));
    assert!(
        page2["next_offset"].is_null(),
        "next_offset should be null when there are no more records, got {}",
        page2["next_offset"]
    );

    // No overlap between pages, and together they cover the full created set.
    let page1_ids: std::collections::HashSet<String> = p1_issues
        .iter()
        .filter_map(|i| i["id"].as_str().map(String::from))
        .collect();
    let page2_ids: std::collections::HashSet<String> = p2_issues
        .iter()
        .filter_map(|i| i["id"].as_str().map(String::from))
        .collect();

    assert!(
        page1_ids.is_disjoint(&page2_ids),
        "pages 1 and 2 must not share issue ids"
    );
    let union: std::collections::HashSet<String> = page1_ids.union(&page2_ids).cloned().collect();
    let all_created: std::collections::HashSet<String> = created_ids.into_iter().collect();
    assert_eq!(
        union, all_created,
        "union of both pages should equal the full created set"
    );

    Ok(())
}

// ── 5. delete_issue removes the issue ────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_removes_issue() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    let created =
        call_tool_json(&client, "create_issue", json!({ "title": "to be deleted" })).await?;
    let id = created["id"].as_str().context("no id")?.to_string();

    let del = call_tool_json(&client, "delete_issue", json!({ "id": id })).await?;
    assert_eq!(del["deleted"].as_str(), Some(id.as_str()));

    // Tool methods return `Result<String, String>`, so the not-found path
    // surfaces as an MCP tool error rather than a JSON success payload.
    let raw = call_tool_raw(&client, "get_issue", json!({ "id": id })).await?;
    assert_eq!(
        raw.is_error,
        Some(true),
        "get_issue on a deleted id should be an error, got {:?}",
        raw.content
    );
    let err_text = raw
        .content
        .first()
        .and_then(|c| c.raw.as_text())
        .map(|t| t.text.clone())
        .unwrap_or_default();
    assert!(
        err_text.to_lowercase().contains("not found"),
        "expected a not-found error message, got: {err_text}"
    );

    Ok(())
}

// ── 6. search_issues finds a matching issue ──────────────────────────────
//
// Uses three issues with distinct titles and queries for a unique word. The
// BM25 index lives inside the per-test TempDir — no embeddings required, so
// this is cheap and deterministic.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn search_finds_matching_issue() -> Result<()> {
    let temp = TempDir::new()?;
    let client = start_server(&temp).await?;

    call_tool_json(
        &client,
        "create_issue",
        json!({ "title": "Login page freezes on Firefox" }),
    )
    .await?;
    let target = call_tool_json(
        &client,
        "create_issue",
        json!({
            "title": "Dashboard widget misaligned",
            "description": "The analytics brainwires-search-needle widget overlaps the header.",
        }),
    )
    .await?;
    let target_id = target["id"].as_str().context("no target id")?.to_string();
    call_tool_json(
        &client,
        "create_issue",
        json!({ "title": "Settings page missing save button" }),
    )
    .await?;

    let resp = call_tool_json(
        &client,
        "search_issues",
        json!({ "query": "brainwires-search-needle", "limit": 10 }),
    )
    .await?;

    let results = resp["issues"]
        .as_array()
        .context("search issues not an array")?;
    assert!(
        !results.is_empty(),
        "search should return at least one match for the unique token"
    );
    let matched = results
        .iter()
        .any(|i| i["id"].as_str() == Some(target_id.as_str()));
    assert!(
        matched,
        "expected the target issue {target_id} in search results, got {results:?}"
    );

    Ok(())
}
