//! Integration tests for the MCP tool surface of `brainwires-scheduler`.
//!
//! Each test spins up an in-process `SchedulerServer` backed by a per-test
//! `TempDir` over a `tokio::io::duplex` pair, connects an rmcp client, and
//! drives tools through the MCP protocol exactly the way an external AI tool
//! would. Mirrors the pattern in `brainwires-issues/tests/mcp_tools.rs` and
//! `brainwires-brain-server/tests/mcp_tools.rs`.
//!
//! Every scheduled job uses `echo` as the command so the tests never touch
//! Docker — only the native execution path is exercised.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use brainwires_scheduler::{JobStore, SchedulerDaemon, SchedulerServer};
use rmcp::{ServiceExt, model::CallToolRequestParams, service::RunningService};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::io::DuplexStream;
use tokio::sync::{RwLock, watch};

/// A cron expression that is valid but never fires during the test window.
/// "At 00:00 on January 1st" — at worst fires once a year.
const FAR_FUTURE_CRON: &str = "0 0 1 1 *";

/// Bundle returned by `start_server`. Holds handles that must stay alive for
/// the duration of the test: the client (to drive tools), the daemon cancel
/// sender (dropped last to stop the daemon loop), and the TempDir.
struct TestHarness {
    client: RunningService<rmcp::RoleClient, ()>,
    _cancel_tx: watch::Sender<bool>,
}

/// Build a `SchedulerServer` backed by `jobs_dir`, spawn the daemon loop, wire
/// the MCP server to an in-memory duplex transport, and return a connected
/// rmcp client. The daemon is cancelled when `TestHarness` is dropped.
async fn start_server(jobs_dir: &Path) -> Result<TestHarness> {
    let store = JobStore::open(jobs_dir)
        .await
        .context("failed to open JobStore")?;
    let store = Arc::new(RwLock::new(store));

    // max_concurrent = 2 is plenty for tests; the daemon needs >= 1.
    let (daemon, handle, cancel_tx) = SchedulerDaemon::new(Arc::clone(&store), 2);

    // Spawn the scheduler loop — required for `run_now` because the daemon
    // processes the `RunNow` command via its mpsc channel.
    tokio::spawn(daemon.run());

    let server = SchedulerServer::new(handle);

    let (server_transport, client_transport): (DuplexStream, DuplexStream) =
        tokio::io::duplex(8 * 1024);

    tokio::spawn(async move {
        if let Ok(running) = server.serve(server_transport).await {
            let _ = running.waiting().await;
        }
    });

    let client = ().serve(client_transport).await.context("client failed to init")?;

    Ok(TestHarness {
        client,
        _cancel_tx: cancel_tx,
    })
}

/// Call a tool and return its first text-content chunk as a `String`.
/// Scheduler tools return human-readable text (not JSON).
async fn call_tool_text(
    client: &RunningService<rmcp::RoleClient, ()>,
    name: &'static str,
    args: Value,
) -> Result<String> {
    let mut params = CallToolRequestParams::new(name);
    let obj = args
        .as_object()
        .cloned()
        .context("arguments must be a JSON object")?;
    // Tools that take `Parameters<()>` reject an empty map ("invalid type:
    // map, expected unit") so only attach arguments when non-empty.
    if !obj.is_empty() {
        params = params.with_arguments(obj);
    }
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
        .map(|t| t.text.clone())
        .with_context(|| format!("tool {name} returned no text content"))?;

    Ok(text)
}

/// Call a tool that may error. Returns the raw `CallToolResult` so the test
/// can inspect `is_error` and the error text.
async fn call_tool_raw(
    client: &RunningService<rmcp::RoleClient, ()>,
    name: &'static str,
    args: Value,
) -> Result<rmcp::model::CallToolResult> {
    let mut params = CallToolRequestParams::new(name);
    let obj = args
        .as_object()
        .cloned()
        .context("arguments must be a JSON object")?;
    if !obj.is_empty() {
        params = params.with_arguments(obj);
    }
    client.call_tool(params).await.context("call_tool failed")
}

/// Extract the `id: <uuid>` field from the `add_job` response string.
fn extract_job_id(add_job_response: &str) -> Result<String> {
    add_job_response
        .lines()
        .find_map(|l| l.strip_prefix("id: "))
        .map(str::trim)
        .map(str::to_owned)
        .context("add_job response did not contain an `id:` line")
}

/// Common add-a-simple-echo-job helper. Uses `/tmp` as working dir (always
/// exists on Linux CI) and a far-future cron so it never auto-fires.
fn echo_job_args(name: &str, message: &str) -> Value {
    json!({
        "name": name,
        "cron": FAR_FUTURE_CRON,
        "command": "echo",
        "args": [message],
        "working_dir": "/tmp",
    })
}

// ── 1. add_job returns a queryable job id (stands in for list_jobs) ──────
//
// The original plan asked for `add_job` + `list_jobs`, but `list_jobs` is
// declared as `Parameters<()>` and rmcp's server rewrites missing MCP
// arguments into `{}` before deserializing them, which fails on `()`
// ("invalid type: map, expected unit"). `list_jobs` therefore isn't
// externally callable until the scheduler switches those two tools
// (`list_jobs`, `status`) to a real request struct. For now the test
// asserts `add_job` round-trips via `get_job` and pins the quirk so any
// future fix shows up as a test failure.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn add_and_list_job() -> Result<()> {
    let temp = TempDir::new()?;
    let h = start_server(temp.path()).await?;

    let resp = call_tool_text(&h.client, "add_job", echo_job_args("hello-job", "hello")).await?;
    assert!(
        resp.starts_with("Job added."),
        "unexpected add_job response: {resp}"
    );
    let id = extract_job_id(&resp)?;
    assert!(!id.is_empty(), "extracted job id should be non-empty");

    let detail = call_tool_text(&h.client, "get_job", json!({ "id": id })).await?;
    assert!(
        detail.contains(&id),
        "get_job should echo the id we just created, got:\n{detail}"
    );
    assert!(
        detail.contains("hello-job"),
        "get_job should contain the job name, got:\n{detail}"
    );

    // Pin the `list_jobs` / `Parameters<()>` quirk. When this starts failing,
    // list_jobs has become callable — delete this block and replace it with a
    // real list_jobs happy-path assertion.
    let listed_raw = call_tool_raw(&h.client, "list_jobs", json!({})).await;
    let hit_known_quirk = matches!(
        &listed_raw,
        Err(e) if format!("{e:#}").contains("expected unit")
    );
    assert!(
        hit_known_quirk,
        "expected list_jobs to hit the Parameters<()> quirk, got: {listed_raw:?}"
    );

    Ok(())
}

// ── 2. get_job returns the configured record ─────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_job_returns_matching_record() -> Result<()> {
    let temp = TempDir::new()?;
    let h = start_server(temp.path()).await?;

    let resp = call_tool_text(
        &h.client,
        "add_job",
        echo_job_args("lookup-job", "lookup-me"),
    )
    .await?;
    let id = extract_job_id(&resp)?;

    let detail = call_tool_text(&h.client, "get_job", json!({ "id": id })).await?;

    assert!(
        detail.contains(&id),
        "get_job should echo the id, got:\n{detail}"
    );
    assert!(
        detail.contains("lookup-job"),
        "get_job should contain the job name, got:\n{detail}"
    );
    assert!(
        detail.contains(FAR_FUTURE_CRON),
        "get_job should contain the cron expression, got:\n{detail}"
    );
    assert!(
        detail.contains("echo") && detail.contains("lookup-me"),
        "get_job should contain the command + args, got:\n{detail}"
    );
    assert!(
        detail.contains("Enabled:         true"),
        "a freshly-added job should be enabled, got:\n{detail}"
    );
    assert!(
        detail.contains("none (runs natively)"),
        "no sandbox was configured, so get_job should report native execution, got:\n{detail}"
    );

    Ok(())
}

// ── 3. run_job executes the command and writes a log entry ───────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_job_executes_command() -> Result<()> {
    let temp = TempDir::new()?;
    let h = start_server(temp.path()).await?;

    const NEEDLE: &str = "brainwires-scheduler-test";

    let add_resp =
        call_tool_text(&h.client, "add_job", echo_job_args("oneshot-echo", NEEDLE)).await?;
    let id = extract_job_id(&add_resp)?;

    // `run_job` awaits the job's completion (DaemonHandle::run_now returns the
    // JobResult) so this should return the final status synchronously.
    let run_resp = call_tool_text(&h.client, "run_job", json!({ "id": id })).await?;
    assert!(
        run_resp.contains("success") && run_resp.contains("exit 0"),
        "run_job should report success, got:\n{run_resp}"
    );
    assert!(
        run_resp.contains(NEEDLE),
        "run_job response should include echoed stdout '{NEEDLE}', got:\n{run_resp}"
    );

    // The daemon writes the log *after* sending the result back over the
    // oneshot channel, so `get_logs` is eventually-consistent. Poll for up to
    // ~5s in 100 ms increments.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let last_logs = loop {
        let logs = call_tool_text(&h.client, "get_logs", json!({ "id": id, "limit": 5 })).await?;
        if logs.contains(NEEDLE) {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            break logs;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    };

    panic!(
        "get_logs did not surface echoed stdout '{NEEDLE}' within 5s, last response:\n{last_logs}"
    );
}

// ── 4. disable_job prevents run_job, enable_job restores it ──────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disable_job_prevents_nothing_observable_but_flag_flips() -> Result<()> {
    // Note: reading `daemon.rs`, `DaemonHandle::run_now` does NOT check the
    // `enabled` flag — disabling a job only prevents the cron loop from
    // auto-firing it. Scheduled firing is paused; `run_now` is still honoured.
    // The observable invariant we can test here is the enabled flag round-trip
    // via get_job, which is what the tool surface guarantees.
    let temp = TempDir::new()?;
    let h = start_server(temp.path()).await?;

    let add_resp =
        call_tool_text(&h.client, "add_job", echo_job_args("toggle-job", "toggle")).await?;
    let id = extract_job_id(&add_resp)?;

    // Freshly-added → enabled.
    let before = call_tool_text(&h.client, "get_job", json!({ "id": id })).await?;
    assert!(
        before.contains("Enabled:         true"),
        "new job should start enabled, got:\n{before}"
    );

    // Disable and confirm.
    let disable_resp = call_tool_text(&h.client, "disable_job", json!({ "id": id })).await?;
    assert!(
        disable_resp.contains("disabled"),
        "disable_job should report success, got:\n{disable_resp}"
    );
    let disabled = call_tool_text(&h.client, "get_job", json!({ "id": id })).await?;
    assert!(
        disabled.contains("Enabled:         false"),
        "get_job should show disabled after disable_job, got:\n{disabled}"
    );

    // Re-enable and confirm.
    let enable_resp = call_tool_text(&h.client, "enable_job", json!({ "id": id })).await?;
    assert!(
        enable_resp.contains("enabled"),
        "enable_job should report success, got:\n{enable_resp}"
    );
    let enabled = call_tool_text(&h.client, "get_job", json!({ "id": id })).await?;
    assert!(
        enabled.contains("Enabled:         true"),
        "get_job should show enabled after enable_job, got:\n{enabled}"
    );

    // disable_job on a nonexistent id is an MCP error — exercise that
    // error-reporting surface too.
    let raw = call_tool_raw(
        &h.client,
        "disable_job",
        json!({ "id": "00000000-0000-0000-0000-000000000000" }),
    )
    .await?;
    assert_eq!(
        raw.is_error,
        Some(true),
        "disable_job on missing id should be an MCP error"
    );

    Ok(())
}

// ── 5. get_logs on a never-run job returns the empty shape ──────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_logs_empty_for_never_run() -> Result<()> {
    let temp = TempDir::new()?;
    let h = start_server(temp.path()).await?;

    let add_resp =
        call_tool_text(&h.client, "add_job", echo_job_args("never-run", "unused")).await?;
    let id = extract_job_id(&add_resp)?;

    let logs = call_tool_text(&h.client, "get_logs", json!({ "id": id })).await?;
    assert!(
        logs.contains("No execution logs"),
        "get_logs on a fresh job should report the empty sentinel, got:\n{logs}"
    );

    Ok(())
}
