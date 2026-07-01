//! Integration tests for the SIGINT -> SIGTERM -> SIGKILL escalation ladder
//! in [`reload_daemon::reload::kill_process`].
//!
//! Each test spawns a real child process whose signal-handling behaviour
//! exercises a specific rung of the escalation ladder:
//!
//! * default handlers          -> SIGINT suffices
//! * `trap "" INT; exec sleep` -> SIGINT ignored, SIGTERM kills
//! * `trap "" INT TERM; exec`  -> both trapped, only SIGKILL succeeds
//! * bogus pid                 -> function returns cleanly (already-gone path)
//!
//! Note the `exec` in the trap scripts: POSIX guarantees that signal
//! dispositions set to `SIG_IGN` are preserved across `execve`. Without the
//! `exec`, bash would be the parent of `sleep`, and `bash`'s `wait`-builtin
//! semantics plus signal-trap bookkeeping can let SIGINT terminate the
//! pipeline even though the trap is installed. Using `exec` means the pid
//! we targeted becomes `sleep` directly, with SIGINT (and optionally SIGTERM)
//! carried over as `SIG_IGN` — the canonical "stubborn process" scenario.
//!
//! The whole file is `#[cfg(unix)]` because the daemon is Unix-only (the
//! escalation APIs are bare `libc::kill` calls). Tests are intentionally
//! serial — the kernel pid space is shared and parallel runs would make the
//! elapsed-time assertions flaky.

#![cfg(unix)]

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use reload_daemon::config::ClientStrategy;
use reload_daemon::reload::kill_process;

/// Poll helper: returns true while the process is still alive. Uses
/// `kill(pid, 0)` which is the canonical "does this pid exist?" probe on
/// Unix — it sends no signal, just performs permission + existence checks.
fn is_alive(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}

/// Spawn a background reaper thread that blocks on `child.wait()`.
///
/// This is crucial for the test to observe realistic `kill(pid, 0)`
/// semantics. When the test binary itself is the direct parent of the
/// target, an exited child lingers as a zombie — and `kill(pid, 0)` still
/// returns 0 for zombies — until we reap it. That would defeat
/// `process_alive`-based polling inside `kill_process` and force every test
/// to escalate all the way to SIGKILL.
///
/// In production the daemon is not the parent of the target (Claude Code,
/// Cursor, …); init/PID 1 (or the user's shell) reaps those, so the zombie
/// window doesn't exist. The reaper thread approximates that: the moment
/// the child exits, `wait` returns and the pid leaves the process table.
///
/// Returns a `JoinHandle` — the caller joins it at the end of the test to
/// confirm the reaper completed (and to observe the exit status for logging
/// if ever needed).
fn spawn_reaper(mut child: std::process::Child) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let _ = child.wait();
    })
}

/// Wait (bounded) for a pid to leave the process table. Used after
/// `kill_process` returns to give the reaper thread a beat to reap.
fn wait_for_pid_gone(pid: i32, deadline: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < deadline {
        if !is_alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    !is_alive(pid)
}

/// Build a `ClientStrategy` with the given per-step timeouts.
/// `process_name` and `restart_args_transform` are irrelevant to
/// `kill_process` — only `kill_signals` / `kill_timeouts_ms` drive the ladder.
fn strategy_with_timeouts(sigint_ms: u64, sigterm_ms: u64) -> ClientStrategy {
    ClientStrategy {
        process_name: "test-target".into(),
        kill_signals: vec!["SIGINT".into(), "SIGTERM".into(), "SIGKILL".into()],
        // SIGKILL uses timeout 0 = "fire and forget", matching the production
        // config. We tighten SIGINT/SIGTERM so the test suite stays fast.
        kill_timeouts_ms: vec![sigint_ms, sigterm_ms, 0],
        restart_args_transform: None,
    }
}

/// Overall safety net so a hung `kill_process` never deadlocks the test
/// process. Generous enough to cover the sum of per-step timeouts plus
/// scheduling jitter.
const SAFETY_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Test 1: SIGINT alone suffices when the child uses default handlers.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn sigint_suffices_if_child_exits_on_interrupt() {
    // `sleep 30` has default signal handlers, so SIGINT terminates it.
    let child = Command::new("sleep")
        .arg("30")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn sleep");
    let pid = child.id() as i32;
    let reaper = spawn_reaper(child);
    assert!(is_alive(pid), "child should be alive before kill_process");

    let strategy = strategy_with_timeouts(800, 800);
    let start = Instant::now();
    let result = tokio::time::timeout(SAFETY_TIMEOUT, kill_process(pid, &strategy))
        .await
        .expect("kill_process hung past safety timeout");
    let elapsed = start.elapsed();
    result.expect("kill_process should succeed on default-handler child");

    assert!(
        wait_for_pid_gone(pid, Duration::from_secs(1)),
        "pid {pid} should be gone after SIGINT"
    );

    // Sanity bound: we expect to exit well inside the SIGINT window, not the
    // full SIGINT + SIGTERM window. Allow headroom for polling granularity
    // (the escalation loop polls every 100 ms).
    assert!(
        elapsed < Duration::from_millis(800),
        "should have exited on SIGINT alone, took {elapsed:?}"
    );

    reaper.join().expect("reaper thread panicked");
}

// ---------------------------------------------------------------------------
// Test 2: SIGTERM escalation when SIGINT is trapped away.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn sigterm_escalation_when_sigint_ignored() {
    // `trap "" INT` installs an empty SIGINT handler that explicitly ignores
    // the signal, but leaves SIGTERM on its default (terminate) disposition.
    let child = Command::new("bash")
        .arg("-c")
        .arg(r#"trap "" INT; exec sleep 30"#)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn bash trap");
    let pid = child.id() as i32;
    let reaper = spawn_reaper(child);
    assert!(is_alive(pid), "child should be alive before kill_process");

    // Give bash a moment to finish `trap` + `exec sleep` so the SIG_IGN
    // disposition is actually in place before we start signalling. Without
    // this, there's a tiny race where SIGINT arrives while bash is still
    // parsing the script and the signal kills it outright.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let sigint_ms = 400;
    let sigterm_ms = 800;
    let strategy = strategy_with_timeouts(sigint_ms, sigterm_ms);
    let start = Instant::now();
    let result = tokio::time::timeout(SAFETY_TIMEOUT, kill_process(pid, &strategy))
        .await
        .expect("kill_process hung past safety timeout");
    let elapsed = start.elapsed();
    result.expect("kill_process should succeed via SIGTERM escalation");

    assert!(
        wait_for_pid_gone(pid, Duration::from_secs(1)),
        "pid {pid} should be gone after SIGTERM"
    );

    // We must have waited out the SIGINT window before escalating.
    assert!(
        elapsed >= Duration::from_millis(sigint_ms),
        "should have waited the full SIGINT window before escalating, took {elapsed:?}"
    );
    // But SIGTERM should land near-instantly once delivered — no need to
    // burn the SIGTERM window.
    assert!(
        elapsed < Duration::from_millis(sigint_ms + sigterm_ms) + Duration::from_millis(500),
        "SIGTERM should have killed near-instantly, took {elapsed:?}"
    );

    reaper.join().expect("reaper thread panicked");
}

// ---------------------------------------------------------------------------
// Test 3: SIGKILL is the last resort when both SIGINT and SIGTERM are trapped.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn sigkill_final_escalation_when_both_ignored() {
    // SIGKILL cannot be trapped or ignored — this is the backstop the daemon
    // relies on for any well-behaved-but-stubborn target.
    let child = Command::new("bash")
        .arg("-c")
        .arg(r#"trap "" INT TERM; exec sleep 30"#)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn bash double-trap");
    let pid = child.id() as i32;
    let reaper = spawn_reaper(child);
    assert!(is_alive(pid), "child should be alive before kill_process");

    // Same race mitigation as the SIGTERM-escalation test: let bash exec
    // into `sleep` before we start the ladder.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let sigint_ms = 300;
    let sigterm_ms = 300;
    let strategy = strategy_with_timeouts(sigint_ms, sigterm_ms);
    let start = Instant::now();
    let result = tokio::time::timeout(SAFETY_TIMEOUT, kill_process(pid, &strategy))
        .await
        .expect("kill_process hung past safety timeout");
    let elapsed = start.elapsed();
    result.expect("kill_process should succeed via SIGKILL escalation");

    // SIGKILL is fire-and-forget in the production code path (timeout 0), so
    // give the reaper thread a beat before probing existence.
    assert!(
        wait_for_pid_gone(pid, Duration::from_secs(1)),
        "pid {pid} should be gone after SIGKILL"
    );

    // Must have burned through both the SIGINT and SIGTERM windows before
    // reaching SIGKILL.
    assert!(
        elapsed >= Duration::from_millis(sigint_ms + sigterm_ms),
        "should have waited both SIGINT and SIGTERM windows, took {elapsed:?}"
    );

    reaper.join().expect("reaper thread panicked");
}

// ---------------------------------------------------------------------------
// Test 4: kill_process is tolerant of a non-existent pid (ESRCH).
// ---------------------------------------------------------------------------
#[tokio::test]
async fn nonexistent_pid_returns_cleanly() {
    // i32::MAX is well beyond any realistic pid_max on Linux / macOS, so
    // the first `kill(pid, SIGINT)` will return ESRCH and the function
    // should short-circuit to Ok(()).
    let pid = i32::MAX;
    assert!(!is_alive(pid), "sanity: i32::MAX pid must not exist");

    let strategy = strategy_with_timeouts(400, 400);
    let result = tokio::time::timeout(SAFETY_TIMEOUT, kill_process(pid, &strategy))
        .await
        .expect("kill_process hung past safety timeout");

    // Current behaviour: ESRCH on the first signal is treated as "already
    // gone" and returns Ok(()). If that contract ever changes, this test
    // should be updated deliberately.
    result.expect("kill_process on a non-existent pid should succeed cleanly");
}
