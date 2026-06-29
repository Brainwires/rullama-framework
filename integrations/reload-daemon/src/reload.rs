use crate::config::{ArgsTransform, ClientStrategy};
use std::process::Command;
use std::time::Duration;

const PROCESS_CHECK_POLL_MS: u64 = 100;

/// Parse a signal name (e.g. "SIGINT") into its numeric value.
#[cfg(unix)]
fn parse_signal(name: &str) -> Option<i32> {
    match name {
        "SIGINT" => Some(libc::SIGINT),
        "SIGTERM" => Some(libc::SIGTERM),
        "SIGKILL" => Some(libc::SIGKILL),
        "SIGHUP" => Some(libc::SIGHUP),
        "SIGUSR1" => Some(libc::SIGUSR1),
        "SIGUSR2" => Some(libc::SIGUSR2),
        _ => None,
    }
}

/// Check whether a process with the given PID is still alive.
#[cfg(unix)]
fn process_alive(pid: i32) -> bool {
    // kill(pid, 0) checks existence without sending a signal.
    unsafe { libc::kill(pid, 0) == 0 }
}

/// Kill a process using the escalating signal strategy from config.
///
/// Iterates through `kill_signals` paired with `kill_timeouts_ms`. For each
/// step it sends the signal and waits up to the timeout for the process to
/// exit. A timeout of 0 means "send and return immediately" (used for
/// SIGKILL where we don't need to poll).
#[cfg(unix)]
pub async fn kill_process(pid: i32, strategy: &ClientStrategy) -> Result<(), String> {
    for (sig_name, &timeout_ms) in strategy
        .kill_signals
        .iter()
        .zip(strategy.kill_timeouts_ms.iter())
    {
        let sig = parse_signal(sig_name).ok_or_else(|| format!("unknown signal: {sig_name}"))?;

        let rc = unsafe { libc::kill(pid, sig) };
        if rc != 0 {
            let errno = std::io::Error::last_os_error();
            // ESRCH = no such process — already dead, success.
            if errno.raw_os_error() == Some(libc::ESRCH) {
                return Ok(());
            }
            return Err(format!("kill({pid}, {sig_name}) failed: {errno}"));
        }

        if timeout_ms == 0 {
            // Fire-and-forget (e.g. SIGKILL).
            return Ok(());
        }

        // Poll for process exit.
        let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            tokio::time::sleep(Duration::from_millis(PROCESS_CHECK_POLL_MS)).await;
            if !process_alive(pid) {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                break; // Escalate to the next signal.
            }
        }
    }

    // After all signals exhausted, check one last time.
    if process_alive(pid) {
        Err(format!(
            "process {pid} still alive after exhausting all signals"
        ))
    } else {
        Ok(())
    }
}

#[cfg(not(unix))]
pub async fn kill_process(_pid: i32, _strategy: &ClientStrategy) -> Result<(), String> {
    Err("kill_process is only supported on Unix".into())
}

/// Build the new argument list for the restarted process.
///
/// - Keeps any flags from `original_args` that appear in `preserve_flags`.
/// - Appends the `replace_trailing` args at the end.
pub fn transform_args(original_args: &[String], transform: &ArgsTransform) -> Vec<String> {
    let mut out = Vec::new();

    let mut i = 0;
    while i < original_args.len() {
        let arg = &original_args[i];
        if transform.preserve_flags.contains(arg) {
            out.push(arg.clone());
        }
        i += 1;
    }

    out.extend(transform.replace_trailing.iter().cloned());
    out
}

/// Spawn a new process and detach it so it outlives this handler.
pub fn spawn_process(program: &str, args: &[String], cwd: &str) -> Result<(), String> {
    let child = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to spawn {program}: {e}"))?;

    // Detach: forget the Child handle so the process is adopted by init/PID 1.
    std::mem::forget(child);
    Ok(())
}
