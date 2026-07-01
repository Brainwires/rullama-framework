//! Tier-B adversarial cases for `rullama_sandbox::SandboxPolicy`.
//!
//! Invariants asserted here:
//! - `validate_mount` REJECTS sources outside the workspace + allowed-roots set.
//! - `validate_mount` REJECTS any path containing parent-dir traversal.
//! - The default `SandboxPolicy` denies network egress (`NetworkPolicy::None`).
//! - The default `SandboxPolicy` mounts the container root filesystem read-only.
//! - The default `SandboxPolicy` whitelists ZERO host paths (mounts must be
//!   explicitly opted in, never permitted by accident).
//!
//! Ground truth: `crates/rullama-sandbox/src/policy.rs`.

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;
use rullama_eval::{EvaluationCase, TrialResult};
use rullama_sandbox::{Mount, NetworkPolicy, SandboxPolicy};

use crate::registry::SecurityCase;

// ── sec.sandbox.mount_whitelist ─────────────────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.sandbox.mount_whitelist",
        crate_name: "rullama-sandbox",
        invariant: "validate_mount rejects sources outside allowed roots",
        factory: || Box::new(MountWhitelistCase),
    }
}

struct MountWhitelistCase;

#[async_trait]
impl EvaluationCase for MountWhitelistCase {
    fn name(&self) -> &str {
        "sec.sandbox.mount_whitelist"
    }
    fn category(&self) -> &str {
        "security.sandbox"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let policy = SandboxPolicy {
            allowed_mount_sources: vec![PathBuf::from("/workspace")],
            ..SandboxPolicy::default()
        };
        for adversarial in &[
            "/etc/passwd",
            "/etc/shadow",
            "/root/.ssh/authorized_keys",
            "/var/run/docker.sock",
        ] {
            let m = Mount {
                source: PathBuf::from(adversarial),
                target: PathBuf::from("/mnt/x"),
                read_only: true,
            };
            if policy.validate_mount(&m).is_ok() {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!("sandbox accepted dangerous mount: {adversarial}"),
                ));
            }
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.sandbox.parent_dir_traversal ────────────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.sandbox.parent_dir_traversal",
        crate_name: "rullama-sandbox",
        invariant: "validate_mount rejects any path with `..` components",
        factory: || Box::new(ParentDirTraversalCase),
    }
}

struct ParentDirTraversalCase;

#[async_trait]
impl EvaluationCase for ParentDirTraversalCase {
    fn name(&self) -> &str {
        "sec.sandbox.parent_dir_traversal"
    }
    fn category(&self) -> &str {
        "security.sandbox"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let policy = SandboxPolicy {
            allowed_mount_sources: vec![PathBuf::from("/workspace")],
            ..SandboxPolicy::default()
        };
        // Each path lexically starts under /workspace but escapes via `..`.
        // The policy must reject BEFORE doing the prefix check, otherwise
        // a lexical-only is_within() lets these through.
        for adversarial in &[
            "/workspace/../etc/passwd",
            "/workspace/sub/../../etc",
            "/workspace/./../etc",
        ] {
            let m = Mount {
                source: PathBuf::from(adversarial),
                target: PathBuf::from("/mnt/x"),
                read_only: true,
            };
            if policy.validate_mount(&m).is_ok() {
                return Ok(TrialResult::failure(
                    0,
                    0,
                    format!("sandbox accepted parent-dir traversal: {adversarial}"),
                ));
            }
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.sandbox.network_default_none ────────────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.sandbox.network_default_none",
        crate_name: "rullama-sandbox",
        invariant: "Default SandboxPolicy denies all network egress",
        factory: || Box::new(NetworkDefaultNoneCase),
    }
}

struct NetworkDefaultNoneCase;

#[async_trait]
impl EvaluationCase for NetworkDefaultNoneCase {
    fn name(&self) -> &str {
        "sec.sandbox.network_default_none"
    }
    fn category(&self) -> &str {
        "security.sandbox"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let policy = SandboxPolicy::default();
        // A future refactor that flips the default to `Full` (or accidentally
        // changes to `Limited(vec![])`) is exactly the silent-regression this
        // case exists to catch.
        if policy.network != NetworkPolicy::None {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "default SandboxPolicy.network is {:?}, expected NetworkPolicy::None",
                    policy.network
                ),
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.sandbox.default_safe_invariants ─────────────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.sandbox.default_safe_invariants",
        crate_name: "rullama-sandbox",
        invariant: "Default SandboxPolicy is safe-by-default: read-only rootfs, no mount whitelist, finite resource caps",
        factory: || Box::new(DefaultSafeInvariantsCase),
    }
}

struct DefaultSafeInvariantsCase;

#[async_trait]
impl EvaluationCase for DefaultSafeInvariantsCase {
    fn name(&self) -> &str {
        "sec.sandbox.default_safe_invariants"
    }
    fn category(&self) -> &str {
        "security.sandbox"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let p = SandboxPolicy::default();
        if !p.read_only_rootfs {
            return Ok(TrialResult::failure(
                0,
                0,
                "default SandboxPolicy.read_only_rootfs is false",
            ));
        }
        if !p.allowed_mount_sources.is_empty() {
            return Ok(TrialResult::failure(
                0,
                0,
                format!(
                    "default allowed_mount_sources is non-empty: {:?} — mounts must be opt-in",
                    p.allowed_mount_sources
                ),
            ));
        }
        if p.workspace_mount.is_some() {
            return Ok(TrialResult::failure(
                0,
                0,
                "default workspace_mount must be None — workspace must be opt-in",
            ));
        }
        if p.memory_limit_mb.is_none() || p.cpu_limit.is_none() || p.pid_limit.is_none() {
            return Ok(TrialResult::failure(
                0,
                0,
                "default resource caps (cpu/memory/pid) must all be Some — no unbounded sandbox",
            ));
        }
        if p.proxy_container_name.is_some() {
            return Ok(TrialResult::failure(
                0,
                0,
                "default proxy_container_name must be None — shared long-lived proxies are opt-in",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}
