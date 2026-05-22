//! Tier-B adversarial cases for `brainwires_agent::file_locks::FileLockManager`.
//!
//! The file-lock manager is the framework's mutual-exclusion primitive for
//! concurrent agents touching the same workspace. Bugs here mean two
//! agents can corrupt the same file or deadlock the entire pool.
//!
//! Invariants:
//! - Two distinct agents CAN hold concurrent READ locks on the same file.
//! - A WRITE lock from agent B is REJECTED while agent A holds a READ
//!   lock on the same file.
//! - A WRITE lock from agent B is REJECTED while agent A holds a WRITE
//!   lock on the same file.
//! - The SAME agent re-acquiring a WRITE lock it already holds succeeds
//!   (idempotent — agents must not deadlock against themselves).

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use brainwires_agent::file_locks::{FileLockManager, LockType};
use brainwires_eval::{EvaluationCase, TrialResult};

use crate::registry::SecurityCase;

// ── sec.agent.file_locks.parallel_reads_allowed ────────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.agent.file_locks.parallel_reads_allowed",
        crate_name: "brainwires-agent",
        invariant: "Two distinct agents can concurrently hold READ locks on the same file",
        factory: || Box::new(ParallelReadsAllowedCase),
    }
}

struct ParallelReadsAllowedCase;

#[async_trait]
impl EvaluationCase for ParallelReadsAllowedCase {
    fn name(&self) -> &str {
        "sec.agent.file_locks.parallel_reads_allowed"
    }
    fn category(&self) -> &str {
        "security.agent"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let mgr = Arc::new(FileLockManager::new());
        let _a = mgr
            .acquire_lock("agent-a", "src/lib.rs", LockType::Read)
            .await?;
        let _b = mgr
            .acquire_lock("agent-b", "src/lib.rs", LockType::Read)
            .await?;
        // Both reads coexisted without error.
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.agent.file_locks.write_blocked_by_other_reader ─────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.agent.file_locks.write_blocked_by_other_reader",
        crate_name: "brainwires-agent",
        invariant: "WRITE lock rejected while another agent holds a READ lock on the same file",
        factory: || Box::new(WriteBlockedByOtherReaderCase),
    }
}

struct WriteBlockedByOtherReaderCase;

#[async_trait]
impl EvaluationCase for WriteBlockedByOtherReaderCase {
    fn name(&self) -> &str {
        "sec.agent.file_locks.write_blocked_by_other_reader"
    }
    fn category(&self) -> &str {
        "security.agent"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let mgr = Arc::new(FileLockManager::new());
        let _reader = mgr
            .acquire_lock("agent-a", "src/lib.rs", LockType::Read)
            .await?;
        // Writer from a DIFFERENT agent must fail while reader is alive.
        let writer = mgr
            .acquire_lock("agent-b", "src/lib.rs", LockType::Write)
            .await;
        if writer.is_ok() {
            return Ok(TrialResult::failure(
                0,
                0,
                "WRITE lock granted while another agent holds a READ — lock invariant broken",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.agent.file_locks.write_blocked_by_other_writer ─────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.agent.file_locks.write_blocked_by_other_writer",
        crate_name: "brainwires-agent",
        invariant: "WRITE lock rejected while another agent holds a WRITE lock on the same file",
        factory: || Box::new(WriteBlockedByOtherWriterCase),
    }
}

struct WriteBlockedByOtherWriterCase;

#[async_trait]
impl EvaluationCase for WriteBlockedByOtherWriterCase {
    fn name(&self) -> &str {
        "sec.agent.file_locks.write_blocked_by_other_writer"
    }
    fn category(&self) -> &str {
        "security.agent"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let mgr = Arc::new(FileLockManager::new());
        let _writer_a = mgr
            .acquire_lock("agent-a", "src/lib.rs", LockType::Write)
            .await?;
        let writer_b = mgr
            .acquire_lock("agent-b", "src/lib.rs", LockType::Write)
            .await;
        if writer_b.is_ok() {
            return Ok(TrialResult::failure(
                0,
                0,
                "Two agents simultaneously granted WRITE locks — exclusivity invariant broken",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}

// ── sec.agent.file_locks.reentrant_write_same_agent ────────────────────────

inventory::submit! {
    SecurityCase {
        id: "sec.agent.file_locks.reentrant_write_same_agent",
        crate_name: "brainwires-agent",
        invariant: "Same agent re-acquiring its own WRITE lock succeeds (no self-deadlock)",
        factory: || Box::new(ReentrantWriteSameAgentCase),
    }
}

struct ReentrantWriteSameAgentCase;

#[async_trait]
impl EvaluationCase for ReentrantWriteSameAgentCase {
    fn name(&self) -> &str {
        "sec.agent.file_locks.reentrant_write_same_agent"
    }
    fn category(&self) -> &str {
        "security.agent"
    }
    async fn run(&self, _trial: usize) -> Result<TrialResult> {
        let mgr = Arc::new(FileLockManager::new());
        let _first = mgr
            .acquire_lock("agent-a", "src/lib.rs", LockType::Write)
            .await?;
        // Same agent re-acquires — must succeed (idempotent).
        let second = mgr
            .acquire_lock("agent-a", "src/lib.rs", LockType::Write)
            .await;
        if second.is_err() {
            return Ok(TrialResult::failure(
                0,
                0,
                "Same agent failed to re-acquire its own WRITE lock — self-deadlock risk",
            ));
        }
        Ok(TrialResult::success(0, 0))
    }
}
