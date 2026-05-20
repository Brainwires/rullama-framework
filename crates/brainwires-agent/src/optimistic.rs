//! Optimistic Concurrency with Conflict Resolution
//!
//! Based on Multi-Agent Coordination Survey research, this module provides
//! optimistic concurrency control that allows agents to proceed with operations
//! without acquiring locks upfront. Conflicts are detected at commit time
//! and resolved using configured strategies.
//!
//! # When to Use Optimistic vs Pessimistic
//!
//! | Scenario | Approach | Rationale |
//! |----------|----------|-----------|
//! | File reads | Optimistic | High contention unlikely |
//! | File writes to different files | Optimistic | No actual conflict |
//! | File writes to same file | Pessimistic | Real conflict likely |
//! | Build operations | Pessimistic | Expensive to retry |
//! | Git staging | Optimistic | Can merge staging areas |
//! | Git commit | Pessimistic | Must be sequential |
//! | Git push | Pessimistic | Remote state matters |
//!
//! # Key Concepts
//!
//! - **OptimisticToken**: Captures the version at the start of an operation
//! - **ResourceVersion**: Tracks version, hash, and modifier for each resource
//! - **ResolutionStrategy**: Configures how conflicts are resolved
//! - **OptimisticConflict**: Describes a detected conflict for resolution

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Optimistic concurrency controller
pub struct OptimisticController {
    /// Version tracking for resources
    versions: RwLock<HashMap<String, ResourceVersion>>,
    /// Conflict resolution strategies by resource pattern
    resolution_strategies: RwLock<HashMap<String, ResolutionStrategy>>,
    /// Default resolution strategy
    default_strategy: ResolutionStrategy,
    /// Conflict history for debugging/analysis
    conflict_history: RwLock<Vec<ConflictRecord>>,
    /// Maximum history entries to keep
    max_history: usize,
}

/// Version information for a resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceVersion {
    /// Monotonic version number
    pub version: u64,
    /// Content hash for change detection
    pub content_hash: String,
    /// Agent that last modified this resource
    pub last_modifier: String,
    /// When the modification occurred
    #[serde(skip, default = "Instant::now")]
    pub modified_at: Instant,
}

impl ResourceVersion {
    /// Create a new resource version
    pub fn new(content_hash: impl Into<String>, modifier: impl Into<String>) -> Self {
        Self {
            version: 1,
            content_hash: content_hash.into(),
            last_modifier: modifier.into(),
            modified_at: Instant::now(),
        }
    }

    /// Increment version with new hash
    pub fn increment(&mut self, content_hash: impl Into<String>, modifier: impl Into<String>) {
        self.version += 1;
        self.content_hash = content_hash.into();
        self.last_modifier = modifier.into();
        self.modified_at = Instant::now();
    }
}

/// Strategy for resolving conflicts
#[derive(Debug, Clone, Default)]
pub enum ResolutionStrategy {
    /// Last writer wins (overwrite)
    LastWriterWins,
    /// First writer wins (reject later)
    #[default]
    FirstWriterWins,
    /// Attempt to merge changes
    Merge(MergeStrategy),
    /// Escalate to orchestrator/user
    Escalate,
    /// Retry the operation with fresh state
    Retry {
        /// Maximum number of retry attempts.
        max_attempts: u32,
    },
}

/// Strategies for merging conflicting changes
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Line-by-line merge for text files
    TextMerge,
    /// JSON deep merge
    JsonMerge,
    /// Append both versions
    Append,
    /// Custom merge function name (for extension)
    Custom(String),
}

/// Describes a conflict between two operations
#[derive(Debug, Clone)]
pub struct OptimisticConflict {
    /// Resource that had the conflict
    pub resource_id: String,
    /// Agent that tried to commit
    pub conflicting_agent: String,
    /// Version the agent expected
    pub expected_version: u64,
    /// Current version when commit was attempted
    pub actual_version: u64,
    /// Agent that made the conflicting change
    pub holder_agent: String,
    /// When the conflict was detected
    pub detected_at: Instant,
}

impl OptimisticConflict {
    /// Get the version difference
    pub fn version_diff(&self) -> u64 {
        self.actual_version.saturating_sub(self.expected_version)
    }
}

/// Full conflict information for resolution
#[derive(Debug, Clone)]
pub struct OptimisticConflictDetails {
    /// The conflicting resource
    pub resource_id: String,
    /// First agent's data
    pub agent_a: String,
    /// Second agent's data
    pub agent_b: String,
    /// First agent's version
    pub version_a: ResourceVersion,
    /// Second agent's version
    pub version_b: ResourceVersion,
    /// Base version before both changes
    pub base_version: ResourceVersion,
    /// First agent's content (if available)
    pub content_a: Option<String>,
    /// Second agent's content (if available)
    pub content_b: Option<String>,
}

/// Result of conflict resolution
#[derive(Debug, Clone)]
pub enum Resolution {
    /// Use version from specified agent
    UseVersion(String),
    /// Use merged content
    Merged(String),
    /// Abort both operations
    AbortBoth,
    /// Keep both as separate resources
    KeepBoth {
        /// Suffix for the first version.
        suffix_a: String,
        /// Suffix for the second version.
        suffix_b: String,
    },
    /// Retry with fresh state
    Retry,
    /// Escalate to higher authority
    Escalate {
        /// Reason for escalation.
        reason: String,
    },
}

/// Token for optimistic operations
#[derive(Debug, Clone)]
pub struct OptimisticToken {
    /// Resource identifier
    pub resource_id: String,
    /// Version at the start of the operation
    pub base_version: u64,
    /// Content hash at the start
    pub base_hash: String,
    /// Agent performing the operation
    pub agent_id: String,
    /// When the token was created
    pub created_at: Instant,
}

impl OptimisticToken {
    /// Check if this token has expired (stale)
    pub fn is_stale(&self, max_age: std::time::Duration) -> bool {
        self.created_at.elapsed() > max_age
    }
}

/// Record of a conflict for history/debugging
#[derive(Debug, Clone)]
pub struct ConflictRecord {
    /// The conflict details
    pub conflict: OptimisticConflict,
    /// How it was resolved
    pub resolution: Resolution,
    /// When resolved
    pub resolved_at: Instant,
}

impl OptimisticController {
    /// Create a new optimistic controller with default settings
    pub fn new() -> Self {
        Self {
            versions: RwLock::new(HashMap::new()),
            resolution_strategies: RwLock::new(HashMap::new()),
            default_strategy: ResolutionStrategy::FirstWriterWins,
            conflict_history: RwLock::new(Vec::new()),
            max_history: 100,
        }
    }

    /// Create with a custom default strategy
    pub fn with_default_strategy(strategy: ResolutionStrategy) -> Self {
        Self {
            versions: RwLock::new(HashMap::new()),
            resolution_strategies: RwLock::new(HashMap::new()),
            default_strategy: strategy,
            conflict_history: RwLock::new(Vec::new()),
            max_history: 100,
        }
    }

    /// Set maximum history entries
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Start an optimistic operation - returns token with current version
    pub async fn begin_optimistic(&self, agent_id: &str, resource_id: &str) -> OptimisticToken {
        let versions = self.versions.read().await;
        let (base_version, base_hash) = versions
            .get(resource_id)
            .map(|v| (v.version, v.content_hash.clone()))
            .unwrap_or((0, String::new()));

        OptimisticToken {
            resource_id: resource_id.to_string(),
            base_version,
            base_hash,
            agent_id: agent_id.to_string(),
            created_at: Instant::now(),
        }
    }

    /// Commit optimistic operation - returns conflict if version changed
    pub async fn commit_optimistic(
        &self,
        token: OptimisticToken,
        new_content_hash: &str,
    ) -> Result<u64, OptimisticConflict> {
        let mut versions = self.versions.write().await;

        // Check for conflict
        if let Some(current) = versions.get(&token.resource_id)
            && current.version != token.base_version
        {
            return Err(OptimisticConflict {
                resource_id: token.resource_id,
                conflicting_agent: token.agent_id,
                expected_version: token.base_version,
                actual_version: current.version,
                holder_agent: current.last_modifier.clone(),
                detected_at: Instant::now(),
            });
        }

        // No conflict - commit the change
        let new_version = token.base_version + 1;
        versions.insert(
            token.resource_id,
            ResourceVersion {
                version: new_version,
                content_hash: new_content_hash.to_string(),
                last_modifier: token.agent_id,
                modified_at: Instant::now(),
            },
        );

        Ok(new_version)
    }

    /// Try to commit, and if conflict occurs, resolve it
    pub async fn commit_or_resolve(
        &self,
        token: OptimisticToken,
        new_content_hash: &str,
        new_content: Option<&str>,
    ) -> Result<CommitResult, String> {
        match self
            .commit_optimistic(token.clone(), new_content_hash)
            .await
        {
            Ok(version) => Ok(CommitResult::Committed { version }),
            Err(conflict) => {
                let resolution = self.resolve_conflict_auto(&conflict, new_content).await;

                // Record the conflict
                self.record_conflict(conflict.clone(), resolution.clone())
                    .await;

                match resolution {
                    Resolution::UseVersion(agent) => {
                        if agent == token.agent_id {
                            // Force our version
                            let version = self
                                .force_commit(&token.resource_id, new_content_hash, &token.agent_id)
                                .await;
                            Ok(CommitResult::Committed { version })
                        } else {
                            Ok(CommitResult::Rejected {
                                reason: format!("Conflict resolved in favor of {}", agent),
                            })
                        }
                    }
                    Resolution::Merged(merged_hash) => {
                        let version = self
                            .force_commit(&token.resource_id, &merged_hash, &token.agent_id)
                            .await;
                        Ok(CommitResult::Merged {
                            version,
                            merged_hash,
                        })
                    }
                    Resolution::Retry => Ok(CommitResult::RetryNeeded {
                        current_version: conflict.actual_version,
                    }),
                    Resolution::AbortBoth => Ok(CommitResult::Aborted {
                        reason: "Both operations aborted due to conflict".to_string(),
                    }),
                    Resolution::KeepBoth { suffix_a, suffix_b } => {
                        Ok(CommitResult::Split { suffix_a, suffix_b })
                    }
                    Resolution::Escalate { reason } => Ok(CommitResult::Escalated { reason }),
                }
            }
        }
    }

    /// Force commit without version check (for resolution)
    async fn force_commit(&self, resource_id: &str, content_hash: &str, agent_id: &str) -> u64 {
        let mut versions = self.versions.write().await;
        let current_version = versions.get(resource_id).map(|v| v.version).unwrap_or(0);
        let new_version = current_version + 1;

        versions.insert(
            resource_id.to_string(),
            ResourceVersion {
                version: new_version,
                content_hash: content_hash.to_string(),
                last_modifier: agent_id.to_string(),
                modified_at: Instant::now(),
            },
        );

        new_version
    }

    /// Resolve a conflict using the configured strategy
    async fn resolve_conflict_auto(
        &self,
        conflict: &OptimisticConflict,
        _new_content: Option<&str>,
    ) -> Resolution {
        let strategies = self.resolution_strategies.read().await;
        let strategy = strategies
            .get(&conflict.resource_id)
            .cloned()
            .unwrap_or_else(|| self.default_strategy.clone());

        match strategy {
            ResolutionStrategy::LastWriterWins => {
                Resolution::UseVersion(conflict.conflicting_agent.clone())
            }
            ResolutionStrategy::FirstWriterWins => {
                Resolution::UseVersion(conflict.holder_agent.clone())
            }
            ResolutionStrategy::Retry { max_attempts } => {
                // Check if we should retry
                if conflict.version_diff() < max_attempts as u64 {
                    Resolution::Retry
                } else {
                    Resolution::Escalate {
                        reason: format!("Max retry attempts ({}) exceeded", max_attempts),
                    }
                }
            }
            ResolutionStrategy::Escalate => Resolution::Escalate {
                reason: "Configured to escalate all conflicts".to_string(),
            },
            ResolutionStrategy::Merge(_strategy) => {
                // Merge requires content - if not available, escalate
                Resolution::Escalate {
                    reason: "Merge requires content, not available".to_string(),
                }
            }
        }
    }

    /// Manually resolve a conflict with full details
    pub async fn resolve_conflict(&self, conflict: &OptimisticConflictDetails) -> Resolution {
        let strategies = self.resolution_strategies.read().await;
        let strategy = strategies
            .get(&conflict.resource_id)
            .cloned()
            .unwrap_or_else(|| self.default_strategy.clone());

        match strategy {
            ResolutionStrategy::LastWriterWins => Resolution::UseVersion(conflict.agent_b.clone()),
            ResolutionStrategy::FirstWriterWins => Resolution::UseVersion(conflict.agent_a.clone()),
            ResolutionStrategy::Merge(merge_strategy) => {
                self.try_merge(conflict, &merge_strategy).await
            }
            ResolutionStrategy::Escalate => Resolution::Escalate {
                reason: "Policy requires manual resolution".to_string(),
            },
            ResolutionStrategy::Retry { .. } => Resolution::Retry,
        }
    }

    /// Attempt to merge content
    async fn try_merge(
        &self,
        conflict: &OptimisticConflictDetails,
        strategy: &MergeStrategy,
    ) -> Resolution {
        match (strategy, &conflict.content_a, &conflict.content_b) {
            (MergeStrategy::Append, Some(a), Some(b)) => {
                let merged = format!("{}\n{}", a, b);
                Resolution::Merged(hash_content(&merged))
            }
            (MergeStrategy::TextMerge, Some(a), Some(b)) => {
                // Line-by-line merge: deduplicate shared lines, append unique lines from both
                let lines_a: Vec<&str> = a.lines().collect();
                let lines_b: Vec<&str> = b.lines().collect();
                let mut merged = Vec::new();
                let mut used_b: Vec<bool> = vec![false; lines_b.len()];

                for line_a in &lines_a {
                    merged.push(*line_a);
                    // Mark matching lines in b as consumed
                    for (i, line_b) in lines_b.iter().enumerate() {
                        if !used_b[i] && line_a == line_b {
                            used_b[i] = true;
                            break;
                        }
                    }
                }
                // Append lines from b that weren't already present
                for (i, line_b) in lines_b.iter().enumerate() {
                    if !used_b[i] {
                        merged.push(*line_b);
                    }
                }

                let merged_content = merged.join("\n");
                Resolution::Merged(hash_content(&merged_content))
            }
            (MergeStrategy::JsonMerge, Some(a), Some(b)) => {
                // JSON deep merge: parse both as JSON objects and merge keys
                match (
                    serde_json::from_str::<serde_json::Value>(a),
                    serde_json::from_str::<serde_json::Value>(b),
                ) {
                    (Ok(mut val_a), Ok(val_b)) => {
                        json_deep_merge(&mut val_a, &val_b);
                        let merged_content = serde_json::to_string_pretty(&val_a)
                            .unwrap_or_else(|_| format!("{}", val_a));
                        Resolution::Merged(hash_content(&merged_content))
                    }
                    _ => Resolution::Escalate {
                        reason: "Failed to parse content as JSON for merge".to_string(),
                    },
                }
            }
            _ => Resolution::Escalate {
                reason: "Content not available for merge".to_string(),
            },
        }
    }

    /// Record a conflict in history
    async fn record_conflict(&self, conflict: OptimisticConflict, resolution: Resolution) {
        let mut history = self.conflict_history.write().await;

        history.push(ConflictRecord {
            conflict,
            resolution,
            resolved_at: Instant::now(),
        });

        // Trim history if needed
        while history.len() > self.max_history {
            history.remove(0);
        }
    }

    /// Register a resolution strategy for a resource pattern
    pub async fn register_strategy(&self, resource_pattern: &str, strategy: ResolutionStrategy) {
        self.resolution_strategies
            .write()
            .await
            .insert(resource_pattern.to_string(), strategy);
    }

    /// Get the current version of a resource
    pub async fn get_version(&self, resource_id: &str) -> Option<ResourceVersion> {
        self.versions.read().await.get(resource_id).cloned()
    }

    /// Check if a resource has been modified since a given version
    pub async fn has_changed(&self, resource_id: &str, since_version: u64) -> bool {
        self.versions
            .read()
            .await
            .get(resource_id)
            .map(|v| v.version > since_version)
            .unwrap_or(false)
    }

    /// Get conflict history
    pub async fn get_conflict_history(&self) -> Vec<ConflictRecord> {
        self.conflict_history.read().await.clone()
    }

    /// Clear conflict history
    pub async fn clear_history(&self) {
        self.conflict_history.write().await.clear();
    }

    /// Get statistics about conflicts
    pub async fn get_stats(&self) -> OptimisticStats {
        let history = self.conflict_history.read().await;
        let versions = self.versions.read().await;

        let total_conflicts = history.len();
        let resolved_by_retry = history
            .iter()
            .filter(|r| matches!(r.resolution, Resolution::Retry))
            .count();
        let escalated = history
            .iter()
            .filter(|r| matches!(r.resolution, Resolution::Escalate { .. }))
            .count();

        OptimisticStats {
            total_resources: versions.len(),
            total_conflicts,
            resolved_by_retry,
            escalated,
        }
    }
}

impl Default for OptimisticController {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a commit operation
#[derive(Debug, Clone)]
pub enum CommitResult {
    /// Successfully committed
    Committed {
        /// New version number.
        version: u64,
    },
    /// Merged with existing changes
    Merged {
        /// New version number.
        version: u64,
        /// Hash of the merged content.
        merged_hash: String,
    },
    /// Need to retry with fresh state
    RetryNeeded {
        /// Current version to retry against.
        current_version: u64,
    },
    /// Commit was rejected
    Rejected {
        /// Reason for rejection.
        reason: String,
    },
    /// Both operations aborted
    Aborted {
        /// Reason for abort.
        reason: String,
    },
    /// Split into separate resources
    Split {
        /// Suffix for the first version.
        suffix_a: String,
        /// Suffix for the second version.
        suffix_b: String,
    },
    /// Escalated to higher authority
    Escalated {
        /// Reason for escalation.
        reason: String,
    },
}

impl CommitResult {
    /// Check if the commit succeeded
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            CommitResult::Committed { .. } | CommitResult::Merged { .. }
        )
    }

    /// Get the new version if successful
    pub fn version(&self) -> Option<u64> {
        match self {
            CommitResult::Committed { version } | CommitResult::Merged { version, .. } => {
                Some(*version)
            }
            _ => None,
        }
    }
}

/// Statistics about optimistic concurrency
#[derive(Debug, Clone)]
pub struct OptimisticStats {
    /// Number of resources being tracked
    pub total_resources: usize,
    /// Total number of conflicts recorded
    pub total_conflicts: usize,
    /// Conflicts resolved by retry
    pub resolved_by_retry: usize,
    /// Conflicts escalated
    pub escalated: usize,
}

/// Helper function to hash content
/// Recursively merge `source` into `target`. For objects, keys from `source`
/// are inserted/overwritten in `target`. For non-object values, `source` wins.
fn json_deep_merge(target: &mut serde_json::Value, source: &serde_json::Value) {
    match (target, source) {
        (serde_json::Value::Object(t), serde_json::Value::Object(s)) => {
            for (key, value) in s {
                json_deep_merge(
                    t.entry(key.clone()).or_insert(serde_json::Value::Null),
                    value,
                );
            }
        }
        (target, source) => {
            *target = source.clone();
        }
    }
}

fn hash_content(content: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_optimistic_commit_success() {
        let controller = OptimisticController::new();

        // Begin optimistic operation
        let token = controller.begin_optimistic("agent-1", "file.txt").await;
        assert_eq!(token.base_version, 0);

        // Commit should succeed
        let result = controller.commit_optimistic(token, "hash123").await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_optimistic_commit_conflict() {
        let controller = OptimisticController::new();

        // Agent 1 begins operation
        let token1 = controller.begin_optimistic("agent-1", "file.txt").await;

        // Agent 2 begins operation
        let token2 = controller.begin_optimistic("agent-2", "file.txt").await;

        // Agent 1 commits first
        let result1 = controller.commit_optimistic(token1, "hash1").await;
        assert!(result1.is_ok());

        // Agent 2 tries to commit - should fail
        let result2 = controller.commit_optimistic(token2, "hash2").await;

        assert!(result2.is_err());
        let conflict = result2.unwrap_err();
        assert_eq!(conflict.expected_version, 0);
        assert_eq!(conflict.actual_version, 1);
        assert_eq!(conflict.holder_agent, "agent-1");
    }

    #[tokio::test]
    async fn test_version_tracking() {
        let controller = OptimisticController::new();

        // First commit
        let token1 = controller.begin_optimistic("agent-1", "file.txt").await;
        controller.commit_optimistic(token1, "hash1").await.unwrap();

        // Second commit
        let token2 = controller.begin_optimistic("agent-1", "file.txt").await;
        assert_eq!(token2.base_version, 1);
        controller.commit_optimistic(token2, "hash2").await.unwrap();

        // Verify version
        let version = controller.get_version("file.txt").await.unwrap();
        assert_eq!(version.version, 2);
        assert_eq!(version.content_hash, "hash2");
    }

    #[tokio::test]
    async fn test_resolution_strategy_last_writer_wins() {
        let controller =
            OptimisticController::with_default_strategy(ResolutionStrategy::LastWriterWins);

        // Two agents start
        let token1 = controller.begin_optimistic("agent-1", "file.txt").await;
        let token2 = controller.begin_optimistic("agent-2", "file.txt").await;

        // Agent 1 commits
        controller.commit_optimistic(token1, "hash1").await.unwrap();

        // Agent 2 commits with resolution
        let result = controller
            .commit_or_resolve(token2, "hash2", None)
            .await
            .unwrap();

        // With LastWriterWins, agent-2 should succeed
        assert!(result.is_success());
    }

    #[tokio::test]
    async fn test_resolution_strategy_first_writer_wins() {
        let controller =
            OptimisticController::with_default_strategy(ResolutionStrategy::FirstWriterWins);

        // Two agents start
        let token1 = controller.begin_optimistic("agent-1", "file.txt").await;
        let token2 = controller.begin_optimistic("agent-2", "file.txt").await;

        // Agent 1 commits
        controller.commit_optimistic(token1, "hash1").await.unwrap();

        // Agent 2 commits with resolution
        let result = controller
            .commit_or_resolve(token2, "hash2", None)
            .await
            .unwrap();

        // With FirstWriterWins, agent-2 should be rejected
        match result {
            CommitResult::Rejected { reason } => {
                assert!(reason.contains("agent-1"));
            }
            _ => panic!("Expected rejection"),
        }
    }

    #[tokio::test]
    async fn test_has_changed() {
        let controller = OptimisticController::new();

        // Initially no changes
        assert!(!controller.has_changed("file.txt", 0).await);

        // Make a commit
        let token = controller.begin_optimistic("agent-1", "file.txt").await;
        controller.commit_optimistic(token, "hash1").await.unwrap();

        // Now changed since version 0
        assert!(controller.has_changed("file.txt", 0).await);
        // But not since version 1
        assert!(!controller.has_changed("file.txt", 1).await);
    }

    #[tokio::test]
    async fn test_conflict_history() {
        let controller = OptimisticController::new();

        // Create a conflict
        let token1 = controller.begin_optimistic("agent-1", "file.txt").await;
        let token2 = controller.begin_optimistic("agent-2", "file.txt").await;

        controller.commit_optimistic(token1, "hash1").await.unwrap();

        // This will create a conflict
        let _ = controller.commit_or_resolve(token2, "hash2", None).await;

        // Check history
        let history = controller.get_conflict_history().await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].conflict.conflicting_agent, "agent-2");
    }

    #[tokio::test]
    async fn test_stats() {
        let controller = OptimisticController::new();

        // Create some commits
        for i in 0..5 {
            let token = controller
                .begin_optimistic("agent-1", &format!("file{}.txt", i))
                .await;
            controller
                .commit_optimistic(token, &format!("hash{}", i))
                .await
                .unwrap();
        }

        let stats = controller.get_stats().await;
        assert_eq!(stats.total_resources, 5);
        assert_eq!(stats.total_conflicts, 0);
    }

    #[test]
    fn test_token_staleness() {
        let token = OptimisticToken {
            resource_id: "test".to_string(),
            base_version: 0,
            base_hash: String::new(),
            agent_id: "agent-1".to_string(),
            created_at: Instant::now() - std::time::Duration::from_secs(120),
        };

        // Token should be stale after 60 seconds
        assert!(token.is_stale(std::time::Duration::from_secs(60)));
        // But not after 180 seconds
        assert!(!token.is_stale(std::time::Duration::from_secs(180)));
    }

    #[tokio::test]
    async fn test_custom_strategy_per_resource() {
        let controller = OptimisticController::new();

        // Register LastWriterWins for specific file
        controller
            .register_strategy("special.txt", ResolutionStrategy::LastWriterWins)
            .await;

        // Conflict on special.txt should use LastWriterWins
        let token1 = controller.begin_optimistic("agent-1", "special.txt").await;
        let token2 = controller.begin_optimistic("agent-2", "special.txt").await;

        controller.commit_optimistic(token1, "hash1").await.unwrap();

        let result = controller
            .commit_or_resolve(token2, "hash2", None)
            .await
            .unwrap();

        // LastWriterWins means agent-2 succeeds
        assert!(result.is_success());
    }
}
