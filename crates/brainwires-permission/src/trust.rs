//! Trust Factor System - Dynamic trust scoring for agents
//!
//! Implements a reputation-based trust system where agents build trust through
//! successful operations and lose trust through violations. Trust levels affect
//! what capabilities an agent can access.
//!
//! # Trust Levels
//!
//! - **System (4)**: Internal system agents, maximum trust
//! - **High (3)**: Established agents with excellent track record (score >= 0.9)
//! - **Medium (2)**: Agents with good track record (score >= 0.7)
//! - **Low (1)**: New or recovering agents (score >= 0.4)
//! - **Untrusted (0)**: Agents with poor track record (score < 0.4)
//!
//! # Example
//!
//! ```rust,ignore
//! use brainwires::permissions::trust::{TrustFactor, TrustLevel, TrustManager};
//!
//! let mut manager = TrustManager::new()?;
//!
//! // Record successful operations
//! manager.record_success("agent-123");
//!
//! // Record violations
//! manager.record_violation("agent-123", ViolationSeverity::Minor);
//!
//! // Check trust level
//! let level = manager.get_trust_level("agent-123");
//! ```

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Trust level enum representing discrete trust categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TrustLevel {
    /// Agent with poor track record (score < 0.4)
    Untrusted = 0,
    /// New or recovering agent (score >= 0.4)
    #[default]
    Low = 1,
    /// Agent with good track record (score >= 0.7)
    Medium = 2,
    /// Established agent with excellent track record (score >= 0.9)
    High = 3,
    /// Internal system agent, maximum trust
    System = 4,
}

impl TrustLevel {
    /// Convert from numeric level
    pub fn from_u8(level: u8) -> Self {
        match level {
            0 => TrustLevel::Untrusted,
            1 => TrustLevel::Low,
            2 => TrustLevel::Medium,
            3 => TrustLevel::High,
            4 => TrustLevel::System,
            _ => TrustLevel::Low,
        }
    }

    /// Convert to numeric level
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    /// Derive trust level from a score (0.0 to 1.0)
    pub fn from_score(score: f32) -> Self {
        if score >= 0.9 {
            TrustLevel::High
        } else if score >= 0.7 {
            TrustLevel::Medium
        } else if score >= 0.4 {
            TrustLevel::Low
        } else {
            TrustLevel::Untrusted
        }
    }
}

impl std::fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrustLevel::Untrusted => write!(f, "Untrusted"),
            TrustLevel::Low => write!(f, "Low"),
            TrustLevel::Medium => write!(f, "Medium"),
            TrustLevel::High => write!(f, "High"),
            TrustLevel::System => write!(f, "System"),
        }
    }
}

/// Severity of a policy violation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ViolationSeverity {
    /// Minor violation (e.g., accessing non-critical denied path)
    Minor,
    /// Major violation (e.g., attempting destructive operation without approval)
    Major,
    /// Critical violation (e.g., attempting to access secrets, bypass security)
    Critical,
}

impl ViolationSeverity {
    /// Get the penalty for this violation severity
    pub fn penalty(&self) -> f32 {
        match self {
            ViolationSeverity::Minor => 0.02,
            ViolationSeverity::Major => 0.08,
            ViolationSeverity::Critical => 0.15,
        }
    }

    /// Get the recent penalty multiplier (violations in last 24h are weighted more)
    pub fn recent_penalty(&self) -> f32 {
        match self {
            ViolationSeverity::Minor => 0.04,
            ViolationSeverity::Major => 0.15,
            ViolationSeverity::Critical => 0.30,
        }
    }
}

/// Violation counts by severity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ViolationCounts {
    /// Total minor violations
    pub minor: u32,
    /// Total major violations
    pub major: u32,
    /// Total critical violations
    pub critical: u32,
    /// Recent minor violations (last 24h)
    pub recent_minor: u32,
    /// Recent major violations (last 24h)
    pub recent_major: u32,
    /// Recent critical violations (last 24h)
    pub recent_critical: u32,
}

impl ViolationCounts {
    /// Calculate total penalty from violations
    pub fn total_penalty(&self) -> f32 {
        let base_penalty = (self.minor as f32 * ViolationSeverity::Minor.penalty())
            + (self.major as f32 * ViolationSeverity::Major.penalty())
            + (self.critical as f32 * ViolationSeverity::Critical.penalty());

        let recent_penalty = (self.recent_minor as f32 * ViolationSeverity::Minor.recent_penalty())
            + (self.recent_major as f32 * ViolationSeverity::Major.recent_penalty())
            + (self.recent_critical as f32 * ViolationSeverity::Critical.recent_penalty());

        base_penalty + recent_penalty
    }

    /// Increment violation count
    pub fn record(&mut self, severity: ViolationSeverity) {
        match severity {
            ViolationSeverity::Minor => {
                self.minor += 1;
                self.recent_minor += 1;
            }
            ViolationSeverity::Major => {
                self.major += 1;
                self.recent_major += 1;
            }
            ViolationSeverity::Critical => {
                self.critical += 1;
                self.recent_critical += 1;
            }
        }
    }

    /// Decay recent violations (should be called periodically)
    pub fn decay_recent(&mut self) {
        self.recent_minor = 0;
        self.recent_major = 0;
        self.recent_critical = 0;
    }
}

/// Trust factor for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustFactor {
    /// Agent ID
    pub agent_id: String,
    /// Current trust score (0.0 to 1.0)
    pub score: f32,
    /// Derived trust level
    pub level: TrustLevel,
    /// Violation counts
    pub violations: ViolationCounts,
    /// Number of successful operations
    pub successful_ops: u64,
    /// Total number of operations
    pub total_ops: u64,
    /// When this factor was last updated
    pub last_updated: DateTime<Utc>,
    /// When recent violations should be decayed
    pub violations_decay_at: DateTime<Utc>,
    /// Whether this is a system agent (always trusted)
    pub is_system: bool,
}

impl TrustFactor {
    /// Create a new trust factor for an agent
    pub fn new(agent_id: &str) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            score: 0.5, // Start neutral
            level: TrustLevel::Low,
            violations: ViolationCounts::default(),
            successful_ops: 0,
            total_ops: 0,
            last_updated: Utc::now(),
            violations_decay_at: Utc::now() + Duration::hours(24),
            is_system: false,
        }
    }

    /// Create a system agent trust factor
    pub fn system(agent_id: &str) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            score: 1.0,
            level: TrustLevel::System,
            violations: ViolationCounts::default(),
            successful_ops: 0,
            total_ops: 0,
            last_updated: Utc::now(),
            violations_decay_at: Utc::now(),
            is_system: true,
        }
    }

    /// Record a successful operation
    pub fn record_success(&mut self) {
        self.successful_ops += 1;
        self.total_ops += 1;
        self.recalculate();
    }

    /// Record a failed operation (not a violation, just a failure)
    pub fn record_failure(&mut self) {
        self.total_ops += 1;
        self.recalculate();
    }

    /// Record a policy violation
    pub fn record_violation(&mut self, severity: ViolationSeverity) {
        self.violations.record(severity);
        self.total_ops += 1;
        self.recalculate();
    }

    /// Recalculate score and level
    fn recalculate(&mut self) {
        if self.is_system {
            return; // System agents always have max trust
        }

        // Check if violations need decay
        if Utc::now() > self.violations_decay_at {
            self.violations.decay_recent();
            self.violations_decay_at = Utc::now() + Duration::hours(24);
        }

        // Calculate base score from success rate
        let base_score = if self.total_ops > 0 {
            self.successful_ops as f32 / self.total_ops as f32
        } else {
            0.5 // Neutral for new agents
        };

        // Apply violation penalty
        let penalty = self.violations.total_penalty();
        self.score = (base_score - penalty).clamp(0.0, 1.0);

        // Derive level from score
        self.level = TrustLevel::from_score(self.score);
        self.last_updated = Utc::now();
    }

    /// Manually set trust level (for overrides)
    pub fn set_level(&mut self, level: TrustLevel) {
        self.level = level;
        // Update score to match level
        self.score = match level {
            TrustLevel::Untrusted => 0.2,
            TrustLevel::Low => 0.5,
            TrustLevel::Medium => 0.75,
            TrustLevel::High => 0.95,
            TrustLevel::System => 1.0,
        };
        self.last_updated = Utc::now();
    }

    /// Reset trust factor to defaults
    pub fn reset(&mut self) {
        self.score = 0.5;
        self.level = TrustLevel::Low;
        self.violations = ViolationCounts::default();
        self.successful_ops = 0;
        self.total_ops = 0;
        self.last_updated = Utc::now();
    }
}

/// Persisted trust store
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TrustStore {
    factors: HashMap<String, TrustFactor>,
    last_saved: DateTime<Utc>,
}

/// Trust manager for managing agent trust factors
#[derive(Debug)]
pub struct TrustManager {
    /// In-memory trust factors
    factors: HashMap<String, TrustFactor>,
    /// Path to persistence file
    store_path: PathBuf,
    /// Whether to persist changes
    persist: bool,
}

impl TrustManager {
    /// Create a new trust manager
    #[cfg(feature = "native")]
    pub fn new() -> Result<Self> {
        let store_path = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Failed to get home directory"))?
            .join(".brainwires")
            .join("trust_store.json");
        let mut manager = Self {
            factors: HashMap::new(),
            store_path,
            persist: true,
        };

        // Load existing trust data
        manager.load()?;

        Ok(manager)
    }

    /// Create a trust manager with custom path
    pub fn with_path(path: PathBuf) -> Result<Self> {
        let mut manager = Self {
            factors: HashMap::new(),
            store_path: path,
            persist: true,
        };

        manager.load()?;
        Ok(manager)
    }

    /// Create an in-memory only trust manager (no persistence)
    pub fn in_memory() -> Self {
        Self {
            factors: HashMap::new(),
            store_path: PathBuf::new(),
            persist: false,
        }
    }

    /// Load trust data from disk
    fn load(&mut self) -> Result<()> {
        if !self.store_path.exists() {
            return Ok(());
        }

        let content = fs::read_to_string(&self.store_path)?;
        let store: TrustStore = serde_json::from_str(&content)?;

        // Apply time-based decay on load
        self.factors = store.factors;
        for factor in self.factors.values_mut() {
            if Utc::now() > factor.violations_decay_at {
                factor.violations.decay_recent();
                factor.violations_decay_at = Utc::now() + Duration::hours(24);
            }
        }

        Ok(())
    }

    /// Save trust data to disk
    pub fn save(&self) -> Result<()> {
        if !self.persist {
            return Ok(());
        }

        // Ensure parent directory exists
        if let Some(parent) = self.store_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let store = TrustStore {
            factors: self.factors.clone(),
            last_saved: Utc::now(),
        };

        let content = serde_json::to_string_pretty(&store)?;
        fs::write(&self.store_path, content)?;

        Ok(())
    }

    /// Get or create a trust factor for an agent
    pub fn get_or_create(&mut self, agent_id: &str) -> &mut TrustFactor {
        self.factors
            .entry(agent_id.to_string())
            .or_insert_with(|| TrustFactor::new(agent_id))
    }

    /// Get trust factor for an agent (if exists)
    pub fn get(&self, agent_id: &str) -> Option<&TrustFactor> {
        self.factors.get(agent_id)
    }

    /// Get trust level for an agent
    pub fn get_trust_level(&self, agent_id: &str) -> TrustLevel {
        self.factors
            .get(agent_id)
            .map(|f| f.level)
            .unwrap_or(TrustLevel::Low)
    }

    /// Record a successful operation
    pub fn record_success(&mut self, agent_id: &str) {
        let factor = self.get_or_create(agent_id);
        factor.record_success();
        let _ = self.save();
    }

    /// Record a failed operation
    pub fn record_failure(&mut self, agent_id: &str) {
        let factor = self.get_or_create(agent_id);
        factor.record_failure();
        let _ = self.save();
    }

    /// Record a violation
    pub fn record_violation(&mut self, agent_id: &str, severity: ViolationSeverity) {
        let factor = self.get_or_create(agent_id);
        factor.record_violation(severity);
        let _ = self.save();
    }

    /// Set trust level for an agent (manual override)
    pub fn set_trust_level(&mut self, agent_id: &str, level: TrustLevel) {
        let factor = self.get_or_create(agent_id);
        factor.set_level(level);
        let _ = self.save();
    }

    /// Reset an agent's trust
    pub fn reset(&mut self, agent_id: &str) {
        if let Some(factor) = self.factors.get_mut(agent_id) {
            factor.reset();
            let _ = self.save();
        }
    }

    /// Remove an agent's trust data
    pub fn remove(&mut self, agent_id: &str) -> Option<TrustFactor> {
        let removed = self.factors.remove(agent_id);
        let _ = self.save();
        removed
    }

    /// Register a system agent
    pub fn register_system_agent(&mut self, agent_id: &str) {
        self.factors
            .insert(agent_id.to_string(), TrustFactor::system(agent_id));
        let _ = self.save();
    }

    /// Get all agent IDs
    pub fn agents(&self) -> Vec<&str> {
        self.factors.keys().map(|s| s.as_str()).collect()
    }

    /// Get statistics
    pub fn statistics(&self) -> TrustStatistics {
        let mut stats = TrustStatistics {
            total_agents: self.factors.len(),
            ..Default::default()
        };

        for factor in self.factors.values() {
            match factor.level {
                TrustLevel::Untrusted => stats.untrusted += 1,
                TrustLevel::Low => stats.low_trust += 1,
                TrustLevel::Medium => stats.medium_trust += 1,
                TrustLevel::High => stats.high_trust += 1,
                TrustLevel::System => stats.system += 1,
            }

            stats.total_violations += factor.violations.minor as usize
                + factor.violations.major as usize
                + factor.violations.critical as usize;
            stats.total_operations += factor.total_ops as usize;
        }

        if stats.total_operations > 0 {
            let total_success: u64 = self.factors.values().map(|f| f.successful_ops).sum();
            stats.average_score = total_success as f32 / stats.total_operations as f32;
        }

        stats
    }
}

/// Trust statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrustStatistics {
    /// Total number of tracked agents.
    pub total_agents: usize,
    /// Number of untrusted agents.
    pub untrusted: usize,
    /// Number of low-trust agents.
    pub low_trust: usize,
    /// Number of medium-trust agents.
    pub medium_trust: usize,
    /// Number of high-trust agents.
    pub high_trust: usize,
    /// Number of system-level agents.
    pub system: usize,
    /// Total policy violations across all agents.
    pub total_violations: usize,
    /// Total operations across all agents.
    pub total_operations: usize,
    /// Average trust score.
    pub average_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_level_from_score() {
        assert_eq!(TrustLevel::from_score(0.95), TrustLevel::High);
        assert_eq!(TrustLevel::from_score(0.9), TrustLevel::High);
        assert_eq!(TrustLevel::from_score(0.85), TrustLevel::Medium);
        assert_eq!(TrustLevel::from_score(0.7), TrustLevel::Medium);
        assert_eq!(TrustLevel::from_score(0.5), TrustLevel::Low);
        assert_eq!(TrustLevel::from_score(0.4), TrustLevel::Low);
        assert_eq!(TrustLevel::from_score(0.3), TrustLevel::Untrusted);
        assert_eq!(TrustLevel::from_score(0.0), TrustLevel::Untrusted);
    }

    #[test]
    fn test_trust_factor_success_increases_score() {
        let mut factor = TrustFactor::new("test-agent");

        // Record many successes
        for _ in 0..10 {
            factor.record_success();
        }

        assert!(factor.score > 0.5);
        assert_eq!(factor.successful_ops, 10);
        assert_eq!(factor.total_ops, 10);
    }

    #[test]
    fn test_trust_factor_violations_decrease_score() {
        let mut factor = TrustFactor::new("test-agent");

        // Start with some successes
        for _ in 0..10 {
            factor.record_success();
        }

        let initial_score = factor.score;

        // Record a major violation
        factor.record_violation(ViolationSeverity::Major);

        assert!(factor.score < initial_score);
    }

    #[test]
    fn test_trust_factor_critical_violation() {
        let mut factor = TrustFactor::new("test-agent");

        // Record a critical violation
        factor.record_violation(ViolationSeverity::Critical);

        // Score should drop significantly
        assert!(factor.score < 0.4);
        assert_eq!(factor.level, TrustLevel::Untrusted);
    }

    #[test]
    fn test_system_agent_always_trusted() {
        let mut factor = TrustFactor::system("system-agent");

        // Even with violations, system agents stay at max trust
        factor.record_violation(ViolationSeverity::Critical);

        assert_eq!(factor.level, TrustLevel::System);
        assert_eq!(factor.score, 1.0);
    }

    #[test]
    fn test_trust_manager() {
        let mut manager = TrustManager::in_memory();

        // Record operations
        manager.record_success("agent-1");
        manager.record_success("agent-1");
        manager.record_violation("agent-2", ViolationSeverity::Minor);

        // Check levels
        assert!(manager.get_trust_level("agent-1") >= TrustLevel::Low);

        // Check statistics
        let stats = manager.statistics();
        assert_eq!(stats.total_agents, 2);
    }

    #[test]
    fn test_violation_counts() {
        let mut counts = ViolationCounts::default();

        counts.record(ViolationSeverity::Minor);
        counts.record(ViolationSeverity::Major);
        counts.record(ViolationSeverity::Critical);

        assert_eq!(counts.minor, 1);
        assert_eq!(counts.major, 1);
        assert_eq!(counts.critical, 1);

        // Penalty should be significant
        let penalty = counts.total_penalty();
        assert!(penalty > 0.2);
    }

    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::System > TrustLevel::High);
        assert!(TrustLevel::High > TrustLevel::Medium);
        assert!(TrustLevel::Medium > TrustLevel::Low);
        assert!(TrustLevel::Low > TrustLevel::Untrusted);
    }

    #[test]
    fn test_reset_trust() {
        let mut manager = TrustManager::in_memory();

        // Build up some trust
        for _ in 0..20 {
            manager.record_success("agent-1");
        }

        // Then violate
        manager.record_violation("agent-1", ViolationSeverity::Critical);

        // Reset
        manager.reset("agent-1");

        let factor = manager.get("agent-1").unwrap();
        assert_eq!(factor.score, 0.5);
        assert_eq!(factor.level, TrustLevel::Low);
        assert_eq!(factor.successful_ops, 0);
    }
}
