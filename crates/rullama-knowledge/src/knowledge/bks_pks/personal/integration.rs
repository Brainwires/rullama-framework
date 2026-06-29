//! PKS Integration Module
//!
//! Provides integration points for:
//! 1. Processing user messages for implicit fact detection
//! 2. Observing tool usage for behavioral inference
//! 3. SSE listener for real-time server updates
//!
//! This module bridges the PKS components with the rest of the application.

use super::{
    PersonalFact, PersonalFactCategory, PersonalFactCollector, PersonalFactSource,
    PersonalKnowledgeCache, PersonalKnowledgeSettings,
};

use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Interval in seconds between behavioral inference emissions.
const KNOWLEDGE_INFERENCE_INTERVAL_SECS: u64 = 300;
/// Interval in seconds between background sync cycles.
const KNOWLEDGE_SYNC_INTERVAL_SECS: u64 = 60;

/// PKS Integration Manager
///
/// Coordinates all PKS integration points:
/// - Message processing for implicit detection
/// - Tool usage observation for behavioral inference
/// - Background sync with server
pub struct PksIntegration {
    /// Settings for PKS behavior
    settings: PersonalKnowledgeSettings,

    /// Collector for implicit fact detection from messages
    collector: PersonalFactCollector,

    /// Local cache (shared reference)
    cache: Option<Arc<Mutex<PersonalKnowledgeCache>>>,

    /// Tool usage tracker for behavioral inference
    tool_usage: ToolUsageTracker,

    /// Channel to send detected facts for processing
    fact_tx: Option<mpsc::UnboundedSender<DetectedFact>>,
}

/// A fact detected from user interaction
#[derive(Debug, Clone)]
pub struct DetectedFact {
    /// The detected personal fact.
    pub fact: PersonalFact,
    /// How the fact was detected.
    pub detection_source: DetectionSource,
}

/// How the fact was detected
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectionSource {
    /// From user message patterns
    ImplicitDetection,
    /// From tool usage observation
    BehavioralInference,
    /// From SSE server push
    ServerSync,
}

impl PksIntegration {
    /// Create a new PKS integration manager
    pub fn new(settings: PersonalKnowledgeSettings) -> Self {
        let collector = PersonalFactCollector::new(
            settings.implicit_detection_confidence,
            settings.enable_implicit_learning,
        );

        Self {
            settings,
            collector,
            cache: None,
            tool_usage: ToolUsageTracker::new(),
            fact_tx: None,
        }
    }

    /// Set the cache reference
    pub fn with_cache(mut self, cache: Arc<Mutex<PersonalKnowledgeCache>>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set the fact output channel
    pub fn with_fact_channel(mut self, tx: mpsc::UnboundedSender<DetectedFact>) -> Self {
        self.fact_tx = Some(tx);
        self
    }

    /// Process a user message for implicit fact detection
    ///
    /// Call this whenever the user sends a message. Detected facts are
    /// queued for storage/sync automatically.
    ///
    /// Returns the number of facts detected.
    pub fn process_user_message(&mut self, message: &str) -> usize {
        if !self.settings.enabled || !self.settings.enable_implicit_learning {
            return 0;
        }

        let facts = self.collector.process_message(message);
        let count = facts.len();

        for fact in facts {
            self.emit_fact(fact, DetectionSource::ImplicitDetection);
        }

        count
    }

    /// Record a tool execution for behavioral inference
    ///
    /// Call this after each successful tool execution to track usage patterns.
    pub fn record_tool_usage(&mut self, tool_name: &str, success: bool) {
        if !self.settings.enabled || !self.settings.enable_observed_learning {
            return;
        }

        self.tool_usage.record(tool_name, success);

        // Check for inferrable patterns
        if let Some(facts) = self.tool_usage.infer_facts() {
            for fact in facts {
                self.emit_fact(fact, DetectionSource::BehavioralInference);
            }
        }
    }

    /// Record the working directory to infer current project
    pub fn record_working_directory(&mut self, path: &str) {
        if !self.settings.enabled || !self.settings.enable_observed_learning {
            return;
        }

        // Extract project name from path
        if let Some(project_name) = extract_project_name(path) {
            let fact = PersonalFact::new(
                PersonalFactCategory::Context,
                "current_project".to_string(),
                project_name,
                Some(format!("Working directory: {}", path)),
                PersonalFactSource::SystemObserved,
                false,
            );
            self.emit_fact(fact, DetectionSource::BehavioralInference);
        }
    }

    /// Emit a detected fact to the channel and/or cache
    fn emit_fact(&mut self, fact: PersonalFact, source: DetectionSource) {
        // Send to channel if available
        if let Some(ref tx) = self.fact_tx {
            let detected = DetectedFact {
                fact: fact.clone(),
                detection_source: source,
            };
            let _ = tx.send(detected);
        }

        // Store directly in cache if available
        if let Some(ref cache) = self.cache
            && let Ok(mut cache) = cache.lock()
            && let Err(e) = cache.upsert_fact(fact)
        {
            tracing::warn!("Failed to store detected fact: {}", e);
        }
    }

    /// Check if PKS is enabled
    pub fn is_enabled(&self) -> bool {
        self.settings.enabled
    }

    /// Get current settings
    pub fn settings(&self) -> &PersonalKnowledgeSettings {
        &self.settings
    }
}

impl Default for PksIntegration {
    fn default() -> Self {
        Self::new(PersonalKnowledgeSettings::default())
    }
}

/// Tracks tool usage patterns for behavioral inference
pub struct ToolUsageTracker {
    /// Tool usage counts: tool_name -> (success_count, failure_count)
    usage: HashMap<String, (u32, u32)>,

    /// Last time we emitted inference facts
    last_inference: Instant,

    /// Minimum interval between inference emissions
    inference_interval: Duration,

    /// Minimum uses before inferring preference
    min_uses_for_inference: u32,
}

impl ToolUsageTracker {
    fn new() -> Self {
        Self {
            usage: HashMap::new(),
            last_inference: Instant::now(),
            inference_interval: Duration::from_secs(KNOWLEDGE_INFERENCE_INTERVAL_SECS), // 5 minutes
            min_uses_for_inference: 5,
        }
    }

    /// Record a tool usage
    fn record(&mut self, tool_name: &str, success: bool) {
        let entry = self.usage.entry(tool_name.to_string()).or_insert((0, 0));
        if success {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }

    /// Infer facts from usage patterns (rate-limited)
    fn infer_facts(&mut self) -> Option<Vec<PersonalFact>> {
        // Rate limit inference
        if self.last_inference.elapsed() < self.inference_interval {
            return None;
        }

        let mut facts = Vec::new();

        // Find frequently used tools
        for (tool_name, (successes, _failures)) in &self.usage {
            if *successes >= self.min_uses_for_inference {
                // Infer tool preference
                let category = categorize_tool(tool_name);
                let key = format!("preferred_{}_tool", category);

                let fact = PersonalFact::new(
                    PersonalFactCategory::Preference,
                    key,
                    tool_name.clone(),
                    Some(format!("Used {} times successfully", successes)),
                    PersonalFactSource::SystemObserved,
                    false,
                );
                facts.push(fact);
            }
        }

        // Infer capabilities from successful tool usage
        let file_ops =
            self.count_category_usage(&["read_file", "write_file", "edit_file", "glob", "grep"]);
        if file_ops >= self.min_uses_for_inference {
            facts.push(PersonalFact::new(
                PersonalFactCategory::Capability,
                "file_operations_proficiency".to_string(),
                "proficient".to_string(),
                Some(format!("Completed {} file operations", file_ops)),
                PersonalFactSource::SystemObserved,
                false,
            ));
        }

        let git_ops =
            self.count_category_usage(&["git_status", "git_diff", "git_log", "git_commit"]);
        if git_ops >= self.min_uses_for_inference {
            facts.push(PersonalFact::new(
                PersonalFactCategory::Capability,
                "git_proficiency".to_string(),
                "proficient".to_string(),
                Some(format!("Completed {} git operations", git_ops)),
                PersonalFactSource::SystemObserved,
                false,
            ));
        }

        if !facts.is_empty() {
            self.last_inference = Instant::now();
            Some(facts)
        } else {
            None
        }
    }

    fn count_category_usage(&self, tools: &[&str]) -> u32 {
        tools
            .iter()
            .filter_map(|t| self.usage.get(*t))
            .map(|(s, _)| s)
            .sum()
    }
}

/// Categorize a tool for preference tracking
fn categorize_tool(tool_name: &str) -> &'static str {
    match tool_name {
        "read_file" | "write_file" | "edit_file" | "glob" | "grep" => "file",
        "bash" | "execute_command" => "shell",
        "git_status" | "git_diff" | "git_log" | "git_commit" => "git",
        "web_search" | "fetch_url" => "web",
        "semantic_search" | "context_recall" => "search",
        _ => "general",
    }
}

/// Extract project name from a path
fn extract_project_name(path: &str) -> Option<String> {
    use std::path::Path;

    let path = Path::new(path);

    // Look for common project indicators
    let indicators = [
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "go.mod",
        ".git",
    ];

    // Check if any indicator exists in this directory
    for indicator in &indicators {
        if path.join(indicator).exists() {
            // Use the directory name as project name
            return path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string());
        }
    }

    // Fallback: just use the directory name if it's not a common system path
    let name = path.file_name()?.to_str()?;
    if !["home", "usr", "var", "tmp", "etc", "Users", "root"].contains(&name) {
        Some(name.to_string())
    } else {
        None
    }
}

// ============================================================================
// Background Sync for Personal Facts
// ============================================================================

/// Background syncer for personal facts using REST API polling.
///
/// Periodically polls the server for updates and uploads local changes.
/// The web frontend uses SSE; this CLI-side integration uses REST polling.
pub struct PksRestPoller {
    /// API client for server communication
    api_client: super::api::PersonalKnowledgeApiClient,

    /// Channel to send received facts
    fact_tx: mpsc::UnboundedSender<DetectedFact>,

    /// Local cache for getting pending facts to upload
    cache: Option<Arc<Mutex<PersonalKnowledgeCache>>>,

    /// Shutdown signal
    shutdown_rx: Option<tokio::sync::oneshot::Receiver<()>>,

    /// Sync interval (default: 60 seconds)
    sync_interval: Duration,

    /// Last sync timestamp (ISO 8601)
    last_sync: Option<String>,
}

impl PksRestPoller {
    /// Create a new background syncer
    pub fn new(server_url: &str, fact_tx: mpsc::UnboundedSender<DetectedFact>) -> Self {
        let api_client = super::api::PersonalKnowledgeApiClient::new(server_url);

        Self {
            api_client,
            fact_tx,
            cache: None,
            shutdown_rx: None,
            sync_interval: Duration::from_secs(KNOWLEDGE_SYNC_INTERVAL_SECS),
            last_sync: None,
        }
    }

    /// Set authentication token
    pub fn with_auth(mut self, token: String) -> Self {
        self.api_client.set_auth_token(token);
        self
    }

    /// Set local cache for bidirectional sync (upload pending facts)
    pub fn with_cache(mut self, cache: Arc<Mutex<PersonalKnowledgeCache>>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set sync interval
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.sync_interval = interval;
        self
    }

    /// Set shutdown receiver
    pub fn with_shutdown(mut self, rx: tokio::sync::oneshot::Receiver<()>) -> Self {
        self.shutdown_rx = Some(rx);
        self
    }

    /// Start the background sync loop (runs until shutdown)
    pub async fn listen(mut self) -> Result<()> {
        tracing::info!(
            "Starting PKS background sync (interval: {:?})",
            self.sync_interval
        );

        // Take shutdown receiver
        let mut shutdown_rx = self.shutdown_rx.take();
        let mut interval = tokio::time::interval(self.sync_interval);

        // Do an initial sync immediately
        if let Err(e) = self.perform_sync().await {
            tracing::debug!("Initial PKS sync failed (may not be logged in): {}", e);
        }

        loop {
            tokio::select! {
                // Check for shutdown signal
                _ = async {
                    if let Some(ref mut rx) = shutdown_rx {
                        rx.await.ok();
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {
                    tracing::info!("PKS background sync shutting down");
                    break;
                }

                // Perform periodic sync
                _ = interval.tick() => {
                    if let Err(e) = self.perform_sync().await {
                        tracing::debug!("PKS sync error: {}", e);
                    }
                }
            }
        }

        Ok(())
    }

    /// Perform a single sync operation
    async fn perform_sync(&mut self) -> Result<()> {
        // Get pending local facts to upload from the submission queue
        let pending_facts: Vec<PersonalFact> = if let Some(ref cache) = self.cache {
            if let Ok(cache) = cache.lock() {
                cache
                    .pending_submissions()
                    .iter()
                    .map(|p| p.fact.clone())
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Perform bidirectional sync via REST API
        let sync_result = self
            .api_client
            .sync(
                self.last_sync.as_deref(),
                None,           // client_id
                &pending_facts, // facts to upload
                &[],            // no feedback
                0.5,            // min_confidence
                100,            // limit
            )
            .await;

        match sync_result {
            Ok(result) => {
                // Update last sync timestamp
                self.last_sync = Some(result.sync_timestamp.clone());

                // Track count before consuming the facts
                let received_count = result.facts.len();
                let uploaded_count = pending_facts.len();

                // Process received facts from server
                for fact in result.facts {
                    let detected = DetectedFact {
                        fact,
                        detection_source: DetectionSource::ServerSync,
                    };

                    if let Err(e) = self.fact_tx.send(detected) {
                        tracing::warn!("Failed to send synced fact to channel: {}", e);
                    }
                }

                // Clear the pending submissions queue after successful upload
                if uploaded_count > 0
                    && let Some(ref cache) = self.cache
                    && let Ok(mut cache) = cache.lock()
                    && let Err(e) = cache.clear_pending_submissions()
                {
                    tracing::warn!("Failed to clear pending submissions: {}", e);
                }

                // Log sync activity (only if something happened)
                if received_count > 0 || uploaded_count > 0 {
                    tracing::debug!(
                        "PKS sync complete: received {} facts, uploaded {} facts",
                        received_count,
                        uploaded_count
                    );
                }

                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

// parse_category, parse_source, parse_timestamp live in api.rs

// ============================================================================
// Background Processor for Detected Facts
// ============================================================================

/// Processes detected facts in the background
///
/// Handles:
/// - Storing facts in local cache
/// - Queuing facts for server sync
/// - Deduplication and conflict resolution
pub struct PksBackgroundProcessor {
    /// Receiver for detected facts
    fact_rx: mpsc::UnboundedReceiver<DetectedFact>,

    /// Local cache
    cache: Arc<Mutex<PersonalKnowledgeCache>>,
}

impl PksBackgroundProcessor {
    /// Create a new background processor
    pub fn new(
        fact_rx: mpsc::UnboundedReceiver<DetectedFact>,
        cache: Arc<Mutex<PersonalKnowledgeCache>>,
        _settings: PersonalKnowledgeSettings,
    ) -> Self {
        Self { fact_rx, cache }
    }

    /// Run the background processor
    pub async fn run(mut self) {
        tracing::info!("PKS background processor started");

        while let Some(detected) = self.fact_rx.recv().await {
            if let Err(e) = self.process_fact(detected) {
                tracing::warn!("Failed to process detected fact: {}", e);
            }
        }

        tracing::info!("PKS background processor stopped");
    }

    /// Process a single detected fact
    fn process_fact(&self, detected: DetectedFact) -> Result<()> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|e| anyhow::anyhow!("Cache lock error: {}", e))?;

        // Check if fact already exists
        if let Some(existing) = cache.get_fact_by_key(&detected.fact.key) {
            // Fact exists - decide whether to update
            match detected.detection_source {
                DetectionSource::ServerSync => {
                    // Server facts always win (unless local-only)
                    if !existing.local_only {
                        cache.upsert_fact(detected.fact)?;
                    }
                }
                DetectionSource::ImplicitDetection | DetectionSource::BehavioralInference => {
                    // Only reinforce if source matches
                    if existing.source == detected.fact.source {
                        // Queue feedback to reinforce the existing fact
                        use super::PersonalFactFeedback;
                        let feedback = PersonalFactFeedback {
                            fact_id: existing.id.clone(),
                            is_reinforcement: true,
                            context: Some(format!(
                                "Detected again via {:?}",
                                detected.detection_source
                            )),
                            timestamp: chrono::Utc::now().timestamp(),
                        };
                        let _ = cache.queue_feedback(feedback);
                    }
                    // Otherwise, inferred facts don't override existing facts
                }
            }
        } else {
            // New fact - store it
            cache.upsert_fact(detected.fact)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pks_integration_creation() {
        let integration = PksIntegration::default();
        assert!(integration.is_enabled());
    }

    #[test]
    fn test_process_user_message() {
        let mut integration = PksIntegration::default();
        let count = integration.process_user_message("My name is John Smith");
        assert!(count > 0);
    }

    #[test]
    fn test_process_user_message_disabled() {
        let settings = PersonalKnowledgeSettings {
            enable_implicit_learning: false,
            ..Default::default()
        };
        let mut integration = PksIntegration::new(settings);
        let count = integration.process_user_message("My name is John Smith");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_tool_usage_tracking() {
        let mut integration = PksIntegration::default();

        // Record multiple uses
        for _ in 0..6 {
            integration.record_tool_usage("read_file", true);
        }

        // Should have tracked usage
        assert!(integration.tool_usage.usage.contains_key("read_file"));
    }

    #[test]
    fn test_extract_project_name() {
        // Would need actual filesystem for full test
        let result = extract_project_name("/home/user/invalid");
        assert!(result.is_some()); // Returns "invalid" as fallback
    }

    #[test]
    fn test_categorize_tool() {
        assert_eq!(categorize_tool("read_file"), "file");
        assert_eq!(categorize_tool("bash"), "shell");
        assert_eq!(categorize_tool("git_status"), "git");
        assert_eq!(categorize_tool("unknown_tool"), "general");
    }
}
