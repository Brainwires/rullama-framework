//! Server API client for behavioral knowledge synchronization
//!
//! Handles communication with the Brainwires server for syncing truths,
//! submitting new truths, and reporting reinforcements/contradictions.

use super::truth::{BehavioralTruth, TruthCategory, TruthFeedback, TruthSource};
use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// API client for the behavioral knowledge server
pub struct KnowledgeApiClient {
    /// HTTP client
    client: Client,

    /// Base URL for the API
    base_url: String,

    /// Authentication token (API key)
    auth_token: Option<String>,
}

impl KnowledgeApiClient {
    /// Create a new API client
    pub fn new(base_url: &str, auth_token: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            auth_token,
        }
    }

    /// Build authorization header
    fn auth_header(&self) -> Option<String> {
        self.auth_token.as_ref().map(|t| format!("Bearer {}", t))
    }

    /// Sync truths from server (bidirectional sync)
    pub async fn sync(&self, request: SyncRequest) -> Result<SyncResponse> {
        let url = format!("{}/api/knowledge/sync", self.base_url);

        let mut req = self.client.post(&url).json(&request);
        if let Some(auth) = self.auth_header() {
            req = req.header("Authorization", auth);
        }

        let response = req.send().await.context("Failed to send sync request")?;

        if response.status().is_success() {
            let sync_response: SyncResponse = response
                .json()
                .await
                .context("Failed to parse sync response")?;
            Ok(sync_response)
        } else {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Sync failed with status {}: {}", status, error_text);
        }
    }

    /// Get truths from server
    pub async fn get_truths(&self, params: GetTruthsParams) -> Result<GetTruthsResponse> {
        let mut url = format!("{}/api/knowledge/truths", self.base_url);

        let mut query_parts = Vec::new();
        if let Some(cat) = &params.category {
            query_parts.push(format!("category={}", cat));
        }
        if let Some(q) = &params.query {
            query_parts.push(format!("query={}", urlencoding::encode(q)));
        }
        if let Some(min) = params.min_confidence {
            query_parts.push(format!("min_confidence={}", min));
        }
        if let Some(lim) = params.limit {
            query_parts.push(format!("limit={}", lim));
        }
        if params.stats {
            query_parts.push("stats=true".to_string());
        }

        if !query_parts.is_empty() {
            url = format!("{}?{}", url, query_parts.join("&"));
        }

        let mut req = self.client.get(&url);
        if let Some(auth) = self.auth_header() {
            req = req.header("Authorization", auth);
        }

        let response = req
            .send()
            .await
            .context("Failed to send get truths request")?;

        if response.status().is_success() {
            let truths_response: GetTruthsResponse = response
                .json()
                .await
                .context("Failed to parse truths response")?;
            Ok(truths_response)
        } else {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Get truths failed with status {}: {}", status, error_text);
        }
    }

    /// Submit a new truth to the server
    pub async fn submit_truth(&self, truth: &TruthSubmission) -> Result<SubmitResponse> {
        let url = format!("{}/api/knowledge/truths", self.base_url);

        let mut req = self.client.post(&url).json(truth);
        if let Some(auth) = self.auth_header() {
            req = req.header("Authorization", auth);
        }

        let response = req.send().await.context("Failed to send submit request")?;

        if response.status().is_success() {
            let submit_response: SubmitResponse = response
                .json()
                .await
                .context("Failed to parse submit response")?;
            Ok(submit_response)
        } else {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Submit failed with status {}: {}", status, error_text);
        }
    }

    /// Report reinforcement of a truth
    pub async fn reinforce(
        &self,
        truth_id: &str,
        context: Option<&str>,
    ) -> Result<ReinforcementResponse> {
        let url = format!(
            "{}/api/knowledge/truths/{}/reinforce",
            self.base_url, truth_id
        );

        let body = ReinforcementRequest {
            context: context.map(|s| s.to_string()),
            ema_alpha: None,
        };

        let mut req = self.client.post(&url).json(&body);
        if let Some(auth) = self.auth_header() {
            req = req.header("Authorization", auth);
        }

        let response = req
            .send()
            .await
            .context("Failed to send reinforce request")?;

        if response.status().is_success() {
            let resp: ReinforcementResponse = response
                .json()
                .await
                .context("Failed to parse reinforce response")?;
            Ok(resp)
        } else {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Reinforce failed with status {}: {}", status, error_text);
        }
    }

    /// Report contradiction of a truth
    pub async fn contradict(
        &self,
        truth_id: &str,
        reason: Option<&str>,
        context: Option<&str>,
    ) -> Result<ContradictionResponse> {
        let url = format!(
            "{}/api/knowledge/truths/{}/contradict",
            self.base_url, truth_id
        );

        let body = ContradictionRequest {
            context: context.map(|s| s.to_string()),
            reason: reason.map(|s| s.to_string()),
            ema_alpha: None,
        };

        let mut req = self.client.post(&url).json(&body);
        if let Some(auth) = self.auth_header() {
            req = req.header("Authorization", auth);
        }

        let response = req
            .send()
            .await
            .context("Failed to send contradict request")?;

        if response.status().is_success() {
            let resp: ContradictionResponse = response
                .json()
                .await
                .context("Failed to parse contradict response")?;
            Ok(resp)
        } else {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Contradict failed with status {}: {}", status, error_text);
        }
    }

    /// Check server health
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/api/health", self.base_url);

        match self.client.get(&url).send().await {
            Ok(response) => Ok(response.status().is_success()),
            Err(_) => Ok(false),
        }
    }
}

// ============ Request/Response Types ============

/// Request for sync endpoint
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SyncRequest {
    /// ISO timestamp - get truths updated since this time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub since: Option<String>,

    /// Client identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,

    /// Minimum confidence threshold
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_confidence: Option<f32>,

    /// Max results
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,

    /// New truths to submit from client
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truths: Option<Vec<TruthSubmission>>,

    /// Feedback from client
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feedback: Option<Vec<TruthFeedback>>,
}

/// Response from sync endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResponse {
    /// Truths updated since the requested timestamp
    pub truths: Vec<ServerTruth>,

    /// Timestamp to use for next sync
    pub sync_timestamp: String,

    /// Whether there are more results
    pub has_more: bool,

    /// Stats about sync
    #[serde(default)]
    pub stats: SyncStats,
}

/// Statistics about a sync operation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SyncStats {
    /// Number of truths received from server.
    pub truths_received: u32,
    /// Number of truths sent to server.
    pub truths_sent: u32,
    /// Number of feedback reports sent.
    pub feedback_sent: u32,
}

/// Truth as returned from server (snake_case fields)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerTruth {
    /// Server-assigned truth ID.
    pub id: String,
    /// Truth category (snake_case).
    pub category: String,
    /// Context pattern this truth applies to.
    pub context_pattern: String,
    /// The behavioral rule.
    pub rule: String,
    /// Rationale for this truth.
    pub rationale: String,
    /// How the truth was learned.
    pub source: String,
    /// Confidence score.
    pub confidence: f32,
    /// Number of reinforcements.
    pub reinforcements: i32,
    /// Number of contradictions.
    pub contradictions: i32,
    /// User who created the truth.
    pub created_by: Option<String>,
    /// Whether this truth has been deleted.
    pub deleted: bool,
    /// Version number.
    pub version: i32,
    /// ISO 8601 creation timestamp.
    pub created_at: String,
    /// ISO 8601 last update timestamp.
    pub updated_at: String,
    /// ISO 8601 last usage timestamp.
    pub last_used: String,
}

impl ServerTruth {
    /// Convert server truth to local BehavioralTruth
    pub fn to_behavioral_truth(&self) -> BehavioralTruth {
        let category = match self.category.as_str() {
            "command_usage" => TruthCategory::CommandUsage,
            "task_strategy" => TruthCategory::TaskStrategy,
            "tool_behavior" => TruthCategory::ToolBehavior,
            "error_recovery" => TruthCategory::ErrorRecovery,
            "resource_management" => TruthCategory::ResourceManagement,
            "pattern_avoidance" => TruthCategory::PatternAvoidance,
            _ => TruthCategory::CommandUsage,
        };

        let source = match self.source.as_str() {
            "explicit_command" => TruthSource::ExplicitCommand,
            "conversation_correction" => TruthSource::ConversationCorrection,
            "success_pattern" => TruthSource::SuccessPattern,
            "failure_pattern" => TruthSource::FailurePattern,
            _ => TruthSource::ExplicitCommand,
        };

        // Parse ISO timestamps to unix timestamps
        let created_at = chrono::DateTime::parse_from_rfc3339(&self.created_at)
            .map(|dt| dt.timestamp())
            .unwrap_or_else(|_| chrono::Utc::now().timestamp());

        let last_used = chrono::DateTime::parse_from_rfc3339(&self.last_used)
            .map(|dt| dt.timestamp())
            .unwrap_or_else(|_| chrono::Utc::now().timestamp());

        BehavioralTruth {
            id: self.id.clone(),
            category,
            context_pattern: self.context_pattern.clone(),
            rule: self.rule.clone(),
            rationale: self.rationale.clone(),
            source,
            confidence: self.confidence,
            reinforcements: self.reinforcements as u32,
            contradictions: self.contradictions as u32,
            created_at,
            last_used,
            created_by: self.created_by.clone(),
            version: self.version as u64,
            deleted: self.deleted,
        }
    }
}

/// Params for get truths endpoint
#[derive(Debug, Clone, Default)]
pub struct GetTruthsParams {
    /// Filter by category.
    pub category: Option<String>,
    /// Search query string.
    pub query: Option<String>,
    /// Minimum confidence threshold.
    pub min_confidence: Option<f32>,
    /// Maximum number of results.
    pub limit: Option<u32>,
    /// Whether to include statistics.
    pub stats: bool,
}

/// Response from get truths endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetTruthsResponse {
    /// Returned truths.
    #[serde(default)]
    pub truths: Vec<ServerTruth>,

    /// Total truth count (when stats=true).
    #[serde(default)]
    pub total_truths: Option<u32>,
    /// Counts by category (when stats=true).
    #[serde(default)]
    pub by_category: Option<std::collections::HashMap<String, u32>>,
    /// Average confidence (when stats=true).
    #[serde(default)]
    pub avg_confidence: Option<f32>,
    /// Total reinforcements (when stats=true).
    #[serde(default)]
    pub total_reinforcements: Option<u32>,
    /// Total contradictions (when stats=true).
    #[serde(default)]
    pub total_contradictions: Option<u32>,
}

/// Truth submission to server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthSubmission {
    /// Truth category (snake_case).
    pub category: String,
    /// Context pattern.
    pub context_pattern: String,
    /// The behavioral rule.
    pub rule: String,
    /// Rationale.
    pub rationale: String,
    /// How the truth was learned.
    pub source: String,
    /// Optional confidence override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

impl From<&BehavioralTruth> for TruthSubmission {
    fn from(truth: &BehavioralTruth) -> Self {
        Self {
            category: truth.category.to_snake_case(),
            context_pattern: truth.context_pattern.clone(),
            rule: truth.rule.clone(),
            rationale: truth.rationale.clone(),
            source: truth.source.to_snake_case(),
            confidence: Some(truth.confidence),
        }
    }
}

/// Response from submit endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitResponse {
    /// The submitted truth as stored on server.
    pub truth: ServerTruth,
}

/// Request body for reinforcement
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReinforcementRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ema_alpha: Option<f32>,
}

/// Response from reinforcement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReinforcementResponse {
    /// Updated truth (if still active).
    pub truth: Option<ServerTruth>,
    /// Server message.
    pub message: String,
}

/// Request body for contradiction
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContradictionRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ema_alpha: Option<f32>,
}

/// Response from contradiction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionResponse {
    /// Updated truth (if still active).
    pub truth: Option<ServerTruth>,
    /// Server message.
    pub message: String,
    /// Whether the truth was deleted due to low confidence.
    #[serde(default)]
    pub was_deleted: bool,
}

// ============ Helper trait for enum serialization ============

trait ToSnakeCase {
    fn to_snake_case(&self) -> String;
}

impl ToSnakeCase for TruthCategory {
    fn to_snake_case(&self) -> String {
        match self {
            TruthCategory::CommandUsage => "command_usage",
            TruthCategory::TaskStrategy => "task_strategy",
            TruthCategory::ToolBehavior => "tool_behavior",
            TruthCategory::ErrorRecovery => "error_recovery",
            TruthCategory::ResourceManagement => "resource_management",
            TruthCategory::PatternAvoidance => "pattern_avoidance",
            TruthCategory::PromptingTechnique => "prompting_technique",
            TruthCategory::ClarifyingQuestions => "clarifying_questions",
        }
        .to_string()
    }
}

impl ToSnakeCase for TruthSource {
    fn to_snake_case(&self) -> String {
        match self {
            TruthSource::ExplicitCommand => "explicit_command",
            TruthSource::ConversationCorrection => "conversation_correction",
            TruthSource::SuccessPattern => "success_pattern",
            TruthSource::FailurePattern => "failure_pattern",
        }
        .to_string()
    }
}

// ============ Mock client for testing ============

#[cfg(test)]
#[allow(missing_docs)]
/// Mock client for testing knowledge API interactions.
pub struct MockKnowledgeApiClient {
    pub truths: Vec<BehavioralTruth>,
    pub submitted: Vec<BehavioralTruth>,
    pub reinforced: Vec<String>,
    pub contradicted: Vec<String>,
}

#[cfg(test)]
#[allow(missing_docs)]
impl Default for MockKnowledgeApiClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(missing_docs)]
impl MockKnowledgeApiClient {
    pub fn new() -> Self {
        Self {
            truths: Vec::new(),
            submitted: Vec::new(),
            reinforced: Vec::new(),
            contradicted: Vec::new(),
        }
    }

    pub fn with_truths(truths: Vec<BehavioralTruth>) -> Self {
        Self {
            truths,
            submitted: Vec::new(),
            reinforced: Vec::new(),
            contradicted: Vec::new(),
        }
    }

    pub async fn sync(&self, _request: SyncRequest) -> Result<SyncResponse> {
        use chrono::{TimeZone, Utc};
        Ok(SyncResponse {
            truths: self
                .truths
                .iter()
                .map(|t| {
                    let created = Utc.timestamp_opt(t.created_at, 0).unwrap();
                    let used = Utc.timestamp_opt(t.last_used, 0).unwrap();
                    ServerTruth {
                        id: t.id.clone(),
                        category: t.category.to_snake_case(),
                        context_pattern: t.context_pattern.clone(),
                        rule: t.rule.clone(),
                        rationale: t.rationale.clone(),
                        source: t.source.to_snake_case(),
                        confidence: t.confidence,
                        reinforcements: t.reinforcements as i32,
                        contradictions: t.contradictions as i32,
                        created_by: t.created_by.clone(),
                        deleted: t.deleted,
                        version: t.version as i32,
                        created_at: created.to_rfc3339(),
                        updated_at: created.to_rfc3339(),
                        last_used: used.to_rfc3339(),
                    }
                })
                .collect(),
            sync_timestamp: Utc::now().to_rfc3339(),
            has_more: false,
            stats: SyncStats::default(),
        })
    }

    pub async fn submit_truth(&mut self, truth: &BehavioralTruth) -> Result<SubmitResponse> {
        use chrono::{TimeZone, Utc};
        self.submitted.push(truth.clone());
        let created = Utc.timestamp_opt(truth.created_at, 0).unwrap();
        let used = Utc.timestamp_opt(truth.last_used, 0).unwrap();
        Ok(SubmitResponse {
            truth: ServerTruth {
                id: truth.id.clone(),
                category: truth.category.to_snake_case(),
                context_pattern: truth.context_pattern.clone(),
                rule: truth.rule.clone(),
                rationale: truth.rationale.clone(),
                source: truth.source.to_snake_case(),
                confidence: truth.confidence,
                reinforcements: truth.reinforcements as i32,
                contradictions: truth.contradictions as i32,
                created_by: truth.created_by.clone(),
                deleted: truth.deleted,
                version: truth.version as i32,
                created_at: created.to_rfc3339(),
                updated_at: created.to_rfc3339(),
                last_used: used.to_rfc3339(),
            },
        })
    }

    pub async fn reinforce(
        &mut self,
        truth_id: &str,
        _context: Option<&str>,
    ) -> Result<ReinforcementResponse> {
        self.reinforced.push(truth_id.to_string());
        Ok(ReinforcementResponse {
            truth: None,
            message: "Reinforced".to_string(),
        })
    }

    pub async fn contradict(
        &mut self,
        truth_id: &str,
        _reason: Option<&str>,
        _context: Option<&str>,
    ) -> Result<ContradictionResponse> {
        self.contradicted.push(truth_id.to_string());
        Ok(ContradictionResponse {
            truth: None,
            message: "Contradicted".to_string(),
            was_deleted: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_truth() -> BehavioralTruth {
        BehavioralTruth::new(
            TruthCategory::CommandUsage,
            "test".to_string(),
            "test rule".to_string(),
            "test rationale".to_string(),
            TruthSource::ExplicitCommand,
            None,
        )
    }

    #[tokio::test]
    async fn test_mock_client() {
        let mut mock = MockKnowledgeApiClient::new();

        let truth = create_test_truth();
        let response = mock.submit_truth(&truth).await.unwrap();

        assert_eq!(response.truth.id, truth.id);
        assert_eq!(mock.submitted.len(), 1);
    }

    #[tokio::test]
    async fn test_mock_sync() {
        let truth = create_test_truth();
        let mock = MockKnowledgeApiClient::with_truths(vec![truth.clone()]);

        let response = mock.sync(SyncRequest::default()).await.unwrap();
        assert_eq!(response.truths.len(), 1);
        assert_eq!(response.truths[0].id, truth.id);
    }

    #[test]
    fn test_truth_submission_from_behavioral() {
        let truth = create_test_truth();
        let submission = TruthSubmission::from(&truth);

        assert_eq!(submission.category, "command_usage");
        assert_eq!(submission.source, "explicit_command");
        assert_eq!(submission.rule, truth.rule);
    }

    #[test]
    fn test_server_truth_to_behavioral() {
        let server = ServerTruth {
            id: "test-id".to_string(),
            category: "task_strategy".to_string(),
            context_pattern: "pattern".to_string(),
            rule: "rule".to_string(),
            rationale: "rationale".to_string(),
            source: "success_pattern".to_string(),
            confidence: 0.9,
            reinforcements: 5,
            contradictions: 1,
            created_by: Some("user".to_string()),
            deleted: false,
            version: 1,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            last_used: "2024-01-01T00:00:00Z".to_string(),
        };

        let truth = server.to_behavioral_truth();

        assert_eq!(truth.id, "test-id");
        assert!(matches!(truth.category, TruthCategory::TaskStrategy));
        assert!(matches!(truth.source, TruthSource::SuccessPattern));
        assert_eq!(truth.confidence, 0.9);
    }
}
