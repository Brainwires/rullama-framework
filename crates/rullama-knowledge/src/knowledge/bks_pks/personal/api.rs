//! API client for Personal Knowledge System server communication
//!
//! Handles sync, submission, and feedback with the server-side PKS.

use super::fact::{PersonalFact, PersonalFactCategory, PersonalFactFeedback, PersonalFactSource};
use anyhow::{Context, Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// API client for personal knowledge server
pub struct PersonalKnowledgeApiClient {
    client: Client,
    base_url: String,
    auth_token: Option<String>,
}

/// Server response for a personal fact
#[derive(Debug, Deserialize)]
struct ServerFact {
    id: String,
    category: String,
    key: String,
    value: String,
    context: Option<String>,
    confidence: f32,
    reinforcements: i32,
    contradictions: i32,
    source: String,
    version: i64,
    deleted: bool,
    created_at: String,
    updated_at: String,
    last_used: String,
}

impl ServerFact {
    fn into_personal_fact(self) -> PersonalFact {
        PersonalFact {
            id: self.id,
            category: parse_category(&self.category),
            key: self.key,
            value: self.value,
            context: self.context,
            confidence: self.confidence,
            reinforcements: self.reinforcements as u32,
            contradictions: self.contradictions as u32,
            last_used: parse_timestamp(&self.last_used),
            created_at: parse_timestamp(&self.created_at),
            updated_at: parse_timestamp(&self.updated_at),
            source: parse_source(&self.source),
            version: self.version as u64,
            deleted: self.deleted,
            local_only: false, // Server facts are never local-only
        }
    }
}

fn parse_category(s: &str) -> PersonalFactCategory {
    match s {
        "identity" => PersonalFactCategory::Identity,
        "preference" => PersonalFactCategory::Preference,
        "capability" => PersonalFactCategory::Capability,
        "context" => PersonalFactCategory::Context,
        "constraint" => PersonalFactCategory::Constraint,
        "relationship" => PersonalFactCategory::Relationship,
        _ => PersonalFactCategory::Preference, // Default
    }
}

fn parse_source(s: &str) -> PersonalFactSource {
    match s {
        "explicit_statement" => PersonalFactSource::ExplicitStatement,
        "inferred_from_behavior" => PersonalFactSource::InferredFromBehavior,
        "profile_setup" => PersonalFactSource::ProfileSetup,
        "system_observed" => PersonalFactSource::SystemObserved,
        _ => PersonalFactSource::ExplicitStatement, // Default
    }
}

fn parse_timestamp(s: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.timestamp())
        .unwrap_or_else(|_| chrono::Utc::now().timestamp())
}

/// Submission format for server
#[derive(Debug, Serialize)]
struct FactSubmission {
    category: String,
    key: String,
    value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<f32>,
}

impl From<&PersonalFact> for FactSubmission {
    fn from(fact: &PersonalFact) -> Self {
        let category = serde_json::to_string(&fact.category)
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        let source = serde_json::to_string(&fact.source)
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();

        Self {
            category,
            key: fact.key.clone(),
            value: fact.value.clone(),
            context: fact.context.clone(),
            source: Some(source),
            confidence: Some(fact.confidence),
        }
    }
}

/// Sync request to server
#[derive(Debug, Serialize)]
struct SyncRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    since: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    client_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    facts: Option<Vec<FactSubmission>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    feedback: Option<Vec<FeedbackSubmission>>,
}

#[derive(Debug, Serialize)]
struct FeedbackSubmission {
    fact_id: String,
    is_reinforcement: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<String>,
}

impl From<&PersonalFactFeedback> for FeedbackSubmission {
    fn from(fb: &PersonalFactFeedback) -> Self {
        Self {
            fact_id: fb.fact_id.clone(),
            is_reinforcement: fb.is_reinforcement,
            context: fb.context.clone(),
        }
    }
}

/// Sync response from server
#[derive(Debug, Deserialize)]
struct SyncResponse {
    facts: Vec<ServerFact>,
    sync_timestamp: String,
    has_more: bool,
    #[serde(default)]
    stats: SyncStats,
}

#[derive(Debug, Default, Deserialize)]
struct SyncStats {
    facts_received: i32,
    facts_sent: i32,
    feedback_sent: i32,
}

/// Result of a sync operation
#[derive(Debug)]
pub struct SyncResult {
    /// Facts received from server.
    pub facts: Vec<PersonalFact>,
    /// Server sync timestamp.
    pub sync_timestamp: String,
    /// Whether more facts are available.
    pub has_more: bool,
    /// Number of facts received.
    pub facts_received: i32,
    /// Number of facts sent.
    pub facts_sent: i32,
    /// Number of feedback reports sent.
    pub feedback_sent: i32,
}

impl PersonalKnowledgeApiClient {
    /// Create a new API client
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            auth_token: None,
        }
    }

    /// Set authentication token
    pub fn set_auth_token(&mut self, token: String) {
        self.auth_token = Some(token);
    }

    /// Check if client is authenticated
    pub fn is_authenticated(&self) -> bool {
        self.auth_token.is_some()
    }

    /// Build request with auth headers
    fn build_request(&self, method: reqwest::Method, endpoint: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url, endpoint);
        let mut req = self.client.request(method, &url);

        if let Some(ref token) = self.auth_token {
            req = req.header("Authorization", format!("Bearer {}", token));
        }

        req.header("Content-Type", "application/json")
    }

    /// Sync personal facts with server
    pub async fn sync(
        &self,
        since: Option<&str>,
        client_id: Option<&str>,
        facts: &[PersonalFact],
        feedback: &[PersonalFactFeedback],
        min_confidence: f32,
        limit: i32,
    ) -> Result<SyncResult> {
        let fact_submissions: Vec<FactSubmission> = facts
            .iter()
            .filter(|f| !f.local_only) // Never sync local-only facts
            .map(FactSubmission::from)
            .collect();

        let feedback_submissions: Vec<FeedbackSubmission> =
            feedback.iter().map(FeedbackSubmission::from).collect();

        let request = SyncRequest {
            since: since.map(String::from),
            client_id: client_id.map(String::from),
            min_confidence: Some(min_confidence),
            limit: Some(limit),
            facts: if fact_submissions.is_empty() {
                None
            } else {
                Some(fact_submissions)
            },
            feedback: if feedback_submissions.is_empty() {
                None
            } else {
                Some(feedback_submissions)
            },
        };

        let response = self
            .build_request(reqwest::Method::POST, "/api/knowledge/personal/sync")
            .json(&request)
            .send()
            .await
            .context("Failed to send sync request")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Sync failed with status {}: {}", status, text));
        }

        let sync_response: SyncResponse = response
            .json()
            .await
            .context("Failed to parse sync response")?;

        Ok(SyncResult {
            facts: sync_response
                .facts
                .into_iter()
                .map(|f| f.into_personal_fact())
                .collect(),
            sync_timestamp: sync_response.sync_timestamp,
            has_more: sync_response.has_more,
            facts_received: sync_response.stats.facts_received,
            facts_sent: sync_response.stats.facts_sent,
            feedback_sent: sync_response.stats.feedback_sent,
        })
    }

    /// Submit a single fact to server
    pub async fn submit_fact(&self, fact: &PersonalFact) -> Result<PersonalFact> {
        if fact.local_only {
            return Err(anyhow!("Cannot submit local-only fact to server"));
        }

        let submission = FactSubmission::from(fact);

        let response = self
            .build_request(reqwest::Method::POST, "/api/knowledge/personal")
            .json(&submission)
            .send()
            .await
            .context("Failed to submit fact")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Submit failed with status {}: {}", status, text));
        }

        #[derive(Deserialize)]
        struct SubmitResponse {
            fact: ServerFact,
        }

        let result: SubmitResponse = response
            .json()
            .await
            .context("Failed to parse submit response")?;

        Ok(result.fact.into_personal_fact())
    }

    /// Reinforce a fact
    pub async fn reinforce_fact(
        &self,
        fact_id: &str,
        context: Option<&str>,
    ) -> Result<PersonalFact> {
        #[derive(Serialize)]
        struct ReinforceRequest {
            #[serde(skip_serializing_if = "Option::is_none")]
            context: Option<String>,
        }

        let request = ReinforceRequest {
            context: context.map(String::from),
        };

        let response = self
            .build_request(
                reqwest::Method::POST,
                &format!("/api/knowledge/personal/{}/reinforce", fact_id),
            )
            .json(&request)
            .send()
            .await
            .context("Failed to reinforce fact")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Reinforce failed with status {}: {}", status, text));
        }

        #[derive(Deserialize)]
        struct ReinforceResponse {
            fact: ServerFact,
        }

        let result: ReinforceResponse = response
            .json()
            .await
            .context("Failed to parse reinforce response")?;

        Ok(result.fact.into_personal_fact())
    }

    /// Contradict a fact
    pub async fn contradict_fact(
        &self,
        fact_id: &str,
        context: Option<&str>,
        reason: Option<&str>,
    ) -> Result<PersonalFact> {
        #[derive(Serialize)]
        struct ContradictRequest {
            #[serde(skip_serializing_if = "Option::is_none")]
            context: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            reason: Option<String>,
        }

        let request = ContradictRequest {
            context: context.map(String::from),
            reason: reason.map(String::from),
        };

        let response = self
            .build_request(
                reqwest::Method::POST,
                &format!("/api/knowledge/personal/{}/contradict", fact_id),
            )
            .json(&request)
            .send()
            .await
            .context("Failed to contradict fact")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!(
                "Contradict failed with status {}: {}",
                status,
                text
            ));
        }

        #[derive(Deserialize)]
        struct ContradictResponse {
            fact: ServerFact,
            #[allow(dead_code)]
            was_deleted: bool,
        }

        let result: ContradictResponse = response
            .json()
            .await
            .context("Failed to parse contradict response")?;

        Ok(result.fact.into_personal_fact())
    }

    /// Get all personal facts from server
    pub async fn get_facts(
        &self,
        category: Option<PersonalFactCategory>,
        search: Option<&str>,
        min_confidence: f32,
        limit: i32,
    ) -> Result<Vec<PersonalFact>> {
        let mut url = format!(
            "/api/knowledge/personal?min_confidence={}&limit={}",
            min_confidence, limit
        );

        if let Some(cat) = category {
            let cat_str = serde_json::to_string(&cat)
                .unwrap_or_default()
                .trim_matches('"')
                .to_string();
            url.push_str(&format!("&category={}", cat_str));
        }

        if let Some(q) = search {
            url.push_str(&format!("&search={}", urlencoding::encode(q)));
        }

        let response = self
            .build_request(reqwest::Method::GET, &url)
            .send()
            .await
            .context("Failed to get facts")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Get facts failed with status {}: {}", status, text));
        }

        #[derive(Deserialize)]
        struct FactsResponse {
            facts: Vec<ServerFact>,
        }

        let result: FactsResponse = response
            .json()
            .await
            .context("Failed to parse facts response")?;

        Ok(result
            .facts
            .into_iter()
            .map(|f| f.into_personal_fact())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = PersonalKnowledgeApiClient::new("https://example.com");
        assert!(!client.is_authenticated());
    }

    #[test]
    fn test_set_auth_token() {
        let mut client = PersonalKnowledgeApiClient::new("https://example.com");
        client.set_auth_token("test_token".to_string());
        assert!(client.is_authenticated());
    }

    #[test]
    fn test_fact_submission_conversion() {
        let fact = PersonalFact::new(
            PersonalFactCategory::Preference,
            "language".to_string(),
            "Rust".to_string(),
            None,
            PersonalFactSource::ExplicitStatement,
            false,
        );

        let submission = FactSubmission::from(&fact);
        assert_eq!(submission.key, "language");
        assert_eq!(submission.value, "Rust");
        assert_eq!(submission.category, "preference");
    }

    #[test]
    fn test_parse_category() {
        assert_eq!(parse_category("identity"), PersonalFactCategory::Identity);
        assert_eq!(
            parse_category("preference"),
            PersonalFactCategory::Preference
        );
        assert_eq!(parse_category("context"), PersonalFactCategory::Context);
        assert_eq!(parse_category("unknown"), PersonalFactCategory::Preference); // Default
    }

    #[test]
    fn test_parse_source() {
        assert_eq!(
            parse_source("explicit_statement"),
            PersonalFactSource::ExplicitStatement
        );
        assert_eq!(
            parse_source("inferred_from_behavior"),
            PersonalFactSource::InferredFromBehavior
        );
        assert_eq!(
            parse_source("unknown"),
            PersonalFactSource::ExplicitStatement
        ); // Default
    }
}
