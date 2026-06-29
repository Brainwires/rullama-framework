//! AI-powered issue investigation.
//!
//! Sends issue details to an AI provider and parses the response to produce
//! a structured [`InvestigationResult`] with affected files, approach, and
//! confidence score.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use brainwires_core::Provider;

use super::forge::{Issue, RepoRef};

/// Result of investigating an issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestigationResult {
    /// Issue that was investigated.
    pub issue_number: u64,
    /// Confidence score (0.0 to 1.0) that the analysis is correct.
    pub confidence: f64,
    /// Summary of the issue and proposed fix approach.
    pub summary: String,
    /// Files likely affected by the issue.
    pub affected_files: Vec<String>,
    /// Proposed approach to fix the issue.
    pub approach: String,
    /// Estimated complexity (low, medium, high).
    pub complexity: String,
}

/// Investigates issues by prompting an AI provider to analyze the problem,
/// identify affected files, and propose a fix approach.
pub struct IssueInvestigator {
    provider: Arc<dyn Provider>,
}

impl IssueInvestigator {
    /// Create a new issue investigator with the given AI provider.
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        Self { provider }
    }

    /// Investigate an issue and produce an analysis.
    pub async fn investigate(
        &self,
        issue: &Issue,
        _repo: &RepoRef,
    ) -> anyhow::Result<InvestigationResult> {
        let prompt = format!(
            "Analyze this issue and determine what needs to be fixed:\n\n\
             Title: {}\n\
             Body:\n{}\n\
             Labels: {}\n\n\
             Provide:\n\
             1. A brief summary of the problem\n\
             2. Which files are likely affected\n\
             3. A proposed approach to fix it\n\
             4. Estimated complexity (low/medium/high)\n\
             5. Your confidence (0.0-1.0) that this analysis is correct",
            issue.title,
            issue.body,
            issue.labels.join(", ")
        );

        let messages = vec![brainwires_core::Message::user(prompt)];
        let options = brainwires_core::ChatOptions::default();

        let response = self.provider.chat(&messages, None, &options).await?;
        let content = response.message.text().unwrap_or_default().to_string();

        // Parse the AI response (best-effort extraction)
        Ok(InvestigationResult {
            issue_number: issue.number,
            confidence: extract_confidence(&content),
            summary: content.clone(),
            affected_files: extract_files(&content),
            approach: content.clone(),
            complexity: extract_complexity(&content),
        })
    }
}

fn extract_confidence(text: &str) -> f64 {
    // Look for patterns like "confidence: 0.8" or "0.85 confidence"
    for line in text.lines() {
        let lower = line.to_lowercase();
        if lower.contains("confidence") {
            for word in lower.split_whitespace() {
                if let Ok(val) = word
                    .trim_matches(|c: char| !c.is_ascii_digit() && c != '.')
                    .parse::<f64>()
                    && (0.0..=1.0).contains(&val)
                {
                    return val;
                }
            }
        }
    }
    0.5 // Default moderate confidence
}

fn extract_files(text: &str) -> Vec<String> {
    let mut files = Vec::new();
    for line in text.lines() {
        let trimmed = line
            .trim()
            .trim_start_matches("- ")
            .trim_start_matches("* ");
        if (trimmed.contains('/') || trimmed.contains('.')) && trimmed.ends_with(".rs")
            || trimmed.ends_with(".ts")
            || trimmed.ends_with(".py")
            || trimmed.ends_with(".js")
            || trimmed.ends_with(".toml")
        {
            // Extract the file path
            let path = trimmed
                .split_whitespace()
                .find(|w| w.contains('.'))
                .unwrap_or(trimmed)
                .trim_matches('`')
                .to_string();
            if !path.is_empty() {
                files.push(path);
            }
        }
    }
    files
}

fn extract_complexity(text: &str) -> String {
    let lower = text.to_lowercase();
    if lower.contains("high complexity") || lower.contains("complexity: high") {
        "high".to_string()
    } else if lower.contains("low complexity") || lower.contains("complexity: low") {
        "low".to_string()
    } else {
        "medium".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_confidence_from_text() {
        assert!((extract_confidence("My confidence: 0.85") - 0.85).abs() < f64::EPSILON);
        assert!((extract_confidence("I have 0.9 confidence in this") - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn extract_confidence_default_when_missing() {
        assert!((extract_confidence("no score here") - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn extract_files_finds_file_paths() {
        let text = "Affected files:\n- src/main.rs\n- src/lib.rs\n- README.md\n";
        let files = extract_files(text);
        assert!(files.contains(&"src/main.rs".to_string()));
        assert!(files.contains(&"src/lib.rs".to_string()));
    }

    #[test]
    fn extract_complexity_variants() {
        assert_eq!(extract_complexity("This is high complexity work"), "high");
        assert_eq!(extract_complexity("Complexity: high"), "high");
        assert_eq!(extract_complexity("This is low complexity"), "low");
        assert_eq!(extract_complexity("Complexity: low"), "low");
        assert_eq!(extract_complexity("Something else entirely"), "medium");
    }
}
