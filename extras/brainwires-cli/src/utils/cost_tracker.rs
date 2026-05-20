//! Cost Tracking System
//!
//! Tracks API usage costs across providers and models.
//! Provides visibility into API spend with budget enforcement.

use anyhow::Result;
use chrono::{DateTime, Datelike, Duration, Local, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Model pricing information (per 1000 tokens in USD)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Cost per 1000 input tokens
    pub input_cost_per_1k: f64,
    /// Cost per 1000 output tokens
    pub output_cost_per_1k: f64,
}

impl ModelPricing {
    pub fn new(input_cost_per_1k: f64, output_cost_per_1k: f64) -> Self {
        Self {
            input_cost_per_1k,
            output_cost_per_1k,
        }
    }

    /// Calculate cost for given token counts
    pub fn calculate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        let input_cost = (input_tokens as f64 / 1000.0) * self.input_cost_per_1k;
        let output_cost = (output_tokens as f64 / 1000.0) * self.output_cost_per_1k;
        input_cost + output_cost
    }
}

/// Default pricing for common models (as of 2025)
// TODO: drive pricing from ModelRegistry::fetch_models() instead of hard-coding
fn default_model_pricing() -> HashMap<String, ModelPricing> {
    let mut pricing = HashMap::new();

    // Anthropic Claude models
    pricing.insert("claude-3-opus".to_string(), ModelPricing::new(0.015, 0.075));
    pricing.insert(
        "claude-3-sonnet".to_string(),
        ModelPricing::new(0.003, 0.015),
    );
    pricing.insert(
        "claude-3-haiku".to_string(),
        ModelPricing::new(0.00025, 0.00125),
    );
    pricing.insert(
        "claude-3.5-sonnet".to_string(),
        ModelPricing::new(0.003, 0.015),
    );
    pricing.insert("claude-opus-4".to_string(), ModelPricing::new(0.015, 0.075));
    pricing.insert(
        "claude-haiku-4-5-20251001".to_string(),
        ModelPricing::new(0.00025, 0.00125),
    );
    pricing.insert(
        "claude-sonnet-4-5-20250929".to_string(),
        ModelPricing::new(0.003, 0.015),
    );
    pricing.insert(
        "claude-opus-4-1-20250805".to_string(),
        ModelPricing::new(0.015, 0.075),
    );

    // OpenAI models
    pricing.insert("gpt-4".to_string(), ModelPricing::new(0.03, 0.06));
    pricing.insert("gpt-4-turbo".to_string(), ModelPricing::new(0.01, 0.03));
    pricing.insert(
        "gpt-3.5-turbo".to_string(),
        ModelPricing::new(0.0005, 0.0015),
    );

    pricing
}

/// A single usage event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,
    /// Provider name (e.g., "anthropic", "openai")
    pub provider: String,
    /// Model name
    pub model: String,
    /// Input tokens used
    pub input_tokens: u32,
    /// Output tokens used
    pub output_tokens: u32,
    /// Calculated cost in USD
    pub cost_usd: f64,
    /// Session ID (for grouping)
    pub session_id: Option<String>,
}

/// Aggregated usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageStats {
    /// Total cost in USD
    pub total_cost_usd: f64,
    /// Total input tokens
    pub total_input_tokens: u64,
    /// Total output tokens
    pub total_output_tokens: u64,
    /// Number of API calls
    pub total_calls: u64,
    /// Cost breakdown by model
    pub by_model: HashMap<String, f64>,
    /// Cost breakdown by provider
    pub by_provider: HashMap<String, f64>,
}

impl UsageStats {
    pub fn add_event(&mut self, event: &UsageEvent) {
        self.total_cost_usd += event.cost_usd;
        self.total_input_tokens += event.input_tokens as u64;
        self.total_output_tokens += event.output_tokens as u64;
        self.total_calls += 1;

        *self.by_model.entry(event.model.clone()).or_insert(0.0) += event.cost_usd;
        *self
            .by_provider
            .entry(event.provider.clone())
            .or_insert(0.0) += event.cost_usd;
    }
}

/// Budget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Daily budget in USD (None = unlimited)
    pub daily_limit_usd: Option<f64>,
    /// Monthly budget in USD (None = unlimited)
    pub monthly_limit_usd: Option<f64>,
    /// Warning threshold as percentage of budget (0.0 - 1.0)
    pub warning_threshold: f64,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            daily_limit_usd: None,
            monthly_limit_usd: None,
            warning_threshold: 0.80, // Warn at 80% of budget
        }
    }
}

/// Budget status
#[derive(Debug, Clone)]
pub enum BudgetStatus {
    /// Under budget, all good
    Ok,
    /// Approaching budget limit
    Warning { used_pct: f64, limit_type: String },
    /// Budget exceeded
    Exceeded { used_pct: f64, limit_type: String },
}

/// Main cost tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTracker {
    /// Usage events (recent history)
    events: Vec<UsageEvent>,
    /// Model pricing lookup
    pricing: HashMap<String, ModelPricing>,
    /// Budget configuration
    budget: BudgetConfig,
    /// Current session ID
    #[serde(skip)]
    current_session: Option<String>,
    /// Data file path
    #[serde(skip)]
    data_path: Option<PathBuf>,
}

impl Default for CostTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CostTracker {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            pricing: default_model_pricing(),
            budget: BudgetConfig::default(),
            current_session: None,
            data_path: None,
        }
    }

    /// Create with a specific data path
    pub fn with_path(path: PathBuf) -> Self {
        let mut tracker = Self::new();
        tracker.data_path = Some(path);
        tracker
    }

    /// Set the current session ID
    pub fn set_session(&mut self, session_id: impl Into<String>) {
        self.current_session = Some(session_id.into());
    }

    /// Set budget limits
    pub fn set_budget(&mut self, config: BudgetConfig) {
        self.budget = config;
    }

    /// Add custom model pricing
    pub fn add_pricing(&mut self, model: impl Into<String>, pricing: ModelPricing) {
        self.pricing.insert(model.into(), pricing);
    }

    /// Load from file
    pub async fn load() -> Result<Self> {
        let path = Self::default_path()?;
        Self::load_from_path(&path).await
    }

    /// Load from specific path
    pub async fn load_from_path(path: &PathBuf) -> Result<Self> {
        if path.exists() {
            let content = tokio::fs::read_to_string(path).await?;
            let mut tracker: CostTracker = serde_json::from_str(&content)?;
            tracker.data_path = Some(path.clone());
            // Ensure we have default pricing even after load
            for (model, price) in default_model_pricing() {
                tracker.pricing.entry(model).or_insert(price);
            }
            Ok(tracker)
        } else {
            let mut tracker = Self::new();
            tracker.data_path = Some(path.clone());
            Ok(tracker)
        }
    }

    /// Save to file
    pub async fn save(&self) -> Result<()> {
        let path = self.data_path.clone().unwrap_or(Self::default_path()?);

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let content = serde_json::to_string_pretty(self)?;
        tokio::fs::write(&path, content).await?;
        Ok(())
    }

    /// Default data path
    fn default_path() -> Result<PathBuf> {
        let data_dir =
            dirs::data_dir().ok_or_else(|| anyhow::anyhow!("Could not find data directory"))?;
        Ok(data_dir.join("brainwires").join("cost_tracker.json"))
    }

    /// Track a usage event
    pub fn track_usage(
        &mut self,
        provider: &str,
        model: &str,
        input_tokens: u32,
        output_tokens: u32,
    ) {
        let cost = self.calculate_cost(model, input_tokens, output_tokens);

        let event = UsageEvent {
            timestamp: Utc::now(),
            provider: provider.to_string(),
            model: model.to_string(),
            input_tokens,
            output_tokens,
            cost_usd: cost,
            session_id: self.current_session.clone(),
        };

        self.events.push(event);

        // Prune old events (keep last 30 days)
        let cutoff = Utc::now() - Duration::days(30);
        self.events.retain(|e| e.timestamp > cutoff);
    }

    /// Legacy method for backward compatibility
    pub fn track_usage_legacy(&mut self, _provider: &str, _model: &str, _tokens: u32) {
        // Legacy - does nothing, use track_usage with separate input/output counts
    }

    /// Calculate cost for a model and token counts
    pub fn calculate_cost(&self, model: &str, input_tokens: u32, output_tokens: u32) -> f64 {
        // Try exact match first
        if let Some(pricing) = self.pricing.get(model) {
            return pricing.calculate_cost(input_tokens, output_tokens);
        }

        // Try partial match (model names often have versions/suffixes)
        for (model_name, pricing) in &self.pricing {
            if model.contains(model_name) || model_name.contains(model) {
                return pricing.calculate_cost(input_tokens, output_tokens);
            }
        }

        // Default fallback pricing (conservative estimate)
        let fallback = ModelPricing::new(0.01, 0.03);
        fallback.calculate_cost(input_tokens, output_tokens)
    }

    /// Get usage stats for a time period
    pub fn get_stats(&self, period: TimePeriod) -> UsageStats {
        let cutoff = match period {
            TimePeriod::Today => {
                let today = Local::now().date_naive();
                today.and_hms_opt(0, 0, 0).unwrap().and_utc()
            }
            TimePeriod::ThisWeek => {
                let today = Local::now();
                let days_since_monday = today.weekday().num_days_from_monday();
                let monday = today - Duration::days(days_since_monday as i64);
                monday.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc()
            }
            TimePeriod::ThisMonth => {
                let today = Local::now();
                NaiveDate::from_ymd_opt(today.year(), today.month(), 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc()
            }
            TimePeriod::Last30Days => Utc::now() - Duration::days(30),
            TimePeriod::AllTime => DateTime::<Utc>::MIN_UTC,
        };

        let mut stats = UsageStats::default();
        for event in &self.events {
            if event.timestamp >= cutoff {
                stats.add_event(event);
            }
        }
        stats
    }

    /// Get human-readable usage summary
    pub fn get_usage_summary(&self, period: &str) -> String {
        let time_period = match period.to_lowercase().as_str() {
            "today" => TimePeriod::Today,
            "week" | "this_week" => TimePeriod::ThisWeek,
            "month" | "this_month" => TimePeriod::ThisMonth,
            "30days" | "last_30_days" => TimePeriod::Last30Days,
            _ => TimePeriod::Today,
        };

        let stats = self.get_stats(time_period);

        if stats.total_calls == 0 {
            return "No usage data".to_string();
        }

        let mut summary = format!(
            "Usage Summary ({}):\n\
             Total Cost: ${:.4}\n\
             API Calls: {}\n\
             Input Tokens: {}\n\
             Output Tokens: {}\n",
            period,
            stats.total_cost_usd,
            stats.total_calls,
            stats.total_input_tokens,
            stats.total_output_tokens,
        );

        if !stats.by_model.is_empty() {
            summary.push_str("\nBy Model:\n");
            let mut models: Vec<_> = stats.by_model.iter().collect();
            models.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (model, cost) in models {
                summary.push_str(&format!("  {}: ${:.4}\n", model, cost));
            }
        }

        summary
    }

    /// Check budget status
    pub fn check_budget(&self) -> BudgetStatus {
        let today_stats = self.get_stats(TimePeriod::Today);
        let month_stats = self.get_stats(TimePeriod::ThisMonth);

        // Check daily limit
        if let Some(daily_limit) = self.budget.daily_limit_usd {
            let used_pct = today_stats.total_cost_usd / daily_limit;
            if used_pct >= 1.0 {
                return BudgetStatus::Exceeded {
                    used_pct,
                    limit_type: "daily".to_string(),
                };
            }
            if used_pct >= self.budget.warning_threshold {
                return BudgetStatus::Warning {
                    used_pct,
                    limit_type: "daily".to_string(),
                };
            }
        }

        // Check monthly limit
        if let Some(monthly_limit) = self.budget.monthly_limit_usd {
            let used_pct = month_stats.total_cost_usd / monthly_limit;
            if used_pct >= 1.0 {
                return BudgetStatus::Exceeded {
                    used_pct,
                    limit_type: "monthly".to_string(),
                };
            }
            if used_pct >= self.budget.warning_threshold {
                return BudgetStatus::Warning {
                    used_pct,
                    limit_type: "monthly".to_string(),
                };
            }
        }

        BudgetStatus::Ok
    }

    /// Get today's cost
    pub fn today_cost(&self) -> f64 {
        self.get_stats(TimePeriod::Today).total_cost_usd
    }

    /// Get session cost
    pub fn session_cost(&self) -> f64 {
        self.events
            .iter()
            .filter(|e| e.session_id == self.current_session)
            .map(|e| e.cost_usd)
            .sum()
    }
}

/// Time periods for aggregation
#[derive(Debug, Clone, Copy)]
pub enum TimePeriod {
    Today,
    ThisWeek,
    ThisMonth,
    Last30Days,
    AllTime,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_tracker_new() {
        let tracker = CostTracker::new();
        assert!(tracker.events.is_empty());
        assert!(!tracker.pricing.is_empty());
    }

    #[test]
    fn test_track_usage() {
        let mut tracker = CostTracker::new();
        tracker.track_usage("anthropic", "claude-3-sonnet", 1000, 500);

        assert_eq!(tracker.events.len(), 1);
        assert!(tracker.events[0].cost_usd > 0.0);
    }

    #[test]
    fn test_calculate_cost() {
        let tracker = CostTracker::new();

        // Claude 3 Sonnet: $0.003/1K input, $0.015/1K output
        let cost = tracker.calculate_cost("claude-3-sonnet", 1000, 1000);
        assert!((cost - 0.018).abs() < 0.001);
    }

    #[test]
    fn test_usage_stats() {
        let mut tracker = CostTracker::new();
        tracker.track_usage("anthropic", "claude-3-sonnet", 1000, 500);
        tracker.track_usage("anthropic", "claude-3-haiku", 2000, 1000);

        let stats = tracker.get_stats(TimePeriod::Today);
        assert_eq!(stats.total_calls, 2);
        assert_eq!(stats.total_input_tokens, 3000);
    }

    #[test]
    fn test_budget_check() {
        let mut tracker = CostTracker::new();
        tracker.set_budget(BudgetConfig {
            daily_limit_usd: Some(1.0),
            monthly_limit_usd: None,
            warning_threshold: 0.80,
        });

        // Under budget
        tracker.track_usage("anthropic", "claude-3-sonnet", 100, 50);
        assert!(matches!(tracker.check_budget(), BudgetStatus::Ok));
    }

    #[test]
    fn test_usage_summary() {
        let mut tracker = CostTracker::new();
        tracker.track_usage("anthropic", "claude-3-sonnet", 1000, 500);

        let summary = tracker.get_usage_summary("today");
        assert!(summary.contains("Total Cost:"));
        assert!(summary.contains("API Calls: 1"));
    }

    #[tokio::test]
    async fn test_load_nonexistent() {
        // Should return a new tracker if file doesn't exist
        let path = PathBuf::from("/tmp/nonexistent_cost_tracker_test.json");
        let result = CostTracker::load_from_path(&path).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_session_cost() {
        let mut tracker = CostTracker::new();
        tracker.set_session("session-123");
        tracker.track_usage("anthropic", "claude-3-sonnet", 1000, 500);
        tracker.track_usage("anthropic", "claude-3-sonnet", 500, 250);

        let session_cost = tracker.session_cost();
        assert!(session_cost > 0.0);
    }
}
