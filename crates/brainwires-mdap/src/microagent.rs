//! Microagent - Minimal Context Single-Step Agent
//!
//! Implements the paper's concept of maximal agentic decomposition (MAD)
//! where each agent handles exactly ONE step (m=1).
//!
//! The microagent is designed to:
//! - Execute a single minimal subtask
//! - Use minimal context to reduce error accumulation
//! - Produce consistent, structured outputs
//! - Be suitable for voting/consensus
//!
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::error::{MdapResult, MicroagentError};
use super::red_flags::{OutputFormat, RedFlagConfig, StandardRedFlagValidator};
use super::voting::{FirstToAheadByKVoter, ResponseMetadata, SampledResponse, VoteResult};

/// A minimal subtask that can be executed by a microagent
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Subtask {
    /// Unique identifier for this subtask
    pub id: String,
    /// Human-readable description of what this subtask does
    pub description: String,
    /// Input state/context for this subtask
    pub input_state: serde_json::Value,
    /// Expected output format for validation
    pub expected_output_format: Option<OutputFormat>,
    /// IDs of subtasks this one depends on
    pub depends_on: Vec<String>,
    /// Complexity estimate (0.0-1.0) for cost estimation
    pub complexity_estimate: f32,
    /// Optional specific instructions for this subtask
    pub instructions: Option<String>,
}

impl Subtask {
    /// Create an atomic (non-decomposable) subtask
    pub fn atomic(description: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.into(),
            input_state: serde_json::Value::Null,
            expected_output_format: None,
            depends_on: Vec::new(),
            complexity_estimate: 0.5,
            instructions: None,
        }
    }

    /// Create a subtask with full configuration
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        input_state: serde_json::Value,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            input_state,
            expected_output_format: None,
            depends_on: Vec::new(),
            complexity_estimate: 0.5,
            instructions: None,
        }
    }

    /// Set expected output format
    pub fn with_format(mut self, format: OutputFormat) -> Self {
        self.expected_output_format = Some(format);
        self
    }

    /// Add dependencies
    pub fn depends_on(mut self, deps: Vec<String>) -> Self {
        self.depends_on = deps;
        self
    }

    /// Set complexity estimate
    pub fn with_complexity(mut self, complexity: f32) -> Self {
        self.complexity_estimate = complexity.clamp(0.0, 1.0);
        self
    }

    /// Set specific instructions
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

/// Output from a subtask execution
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubtaskOutput {
    /// The subtask ID this output is for
    pub subtask_id: String,
    /// The output value
    pub value: serde_json::Value,
    /// Optional next state (for stateful subtasks)
    pub next_state: Option<serde_json::Value>,
}

impl SubtaskOutput {
    /// Create a new output
    pub fn new(subtask_id: impl Into<String>, value: serde_json::Value) -> Self {
        Self {
            subtask_id: subtask_id.into(),
            value,
            next_state: None,
        }
    }

    /// Create with next state
    pub fn with_state(mut self, state: serde_json::Value) -> Self {
        self.next_state = Some(state);
        self
    }
}

/// Configuration for microagent execution
#[derive(Clone, Debug)]
pub struct MicroagentConfig {
    /// Maximum output tokens (strict limit for red-flagging, paper: ~750)
    pub max_output_tokens: u32,
    /// Sampling temperature (paper used low temp for consistency)
    pub temperature: f32,
    /// System prompt template
    pub system_prompt_template: String,
    /// Red-flag configuration
    pub red_flag_config: RedFlagConfig,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for MicroagentConfig {
    fn default() -> Self {
        Self {
            max_output_tokens: 750,
            temperature: 0.1, // Low temperature for consistency
            system_prompt_template: MICROAGENT_SYSTEM_PROMPT.to_string(),
            red_flag_config: RedFlagConfig::strict(),
            timeout_ms: 30000,
        }
    }
}

const MICROAGENT_SYSTEM_PROMPT: &str = r#"You are a focused execution agent. Your job is to complete ONE specific subtask.

RULES:
1. Complete ONLY the specified subtask - nothing more, nothing less
2. Output ONLY the requested format - no explanations unless required
3. If you're unsure, output your best answer - do NOT hedge or explain uncertainty
4. Do NOT use phrases like "Wait,", "Actually,", "Let me reconsider" - just give the answer
5. Be concise and direct

Your subtask: {subtask_description}
Expected output format: {output_format}"#;

/// A focused agent for executing a single minimal subtask
///
/// This implements the paper's concept of maximal agentic decomposition (MAD)
/// where each agent handles exactly ONE step (m=1).
pub struct Microagent<P> {
    /// The provider for making LLM calls
    provider: Arc<P>,
    /// The subtask to execute
    subtask: Subtask,
    /// Configuration
    config: MicroagentConfig,
    /// Red-flag validator
    red_flag_validator: StandardRedFlagValidator,
}

/// Trait for providers that can be used with microagents
#[async_trait::async_trait]
pub trait MicroagentProvider: Send + Sync {
    /// Execute a chat completion
    async fn chat(
        &self,
        system: &str,
        user: &str,
        temperature: f32,
        max_tokens: u32,
    ) -> MdapResult<MicroagentResponse>;

    /// Get available tools for intent expression (not execution)
    ///
    /// These tools are described to the LLM so it can express intent to use them.
    /// Actual execution happens after voting consensus, outside the microagent.
    fn available_tools(&self) -> Vec<super::tool_intent::ToolSchema> {
        vec![] // Default: no tools available
    }

    /// Check if this provider has tools available
    fn has_tools(&self) -> bool {
        !self.available_tools().is_empty()
    }
}

/// Response from a microagent provider
#[derive(Clone, Debug)]
pub struct MicroagentResponse {
    /// The response text
    pub text: String,
    /// Number of input tokens
    pub input_tokens: u32,
    /// Number of output tokens
    pub output_tokens: u32,
    /// Finish reason (if available)
    pub finish_reason: Option<String>,
    /// Response time in milliseconds
    pub response_time_ms: u64,
}

impl<P: MicroagentProvider + 'static> Microagent<P> {
    /// Create a new microagent
    pub fn new(provider: Arc<P>, subtask: Subtask, config: MicroagentConfig) -> Self {
        let red_flag_validator = StandardRedFlagValidator::new(
            config.red_flag_config.clone(),
            subtask.expected_output_format.clone(),
        );

        Self {
            provider,
            subtask,
            config,
            red_flag_validator,
        }
    }

    /// Create with default configuration
    pub fn with_defaults(provider: Arc<P>, subtask: Subtask) -> Self {
        Self::new(provider, subtask, MicroagentConfig::default())
    }

    /// Execute the subtask once (single sample for voting)
    pub async fn execute_once(
        &self,
        input: &serde_json::Value,
    ) -> MdapResult<SampledResponse<SubtaskOutput>> {
        let system_prompt = self.build_system_prompt();
        let user_prompt = self.build_user_prompt(input);

        let start = std::time::Instant::now();

        let response = self
            .provider
            .chat(
                &system_prompt,
                &user_prompt,
                self.config.temperature,
                self.config.max_output_tokens,
            )
            .await
            .map_err(|e| MicroagentError::ProviderError(e.to_string()))?;

        let elapsed = start.elapsed();

        let metadata = ResponseMetadata {
            token_count: response.output_tokens,
            response_time_ms: elapsed.as_millis() as u64,
            format_valid: true, // Will be validated by red-flag checker
            finish_reason: response.finish_reason,
            model: None,
        };

        let output = self.parse_output(&response.text)?;

        // Extract confidence from response (CISC paper: arxiv:2502.06233v1)
        let confidence = extract_response_confidence(&response.text, &metadata);

        Ok(SampledResponse {
            value: output,
            metadata,
            raw_response: response.text,
            confidence,
        })
    }

    /// Execute subtask with voting for error correction
    pub async fn execute_with_voting(
        &self,
        input: &serde_json::Value,
        voter: &FirstToAheadByKVoter,
    ) -> MdapResult<VoteResult<SubtaskOutput>> {
        let input = input.clone();
        let provider = self.provider.clone();
        let subtask = self.subtask.clone();
        let config = self.config.clone();

        // Create a closure that captures necessary state
        voter
            .vote(
                || {
                    let provider = provider.clone();
                    let subtask = subtask.clone();
                    let config = config.clone();
                    let input = input.clone();

                    async move {
                        let agent = Microagent::new(provider, subtask, config);
                        agent.execute_once(&input).await
                    }
                },
                &self.red_flag_validator,
                |output: &SubtaskOutput| {
                    // Use the value as the key for voting
                    serde_json::to_string(&output.value).unwrap_or_default()
                },
            )
            .await
    }

    /// Build the system prompt
    fn build_system_prompt(&self) -> String {
        let format_desc = self
            .subtask
            .expected_output_format
            .as_ref()
            .map(|f| f.description())
            .unwrap_or_else(|| "Plain text response".to_string());

        self.config
            .system_prompt_template
            .replace("{subtask_description}", &self.subtask.description)
            .replace("{output_format}", &format_desc)
    }

    /// Build the user prompt
    fn build_user_prompt(&self, input: &serde_json::Value) -> String {
        let mut prompt = String::new();

        // Add specific instructions if provided
        if let Some(ref instructions) = self.subtask.instructions {
            prompt.push_str("Instructions:\n");
            prompt.push_str(instructions);
            prompt.push_str("\n\n");
        }

        // Add input state
        prompt.push_str("Input:\n");
        prompt.push_str(&serde_json::to_string_pretty(input).unwrap_or_default());
        prompt.push_str("\n\n");

        prompt.push_str("Provide your output:");

        prompt
    }

    /// Parse the output from the response
    fn parse_output(&self, response: &str) -> MdapResult<SubtaskOutput> {
        let trimmed = response.trim();

        // Try to parse as JSON first
        let value = if let Ok(json) = serde_json::from_str::<serde_json::Value>(trimmed) {
            json
        } else {
            // Fall back to string value
            serde_json::Value::String(trimmed.to_string())
        };

        Ok(SubtaskOutput::new(self.subtask.id.clone(), value))
    }

    /// Get the subtask
    pub fn subtask(&self) -> &Subtask {
        &self.subtask
    }

    /// Get the configuration
    pub fn config(&self) -> &MicroagentConfig {
        &self.config
    }
}

/// Builder for microagent configuration
pub struct MicroagentConfigBuilder {
    config: MicroagentConfig,
}

impl Default for MicroagentConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MicroagentConfigBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: MicroagentConfig::default(),
        }
    }

    /// Set the maximum output tokens.
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.config.max_output_tokens = tokens;
        self
    }

    /// Set the temperature for sampling (clamped to 0.0-2.0).
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp.clamp(0.0, 2.0);
        self
    }

    /// Set the system prompt template.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt_template = prompt.into();
        self
    }

    /// Set the red-flag validation configuration.
    pub fn red_flag_config(mut self, config: RedFlagConfig) -> Self {
        self.config.red_flag_config = config;
        self
    }

    /// Set the execution timeout in milliseconds.
    pub fn timeout_ms(mut self, timeout: u64) -> Self {
        self.config.timeout_ms = timeout;
        self
    }

    /// Build the microagent configuration.
    pub fn build(self) -> MicroagentConfig {
        self.config
    }
}

/// Extract confidence from a microagent response (CISC paper: arxiv:2502.06233v1)
///
/// Analyzes the response text and metadata to determine confidence level.
/// This replaces the hardcoded 0.75 with dynamic extraction based on:
/// 1. Finish reason (completion status)
/// 2. Response length (too short/long indicates issues)
/// 3. Language patterns (hedging, self-correction, assertions)
fn extract_response_confidence(text: &str, metadata: &ResponseMetadata) -> f64 {
    let mut confidence = 0.75; // Start with baseline

    // 1. Adjust based on finish reason
    match metadata.finish_reason.as_deref() {
        Some("stop") | Some("end_turn") => confidence += 0.10,
        Some("length") | Some("max_tokens") => confidence -= 0.25, // Truncated
        _ => {}
    }

    // 2. Adjust based on response length
    let token_estimate = metadata.token_count as usize;
    if token_estimate < 10 {
        confidence -= 0.20; // Very short, possibly incomplete
    } else if token_estimate > 700 {
        confidence -= 0.15; // Near token limit, possibly verbose
    }

    // 3. Check for hedging/uncertainty patterns (reduces confidence)
    let text_lower = text.to_lowercase();
    let hedging_patterns = [
        "i'm not sure",
        "i think",
        "possibly",
        "might be",
        "could be",
        "probably",
        "perhaps",
        "maybe",
        "unclear",
        "i guess",
    ];
    let hedging_count = hedging_patterns
        .iter()
        .filter(|p| text_lower.contains(*p))
        .count();
    confidence -= (hedging_count as f64 * 0.08).min(0.30);

    // 4. Check for self-correction patterns (reduces confidence more)
    let self_correction_patterns = [
        "wait,",
        "actually,",
        "let me reconsider",
        "i made a mistake",
        "correction:",
        "i was wrong",
        "on second thought",
    ];
    let correction_count = self_correction_patterns
        .iter()
        .filter(|p| text_lower.contains(*p))
        .count();
    confidence -= (correction_count as f64 * 0.15).min(0.30);

    // 5. Check for confident assertion patterns (slight boost)
    let confident_patterns = [
        "the answer is",
        "definitely",
        "certainly",
        "clearly",
        "the solution is",
        "this will work",
    ];
    let confident_count = confident_patterns
        .iter()
        .filter(|p| text_lower.contains(*p))
        .count();
    confidence += (confident_count as f64 * 0.05).min(0.10);

    // 6. Check format validity from metadata
    if !metadata.format_valid {
        confidence -= 0.20;
    }

    confidence.clamp(0.1, 0.99)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockProvider {
        response: String,
    }

    #[async_trait::async_trait]
    impl MicroagentProvider for MockProvider {
        async fn chat(
            &self,
            _system: &str,
            _user: &str,
            _temperature: f32,
            _max_tokens: u32,
        ) -> MdapResult<MicroagentResponse> {
            Ok(MicroagentResponse {
                text: self.response.clone(),
                input_tokens: 100,
                output_tokens: 50,
                finish_reason: Some("stop".to_string()),
                response_time_ms: 100,
            })
        }
    }

    #[test]
    fn test_subtask_creation() {
        let subtask = Subtask::atomic("Calculate 2 + 2");
        assert_eq!(subtask.description, "Calculate 2 + 2");
        assert!(subtask.depends_on.is_empty());
    }

    #[test]
    fn test_subtask_builder() {
        let subtask = Subtask::new("task_1", "Add numbers", serde_json::json!({"a": 1, "b": 2}))
            .with_complexity(0.3)
            .with_format(OutputFormat::Json)
            .depends_on(vec!["task_0".to_string()]);

        assert_eq!(subtask.id, "task_1");
        assert_eq!(subtask.complexity_estimate, 0.3);
        assert_eq!(subtask.depends_on, vec!["task_0"]);
    }

    #[test]
    fn test_subtask_output() {
        let output = SubtaskOutput::new("task_1", serde_json::json!(42))
            .with_state(serde_json::json!({"done": true}));

        assert_eq!(output.subtask_id, "task_1");
        assert_eq!(output.value, serde_json::json!(42));
        assert!(output.next_state.is_some());
    }

    #[test]
    fn test_microagent_config_builder() {
        let config = MicroagentConfigBuilder::new()
            .max_output_tokens(500)
            .temperature(0.5)
            .timeout_ms(60000)
            .build();

        assert_eq!(config.max_output_tokens, 500);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.timeout_ms, 60000);
    }

    #[tokio::test]
    async fn test_microagent_execute_once() {
        let provider = Arc::new(MockProvider {
            response: "42".to_string(),
        });

        let subtask = Subtask::atomic("Calculate 2 + 2");
        let agent = Microagent::with_defaults(provider, subtask);

        let result = agent
            .execute_once(&serde_json::json!({"expression": "2 + 2"}))
            .await
            .unwrap();

        // "42" is valid JSON, so it parses as a number
        assert_eq!(result.value.value, serde_json::json!(42));
    }

    #[tokio::test]
    async fn test_microagent_parse_json() {
        let provider = Arc::new(MockProvider {
            response: r#"{"result": 42}"#.to_string(),
        });

        let subtask = Subtask::atomic("Return JSON").with_format(OutputFormat::Json);
        let agent = Microagent::with_defaults(provider, subtask);

        let result = agent.execute_once(&serde_json::Value::Null).await.unwrap();

        assert!(result.value.value.is_object());
        assert_eq!(result.value.value["result"], 42);
    }

    #[test]
    fn test_system_prompt_generation() {
        let provider = Arc::new(MockProvider {
            response: "".to_string(),
        });

        let subtask = Subtask::atomic("Test task").with_format(OutputFormat::Json);
        let agent = Microagent::with_defaults(provider, subtask);

        let prompt = agent.build_system_prompt();
        assert!(prompt.contains("Test task"));
        assert!(prompt.contains("JSON"));
    }

    #[test]
    fn test_user_prompt_generation() {
        let provider = Arc::new(MockProvider {
            response: "".to_string(),
        });

        let subtask = Subtask::atomic("Test task").with_instructions("Be precise");
        let agent = Microagent::with_defaults(provider, subtask);

        let prompt = agent.build_user_prompt(&serde_json::json!({"x": 1}));
        assert!(prompt.contains("Be precise"));
        assert!(prompt.contains("\"x\": 1"));
    }
}
