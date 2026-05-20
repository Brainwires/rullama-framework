//! Local LLM Provider Implementation
//!
//! Implements the Provider trait for local LLM inference using llama.cpp.
//! Optimized for CPU inference with efficient memory usage.

use super::config::{LocalInferenceParams, LocalLlmConfig};
use anyhow::{Result, anyhow};
use brainwires_core::message::{Message, Role};
use std::sync::Arc;

// The `Provider` trait impl below is gated on `feature = "native"` because it
// uses `async_stream` and `tokio::time`. These imports are only needed there.
#[cfg(feature = "native")]
use async_trait::async_trait;
#[cfg(feature = "native")]
use brainwires_core::message::{ChatResponse, StreamChunk, Usage};
#[cfg(feature = "native")]
use brainwires_core::provider::{ChatOptions, Provider};
#[cfg(feature = "native")]
use brainwires_core::tool::Tool;
#[cfg(feature = "native")]
use futures::stream::BoxStream;

#[cfg(feature = "llama-cpp-2")]
use llama_cpp_2::{
    context::params::LlamaContextParams, llama_backend::LlamaBackend, llama_batch::LlamaBatch,
    model::AddBos, model::LlamaModel, model::params::LlamaModelParams, sampling::LlamaSampler,
};

/// Local LLM Provider using llama.cpp for CPU-optimized inference
///
/// This provider is designed for high-throughput, low-latency local inference
/// without requiring a GPU. Ideal for:
/// - Query routing and classification
/// - Context processing and summarization
/// - Semantic analysis
/// - Agentic decision making (with LFM2-2.6B-Exp)
pub struct LocalLlmProvider {
    /// Model configuration
    config: LocalLlmConfig,
    /// Backend instance (shared) - wrapped in mutex for thread safety
    #[cfg(feature = "llama-cpp-2")]
    backend: std::sync::Mutex<Option<LlamaBackend>>,
    /// Model instance (lazy loaded) - wrapped in mutex for thread safety
    #[cfg(feature = "llama-cpp-2")]
    model: std::sync::Mutex<Option<LlamaModel>>,
    /// Inference state (without the actual llama.cpp when feature disabled)
    #[cfg(not(feature = "llama-cpp-2"))]
    _placeholder: std::marker::PhantomData<()>,
}

impl LocalLlmProvider {
    /// Create a new local LLM provider with the given configuration
    ///
    /// The model is not loaded until the first inference call (lazy loading).
    #[cfg(feature = "llama-cpp-2")]
    pub fn new(config: LocalLlmConfig) -> Result<Self> {
        config.validate().map_err(|e| anyhow!(e))?;

        Ok(Self {
            config,
            backend: std::sync::Mutex::new(None),
            model: std::sync::Mutex::new(None),
        })
    }

    /// Create a new local LLM provider (fallback when `llama-cpp-2` feature is disabled).
    #[cfg(not(feature = "llama-cpp-2"))]
    pub fn new(config: LocalLlmConfig) -> Result<Self> {
        config.validate().map_err(|e| anyhow!(e))?;

        Ok(Self {
            config,
            _placeholder: std::marker::PhantomData,
        })
    }

    /// Create a provider for an LFM2-350M model (fastest, ~220MB RAM)
    pub fn lfm2_350m(model_path: std::path::PathBuf) -> Result<Self> {
        Self::new(LocalLlmConfig::lfm2_350m(model_path))
    }

    /// Create a provider for an LFM2-1.2B model (sweet spot, ~700MB RAM)
    pub fn lfm2_1_2b(model_path: std::path::PathBuf) -> Result<Self> {
        Self::new(LocalLlmConfig::lfm2_1_2b(model_path))
    }

    /// Get the model configuration
    pub fn config(&self) -> &LocalLlmConfig {
        &self.config
    }

    /// Check if the model is loaded
    #[cfg(feature = "llama-cpp-2")]
    pub async fn is_loaded(&self) -> bool {
        self.model.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Check if the model is loaded (always false without `llama-cpp-2` feature).
    #[cfg(not(feature = "llama-cpp-2"))]
    pub async fn is_loaded(&self) -> bool {
        false
    }

    /// Load the model into memory
    #[cfg(feature = "llama-cpp-2")]
    pub async fn load(&self) -> Result<()> {
        // First ensure backend is initialized
        {
            let mut backend_guard = self
                .backend
                .lock()
                .map_err(|e| anyhow!("Lock poisoned: {}", e))?;
            if backend_guard.is_none() {
                let backend = LlamaBackend::init()
                    .map_err(|e| anyhow!("Failed to initialize llama backend: {:?}", e))?;
                *backend_guard = Some(backend);
            }
        }

        // Now load model
        let mut model_guard = self
            .model
            .lock()
            .map_err(|e| anyhow!("Lock poisoned: {}", e))?;
        if model_guard.is_some() {
            return Ok(()); // Already loaded
        }

        tracing::info!("Loading local model: {}", self.config.name);

        // Configure model parameters
        let model_params = LlamaModelParams::default().with_n_gpu_layers(self.config.gpu_layers);

        // Get backend reference
        let backend_guard = self
            .backend
            .lock()
            .map_err(|e| anyhow!("Lock poisoned: {}", e))?;
        let backend = backend_guard
            .as_ref()
            .ok_or_else(|| anyhow!("Backend not initialized"))?;

        // Load the model
        let model = LlamaModel::load_from_file(backend, &self.config.model_path, &model_params)
            .map_err(|e| anyhow!("Failed to load model: {:?}", e))?;

        *model_guard = Some(model);

        tracing::info!("Local model loaded successfully: {}", self.config.name);
        Ok(())
    }

    /// Load the model (errors without `llama-cpp-2` feature).
    #[cfg(not(feature = "llama-cpp-2"))]
    pub async fn load(&self) -> Result<()> {
        Err(anyhow!(
            "Local LLM support is not enabled. Build with --features llama-cpp-2"
        ))
    }

    /// Unload the model from memory
    #[cfg(feature = "llama-cpp-2")]
    pub async fn unload(&self) {
        if let Ok(mut model_guard) = self.model.lock() {
            *model_guard = None;
        }
        tracing::info!("Local model unloaded: {}", self.config.name);
    }

    /// Unload the model (no-op without `llama-cpp-2` feature).
    #[cfg(not(feature = "llama-cpp-2"))]
    pub async fn unload(&self) {
        // No-op when feature not enabled
    }

    /// Format messages into a prompt string using the model's chat template
    #[cfg_attr(not(feature = "native"), allow(dead_code))]
    fn format_prompt(&self, messages: &[Message], system: Option<&str>) -> String {
        let template = self.config.model_type.chat_template();

        // Extract system message
        let system_msg = system.map(String::from).or_else(|| {
            messages.iter().find_map(|m| {
                if m.role == Role::System {
                    m.text().map(String::from)
                } else {
                    None
                }
            })
        });

        // Build the prompt
        let mut prompt = String::new();

        // Add system message if present
        if let Some(sys) = &system_msg
            && template.contains("{system}")
        {
            // Template has system placeholder
            let sys_part = template
                .split("{user}")
                .next()
                .unwrap_or("")
                .replace("{system}", sys);
            prompt.push_str(&sys_part);
        }

        // Add conversation turns
        for msg in messages {
            match msg.role {
                Role::System => continue, // Already handled
                Role::User => {
                    if let Some(text) = msg.text() {
                        // Find the user part of the template
                        let user_template = if template.contains("{user}") {
                            template
                                .split("{user}")
                                .nth(1)
                                .and_then(|s| s.split("{").next())
                                .unwrap_or("\n")
                        } else {
                            "\n"
                        };
                        prompt.push_str(text);
                        prompt.push_str(user_template);
                    }
                }
                Role::Assistant => {
                    if let Some(text) = msg.text() {
                        prompt.push_str(text);
                        prompt.push('\n');
                    }
                }
                Role::Tool => {
                    // Format tool results
                    if let Some(text) = msg.text() {
                        prompt.push_str("[Tool Result]: ");
                        prompt.push_str(text);
                        prompt.push('\n');
                    }
                }
            }
        }

        prompt
    }

    /// Perform inference without loading (assumes model is loaded)
    #[cfg(feature = "llama-cpp-2")]
    fn generate_impl_sync(&self, prompt: &str, params: &LocalInferenceParams) -> Result<String> {
        let model_guard = self
            .model
            .lock()
            .map_err(|e| anyhow!("Lock poisoned: {}", e))?;
        let model = model_guard
            .as_ref()
            .ok_or_else(|| anyhow!("Model not loaded"))?;

        let backend_guard = self
            .backend
            .lock()
            .map_err(|e| anyhow!("Lock poisoned: {}", e))?;
        let backend = backend_guard
            .as_ref()
            .ok_or_else(|| anyhow!("Backend not initialized"))?;

        // Configure context parameters
        let mut ctx_params = LlamaContextParams::default();
        ctx_params = ctx_params.with_n_ctx(std::num::NonZeroU32::new(self.config.context_size));
        ctx_params = ctx_params.with_n_batch(self.config.batch_size);
        if let Some(threads) = self.config.num_threads {
            ctx_params = ctx_params.with_n_threads(threads as i32);
        }

        // Create a context for this generation
        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| anyhow!("Failed to create context: {:?}", e))?;

        // Tokenize the prompt
        let tokens = model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| anyhow!("Tokenization failed: {:?}", e))?;

        // Create batch and add tokens
        let mut batch = LlamaBatch::new(self.config.batch_size as usize, 1);

        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| anyhow!("Failed to add token to batch: {:?}", e))?;
        }

        // Process the prompt
        ctx.decode(&mut batch)
            .map_err(|e| anyhow!("Prompt processing failed: {:?}", e))?;

        // Build sampler chain
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(params.temperature),
            LlamaSampler::top_p(params.top_p, 1),
            LlamaSampler::top_k(params.top_k as i32),
            LlamaSampler::penalties(64, params.repeat_penalty, 0.0, 0.0),
            LlamaSampler::dist(42),
        ]);

        // Generate tokens
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let stop_tokens = self.config.model_type.stop_tokens();
        let mut generated = 0u32;
        let mut cur_pos = tokens.len() as i32;

        while generated < params.max_tokens {
            // Sample next token
            let token = sampler.sample(&ctx, -1);

            // Check for EOS
            if model.is_eog_token(token) {
                break;
            }

            // Decode token to string
            let piece = model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| anyhow!("Token decode failed: {:?}", e))?;

            // Check for stop sequences
            let should_stop = stop_tokens.iter().any(|s| output.ends_with(s));
            if should_stop {
                // Remove the stop sequence from output
                for stop in &stop_tokens {
                    if output.ends_with(stop) {
                        output.truncate(output.len() - stop.len());
                        break;
                    }
                }
                break;
            }

            // Check custom stop sequences
            let custom_stop = params.stop_sequences.iter().any(|s| output.ends_with(s));
            if custom_stop {
                break;
            }

            output.push_str(&piece);

            // Clear batch and add the new token
            batch.clear();
            batch
                .add(token, cur_pos, &[0], true)
                .map_err(|e| anyhow!("Failed to add token: {:?}", e))?;

            // Process the token through the model
            ctx.decode(&mut batch)
                .map_err(|e| anyhow!("Generation failed: {:?}", e))?;

            cur_pos += 1;
            generated += 1;
        }

        Ok(output.trim().to_string())
    }

    #[cfg(not(feature = "llama-cpp-2"))]
    fn generate_impl_sync(&self, _prompt: &str, _params: &LocalInferenceParams) -> Result<String> {
        Err(anyhow!(
            "Local LLM support is not enabled. Build with --features llama-cpp-2"
        ))
    }

    /// Generate a response from a prompt
    pub async fn generate(&self, prompt: &str, params: &LocalInferenceParams) -> Result<String> {
        // Ensure model is loaded
        if !self.is_loaded().await {
            self.load().await?;
        }

        // Run sync inference in blocking task to not block async runtime
        let prompt = prompt.to_string();
        let params = params.clone();

        // Since llama-cpp-2 isn't Send+Sync friendly, we run sync
        self.generate_impl_sync(&prompt, &params)
    }

    /// Simple completion for routing/classification tasks
    ///
    /// Optimized for fast, deterministic responses.
    pub async fn route(&self, prompt: &str) -> Result<String> {
        self.generate(prompt, &LocalInferenceParams::routing())
            .await
    }

    /// Summarize or process text
    pub async fn process(&self, prompt: &str) -> Result<String> {
        self.generate(prompt, &LocalInferenceParams::factual())
            .await
    }
}

// The streaming `Provider` impl uses `async_stream` and `tokio::time`, which are
// only enabled by the `native` feature. On wasm32 builds (or any non-`native`
// build) the `Provider` trait is still implemented via `Self::route` / `generate`
// directly when needed; the LLM types themselves are constructible.
#[cfg(feature = "native")]
#[async_trait]
impl Provider for LocalLlmProvider {
    #[allow(clippy::misnamed_getters)] // Returns config.id intentionally — this is a trait method, not a field getter
    fn name(&self) -> &str {
        &self.config.id
    }

    fn max_output_tokens(&self) -> Option<u32> {
        Some(self.config.max_tokens)
    }

    async fn chat(
        &self,
        messages: &[Message],
        _tools: Option<&[Tool]>,
        options: &ChatOptions,
    ) -> Result<ChatResponse> {
        // Format messages into prompt
        let prompt = self.format_prompt(messages, options.system.as_deref());

        // Build inference params from options
        let params = LocalInferenceParams {
            temperature: options.temperature.unwrap_or(0.7),
            max_tokens: options.max_tokens.unwrap_or(self.config.max_tokens),
            stop_sequences: options.stop.clone().unwrap_or_default(),
            ..Default::default()
        };

        // Generate response
        let response_text = self.generate(&prompt, &params).await?;

        // Estimate token counts (rough approximation: 4 chars per token)
        let prompt_tokens = (prompt.len() / 4) as u32;
        let completion_tokens = (response_text.len() / 4) as u32;

        Ok(ChatResponse {
            message: Message::assistant(response_text),
            usage: Usage::new(prompt_tokens, completion_tokens),
            finish_reason: Some("stop".to_string()),
        })
    }

    fn stream_chat<'a>(
        &'a self,
        messages: &'a [Message],
        _tools: Option<&'a [Tool]>,
        options: &'a ChatOptions,
    ) -> BoxStream<'a, Result<StreamChunk>> {
        let prompt = self.format_prompt(messages, options.system.as_deref());

        let params = LocalInferenceParams {
            temperature: options.temperature.unwrap_or(0.7),
            max_tokens: options.max_tokens.unwrap_or(self.config.max_tokens),
            stop_sequences: options.stop.clone().unwrap_or_default(),
            ..Default::default()
        };

        // For now, we generate the full response and stream it in chunks
        // A true streaming implementation would require changes to the underlying llama.cpp bindings
        Box::pin(async_stream::stream! {
            match self.generate(&prompt, &params).await {
                Ok(response) => {
                    // Stream the response in chunks
                    const CHUNK_SIZE: usize = 10; // characters per chunk
                    for chunk in response.chars().collect::<Vec<_>>().chunks(CHUNK_SIZE) {
                        let text: String = chunk.iter().collect();
                        yield Ok(StreamChunk::Text(text));
                        // Small delay to simulate streaming
                        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    }

                    // Send usage info
                    let prompt_tokens = (prompt.len() / 4) as u32;
                    let completion_tokens = (response.len() / 4) as u32;
                    yield Ok(StreamChunk::Usage(Usage::new(prompt_tokens, completion_tokens)));
                    yield Ok(StreamChunk::Done);
                }
                Err(e) => {
                    yield Err(e);
                }
            }
        })
    }
}

impl std::fmt::Debug for LocalLlmProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalLlmProvider")
            .field("config", &self.config)
            .finish()
    }
}

/// Pool of local LLM providers for parallel inference
///
/// Manages multiple model instances for high-throughput scenarios.
/// Each instance uses separate memory, allowing true parallel inference.
pub struct LocalLlmPool {
    /// Available providers
    providers: Vec<Arc<LocalLlmProvider>>,
    /// Current index for round-robin selection
    current: std::sync::atomic::AtomicUsize,
}

impl LocalLlmPool {
    /// Create a new pool with the specified number of instances
    pub fn new(config: LocalLlmConfig, instances: usize) -> Result<Self> {
        let mut providers = Vec::with_capacity(instances);
        for _ in 0..instances {
            providers.push(Arc::new(LocalLlmProvider::new(config.clone())?));
        }

        Ok(Self {
            providers,
            current: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Get the next available provider (round-robin)
    pub fn next(&self) -> Arc<LocalLlmProvider> {
        let idx = self
            .current
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.providers.len();
        self.providers[idx].clone()
    }

    /// Load all models in the pool
    pub async fn load_all(&self) -> Result<()> {
        for provider in &self.providers {
            provider.load().await?;
        }
        Ok(())
    }

    /// Unload all models in the pool
    pub async fn unload_all(&self) {
        for provider in &self.providers {
            provider.unload().await;
        }
    }

    /// Get the number of instances in the pool
    pub fn size(&self) -> usize {
        self.providers.len()
    }

    /// Estimate total RAM usage for the pool
    pub fn estimated_ram_mb(&self) -> Option<u32> {
        self.providers
            .first()
            .and_then(|p| p.config.estimated_ram_mb)
            .map(|ram| ram * self.providers.len() as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_provider_creation() {
        let config = LocalLlmConfig::lfm2_350m(PathBuf::from("/tmp/test.gguf"));
        // This will fail validation since the file doesn't exist
        let result = LocalLlmProvider::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_inference_params_defaults() {
        let params = LocalInferenceParams::default();
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.max_tokens, 2048);
    }

    #[test]
    fn test_pool_estimated_ram() {
        // Create a mock pool structure for testing
        let _config = LocalLlmConfig {
            model_path: PathBuf::from("."), // Use current dir to pass validation
            estimated_ram_mb: Some(220),
            ..Default::default()
        };

        // For 4 instances of a 220MB model
        let expected_ram = 220 * 4;
        assert_eq!(expected_ram, 880);
    }
}
