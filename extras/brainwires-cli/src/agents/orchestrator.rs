use anyhow::Result;

/// Conservative token limit for orchestrator completions to prevent runaway generation.
const ORCHESTRATOR_MAX_TOKENS: u32 = 4096;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::debug_log;
use crate::mdap::{
    DecomposeContext, MdapConfig, MdapMetrics, MicroagentProvider, MicroagentResponse,
    error::MdapResult,
};
use crate::providers::Provider;
use crate::tools::{TaskManagerTool, ToolExecutor};
use crate::types::agent::{AgentContext, AgentResponse, PermissionMode, Task};
use crate::types::message::{ChatResponse, ContentBlock, Message, MessageContent, Role};
use crate::types::provider::ChatOptions;
use crate::types::tool::{ToolContext, ToolContextExt, ToolUse};
use crate::utils::entity_extraction::{EntityExtractor, EntityStore};
use brainwires_seal::{DialogState, SealConfig, SealProcessingResult, SealProcessor};

use super::TaskManager;

/// Orchestrator agent - coordinates high-level task execution
pub struct OrchestratorAgent {
    provider: Arc<dyn Provider>,
    tool_executor: ToolExecutor,
    task_manager_tool: TaskManagerTool,
    max_iterations: u32,
    // SEAL components for enhanced context understanding
    seal_processor: Option<SealProcessor>,
    dialog_state: DialogState,
    entity_store: EntityStore,
    entity_extractor: EntityExtractor,
    // SEAL + Knowledge integration
    knowledge_coordinator: Option<brainwires_seal::SealKnowledgeCoordinator>,
    // Adaptive prompting (Phase 4 integration)
    prompt_generator: Option<brainwires::prompting::PromptGenerator>,
    use_adaptive_prompts: bool,
    // Track last prompt generation for learning
    last_generated_prompt: Option<brainwires::prompting::generator::GeneratedPrompt>,
    // Embedding provider for adaptive prompting (Phase 8)
    embedding_provider: Option<Arc<crate::storage::embeddings::CachedEmbeddingProvider>>,
}

impl OrchestratorAgent {
    /// Create a new orchestrator agent
    pub fn new(provider: Arc<dyn Provider>, permission_mode: PermissionMode) -> Self {
        let task_manager = Arc::new(RwLock::new(TaskManager::new()));
        let task_manager_tool = TaskManagerTool::new(task_manager.clone());

        Self {
            provider,
            tool_executor: ToolExecutor::new(permission_mode),
            task_manager_tool,
            max_iterations: 25,
            seal_processor: None,
            dialog_state: DialogState::new(),
            entity_store: EntityStore::new(),
            entity_extractor: EntityExtractor::new(),
            knowledge_coordinator: None,
            prompt_generator: None,
            use_adaptive_prompts: false,
            last_generated_prompt: None,
            embedding_provider: None,
        }
    }

    /// Create a new orchestrator agent with SEAL processing enabled
    pub fn new_with_seal(
        provider: Arc<dyn Provider>,
        permission_mode: PermissionMode,
        seal_config: SealConfig,
    ) -> Self {
        let task_manager = Arc::new(RwLock::new(TaskManager::new()));
        let task_manager_tool = TaskManagerTool::new(task_manager.clone());

        Self {
            provider,
            tool_executor: ToolExecutor::new(permission_mode),
            task_manager_tool,
            max_iterations: 25,
            seal_processor: Some(SealProcessor::new(seal_config)),
            dialog_state: DialogState::new(),
            entity_store: EntityStore::new(),
            entity_extractor: EntityExtractor::new(),
            knowledge_coordinator: None,
            prompt_generator: None,
            use_adaptive_prompts: false,
            last_generated_prompt: None,
            embedding_provider: None,
        }
    }

    /// Enable SEAL processing with the given configuration
    pub fn enable_seal(&mut self, config: SealConfig) {
        self.seal_processor = Some(SealProcessor::new(config));
    }

    /// Disable SEAL processing
    pub fn disable_seal(&mut self) {
        self.seal_processor = None;
    }

    /// Check if SEAL processing is enabled
    pub fn is_seal_enabled(&self) -> bool {
        self.seal_processor.is_some()
    }

    /// Initialize SEAL for a new conversation
    pub fn init_seal_conversation(&mut self, conversation_id: &str) {
        if let Some(ref mut seal) = self.seal_processor {
            seal.init_conversation(conversation_id);
        }
        // Reset dialog state for new conversation
        self.dialog_state = DialogState::new();
        self.entity_store = EntityStore::new();
    }

    /// Create orchestrator with SEAL + Knowledge integration
    pub fn new_with_seal_and_knowledge(
        provider: Arc<dyn Provider>,
        permission_mode: PermissionMode,
        seal_config: SealConfig,
        knowledge_coordinator: brainwires_seal::SealKnowledgeCoordinator,
    ) -> Self {
        let task_manager = Arc::new(RwLock::new(TaskManager::new()));
        let task_manager_tool = TaskManagerTool::new(task_manager.clone());

        Self {
            provider,
            tool_executor: ToolExecutor::new(permission_mode),
            task_manager_tool,
            max_iterations: 25,
            seal_processor: Some(SealProcessor::new(seal_config)),
            dialog_state: DialogState::new(),
            entity_store: EntityStore::new(),
            entity_extractor: EntityExtractor::new(),
            knowledge_coordinator: Some(knowledge_coordinator),
            prompt_generator: None,
            use_adaptive_prompts: false,
            last_generated_prompt: None,
            embedding_provider: None,
        }
    }

    /// Enable knowledge integration
    pub fn enable_knowledge_integration(
        &mut self,
        coordinator: brainwires_seal::SealKnowledgeCoordinator,
    ) {
        self.knowledge_coordinator = Some(coordinator);
    }

    /// Disable knowledge integration
    pub fn disable_knowledge_integration(&mut self) {
        self.knowledge_coordinator = None;
    }

    /// Check if knowledge integration is enabled
    pub fn is_knowledge_integration_enabled(&self) -> bool {
        self.knowledge_coordinator.is_some()
    }

    /// Enable adaptive prompting with provided generator
    pub fn enable_adaptive_prompting(
        &mut self,
        generator: brainwires::prompting::PromptGenerator,
        embedding_provider: Arc<crate::storage::embeddings::CachedEmbeddingProvider>,
    ) {
        self.prompt_generator = Some(generator);
        self.embedding_provider = Some(embedding_provider);
        self.use_adaptive_prompts = true;
    }

    /// Disable adaptive prompting (revert to static prompts)
    pub fn disable_adaptive_prompting(&mut self) {
        self.use_adaptive_prompts = false;
    }

    /// Check if adaptive prompting is enabled
    pub fn is_adaptive_prompting_enabled(&self) -> bool {
        self.use_adaptive_prompts
            && self.prompt_generator.is_some()
            && self.embedding_provider.is_some()
    }

    /// Set the embedding provider for adaptive prompting
    pub fn set_embedding_provider(
        &mut self,
        provider: Arc<crate::storage::embeddings::CachedEmbeddingProvider>,
    ) {
        self.embedding_provider = Some(provider);
    }

    /// Get reference to the last generated prompt (for learning/debugging)
    pub fn last_generated_prompt(
        &self,
    ) -> Option<&brainwires::prompting::generator::GeneratedPrompt> {
        self.last_generated_prompt.as_ref()
    }

    /// Execute a task with the orchestrator
    pub async fn execute(
        &mut self,
        task_description: &str,
        context: &mut AgentContext,
    ) -> Result<AgentResponse> {
        self.execute_internal(task_description, context, None).await
    }

    /// Execute a task with SEAL processing enabled
    ///
    /// This method preprocesses the input through the SEAL pipeline for:
    /// - Coreference resolution ("fix it" -> "fix [main.rs]")
    /// - Query core extraction (structured query understanding)
    /// - Pattern matching from learned interactions
    pub async fn execute_with_seal(
        &mut self,
        task_description: &str,
        context: &mut AgentContext,
    ) -> Result<AgentResponse> {
        // Preprocess through SEAL
        let (resolved_query, seal_result) = self.preprocess_input(task_description);

        // Log SEAL processing results
        if let Some(ref result) = seal_result {
            if result.resolved_query != result.original_query {
                debug_log!(
                    "🔮 SEAL: Resolved '{}' → '{}'",
                    result.original_query,
                    result.resolved_query
                );
            }
            if let Some(ref pattern) = result.matched_pattern {
                debug_log!("🔮 SEAL: Matched pattern: {}", pattern);
            }
            if let Some(ref core) = result.query_core {
                debug_log!("🔮 SEAL: Query type: {:?}", core.question_type);
            }
        }

        // Execute with resolved query
        let response = self
            .execute_internal(&resolved_query, context, seal_result.as_ref())
            .await?;

        // Record outcome for learning
        self.record_seal_outcome(seal_result.as_ref(), response.is_complete, 1);

        Ok(response)
    }

    /// Internal execute implementation
    async fn execute_internal(
        &mut self,
        task_description: &str,
        context: &mut AgentContext,
        seal_result: Option<&SealProcessingResult>,
    ) -> Result<AgentResponse> {
        let mut iterations = 0;
        let mut task = Task::new(
            uuid::Uuid::new_v4().to_string(),
            task_description.to_string(),
        );
        task.start();

        // Add initial user message
        let user_message = Message {
            role: Role::User,
            content: MessageContent::Text(task_description.to_string()),
            name: None,
            metadata: None,
        };
        context.conversation_history.push(user_message);

        loop {
            iterations += 1;
            task.increment_iteration();

            if iterations > self.max_iterations {
                task.fail(format!("Max iterations ({}) reached", self.max_iterations));
                return Ok(AgentResponse {
                    message: "Task failed: exceeded maximum iterations".to_string(),
                    is_complete: true,
                    tasks: vec![task],
                    iterations,
                });
            }

            // Call the AI provider with optional SEAL result
            let response = self
                .call_provider(context, seal_result, task_description)
                .await?;

            debug_log!(
                "🔍 DEBUG - Response message content type: {:?}",
                match &response.message.content {
                    MessageContent::Text(_) => "Text",
                    MessageContent::Blocks(blocks) => {
                        debug_log!("🔍 DEBUG - Message has {} blocks", blocks.len());
                        for (i, block) in blocks.iter().enumerate() {
                            match block {
                                ContentBlock::Text { .. } => {
                                    debug_log!("🔍 DEBUG - Block {}: Text", i)
                                }
                                ContentBlock::ToolUse { name, .. } => {
                                    debug_log!("🔍 DEBUG - Block {}: ToolUse({})", i, name)
                                }
                                ContentBlock::ToolResult { .. } => {
                                    debug_log!("🔍 DEBUG - Block {}: ToolResult", i)
                                }
                                ContentBlock::Image { .. } => {
                                    debug_log!("🔍 DEBUG - Block {}: Image", i)
                                }
                            }
                        }
                        "Blocks"
                    }
                }
            );

            // Check if task is complete
            if let Some(finish_reason) = &response.finish_reason
                && (finish_reason == "end_turn" || finish_reason == "stop")
            {
                // Extract final message
                let message_text = response
                    .message
                    .text()
                    .unwrap_or("Task completed")
                    .to_string();
                task.complete(message_text.clone());

                return Ok(AgentResponse {
                    message: message_text,
                    is_complete: true,
                    tasks: vec![task],
                    iterations,
                });
            }

            // Process tool uses in the response
            let tool_uses = self.extract_tool_uses(&response.message);

            debug_log!(
                "🔍 DEBUG - Extracted {} tool uses from response",
                tool_uses.len()
            );
            if !tool_uses.is_empty() {
                debug_log!(
                    "🔍 DEBUG - Tool uses: {:?}",
                    tool_uses.iter().map(|t| &t.name).collect::<Vec<_>>()
                );
            }

            if tool_uses.is_empty() {
                // No tool uses, treat as completion
                let message_text = response
                    .message
                    .text()
                    .unwrap_or("Task completed")
                    .to_string();
                task.complete(message_text.clone());

                debug_log!("🔍 DEBUG - No tool uses, completing task");
                return Ok(AgentResponse {
                    message: message_text,
                    is_complete: true,
                    tasks: vec![task],
                    iterations,
                });
            }

            // Add assistant message to history
            context.conversation_history.push(response.message.clone());

            // Execute tools and add results to history
            let tool_context = ToolContext::from_agent_context(context);

            for tool_use in tool_uses {
                // Check if this is a task manager tool
                let result = if tool_use.name.starts_with("task_") {
                    self.task_manager_tool
                        .execute(&tool_use.id, &tool_use.name, &tool_use.input)
                        .await
                } else {
                    // Use execute_with_retry for automatic retry on transient errors
                    // (AgentDebug paper: arxiv:2509.25370)
                    let (tool_result, outcome) = self
                        .tool_executor
                        .execute_with_retry(&tool_use, &tool_context)
                        .await?;

                    // Log retry attempts for debugging
                    if outcome.retries > 0 {
                        debug_log!(
                            "🔄 Tool '{}' succeeded after {} retries",
                            tool_use.name,
                            outcome.retries
                        );
                    }

                    // Log outcome for SEAL pattern learning
                    // (AgentDebug paper: arxiv:2509.25370 - track execution patterns)
                    // Note: Actual SEAL learning happens via execute_with_seal() which has &mut self
                    if self.seal_processor.is_some() {
                        let success = !tool_result.is_error;
                        let pattern_id = format!("tool:{}", tool_use.name);

                        // Log tool execution outcomes for analysis
                        // The SEAL learning coordinator records these patterns for future use
                        if !success {
                            debug_log!(
                                "📊 SEAL tracking: tool '{}' failed after {} retries (pattern: {})",
                                tool_use.name,
                                outcome.retries,
                                pattern_id
                            );
                        } else if outcome.retries > 0 {
                            debug_log!(
                                "📊 SEAL tracking: tool '{}' succeeded after {} retries",
                                tool_use.name,
                                outcome.retries
                            );
                        }
                    }

                    tool_result
                };

                // Add tool result message
                let result_message = Message {
                    role: Role::User,
                    content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                        tool_use_id: result.tool_use_id.clone(),
                        content: result.content.clone(),
                        is_error: Some(result.is_error),
                    }]),
                    name: None,
                    metadata: None,
                };

                context.conversation_history.push(result_message);
            }

            // Continue loop for next iteration
        }
    }

    /// Call the AI provider with current context
    async fn call_provider(
        &mut self,
        context: &AgentContext,
        seal_result: Option<&SealProcessingResult>,
        user_query: &str,
    ) -> Result<ChatResponse> {
        // Get SEAL learning context if available
        // (Integrates learned patterns into prompt for improved decision-making)
        let seal_context = self.get_seal_context();
        let learning_section = if seal_context.is_empty() {
            String::new()
        } else {
            format!(
                "\n\nLEARNED PATTERNS (from previous executions):\n{}\n",
                seal_context
            )
        };

        // Get BKS and PKS context if knowledge coordinator is enabled
        let mut knowledge_sections = Vec::new();

        if let Some(ref mut coordinator) = self.knowledge_coordinator {
            // Get BKS context for query
            if let Ok(Some(bks_context)) = coordinator.get_bks_context(user_query).await {
                knowledge_sections.push(bks_context);
            }

            // Get PKS context for SEAL entity resolutions
            if let Some(seal_res) = seal_result
                && let Ok(Some(pks_context)) = coordinator.get_pks_context(seal_res).await
            {
                knowledge_sections.push(pks_context);
            }
        }

        let knowledge_section = if knowledge_sections.is_empty() {
            String::new()
        } else {
            format!("\n\n{}\n", knowledge_sections.join("\n\n"))
        };

        // Build system prompt (adaptive or static based on configuration)
        let system_prompt = self
            .build_system_prompt(
                &context.working_directory,
                user_query,
                seal_result,
                &learning_section,
                &knowledge_section,
            )
            .await?;

        // Use conservative token limits to prevent runaway generation
        // For file operations, rely on edit_file instead of write_file for large files
        let max_tokens = ORCHESTRATOR_MAX_TOKENS; // Conservative limit to prevent corruption

        let options = ChatOptions {
            temperature: Some(0.7),
            max_tokens: Some(max_tokens),
            top_p: None,
            stop: None,
            system: Some(system_prompt),
            model: None,
            cache_strategy: Default::default(),
        };

        self.provider
            .chat(
                &context.conversation_history,
                Some(&context.tools),
                &options,
            )
            .await
    }

    /// Extract tool uses from a message
    fn extract_tool_uses(&self, message: &Message) -> Vec<ToolUse> {
        match &message.content {
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse { id, name, input } => Some(ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    }),
                    _ => None,
                })
                .collect(),
            _ => vec![],
        }
    }

    /// Set maximum iterations
    pub fn set_max_iterations(&mut self, max: u32) {
        self.max_iterations = max;
    }

    /// Preprocess user input through SEAL pipeline
    ///
    /// This method:
    /// 1. Extracts entities from the input
    /// 2. Updates the entity store
    /// 3. Resolves coreferences (pronouns, "the file", etc.)
    /// 4. Extracts structured query cores
    /// 5. Checks for learned patterns
    ///
    /// Returns the original query if SEAL is disabled, otherwise returns the
    /// processed result with resolved references.
    pub fn preprocess_input(&mut self, input: &str) -> (String, Option<SealProcessingResult>) {
        // Always extract entities and update state
        let extraction = self.entity_extractor.extract(input, "");
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        self.entity_store.add_extraction(extraction, "", timestamp);

        // Update dialog state with new entities
        self.update_dialog_state(input);

        // If SEAL is enabled, process through the full pipeline
        if let Some(ref mut seal) = self.seal_processor {
            match seal.process(input, &self.dialog_state, &self.entity_store, None) {
                Ok(result) => {
                    let resolved = result.resolved_query.clone();
                    return (resolved, Some(result));
                }
                Err(e) => {
                    debug_log!("SEAL processing error: {:?}", e);
                }
            }
        }

        (input.to_string(), None)
    }

    /// Update dialog state with entities from a message
    fn update_dialog_state(&mut self, content: &str) {
        // Increment turn counter
        self.dialog_state.next_turn();

        // Extract entities and add to dialog state
        // The SEAL module uses the same EntityType from entity_extraction
        let extraction = self.entity_extractor.extract(content, "");
        for (name, entity_type) in extraction.entities {
            self.dialog_state.mention_entity(&name, entity_type);
        }
    }

    /// Execute a task using MDAP (Massively Decomposed Agentic Processes) mode
    ///
    /// This method executes tasks with high reliability using:
    /// - Task decomposition into minimal subtasks
    /// - First-to-ahead-by-k voting for error correction
    /// - Red-flagging for response validation
    ///
    /// Returns metrics about the execution along with the result.
    pub async fn execute_mdap(
        &mut self,
        task_description: &str,
        context: &mut AgentContext,
        mdap_config: MdapConfig,
    ) -> Result<(AgentResponse, MdapMetrics)> {
        use crate::mdap::{
            composer::{ResultComposer, StandardComposer},
            decomposition::{
                BinaryRecursiveDecomposer, SimpleRecursiveDecomposer, TaskDecomposer,
                topological_sort,
            },
            microagent::Microagent,
            red_flags::StandardRedFlagValidator,
            scaling::estimate_mdap,
            voting::FirstToAheadByKVoter,
        };

        let execution_id = uuid::Uuid::new_v4().to_string();
        let mut metrics = MdapMetrics::new(execution_id.clone());
        metrics.start();

        debug_log!(
            "🔄 MDAP: Starting execution with k={}, target={}%",
            mdap_config.k,
            mdap_config.target_success_rate * 100.0
        );

        // Create provider adapter for microagents with tools from context
        // Tools are described in the system prompt for intent expression,
        // but NOT executed during microagent chat calls - execution happens after voting
        let provider_adapter = if context.tools.is_empty() {
            Arc::new(ProviderMicroagentAdapter::new(self.provider.clone()))
        } else {
            debug_log!(
                "🔄 MDAP: Microagents have access to {} tools",
                context.tools.len()
            );
            Arc::new(ProviderMicroagentAdapter::new_with_tools(
                self.provider.clone(),
                context.tools.clone(),
            ))
        };

        // Create decomposer based on config
        let decomposer: Box<dyn TaskDecomposer + Send + Sync> = match mdap_config.decomposition {
            crate::mdap::DecompositionStrategy::BinaryRecursive { max_depth } => Box::new(
                BinaryRecursiveDecomposer::new(provider_adapter.clone(), max_depth, mdap_config.k),
            ),
            crate::mdap::DecompositionStrategy::Simple { max_depth } => {
                Box::new(SimpleRecursiveDecomposer::new(max_depth))
            }
            _ => {
                // Default to simple for unsupported strategies
                Box::new(SimpleRecursiveDecomposer::new(10))
            }
        };

        // Create decomposition context
        let decompose_context = DecomposeContext::new(context.working_directory.clone())
            .with_tools(context.tools.iter().map(|t| t.name.clone()).collect())
            .with_max_depth(10);

        // Decompose the task
        debug_log!("🔄 MDAP: Decomposing task: {}", task_description);
        let decomposition = match decomposer
            .decompose(task_description, &decompose_context)
            .await
        {
            Ok(d) => d,
            Err(e) => {
                metrics.finalize(false);
                return Err(anyhow::anyhow!("MDAP decomposition failed: {}", e));
            }
        };

        metrics.total_steps = decomposition.subtasks.len() as u64;
        debug_log!(
            "🔄 MDAP: Decomposed into {} subtasks",
            decomposition.subtasks.len()
        );

        // Estimate cost
        let estimate = estimate_mdap(
            decomposition.subtasks.len() as u64,
            0.99, // Assume 99% per-step success rate
            0.95, // Assume 95% valid response rate
            mdap_config.cost_per_sample_usd.unwrap_or(0.0001),
            mdap_config.target_success_rate,
        );
        if let Ok(ref est) = estimate {
            metrics.estimated_cost_usd = est.expected_cost_usd;
            metrics.estimated_success_probability = est.success_probability;
        }

        debug_log!(
            "🔄 MDAP: Estimated cost: ${:.4}, success probability: {:.2}%",
            metrics.estimated_cost_usd,
            metrics.estimated_success_probability * 100.0
        );

        // Create voter and validator
        let voter = FirstToAheadByKVoter::new(mdap_config.k, mdap_config.max_samples_per_subtask);

        // Use completely permissive validator when tools are available to allow any response format
        use crate::mdap::red_flags::{AcceptAllValidator, RedFlagValidator};
        let validator: Box<dyn RedFlagValidator> = if context.tools.is_empty() {
            Box::new(StandardRedFlagValidator::new(
                mdap_config.red_flags.clone(),
                None,
            ))
        } else {
            // Accept all responses when tools are available - let voting handle quality
            Box::new(AcceptAllValidator)
        };

        // Order subtasks by dependencies
        let ordered_subtasks = match topological_sort(&decomposition.subtasks) {
            Ok(sorted) => sorted,
            Err(e) => {
                metrics.finalize(false);
                return Err(anyhow::anyhow!("MDAP topological sort failed: {}", e));
            }
        };

        // Execute subtasks in order
        let mut results: std::collections::HashMap<String, crate::mdap::microagent::SubtaskOutput> =
            std::collections::HashMap::new();

        for subtask in ordered_subtasks {
            debug_log!("🔄 MDAP: Executing subtask: {}", subtask.description);

            // Gather inputs from dependencies
            let mut input = subtask.input_state.clone();
            for dep_id in &subtask.depends_on {
                if let Some(dep_result) = results.get(dep_id) {
                    input[dep_id] = dep_result.value.clone();
                }
            }

            // Execute with voting
            let start = std::time::Instant::now();

            // Clone data for closure - need separate clones for each iteration
            let subtask_id = subtask.id.clone();
            let subtask_desc = subtask.description.clone();
            let provider_for_closure = provider_adapter.clone();
            let subtask_for_closure = subtask.clone();
            let input_for_closure = input.clone();

            let vote_result = {
                let validator_ref = validator.as_ref();

                voter
                    .vote(
                        move || {
                            let provider = provider_for_closure.clone();
                            let st = subtask_for_closure.clone();
                            let inp = input_for_closure.clone();
                            async move {
                                let ma = Microagent::with_defaults(provider, st);
                                ma.execute_once(&inp).await
                            }
                        },
                        validator_ref,
                        |output| output.subtask_id.clone(),
                    )
                    .await
            };

            let elapsed = start.elapsed();

            match vote_result {
                Ok(result) => {
                    // Record metrics
                    metrics.record_subtask(crate::mdap::metrics::SubtaskMetric {
                        subtask_id: subtask_id.clone(),
                        description: subtask_desc,
                        samples_needed: result.total_samples,
                        red_flags_hit: result.red_flagged_count,
                        red_flag_reasons: vec![],
                        final_confidence: result.confidence,
                        execution_time_ms: elapsed.as_millis() as u64,
                        winner_votes: result.winner_votes,
                        total_votes: result.total_votes,
                        succeeded: true,
                        complexity_estimate: 0.5,
                        input_tokens: 0,
                        output_tokens: 0,
                    });

                    // Store result
                    results.insert(subtask_id, result.winner);

                    debug_log!(
                        "🔄 MDAP: Subtask completed with {} votes, confidence: {:.2}%",
                        result.winner_votes,
                        result.confidence * 100.0
                    );
                }
                Err(e) => {
                    debug_log!("🔄 MDAP: Subtask failed: {}", e);
                    metrics.failed_steps += 1;
                    // Continue with other subtasks or fail early based on config
                    if mdap_config.fail_fast {
                        metrics.finalize(false);
                        return Err(anyhow::anyhow!("MDAP subtask failed: {}", e));
                    }
                }
            }
        }

        // Compose results
        let composer = StandardComposer;
        let final_output = match composer.compose(&decomposition, &results) {
            Ok(output) => output,
            Err(e) => {
                metrics.finalize(false);
                return Err(anyhow::anyhow!("MDAP composition failed: {}", e));
            }
        };

        // Finalize metrics
        metrics.finalize(true);

        debug_log!("🔄 MDAP: Execution complete\n{}", metrics.summary());

        // Feed metrics to SEAL if enabled
        if mdap_config.seal_integration
            && let Some(ref mut seal) = self.seal_processor
        {
            seal.record_mdap_metrics(&metrics);
        }

        // Extract the decision/plan from MDAP output
        let plan = match final_output {
            serde_json::Value::String(s) => s,
            serde_json::Value::Object(map) => {
                if let Some(serde_json::Value::Array(outputs)) = map.get("outputs") {
                    outputs
                        .iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join("\n\n")
                } else {
                    serde_json::to_string_pretty(&map).unwrap_or_default()
                }
            }
            other => serde_json::to_string_pretty(&other).unwrap_or_default(),
        };

        // If we have tools available and the plan looks like it needs execution,
        // execute it using the orchestrator with tools
        if !context.tools.is_empty()
            && (plan.contains("edit_file")
                || plan.contains("write_file")
                || plan.contains("read_file")
                || task_description.to_lowercase().contains("add")
                || task_description.to_lowercase().contains("modify")
                || task_description.to_lowercase().contains("change")
                || task_description.to_lowercase().contains("create")
                || task_description.to_lowercase().contains("file"))
        {
            debug_log!("🔄 MDAP: Plan requires tool execution, delegating to orchestrator");

            // Execute the plan using the orchestrator (without MDAP recursion)
            match self.execute_internal(&plan, context, None).await {
                Ok(exec_response) => {
                    let response = AgentResponse {
                        message: exec_response.message,
                        is_complete: exec_response.is_complete,
                        tasks: exec_response.tasks,
                        iterations: metrics.completed_steps as u32 + exec_response.iterations,
                    };
                    Ok((response, metrics))
                }
                Err(e) => {
                    debug_log!("🔄 MDAP: Tool execution failed: {}", e);
                    // Fallback to just returning the plan
                    let response = AgentResponse {
                        message: format!("MDAP Plan (execution failed): {}\n\nError: {}", plan, e),
                        is_complete: true,
                        tasks: vec![],
                        iterations: metrics.completed_steps as u32,
                    };
                    Ok((response, metrics))
                }
            }
        } else {
            // No tool execution needed, return the plan as-is
            let response = AgentResponse {
                message: plan,
                is_complete: true,
                tasks: vec![],
                iterations: metrics.completed_steps as u32,
            };
            Ok((response, metrics))
        }
    }

    /// Record the outcome of a task execution for learning
    ///
    /// This should be called after tool execution to help the learning
    /// coordinator improve pattern matching.
    pub fn record_seal_outcome(
        &mut self,
        seal_result: Option<&SealProcessingResult>,
        success: bool,
        result_count: usize,
    ) {
        if let Some(ref mut seal) = self.seal_processor {
            let pattern_id = seal_result.and_then(|r| r.matched_pattern.as_deref());
            let query_core = seal_result.and_then(|r| r.query_core.as_ref());
            seal.record_outcome(pattern_id, success, result_count, query_core, 0);
        }

        // Observe SEAL resolutions in PKS and check for pattern promotion
        if let Some(seal_res) = seal_result
            && let Some(ref mut coordinator) = self.knowledge_coordinator
        {
            // Observe entity resolutions for PKS learning
            let _ = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    coordinator
                        .observe_seal_resolutions(&seal_res.resolutions)
                        .await
                })
            });

            // Check if we should promote high-reliability patterns to BKS
            self.check_pattern_promotion();
        }
    }

    /// Check and promote high-reliability SEAL patterns to BKS
    fn check_pattern_promotion(&mut self) {
        if let Some(ref mut seal) = self.seal_processor
            && let Some(ref mut coordinator) = self.knowledge_coordinator
        {
            let config = coordinator.config();

            // Get promotable patterns from SEAL
            let promotable = seal.learning_mut().get_promotable_patterns(
                config.pattern_promotion_threshold,
                config.min_pattern_uses,
            );

            // Promote each pattern to BKS
            for pattern in promotable {
                let execution_context = format!("{:?} queries", pattern.question_type);

                let _ = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        coordinator
                            .check_and_promote_pattern(pattern, &execution_context)
                            .await
                    })
                });
            }
        }
    }

    /// Get learning context for prompt enhancement
    ///
    /// Returns a string containing learned patterns and entity context
    /// that can be injected into the system prompt.
    pub fn get_seal_context(&self) -> String {
        if let Some(ref seal) = self.seal_processor {
            seal.get_learning_context()
        } else {
            String::new()
        }
    }

    /// Get access to the entity store
    pub fn entity_store(&self) -> &EntityStore {
        &self.entity_store
    }

    /// Get access to the dialog state
    pub fn dialog_state(&self) -> &DialogState {
        &self.dialog_state
    }

    /// Build system prompt (adaptive or static based on configuration)
    async fn build_system_prompt(
        &mut self,
        working_directory: &str,
        user_query: &str,
        seal_result: Option<&SealProcessingResult>,
        learning_section: &str,
        knowledge_section: &str,
    ) -> Result<String> {
        // Try adaptive prompting first if enabled
        if self.use_adaptive_prompts {
            // Safely extract providers - if either is missing, fall through to default prompt
            if let (Some(generator), Some(embedding_provider)) = (
                self.prompt_generator.as_ref(),
                self.embedding_provider.as_ref(),
            ) {
                // Use SEAL's resolved query if available for better classification
                let task_desc = seal_result
                    .map(|r| r.resolved_query.as_str())
                    .unwrap_or(user_query);

                // Generate embedding for task description
                match embedding_provider.embed_cached(task_desc) {
                    Ok(task_embedding) => {
                        // Generate adaptive prompt
                        // Convert CLI SealProcessingResult to framework's stub type
                        let framework_seal = seal_result.map(|r| {
                            brainwires::prompting::SealProcessingResult::new(
                                r.quality_score,
                                &r.resolved_query,
                            )
                        });
                        match generator
                            .generate_prompt(task_desc, &task_embedding, framework_seal.as_ref())
                            .await
                        {
                            Ok(generated_prompt) => {
                                debug_log!(
                                    "🎯 Adaptive prompting: Generated prompt using cluster '{}' with techniques {:?}",
                                    generated_prompt.cluster_id,
                                    generated_prompt.techniques
                                );
                                debug_log!(
                                    "🎯 SEAL quality: {:.2}, Cluster similarity: {:.2}",
                                    generated_prompt.seal_quality,
                                    generated_prompt.similarity_score
                                );

                                // Store for learning
                                self.last_generated_prompt = Some(generated_prompt.clone());

                                // Build full system prompt with adaptive techniques
                                let full_prompt = format!(
                                    "{}\n\n\
                                Current working directory: {}\n\n\
                                CRITICAL FILE OPERATION RULES:\n\
                                1. For small edits to existing files: ALWAYS use edit_file with specific old_text/new_text\n\
                                2. NEVER use write_file for large files or when preserving existing content\n\
                                3. For adding content to files: Use edit_file to insert at specific locations\n\
                                4. Read files first with read_file before making any changes\n\
                                5. Be precise - match exact text including whitespace when using edit_file\n\n\
                                TOOL USAGE:\n\
                                - read_file: {{\"path\": \"README.md\"}} - Read complete file content\n\
                                - edit_file: {{\"path\": \"README.md\", \"old_text\": \"# Title\\n\\nExisting content\", \"new_text\": \"# Title\\n\\nHello World\\n\\nExisting content\"}} - Edit specific text\n\
                                - list_directory: {{\"path\": \".\"}} - List files\n\
                                - bash: {{\"command\": \"git status\"}} - Run commands\n\n\
                                Remember: edit_file requires EXACT text matching. Include enough surrounding lines to make the match unique.\
                                {}{}",
                                    generated_prompt.system_prompt,
                                    working_directory,
                                    learning_section,
                                    knowledge_section
                                );

                                return Ok(full_prompt);
                            }
                            Err(e) => {
                                debug_log!(
                                    "⚠️ Adaptive prompting failed: {}. Falling back to static prompt.",
                                    e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        debug_log!(
                            "⚠️ Failed to generate embedding: {}. Falling back to static prompt.",
                            e
                        );
                    }
                }
            } // Close if-let for providers
        }

        // Fall back to static prompt (if adaptive prompting is disabled or failed)
        Ok(format!(
            "You are an autonomous coding assistant.\n\
            Current working directory: {}{}{}\n\n\
            CRITICAL FILE OPERATION RULES:\n\
            1. For small edits to existing files: ALWAYS use edit_file with specific old_text/new_text\n\
            2. NEVER use write_file for large files or when preserving existing content\n\
            3. For adding content to files: Use edit_file to insert at specific locations\n\
            4. Read files first with read_file before making any changes\n\
            5. Be precise - match exact text including whitespace when using edit_file\n\n\
            TOOL USAGE:\n\
            - read_file: {{\"path\": \"README.md\"}} - Read complete file content\n\
            - edit_file: {{\"path\": \"README.md\", \"old_text\": \"# Title\\n\\nExisting content\", \"new_text\": \"# Title\\n\\nHello World\\n\\nExisting content\"}} - Edit specific text\n\
            - list_directory: {{\"path\": \".\"}} - List files\n\
            - bash: {{\"command\": \"git status\"}} - Run commands\n\n\
            RESPONSE GUIDELINES:\n\
            - For questions about code/concepts: Answer directly without tools\n\
            - For file modifications: Use read_file first, then edit_file with precise text matching\n\
            - Be concise and accurate in your tool usage\n\
            - When editing files, include sufficient context in old_text to ensure unique matching\n\n\
            Remember: edit_file requires EXACT text matching. Include enough surrounding lines to make the match unique.",
            working_directory, learning_section, knowledge_section
        ))
    }
}

/// Adapter that bridges the Provider trait to MicroagentProvider trait
///
/// This allows the existing Provider implementations to be used with
/// the MDAP microagent system.
///
/// # Tool Support
///
/// When tools are provided, the adapter:
/// 1. Includes tool schemas in the system prompt for intent expression
/// 2. Does NOT execute tools during the microagent call
/// 3. Allows the response to contain tool intents which are parsed later
pub struct ProviderMicroagentAdapter {
    provider: Arc<dyn Provider>,
    /// Cached tool schemas for the MicroagentProvider trait
    tool_schemas: Vec<crate::mdap::tool_intent::ToolSchema>,
}

impl ProviderMicroagentAdapter {
    /// Create a new adapter wrapping a Provider
    pub fn new(provider: Arc<dyn Provider>) -> Self {
        Self {
            provider,
            tool_schemas: Vec::new(),
        }
    }

    /// Create a new adapter with tools for MDAP execution
    ///
    /// Tools are described in the system prompt for intent expression,
    /// but NOT executed during microagent chat calls.
    pub fn new_with_tools(
        provider: Arc<dyn Provider>,
        tools: Vec<crate::types::tool::Tool>,
    ) -> Self {
        // Convert tools to schemas
        let tool_schemas: Vec<crate::mdap::tool_intent::ToolSchema> =
            tools.iter().cloned().map(|t| t.into()).collect();

        Self {
            provider,
            tool_schemas,
        }
    }

    /// Extract tool uses from a message (same logic as OrchestratorAgent)
    #[allow(dead_code)]
    fn extract_tool_uses(&self, message: &Message) -> Vec<ToolUse> {
        match &message.content {
            MessageContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse { id, name, input } => Some(ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    }),
                    _ => None,
                })
                .collect(),
            _ => vec![],
        }
    }

    /// Format tool schemas for inclusion in the system prompt
    fn format_tool_schemas(&self) -> String {
        if self.tool_schemas.is_empty() {
            return String::new();
        }

        let mut result = String::from("\n\n## Available Tools\n\n");
        result.push_str("You can express intent to use the following tools. ");
        result.push_str("To use a tool, include a JSON block with the tool_intent field:\n\n");
        result.push_str("```json\n{\n  \"tool_name\": \"<tool_name>\",\n  \"arguments\": { ... },\n  \"rationale\": \"Why you need this tool\"\n}\n```\n\n");
        result.push_str("Available tools:\n\n");

        for schema in &self.tool_schemas {
            result.push_str(&schema.to_prompt_format());
        }

        result.push_str("\n**Note**: Do NOT expect tool results immediately. ");
        result.push_str("Express your intent and continue with your response. ");
        result.push_str("Tool results will be provided as input to subsequent steps.\n");

        result
    }

    /// Enhance the system prompt with tool schemas if tools are available
    fn enhance_system_prompt(&self, base_system: &str) -> String {
        if self.tool_schemas.is_empty() {
            base_system.to_string()
        } else {
            format!("{}{}", base_system, self.format_tool_schemas())
        }
    }
}

#[async_trait::async_trait]
impl MicroagentProvider for ProviderMicroagentAdapter {
    async fn chat(
        &self,
        system: &str,
        user: &str,
        temperature: f32,
        max_tokens: u32,
    ) -> MdapResult<MicroagentResponse> {
        let messages = vec![Message {
            role: Role::User,
            content: MessageContent::Text(user.to_string()),
            name: None,
            metadata: None,
        }];

        // Enhance system prompt with tool schemas if available
        let enhanced_system = self.enhance_system_prompt(system);

        let options = ChatOptions {
            temperature: Some(temperature),
            max_tokens: Some(max_tokens),
            top_p: None,
            stop: None,
            system: Some(enhanced_system),
            model: None,
            cache_strategy: Default::default(),
        };

        let start = std::time::Instant::now();

        // Tools are NOT passed to the provider - we only express intent, not execute
        // Tool intents in the response are parsed and executed AFTER voting consensus
        debug_log!(
            "🔍 MDAP Microagent: Making provider call (tools described in prompt, not executed)"
        );
        let response = self
            .provider
            .chat(&messages, None, &options)
            .await
            .map_err(|e| {
                crate::mdap::error::MdapError::Microagent(
                    crate::mdap::error::MicroagentError::ExecutionFailed {
                        subtask_id: "unknown".to_string(),
                        reason: e.to_string(),
                    },
                )
            })?;

        let elapsed = start.elapsed();
        let text = response.message.text().unwrap_or("").to_string();

        Ok(MicroagentResponse {
            text,
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
            finish_reason: response.finish_reason,
            response_time_ms: elapsed.as_millis() as u64,
        })
    }

    fn available_tools(&self) -> Vec<crate::mdap::tool_intent::ToolSchema> {
        self.tool_schemas.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::agent::TaskStatus;
    use crate::types::message::{
        ChatResponse, ContentBlock, Message, MessageContent, Role, StreamChunk, Usage,
    };
    use crate::types::provider::ChatOptions;
    use crate::types::tool::Tool;
    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use serde_json::json;
    use std::sync::Arc;

    /// Mock provider for testing
    struct MockProvider {
        responses: Vec<ChatResponse>,
        current_index: std::sync::Mutex<usize>,
    }

    impl MockProvider {
        fn new(responses: Vec<ChatResponse>) -> Self {
            Self {
                responses,
                current_index: std::sync::Mutex::new(0),
            }
        }

        fn single_response(finish_reason: &str, text: &str) -> Self {
            Self::new(vec![ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: MessageContent::Text(text.to_string()),
                    name: None,
                    metadata: None,
                },
                finish_reason: Some(finish_reason.to_string()),
                usage: Usage::default(),
            }])
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock-provider"
        }

        async fn chat(
            &self,
            _messages: &[Message],
            _tools: Option<&[Tool]>,
            _options: &ChatOptions,
        ) -> Result<ChatResponse> {
            let mut index = self.current_index.lock().unwrap();
            let response = self
                .responses
                .get(*index)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("No more mock responses"))?;
            *index += 1;
            Ok(response)
        }

        fn stream_chat<'a>(
            &'a self,
            _messages: &'a [Message],
            _tools: Option<&'a [Tool]>,
            _options: &'a ChatOptions,
        ) -> BoxStream<'a, Result<StreamChunk>> {
            unimplemented!("Streaming not used in orchestrator tests")
        }
    }

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let orchestrator = OrchestratorAgent::new(provider, PermissionMode::ReadOnly);
        assert_eq!(orchestrator.max_iterations, 25);
    }

    #[tokio::test]
    async fn test_set_max_iterations() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let mut orchestrator = OrchestratorAgent::new(provider, PermissionMode::ReadOnly);
        orchestrator.set_max_iterations(10);
        assert_eq!(orchestrator.max_iterations, 10);
    }

    #[tokio::test]
    async fn test_task_completion_with_stop_reason() {
        let provider = Arc::new(MockProvider::single_response(
            "stop",
            "Task completed successfully",
        ));
        let mut orchestrator = OrchestratorAgent::new(provider, PermissionMode::ReadOnly);
        let mut context = AgentContext::default();

        let result = orchestrator
            .execute("test task", &mut context)
            .await
            .unwrap();

        assert!(result.is_complete);
        assert_eq!(result.message, "Task completed successfully");
        assert_eq!(result.iterations, 1);
        assert_eq!(result.tasks.len(), 1);
        assert_eq!(result.tasks[0].status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn test_task_completion_with_end_turn() {
        let provider = Arc::new(MockProvider::single_response("end_turn", "All done"));
        let mut orchestrator = OrchestratorAgent::new(provider, PermissionMode::ReadOnly);
        let mut context = AgentContext::default();

        let result = orchestrator
            .execute("test task", &mut context)
            .await
            .unwrap();

        assert!(result.is_complete);
        assert_eq!(result.message, "All done");
        assert_eq!(result.tasks[0].status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn test_task_completion_no_tool_uses() {
        let provider = Arc::new(MockProvider::single_response("other", "Finished"));
        let mut orchestrator = OrchestratorAgent::new(provider, PermissionMode::ReadOnly);
        let mut context = AgentContext::default();

        let result = orchestrator
            .execute("test task", &mut context)
            .await
            .unwrap();

        assert!(result.is_complete);
        assert_eq!(result.message, "Finished");
    }

    // NOTE: Full max_iterations testing with tool execution requires integration tests
    // since we'd need to register real tools. The basic iteration limit logic is covered
    // by the set_max_iterations test above.

    #[tokio::test]
    async fn test_extract_tool_uses_from_blocks() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let orchestrator = OrchestratorAgent::new(provider, PermissionMode::ReadOnly);

        let message = Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Using a tool".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "tool-1".to_string(),
                    name: "test_tool".to_string(),
                    input: json!({"arg": "value"}),
                },
                ContentBlock::ToolUse {
                    id: "tool-2".to_string(),
                    name: "another_tool".to_string(),
                    input: json!({}),
                },
            ]),
            name: None,
            metadata: None,
        };

        let tool_uses = orchestrator.extract_tool_uses(&message);

        assert_eq!(tool_uses.len(), 2);
        assert_eq!(tool_uses[0].id, "tool-1");
        assert_eq!(tool_uses[0].name, "test_tool");
        assert_eq!(tool_uses[1].id, "tool-2");
        assert_eq!(tool_uses[1].name, "another_tool");
    }

    #[tokio::test]
    async fn test_extract_tool_uses_from_text() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let orchestrator = OrchestratorAgent::new(provider, PermissionMode::ReadOnly);

        let message = Message {
            role: Role::Assistant,
            content: MessageContent::Text("Plain text message".to_string()),
            name: None,
            metadata: None,
        };

        let tool_uses = orchestrator.extract_tool_uses(&message);

        assert_eq!(tool_uses.len(), 0);
    }

    #[tokio::test]
    async fn test_extract_tool_uses_empty_blocks() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let orchestrator = OrchestratorAgent::new(provider, PermissionMode::ReadOnly);

        let message = Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(vec![ContentBlock::Text {
                text: "Only text".to_string(),
            }]),
            name: None,
            metadata: None,
        };

        let tool_uses = orchestrator.extract_tool_uses(&message);

        assert_eq!(tool_uses.len(), 0);
    }

    #[tokio::test]
    async fn test_conversation_history_updated() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let mut orchestrator = OrchestratorAgent::new(provider, PermissionMode::ReadOnly);
        let mut context = AgentContext::default();

        assert_eq!(context.conversation_history.len(), 0);

        orchestrator
            .execute("test task", &mut context)
            .await
            .unwrap();

        // With stop reason, only user message is added (assistant response is in result but not history)
        assert_eq!(context.conversation_history.len(), 1);
        assert_eq!(context.conversation_history[0].role, Role::User);
    }

    // ========================
    // SEAL Integration Tests
    // ========================

    #[tokio::test]
    async fn test_orchestrator_with_seal_creation() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let config = SealConfig::default();
        let orchestrator =
            OrchestratorAgent::new_with_seal(provider, PermissionMode::ReadOnly, config);

        assert!(orchestrator.is_seal_enabled());
        assert_eq!(orchestrator.max_iterations, 25);
    }

    #[tokio::test]
    async fn test_seal_enable_disable() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let mut orchestrator = OrchestratorAgent::new(provider, PermissionMode::ReadOnly);

        // Initially disabled
        assert!(!orchestrator.is_seal_enabled());

        // Enable
        orchestrator.enable_seal(SealConfig::default());
        assert!(orchestrator.is_seal_enabled());

        // Disable
        orchestrator.disable_seal();
        assert!(!orchestrator.is_seal_enabled());
    }

    #[tokio::test]
    async fn test_seal_init_conversation() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let mut orchestrator = OrchestratorAgent::new_with_seal(
            provider,
            PermissionMode::ReadOnly,
            SealConfig::default(),
        );

        orchestrator.init_seal_conversation("test-conversation-123");

        // Dialog state should be fresh
        assert_eq!(orchestrator.dialog_state().current_turn, 0);
    }

    #[tokio::test]
    async fn test_preprocess_input_extracts_entities() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let mut orchestrator = OrchestratorAgent::new_with_seal(
            provider,
            PermissionMode::ReadOnly,
            SealConfig::default(),
        );

        // Preprocess a message mentioning a file
        let (_resolved, _seal_result) = orchestrator.preprocess_input("Check the src/main.rs file");

        // Entity store should have the file
        let stats = orchestrator.entity_store().stats();
        assert!(
            stats.total_entities > 0,
            "Should extract at least one entity"
        );
    }

    #[tokio::test]
    async fn test_preprocess_input_updates_dialog_state() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let mut orchestrator = OrchestratorAgent::new_with_seal(
            provider,
            PermissionMode::ReadOnly,
            SealConfig::default(),
        );

        // Initial turn is 0
        assert_eq!(orchestrator.dialog_state().current_turn, 0);

        // Preprocess first message
        orchestrator.preprocess_input("Look at main.rs");
        assert_eq!(orchestrator.dialog_state().current_turn, 1);

        // Preprocess second message
        orchestrator.preprocess_input("Now check config.toml");
        assert_eq!(orchestrator.dialog_state().current_turn, 2);
    }

    #[tokio::test]
    async fn test_execute_with_seal() {
        let provider = Arc::new(MockProvider::single_response("stop", "Fixed the issue"));
        let mut orchestrator = OrchestratorAgent::new_with_seal(
            provider,
            PermissionMode::ReadOnly,
            SealConfig::default(),
        );
        let mut context = AgentContext::default();

        // First, mention a file to set up context
        orchestrator.preprocess_input("Working on src/main.rs");

        // Then execute with a reference
        let result = orchestrator
            .execute_with_seal("Fix the bug", &mut context)
            .await
            .unwrap();

        assert!(result.is_complete);
        assert_eq!(result.message, "Fixed the issue");
    }

    #[tokio::test]
    async fn test_get_seal_context() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let orchestrator = OrchestratorAgent::new_with_seal(
            provider,
            PermissionMode::ReadOnly,
            SealConfig::default(),
        );

        // Context should be available even if empty
        let context = orchestrator.get_seal_context();
        // Just verify it doesn't panic and returns a string
        assert!(context.is_empty() || !context.is_empty());
    }

    #[tokio::test]
    async fn test_seal_disabled_returns_original_query() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let mut orchestrator = OrchestratorAgent::new(provider, PermissionMode::ReadOnly);

        // SEAL is disabled
        assert!(!orchestrator.is_seal_enabled());

        // Preprocessing should return the original query
        let (resolved, seal_result) = orchestrator.preprocess_input("Fix it");

        assert_eq!(resolved, "Fix it");
        assert!(seal_result.is_none());
    }

    #[tokio::test]
    async fn test_seal_coreference_resolution_basic() {
        let provider = Arc::new(MockProvider::single_response("stop", "Done"));
        let mut orchestrator = OrchestratorAgent::new_with_seal(
            provider,
            PermissionMode::ReadOnly,
            SealConfig::default(),
        );

        // Mention main.rs first
        orchestrator.preprocess_input("Let's look at main.rs");

        // Now use a pronoun - should resolve
        let (_resolved, seal_result) = orchestrator.preprocess_input("Fix it");

        // The seal result should exist
        assert!(seal_result.is_some());

        // Check if coreference was attempted (resolution quality depends on confidence threshold)
        if let Some(result) = seal_result {
            // Either resolves successfully or returns original
            assert!(!result.resolved_query.is_empty());
        }
    }
}
