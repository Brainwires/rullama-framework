#![deny(missing_docs)]
//! # Brainwires
//!
//! The Brainwires Agent Framework — build any AI application in Rust.
//!
//! Re-exports all framework sub-crates via feature flags for convenient access.
//!
//! ## Quick Start
//!
//! ```toml
//! [dependencies]
//! brainwires = { version = "0.10", features = ["full"] }
//! ```
//!
//! ```rust
//! use brainwires::prelude::*;
//! ```

/// Core types and traits — available via `brainwires::core::*` or `brainwires::prelude::*`.
pub mod core {
    pub use brainwires_core::*;
}

/// Model tools — both halves at one path. Runtime types (`ToolExecutor`,
/// `ToolRegistry`, error taxonomy, validation, transactions, smart router,
/// plus optional orchestrator / OAuth / OpenAPI / sandbox / sessions /
/// RAG-tool modules) and the concrete builtins (`BashTool`, `FileOpsTool`,
/// `GitTool`, `WebTool`, `SearchTool`, `BuiltinToolExecutor`,
/// `registry_with_builtins`, plus optional `code_exec` / `interpreters` /
/// `semantic_search` / `browser` / `email` / `calendar` / `system`).
///
/// Underlying crates: [`brainwires-tool-runtime`](https://docs.rs/brainwires-tool-runtime)
/// and [`brainwires-tool-builtins`](https://docs.rs/brainwires-tool-builtins).
/// Depend on those directly if you don't want the whole umbrella.
#[cfg(feature = "tools")]
pub mod tools {
    pub use brainwires_tool_builtins::*;
    pub use brainwires_tool_runtime::*;
}

/// Agent runtime, communication hub, task management, and validation.
///
/// Spreads `brainwires-agent` (coordination + patterns + schema) and
/// `brainwires-inference` (LLM-driven workhorses — `ChatAgent`,
/// `TaskAgent`, planner/judge/validator helpers) under one module path
/// for back-compat. New code can import either crate directly.
#[cfg(feature = "agents")]
pub mod agents {
    pub use brainwires_agent::*;
    #[cfg(feature = "inference")]
    pub use brainwires_inference::*;
}

/// LLM-driven workhorses — chat agent, task agent, planner / judge /
/// validator helpers, cycle orchestrator, summarization, system prompts.
#[cfg(feature = "inference")]
pub mod inference {
    pub use brainwires_inference::*;
}

/// Reasoning — planners, validators, routers, strategies, output parsers.
#[cfg(feature = "reasoning")]
pub mod reasoning {
    pub use brainwires_reasoning::*;
}

/// Persistent storage primitives — `StorageBackend` trait, embeddings, BM25.
#[cfg(feature = "storage")]
pub mod storage {
    pub use brainwires_storage::*;
}

/// Tiered hot/warm/cold agent memory — message/summary/fact stores
/// (schema, from `brainwires-stores`) plus the `TieredMemory` orchestration
/// (from `brainwires-memory`, when the `tiered` feature is on).
#[cfg(feature = "memory")]
pub mod memory {
    #[cfg(feature = "tiered")]
    pub use brainwires_memory::{
        CanonicalWriteToken, MultiFactorScore, TieredMemory, TieredMemoryConfig, TieredMemoryStats,
        TieredSearchResult,
    };
    pub use brainwires_stores::*;
}

/// MCP client — connect to external MCP servers and use their tools.
#[cfg(feature = "mcp")]
pub mod mcp {
    pub use brainwires_mcp_client::*;
}

/// MDAP — Multi-Dimensional Adaptive Planning with MAKER voting.
#[cfg(feature = "mdap")]
pub mod mdap {
    pub use brainwires_mdap::*;
}

/// Adaptive prompting — technique library, clustering, temperature optimization.
#[cfg(feature = "prompting")]
pub mod prompting {
    pub use brainwires_prompting::*;
}

/// Permissions — capability profiles, policy engine, audit logging.
#[cfg(feature = "permissions")]
pub mod permissions {
    pub use brainwires_permission::*;
}

/// AI provider implementations — OpenAI, Anthropic, Google, Ollama, and more.
#[cfg(feature = "providers")]
pub mod providers {
    pub use brainwires_provider::*;
}

/// Chat provider implementations (Provider trait wrappers over API clients).
///
/// Re-exported from `brainwires_provider` — Groq, Together, Fireworks, and
/// Anyscale are now served by `OpenAiChatProvider` with a custom provider name.
#[cfg(feature = "chat")]
pub mod chat {
    pub use brainwires_provider::{
        AnthropicChatProvider, ChatProviderFactory, GoogleChatProvider, OllamaChatProvider,
        OpenAiChatProvider, OpenAiResponsesProvider,
    };
}

/// SEAL — Self-Evolving Adaptive Learning for coreference and knowledge.
#[cfg(feature = "seal")]
pub mod seal {
    pub use brainwires_seal::*;
}

// Orchestrator is re-exported via brainwires_tool_runtime::orchestrator when orchestrator feature is on

/// RAG — codebase indexing, semantic search, and retrieval-augmented generation.
#[cfg(feature = "rag")]
pub mod rag {
    pub use brainwires_rag::rag::*;
}

/// Sandboxed code interpreters — Rhai, Lua, JavaScript (Boa), Python (RustPython).
#[cfg(feature = "interpreters")]
pub mod interpreters {
    pub use brainwires_tool_builtins::interpreters::*;
}

/// Agent network — IPC, remote bridge, mesh networking, routing, discovery.
#[cfg(feature = "agent-network")]
pub mod agent_network {
    pub use brainwires_network::*;
}

/// MCP server framework — build MCP-compliant tool servers with middleware.
#[cfg(feature = "mcp-server-framework")]
pub mod mcp_server_framework {
    pub use brainwires_mcp_server::*;
}

/// Skills — SKILL.md parsing, skill registry, and execution.
#[cfg(feature = "skills")]
pub mod skills {
    pub use brainwires_skills::*;
}

/// Evaluation framework — Monte Carlo runner, Wilson CI, adversarial tests.
#[cfg(feature = "eval")]
pub mod eval {
    pub use brainwires_eval::*;
}

// proxy module removed — brainwires-proxy is an extras app, use it directly

/// A2A (Agent-to-Agent) protocol support.
#[cfg(feature = "a2a")]
pub mod a2a {
    pub use brainwires_a2a::*;
}

/// Distributed mesh networking — topology, discovery, federation, routing.
#[cfg(feature = "mesh")]
pub mod mesh {
    pub use brainwires_network::mesh::*;
}

/// Hardware I/O — audio, GPIO, Bluetooth, camera, USB, voice assistant.
#[cfg(any(
    feature = "audio",
    feature = "gpio",
    feature = "bluetooth",
    feature = "camera",
    feature = "usb",
    feature = "vad",
    feature = "wake-word",
    feature = "voice-assistant"
))]
pub mod hardware {
    pub use brainwires_hardware::*;
}

/// LAN inspection tooling — moved from `brainwires-hardware::network` into `brainwires-network::lan`.
#[cfg(feature = "lan-tools")]
pub mod lan {
    pub use brainwires_network::lan::*;
}

/// Audio — capture, playback, speech-to-text, text-to-speech.
#[cfg(feature = "audio")]
pub mod audio {
    pub use brainwires_hardware::audio::*;
}

/// Camera and webcam frame capture.
#[cfg(feature = "camera")]
pub mod camera {
    pub use brainwires_hardware::camera::*;
}

/// Raw USB device access and transfers.
#[cfg(feature = "usb")]
pub mod usb {
    pub use brainwires_hardware::usb::*;
}

/// Voice Activity Detection.
#[cfg(feature = "vad")]
pub mod vad {
    pub use brainwires_hardware::audio::vad::*;
}

/// Wake word detection.
#[cfg(feature = "wake-word")]
pub mod wake_word {
    pub use brainwires_hardware::audio::wake_word::*;
}

/// Voice assistant pipeline.
#[cfg(feature = "voice-assistant")]
pub mod voice_assistant {
    pub use brainwires_hardware::audio::assistant::*;
}

/// Training data pipelines — JSONL, format conversion, tokenization, dedup.
#[cfg(feature = "datasets")]
pub mod datasets {
    pub use brainwires_finetune::datasets::*;
}

/// Model training — cloud fine-tuning, local Burn-based LoRA/QLoRA/DoRA.
#[cfg(feature = "training")]
pub mod training {
    pub use brainwires_finetune::*;
}

// autonomy module requires brainwires-autonomy (publish = false, workspace-only).
// Available when building from the workspace with the `autonomy` feature.

/// Generic OS-level primitives — filesystem event reactor, service management.
#[cfg(feature = "system")]
pub mod system {
    pub use brainwires_tool_builtins::system::*;
}

/// Offline memory consolidation — summarization, fact extraction, hot/warm/cold tier transitions.
#[cfg(feature = "dream")]
pub mod dream {
    pub use brainwires_memory::dream::*;
}

/// Telemetry — analytics events, billing hooks, SQLite persistence, and cost/usage queries.
#[cfg(feature = "telemetry")]
pub mod telemetry {
    pub use brainwires_telemetry::*;
}

/// Central knowledge — BKS, PKS, entity graphs, thought processing.
#[cfg(feature = "knowledge")]
pub mod knowledge {
    pub use brainwires_knowledge::knowledge::*;
}

/// Re-exports for building MCP servers (rmcp, schemars, CancellationToken).
///
/// Enabled with the `mcp-server` feature.
#[cfg(feature = "mcp-server")]
pub mod mcp_server_support {
    pub use rmcp;
    pub use schemars;
    pub use tokio_util::sync::CancellationToken;
}

/// Convenience prelude — import everything commonly needed.
///
/// ```rust
/// use brainwires::prelude::*;
/// ```
pub mod prelude {
    // Core types — always available
    pub use brainwires_core::{
        // Tasks
        AgentResponse,
        // Providers
        ChatOptions,
        // Messages
        ChatResponse,
        ContentBlock,
        EdgeType,
        // Embeddings & vector store
        EmbeddingProvider,
        EntityStoreT,
        // Graph types & traits
        EntityType,
        // Errors
        FrameworkError,
        FrameworkResult,
        GraphEdge,
        GraphNode,
        ImageSource,
        Message,
        MessageContent,
        // Permissions
        PermissionMode,
        // Plans
        PlanMetadata,
        PlanStatus,
        Provider,
        RelationshipGraphT,
        Role,
        StreamChunk,
        Task,
        TaskPriority,
        TaskStatus,
        // Tools
        Tool,
        ToolCaller,
        ToolContext,
        ToolInputSchema,
        ToolMode,
        ToolResult,
        ToolUse,
        Usage,
        VectorSearchResult,
        VectorStore,
        // Working set
        WorkingSet,
        WorkingSetConfig,
        serialize_messages_to_stateless_history,
    };

    // Tools — available with "tools" feature
    #[cfg(feature = "tools")]
    pub use brainwires_tool_builtins::{BashTool, FileOpsTool, GitTool, SearchTool, WebTool};
    #[cfg(feature = "tools")]
    pub use brainwires_tool_runtime::{
        RetryStrategy, ToolCategory, ToolErrorCategory, ToolOutcome, ToolRegistry, ToolSearchTool,
        ValidationTool, classify_error,
    };

    // Agents — available with "agents" feature (coordination only)
    #[cfg(feature = "agents")]
    pub use brainwires_agent::{
        // Access control
        AccessControlManager,
        CommunicationHub,
        ContentionStrategy,
        FileLockManager,
        // Git coordination
        GitCoordinator,
        LockPersistence,
        TaskManager,
        TaskQueue,
    };

    // Inference workhorses — available with "inference" feature
    #[cfg(feature = "inference")]
    pub use brainwires_inference::{
        AgentExecutionResult, AgentRuntime, ExecutionApprovalMode, PlanExecutionConfig,
        PlanExecutionStatus, PlanExecutorAgent, ValidationCheck, ValidationConfig,
        ValidationSeverity, run_agent_loop,
    };

    // Storage — available with "storage" feature
    #[cfg(feature = "storage")]
    pub use brainwires_storage::CachedEmbeddingProvider;
    // Tiered memory orchestration — available with "tiered" feature
    #[cfg(feature = "tiered")]
    pub use brainwires_memory::TieredMemory;

    // MCP — available with "mcp" feature
    #[cfg(feature = "mcp")]
    pub use brainwires_mcp_client::{McpClient, McpConfigManager, McpServerConfig};

    // MDAP — available with "mdap" feature
    #[cfg(feature = "mdap")]
    pub use brainwires_mdap::{
        Composer, FirstToAheadByKVoter, MdapError, MdapEstimate, MdapResult, MicroagentConfig,
        StandardRedFlagValidator,
    };

    // Knowledge — available with "knowledge" feature (now in brainwires-knowledge::knowledge)
    #[cfg(feature = "knowledge")]
    pub use brainwires_knowledge::knowledge::bks_pks::{
        BehavioralKnowledgeCache, BehavioralTruth, PersonalKnowledgeCache, TruthCategory,
    };

    // Prompting — available with "prompting" feature
    #[cfg(feature = "prompting")]
    pub use brainwires_prompting::{
        GeneratedPrompt, PromptGenerator, PromptingTechnique, TaskClusterManager, TechniqueLibrary,
        TemperatureOptimizer,
    };

    // Permissions — available with "permissions" feature
    #[cfg(feature = "permissions")]
    pub use brainwires_permission::{
        AgentCapabilities, ApprovalAction, ApprovalResponse, ApprovalSeverity, AuditLogger,
        CapabilityProfile, PermissionsConfig, PolicyEngine, TrustLevel, TrustManager,
    };

    // Audio — available with "audio" feature
    #[cfg(feature = "audio")]
    pub use brainwires_hardware::{
        AudioBuffer, AudioCapture, AudioConfig, AudioDevice, AudioError, AudioPlayback,
        AudioResult, SpeechToText, SttOptions, TextToSpeech, Transcript, TtsOptions, Voice,
    };

    // A2A protocol — available with "a2a" feature
    #[cfg(feature = "a2a")]
    pub use brainwires_a2a::AgentCard;

    // Mesh networking — available with "mesh" feature
    #[cfg(feature = "mesh")]
    pub use brainwires_network::mesh::{MeshTopology, TopologyType};
}
