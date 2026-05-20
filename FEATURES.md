# Brainwires Framework — Complete Feature List

A comprehensive catalog of every feature provided by the framework's 32 crates and 18 extras.

---

## Table of Contents

- [Core Types & Traits](#core-types--traits)
- [AI Providers](#ai-providers)
- [Agent Orchestration](#agent-orchestration)
- [Tool System](#tool-system)
- [MCP Protocol](#mcp-protocol)
- [MCP Server Framework](#mcp-server-framework)
- [Agent Networking](#agent-networking)
- [MDAP Voting](#mdap-voting)
- [Storage & Memory](#storage--memory)
- [RAG & Code Search](#rag--code-search)
- [Knowledge & Brain](#knowledge--brain)
- [Adaptive Prompting](#adaptive-prompting)
- [SEAL (Self-Evolving Agentic Learning)](#seal-self-evolving-agentic-learning)
- [Permissions & Security](#permissions--security)
- [Hardware I/O](#hardware-io)
  - [Audio](#audio-feature-audio)
  - [Voice Activity Detection](#voice-activity-detection-always-available-with-audio-webrtcvad-requires-feature-vad)
  - [Wake Word Detection](#wake-word-detection-feature-wake-word)
  - [Voice Assistant Pipeline](#voice-assistant-pipeline-feature-voice-assistant)
  - [GPIO](#gpio-feature-gpio-linux)
  - [Bluetooth](#bluetooth-feature-bluetooth)
  - [Network Hardware](#network-hardware-feature-network)
  - [Camera](#camera-feature-camera)
  - [USB](#usb-feature-usb)
  - [Home Automation](#home-automation)
- [Code Interpreters](#code-interpreters)
- [Skills System](#skills-system)
- [Channels](#channels)
- [Datasets & Training Data](#datasets--training-data)
- [Model Training & Fine-Tuning](#model-training--fine-tuning)
- [Distributed Mesh Networking](#distributed-mesh-networking)
- [Agent-to-Agent (A2A) Protocol](#agent-to-agent-a2a-protocol)
- [Autonomous Operations](#autonomous-operations)
- [Reasoning & Inference](#reasoning--inference)
- [Evaluation Framework](#evaluation-framework)
- [Analytics](#analytics)
- [Proxy Framework](#proxy-framework)
- [WASM Bindings](#wasm-bindings)
- [Extras & Standalone Binaries](#extras--standalone-binaries)
- [Facade Crate & Feature Flags](#facade-crate--feature-flags)

---

## Core Types & Traits

**Crate:** `brainwires-core`

Foundation types shared by all framework crates.

- **Message system** — `Message`, `Role`, `ContentBlock`, `ImageSource`, `MessageContent`, streaming `StreamChunk`, `ChatResponse`, `Usage` tracking
- **Stateless history** — `serialize_messages_to_stateless_history()` for API-ready conversation formatting
- **Tool definitions** — `Tool`, `ToolInputSchema`, `ToolResult`, `ToolUse`, `ToolCaller`, `ToolContext`, `ToolMode`
- **Idempotency** — `IdempotencyRecord`, `IdempotencyRegistry` for deduplicating tool calls
- **Staged writes** — `StagedWrite`, `StagingBackend`, `CommitResult` for transactional file operations
- **Task system** — `Task`, `TaskStatus`, `TaskPriority`, `AgentResponse`
- **Plan system** — `PlanMetadata`, `PlanStatus`, step budgets, serializable plans
- **Plan parsing** — `parse_plan_steps()`, `steps_to_tasks()`, structured output parsers (`JsonOutputParser`, `RegexOutputParser`) (feature: `planning`)
- **Provider trait** — `Provider` async trait, `ChatOptions` (temperature, max tokens, top-p, stop sequences, **per-request model override**)
- **Permission modes** — `PermissionMode` (auto, ask, reject)
- **Knowledge graph types** — `EntityType`, `EdgeType`, `GraphNode`, `GraphEdge`, `EntityStoreT`, `RelationshipGraphT` traits
- **Embedding trait** — `EmbeddingProvider` for pluggable embedding backends
- **Vector store trait** — `VectorStore`, `VectorSearchResult` for similarity search abstraction
- **Working set** — `WorkingSet` with LRU eviction, `WorkingSetConfig`, token estimation utilities
- **Content source** — `ContentSource` for tracking where content originates
- **Lifecycle hooks** — Interceptors for framework events
- **Error handling** — `FrameworkError`, `FrameworkResult`
- **WASM support** — `wasm` feature flag for browser-compatible builds

---

## AI Providers

**Crate:** `brainwires-provider`

Unified multi-provider AI interface with 18 provider types.

### Chat Providers

| Provider | Protocol | Auth |
|----------|----------|------|
| **Anthropic** (Claude) | Anthropic Messages | `x-api-key` header |
| **OpenAI** (GPT) | Chat Completions | Bearer token |
| **OpenAI Responses** | Responses API (`/v1/responses`) | Bearer token |
| **Google** (Gemini) | `generateContent` | Bearer token |
| **Ollama** | Native chat (`/api/chat`) | None (local) |
| **Groq** | OpenAI-compatible | Bearer token |
| **Together AI** | OpenAI-compatible | Bearer token |
| **Fireworks AI** | OpenAI-compatible | Bearer token |
| **Anyscale** | OpenAI-compatible | Bearer token |
| **Amazon Bedrock** | Anthropic Messages via AWS SigV4 | AWS SigV4 signing |
| **Google Vertex AI** | Anthropic Messages via OAuth2 | Google OAuth |
| **Brainwires HTTP** | Custom relay protocol | Bearer token |
| **Custom** | User-defined | Configurable |

### Audio API Clients

| Client | Capabilities |
|--------|-------------|
| **ElevenLabs** | TTS + STT |
| **Deepgram** | TTS + STT |
| **Google Cloud TTS** | TTS |
| **Azure Speech** | TTS + STT |
| **Fish Audio** | TTS + ASR |
| **Cartesia** | TTS |
| **Murf AI** | TTS |

### Per-Request Model Override

All chat providers honour `ChatOptions::model: Option<String>`. When `Some`, providers substitute the override for their configured default on that request only. Enables per-session model switching (e.g. the `/model` slash command in BrainClaw) without recreating the provider instance.

### Infrastructure

- **ChatProviderFactory** — Registry-driven protocol dispatch, creates providers from `ProviderConfig`
- **Provider registry** — Static metadata (protocol, auth scheme, endpoint, model listing URL) for all providers
- **RateLimitedClient** — HTTP client with built-in rate limiting
- **RateLimiter** — Token-bucket rate limiter
- **Model listing** — `ModelLister`, `AvailableModel`, `ModelCapability` for querying available models
- **Local LLM** — `llama-cpp-2` integration for local inference (feature: `llama-cpp-2`)
- **Streaming** — All providers return async streams for real-time output; `StreamChunk::ContextCompacted { summary, tokens_freed }` emitted when Claude auto-summarizes context mid-stream (Claude 4.6+)
- **Default models** — Updated to GA Claude 4.6 IDs: Anthropic → `claude-sonnet-4-6`, Bedrock → `anthropic.claude-sonnet-4-6-v1:0`, VertexAI → `claude-sonnet-4-6`
- **HTTP client transport** (`brainwires-mcp-client`, feature `http`) — `HttpTransport` for stateless JSON-RPC-over-HTTP MCP clients

---

## Agent Orchestration

**Crate:** `brainwires-agent`

Multi-agent infrastructure for autonomous task execution.

### Chat

- **ChatAgent** — Reusable streaming completion loop for interactive sessions. `restore_messages()` reloads history from a `SessionStore`; `compact_history()` trims old messages.
- **SessionStore** trait / **JsonFileStore** — Persist and reload conversation history across restarts; wired into agents via config.

### Agent Types

- **TaskAgent** — Autonomous agent executing tasks with tool access, configurable iteration limits, validation loops
- **ValidatorAgent** — Rule-based validation agent for quality checks
- **PlanExecutorAgent** — Executes multi-step plans with approval modes (auto, manual, checkpoint)
- **TaskOrchestrator** — Hierarchical task decomposition with failure policies (fail-fast, continue, retry)

### Workflow Graph Builder

- **WorkflowBuilder** — Declarative DAG-based workflow pipelines with `node()`, `edge()`, `conditional()`, and `build()`
- **WorkflowContext** — Shared state map accessible to all workflow nodes during execution
- **WorkflowResult** — Collected per-node results after execution
- **Parallel fan-out / fan-in** — Nodes with shared predecessors run concurrently via `tokio::spawn`
- **Conditional routing** — Skip downstream branches based on runtime conditions
- **Cycle detection** — Compile-time validation via `petgraph::algo::is_cyclic_directed`

### Runtime & Lifecycle

- **AgentRuntime** — Core agent execution loop with `run_agent_loop()`
- **AgentPool** — Concurrent agent management with lifecycle tracking and pool statistics
- **AgentContext** — Working directory, tool registry, capabilities per agent
- **ExecutionGraph** — Step-by-step telemetry recording (`StepNode`, `ToolCallRecord`, `RunTelemetry`)

### Communication

- **CommunicationHub** — Inter-agent messaging bus with 50+ message types
- **AgentMessage** — Typed messages: `StatusUpdate`, `HelpRequest`, `TaskResult`, `ToolRequest`, conflict info
- **ConflictInfo** — Git operation conflict detection and reporting

### Coordination Patterns

- **ContractNet** — Bidding protocol for agent task negotiation
- **SagaExecutor** — Compensating transactions for distributed operations with rollback
- **OptimisticController** — Optimistic locking with version-based conflict detection
- **MarketAllocator** — Market-based task allocation
- **WaitQueue** — Queue-based coordination primitives
- **ThreeStateModel** — State snapshots for rollback support (`StateSnapshot`, proposed operations)

### File & Resource Coordination

- **FileLockManager** — File-level read/write locks with deadlock prevention
- **ResourceLockManager** — Scoped resource locking with heartbeat-based liveness
- **AccessControlManager** — Advanced access control with contention strategies and lock persistence
- **OperationTracker** — Operation tracking with heartbeat-based liveness checking

### Task Management

- **TaskManager** — Hierarchical task decomposition and dependency tracking
- **TaskQueue** — Priority-based scheduling with dependency awareness

### Git Coordination

- **GitCoordinator** — Git operation locking with `GitLockRequirements`
- **GitOperationRunner** — Safe concurrent git operations
- **WorktreeManager** — Git worktree management for agent isolation (feature: `native`)

### OpenTelemetry Export (feature: `otel`)

- **export_to_otel()** — Maps `ExecutionGraph` and `RunTelemetry` to hierarchical OpenTelemetry spans
- **Span hierarchy** — Root `agent.run` → `agent.iteration.{N}` → `agent.tool.{name}`
- **Attributes** — Token counts, costs, timing, and error information attached as span attributes
- **Compatible** — Works with Jaeger, Datadog, Grafana, and any OpenTelemetry-compatible backend

### Validation

- **ValidationLoop** — Quality checks before agent completion
- **ValidationConfig** — Configurable checks: file existence, duplicate detection, syntax, build verification
- **ResourceChecker** — Conflict detection and resolution
- **Confidence scoring** — `extract_confidence()`, `quick_confidence_check()`, `ResponseConfidence`

---

## Tool System

**Crates:** `brainwires-tool-runtime` (registry, executor, dispatch, validation, orchestrator) + `brainwires-tool-builtins` (concrete bash / file_ops / git / web / search / interpreters / email / calendar tools)

Composable tool implementations for agent use. The 0.11 split moved the runtime primitives into `brainwires-tool-runtime` and the always-available concrete tools into `brainwires-tool-builtins`.

### Built-in Tools (always available)

- **BashTool** — Shell command execution with proactive output management
- **FileOpsTool** — Read, write, edit, patch, list, search, delete, create directory
- **GitTool** — Status, diff, log, stage, commit, push, pull, branch, checkout, and more
- **WebTool** — URL fetching
- **SearchTool** — Regex-based code search (respects `.gitignore`)
- **ValidationTool** — Code quality checks (duplicate detection, build verification, syntax checking)
- **ToolSearchTool** — Meta-tool for dynamic tool discovery at runtime

### Tool Infrastructure

- **ToolRegistry** — Composable container with `with_builtins()` for all tools, category-based organization
- **BuiltinToolExecutor** — Centralized dispatcher for all built-in tools; eliminates ad-hoc dispatch duplication
- **ToolExecutor** — Permission checking, lock acquisition, working set tracking, error handling
- **ToolPreHook** — Pre-execution hooks with `PreHookDecision` (allow/deny/modify)
- **TransactionManager** — Transactional file operations with commit/rollback (feature: `native`)
- **Error taxonomy** — `classify_error()`, `ToolErrorCategory`, `RetryStrategy`, `ToolOutcome`
- **Sanitization** — `contains_sensitive_data()`, `is_injection_attempt()`, `redact_sensitive_data()`, content source wrapping

### Feature-Gated Tools

- **OrchestratorTool** — Rhai script orchestration (feature: `orchestrator`)
- **CodeExecTool** — Sandboxed multi-language code execution (feature: `interpreters`)
- **SemanticSearchTool** — RAG-powered semantic codebase search (feature: `rag`)

### OpenAPI Tool Generation (feature: `openapi`)

- **openapi_to_tools()** — Parse OpenAPI 3.x specs (JSON or YAML) and generate `Tool` definitions
- **OpenApiTool** — Pairs a `Tool` definition with its `OpenApiEndpoint` metadata (method, path, parameters, base URL)
- **execute_openapi_tool()** — Execute an OpenAPI-generated tool call against the live API
- **OpenApiAuth** — Authentication support: `Bearer`, `ApiKey` (header/query), `Basic`
- **OpenApiParam** — Parameter extraction from path, query, header, and request body
- **HttpMethod** — GET, POST, PUT, PATCH, DELETE support

---

## MCP Protocol

**Crate:** `brainwires-mcp-client` (the old aggregate `brainwires-mcp` was split into `brainwires-mcp-client` + `brainwires-mcp-server` in 0.10)

MCP client for connecting to external MCP servers.

- **McpClient** — Connect, list/call tools, read resources, get prompts
- **StdioTransport** — Stdio-based transport layer
- **McpConfigManager** — Server configuration management
- **JSON-RPC 2.0** — Full request/response/notification/error types
- **MCP types** — `McpTool`, `McpResource`, `McpPrompt`, capabilities, initialization
- **Progress tracking** — `ProgressParams`, `McpNotification`
- **Resource reading** — `ReadResourceParams`, `ResourceContent`
- **Prompt system** — `GetPromptParams`, `PromptMessage`, `PromptArgument`

---

## MCP Server Framework

**Crate:** `brainwires-mcp-server`

Build MCP-compliant tool servers with a composable middleware pipeline. Conforms to the MCP 2026 specification.

- **McpServer** — Async event loop: reads JSON-RPC requests, runs middleware chain, dispatches to handler
- **McpHandler** — Trait defining server identity, capabilities, and tool dispatch (`server_info()`, `list_tools()`, `call_tool()`)
- **McpToolRegistry** — Declarative tool registration with `McpToolDef` and `ToolHandler`; automatic dispatch by tool name
- **ServerTransport** / **StdioServerTransport** — Pluggable request/response I/O; stdio included
- **MiddlewareChain** — Ordered onion-model pipeline; middlewares wrap each request and response
- **Middleware implementations:**
  - `AuthMiddleware` — Bearer token validation
  - `LoggingMiddleware` — Structured request/response logging via `tracing`
  - `RateLimitMiddleware` — Token-bucket rate limiting with per-tool limits
  - `ToolFilterMiddleware` — Allow-list or deny-list for tool access
  - `OAuthMiddleware` (feature `oauth`) — OAuth 2.1 JWT validation; HS256 (shared secret) or RS256 (RSA PEM); configurable `iss`/`aud` claim enforcement; token cached per session
- **`RequestContext`** — Per-request client info passed through the middleware chain
- **`AgentNetworkError`** — Unified error type

### HTTP Transport (feature `http`, MCP 2026 spec)

- **`HttpServerTransport`** — Stateless HTTP + SSE transport; `bind(addr, server_card, oauth_resource)` spawns an axum server and returns a transport compatible with `McpServer::with_transport()`
  - `POST /mcp` — JSON-RPC request/response with configurable timeout (`REQUEST_TIMEOUT_SECS = 30`)
  - `GET /mcp/events` — Server-sent events for server-initiated messages; keep-alive pings every `SSE_KEEPALIVE_INTERVAL_SECS = 15` seconds
  - `GET /.well-known/mcp/server-card.json` — MCP Server Card (SEP-1649) for registry discoverability
  - `GET /.well-known/oauth-protected-resource` — RFC9728 OAuth Protected Resource metadata
  - Bounded request queue (`REQUEST_CHANNEL_CAPACITY = 128` in-flight requests)
- **`McpServerCard`** / `build_server_card()` — SEP-1649 server card types: `McpToolCardEntry`, `McpAuthInfo`, `McpTransportInfo`
- **`OAuthProtectedResource`** — RFC9728 response body: `resource`, `authorization_servers`, `scopes_supported`, `bearer_methods_supported`

### Tasks Primitive (SEP-1686)

- **`McpTaskStore`** — Thread-safe in-memory store for long-running async tool calls
  - 5-state lifecycle: `Working → Completed`, `Working → Failed`, `Working → Cancelled`, `Working ↔ InputRequired`
  - TTL-based expiry with `evict_expired()` returning eviction count
  - Typed transitions: `complete(id, result)`, `fail(id, error)`, `cancel(id)`, `update_state(id, state)`
  - `DEFAULT_MAX_RETRIES = 3` default retry budget per task
- **`McpTask`** — Task entry: `id` (UUID v4), `state`, `created_at`, `expires_at`, `result`, `error`, `retry_count`, `max_retries`
- **`McpTaskState`** — `Working | InputRequired | Completed | Failed | Cancelled`

---

## Agent Networking

**Crate:** `brainwires-network`

Agent IPC, remote bridge, 5-layer protocol stack, device allowlists, permission relay, and optional mesh networking. MCP server framework has been extracted to `brainwires-mcp-server`.

### Agent Communication

- **IPC** — Inter-process communication socket protocol
- **Remote relay** — Bridge and realtime protocol for remote agent communication
- **Auth** — Authentication for relay connections

### Agent Management

- **AgentManager** — Agent lifecycle management (`AgentInfo`, `AgentResult`, `SpawnConfig`)
- **AgentToolRegistry** — Pre-built MCP tools for agent operations (spawn, list, status, stop, await)

### Relay Client

- **AgentNetworkClient** — Connect to remote agent network servers (feature: `client`)

### Security & Device Management

- **DeviceAllowlist** — `DeviceStatus` (Allowed/Blocked/Pending), `OrgPolicies` for organization-level enforcement
- **Device fingerprinting** — Bridge computes SHA-256 of machine-id + hostname + OS and sends it in every `Register` message; connection is refused if server responds `Blocked`
- **Sender verification** — Channel-type and channel-ID allowlists; master `channels_enabled` switch evaluated at handshake time
- **PermissionRelay** — `PermissionRequest`/`PermissionResponse` protocol messages for remote human-in-the-loop approval. `PermissionRelay` module: pending request map (oneshot channels per request ID), session-allowed list for pre-approved tools, configurable timeout. `RemoteBridge::send_permission_request()` sends request and awaits response; auto-denies on timeout.

---

## MDAP Voting

**Crate:** `brainwires-agent` (feature `mdap`)

Multi-Dimensional Adaptive Planning implementing the MAKER paper.

- **FirstToAheadByKVoter** — Consensus algorithm where k agents vote for error correction
- **Microagent system** — Minimal-context single-step agents (m=1 decomposition), `MicroagentConfig`, `MicroagentProvider`
- **Task decomposition:**
  - `SequentialDecomposer` — Linear step-by-step decomposition
  - `AtomicDecomposer` — Single-step atomic tasks
  - `BinaryRecursiveDecomposer` — Divide-and-conquer splitting
  - `SimpleRecursiveDecomposer` — Simple recursive breakdown
- **Red flag validation** — `StandardRedFlagValidator`, `RedFlagConfig`, output format checking
- **Cost estimation** — `estimate_mdap()`, `ModelCosts`, probability optimization
- **Metrics** — `MdapMetrics` for execution tracking and reporting
- **Composer** — `StandardComposer`, `CompositionBuilder` for assembling subtask outputs
- **Tool intent** — `ToolIntent`, `ToolSchema`, `ToolCategory` for stateless execution

---

## Storage & Memory

Three-layer storage architecture (see ADR-0005):

- **`brainwires-storage`** — substrate: `StorageBackend` trait,
  9 backends (LanceDB / nornicdb / sqlite / sled / etc.), embeddings,
  BM25 keyword search, file-context primitives.
- **`brainwires-stores`** — schema + CRUD for the framework's
  opinionated minimum data-store set.
- **`brainwires-memory`** — orchestration over the tier schema stores
  (`TieredMemory` adaptive search) + offline `dream` consolidation.

### Substrate  *(crate: `brainwires-storage`)*

- **LanceDatabase** — LanceDB connection and table management
- **FastEmbedManager** — Text embeddings via FastEmbed ONNX model (all-MiniLM-L6-v2)
- **CachedEmbeddingProvider** — LRU-cached embedding provider

### Data Stores  *(crate: `brainwires-stores`)*

- **SessionStore** — Pluggable transcript persistence (in-memory + sqlite impls)
- **ConversationStore** — Conversation catalog metadata (id, title, model, message count)
- **TaskStore** + **AgentStateStore** — Task / agent-state persistence
- **PlanStore** — Execution plan storage with markdown export
- **TemplateStore** — Reusable plan template storage
- **LockStore** — Cross-process lock coordination with statistics
- **ImageStore** — Image analysis storage with metadata + sha256 hashing
- **MessageStore** — Conversation messages with vector search (tier schema)
- **SummaryStore** — Warm-tier compressed summaries (tier schema)
- **FactStore** — Cold-tier extracted key facts (tier schema)
- **MentalModelStore** — Synthesised behavioural / structural / causal /
  procedural beliefs the agent built up (tier schema)
- **TierMetadataStore** — Tier tracking metadata, access counts,
  importance scoring (tier schema)

Default features: `session`, `task`, `plan`, `conversation`. Opt-in:
`memory` (tier schemas), `lock`, `image`, `sqlite`.

### Tiered Memory Orchestration  *(crate: `brainwires-memory`)*

- **TieredMemory** — Multi-factor adaptive search across the four tiers
  (similarity × recency × importance), promotion / demotion based on
  access patterns. Uses the schema stores from `brainwires-stores`.
- **CanonicalWriteToken** — Capability gate for the `Canonical`
  authority tier
- **MultiFactorScore** — Combined retrieval score that blends
  similarity, recency, and stored importance
- **dream** *(feature)* — Offline consolidation engine: summarises
  hot-tier messages into warm-tier summaries, extracts cold-tier facts,
  demotes by retention score

### File Context

- **FileContextManager** — File content management with chunking (`FileChunk`, `FileContent`)

### Agent Integration

- **PersistentTaskManager** — Persistent task management bridging storage and agents (feature: `agents`)

---

## RAG & Code Search

**Crate:** `brainwires-knowledge` (feature: `rag`)

RAG-based codebase indexing and semantic search.

- **RagClient** — Core library API combining all functionality
- **Hybrid search** — Vector similarity (FastEmbed) + BM25 keyword matching (Tantivy) with Reciprocal Rank Fusion
- **Dual database support** — LanceDB (embedded, default) or Qdrant (external server)
- **Smart indexing** — Auto-detects full vs incremental updates with persistent file hash caching
- **AST-based chunking** — Tree-sitter parsing for 12 programming languages (feature: `tree-sitter-languages`)
- **Git history search** — Semantic search over commit history with on-demand indexing
- **Code relations** — Definition finding, reference finding, call graph extraction
- **Document processing** — PDF, markdown, etc. (feature: `documents`)
- **Multi-project support** — Project-scoped indexing and querying
- **Configuration** — Environment variable support, customizable chunk sizes and thresholds

---

## Knowledge & Brain

**Crate:** `brainwires-knowledge` (feature: `knowledge`)

Central knowledge crate for persistent thought storage and entity graphs.

- **BrainClient** — Persistent thought storage with semantic search
- **Thought system** — `Thought`, `ThoughtCategory`, `ThoughtSource` with full CRUD operations
- **Knowledge systems:**
  - **BKS** (Behavioral Knowledge Store) — Behavioral truths and patterns
  - **PKS** (Personal Knowledge Store) — Personal facts and preferences
- **Entity graph:**
  - **EntityStore** — Entity types, extraction results, contradiction detection (`ContradictionEvent`, `ContradictionKind`)
  - **RelationshipGraph** — Edge types, entity context, graph traversal
- **Fact extraction** — Automatic categorization and tag extraction from text
- **MCP tool types** — Request/response types for search, capture, delete, list, and memory stats

---

## Adaptive Prompting

**Crate:** `brainwires-knowledge` (feature: `prompting`)

Implements "Adaptive Selection of Prompting Techniques" (arXiv:2510.18162).

- **15 prompting techniques** — Chain-of-thought, few-shot, zero-shot, and more with `TechniqueCategory` and `ComplexityLevel`
- **Task clustering** — K-means clustering by semantic similarity with `TaskClusterManager`
- **Technique library** — Metadata with BKS integration for technique selection
- **Prompt generator** — Dynamic multi-source prompt generation with `GeneratedPrompt`
- **Learning coordinator** — Technique effectiveness tracking, BKS promotion, cluster summaries
- **Temperature optimization** — Adaptive temperature per cluster with performance tracking
- **Storage** — SQLite persistence for clusters and performance data (feature: `native`)
- **SEAL integration** — `SealProcessingResult` for connecting with SEAL pipeline

---

## SEAL (Self-Evolving Agentic Learning)

**Crate:** `brainwires-agent` (feature: `seal`)

Self-evolving agent capabilities without retraining.

- **SealProcessor** — Main pipeline orchestrating all components
- **Coreference resolution** — Resolves pronouns and elliptical references ("it", "the file", "that function") to concrete entities from dialog history
- **Query core extraction** — Structured S-expression-like query cores (`QueryCore`, `QueryOp`, `QueryExpr`) for graph traversal
- **Self-evolving learning** — `LearningCoordinator` with `GlobalMemory` and `LocalMemory`, pattern matching and reliability tracking
- **Reflection module** — Post-execution analysis, error correction, quality scoring with `ReflectionReport`
- **Knowledge integration** — Entity resolution strategies, SEAL-Brain coordinator (feature: `knowledge`)
- **MDAP integration** — Record MDAP execution metrics for learning (feature: `mdap`)

---

## Permissions & Security

**Crate:** `brainwires-permission`

Capability-based permission system.

### Capabilities

- **AgentCapabilities** — Granular control over:
  - `FilesystemCapabilities` — Path patterns, read/write/execute
  - `ToolCapabilities` — Tool categories, allow/deny lists
  - `NetworkCapabilities` — Domain restrictions, protocols
  - `GitCapabilities` — Operation-level control (clone, push, force-push)
  - `SpawningCapabilities` — Agent spawning limits
  - `ResourceQuotas` — CPU, memory, disk limits

### Profiles

- Pre-defined capability sets: `read_only`, `standard_dev`, `full_access`

### Policy Engine

- **PolicyEngine** — Rule-based enforcement with conditions and actions
- **EnforcementMode** — Strict, permissive, audit-only
- **PolicyCondition** / **PolicyAction** / **PolicyDecision**

### Audit & Trust

- **AuditLogger** — Event logging with querying and statistics
- **AuditEvent** — Typed events with outcomes and feedback signals
- **TrustManager** — Trust levels, violation tracking, trust factor management
- **AnomalyDetector** — Anomaly detection with configurable thresholds

### Approval System

- **ApprovalRequest** / **ApprovalResponse** — Severity-based approval workflow

---

## Hardware I/O

**Crate:** `brainwires-hardware`

Unified hardware abstraction — audio, GPIO, Bluetooth, network hardware, camera, USB.

### Audio (feature: `audio`)

Audio capture, playback, speech-to-text, and text-to-speech.

#### Core

- **AudioCapture** trait — Audio input abstraction
- **AudioPlayback** trait — Audio output abstraction
- **SpeechToText** trait — STT abstraction
- **TextToSpeech** trait — TTS abstraction
- **AudioRingBuffer** — Ring buffer for streaming audio data
- **WAV utilities** — `encode_wav()`, `decode_wav()`
- **Device enumeration** — `AudioDevice`, `DeviceDirection`
- **CpalCapture** — Hardware audio capture via cpal
- **CpalPlayback** — Hardware audio playback via cpal

#### Cloud API Integrations

| Implementation | Type | Provider |
|---------------|------|----------|
| `OpenAiTts` | TTS | OpenAI |
| `OpenAiStt` | STT | OpenAI |
| `ElevenLabsTts` | TTS | ElevenLabs |
| `ElevenLabsStt` | STT | ElevenLabs |
| `DeepgramTts` | TTS | Deepgram |
| `DeepgramStt` | STT | Deepgram |
| `GoogleTts` | TTS | Google Cloud |
| `AzureTts` | TTS | Azure |
| `AzureStt` | STT | Azure |
| `FishTts` | TTS | Fish Audio |
| `FishStt` | STT | Fish Audio |
| `CartesiaTts` | TTS | Cartesia |
| `MurfTts` | TTS | Murf AI |

#### Local Inference

- **WhisperStt** — Local STT via whisper.cpp (feature: `local-stt`)
- **FLAC support** — `encode_flac()`, `decode_flac()` (feature: `flac`)

### GPIO (feature: `gpio`, Linux)

Safe GPIO pin access using the Linux character device API (`gpio-cdev`).

- **GpioPinManager** — Pin allocation, direction, auto-release on agent timeout
- **GpioSafetyPolicy** — Explicit allow-list: no pin is accessible unless listed
- **GpioChipInfo** / **GpioLineInfo** — Chip and line discovery
- **PwmConfig** — Software PWM (frequency, duty cycle validation)

### Bluetooth (feature: `bluetooth`)

Cross-platform BLE scanning via `btleplug` (Linux/BlueZ, macOS, Windows).

- **`list_adapters()`** — Enumerate local Bluetooth radios
- **`scan_ble(duration)`** — Scan for BLE advertisement packets
- **BluetoothDevice** — Address, name, RSSI, services
- **BluetoothAdapter** — Adapter ID and name

### Network Hardware (feature: `network`)

Network interface enumeration, IP configuration, ARP discovery, and port scanning.

- **`list_interfaces()`** — Enumerate NICs (wired, wireless, loopback, virtual)
- **`get_ip_configs()`** — IP addresses and default gateways per interface
- **`arp_scan(subnet)`** — ARP host discovery on local subnet (requires `CAP_NET_RAW`)
- **`arp_probe(hosts)`** — ARP probe a list of specific hosts
- **`scan_ports(host, ports, timeout, concurrency)`** — Async TCP connect port scan
- **`scan_range(host, start, end, ...)`** — Scan a contiguous port range
- **`scan_common_ports(host, timeout)`** — Scan 21 well-known service ports
- **NetworkInterface** — Name, kind, MAC, addresses, up/down status
- **InterfaceKind** — `Wired`, `Wireless`, `Loopback`, `Virtual`, `Unknown`
- **PortScanResult** / **PortState** — Per-port result (`Open`, `Closed`, `Filtered`)
- **DiscoveredHost** — IP, MAC, hostname from ARP replies

### Camera (feature: `camera`)

Cross-platform webcam and camera frame capture via `nokhwa` (V4L2 on Linux, AVFoundation on macOS, Media Foundation on Windows).

- **`list_cameras()`** — Enumerate connected cameras with index, name, and description
- **`open_camera(index, format)`** — Open a camera with an optional format request; falls back to highest frame rate if `None`
- **`CameraCapture` trait** — `format()`, `capture_frame()` (async), `stop()`
- **`NokhwaCapture`** — `CameraCapture` implementation; internally uses `spawn_blocking` for sync nokhwa API
- **CameraDevice** — Index, name, description
- **CameraFrame** — Width, height, pixel format, raw data bytes, timestamp (ms since first frame)
- **CameraFormat** — Resolution, frame rate (numerator/denominator), pixel format
- **PixelFormat** — `Rgb`, `Bgr`, `Rgba`, `Yuv422`, `Mjpeg`, `Unknown`; MJPEG frames are automatically decoded to RGB
- **Resolution** — Width × height; `Display` as `1920x1080`
- **FrameRate** — Rational (numerator/denominator); `Display` as `30fps`

### USB (feature: `usb`)

Raw USB device enumeration and async bulk/control/interrupt transfers via `nusb` (pure Rust, no libusb system dependency).

- **`list_usb_devices()`** — Enumerate all USB devices; reads string descriptors (manufacturer, product, serial) on a best-effort basis
- **`find_device(vendor_id, product_id)`** — Find the first matching device or return `UsbError::DeviceNotFound`
- **`UsbHandle::open(vendor_id, product_id, interface)`** — Open a device and claim an interface; auto-discovers bulk IN/OUT endpoints
- **`UsbHandle::control_in()`** / **`control_out()`** — USB control transfers (standard/class/vendor)
- **`UsbHandle::bulk_read(endpoint, len, timeout)`** / **`bulk_write()`** — Bulk endpoint transfers with auto-endpoint fallback
- **`UsbHandle::interrupt_read()`** / **`interrupt_write()`** — Interrupt endpoint transfers
- **UsbDevice** — Bus, device address, vendor/product ID, class, speed, and optional string descriptors
- **UsbClass** — Full USB-IF class code mapping (HID, MassStorage, Audio, Video, Hub, …, `Unknown(u8)`)
- **UsbSpeed** — `Low`, `Full`, `High`, `Super`, `SuperPlus`, `Unknown`
- **Linux udev** — No root required; add a udev rule for your vendor/product ID to grant user access

### Voice Activity Detection (always available with `audio`; `WebRtcVad` requires feature `vad`)

Classify audio frames as speech or silence.

- **`VoiceActivityDetector` trait** — `is_speech(audio)`, `detect_segments(audio, frame_ms)` → `Vec<SpeechSegment>`
- **`EnergyVad`** — Pure-Rust RMS energy threshold (default: -40 dBFS). Zero extra dependencies.
- **`WebRtcVad`** — WebRTC VAD algorithm (feature: `vad`). Four aggressiveness modes via `VadMode`: `Quality`, `LowBitrate`, `Aggressive`, `VeryAggressive`. Supports 8 / 16 / 32 / 48 kHz with 10, 20, or 30 ms frames.
- **`SpeechSegment`** — `is_speech`, `start_sample`, `end_sample`, `len()`, `is_empty()`
- **Helpers** — `rms_db(audio)` (dBFS), `pcm_to_i16_mono(audio)`, `pcm_to_f32(audio)`

### Wake Word Detection (feature: `wake-word`)

Keyword-triggered activation for the voice assistant pipeline.

- **`WakeWordDetector` trait** — `sample_rate()`, `frame_size()`, `process_frame(samples) -> Option<WakeWordDetection>`
- **`WakeWordDetection`** — `keyword: String`, `score: f32` (0–1), `timestamp_ms: u64`
- **`EnergyTriggerDetector`** — Zero-dependency energy-burst trigger. Fires when audio exceeds a configurable dB threshold for N consecutive 30 ms frames. Useful as a zero-cost "tap-to-wake" or "clap-to-wake" fallback.
- **`RustpotterDetector`** (feature: `wake-word-rustpotter`) — Pure-Rust wake word detection using DTW or ONNX neural models (`.rpw` files). `from_model_file(path, threshold)`, `from_model_files(paths, threshold)`.

### Voice Assistant Pipeline (feature: `voice-assistant`)

End-to-end orchestration: mic capture → wake word → VAD-gated accumulation → STT → handler → TTS → playback.

- **`VoiceAssistant`** — Main pipeline struct. `builder(capture, stt)` → `VoiceAssistantBuilder`. Methods: `run(&handler)` (async event loop), `listen_once()` (single-shot transcript), `stop()`, `state()`.
- **`VoiceAssistantBuilder`** — Fluent builder: `with_playback()`, `with_tts()`, `with_wake_word()`, `with_vad()`, `with_config()`, `build()`.
- **`VoiceAssistantConfig`** — `capture_config`, `silence_threshold_db` (-40 dB default), `silence_duration_ms` (800 ms default), `max_record_secs` (30 s), `listen_timeout_secs` (10 s), `stt_options`, `tts_options`, `microphone`, `speaker`.
- **`VoiceAssistantHandler` trait** — `on_wake_word(&detection)`, `on_speech(&transcript) -> Option<String>`, `on_error(&error)`.
- **`AssistantState`** — `Idle`, `Listening`, `Processing`, `Speaking`.
- **Pipeline loop** — Stream mic chunks at 16 kHz → accumulate frame buffer → wake word detection (if configured) → VAD-gated ring buffer accumulation → STT transcription → handler callback → optional TTS synthesis + playback → loop.

### Home Automation

Home automation protocols (Matter 1.3, Zigbee 3.0, Z-Wave Plus v2, Thread 1.3.0) and the `matter-tool` CLI moved to the standalone `future/home-automation/brainwires-homeauto` workspace in 0.11 — see `future/home-automation/README.md`.

---

## Code Interpreters

**Crate:** `brainwires-tool-builtins` (absorbed from `brainwires-code-interpreters` in 0.10)

Sandboxed multi-language code execution.

| Language | Feature | Engine | Notes |
|----------|---------|--------|-------|
| **Rhai** | `rhai` | Native Rust | Fastest startup |
| **Lua** | `lua` | mlua | Small runtime, good stdlib |
| **JavaScript** | `javascript` | Boa | ECMAScript compliant |
| **Python** | `python` | RustPython | CPython 3.12 compatible |

- **Executor** — Unified execution interface with `ExecutionRequest`
- **WASM support** — Browser-compatible execution (feature: `wasm`)
- **Language detection** — `supported_languages()`, `is_language_supported()`

---

## Skills System

**Crate:** `brainwires-agent` (absorbed from `brainwires-skills`)

Markdown-based agent skill packages.

- **SKILL.md format** — YAML frontmatter (name, description, allowed-tools, model, metadata) + markdown body
- **SkillRegistry** — Skill registration and lookup
- **SkillRouter** — Automatic skill matching from user input
- **SkillExecutor** — Execution modes: `SubagentPrepared` (delegate to subagent) or `ScriptPrepared`
- **Progressive disclosure** — Metadata loaded at startup, full content loaded on-demand
- **SkillSource** — Multiple sources (built-in, user, project)
- **SkillPackage** — Distributable package format: manifest (name, semver, author, license, tags, deps), skill_content, SHA-256 checksum, optional ed25519 signature
- **RegistryClient** — HTTP client for publishing to and downloading from a skill registry server
- **ed25519 signing** (feature `signing`) — Sign and verify skill packages for supply-chain safety

---

## Channels

**Crate:** `brainwires-network` (absorbed from `brainwires-channels`)

Universal messaging channel contract for adapter implementations (Discord, Telegram, Slack, etc.).

- **Channel** trait — Core interface that all messaging adapters must implement
- **ChannelMessage** — Core message types with attachments, embeds, and media
- **ChannelEvent** — Events: message received, edited, deleted, reactions, presence changes, and 10 WebRTC variants (feature-gated)
- **ChannelCapabilities** — 14 bitflags: rich text, media, threads, reactions, voice, video, data channels, encrypted media, etc.
- **ChannelUser** / **ChannelSession** — User and session identity types
- **ChannelHandshake** — Gateway handshake protocol for adapter registration
- **Conversion** — Bidirectional conversion between `ChannelMessage` and agent-network `MessageEnvelope`

### WebRTC Real-Time Media (feature: `webrtc`)

Full peer-to-peer audio/video/DataChannel support via the Brainwires `webrtc-rs` fork.

- **`WebRtcSession`** — One `PeerConnection` per call; offer/answer, trickle ICE, DTLS-SRTP
  - `add_audio_track()` / `add_video_track()` — push encoded frames via `write_sample()`
  - `create_data_channel()` — bi-directional binary/text DataChannels
  - `get_remote_track(id)` — read incoming RTP packets from remote peers
  - `get_stats()` — `RTCStatsReport` snapshot (jitter, packet loss, RTT, bitrate, frame stats)
  - `subscribe()` — broadcast receiver for all 10 WebRTC `ChannelEvent` variants
- **`WebRtcConfig`** — Serde-serializable: ICE servers, DTLS role, mDNS, TCP candidates, bind addresses, codec preferences, bandwidth constraints
- **`WebRtcSignaling`** trait + `BroadcastSignaling` (in-process) + `ChannelMessageSignaling` (piggybacks on existing channel messages)
- **`WebRtcChannel`** trait — adapter extension: `initiate_session()`, `get_session()`, `close_session()`, `signaling()`
- **`RemoteTrack`** — handle to incoming remote media; `poll() -> Option<TrackRemoteEvent>`

### Advanced Congestion Control (feature: `webrtc-advanced`)

- **GCC** (Google Congestion Control) — adaptive bitrate from TWCC feedback; `session.target_bitrate_bps()`
- **JitterBuffer** — adaptive playout delay; reorders out-of-sequence packets
- **TwccSender** — transport-wide sequence numbers enabling the GCC feedback loop

---

## Datasets & Training Data

**Crate:** `brainwires-finetune` (feature `datasets-full` — absorbed from the deprecated `brainwires-datasets` crate)

Training data pipelines for fine-tuning workflows.

### I/O

- **JsonlReader** / **JsonlWriter** — Streaming JSONL I/O for training examples and preference pairs

### Data Types

- **TrainingExample** — Messages with roles and content
- **PreferencePair** — Chosen/rejected response pairs for RLHF/DPO
- **TrainingMessage** / **TrainingRole** — Message-level types

### Format Conversion

- **OpenAiFormat** — OpenAI fine-tuning format
- **TogetherFormat** — Together AI format
- **AlpacaFormat** — Alpaca instruction format
- **ShareGptFormat** — ShareGPT conversation format
- **ChatMlFormat** — ChatML format
- **detect_format()** — Automatic format detection

### Quality & Validation

- **DataValidator** — Configurable validation with `ValidationReport`
- **DatasetStats** — Token distributions, role counts, histogram buckets
- **Deduplicator** — Exact deduplication for examples and preference pairs (feature: `dedup`)

### Sampling

- **train_eval_split()** — Train/eval splitting with configurable ratios
- **curriculum_order()** — Curriculum learning ordering
- **sample_n()** — Random sampling

### Tokenization

- **HfTokenizer** — Hugging Face tokenizers (feature: `hf-tokenizer`)
- **TiktokenTokenizer** — OpenAI tiktoken (feature: `tiktoken`)

---

## Model Training & Fine-Tuning

**Crate:** `brainwires-finetune` (cloud only since v0.11). Local PEFT and
training-from-scratch live in the sibling `rullama` workspace as
`rullama-finetune` and `rullama-training`.

### Cloud Fine-Tuning (feature: `cloud`)

- **FineTuneProvider** trait — Unified interface for all cloud providers
- **FineTuneProviderFactory** — Create providers from config
- Supported providers: **OpenAI**, **Together**, **Fireworks**, **Anyscale**, **Bedrock**, **Vertex AI**

### Local Training (`rullama-finetune`, separate workspace)

- **LoRA** — Low-Rank Adaptation
- **QLoRA** — Quantized LoRA
- **DoRA** — Weight-Decomposed LoRA
- **Burn framework** — GPU-accelerated training via wgpu + ndarray backends
- **ComputeDevice** — CPU, GPU, or auto-detect
- **TrainedModelArtifact** — Output artifacts with SafeTensors weight loading

### Configuration

- **TrainingHyperparams** — Learning rate, epochs, batch size, warmup
- **LoraConfig** — Rank, alpha, dropout, target modules
- **AdapterMethod** — LoRA, QLoRA, DoRA selection
- **AlignmentMethod** — SFT, DPO, RLHF
- **LrScheduler** — Cosine, linear, constant, warmup

### Job Management

- **TrainingManager** — Job lifecycle management
- **TrainingJobStatus** — Queued, running, completed, failed
- **TrainingProgress** — Step counts, loss tracking, ETA

---

## Distributed Mesh Networking

**Crate:** `brainwires-network` (feature: `mesh`)

Connect agents across processes and machines.

- **MeshTopology** — Topology management with layout types (`TopologyType`)
- **MeshNode** — Node definitions with `NodeCapabilities` and `NodeState`
- **MessageRouter** — Message routing with multiple strategies (`RoutingStrategy`)
- **RouteEntry** — Route table entries
- **PeerDiscovery** — Peer discovery protocols (`DiscoveryProtocol`)
- **FederationGateway** — Cross-mesh communication with `FederationPolicy`

---

## Agent-to-Agent (A2A) Protocol

**Crate:** `brainwires-a2a`

Implementation of Google's A2A protocol for interoperable agent communication.

- **AgentCard** — Discovery metadata describing capabilities and skills
- **Task lifecycle** — Submission, execution tracking, artifact delivery (`TaskState`, `TaskSendParams`, `TaskQueryParams`)
- **Message types** — Text, file, and structured data parts (`Part`, `Artifact`)
- **Authentication** — Pluggable auth schemes: API key, OAuth2, JWT, Bearer
- **AgentProvider** / **AgentSkill** — Provider and skill metadata
- **JSON-RPC 2.0** — Full request/response envelopes with typed method constants
- **Push notifications** — `TaskPushNotificationConfig`, `AuthenticationInfo`
- **Streaming** — Server-Sent Events for real-time task updates
- **Client** — HTTP client with JSON-RPC and REST transports (feature: `client`)
- **Server** — HTTP server with JSON-RPC and REST routers (feature: `server`)
- **gRPC** — Protocol Buffers types, client transport, and server service (feature: `grpc`)

---

## Autonomous Operations

**Crate:** `brainwires-autonomy`

Self-improvement, Git workflows, and human-out-of-loop execution.

### Self-Improvement (feature: `self-improve`)

- **SelfImprovementController** — Autonomous improvement cycles
- **ImprovementStrategy** / **ImprovementCategory** — Strategy definitions
- **TaskGenerator** — Generate improvement tasks
- **Comparator** — Compare before/after results (`ComparisonResult`, `PathResult`)

### Eval-Driven Feedback (feature: `eval-driven`)

- **AutonomousFeedbackLoop** — Continuous evaluation and improvement
- **FeedbackLoopConfig** / **FeedbackLoopReport**
- **Empirical scoring eval cases** (`brainwires_autonomy::eval`) — validates scoring heuristics produce correct relative orderings via NDCG:
  - `EntityImportanceRankingCase` — hub vs. peripheral entity ranking
  - `EntitySingleMentionCase` — ln(1)=0 zero-contribution is compensated by type bonus
  - `EntityTypeBonusCase` — type-bonus ordering matches hardcoded priority table
  - `MultiFactorRankingCase` — 4 scenarios (similarity dominance, recency decay, fast decay, importance tiebreaker)
  - `TierDemotionCase` — `TierMetadata::retention_score` orders demotion candidates correctly
  - `entity_importance_suite()` / `multi_factor_suite()` — convenience constructors for `AutonomousFeedbackLoop`

### Git Workflow Pipeline (feature: `git-workflow`)

- **GitWorkflowPipeline** — Full pipeline: trigger -> investigate -> branch -> fix -> PR -> merge
- **GitForge** trait — Abstraction over GitHub, GitLab, etc.
- **IssueInvestigator** — Analyze issues to determine fix approach
- **BranchManager** — Branch creation and management
- **ChangeMaker** — Apply code changes
- **PullRequestManager** — PR creation and management
- **MergePolicy** — Automated merge decisions
- **WorkflowTrigger** — Event triggers (programmatic, webhook)
- **WebhookServer** — HTTP server for Git forge events (feature: `webhook`)

### Agent Operations

- **AgentSupervisor** — Health monitoring and recovery (feature: `supervisor`)
- **AttentionMechanism** — RAG-integrated attention (feature: `attention`)
- **ParallelCoordinator** — Parallel agent coordination with optional MDAP (feature: `parallel`)
- **HealthMonitor** — `HealthStatus`, `DegradationSignal`, `PerformanceMetrics`
- **HibernateManifest** — Session hibernation and resume

### Safety

- **SafetyGuard** — Safety checks for autonomous operations
- **ApprovalPolicy** — Human approval requirements
- **AutonomousOperation** — Operation classification

### Metrics

- **SessionMetrics** — Per-session performance tracking
- **SessionReport** — Summary reports

### Dream — Memory Consolidation (feature: `dream`)

- **DreamConsolidator** — 4-phase consolidation cycle: orient (scope selection) → gather (conversation sampling) → consolidate (LLM compression) → prune (demotion by policy)
- **DemotionPolicy** — Configurable thresholds for age, importance score, and memory budget
- **DreamSummarizer** — LLM-powered conversation compression; reduces working memory while preserving intent
- **FactExtractor** — Extracts durable knowledge into 5 categories: entities, relationships, events, preferences, habits
- **DreamMetrics** / **DreamReport** — Consolidation health tracking with per-phase timing and retention rates
- **DreamTask** — Wraps a consolidation run as a scheduled task via `AutonomyScheduler`

---

## Reasoning & Inference

**Crate:** `brainwires-reasoning` (facade: `brainwires::reasoning` behind the `reasoning` feature)

Provider-agnostic inference components for quality and cost optimization.

### Named Reasoning Strategies

- **ReasoningStrategy** trait — Common interface for reasoning loop control (`system_prompt()`, `is_complete()`, `next_action()`)
- **ReActStrategy** — Thought → Action → Observation loop (Yao et al., 2022) with configurable max steps
- **ReflexionStrategy** — Self-critique after each action with revised plans (Shinn et al., 2023)
- **ChainOfThoughtStrategy** — "Let's think step by step" structured reasoning (Wei et al., 2022)
- **TreeOfThoughtsStrategy** — Multi-branch exploration with pruning and best-path selection (Yao et al., 2023)
- **StrategyStep** — Typed reasoning trace steps: `Thought`, `Action`, `Observation`, `Reflection`, `Branch`

### Tier 1 — Quick Wins

- **LocalRouter** — Semantic query classification for tool routing
- **ComplexityScorer** — Task complexity scoring for adaptive MDAP k values
- **LocalValidator** — Response validation for red-flagging

### Tier 2 — Context & Retrieval

- **LocalSummarizer** — Context summarization for tiered memory demotion
- **RetrievalClassifier** — Enhanced retrieval gating with semantic understanding
- **RelevanceScorer** — Context re-ranking based on semantic relevance
- **StrategySelector** — Decomposition strategy selection for MDAP
- **EntityEnhancer** — Semantic entity extraction beyond regex patterns

All components accept `Arc<dyn Provider>` and fall back to pattern-based logic when unavailable.

---

## Evaluation Framework

**Module:** `brainwires-agent::eval` (feature: `eval`)

Monte Carlo evaluation framework for agent quality assurance.

- **EvaluationSuite** — N-trial Monte Carlo runner with `SuiteConfig`
- **EvaluationCase** trait — Single evaluatable scenario, with built-in helpers (`AlwaysPassCase`, `AlwaysFailCase`, `StochasticCase`)
- **TrialResult** / **EvaluationStats** — Per-trial results with Wilson-score 95% confidence intervals
- **ToolSequenceRecorder** — Record and diff tool call sequences (`SequenceDiff`)
- **AdversarialTestCase** — Prompt injection, ambiguity, budget stress tests
- **Regression tests** — Regression detection across versions
- **Stability tests** — Consistency checks
- **Fault reports** — Structured fault documentation

---

## Telemetry & Analytics

**Crate:** `brainwires-telemetry` (previously `brainwires-analytics` — renamed in the 0.10 consolidation)

Unified analytics collection, persistence, and querying — zero-friction observability for all framework components. Includes EU AI Act / GDPR compliance tooling.

### Event Types (`AnalyticsEvent`)

10 fully serializable typed event variants:

| Variant | Key fields |
|---------|-----------|
| `ProviderCall` | provider, model, prompt/completion tokens, cost, latency, success, `compliance?` |
| `AgentRun` | agent_id, task_id, iterations, tool calls, token totals, cost, duration, `compliance?` |
| `ToolCall` | agent_id, tool_name, tool_use_id, is_error, duration |
| `McpRequest` | server_name, tool_name, success, duration |
| `ChannelMessage` | channel_type, direction, message length |
| `StorageOp` | store_type, operation, success, duration |
| `NetworkMessage` | protocol, direction, bytes, success |
| `DreamCycle` | sessions processed, messages summarized, facts extracted, token reduction |
| `AutonomySession` | tasks attempted/succeeded/failed, total cost, duration |
| `Custom` | name, arbitrary JSON payload |

`ProviderCall` and `AgentRun` carry an optional `ComplianceMetadata` field (`#[serde(default)]` — backward-compatible with existing serialized events).

### Compliance Metadata (`ComplianceMetadata`)

Attach to `ProviderCall` / `AgentRun` events for EU AI Act, GDPR, HIPAA, SOC2 audit trails:

- `data_region` — ISO 3166-1 alpha-2 region (e.g. `"EU"`, `"US"`)
- `pii_present` — Whether the event payload may contain PII
- `retention_days` — Minimum retention period before deletion
- `regulation` — Applicable regulation (`"GDPR"`, `"HIPAA"`, `"EU_AI_ACT"`, etc.)
- `audit_required` — Include in compliance audit trail

### Collection

- **`AnalyticsCollector`** — Multi-sink dispatcher; call `record(event)` from any instrumented site. Clone-safe (`Arc`-backed).
- **`AnalyticsLayer`** — `tracing-subscriber` layer that automatically intercepts known span names (`provider.chat`, etc.) without modifying instrumented code. Register alongside your existing tracing setup.

### Sinks

- **`MemoryAnalyticsSink`** — In-process ring buffer (`DEFAULT_CAPACITY = 1_000`); useful for testing and dashboards. Helpers: `deposit()` (sync), `drain_matching(pred)`, `retain(pred)`.
- **`SqliteAnalyticsSink`** (feature `sqlite`) — Persists events to a local SQLite database at `<data_dir>/brainwires-telemetry/analytics.db`.

### Querying (feature `sqlite`)

- **`AnalyticsQuery`** — Aggregated reporting from the SQLite sink.
  - `cost_by_model(start, end)` → `Vec<CostByModelRow>` (model, total cost, call count)
  - `tool_frequency(start, end)` → `Vec<ToolFrequencyRow>` (tool name, call count, error count)
  - `daily_summary(start, end)` → `Vec<DailySummaryRow>` (date, calls, tokens, cost)
  - `rebuild_summaries()` — Refresh materialized summary tables

### Audit Export (`AuditExporter`)

Time-range filtered export from `MemoryAnalyticsSink`:

- `export_json(start, end)` — JSON array of matching events
- `export_csv(start, end)` — CSV with columns `event_type,session_id,timestamp,payload_json`
- `apply_retention_policy(days)` — Remove events older than N days; returns deleted count

### PII Redaction (`PiiRedactionRules` / `redact_event()`)

Configurable PII scrubbing before events reach storage sinks:

- `hash_session_ids` — Replace session IDs with a one-way hash (events remain groupable)
- `redact_prompt_content` — Replace `Custom` event payloads with `"[REDACTED]"`
- `custom_patterns` — Substring patterns; any matching string field is replaced with `"[REDACTED]"`
- `redact_event(event, rules)` — Pure function; returns a new scrubbed event

---

## Proxy Framework

**Crate:** `brainwires-proxy` *(extras/)*

Protocol-agnostic proxy for debugging AI API traffic.

- **ProxyBuilder** — Fluent API for proxy construction
- **ProxyService** — Core proxy engine
- **Transports:**
  - HTTP/HTTPS via hyper (feature: `http`)
  - WebSocket via tokio-tungstenite (feature: `websocket`)
  - TLS termination via tokio-rustls (feature: `tls`)
- **Middleware stack** — `ProxyLayer`, `LayerAction`, composable `MiddlewareStack`
- **Format conversion** — `Converter`, `StreamConverter`, `ConversionRegistry`, `FormatDetector`
- **Inspector API** — HTTP query API for captured traffic (feature: `inspector-api`)
- **Request tracking** — `RequestId` for correlating requests/responses

---

## WASM Bindings

**Crate:** `brainwires-wasm`

Browser-compatible WASM bindings via `wasm-bindgen`.

- **Message validation** — `validate_message()` for JSON message normalization
- **Tool validation** — `validate_tool()` for JSON tool definition validation
- **History serialization** — `serialize_history()` for stateless protocol format
- **Version** — `version()` for framework version
- **Code interpreters** — WASM interpreter support (feature: `interpreters`)
- **Orchestrator** — WASM orchestrator with execution limits (feature: `orchestrator`)

---

## Extras & Standalone Binaries

### voice-assistant *(extras/)*

Personal voice assistant binary built on `brainwires-hardware`. Mic capture → optional energy wake trigger → VAD-gated speech accumulation → OpenAI Whisper STT → LLM response (OpenAI chat completions) → OpenAI TTS playback. CLI: `--config <path.toml>`, `--list-devices`, `--wake-word <model>`, `--verbose`. TOML config covers STT model, TTS voice/model, silence tuning, wake word path, LLM model, system prompt, device names, and API key (or `OPENAI_API_KEY` env var). Graceful Ctrl-C shutdown.

### brainwires-issues *(extras/)*

Lightweight MCP-native issue tracking server inspired by Linear's agent interface. 10 tools: `create_issue` (title, description, priority, assignee, project, parent_id, labels), `get_issue` (UUID or `#number` display shorthand), `list_issues` (filter by project/status/assignee/label; offset-based pagination with `next_offset`), `update_issue`, `close_issue` (done or cancelled), `delete_issue` (optional comment cascade), `search_issues` (BM25 full-text; in-memory fallback), `add_comment`, `list_comments` (offset pagination), `delete_comment` (existence-checked). 4 prompts: `/create`, `/list`, `/search`, `/triage`. Data model: `Issue` (UUID + auto-incrementing display number, 6 status states, 5 priority levels, labels, assignee, project, parent_id for sub-issues, timestamps), `Comment`. Storage: LanceDB at `<data_dir>/brainwires-issues/lancedb/`; BM25 index at `<data_dir>/brainwires-issues/bm25/`.

### brainwires-brain-server *(extras/)*

MCP server binary wrapping `brainwires-knowledge::knowledge` for use with AI assistants (Claude Desktop, etc.). The underlying "brain" subsystem is now part of `brainwires-knowledge`.

### brainwires-rag-server *(extras/)*

MCP server binary wrapping `brainwires-knowledge::rag` (formerly the standalone `brainwires-rag` crate) for semantic code search via MCP protocol.

### agent-chat *(extras/)*

Minimal reference implementation of a chat client — small, readable, and purpose-built for learning the framework. Includes CLI commands for config, models, and auth. For a full-featured CLI, see `brainwires-cli` below.

### brainwires-cli *(extras/)*

Full-featured AI-powered agentic CLI with multi-agent orchestration (`TaskAgent`, `WorkerAgent`, `OrchestratorAgent`), MCP server mode (expose the CLI as an MCP tool server for hierarchical AI workflows), TUI (fullscreen ratatui interface), infinite context (LanceDB-backed semantic memory), extensive tool integration (file ops, bash, git, web, code search, validation), per-session model switching (`/model`), and support for all cloud providers (Anthropic, OpenAI, Google, Ollama, Groq, Together, Fireworks, Bedrock, Vertex AI). Migrated from a standalone repository; now a root workspace member at `extras/brainwires-cli/`.

### reload-daemon *(extras/)*

File-watching daemon for automatic server reloading during development.

### brainclaw *(extras/brainclaw/)*

Self-hosted personal AI assistant daemon. Multi-provider (Anthropic, OpenAI, Google, Ollama, etc.), per-user agent sessions, TOML config. Bundles the gateway, security middleware, and all channel adapters into a single service. Feature flags: `native-tools` (default), `email` (IMAP/SMTP/Gmail), `calendar` (Google Calendar/CalDAV).

### brainwires-gateway *(extras/brainclaw/)*

WebSocket/HTTP hub for routing channel adapters to AI agent sessions. `InboundHandler` trait for custom message processing; built-in `AgentInboundHandler` wires `ChatAgent` sessions per user. WebChat browser UI served at `/chat`. Media pipeline for attachment download, image description, and audio transcription. Admin API (`/admin/*`) with Bearer token auth. Admin browser UI at `/admin/ui` (dark-themed single-file dashboard; Dashboard, Channels, Sessions, Cron Jobs, Identity, Broadcast sections). Webhook endpoint with HMAC-SHA256 verification. Audit logger (structured JSON, ring buffer). In-memory metrics counters. **`/model` slash command** for per-session model switching stored in a `DashMap`; fires `/model list`, `/model <name>`, `/model default`.

### brainwires-discord-channel *(extras/brainclaw/)*

Discord channel adapter (serenity) implementing the `Channel` trait. Reference implementation for building additional platform adapters. Optional MCP tool server mode (`--mcp`) for programmatic Discord access.

### brainwires-telegram-channel *(extras/brainclaw/)*

Telegram channel adapter (teloxide) implementing the `Channel` trait. Bidirectional gateway relay. Optional MCP tool server mode (`--mcp`).

### brainwires-slack-channel *(extras/brainclaw/)*

Slack channel adapter using Socket Mode (reqwest) — no public URL required. Implements the `Channel` trait. Optional MCP tool server mode (`--mcp`).

### brainwires-mattermost-channel *(extras/brainclaw/)*

Mattermost channel adapter. Connects via Mattermost WebSocket API (`/api/v4/websocket`) for real-time events. Implements the `Channel` trait. Filtering: self-messages, channel allowlist, @mention requirement, team scoping. Optional MCP tool server mode (`--mcp`): `send_message`, `edit_message`, `delete_message`, `get_history`, `add_reaction`. Capabilities: `RICH_TEXT | THREADS | REACTIONS | TYPING_INDICATOR | EDIT_MESSAGES | DELETE_MESSAGES | MENTIONS`.

### brainwires-signal-channel *(extras/brainclaw/)*

Signal messenger channel adapter via `signal-cli-rest-api`. WebSocket push mode (`/v1/events`) with polling fallback (`GET /v1/receive/{number}`). Filtering: self-messages, sender allowlist (E.164 numbers), group allowlist (base64 IDs), @mention/keyword trigger for groups. Optional MCP tool server mode (`--mcp`): `send_message` (phone or `group.<id>`), `add_reaction` (composite `recipient:author:timestamp` ID). Capabilities: `REACTIONS`.

### brainwires-skill-registry *(extras/brainclaw/)*

HTTP skill registry server. SQLite with FTS5 full-text search. Endpoints: publish (`POST /api/skills`), search by query + tags, get manifest (latest or versioned), download package. Schema auto-created on first run.

---

## Facade Crate & Feature Flags

**Crate:** `brainwires`

Re-exports all framework crates behind feature flags.

| Feature | Default | Description |
|---------|---------|-------------|
| `core` | Always | Core types and traits |
| `tools` | Yes | Tool definitions and execution |
| `agents` | Yes | Multi-agent orchestration |
| `providers` | No | AI provider integrations |
| `chat` | No | Chat provider wrappers (alias for `providers`) |
| `storage` | No | Vector storage and semantic search |
| `mcp` | No | MCP client support |
| `mcp-server` | No | MCP server re-exports (rmcp, schemars) |
| `mdap` | No | MDAP voting framework |
| `prompting` | No | Adaptive prompting techniques |
| `knowledge` | No | BKS/PKS knowledge systems (alias for `brain`) |
| `brain` | No | Central knowledge crate |
| `permissions` | No | Capability-based permissions |
| `seal` | No | Self-Evolving Agentic Learning |
| `agent-network` | No | MCP server, IPC, remote bridge |
| `rag` | No | RAG engine with code search |
| `rag-full-languages` | No | RAG + all Tree-sitter language parsers |
| `interpreters` | No | Sandboxed code interpreters |
| `orchestrator` | No | Rhai script orchestration |
| `reasoning` | No | Local inference components |
| `openapi` | No | OpenAPI 3.x spec → Tool generation |
| `otel` | No | OpenTelemetry span export for agent traces |
| `eval` | No | Evaluation framework |
| `skills` | No | SKILL.md skill system |
| `audio` | No | Audio capture, STT, TTS |
| `gpio` | No | GPIO pin control with safety allow-lists (Linux) |
| `bluetooth` | No | BLE advertisement scanning and adapter enumeration |
| `network-hardware` | No | NIC enumeration, IP config, ARP discovery, port scanning |
| `camera` | No | Webcam/camera frame capture (V4L2/AVFoundation/MSMF) |
| `usb` | No | Raw USB device enumeration and transfers (no libusb) |
| `datasets` | No | Training data pipelines |
| `training` | No | Model training (base types) |
| `training-cloud` | No | Cloud fine-tuning providers |
| `training-local` | No | Local LoRA/QLoRA/DoRA training |
| `training-full` | No | All training + all datasets |
| `channels` | No | Universal messaging channel contract (Channel trait, message/event types) |
| `mcp-server-framework` | No | MCP server building blocks (McpServer, McpHandler, middleware pipeline) |
| `autonomy` | No | Autonomous operations |
| `dream` | No | Autodream memory consolidation (requires `autonomy`) |
| `mesh` | No | Distributed agent mesh |
| `a2a` | No | Agent-to-Agent protocol |
| `proxy` | No | Protocol proxy framework |
| `wasm` | No | WASM browser bindings |
| `bedrock` | No | Amazon Bedrock provider |
| `vertex-ai` | No | Google Vertex AI provider |
| `llama-cpp-2` | No | Local LLM inference |
| `learning` | No | SEAL + knowledge integration |
| `agent-full` | No | agents + permissions + prompting + tools |
| `researcher` | No | providers + agents + storage + rag + training + datasets |
| `full` | No | Everything enabled |
