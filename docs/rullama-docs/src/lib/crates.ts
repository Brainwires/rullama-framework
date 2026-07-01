export type CrateTier = "facade" | "foundation" | "intelligence" | "hardware" | "network" | "utility";

export interface CrateMeta {
  name: string;
  description: string;
  tier: CrateTier;
  features?: string[];
}

export interface ExtraMeta {
  name: string;
  description: string;
  hasReadme: boolean;
}

export const CRATE_LIST: CrateMeta[] = [
  { name: "rullama", description: "Unified facade crate — build any AI application in Rust.", tier: "facade", features: ["full", "agent-full", "researcher", "learning"] },
  { name: "rullama-core", description: "Core types, traits, and error handling for the agent framework.", tier: "foundation" },
  { name: "rullama-a2a", description: "Agent-to-Agent (A2A) protocol — JSON-RPC, REST, and gRPC bindings for interoperable agent communication.", tier: "foundation" },
  { name: "rullama-skills", description: "SKILL.md skills system — manifest parsing, registry, smart routing, sandboxed execution, optional ed25519 signing.", tier: "foundation" },
  { name: "rullama-permission", description: "Permission policies, audit logging, and trust profiles.", tier: "foundation" },
  { name: "rullama-call-policy", description: "Resilience middleware — retry, circuit breaker, budget, cache, and error classification.", tier: "foundation" },
  { name: "rullama-sandbox", description: "Container-based sandboxing (Docker / Podman / host) for tool execution.", tier: "foundation" },
  { name: "rullama-provider", description: "LLM provider implementations — Anthropic, OpenAI (Chat + Responses), Google Gemini, Ollama, Bedrock, Vertex AI, local llama.cpp.", tier: "intelligence", features: ["anthropic", "openai", "google", "ollama", "llama-cpp"] },
  { name: "rullama-provider-speech", description: "Speech (TTS / STT) provider clients — Azure, Cartesia, Deepgram, ElevenLabs, Fish Audio, Google, Murf, and browser web-speech.", tier: "intelligence" },
  { name: "rullama-agent", description: "Agent coordination primitives + multi-agent patterns — communication hub, locks, task queue, contract net, saga, workflow graph.", tier: "intelligence" },
  { name: "rullama-inference", description: "LLM-driven workhorses — chat agent, planner / judge / validator, task agent, cycle orchestrator, summarization.", tier: "intelligence", features: ["tools", "mcp"] },
  { name: "rullama-knowledge", description: "Knowledge layer — knowledge graphs, BKS/PKS, brain client, and entity extraction.", tier: "intelligence" },
  { name: "rullama-prompting", description: "Adaptive prompting techniques, K-means task clustering, and temperature optimization.", tier: "intelligence" },
  { name: "rullama-rag", description: "Codebase indexing + hybrid retrieval (vector + BM25) — AST-aware chunking (tree-sitter, 12 languages), Git history search, reranking.", tier: "intelligence", features: ["code-search"] },
  { name: "rullama-reasoning", description: "Reasoning layer — plan/output parsers and provider-agnostic local-inference scorers (router, complexity, relevance, strategy, validator).", tier: "intelligence" },
  { name: "rullama-mdap", description: "Multi-Dimensional Adaptive Planning (MAKER voting) — k-out-of-n voting, recursive decomposition, microagent dispatch, red-flag validation.", tier: "intelligence" },
  { name: "rullama-memory", description: "Tiered hot/warm/cold agent memory orchestration, plus the `dream` offline consolidation engine.", tier: "intelligence" },
  { name: "rullama-seal", description: "Self-Evolving Agentic Learning (SEAL) — coreference resolution, query-core extraction, learned-pattern store, reflection.", tier: "intelligence" },
  { name: "rullama-finetune", description: "Cloud fine-tune APIs — OpenAI, Anthropic, Together, Fireworks, Anyscale, Bedrock, Vertex AI.", tier: "intelligence", features: ["training-cloud"] },
  { name: "rullama-datasets", description: "Training data pipelines — JSONL I/O, tokenization, dedup, and format conversion (Alpaca / ChatML / OpenAI / ShareGPT / Together).", tier: "intelligence" },
  { name: "rullama-eval", description: "Evaluation harness — fixtures, regression suites, stability + adversarial tests, ranking metrics (NDCG / MRR / precision@k).", tier: "intelligence" },
  { name: "rullama-tool-runtime", description: "Tool execution runtime — executor trait, registry, error taxonomy, sanitization, validation, transactions, smart router.", tier: "intelligence" },
  { name: "rullama-tool-builtins", description: "Built-in concrete tools — bash, file ops, git, web, search, code exec, semantic search, browser, email, calendar, system.", tier: "intelligence" },
  { name: "rullama-storage", description: "Backend-agnostic storage, tiered memory, and document management.", tier: "intelligence", features: ["storage-sqlite", "storage-postgres", "storage-redis", "storage-lancedb"] },
  { name: "rullama-stores", description: "Opinionated minimum data-store set — sessions, conversations, tasks, plans, locks, images, and tiered memory.", tier: "intelligence" },
  { name: "rullama-hardware", description: "Hardware I/O — audio TTS/STT/VAD, GPIO, Bluetooth, and network.", tier: "hardware", features: ["audio", "gpio", "bluetooth", "vad", "wake-word"] },
  { name: "rullama-mcp-client", description: "MCP client, transport, and protocol types.", tier: "network", features: ["mcp-sse", "mcp-stdio"] },
  { name: "rullama-mcp-server", description: "MCP server framework with a composable middleware pipeline.", tier: "network" },
  { name: "rullama-network", description: "Agent-to-agent networking — IPC, remote bridge, mesh networking, routing, and discovery.", tier: "network" },
  { name: "rullama-telemetry", description: "Unified telemetry — analytics events, SQLite persistence, cost/usage queries, and a `BillingHook` trait.", tier: "utility" },
  { name: "rullama-sandbox-proxy", description: "Egress allowlist proxy used by the Docker sandbox to enforce a limited network policy.", tier: "utility" },
  { name: "rullama-test-fixtures", description: "Internal test fixtures — shared mock providers, recording wrappers, tool registries, and sandbox helpers.", tier: "utility" },
  { name: "rullama-test-harness", description: "Cross-crate test harness — feature determinism, security adversarial, and golden-path assembly tiers.", tier: "utility" },
];

export const EXTRAS_LIST: ExtraMeta[] = [
  // sdks/
  { name: "rullama-autonomy", description: "Autonomous agent operations — self-improvement, Git workflows, and human-out-of-loop execution.", hasReadme: true },
  { name: "rullama-billing", description: "Full billing implementation for rullama agents — ledger, wallet, and Stripe integration.", hasReadme: true },
  { name: "rullama-proxy", description: "Protocol-agnostic proxy framework for debugging app traffic.", hasReadme: true },
  { name: "rullama-wasm", description: "WebAssembly bindings for rullama.", hasReadme: true },
  // servers/
  { name: "rullama-brain-server", description: "Open Brain MCP server binary — exposes persistent knowledge (thoughts, PKS, BKS) to any AI tool over MCP.", hasReadme: true },
  { name: "rullama-issues", description: "Issue-tracking MCP server — lightweight project issue and bug tracking.", hasReadme: true },
  { name: "rullama-memory-server", description: "Mem0-compatible memory REST API server for rullama agents.", hasReadme: true },
  { name: "rullama-rag-server", description: "Project-RAG MCP server binary — RAG-based codebase indexing and semantic search.", hasReadme: true },
  { name: "rullama-scheduler", description: "Local-machine MCP scheduler — cron-based job scheduling with optional Docker sandboxing.", hasReadme: true },
  // integrations/
  { name: "claude-brain", description: "Claude Code context manager — replaces compaction with rullama tiered memory, dream consolidation, and semantic recall.", hasReadme: true },
  { name: "reload-daemon", description: "Minimal MCP server that lets AI coding clients kill and restart themselves with transformed arguments.", hasReadme: true },
  // examples/
  { name: "agent-chat", description: "Simplified AI chat client — a small, readable example built on rullama.", hasReadme: true },
  { name: "audio-demo", description: "Cross-platform desktop GUI (Avalonia .NET 9) demoing TTS and STT across all rullama audio providers.", hasReadme: true },
  { name: "audio-demo-ffi", description: "UniFFI bindings for rullama-hardware, exposing TTS/STT to C#, Kotlin, Swift, and Python.", hasReadme: true },
  { name: "rullama-chat-native", description: "Native chat example built on rullama.", hasReadme: true },
  { name: "rullama-web-search-agent", description: "Minimal end-to-end example — ChatAgent + WebTool + BudgetProvider answering a question from the open web.", hasReadme: true },
  { name: "rullama-webchat", description: "Web chat example built on rullama.", hasReadme: true },
  { name: "voice-assistant", description: "Personal voice assistant built on the rullama framework.", hasReadme: true },
];

export const TIER_LABELS: Record<CrateTier, string> = {
  facade: "Facade", foundation: "Foundation", intelligence: "Intelligence",
  hardware: "Hardware", network: "Network", utility: "Utility",
};

export const TIER_COLORS: Record<CrateTier, string> = {
  facade: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
  foundation: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  intelligence: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
  hardware: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
  network: "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-200",
  utility: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200",
};
