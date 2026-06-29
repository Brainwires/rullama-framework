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
  { name: "brainwires", description: "Unified facade crate — build any AI application in Rust.", tier: "facade", features: ["full", "agent-full", "researcher", "learning"] },
  { name: "brainwires-core", description: "Core types, traits, and error handling.", tier: "foundation" },
  { name: "brainwires-a2a", description: "Full Rust implementation of the Agent-to-Agent (A2A) protocol — the open standard for interoperable agent communication.", tier: "foundation" },
  { name: "brainwires-code-interpreters", description: "Sandboxed code execution for multiple languages (Rhai, Lua, JavaScript, Python).", tier: "foundation" },
  { name: "brainwires-skills", description: "Composable agent skills system.", tier: "foundation" },
  { name: "brainwires-permissions", description: "Capability-based permission system with policy engine, audit logging, trust management, and anomaly detection.", tier: "foundation" },
  { name: "brainwires-providers", description: "AI provider implementations: Anthropic, OpenAI, Google, Ollama, llama.cpp.", tier: "intelligence", features: ["anthropic", "openai", "google", "ollama", "llama-cpp"] },
  { name: "brainwires-agents", description: "Agent orchestration, coordination, and lifecycle management.", tier: "intelligence", features: ["tools", "mcp"] },
  { name: "brainwires-cognition", description: "Unified intelligence layer — knowledge graphs, adaptive prompting, RAG, spectral math, and code analysis.", tier: "intelligence", features: ["knowledge", "prompting", "rag", "code-search"] },
  { name: "brainwires-training", description: "Model training and fine-tuning — cloud fine-tuning and local LoRA/QLoRA/DoRA training.", tier: "intelligence", features: ["training-cloud", "training-local", "training-full"] },
  { name: "brainwires-storage", description: "Backend-agnostic storage, tiered memory, and document management across 9 backends.", tier: "intelligence", features: ["storage-sqlite", "storage-postgres", "storage-redis", "storage-lancedb"] },
  { name: "brainwires-tool-system", description: "Tooling layer: file ops, shell, Git, web access, code search, validation, transactions. Composable — register only what you need.", tier: "intelligence" },
  { name: "brainwires-datasets", description: "Training data pipelines — JSONL I/O, tokenization, deduplication, format conversion.", tier: "intelligence" },
  { name: "brainwires-autonomy", description: "Autonomous agent operations — self-improvement, Git workflows, environment interaction, and human-out-of-loop execution.", tier: "intelligence" },
  { name: "brainwires-hardware", description: "Hardware I/O — audio TTS/STT/VAD, GPIO, Bluetooth, USB, camera, wake word.", tier: "hardware", features: ["audio", "gpio", "bluetooth", "usb", "camera", "vad", "wake-word"] },
  { name: "brainwires-mcp", description: "MCP client, transport, and protocol types.", tier: "network", features: ["mcp-client", "mcp-sse", "mcp-stdio"] },
  { name: "brainwires-agent-network", description: "Agent networking layer — discovery, routing, and coordination across agent clusters.", tier: "network" },
  { name: "brainwires-channels", description: "Universal messaging channel contract — Discord, Telegram, Slack, Signal, Matrix, and more.", tier: "network", features: ["discord", "telegram", "slack", "signal", "matrix", "mattermost"] },
  { name: "brainwires-mcp-server", description: "MCP server framework with composable middleware.", tier: "network" },
  { name: "brainwires-analytics", description: "Unified analytics collection, persistence, and querying.", tier: "utility" },
  { name: "brainwires-wasm", description: "WebAssembly bindings for the Brainwires Agent Framework.", tier: "utility" },
];

export const EXTRAS_LIST: ExtraMeta[] = [
  { name: "brainwires-cli", description: "AI-powered agentic CLI tool for autonomous coding assistance, built in Rust.", hasReadme: true },
  { name: "brainwires-proxy", description: "HTTP protocol proxy for the Brainwires Framework.", hasReadme: true },
  { name: "brainwires-brain-server", description: "Standalone MCP server binary for the Open Brain knowledge system.", hasReadme: true },
  { name: "brainwires-rag-server", description: "Standalone MCP server binary for codebase RAG (Retrieval-Augmented Generation).", hasReadme: true },
  { name: "brainwires-issues", description: "Standalone MCP server binary for lightweight project issue and bug tracking.", hasReadme: true },
  { name: "agent-chat", description: "Minimal reference implementation — a small, readable example of building a chat client on the Brainwires Framework.", hasReadme: true },
  { name: "audio-demo", description: "Cross-platform desktop GUI (Avalonia .NET 9) for demoing TTS and STT across all Brainwires audio providers.", hasReadme: true },
  { name: "audio-demo-ffi", description: "UniFFI bindings for brainwires-hardware exposing audio APIs to C#, Kotlin, Swift, and Python.", hasReadme: true },
  { name: "reload-daemon", description: "Minimal MCP server that enables AI coding clients to kill and restart themselves with transformed arguments.", hasReadme: true },
  { name: "brainclaw", description: "Multi-provider personal AI assistant daemon with gateway and channel adapters.", hasReadme: false },
  { name: "voice-assistant", description: "Voice-driven assistant built on the Brainwires hardware and agents stack.", hasReadme: false },
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
