/**
 * @module @rullama/core
 *
 * Foundation types, traits, and error handling for the rullama.
 * Equivalent to Rust's `rullama-core` crate.
 */

// Content source types
export {
  canOverride,
  type ContentSource,
  requiresSanitization,
} from "./content_source.ts";

// Embedding provider
export { type EmbeddingProvider } from "./embedding.ts";

// Framework errors
export { FrameworkError, type FrameworkErrorKind } from "./error.ts";

// Correlation / tracing envelope
export { type Event, EventEnvelope, newTraceId } from "./event.ts";

// Workflow checkpoint / crash-safe retry
export {
  defaultWorkflowStatePath,
  FsWorkflowStateStore,
  InMemoryWorkflowStateStore,
  isCompleted,
  newCheckpoint,
  newSideEffectRecord,
  type SideEffectRecord,
  type WorkflowCheckpoint,
  type WorkflowStateStore,
} from "./workflow_state.ts";

// Knowledge graph types
export {
  type EdgeType,
  edgeTypeWeight,
  type EntityStoreT,
  type EntityType,
  type GraphEdge,
  type GraphNode,
  type RelationshipGraphT,
} from "./graph.ts";

// Lifecycle hooks
export {
  defaultEventFilter,
  eventAgentId,
  type EventFilter,
  eventToolName,
  eventType,
  filterMatches,
  HookRegistry,
  type HookResult,
  type LifecycleEvent,
  type LifecycleHook,
} from "./lifecycle.ts";

// Message types
export {
  type ChatResponse,
  type ContentBlock,
  createUsage,
  type ImageBlock,
  type ImageSource,
  Message,
  type MessageContent,
  type MessageData,
  type Role,
  serializeMessagesToStatelessHistory,
  type StreamChunk,
  type TextBlock,
  type ToolResultBlock,
  type ToolUseBlock,
  type Usage,
} from "./message.ts";

// Permission modes
export {
  DEFAULT_PERMISSION_MODE,
  parsePermissionMode,
  type PermissionMode,
} from "./permission.ts";

// Plan types
export {
  parsePlanStatus,
  PlanBudget,
  PlanMetadata,
  type PlanStatus,
  type PlanStep,
  SerializablePlan,
} from "./plan.ts";

// Provider types
export { ChatOptions, type Provider } from "./provider.ts";

// Search types
export {
  type ChunkMetadata,
  type DatabaseStats,
  type SearchResult,
} from "./search.ts";

// Task types
export {
  type AgentResponse,
  Task,
  TASK_PRIORITY_VALUES,
  type TaskPriority,
  type TaskStatus,
} from "./task.ts";

// Tool types
export {
  type CommitResult,
  defaultToolInputSchema,
  type IdempotencyRecord,
  IdempotencyRegistry,
  objectSchema,
  type StagedWrite,
  type StagingBackend,
  type Tool,
  type ToolCaller,
  ToolContext,
  type ToolInputSchema,
  type ToolMode,
  toolModeDisplayName,
  ToolResult,
  type ToolUse,
} from "./tool.ts";

// Output parsers
export {
  extractJson,
  JsonListParser,
  JsonOutputParser,
  type OutputParser,
  RegexOutputParser,
} from "./output_parser.ts";

// Vector store types
export { type VectorSearchResult, type VectorStore } from "./vector_store.ts";

// Working set
export {
  DEFAULT_MAX_FILES,
  DEFAULT_MAX_TOKENS,
  defaultWorkingSetConfig,
  estimateTokens,
  estimateTokensFromSize,
  WorkingSet,
  type WorkingSetConfig,
  type WorkingSetEntry,
} from "./working_set.ts";

// Confidence (moved from agents → core in v0.11.0)
export {
  type ConfidenceFactors,
  confidenceLevel,
  defaultConfidenceFactors,
  defaultResponseConfidence,
  extractConfidence,
  isHighConfidence,
  isLowConfidence,
  quickConfidenceCheck,
  type ResponseConfidence,
  weakestFactor,
} from "./confidence.ts";

// Platform paths (moved from storage → core in v0.11.0)
export { PlatformPaths } from "./paths.ts";

// File context manager (moved from storage → core in v0.11.0)
export {
  type FileChunk,
  type FileContent,
  FileContextManager,
} from "./file_context.ts";
