/**
 * @module @brainwires/core
 *
 * Foundation types, traits, and error handling for the Brainwires Agent Framework.
 * Equivalent to Rust's `brainwires-core` crate.
 */

// Content source types
export {
  canOverride,
  requiresSanitization,
  type ContentSource,
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
  edgeTypeWeight,
  type EdgeType,
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
  eventToolName,
  eventType,
  filterMatches,
  HookRegistry,
  type EventFilter,
  type HookResult,
  type LifecycleEvent,
  type LifecycleHook,
} from "./lifecycle.ts";

// Message types
export {
  createUsage,
  Message,
  serializeMessagesToStatelessHistory,
  type ChatResponse,
  type ContentBlock,
  type ImageBlock,
  type ImageSource,
  type MessageContent,
  type MessageData,
  type Role,
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
  SerializablePlan,
  type PlanStatus,
  type PlanStep,
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
  Task,
  TASK_PRIORITY_VALUES,
  type AgentResponse,
  type TaskPriority,
  type TaskStatus,
} from "./task.ts";

// Tool types
export {
  defaultToolInputSchema,
  IdempotencyRegistry,
  objectSchema,
  ToolContext,
  toolModeDisplayName,
  ToolResult,
  type CommitResult,
  type IdempotencyRecord,
  type StagedWrite,
  type StagingBackend,
  type Tool,
  type ToolCaller,
  type ToolInputSchema,
  type ToolMode,
  type ToolUse,
} from "./tool.ts";

// Output parsers
export {
  extractJson,
  JsonListParser,
  JsonOutputParser,
  RegexOutputParser,
  type OutputParser,
} from "./output_parser.ts";

// Vector store types
export {
  type VectorSearchResult,
  type VectorStore,
} from "./vector_store.ts";

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
