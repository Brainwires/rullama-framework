// deno-lint-ignore-file no-explicit-any

/** Specifies which contexts can invoke a tool.
 * Equivalent to Rust's `ToolCaller` in rullama-core. */
export type ToolCaller = "direct" | "code_execution";

/** A tool that can be used by the AI agent.
 * Equivalent to Rust's `Tool` in rullama-core. */
export interface Tool {
  name: string;
  description: string;
  input_schema: ToolInputSchema;
  requires_approval?: boolean;
  defer_loading?: boolean;
  allowed_callers?: ToolCaller[];
  input_examples?: any[];
}

/** JSON Schema for tool input.
 * Equivalent to Rust's `ToolInputSchema` in rullama-core. */
export interface ToolInputSchema {
  type: string;
  properties?: Record<string, any>;
  required?: string[];
}

/** Create a default ToolInputSchema. */
export function defaultToolInputSchema(): ToolInputSchema {
  return { type: "object" };
}

/** Create an object schema with properties and required fields. */
export function objectSchema(
  properties: Record<string, any>,
  required: string[],
): ToolInputSchema {
  return { type: "object", properties, required };
}

/** A tool use request from the AI.
 * Equivalent to Rust's `ToolUse` in rullama-core. */
export interface ToolUse {
  id: string;
  name: string;
  input: any;
}

/** Result of a tool execution.
 * Equivalent to Rust's `ToolResult` in rullama-core. */
export class ToolResult {
  tool_use_id: string;
  content: string;
  is_error: boolean;

  constructor(toolUseId: string, content: string, isError: boolean) {
    this.tool_use_id = toolUseId;
    this.content = content;
    this.is_error = isError;
  }

  /** Create a successful tool result. */
  static success(toolUseId: string, content: string): ToolResult {
    return new ToolResult(toolUseId, content, false);
  }

  /** Create an error tool result. */
  static error(toolUseId: string, error: string): ToolResult {
    return new ToolResult(toolUseId, error, true);
  }
}

/** Record of a completed idempotent write operation.
 * Equivalent to Rust's `IdempotencyRecord` in rullama-core. */
export interface IdempotencyRecord {
  executed_at: number;
  cached_result: string;
}

/** Shared registry that deduplicates mutating file-system tool calls within a run.
 * Equivalent to Rust's `IdempotencyRegistry` in rullama-core. */
export class IdempotencyRegistry {
  private records: Map<string, IdempotencyRecord> = new Map();

  /** Return the cached result for `key`, or undefined if not yet executed. */
  get(key: string): IdempotencyRecord | undefined {
    return this.records.get(key);
  }

  /** Record that `key` produced `result`. First result wins. */
  record(key: string, result: string): void {
    if (!this.records.has(key)) {
      this.records.set(key, {
        executed_at: Math.floor(Date.now() / 1000),
        cached_result: result,
      });
    }
  }

  /** Number of recorded operations. */
  get length(): number {
    return this.records.size;
  }

  /** Returns true if no operations have been recorded yet. */
  isEmpty(): boolean {
    return this.records.size === 0;
  }
}

/** A single write operation that has been staged but not yet committed.
 * Equivalent to Rust's `StagedWrite` in rullama-core. */
export interface StagedWrite {
  key: string;
  target_path: string;
  content: string;
}

/** Result returned by a successful commit.
 * Equivalent to Rust's `CommitResult` in rullama-core. */
export interface CommitResult {
  committed: number;
  paths: string[];
}

/** Interface for staging write operations before committing to the filesystem.
 * Equivalent to Rust's `StagingBackend` trait in rullama-core. */
export interface StagingBackend {
  stage(write: StagedWrite): boolean;
  commit(): CommitResult;
  rollback(): void;
  pendingCount(): number;
}

/** Execution context for a tool.
 * Equivalent to Rust's `ToolContext` in rullama-core. */
export class ToolContext {
  working_directory: string;
  user_id?: string;
  metadata: Record<string, string>;
  capabilities?: any;
  idempotency_registry?: IdempotencyRegistry;
  staging_backend?: StagingBackend;

  constructor(opts?: Partial<ToolContext>) {
    this.working_directory = opts?.working_directory ?? Deno.cwd();
    this.user_id = opts?.user_id;
    this.metadata = opts?.metadata ?? {};
    this.capabilities = opts?.capabilities;
    this.idempotency_registry = opts?.idempotency_registry;
    this.staging_backend = opts?.staging_backend;
  }

  /** Attach a fresh idempotency registry (builder pattern). */
  withIdempotencyRegistry(): this {
    this.idempotency_registry = new IdempotencyRegistry();
    return this;
  }

  /** Attach a staging backend (builder pattern). */
  withStagingBackend(backend: StagingBackend): this {
    this.staging_backend = backend;
    return this;
  }
}

/** Tool selection mode.
 * Equivalent to Rust's `ToolMode` in rullama-core. */
export type ToolMode =
  | { type: "full" }
  | { type: "explicit"; tools: string[] }
  | { type: "smart" }
  | { type: "core" }
  | { type: "none" };

/** Get a display name for a ToolMode. */
export function toolModeDisplayName(mode: ToolMode): string {
  return mode.type === "explicit" ? "explicit" : mode.type;
}
