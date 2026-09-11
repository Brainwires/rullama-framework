// deno-lint-ignore-file no-explicit-any

/** Specifies which contexts can invoke a tool.
 * Equivalent to Rust's `ToolCaller` in rullama-core. */
export type ToolCaller = "direct" | "code_execution";

/** A tool that can be used by the AI agent.
 * Equivalent to Rust's `Tool` in rullama-core. */
export interface Tool {
  /** Unique name the model uses to call the tool. */
  name: string;
  /** What the tool does and when to use it, shown to the model. */
  description: string;
  /** JSON Schema for the tool's input arguments. */
  input_schema: ToolInputSchema;
  /** When true, a call must be approved by the user before it runs. */
  requires_approval?: boolean;
  /** When true, the tool is not loaded up front; it is surfaced only via tool search. */
  defer_loading?: boolean;
  /** Contexts allowed to invoke the tool; unrestricted when omitted. */
  allowed_callers?: ToolCaller[];
  /** Example input objects that show the model valid argument shapes. */
  input_examples?: any[];
}

/** JSON Schema for tool input.
 * Equivalent to Rust's `ToolInputSchema` in rullama-core. */
export interface ToolInputSchema {
  /** JSON Schema `type` keyword (normally `"object"`). */
  type: string;
  /** JSON Schema for each named argument. */
  properties?: Record<string, any>;
  /** Names of the arguments the model must supply. */
  required?: string[];
}

/** Create a default ToolInputSchema. */
export function defaultToolInputSchema(): ToolInputSchema {
  return { type: "object" };
}

/**
 * The JSON Schema object a provider must send for a tool's input: always
 * `type: "object"` with `properties` and `required` present. Five of the six
 * providers used to send `input_schema.properties` alone, dropping `type` and
 * `required`, so models were never told which arguments were mandatory.
 */
export function toolInputJsonSchema(
  schema: ToolInputSchema,
): { type: "object"; properties: Record<string, any>; required: string[] } {
  return {
    type: "object",
    properties: schema.properties ?? {},
    required: schema.required ?? [],
  };
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
  /** Provider-assigned ID that the matching {@link ToolResult} echoes back. */
  id: string;
  /** Name of the tool to invoke. */
  name: string;
  /** Arguments the model supplied, shaped by the tool's input schema. */
  input: any;
}

/** Result of a tool execution.
 * Equivalent to Rust's `ToolResult` in rullama-core. */
export class ToolResult {
  /** ID of the {@link ToolUse} this result answers. */
  tool_use_id: string;
  /** Output text returned to the model (or the error message when `is_error`). */
  content: string;
  /** True when the execution failed and `content` holds the error message. */
  is_error: boolean;

  /**
   * Build a result for a tool call; prefer {@link ToolResult.success} and
   * {@link ToolResult.error}.
   * @param toolUseId ID of the tool call being answered.
   * @param content Output text, or the error message when `isError` is true.
   * @param isError Whether the call failed.
   */
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
  /** Unix timestamp (seconds) when the operation first ran. */
  executed_at: number;
  /** Result text of the first execution, replayed for later duplicate calls. */
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
  /** Idempotency key identifying the write; a repeat of the same key is not staged twice. */
  key: string;
  /** Filesystem path the content will be written to on commit. */
  target_path: string;
  /** Full file content to write. */
  content: string;
}

/** Result returned by a successful commit.
 * Equivalent to Rust's `CommitResult` in rullama-core. */
export interface CommitResult {
  /** Number of staged writes that were applied. */
  committed: number;
  /** Target paths of the applied writes. */
  paths: string[];
}

/** Interface for staging write operations before committing to the filesystem.
 * Equivalent to Rust's `StagingBackend` trait in rullama-core. */
export interface StagingBackend {
  /** Queue a write; returns false when a write with the same key is already staged. */
  stage(write: StagedWrite): boolean;
  /** Apply every pending write to the filesystem and clear the queue. */
  commit(): CommitResult;
  /** Discard every pending write without applying it. */
  rollback(): void;
  /** Number of writes staged but not yet committed. */
  pendingCount(): number;
}

/** Execution context for a tool.
 * Equivalent to Rust's `ToolContext` in rullama-core. */
export class ToolContext {
  /** Directory relative paths resolve against; defaults to `Deno.cwd()`. */
  working_directory: string;
  /** Identifier of the user on whose behalf the tool runs, if known. */
  user_id?: string;
  /** Free-form string key/value pairs tools read for out-of-band configuration (session IDs, calendar config, ...). */
  metadata: Record<string, string>;
  /** Capability profile that narrows what the tool may do; `any` because the profile type lives in a downstream package. */
  capabilities?: any;
  /** Registry that de-duplicates mutating calls within a run, when attached. */
  idempotency_registry?: IdempotencyRegistry;
  /** Backend that stages writes for a later commit, when attached. */
  staging_backend?: StagingBackend;

  /**
   * Create a context, filling every field left out of `opts` with its default
   * (`working_directory` = `Deno.cwd()`, `metadata` = `{}`).
   */
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
