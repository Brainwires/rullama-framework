/**
 * Enforcing tool executor — where permissions become real.
 *
 * `@rullama/permission` ships a policy engine, capability profiles and approval
 * types; `@rullama/tool-runtime` ships output sanitization and a `ToolPreHook`
 * seam. Until 0.12.1 nothing called any of them: every `Tool.requires_approval`
 * flag, every policy rule and every sanitizer was advisory. {@link EnforcingExecutor}
 * wraps any {@link ToolExecutor} and applies all of them, in this order, before
 * and after each call:
 *
 * 1. **Permission mode** — `"read-only"` admits only read/search/planning tools.
 * 2. **Capabilities** — an optional {@link AgentCapabilities} profile narrows
 *    tools, file paths, domains and git operations.
 * 3. **Policy** — the {@link PolicyEngine} decides allow / deny / require approval
 *    (default: `PolicyEngine.withDefaults()`, which denies `.env`, secret and
 *    credential files and asks before `git reset` / `git rebase`).
 * 4. **Pre-execute hooks** — every {@link ToolPreHook} may still reject.
 * 5. **Output filtering** — secrets are redacted from every result and results of
 *    web-fetching tools are wrapped as untrusted external content.
 *
 * `@rullama/inference` wraps the executor it is given with this class by
 * default; construct it directly with {@link enforce} anywhere else.
 *
 * @module
 */

import {
  DEFAULT_PERMISSION_MODE,
  type PermissionMode,
  type Tool,
  type ToolContext,
  ToolResult,
  type ToolUse,
} from "@rullama/core";
import {
  AgentCapabilities,
  createPolicyRequest,
  type GitOperation,
  type PolicyDecision,
  PolicyEngine,
  type PolicyRequest,
} from "@rullama/permission";
import type { ToolExecutor, ToolPreHook } from "./executor.ts";
import { filterToolOutput, wrapWithContentSource } from "./sanitization.ts";

/**
 * Decides an approval request. Return `true` to let the call proceed.
 *
 * Called when a policy answers `require_approval`, and — when a handler is
 * configured and the mode is `"auto"` — for tools flagged `requires_approval`.
 */
export type ApprovalHandler = (
  toolUse: ToolUse,
  decision: PolicyDecision,
  context: ToolContext,
) => Promise<boolean> | boolean;

/** What the executor decided about one tool call. */
export type EnforcementOutcome =
  | "allowed"
  | "denied"
  | "approved"
  | "rejected";

/** One enforcement decision, delivered to {@link EnforcementOptions.onDecision}. */
export interface EnforcementEvent {
  /** The tool call. */
  toolUse: ToolUse;
  /** The policy decision (synthetic for mode / capability / hook denials). */
  decision: PolicyDecision;
  /** How the call was resolved. */
  outcome: EnforcementOutcome;
  /** Human-readable reason for a denial or rejection. */
  reason?: string;
}

/** Options for {@link EnforcingExecutor} / {@link enforce}. */
export interface EnforcementOptions {
  /** Policy engine. Default: {@link PolicyEngine.withDefaults}. */
  policy?: PolicyEngine;
  /** Capability profile to narrow tools/paths/domains/git ops. Default: none. */
  capabilities?: AgentCapabilities;
  /** Permission mode. Default: `"auto"`. */
  mode?: PermissionMode;
  /** Approval handler. Default: deny every approval request. */
  approve?: ApprovalHandler;
  /** Pre-execute hooks, consulted after policy. */
  preHooks?: readonly ToolPreHook[];
  /**
   * Tools whose output is untrusted external content and gets wrapped with
   * delimiters. Default: {@link DEFAULT_EXTERNAL_CONTENT_TOOLS}.
   */
  externalContentTools?: Iterable<string>;
  /** Redact secrets / injection lines from every result. Default: `true`. */
  filterOutput?: boolean;
  /** Receives every decision (for audit logging). */
  onDecision?: (event: EnforcementEvent) => void;
}

/** Built-in tools whose output comes from the open internet. */
export const DEFAULT_EXTERNAL_CONTENT_TOOLS: readonly string[] = [
  "fetch_url",
  "web_search",
  "web_browse",
  "web_scrape",
];

/** Tool categories that a `"read-only"` agent may still use. */
const READ_ONLY_CATEGORIES = new Set(["FileRead", "Search", "Planning"]);

/** Built-in git tool name → the operation it performs. */
const GIT_OPERATIONS: Readonly<Record<string, GitOperation>> = {
  git_status: "Status",
  git_diff: "Diff",
  git_log: "Log",
  git_add: "Add",
  git_commit: "Commit",
  git_push: "Push",
  git_pull: "Pull",
  git_fetch: "Fetch",
  git_branch: "Branch",
  git_checkout: "Checkout",
  git_merge: "Merge",
  git_rebase: "Rebase",
  git_reset: "Reset",
  git_stash: "Stash",
  git_tag: "Tag",
};

function inputOf(toolUse: ToolUse): Record<string, unknown> {
  const input = toolUse.input;
  return typeof input === "object" && input !== null
    ? input as Record<string, unknown>
    : {};
}

/** The git operation a built-in git tool call performs, if it is one. */
export function gitOperationForToolUse(
  toolUse: ToolUse,
): GitOperation | undefined {
  const op = GIT_OPERATIONS[toolUse.name];
  if (op === "Push" && inputOf(toolUse).force === true) return "ForcePush";
  return op;
}

/** The file path a tool call targets (`path` or `file_path` input), if any. */
export function filePathForToolUse(toolUse: ToolUse): string | undefined {
  const input = inputOf(toolUse);
  const path = input.path ?? input.file_path;
  return typeof path === "string" ? path : undefined;
}

/** The host a tool call reaches (`url` input), if it is a valid URL. */
export function domainForToolUse(toolUse: ToolUse): string | undefined {
  const url = inputOf(toolUse).url;
  if (typeof url !== "string") return undefined;
  try {
    return new URL(url).hostname;
  } catch {
    return undefined;
  }
}

/**
 * Build the {@link PolicyRequest} for a tool call: name, category, and the file
 * path / domain / git operation derived from its input.
 */
export function policyRequestForToolUse(
  toolUse: ToolUse,
  context?: ToolContext,
): PolicyRequest {
  return createPolicyRequest({
    tool_name: toolUse.name,
    tool_category: AgentCapabilities.categorizeTool(toolUse.name),
    file_path: filePathForToolUse(toolUse),
    domain: domainForToolUse(toolUse),
    git_operation: gitOperationForToolUse(toolUse),
    agent_id: context?.metadata?.agent_id,
  });
}

function syntheticDecision(reason: string): PolicyDecision {
  return {
    action: { type: "deny_with_message", message: reason },
    matched_policy: undefined,
    reason,
    audit: true,
  };
}

/**
 * A {@link ToolExecutor} that enforces permission mode, capabilities, policy,
 * pre-execute hooks and output filtering around an inner executor.
 */
export class EnforcingExecutor implements ToolExecutor {
  /** The wrapped executor. */
  readonly inner: ToolExecutor;
  /** The policy engine consulted for every call. */
  readonly policy: PolicyEngine;
  /** The capability profile, if one narrows this executor. */
  readonly capabilities: AgentCapabilities | undefined;
  /** The permission mode. */
  readonly mode: PermissionMode;
  readonly #approve: ApprovalHandler | undefined;
  readonly #preHooks: readonly ToolPreHook[];
  readonly #external: Set<string>;
  readonly #filterOutput: boolean;
  readonly #onDecision: ((event: EnforcementEvent) => void) | undefined;

  constructor(inner: ToolExecutor, options: EnforcementOptions = {}) {
    this.inner = inner;
    this.policy = options.policy ?? PolicyEngine.withDefaults();
    this.capabilities = options.capabilities;
    this.mode = options.mode ?? DEFAULT_PERMISSION_MODE;
    this.#approve = options.approve;
    this.#preHooks = options.preHooks ?? [];
    this.#external = new Set(
      options.externalContentTools ?? DEFAULT_EXTERNAL_CONTENT_TOOLS,
    );
    this.#filterOutput = options.filterOutput ?? true;
    this.#onDecision = options.onDecision;
  }

  /** Tools of the inner executor (enforcement never hides a tool's existence). */
  availableTools(): Tool[] {
    return this.inner.availableTools();
  }

  /** Gate the call; on success run it and filter the result. */
  async execute(toolUse: ToolUse, context: ToolContext): Promise<ToolResult> {
    const denial = await this.gate(toolUse, context);
    if (denial !== undefined) return ToolResult.error(toolUse.id, denial);
    const result = await this.inner.execute(toolUse, context);
    return this.postProcess(toolUse, result);
  }

  /** Run every check; returns the denial reason, or `undefined` to proceed. */
  private async gate(
    toolUse: ToolUse,
    context: ToolContext,
  ): Promise<string | undefined> {
    const early = this.checkMode(toolUse) ?? this.checkCapabilities(toolUse);
    if (early !== undefined) {
      this.emit(toolUse, syntheticDecision(early), "denied", early);
      return early;
    }
    const policyDenial = await this.checkPolicy(toolUse, context);
    if (policyDenial !== undefined) return policyDenial;
    return await this.runPreHooks(toolUse, context);
  }

  private checkMode(toolUse: ToolUse): string | undefined {
    if (this.mode !== "read-only") return undefined;
    const category = AgentCapabilities.categorizeTool(toolUse.name);
    if (READ_ONLY_CATEGORIES.has(category)) return undefined;
    return `tool '${toolUse.name}' is not permitted in read-only mode`;
  }

  private checkCapabilities(toolUse: ToolUse): string | undefined {
    const caps = this.capabilities;
    if (caps === undefined || this.mode === "full") return undefined;
    if (!caps.allowsTool(toolUse.name)) {
      return `tool '${toolUse.name}' is outside the agent's capabilities`;
    }
    const request = policyRequestForToolUse(toolUse);
    return this.checkFileCapability(caps, request) ??
      this.checkNetworkAndGitCapability(caps, request);
  }

  private checkFileCapability(
    caps: AgentCapabilities,
    request: PolicyRequest,
  ): string | undefined {
    const path = request.file_path;
    if (path === undefined) return undefined;
    const write = request.tool_category === "FileWrite";
    const allowed = write ? caps.allowsWrite(path) : caps.allowsRead(path);
    if (allowed) return undefined;
    return `${write ? "write" : "read"} access to '${path}' is not permitted`;
  }

  private checkNetworkAndGitCapability(
    caps: AgentCapabilities,
    request: PolicyRequest,
  ): string | undefined {
    const { domain, git_operation: op } = request;
    if (domain !== undefined && !caps.allowsDomain(domain)) {
      return `network access to '${domain}' is not permitted`;
    }
    if (op !== undefined && !caps.allowsGitOp(op)) {
      return `git operation '${op}' is not permitted`;
    }
    return undefined;
  }

  private async checkPolicy(
    toolUse: ToolUse,
    context: ToolContext,
  ): Promise<string | undefined> {
    const decision = this.policy.evaluate(
      policyRequestForToolUse(toolUse, context),
    );
    switch (decision.action.type) {
      case "deny": {
        const reason = decision.reason ??
          `denied by policy '${decision.matched_policy ?? "default"}'`;
        this.emit(toolUse, decision, "denied", reason);
        return reason;
      }
      case "deny_with_message":
        this.emit(toolUse, decision, "denied", decision.action.message);
        return decision.action.message;
      case "require_approval":
        return await this.askApproval(toolUse, decision, context);
      default:
        return await this.checkFlaggedTool(toolUse, decision, context);
    }
  }

  /** In `"auto"` mode a configured approver also gates `requires_approval` tools. */
  private async checkFlaggedTool(
    toolUse: ToolUse,
    decision: PolicyDecision,
    context: ToolContext,
  ): Promise<string | undefined> {
    const flagged = this.availableTools().some((t) =>
      t.name === toolUse.name && t.requires_approval === true
    );
    if (!flagged || this.#approve === undefined || this.mode !== "auto") {
      this.emit(toolUse, decision, "allowed");
      return undefined;
    }
    const synthetic: PolicyDecision = {
      action: { type: "require_approval" },
      matched_policy: undefined,
      reason: `tool '${toolUse.name}' is flagged requires_approval`,
      audit: true,
    };
    return await this.askApproval(toolUse, synthetic, context);
  }

  private async askApproval(
    toolUse: ToolUse,
    decision: PolicyDecision,
    context: ToolContext,
  ): Promise<string | undefined> {
    const granted = this.mode === "full" ||
      (this.#approve !== undefined &&
        await this.#approve(toolUse, decision, context));
    if (granted) {
      this.emit(toolUse, decision, "approved");
      return undefined;
    }
    const reason =
      `tool '${toolUse.name}' requires approval and none was granted` +
      (decision.reason ? ` (${decision.reason})` : "");
    this.emit(toolUse, decision, "rejected", reason);
    return reason;
  }

  private async runPreHooks(
    toolUse: ToolUse,
    context: ToolContext,
  ): Promise<string | undefined> {
    for (const hook of this.#preHooks) {
      const verdict = await hook.beforeExecute(toolUse, context);
      if (verdict.type === "Reject") {
        const reason = `rejected by pre-execute hook: ${verdict.reason}`;
        this.emit(toolUse, syntheticDecision(reason), "rejected", reason);
        return reason;
      }
    }
    return undefined;
  }

  private postProcess(toolUse: ToolUse, result: ToolResult): ToolResult {
    if (!this.#filterOutput) return result;
    let content = filterToolOutput(result.content);
    if (!result.is_error && this.#external.has(toolUse.name)) {
      content = wrapWithContentSource(content, "ExternalContent");
    }
    return new ToolResult(result.tool_use_id, content, result.is_error);
  }

  private emit(
    toolUse: ToolUse,
    decision: PolicyDecision,
    outcome: EnforcementOutcome,
    reason?: string,
  ): void {
    this.#onDecision?.({ toolUse, decision, outcome, reason });
  }
}

/**
 * Wrap `inner` in an {@link EnforcingExecutor}. An executor that already
 * enforces is returned as-is, so wrapping is idempotent.
 */
export function enforce(
  inner: ToolExecutor,
  options?: EnforcementOptions,
): EnforcingExecutor {
  return inner instanceof EnforcingExecutor
    ? inner
    : new EnforcingExecutor(inner, options);
}
