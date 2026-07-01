/**
 * Runtime approval request/response types.
 *
 * Provides types for interactive approval workflows where tool execution
 * requires user consent before proceeding.
 *
 * Rust equivalent: `rullama-permissions/src/approval.rs`
 * @module
 */

// ── Approval Action ─────────────────────────────────────────────────

/**
 * The type of action being performed.
 *
 * Rust equivalent: `ApprovalAction` enum
 */
export type ApprovalAction =
  | { type: "write_file"; path: string }
  | { type: "edit_file"; path: string }
  | { type: "delete_file"; path: string }
  | { type: "create_directory"; path: string }
  | { type: "execute_command"; command: string }
  | { type: "git_modify"; operation: string }
  | { type: "network_access"; domain: string }
  | { type: "other"; description: string };

/**
 * Get a human-readable description of the action.
 *
 * Rust equivalent: `ApprovalAction::description()`
 */
export function approvalActionDescription(action: ApprovalAction): string {
  switch (action.type) {
    case "write_file":
      return `Write file: ${action.path}`;
    case "edit_file":
      return `Edit file: ${action.path}`;
    case "delete_file":
      return `Delete: ${action.path}`;
    case "create_directory":
      return `Create directory: ${action.path}`;
    case "execute_command": {
      const truncated = action.command.length > 50
        ? action.command.slice(0, 50) + "..."
        : action.command;
      return `Execute: ${truncated}`;
    }
    case "git_modify":
      return `Git: ${action.operation}`;
    case "network_access":
      return `Network: ${action.domain}`;
    case "other":
      return action.description;
  }
}

/**
 * Get the category name for display.
 *
 * Rust equivalent: `ApprovalAction::category()`
 */
export function approvalActionCategory(action: ApprovalAction): string {
  switch (action.type) {
    case "write_file":
      return "File Write";
    case "edit_file":
      return "File Edit";
    case "delete_file":
      return "Delete";
    case "create_directory":
      return "Create Directory";
    case "execute_command":
      return "Shell Command";
    case "git_modify":
      return "Git Operation";
    case "network_access":
      return "Network Access";
    case "other":
      return "Other";
  }
}

/**
 * Get the severity level for the action.
 *
 * Rust equivalent: `ApprovalAction::severity()`
 */
export function approvalActionSeverity(
  action: ApprovalAction,
): ApprovalSeverity {
  switch (action.type) {
    case "delete_file":
    case "execute_command":
      return "high";
    case "git_modify":
    case "write_file":
    case "edit_file":
    case "other":
      return "medium";
    case "create_directory":
    case "network_access":
      return "low";
  }
}

// ── Approval Severity ───────────────────────────────────────────────

/**
 * Severity level for approval actions (affects UI presentation).
 *
 * Rust equivalent: `ApprovalSeverity` enum
 */
export type ApprovalSeverity = "low" | "medium" | "high";

// ── Approval Details ────────────────────────────────────────────────

/**
 * Additional details about an approval request.
 *
 * Rust equivalent: `ApprovalDetails` struct
 */
export interface ApprovalDetails {
  /** Description of the tool. */
  tool_description: string;
  /** The parameters being passed to the tool. */
  parameters: unknown;
}

// ── Approval Request ────────────────────────────────────────────────

/**
 * A request for user approval before executing a tool.
 *
 * Note: The Rust version uses a tokio oneshot channel for `response_tx`.
 * In Deno/TS we use a Promise-based approach instead.
 *
 * Rust equivalent: `ApprovalRequest` struct
 */
export interface ApprovalRequest {
  /** Unique identifier for this request. */
  id: string;
  /** Name of the tool requesting approval. */
  tool_name: string;
  /** The action being performed. */
  action: ApprovalAction;
  /** Additional details about the action. */
  details: ApprovalDetails;
  /** Resolve function for the response (replaces tokio oneshot). */
  respond: (response: ApprovalResponse) => void;
}

// ── Approval Response ───────────────────────────────────────────────

/**
 * User's response to an approval request.
 *
 * Rust equivalent: `ApprovalResponse` enum
 */
export type ApprovalResponse =
  | "approve"
  | "deny"
  | "approve_for_session"
  | "deny_for_session";

/**
 * Check if this is an approval (yes or always).
 *
 * Rust equivalent: `ApprovalResponse::is_approved()`
 */
export function isApprovalResponseApproved(
  response: ApprovalResponse,
): boolean {
  return response === "approve" || response === "approve_for_session";
}

/**
 * Check if this should be remembered for the session.
 *
 * Rust equivalent: `ApprovalResponse::is_session_persistent()`
 */
export function isApprovalResponseSessionPersistent(
  response: ApprovalResponse,
): boolean {
  return response === "approve_for_session" || response === "deny_for_session";
}
