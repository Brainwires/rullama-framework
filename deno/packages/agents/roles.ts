/**
 * Agent role definitions for constrained, least-privilege execution.
 *
 * Each {@link AgentRole} maps to a specific tool allow-list and a short
 * system-prompt suffix that reinforces the role boundary.
 *
 * Equivalent to Rust's `brainwires_agents::roles` module.
 */

import type { Tool } from "@brainwires/core";

/** Role assigned to a TaskAgent that restricts its available tools. */
export type AgentRole = "exploration" | "planning" | "verification" | "execution";

const EXPLORATION_TOOLS: readonly string[] = [
  "read_file",
  "list_directory",
  "search_code",
  "query_codebase",
  "fetch_url",
  "web_search",
  "glob",
  "grep",
  "context_recall",
  "task_get",
  "task_list",
];

const PLANNING_TOOLS: readonly string[] = [
  "read_file",
  "list_directory",
  "glob",
  "grep",
  "task_create",
  "task_update",
  "task_add_subtask",
  "task_list",
  "task_get",
  "plan_task",
  "context_recall",
];

const VERIFICATION_TOOLS: readonly string[] = [
  "read_file",
  "list_directory",
  "glob",
  "grep",
  "execute_command",
  "check_duplicates",
  "verify_build",
  "check_syntax",
  "task_get",
  "task_list",
  "context_recall",
];

/** Tool names allowed for the role, or null if all tools are permitted. */
export function allowedTools(role: AgentRole): readonly string[] | null {
  switch (role) {
    case "exploration":
      return EXPLORATION_TOOLS;
    case "planning":
      return PLANNING_TOOLS;
    case "verification":
      return VERIFICATION_TOOLS;
    case "execution":
      return null;
  }
}

/** Filter a tool list to only those permitted by the role. */
export function filterTools(role: AgentRole, tools: Tool[]): Tool[] {
  const allow = allowedTools(role);
  if (allow === null) return [...tools];
  const allowSet = new Set(allow);
  return tools.filter((t) => allowSet.has(t.name));
}

/** Short system-prompt suffix that reminds the model of its constraints. */
export function systemPromptSuffix(role: AgentRole): string {
  switch (role) {
    case "exploration":
      return "\n\n[ROLE: Exploration] You may only read files and search. " +
        "Do not attempt to write files, run commands, or spawn agents.";
    case "planning":
      return "\n\n[ROLE: Planning] You may read files and manage tasks. " +
        "Do not write files or execute code — produce a plan only.";
    case "verification":
      return "\n\n[ROLE: Verification] You may read files and run build/test commands. " +
        "Do not write or delete files.";
    case "execution":
      return "";
  }
}

/** Display name for logs and prompts (matches Rust Debug formatting). */
export function roleDisplayName(role: AgentRole): string {
  switch (role) {
    case "exploration":
      return "Exploration";
    case "planning":
      return "Planning";
    case "verification":
      return "Verification";
    case "execution":
      return "Execution";
  }
}
