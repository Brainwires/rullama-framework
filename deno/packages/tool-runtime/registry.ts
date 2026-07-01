/**
 * Tool Registry - Composable container for tool definitions
 *
 * Provides a `ToolRegistry` that stores tool definitions and supports
 * deferred loading, category filtering, and search.
 */

import type { Tool } from "@rullama/core";

/** Tool categories for filtering tools by purpose. */
export type ToolCategory =
  | "FileOps"
  | "Search"
  | "SemanticSearch"
  | "Git"
  | "TaskManager"
  | "AgentPool"
  | "Web"
  | "WebSearch"
  | "Bash"
  | "Planning"
  | "Context"
  | "Orchestrator"
  | "CodeExecution"
  | "SessionTask"
  | "Validation";

/** Mapping from ToolCategory to the tool names that belong to it. */
const CATEGORY_TOOL_NAMES: Record<ToolCategory, string[]> = {
  FileOps: [
    "read_file",
    "write_file",
    "edit_file",
    "patch_file",
    "list_directory",
    "search_files",
    "delete_file",
    "create_directory",
  ],
  Search: ["search_code", "search_files"],
  SemanticSearch: [
    "index_codebase",
    "query_codebase",
    "search_with_filters",
    "get_rag_statistics",
    "clear_rag_index",
    "search_git_history",
  ],
  Git: [
    "git_status",
    "git_diff",
    "git_log",
    "git_stage",
    "git_unstage",
    "git_commit",
    "git_push",
    "git_pull",
    "git_fetch",
    "git_discard",
    "git_branch",
  ],
  TaskManager: [
    "task_create",
    "task_start",
    "task_complete",
    "task_list",
    "task_skip",
    "task_add",
    "task_block",
    "task_depends",
    "task_ready",
    "task_time",
  ],
  AgentPool: [
    "agent_spawn",
    "agent_status",
    "agent_list",
    "agent_stop",
    "agent_await",
  ],
  Web: ["fetch_url"],
  WebSearch: ["web_search", "web_browse", "web_scrape"],
  Bash: ["execute_command"],
  Planning: ["plan_task"],
  Context: ["recall_context"],
  Orchestrator: ["execute_script"],
  CodeExecution: ["execute_code"],
  SessionTask: ["task_list_write"],
  Validation: ["check_duplicates", "verify_build", "check_syntax"],
};

/** Core tool names for basic project exploration. */
const CORE_TOOL_NAMES = [
  "read_file",
  "write_file",
  "edit_file",
  "list_directory",
  "search_code",
  "execute_command",
  "git_status",
  "git_diff",
  "git_log",
  "git_stage",
  "git_commit",
  "search_tools",
  "index_codebase",
  "query_codebase",
];

/** Primary meta-tool names. */
const PRIMARY_TOOL_NAMES = ["execute_script", "search_tools"];

/**
 * Composable tool registry - stores and queries tool definitions.
 *
 * Unlike the CLI's registry which auto-registers all tools, this registry
 * is empty by default. Callers compose it by registering tools from
 * whichever modules they need.
 */
export class ToolRegistry {
  private tools: Tool[] = [];

  /** Create an empty registry. */
  constructor() {}

  /**
   * Create a registry pre-populated with all built-in tools.
   * Import and register tool classes as needed.
   */
  static withBuiltins(): ToolRegistry {
    const registry = new ToolRegistry();
    // Lazily import to avoid circular dependency issues.
    // Callers should register tools explicitly:
    //   registry.registerTools(BashTool.getTools());
    //   registry.registerTools(FileOpsTool.getTools());
    //   etc.
    return registry;
  }

  /** Register a single tool. */
  register(tool: Tool): void {
    this.tools.push(tool);
  }

  /** Register multiple tools at once. */
  registerTools(tools: Tool[]): void {
    this.tools.push(...tools);
  }

  /** Get all registered tools. */
  getAll(): readonly Tool[] {
    return this.tools;
  }

  /** Get all tools including additional external tools (e.g., MCP tools). */
  getAllWithExtra(extra: Tool[]): Tool[] {
    return [...this.tools, ...extra];
  }

  /** Look up a tool by name. */
  get(name: string): Tool | undefined {
    return this.tools.find((t) => t.name === name);
  }

  /** Get tools that should be loaded initially (defer_loading = false or undefined). */
  getInitialTools(): Tool[] {
    return this.tools.filter((t) => !t.defer_loading);
  }

  /** Get only deferred tools (defer_loading = true). */
  getDeferredTools(): Tool[] {
    return this.tools.filter((t) => t.defer_loading === true);
  }

  /** Search tools by query string matching name and description. */
  searchTools(query: string): Tool[] {
    const queryLower = query.toLowerCase();
    const queryTerms = queryLower.split(/\s+/);

    return this.tools.filter((tool) => {
      const nameLower = tool.name.toLowerCase();
      const descLower = tool.description.toLowerCase();
      return queryTerms.some(
        (term) => nameLower.includes(term) || descLower.includes(term),
      );
    });
  }

  /** Get tools by category. */
  getByCategory(category: ToolCategory): Tool[] {
    const names = CATEGORY_TOOL_NAMES[category] ?? [];
    return this.tools.filter((t) => names.includes(t.name));
  }

  /** Get all tools including MCP tools. */
  getAllWithMcp(mcpTools: Tool[]): Tool[] {
    return this.getAllWithExtra(mcpTools);
  }

  /** Get core tools for basic project exploration. */
  getCore(): Tool[] {
    return this.tools.filter((t) => CORE_TOOL_NAMES.includes(t.name));
  }

  /** Get primary meta-tools (always available). */
  getPrimary(): Tool[] {
    return this.tools.filter((t) => PRIMARY_TOOL_NAMES.includes(t.name));
  }

  /** Total number of registered tools. */
  get length(): number {
    return this.tools.length;
  }

  /** Whether the registry is empty. */
  isEmpty(): boolean {
    return this.tools.length === 0;
  }
}
