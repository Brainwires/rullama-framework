/**
 * @module agent_tools
 *
 * Pre-built MCP tools for agent operations.
 * Equivalent to Rust's `AgentToolRegistry`.
 */

import type { Tool } from "@rullama/core";

/**
 * Registry of pre-built agent management tools.
 * Equivalent to Rust `AgentToolRegistry`.
 */
export class AgentToolRegistry {
  private readonly tools: Tool[];

  constructor() {
    this.tools = [
      {
        name: "agent_spawn",
        description:
          "Spawn a new task agent to work on a subtask autonomously. " +
          "The agent will execute the task in the background and report results. " +
          "Useful for breaking down large workloads hierarchically.",
        input_schema: {
          type: "object",
          properties: {
            description: {
              type: "string",
              description: "Description of the task for the agent to execute",
            },
            working_directory: {
              type: "string",
              description:
                "Optional working directory for file operations. If not specified, uses the MCP server's current directory.",
            },
            max_iterations: {
              type: "integer",
              description:
                "Optional maximum number of iterations before the agent stops (default: 100).",
            },
            enable_validation: {
              type: "boolean",
              description:
                "Enable automatic validation checks before completion (default: true).",
            },
            build_type: {
              type: "string",
              enum: ["npm", "cargo", "typescript"],
              description: "Optional build type for validation.",
            },
          },
          required: ["description"],
        },
      },
      {
        name: "agent_list",
        description: "List all currently running task agents and their status",
        input_schema: { type: "object", properties: {} },
      },
      {
        name: "agent_status",
        description: "Get detailed status of a specific task agent",
        input_schema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "ID of the agent to query",
            },
          },
          required: ["agent_id"],
        },
      },
      {
        name: "agent_stop",
        description: "Stop a running task agent",
        input_schema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "ID of the agent to stop",
            },
          },
          required: ["agent_id"],
        },
      },
      {
        name: "agent_await",
        description:
          "Wait for a task agent to complete and return its result. " +
          "Unlike agent_status which returns immediately, this tool blocks " +
          "until the agent finishes and returns the final result.",
        input_schema: {
          type: "object",
          properties: {
            agent_id: {
              type: "string",
              description: "ID of the agent to wait for",
            },
            timeout_secs: {
              type: "integer",
              description:
                "Optional timeout in seconds. If not provided, waits indefinitely.",
            },
          },
          required: ["agent_id"],
        },
      },
      {
        name: "agent_pool_stats",
        description: "Get statistics about the agent pool",
        input_schema: { type: "object", properties: {} },
      },
      {
        name: "agent_file_locks",
        description: "List all currently held file locks by agents",
        input_schema: { type: "object", properties: {} },
      },
      {
        name: "self_improve_start",
        description:
          "Start an autonomous self-improvement loop that analyzes the codebase " +
          "and spawns agents to fix issues (clippy warnings, TODOs, missing docs, " +
          "dead code, test gaps, code smells)",
        input_schema: {
          type: "object",
          properties: {
            max_cycles: {
              type: "integer",
              description: "Maximum number of improvement cycles (default: 10)",
            },
            max_budget: {
              type: "number",
              description: "Maximum budget in dollars (default: 10.0)",
            },
            dry_run: {
              type: "boolean",
              description: "List tasks without executing (default: false)",
            },
            strategies: {
              type: "string",
              description:
                "Comma-separated list of strategies: clippy,todo_scanner,doc_gaps,test_coverage,refactoring,dead_code (empty = all)",
            },
          },
        },
      },
      {
        name: "self_improve_status",
        description: "Get the status of a running self-improvement session",
        input_schema: { type: "object", properties: {} },
      },
      {
        name: "self_improve_stop",
        description: "Stop a running self-improvement session",
        input_schema: { type: "object", properties: {} },
      },
    ];
  }

  /** Get all agent management tools. */
  getTools(): Tool[] {
    return this.tools;
  }
}
