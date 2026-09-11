/**
 * A concrete {@link ToolExecutor} over the built-in tool classes.
 *
 * Each built-in tool class (`BashTool`, `FileOpsTool`, …) exposes a static
 * `getTools()` and a static `execute(toolUseId, toolName, input, context)`;
 * {@link BuiltinToolExecutor} indexes those by tool name and dispatches calls to
 * the class that owns the tool. {@link createBuiltinExecutor} is the one-liner
 * most applications want: the default tool set behind the enforcing executor
 * from `@rullama/tool-runtime` (policy, capabilities, output filtering).
 *
 * @module
 */

import type { Tool, ToolContext, ToolResult, ToolUse } from "@rullama/core";
import { ToolResult as ToolResultClass } from "@rullama/core";
import {
  enforce,
  type EnforcementOptions,
  type EnforcingExecutor,
  type ToolExecutor,
} from "@rullama/tool-runtime";
import { BashTool } from "./bash.ts";
import { FileOpsTool } from "./file_ops.ts";
import { GitTool } from "./git.ts";
import { SearchTool } from "./search.ts";
import { WebTool } from "./web.ts";

/**
 * Anything that owns a set of tools and can run them by name. The built-in
 * tool classes satisfy this statically (`BashTool`, `FileOpsTool`, …), and an
 * instance-based tool such as `SessionsTool` can be adapted in a few lines.
 */
export interface ToolProvider {
  /** The tool definitions this provider owns. */
  getTools(): Tool[];
  /** Run one of this provider's tools. Should never throw; return an error result. */
  execute(
    toolUseId: string,
    toolName: string,
    // deno-lint-ignore no-explicit-any
    input: any,
    context: ToolContext,
  ): Promise<ToolResult>;
}

/** The providers {@link BuiltinToolExecutor.withDefaults} registers. */
export const DEFAULT_TOOL_PROVIDERS: readonly ToolProvider[] = [
  BashTool,
  FileOpsTool,
  GitTool,
  SearchTool,
  WebTool,
];

/** Dispatches tool calls to the {@link ToolProvider} that owns each tool. */
export class BuiltinToolExecutor implements ToolExecutor {
  readonly #owners = new Map<string, ToolProvider>();
  readonly #tools: Tool[] = [];

  /**
   * @param providers Tool providers; a tool name owned by two providers is an error.
   */
  constructor(providers: Iterable<ToolProvider>) {
    for (const provider of providers) {
      for (const tool of provider.getTools()) {
        if (this.#owners.has(tool.name)) {
          throw new Error(`Tool '${tool.name}' is registered twice`);
        }
        this.#owners.set(tool.name, provider);
        this.#tools.push(tool);
      }
    }
  }

  /** Bash, file operations, git, code search and web fetch. */
  static withDefaults(): BuiltinToolExecutor {
    return new BuiltinToolExecutor(DEFAULT_TOOL_PROVIDERS);
  }

  /** Every registered tool definition. */
  availableTools(): Tool[] {
    return [...this.#tools];
  }

  /** Run a tool; an unknown tool name yields an error result, never a throw. */
  execute(toolUse: ToolUse, context: ToolContext): Promise<ToolResult> {
    const owner = this.#owners.get(toolUse.name);
    if (owner === undefined) {
      return Promise.resolve(
        ToolResultClass.error(toolUse.id, `Unknown tool: ${toolUse.name}`),
      );
    }
    return owner.execute(toolUse.id, toolUse.name, toolUse.input, context);
  }
}

/** Options for {@link createBuiltinExecutor}. */
export interface BuiltinExecutorOptions extends EnforcementOptions {
  /** Tool providers. Default: {@link DEFAULT_TOOL_PROVIDERS}. */
  providers?: Iterable<ToolProvider>;
}

/**
 * The built-in tools behind the enforcing executor: the recommended default
 * for `AgentContext` and any other consumer of a `ToolExecutor`.
 */
export function createBuiltinExecutor(
  options: BuiltinExecutorOptions = {},
): EnforcingExecutor {
  const { providers, ...enforcement } = options;
  return enforce(
    new BuiltinToolExecutor(providers ?? DEFAULT_TOOL_PROVIDERS),
    enforcement,
  );
}
