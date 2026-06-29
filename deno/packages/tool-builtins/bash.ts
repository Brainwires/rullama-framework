/**
 * Bash execution tool implementation.
 * Uses Deno.Command for subprocess execution.
 */

// deno-lint-ignore-file no-explicit-any

import { objectSchema, type ToolContext, ToolResult } from "@rullama/core";
import type { Tool } from "@rullama/core";

/** Output limiting mode for proactive context management. */
export type OutputMode =
  | "full"
  | "head"
  | "tail"
  | "filter"
  | "count"
  | "smart";

/** Stderr handling mode. */
export type StderrMode = "separate" | "combined" | "stderr_only" | "suppress";

/** Output limiting configuration. */
interface OutputLimits {
  maxLines?: number;
  outputMode: OutputMode;
  filterPattern?: string;
  stderrMode: StderrMode;
  autoLimit: boolean;
}

/** Interactive commands that should be rejected. */
const INTERACTIVE_COMMANDS = [
  "vim",
  "vi",
  "nvim",
  "nano",
  "emacs",
  "pico",
  "less",
  "more",
  "most",
  "top",
  "htop",
  "btop",
  "glances",
  "man",
  "info",
  "ssh",
  "telnet",
  "ftp",
  "sftp",
  "python",
  "python3",
  "node",
  "irb",
  "ghci",
  "lua",
  "mysql",
  "psql",
  "sqlite3",
  "mongo",
  "redis-cli",
];

/** Dangerous command patterns. */
const DANGEROUS_PATTERNS = [
  "rm -rf /",
  "mkfs",
  "> /dev/sda",
  "dd if=/dev/zero",
  ":(){ :|:& };:",
];

interface CommandOutput {
  stdout: string;
  stderr: string;
  exitCode: number;
}

interface ParsedParams {
  command: string;
  timeout: number;
  maxLines?: number;
  outputMode: OutputMode;
  filterPattern?: string;
  stderrMode: StderrMode;
  autoLimit: boolean;
}

/** Bash execution tool. */
export class BashTool {
  /** Get all bash tool definitions. */
  static getTools(): Tool[] {
    return [BashTool.executeCommandTool()];
  }

  private static executeCommandTool(): Tool {
    return {
      name: "execute_command",
      description:
        "Execute a bash command and return the output. Supports proactive output limiting to manage context size.",
      input_schema: objectSchema(
        {
          command: {
            type: "string",
            description: "The bash command to execute",
          },
          timeout: {
            type: "number",
            description: "Timeout in seconds (default: 30)",
            default: 30,
          },
          max_lines: {
            type: "number",
            description:
              "Maximum output lines. Applies head -n or tail -n based on output_mode.",
          },
          output_mode: {
            type: "string",
            enum: ["full", "head", "tail", "filter", "count", "smart"],
            description:
              "Output limiting mode: full, head, tail, filter, count, smart",
            default: "smart",
          },
          filter_pattern: {
            type: "string",
            description:
              "Grep pattern to filter output (used when output_mode is 'filter')",
          },
          stderr_mode: {
            type: "string",
            enum: ["separate", "combined", "stderr_only", "suppress"],
            description: "Stderr handling mode",
            default: "combined",
          },
          auto_limit: {
            type: "boolean",
            description:
              "Automatically apply smart output limits based on command type (default: true)",
            default: true,
          },
        },
        ["command"],
      ),
      requires_approval: true,
    };
  }

  /** Execute a bash command tool. */
  static async execute(
    toolUseId: string,
    toolName: string,
    input: any,
    context: ToolContext,
  ): Promise<ToolResult> {
    if (toolName !== "execute_command") {
      return ToolResult.error(toolUseId, `Unknown bash tool: ${toolName}`);
    }

    try {
      const output = await BashTool.executeCommand(input, context);
      return ToolResult.success(toolUseId, output);
    } catch (e) {
      return ToolResult.error(
        toolUseId,
        `Command execution failed: ${(e as Error).message}`,
      );
    }
  }

  private static async executeCommand(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const params = BashTool.parseParams(input);

    if (BashTool.isInteractiveCommand(params.command)) {
      const firstWord = params.command.split(/\s+/)[0];
      throw new Error(
        `Interactive command detected: '${firstWord}'. Use non-interactive alternatives instead.`,
      );
    }

    BashTool.validateCommand(params.command);

    const limits = BashTool.resolveOutputLimits(params);
    const transformedCommand = BashTool.transformCommand(
      params.command,
      limits,
    );

    const output = await BashTool.runCommandWithTimeout(
      transformedCommand,
      context.working_directory,
      params.timeout * 1000,
    );

    return BashTool.formatOutput(
      params.command,
      transformedCommand,
      output,
      limits,
    );
  }

  private static parseParams(input: any): ParsedParams {
    return {
      command: input.command,
      timeout: input.timeout ?? 30,
      maxLines: input.max_lines,
      outputMode: input.output_mode ?? "smart",
      filterPattern: input.filter_pattern,
      stderrMode: input.stderr_mode ?? "combined",
      autoLimit: input.auto_limit ?? true,
    };
  }

  /** Check if a command is interactive. */
  static isInteractiveCommand(command: string): boolean {
    const firstWord = command.split(/\s+/)[0];
    const effective = (firstWord === "sudo" || firstWord === "env")
      ? (command.split(/\s+/)[1] ?? "")
      : firstWord;
    return INTERACTIVE_COMMANDS.includes(effective);
  }

  /** Validate command for dangerous patterns. */
  static validateCommand(command: string): void {
    for (const pattern of DANGEROUS_PATTERNS) {
      if (command.includes(pattern)) {
        throw new Error(
          `Command contains potentially dangerous pattern: ${pattern}`,
        );
      }
    }
  }

  /** Get smart output limits based on command type. */
  static getSmartLimits(command: string): OutputLimits {
    const cmdLower = command.toLowerCase();
    const firstWord = command.split(/\s+/)[0];

    const defaults: OutputLimits = {
      outputMode: "full",
      stderrMode: "separate",
      autoLimit: false,
    };

    if (firstWord === "cargo") {
      if (cmdLower.includes("build")) {
        return {
          maxLines: 80,
          outputMode: "head",
          stderrMode: "combined",
          autoLimit: false,
        };
      }
      if (cmdLower.includes("test")) {
        return {
          maxLines: 100,
          outputMode: "head",
          stderrMode: "combined",
          autoLimit: false,
        };
      }
      if (cmdLower.includes("check")) {
        return {
          maxLines: 60,
          outputMode: "head",
          stderrMode: "combined",
          autoLimit: false,
        };
      }
      if (cmdLower.includes("clippy")) {
        return {
          maxLines: 80,
          outputMode: "head",
          stderrMode: "combined",
          autoLimit: false,
        };
      }
    }

    const buildTools: Record<string, OutputLimits> = {
      npm: {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "combined",
        autoLimit: false,
      },
      yarn: {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "combined",
        autoLimit: false,
      },
      pnpm: {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "combined",
        autoLimit: false,
      },
      bun: {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "combined",
        autoLimit: false,
      },
      make: {
        maxLines: 100,
        outputMode: "head",
        stderrMode: "combined",
        autoLimit: false,
      },
      cmake: {
        maxLines: 100,
        outputMode: "head",
        stderrMode: "combined",
        autoLimit: false,
      },
      ninja: {
        maxLines: 100,
        outputMode: "head",
        stderrMode: "combined",
        autoLimit: false,
      },
    };

    if (buildTools[firstWord]) return buildTools[firstWord];

    if (
      firstWord === "go" &&
      (cmdLower.includes("build") || cmdLower.includes("test"))
    ) {
      return {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "combined",
        autoLimit: false,
      };
    }

    const simpleTools: Record<string, OutputLimits> = {
      find: {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "separate",
        autoLimit: false,
      },
      fd: {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "separate",
        autoLimit: false,
      },
      locate: {
        maxLines: 30,
        outputMode: "head",
        stderrMode: "separate",
        autoLimit: false,
      },
      ps: {
        maxLines: 30,
        outputMode: "head",
        stderrMode: "separate",
        autoLimit: false,
      },
      ls: {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "separate",
        autoLimit: false,
      },
      tree: {
        maxLines: 80,
        outputMode: "head",
        stderrMode: "separate",
        autoLimit: false,
      },
      grep: {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "separate",
        autoLimit: false,
      },
      rg: {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "separate",
        autoLimit: false,
      },
      ag: {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "separate",
        autoLimit: false,
      },
      ack: {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "separate",
        autoLimit: false,
      },
    };

    if (simpleTools[firstWord]) return simpleTools[firstWord];

    if (firstWord === "git") {
      if (cmdLower.includes("log")) {
        return {
          maxLines: 30,
          outputMode: "head",
          stderrMode: "separate",
          autoLimit: false,
        };
      }
      if (cmdLower.includes("diff")) {
        return {
          maxLines: 100,
          outputMode: "head",
          stderrMode: "separate",
          autoLimit: false,
        };
      }
      if (cmdLower.includes("status")) {
        return {
          maxLines: 50,
          outputMode: "head",
          stderrMode: "separate",
          autoLimit: false,
        };
      }
    }

    if (firstWord === "docker") {
      if (cmdLower.includes("logs")) {
        return {
          maxLines: 50,
          outputMode: "tail",
          stderrMode: "separate",
          autoLimit: false,
        };
      }
      if (cmdLower.includes("ps")) {
        return {
          maxLines: 30,
          outputMode: "head",
          stderrMode: "separate",
          autoLimit: false,
        };
      }
    }

    if (firstWord === "kubectl") {
      if (cmdLower.includes("logs")) {
        return {
          maxLines: 50,
          outputMode: "tail",
          stderrMode: "separate",
          autoLimit: false,
        };
      }
      return {
        maxLines: 50,
        outputMode: "head",
        stderrMode: "separate",
        autoLimit: false,
      };
    }

    if (firstWord === "journalctl") {
      return {
        maxLines: 100,
        outputMode: "tail",
        stderrMode: "separate",
        autoLimit: false,
      };
    }

    return defaults;
  }

  private static resolveOutputLimits(params: ParsedParams): OutputLimits {
    const limits: OutputLimits = {
      maxLines: params.maxLines,
      outputMode: params.outputMode,
      filterPattern: params.filterPattern,
      stderrMode: params.stderrMode,
      autoLimit: params.autoLimit,
    };

    if (limits.autoLimit && limits.outputMode === "smart") {
      const smartLimits = BashTool.getSmartLimits(params.command);
      if (limits.maxLines === undefined) {
        limits.maxLines = smartLimits.maxLines;
      }
      if (limits.outputMode === "smart") {
        limits.outputMode = smartLimits.outputMode;
      }
      if (limits.stderrMode === "separate") {
        limits.stderrMode = smartLimits.stderrMode;
      }
    }

    return limits;
  }

  /** Transform command with output limiting (pipes). */
  static transformCommand(command: string, limits: OutputLimits): string {
    let cmd = command;

    // No transformation needed for default/full mode with no limits
    if (
      limits.maxLines === undefined &&
      !limits.filterPattern &&
      limits.stderrMode === "separate" &&
      limits.outputMode === "full"
    ) {
      return command;
    }

    switch (limits.stderrMode) {
      case "combined":
        cmd = `${cmd} 2>&1`;
        break;
      case "stderr_only":
        cmd = `${cmd} 2>&1 >/dev/null`;
        break;
      case "suppress":
        cmd = `${cmd} 2>/dev/null`;
        break;
      case "separate":
        break;
    }

    if (limits.filterPattern) {
      const escaped = limits.filterPattern.replace(/'/g, "'\\''");
      cmd = `${cmd} | grep -E '${escaped}'`;
    }

    if (limits.maxLines !== undefined) {
      switch (limits.outputMode) {
        case "tail":
          cmd = `${cmd} | tail -n ${limits.maxLines}`;
          break;
        case "count":
          cmd = `${cmd} | wc -l`;
          break;
        case "head":
        case "smart":
        case "filter":
          cmd = `${cmd} | head -n ${limits.maxLines}`;
          break;
        case "full":
          break;
      }
    }

    if (cmd !== command) {
      cmd = `set -o pipefail; ${cmd}`;
    }

    return cmd;
  }

  private static async runCommandWithTimeout(
    command: string,
    workingDir: string,
    timeoutMs: number,
  ): Promise<CommandOutput> {
    const cmd = new Deno.Command("bash", {
      args: ["-o", "pipefail", "-c", command],
      cwd: workingDir,
      stdout: "piped",
      stderr: "piped",
    });

    const abortController = new AbortController();
    const timer = setTimeout(() => abortController.abort(), timeoutMs);

    try {
      const process = cmd.spawn();
      const output = await process.output();
      clearTimeout(timer);

      return {
        stdout: new TextDecoder().decode(output.stdout),
        stderr: new TextDecoder().decode(output.stderr),
        exitCode: output.code,
      };
    } catch (e) {
      clearTimeout(timer);
      throw new Error(`Failed to execute command: ${(e as Error).message}`);
    }
  }

  private static formatOutput(
    originalCommand: string,
    transformedCommand: string,
    output: CommandOutput,
    limits: OutputLimits,
  ): string {
    let result = `Command: ${originalCommand}\n`;
    if (transformedCommand !== originalCommand) {
      result += `Transformed: ${transformedCommand}\n`;
    }
    result += `Exit Code: ${output.exitCode}\n\n`;

    if (
      limits.stderrMode === "combined" || limits.stderrMode === "stderr_only"
    ) {
      result += `Output:\n${output.stdout}`;
      if (output.stderr.length > 0) {
        result += `\n\nStderr (unmerged):\n${output.stderr}`;
      }
    } else {
      result += `Stdout:\n${output.stdout}\n\nStderr:\n${output.stderr}`;
    }

    return result;
  }
}
