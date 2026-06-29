/**
 * Git operations tool implementation.
 * Uses Deno.Command for git subprocess execution.
 */

// deno-lint-ignore-file no-explicit-any

import { objectSchema, type ToolContext, ToolResult } from "@rullama/core";
import type { Tool } from "@rullama/core";

/** Git operations tool. */
export class GitTool {
  /** Get all git tool definitions. */
  static getTools(): Tool[] {
    return [
      GitTool.gitStatusTool(),
      GitTool.gitDiffTool(),
      GitTool.gitLogTool(),
      GitTool.gitStageTool(),
      GitTool.gitUnstageTool(),
      GitTool.gitCommitTool(),
      GitTool.gitPushTool(),
      GitTool.gitPullTool(),
      GitTool.gitFetchTool(),
      GitTool.gitDiscardTool(),
      GitTool.gitBranchTool(),
    ];
  }

  private static gitStatusTool(): Tool {
    return {
      name: "git_status",
      description: "Get git repository status",
      input_schema: objectSchema({}, []),
      requires_approval: false,
    };
  }

  private static gitDiffTool(): Tool {
    return {
      name: "git_diff",
      description: "Get git diff of changes",
      input_schema: objectSchema({}, []),
      requires_approval: false,
    };
  }

  private static gitLogTool(): Tool {
    return {
      name: "git_log",
      description: "Get git commit history",
      input_schema: objectSchema(
        {
          limit: {
            type: "number",
            description: "Number of commits",
            default: 10,
          },
        },
        [],
      ),
      requires_approval: false,
    };
  }

  private static gitStageTool(): Tool {
    return {
      name: "git_stage",
      description: "Stage files for commit.",
      input_schema: objectSchema(
        {
          files: {
            type: "array",
            items: { type: "string" },
            description: "Files to stage. Use '.' for all.",
          },
        },
        ["files"],
      ),
      requires_approval: true,
    };
  }

  private static gitUnstageTool(): Tool {
    return {
      name: "git_unstage",
      description: "Unstage files from the staging area.",
      input_schema: objectSchema(
        {
          files: {
            type: "array",
            items: { type: "string" },
            description: "Files to unstage.",
          },
        },
        ["files"],
      ),
      requires_approval: true,
    };
  }

  private static gitCommitTool(): Tool {
    return {
      name: "git_commit",
      description: "Create a git commit with staged changes.",
      input_schema: objectSchema(
        {
          message: {
            type: "string",
            description: "Commit message",
          },
          all: {
            type: "boolean",
            description: "Stage all modified files before committing",
            default: false,
          },
        },
        ["message"],
      ),
      requires_approval: true,
    };
  }

  private static gitPushTool(): Tool {
    return {
      name: "git_push",
      description: "Push commits to a remote repository.",
      input_schema: objectSchema(
        {
          remote: {
            type: "string",
            description: "Remote name (default: origin)",
            default: "origin",
          },
          branch: {
            type: "string",
            description: "Branch to push",
          },
          set_upstream: {
            type: "boolean",
            description: "Set upstream tracking (-u)",
            default: false,
          },
        },
        [],
      ),
      requires_approval: true,
    };
  }

  private static gitPullTool(): Tool {
    return {
      name: "git_pull",
      description: "Pull changes from a remote repository.",
      input_schema: objectSchema(
        {
          remote: {
            type: "string",
            description: "Remote name (default: origin)",
            default: "origin",
          },
          branch: {
            type: "string",
            description: "Branch to pull",
          },
          rebase: {
            type: "boolean",
            description: "Use rebase instead of merge",
            default: false,
          },
        },
        [],
      ),
      requires_approval: true,
    };
  }

  private static gitFetchTool(): Tool {
    return {
      name: "git_fetch",
      description: "Fetch changes from a remote without merging.",
      input_schema: objectSchema(
        {
          remote: {
            type: "string",
            description: "Remote name (default: origin)",
            default: "origin",
          },
          all: {
            type: "boolean",
            description: "Fetch all remotes",
            default: false,
          },
          prune: {
            type: "boolean",
            description: "Remove stale remote-tracking refs",
            default: false,
          },
        },
        [],
      ),
      requires_approval: false,
    };
  }

  private static gitDiscardTool(): Tool {
    return {
      name: "git_discard",
      description: "Discard uncommitted changes. WARNING: Permanent!",
      input_schema: objectSchema(
        {
          files: {
            type: "array",
            items: { type: "string" },
            description: "Files to discard changes for.",
          },
        },
        ["files"],
      ),
      requires_approval: true,
    };
  }

  private static gitBranchTool(): Tool {
    return {
      name: "git_branch",
      description: "Manage git branches: list, create, switch, or delete.",
      input_schema: objectSchema(
        {
          name: {
            type: "string",
            description: "Branch name",
          },
          action: {
            type: "string",
            enum: ["list", "create", "switch", "delete"],
            description: "Action to perform",
            default: "list",
          },
          force: {
            type: "boolean",
            description: "Force the action",
            default: false,
          },
        },
        [],
      ),
      requires_approval: true,
    };
  }

  /** Execute a git tool. */
  static async execute(
    toolUseId: string,
    toolName: string,
    input: any,
    context: ToolContext,
  ): Promise<ToolResult> {
    try {
      let output: string;
      switch (toolName) {
        case "git_status":
          output = await GitTool.gitStatus(context);
          break;
        case "git_diff":
          output = await GitTool.gitDiff(context);
          break;
        case "git_log":
          output = await GitTool.gitLog(input, context);
          break;
        case "git_stage":
          output = await GitTool.gitStage(input, context);
          break;
        case "git_unstage":
          output = await GitTool.gitUnstage(input, context);
          break;
        case "git_commit":
          output = await GitTool.gitCommit(input, context);
          break;
        case "git_push":
          output = await GitTool.gitPush(input, context);
          break;
        case "git_pull":
          output = await GitTool.gitPull(input, context);
          break;
        case "git_fetch":
          output = await GitTool.gitFetch(input, context);
          break;
        case "git_discard":
          output = await GitTool.gitDiscard(input, context);
          break;
        case "git_branch":
          output = await GitTool.gitBranch(input, context);
          break;
        default:
          return ToolResult.error(
            toolUseId,
            `Unknown git tool: ${toolName}`,
          );
      }
      return ToolResult.success(toolUseId, output);
    } catch (e) {
      return ToolResult.error(
        toolUseId,
        `Git operation failed: ${(e as Error).message}`,
      );
    }
  }

  /** Run a git command and return stdout. */
  private static async runGit(
    args: string[],
    cwd: string,
  ): Promise<{ stdout: string; stderr: string; success: boolean }> {
    const cmd = new Deno.Command("git", {
      args,
      cwd,
      stdout: "piped",
      stderr: "piped",
    });
    const output = await cmd.output();
    return {
      stdout: new TextDecoder().decode(output.stdout),
      stderr: new TextDecoder().decode(output.stderr),
      success: output.success,
    };
  }

  private static async gitStatus(context: ToolContext): Promise<string> {
    const result = await GitTool.runGit(
      ["status", "--porcelain=v1"],
      context.working_directory,
    );
    if (!result.success) {
      throw new Error(`git status failed: ${result.stderr}`);
    }
    return `Git Status:\n\n${result.stdout || "(clean)"}`;
  }

  private static async gitDiff(context: ToolContext): Promise<string> {
    const result = await GitTool.runGit(
      ["diff"],
      context.working_directory,
    );
    if (!result.success) {
      throw new Error(`git diff failed: ${result.stderr}`);
    }
    return `Git Diff:\n\n${result.stdout || "(no changes)"}`;
  }

  private static async gitLog(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const limit = input.limit ?? 10;
    const result = await GitTool.runGit(
      ["log", `--max-count=${limit}`, "--oneline"],
      context.working_directory,
    );
    if (!result.success) {
      throw new Error(`git log failed: ${result.stderr}`);
    }
    return `Git Log:\n\n${result.stdout}`;
  }

  private static async gitStage(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const files: string[] = input.files;
    const result = await GitTool.runGit(
      ["add", ...files],
      context.working_directory,
    );
    if (!result.success) {
      throw new Error(`Failed to stage files: ${result.stderr}`);
    }
    return `Successfully staged ${files.length} file(s)`;
  }

  private static async gitUnstage(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const files: string[] = input.files;
    const result = await GitTool.runGit(
      ["reset", "HEAD", "--", ...files],
      context.working_directory,
    );
    if (!result.success) {
      throw new Error(`Failed to unstage files: ${result.stderr}`);
    }
    return `Successfully unstaged ${files.length} file(s)`;
  }

  private static async gitCommit(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const args = ["commit"];
    if (input.all) args.push("-a");
    args.push("-m", input.message);

    const result = await GitTool.runGit(args, context.working_directory);
    if (!result.success) {
      throw new Error(`Commit failed: ${result.stderr}`);
    }
    return `Commit successful:\n${result.stdout}`;
  }

  private static async gitPush(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const remote = input.remote ?? "origin";
    const args = ["push"];
    if (input.set_upstream) args.push("-u");
    args.push(remote);
    if (input.branch) args.push(input.branch);

    const result = await GitTool.runGit(args, context.working_directory);
    if (!result.success) {
      throw new Error(`Push failed: ${result.stderr}`);
    }
    return `Push successful:\n${result.stdout}${result.stderr}`;
  }

  private static async gitPull(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const remote = input.remote ?? "origin";
    const args = ["pull"];
    if (input.rebase) args.push("--rebase");
    args.push(remote);
    if (input.branch) args.push(input.branch);

    const result = await GitTool.runGit(args, context.working_directory);
    if (!result.success) {
      throw new Error(`Pull failed: ${result.stderr}`);
    }
    return `Pull successful:\n${result.stdout}`;
  }

  private static async gitFetch(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const remote = input.remote ?? "origin";
    const args = ["fetch"];
    if (input.all) {
      args.push("--all");
    } else {
      args.push(remote);
    }
    if (input.prune) args.push("--prune");

    const result = await GitTool.runGit(args, context.working_directory);
    if (!result.success) {
      throw new Error(`Fetch failed: ${result.stderr}`);
    }
    const fetchOutput = (!result.stdout && !result.stderr)
      ? "Already up to date."
      : `${result.stdout}${result.stderr}`;
    return `Fetch successful:\n${fetchOutput}`;
  }

  private static async gitDiscard(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const files: string[] = input.files;
    const result = await GitTool.runGit(
      ["checkout", "--", ...files],
      context.working_directory,
    );
    if (!result.success) {
      throw new Error(`Failed to discard changes: ${result.stderr}`);
    }
    return `Successfully discarded changes to ${files.length} file(s)`;
  }

  private static async gitBranch(
    input: any,
    context: ToolContext,
  ): Promise<string> {
    const action = input.action ?? "list";
    const name: string | undefined = input.name;
    const force = input.force ?? false;

    let args: string[];
    switch (action) {
      case "list":
        args = ["branch", "-a", "-v"];
        break;
      case "create":
        if (!name) throw new Error("Branch name required");
        args = ["branch", name];
        break;
      case "switch":
        if (!name) throw new Error("Branch name required");
        args = ["checkout", name];
        break;
      case "delete":
        if (!name) throw new Error("Branch name required");
        args = force ? ["branch", "-D", name] : ["branch", "-d", name];
        break;
      default:
        throw new Error(`Unknown branch action: ${action}`);
    }

    const result = await GitTool.runGit(args, context.working_directory);
    if (!result.success) {
      throw new Error(`Branch operation failed: ${result.stderr}`);
    }

    switch (action) {
      case "list":
        return `Branches:\n${result.stdout}`;
      case "create":
        return `Created branch '${name}'`;
      case "switch":
        return `Switched to branch '${name}'`;
      case "delete":
        return `Deleted branch '${name}'`;
      default:
        return result.stdout;
    }
  }
}
