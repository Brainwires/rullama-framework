/**
 * Smart Tool Router
 *
 * Analyzes user queries to determine which tool categories are relevant.
 * Uses pure keyword-based pattern matching (no AI/inference dependencies).
 */

import type { Tool } from "@rullama/core";
import type { Message } from "@rullama/core";
import type { ToolCategory, ToolRegistry } from "./registry.ts";

/** Keyword patterns for each tool category. */
interface CategoryPatterns {
  category: ToolCategory;
  keywords: string[];
}

const CATEGORY_PATTERNS: CategoryPatterns[] = [
  {
    category: "FileOps",
    keywords: [
      "file",
      "read",
      "write",
      "edit",
      "create",
      "delete",
      "directory",
      "folder",
      "list",
      "content",
      "save",
      "open",
      "path",
      "rename",
      "move",
      "copy",
      "mkdir",
      "touch",
      "cat",
      "ls",
    ],
  },
  {
    category: "Search",
    keywords: [
      "search",
      "find",
      "grep",
      "look",
      "where",
      "locate",
      "pattern",
      "match",
      "regex",
      "occurrence",
      "rg",
      "ripgrep",
    ],
  },
  {
    category: "SemanticSearch",
    keywords: [
      "semantic",
      "meaning",
      "similar",
      "related",
      "understand",
      "codebase",
      "index",
      "rag",
      "embedding",
      "concept",
      "query",
    ],
  },
  {
    category: "Git",
    keywords: [
      "git",
      "commit",
      "diff",
      "branch",
      "merge",
      "push",
      "pull",
      "clone",
      "status",
      "log",
      "history",
      "version",
      "repository",
      "repo",
      "checkout",
      "stash",
      "rebase",
      "cherry-pick",
    ],
  },
  {
    category: "TaskManager",
    keywords: [
      "task",
      "todo",
      "progress",
      "complete",
      "pending",
      "assign",
      "track",
      "subtask",
      "dependency",
    ],
  },
  {
    category: "AgentPool",
    keywords: [
      "agent",
      "spawn",
      "parallel",
      "concurrent",
      "worker",
      "pool",
      "background",
      "async",
      "thread",
    ],
  },
  {
    category: "Web",
    keywords: [
      "url",
      "fetch",
      "http",
      "api",
      "endpoint",
      "request",
      "download",
      "curl",
      "get",
      "post",
    ],
  },
  {
    category: "WebSearch",
    keywords: [
      "web",
      "search",
      "google",
      "browse",
      "scrape",
      "internet",
      "online",
      "website",
      "page",
      "html",
      "duckduckgo",
      "bing",
    ],
  },
  {
    category: "Bash",
    keywords: [
      "run",
      "execute",
      "command",
      "shell",
      "bash",
      "terminal",
      "script",
      "npm",
      "cargo",
      "pip",
      "make",
      "build",
      "install",
      "test",
      "compile",
      "yarn",
      "pnpm",
      "docker",
      "kubectl",
    ],
  },
  {
    category: "Planning",
    keywords: [
      "plan",
      "design",
      "architect",
      "strategy",
      "approach",
      "implement",
      "roadmap",
      "outline",
    ],
  },
  {
    category: "Context",
    keywords: [
      "context",
      "remember",
      "recall",
      "previous",
      "earlier",
      "mentioned",
      "before",
      "history",
    ],
  },
  {
    category: "Orchestrator",
    keywords: [
      "batch",
      "loop",
      "iterate",
      "for each",
      "every",
      "all files",
      "multiple",
      "process all",
      "count all",
      "script",
      "orchestrate",
      "workflow",
      "automate",
      "chain",
      "pipeline",
      "sequential",
      "conditional",
      "each file",
    ],
  },
  {
    category: "CodeExecution",
    keywords: [
      "execute code",
      "run code",
      "python",
      "javascript",
      "compile",
      "interpreter",
      "sandbox",
      "piston",
      "run script",
    ],
  },
];

/**
 * Get context for analysis based on adaptive window.
 * - Short prompts (< 50 chars): analyze last 3 messages
 * - Medium prompts (50-150 chars): analyze last 2 messages
 * - Detailed prompts (> 150 chars): analyze current message only
 */
export function getContextForAnalysis(messages: Message[]): string {
  const lastMsg = messages.length > 0
    ? messages[messages.length - 1].text() ?? ""
    : "";
  const msgLen = lastMsg.length;

  if (msgLen > 150) {
    return lastMsg;
  } else if (msgLen > 50) {
    return messages
      .slice(-2)
      .reverse()
      .map((m) => m.text())
      .filter((t): t is string => t !== undefined)
      .join(" ");
  } else {
    return messages
      .slice(-3)
      .reverse()
      .map((m) => m.text())
      .filter((t): t is string => t !== undefined)
      .join(" ");
  }
}

/** Analyze a query and return relevant tool categories. */
export function analyzeQuery(query: string): ToolCategory[] {
  const queryLower = query.toLowerCase();
  const words = new Set(queryLower.split(/\s+/));

  const matchedCategories: ToolCategory[] = [];

  for (const pattern of CATEGORY_PATTERNS) {
    const hasMatch = pattern.keywords.some(
      (kw) => words.has(kw) || queryLower.includes(kw),
    );
    if (hasMatch) {
      matchedCategories.push(pattern.category);
    }
  }

  // If no specific categories matched, return default set
  if (matchedCategories.length === 0) {
    return ["FileOps", "Search", "Bash"];
  }

  // Always include FileOps as it's almost always useful
  if (!matchedCategories.includes("FileOps")) {
    matchedCategories.push("FileOps");
  }

  return matchedCategories;
}

/** Analyze conversation messages and return relevant tool categories. */
export function analyzeMessages(messages: Message[]): ToolCategory[] {
  const context = getContextForAnalysis(messages);
  return analyzeQuery(context);
}

/** Get tools for the given categories from the registry. */
export function getToolsForCategories(
  registry: ToolRegistry,
  categories: ToolCategory[],
): Tool[] {
  const tools: Tool[] = [];
  const seenNames = new Set<string>();

  for (const category of categories) {
    for (const tool of registry.getByCategory(category)) {
      if (!seenNames.has(tool.name)) {
        seenNames.add(tool.name);
        tools.push(tool);
      }
    }
  }

  return tools;
}

/** Get smart-routed tools for the given messages. */
export function getSmartTools(
  messages: Message[],
  registry: ToolRegistry,
): Tool[] {
  const categories = analyzeMessages(messages);
  return getToolsForCategories(registry, categories);
}

/**
 * Check if an MCP tool matches any of the detected categories based on keywords.
 */
function mcpToolMatchesCategories(
  tool: Tool,
  categories: ToolCategory[],
): boolean {
  const text = `${tool.name} ${tool.description}`.toLowerCase();

  for (const category of categories) {
    let matches = false;
    switch (category) {
      case "FileOps":
        matches = text.includes("file") || text.includes("read") ||
          text.includes("write") || text.includes("directory") ||
          text.includes("path") || text.includes("folder");
        break;
      case "Search":
        matches = text.includes("search") || text.includes("find") ||
          text.includes("query") || text.includes("lookup") ||
          text.includes("grep");
        break;
      case "SemanticSearch":
        matches = text.includes("semantic") || text.includes("embedding") ||
          text.includes("rag") || text.includes("vector") ||
          text.includes("similarity");
        break;
      case "Git":
        matches = text.includes("git") || text.includes("commit") ||
          text.includes("branch") || text.includes("pull") ||
          text.includes("push") || text.includes("repository");
        break;
      case "Web":
        matches = text.includes("http") || text.includes("url") ||
          text.includes("api") || text.includes("fetch") ||
          text.includes("request");
        break;
      case "WebSearch":
        matches = text.includes("web") || text.includes("browse") ||
          text.includes("scrape") || text.includes("google");
        break;
      case "Bash":
        matches = text.includes("shell") || text.includes("exec") ||
          text.includes("command") || text.includes("run") ||
          text.includes("terminal");
        break;
      case "TaskManager":
        matches = text.includes("task") || text.includes("todo") ||
          text.includes("issue") || text.includes("ticket");
        break;
      case "AgentPool":
        matches = text.includes("agent") || text.includes("spawn") ||
          text.includes("worker") || text.includes("parallel");
        break;
      case "Planning":
        matches = text.includes("plan") || text.includes("design") ||
          text.includes("architect");
        break;
      case "Context":
        matches = text.includes("context") || text.includes("recall") ||
          text.includes("memory");
        break;
      case "Orchestrator":
        matches = text.includes("script") || text.includes("orchestrat") ||
          text.includes("automat") || text.includes("workflow") ||
          text.includes("batch");
        break;
      case "CodeExecution":
        matches = text.includes("execute") || text.includes("run") ||
          text.includes("code") || text.includes("python") ||
          text.includes("javascript") || text.includes("compile");
        break;
      case "SessionTask":
        // Session tasks are internal-only, never match external MCP tools
        matches = false;
        break;
      case "Validation":
        matches = text.includes("valid") || text.includes("check") ||
          text.includes("verify") || text.includes("lint") ||
          text.includes("build") || text.includes("syntax") ||
          text.includes("duplicate") || text.includes("test");
        break;
    }
    if (matches) return true;
  }
  return false;
}

/** Get smart-routed tools including MCP tools that match detected categories. */
export function getSmartToolsWithMcp(
  messages: Message[],
  registry: ToolRegistry,
  mcpTools: Tool[],
): Tool[] {
  const categories = analyzeMessages(messages);
  const tools = getToolsForCategories(registry, categories);
  const seenNames = new Set(tools.map((t) => t.name));

  for (const mcpTool of mcpTools) {
    if (
      !seenNames.has(mcpTool.name) &&
      mcpToolMatchesCategories(mcpTool, categories)
    ) {
      tools.push(mcpTool);
      seenNames.add(mcpTool.name);
    }
  }

  return tools;
}
