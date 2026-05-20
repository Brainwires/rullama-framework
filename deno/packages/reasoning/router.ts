/**
 * Semantic query → tool-category router.
 *
 * Categories match the strings used by `@brainwires/tools`' ToolCategory
 * enum. Consumers can remap them to their own taxonomy.
 *
 * Equivalent to Rust's `brainwires_reasoning::router` module.
 */

import { ChatOptions, Message, type Provider } from "@brainwires/core";

/** Tool category tag — same labels as `@brainwires/tools`' ToolCategory. */
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
  | "CodeExecution";

export interface RouteResult {
  categories: ToolCategory[];
  confidence: number;
  used_local_llm: boolean;
}

export function routeFromFallback(categories: ToolCategory[]): RouteResult {
  return { categories, confidence: 0.5, used_local_llm: false };
}

export function routeFromLocal(categories: ToolCategory[], confidence: number): RouteResult {
  return { categories, confidence, used_local_llm: true };
}

const SYSTEM_PROMPT = `You are a tool category classifier. Given a user query, output the relevant tool categories.

Available categories:
- FileOps: File operations (read, write, edit, create, delete, list files/directories)
- Search: Text search (grep, find patterns, locate text)
- SemanticSearch: Semantic/concept search (codebase queries, embeddings, RAG)
- Git: Git operations (commit, diff, branch, merge, status, log)
- TaskManager: Task tracking (todos, progress, subtasks)
- AgentPool: Multi-agent operations (spawn, parallel, background)
- Web: HTTP/API operations (fetch, request, download)
- WebSearch: Internet search (google, browse, scrape)
- Bash: Shell commands (run, execute, npm, cargo, pip, docker)
- Planning: Design/architecture (plan, strategy, roadmap)
- Context: Memory/recall (remember, previous, earlier)
- Orchestrator: Script automation (workflow, batch)
- CodeExecution: Code execution (run code, python, javascript)

Rules:
1. Output ONLY category names, comma-separated
2. Include multiple categories if query spans multiple domains
3. Always include FileOps if file operations might be needed
4. Be conservative - only include clearly relevant categories`;

const KEYWORD_CATEGORIES: Array<[string, ToolCategory]> = [
  ["fileops", "FileOps"],
  ["file", "FileOps"],
  ["search", "Search"],
  ["semanticsearch", "SemanticSearch"],
  ["semantic", "SemanticSearch"],
  ["git", "Git"],
  ["taskmanager", "TaskManager"],
  ["task", "TaskManager"],
  ["agentpool", "AgentPool"],
  ["agent", "AgentPool"],
  ["web", "Web"],
  ["websearch", "WebSearch"],
  ["bash", "Bash"],
  ["shell", "Bash"],
  ["planning", "Planning"],
  ["plan", "Planning"],
  ["context", "Context"],
  ["orchestrator", "Orchestrator"],
  ["codeexecution", "CodeExecution"],
  ["code", "CodeExecution"],
];

/** Extract the categories referenced in free-form text. */
export function parseCategories(output: string): ToolCategory[] {
  const lower = output.toLowerCase();
  const out: ToolCategory[] = [];
  for (const [kw, cat] of KEYWORD_CATEGORIES) {
    if (lower.includes(kw) && !out.includes(cat)) out.push(cat);
  }
  return out;
}

/** Provider-backed semantic router. */
export class LocalRouter {
  readonly provider: Provider;
  readonly model_id: string;

  constructor(provider: Provider, model_id: string) {
    this.provider = provider;
    this.model_id = model_id;
  }

  async classify(query: string): Promise<RouteResult | null> {
    const user = `Classify this query into tool categories. Output ONLY the category names, comma-separated.

Query: ${query}`;
    const options = ChatOptions.deterministic(50);
    options.setSystem(SYSTEM_PROMPT);
    try {
      const resp = await this.provider.chat([Message.user(user)], undefined, options);
      const text = resp.message.textOrSummary();
      const cats = parseCategories(text);
      if (cats.length === 0) return null;
      return routeFromLocal(cats, 0.85);
    } catch {
      return null;
    }
  }
}
