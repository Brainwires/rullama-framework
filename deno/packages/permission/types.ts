/**
 * Core permission system types.
 *
 * Defines the capability-based permission model for agents, including
 * filesystem, tool, network, spawning, git, and quota capabilities.
 *
 * Rust equivalent: `rullama-permissions/src/types.rs`
 * @module
 */

// ── Tool Categories ─────────────────────────────────────────────────

/**
 * Tool categories for permission grouping.
 *
 * Rust equivalent: `ToolCategory` enum
 */
export type ToolCategory =
  | "FileRead"
  | "FileWrite"
  | "Search"
  | "Git"
  | "GitDestructive"
  | "Bash"
  | "Web"
  | "CodeExecution"
  | "AgentSpawn"
  | "Planning"
  | "System";

/** All valid ToolCategory values. */
export const ALL_TOOL_CATEGORIES: readonly ToolCategory[] = [
  "FileRead",
  "FileWrite",
  "Search",
  "Git",
  "GitDestructive",
  "Bash",
  "Web",
  "CodeExecution",
  "AgentSpawn",
  "Planning",
  "System",
] as const;

// ── Git Operations ──────────────────────────────────────────────────

/**
 * Git operations.
 *
 * Rust equivalent: `GitOperation` enum
 */
export type GitOperation =
  | "Status"
  | "Diff"
  | "Log"
  | "Add"
  | "Commit"
  | "Push"
  | "Pull"
  | "Fetch"
  | "Branch"
  | "Checkout"
  | "Merge"
  | "Rebase"
  | "Reset"
  | "Stash"
  | "Tag"
  | "ForcePush";

/**
 * Check if a git operation is destructive.
 *
 * Rust equivalent: `GitOperation::is_destructive()`
 */
export function isDestructiveGitOp(op: GitOperation): boolean {
  return op === "Rebase" || op === "Reset" || op === "ForcePush" ||
    op === "Merge";
}

// ── Path Pattern ────────────────────────────────────────────────────

/**
 * Path pattern for glob matching.
 *
 * Rust equivalent: `PathPattern` struct (serialized as transparent string)
 */
export class PathPattern {
  readonly #pattern: string;

  constructor(pattern: string) {
    this.#pattern = pattern;
  }

  /** Create a glob pattern (alias for constructor). Rust equivalent: `PathPattern::glob()` */
  static glob(pattern: string): PathPattern {
    return new PathPattern(pattern);
  }

  /** Get the pattern string. Rust equivalent: `PathPattern::pattern()` */
  pattern(): string {
    return this.#pattern;
  }

  /**
   * Check if a path matches this pattern.
   *
   * Implements simple glob matching: `*` matches any single segment,
   * `**` matches zero or more segments, `?` matches any single character.
   *
   * Rust equivalent: `PathPattern::matches()`
   */
  matches(path: string): boolean {
    return globMatch(this.#pattern, path);
  }

  /** JSON serialization produces a plain string. */
  toJSON(): string {
    return this.#pattern;
  }

  /** Deserialize from a plain string. */
  static fromJSON(value: string): PathPattern {
    return new PathPattern(value);
  }
}

/**
 * Simple glob matcher supporting `*`, `**`, and `?`.
 *
 * This is a pragmatic port — it covers the patterns used in the
 * Rust crate without pulling in a full glob library.
 */
function globMatch(pattern: string, path: string): boolean {
  // Convert glob pattern to regex
  let regexStr = "^";
  let i = 0;
  while (i < pattern.length) {
    const c = pattern[i];
    if (c === "*") {
      if (i + 1 < pattern.length && pattern[i + 1] === "*") {
        // `**` — match anything (including slashes)
        i += 2;
        if (i < pattern.length && pattern[i] === "/") {
          i++; // skip trailing slash after **
          regexStr += "(?:.*/)?";
        } else {
          regexStr += ".*";
        }
      } else {
        // `*` — match anything except `/`
        regexStr += "[^/]*";
        i++;
      }
    } else if (c === "?") {
      regexStr += "[^/]";
      i++;
    } else if (c === ".") {
      regexStr += "\\.";
      i++;
    } else if (
      c === "(" || c === ")" || c === "{" || c === "}" || c === "+" ||
      c === "|" || c === "^" || c === "$" || c === "\\"
    ) {
      regexStr += "\\" + c;
      i++;
    } else {
      regexStr += c;
      i++;
    }
  }
  regexStr += "$";

  try {
    const re = new RegExp(regexStr);
    return re.test(path);
  } catch {
    // Fall back to simple string matching if pattern is invalid
    return path.includes(pattern);
  }
}

// ── Filesystem Capabilities ─────────────────────────────────────────

/**
 * File system capabilities.
 *
 * Rust equivalent: `FilesystemCapabilities` struct
 */
export interface FilesystemCapabilities {
  /** Allowed read paths (glob patterns). */
  read_paths: PathPattern[];
  /** Allowed write paths (glob patterns). */
  write_paths: PathPattern[];
  /** Denied paths (override allows). */
  denied_paths: PathPattern[];
  /** Can follow symlinks outside allowed paths. */
  follow_symlinks: boolean;
  /** Can access hidden files (dotfiles). */
  access_hidden: boolean;
  /** Maximum file size for write operations (bytes). */
  max_write_size: number | undefined;
  /** Can delete files. */
  can_delete: boolean;
  /** Can create directories. */
  can_create_dirs: boolean;
}

/** Create default filesystem capabilities. Rust equivalent: `FilesystemCapabilities::default()` */
export function defaultFilesystemCapabilities(): FilesystemCapabilities {
  return {
    read_paths: [new PathPattern("**/*")],
    write_paths: [],
    denied_paths: [
      new PathPattern("**/.env*"),
      new PathPattern("**/*credentials*"),
      new PathPattern("**/*secret*"),
    ],
    follow_symlinks: true,
    access_hidden: true,
    max_write_size: undefined,
    can_delete: false,
    can_create_dirs: true,
  };
}

/** Create full-access filesystem capabilities. Rust equivalent: `FilesystemCapabilities::full()` */
export function fullFilesystemCapabilities(): FilesystemCapabilities {
  return {
    read_paths: [new PathPattern("**/*")],
    write_paths: [new PathPattern("**/*")],
    denied_paths: [],
    follow_symlinks: true,
    access_hidden: true,
    max_write_size: undefined,
    can_delete: true,
    can_create_dirs: true,
  };
}

// ── Tool Capabilities ───────────────────────────────────────────────

/**
 * Tool execution capabilities.
 *
 * Rust equivalent: `ToolCapabilities` struct
 */
export interface ToolCapabilities {
  /** Tool categories allowed. */
  allowed_categories: Set<ToolCategory>;
  /** Specific tools denied (overrides category allows). */
  denied_tools: Set<string>;
  /** Specific tools allowed (if not using categories). */
  allowed_tools: Set<string> | undefined;
  /** Require approval for these tools regardless of trust. */
  always_approve: Set<string>;
}

/** Create default tool capabilities. Rust equivalent: `ToolCapabilities::default()` */
export function defaultToolCapabilities(): ToolCapabilities {
  return {
    allowed_categories: new Set<ToolCategory>(["FileRead", "Search", "Web"]),
    denied_tools: new Set<string>(),
    allowed_tools: undefined,
    always_approve: new Set<string>(),
  };
}

/** Create full-access tool capabilities. Rust equivalent: `ToolCapabilities::full()` */
export function fullToolCapabilities(): ToolCapabilities {
  return {
    allowed_categories: new Set<ToolCategory>(ALL_TOOL_CATEGORIES),
    denied_tools: new Set<string>(),
    allowed_tools: undefined,
    always_approve: new Set<string>(),
  };
}

// ── Network Capabilities ────────────────────────────────────────────

/**
 * Network capabilities.
 *
 * Rust equivalent: `NetworkCapabilities` struct
 */
export interface NetworkCapabilities {
  /** Allowed domains (supports wildcards like *.github.com). */
  allowed_domains: string[];
  /** Denied domains (override allows). */
  denied_domains: string[];
  /** Allow all domains (use with caution). */
  allow_all: boolean;
  /** Rate limit (requests per minute). */
  rate_limit: number | undefined;
  /** Can make external API calls. */
  allow_api_calls: boolean;
  /** Maximum response size to process (bytes). */
  max_response_size: number | undefined;
}

/** Create default network capabilities. Rust equivalent: `NetworkCapabilities::default()` */
export function defaultNetworkCapabilities(): NetworkCapabilities {
  return {
    allowed_domains: [],
    denied_domains: [],
    allow_all: false,
    rate_limit: 60,
    allow_api_calls: false,
    max_response_size: 10 * 1024 * 1024,
  };
}

/** Create disabled network capabilities. Rust equivalent: `NetworkCapabilities::disabled()` */
export function disabledNetworkCapabilities(): NetworkCapabilities {
  return {
    allowed_domains: [],
    denied_domains: [],
    allow_all: false,
    rate_limit: 0,
    allow_api_calls: false,
    max_response_size: undefined,
  };
}

/** Create full network capabilities. Rust equivalent: `NetworkCapabilities::full()` */
export function fullNetworkCapabilities(): NetworkCapabilities {
  return {
    allowed_domains: [],
    denied_domains: [],
    allow_all: true,
    rate_limit: undefined,
    allow_api_calls: true,
    max_response_size: undefined,
  };
}

// ── Spawning Capabilities ───────────────────────────────────────────

/**
 * Agent spawning capabilities.
 *
 * Rust equivalent: `SpawningCapabilities` struct
 */
export interface SpawningCapabilities {
  /** Can spawn child agents. */
  can_spawn: boolean;
  /** Maximum concurrent child agents. */
  max_children: number;
  /** Maximum depth of agent hierarchy. */
  max_depth: number;
  /** Can spawn agents with elevated privileges (requires approval). */
  can_elevate: boolean;
}

/** Create default spawning capabilities. Rust equivalent: `SpawningCapabilities::default()` */
export function defaultSpawningCapabilities(): SpawningCapabilities {
  return {
    can_spawn: false,
    max_children: 3,
    max_depth: 2,
    can_elevate: false,
  };
}

/** Create disabled spawning capabilities. Rust equivalent: `SpawningCapabilities::disabled()` */
export function disabledSpawningCapabilities(): SpawningCapabilities {
  return {
    can_spawn: false,
    max_children: 0,
    max_depth: 0,
    can_elevate: false,
  };
}

/** Create full spawning capabilities. Rust equivalent: `SpawningCapabilities::full()` */
export function fullSpawningCapabilities(): SpawningCapabilities {
  return { can_spawn: true, max_children: 10, max_depth: 5, can_elevate: true };
}

// ── Git Capabilities ────────────────────────────────────────────────

/**
 * Git operation capabilities.
 *
 * Rust equivalent: `GitCapabilities` struct
 */
export interface GitCapabilities {
  /** Allowed operations. */
  allowed_ops: Set<GitOperation>;
  /** Protected branches (cannot push directly). */
  protected_branches: string[];
  /** Can force push (dangerous). */
  can_force_push: boolean;
  /** Can perform destructive operations. */
  can_destructive: boolean;
  /** Require PR for these branches. */
  require_pr_branches: string[];
}

/** Create default git capabilities. Rust equivalent: `GitCapabilities::default()` */
export function defaultGitCapabilities(): GitCapabilities {
  return {
    allowed_ops: new Set<GitOperation>(["Status", "Diff", "Log"]),
    protected_branches: ["main", "master"],
    can_force_push: false,
    can_destructive: false,
    require_pr_branches: [],
  };
}

/** Create read-only git capabilities. Rust equivalent: `GitCapabilities::read_only()` */
export function readOnlyGitCapabilities(): GitCapabilities {
  return {
    allowed_ops: new Set<GitOperation>(["Status", "Diff", "Log", "Fetch"]),
    protected_branches: ["main", "master"],
    can_force_push: false,
    can_destructive: false,
    require_pr_branches: [],
  };
}

/** Create standard git capabilities. Rust equivalent: `GitCapabilities::standard()` */
export function standardGitCapabilities(): GitCapabilities {
  return {
    allowed_ops: new Set<GitOperation>([
      "Status",
      "Diff",
      "Log",
      "Add",
      "Commit",
      "Push",
      "Pull",
      "Fetch",
      "Branch",
      "Checkout",
      "Stash",
    ]),
    protected_branches: ["main", "master"],
    can_force_push: false,
    can_destructive: false,
    require_pr_branches: [],
  };
}

/** Create full git capabilities. Rust equivalent: `GitCapabilities::full()` */
export function fullGitCapabilities(): GitCapabilities {
  return {
    allowed_ops: new Set<GitOperation>([
      "Status",
      "Diff",
      "Log",
      "Add",
      "Commit",
      "Push",
      "Pull",
      "Fetch",
      "Branch",
      "Checkout",
      "Merge",
      "Rebase",
      "Reset",
      "Stash",
      "Tag",
      "ForcePush",
    ]),
    protected_branches: [],
    can_force_push: true,
    can_destructive: true,
    require_pr_branches: [],
  };
}

// ── Resource Quotas ─────────────────────────────────────────────────

/**
 * Resource quota limits.
 *
 * Rust equivalent: `ResourceQuotas` struct
 */
export interface ResourceQuotas {
  /** Maximum execution time (seconds). */
  max_execution_time: number | undefined;
  /** Maximum memory usage (bytes). */
  max_memory: number | undefined;
  /** Maximum API tokens consumed. */
  max_tokens: number | undefined;
  /** Maximum tool calls per session. */
  max_tool_calls: number | undefined;
  /** Maximum files modified per session. */
  max_files_modified: number | undefined;
}

/** Create default resource quotas. Rust equivalent: `ResourceQuotas::default()` */
export function defaultResourceQuotas(): ResourceQuotas {
  return {
    max_execution_time: 30 * 60,
    max_memory: undefined,
    max_tokens: 100_000,
    max_tool_calls: 500,
    max_files_modified: 50,
  };
}

/** Create conservative resource quotas. Rust equivalent: `ResourceQuotas::conservative()` */
export function conservativeResourceQuotas(): ResourceQuotas {
  return {
    max_execution_time: 5 * 60,
    max_memory: 512 * 1024 * 1024,
    max_tokens: 10_000,
    max_tool_calls: 50,
    max_files_modified: 10,
  };
}

/** Create standard resource quotas. Rust equivalent: `ResourceQuotas::standard()` */
export function standardResourceQuotas(): ResourceQuotas {
  return defaultResourceQuotas();
}

/** Create generous resource quotas. Rust equivalent: `ResourceQuotas::generous()` */
export function generousResourceQuotas(): ResourceQuotas {
  return {
    max_execution_time: 2 * 60 * 60,
    max_memory: undefined,
    max_tokens: 500_000,
    max_tool_calls: 2000,
    max_files_modified: 200,
  };
}

// ── Agent Capabilities ──────────────────────────────────────────────

/**
 * Agent capabilities — explicit permissions granted to an agent.
 *
 * Rust equivalent: `AgentCapabilities` struct
 */
export class AgentCapabilities {
  /** Unique capability set ID for auditing. */
  capability_id: string;
  /** File system capabilities. */
  filesystem: FilesystemCapabilities;
  /** Tool execution capabilities. */
  tools: ToolCapabilities;
  /** Network capabilities. */
  network: NetworkCapabilities;
  /** Agent spawning capabilities. */
  spawning: SpawningCapabilities;
  /** Git operation capabilities. */
  git: GitCapabilities;
  /** Resource quota limits. */
  quotas: ResourceQuotas;

  constructor(init?: Partial<AgentCapabilities>) {
    this.capability_id = init?.capability_id ?? crypto.randomUUID();
    this.filesystem = init?.filesystem ?? defaultFilesystemCapabilities();
    this.tools = init?.tools ?? defaultToolCapabilities();
    this.network = init?.network ?? defaultNetworkCapabilities();
    this.spawning = init?.spawning ?? defaultSpawningCapabilities();
    this.git = init?.git ?? defaultGitCapabilities();
    this.quotas = init?.quotas ?? defaultResourceQuotas();
  }

  /**
   * Categorize a tool by name into a ToolCategory.
   *
   * Rust equivalent: `AgentCapabilities::categorize_tool()`
   */
  static categorizeTool(toolName: string): ToolCategory {
    switch (toolName) {
      case "read_file":
      case "list_directory":
      case "search_files":
        return "FileRead";
      case "write_file":
      case "edit_file":
      case "patch_file":
      case "delete_file":
      case "create_directory":
        return "FileWrite";
      case "search_code":
      case "index_codebase":
      case "query_codebase":
      case "search_with_filters":
      case "get_rag_statistics":
      case "clear_rag_index":
      case "search_git_history":
      case "recall_context":
      case "search_tools":
        return "Search";
      case "execute_command":
        return "Bash";
      case "fetch_url":
      case "web_search":
      case "web_browse":
      case "web_scrape":
        return "Web";
      case "execute_code":
      case "execute_script":
        return "CodeExecution";
      case "agent_spawn":
      case "agent_stop":
      case "agent_status":
      case "agent_list":
      case "agent_pool_stats":
      case "agent_file_locks":
        return "AgentSpawn";
      case "plan_task":
      case "task_create":
      case "task_add_subtask":
      case "task_start":
      case "task_complete":
      case "task_fail":
      case "task_list":
      case "task_get":
        return "Planning";
      default:
        if (toolName.startsWith("git_")) {
          if (
            toolName.includes("force") ||
            toolName.includes("reset") ||
            toolName.includes("rebase") ||
            toolName.includes("delete_branch")
          ) {
            return "GitDestructive";
          }
          return "Git";
        }
        if (toolName.startsWith("mcp_")) {
          return "System";
        }
        return "System";
    }
  }

  /**
   * Check if a tool is allowed by the current capabilities.
   *
   * Rust equivalent: `AgentCapabilities::allows_tool()`
   */
  allowsTool(toolName: string): boolean {
    if (this.tools.denied_tools.has(toolName)) return false;

    if (this.tools.allowed_tools !== undefined) {
      return this.tools.allowed_tools.has(toolName);
    }

    const category = AgentCapabilities.categorizeTool(toolName);
    return this.tools.allowed_categories.has(category);
  }

  /**
   * Check if a tool requires explicit approval.
   *
   * Rust equivalent: `AgentCapabilities::requires_approval()`
   */
  requiresApproval(toolName: string): boolean {
    return this.tools.always_approve.has(toolName);
  }

  /**
   * Check if a file path is allowed for reading.
   *
   * Rust equivalent: `AgentCapabilities::allows_read()`
   */
  allowsRead(path: string): boolean {
    for (const denied of this.filesystem.denied_paths) {
      if (denied.matches(path)) return false;
    }
    for (const allowed of this.filesystem.read_paths) {
      if (allowed.matches(path)) return true;
    }
    return false;
  }

  /**
   * Check if a file path is allowed for writing.
   *
   * Rust equivalent: `AgentCapabilities::allows_write()`
   */
  allowsWrite(path: string): boolean {
    for (const denied of this.filesystem.denied_paths) {
      if (denied.matches(path)) return false;
    }
    for (const allowed of this.filesystem.write_paths) {
      if (allowed.matches(path)) return true;
    }
    return false;
  }

  /**
   * Check if a domain is allowed for network access.
   *
   * Rust equivalent: `AgentCapabilities::allows_domain()`
   */
  allowsDomain(domain: string): boolean {
    for (const denied of this.network.denied_domains) {
      if (domainMatches(denied, domain)) return false;
    }
    if (this.network.allow_all) return true;
    for (const allowed of this.network.allowed_domains) {
      if (domainMatches(allowed, domain)) return true;
    }
    return false;
  }

  /**
   * Check if a git operation is allowed.
   *
   * Rust equivalent: `AgentCapabilities::allows_git_op()`
   */
  allowsGitOp(op: GitOperation): boolean {
    if (isDestructiveGitOp(op) && !this.git.can_destructive) return false;
    if (op === "ForcePush" && !this.git.can_force_push) return false;
    return this.git.allowed_ops.has(op);
  }

  /**
   * Check if spawning agents is allowed.
   *
   * Rust equivalent: `AgentCapabilities::can_spawn_agent()`
   */
  canSpawnAgent(currentChildren: number, currentDepth: number): boolean {
    if (!this.spawning.can_spawn) return false;
    if (currentChildren >= this.spawning.max_children) return false;
    if (currentDepth >= this.spawning.max_depth) return false;
    return true;
  }

  // ── Profile factory methods ─────────────────────────────────────

  /**
   * Read-only exploration — safe for untrusted agents.
   *
   * Rust equivalent: `AgentCapabilities::read_only()`
   */
  static readOnly(): AgentCapabilities {
    return new AgentCapabilities({
      filesystem: {
        read_paths: [new PathPattern("**/*")],
        write_paths: [],
        denied_paths: [
          new PathPattern("**/.env*"),
          new PathPattern("**/*credentials*"),
          new PathPattern("**/*secret*"),
          new PathPattern("**/*.pem"),
          new PathPattern("**/*.key"),
        ],
        follow_symlinks: false,
        access_hidden: false,
        can_delete: false,
        can_create_dirs: false,
        max_write_size: undefined,
      },
      tools: {
        allowed_categories: new Set<ToolCategory>(["FileRead", "Search"]),
        denied_tools: new Set<string>(),
        allowed_tools: undefined,
        always_approve: new Set<string>(),
      },
      network: disabledNetworkCapabilities(),
      spawning: disabledSpawningCapabilities(),
      git: readOnlyGitCapabilities(),
      quotas: conservativeResourceQuotas(),
    });
  }

  /**
   * Standard development — balanced safety and utility.
   *
   * Rust equivalent: `AgentCapabilities::standard_dev()`
   */
  static standardDev(): AgentCapabilities {
    return new AgentCapabilities({
      filesystem: {
        read_paths: [new PathPattern("**/*")],
        write_paths: [
          new PathPattern("src/**"),
          new PathPattern("tests/**"),
          new PathPattern("docs/**"),
          new PathPattern("scripts/**"),
          new PathPattern("*.toml"),
          new PathPattern("*.json"),
          new PathPattern("*.yaml"),
          new PathPattern("*.yml"),
          new PathPattern("*.md"),
          new PathPattern("Makefile"),
          new PathPattern(".gitignore"),
        ],
        denied_paths: [
          new PathPattern("**/.env*"),
          new PathPattern("**/*credentials*"),
          new PathPattern("**/*secret*"),
          new PathPattern("**/node_modules/**"),
          new PathPattern("**/target/**"),
          new PathPattern("**/.git/**"),
        ],
        follow_symlinks: true,
        access_hidden: true,
        can_delete: true,
        can_create_dirs: true,
        max_write_size: 1024 * 1024,
      },
      tools: {
        allowed_categories: new Set<ToolCategory>([
          "FileRead",
          "FileWrite",
          "Search",
          "Git",
          "Planning",
          "Web",
        ]),
        denied_tools: new Set<string>(["execute_code"]),
        allowed_tools: undefined,
        always_approve: new Set<string>(["delete_file", "execute_command"]),
      },
      network: {
        allowed_domains: [
          "github.com",
          "*.github.com",
          "docs.rs",
          "crates.io",
          "npmjs.com",
          "*.npmjs.com",
          "pypi.org",
          "stackoverflow.com",
        ],
        denied_domains: [],
        allow_all: false,
        rate_limit: 60,
        allow_api_calls: true,
        max_response_size: 10 * 1024 * 1024,
      },
      spawning: {
        can_spawn: true,
        max_children: 3,
        max_depth: 2,
        can_elevate: false,
      },
      git: standardGitCapabilities(),
      quotas: standardResourceQuotas(),
    });
  }

  /**
   * Full access — for trusted orchestrators.
   *
   * Rust equivalent: `AgentCapabilities::full_access()`
   */
  static fullAccess(): AgentCapabilities {
    return new AgentCapabilities({
      filesystem: fullFilesystemCapabilities(),
      tools: fullToolCapabilities(),
      network: fullNetworkCapabilities(),
      spawning: fullSpawningCapabilities(),
      git: fullGitCapabilities(),
      quotas: generousResourceQuotas(),
    });
  }

  /**
   * Create capabilities from a profile.
   *
   * Rust equivalent: `AgentCapabilities::from_profile()`
   */
  static fromProfile(profile: CapabilityProfile): AgentCapabilities {
    switch (profile) {
      case "read_only":
        return AgentCapabilities.readOnly();
      case "standard_dev":
        return AgentCapabilities.standardDev();
      case "full_access":
        return AgentCapabilities.fullAccess();
      case "custom":
        return new AgentCapabilities();
    }
  }

  /**
   * Create a child capability set that is a subset of the parent.
   * Child capabilities can never exceed parent capabilities.
   *
   * Rust equivalent: `AgentCapabilities::derive_child()`
   */
  deriveChild(): AgentCapabilities {
    const child = new AgentCapabilities({
      capability_id: crypto.randomUUID(),
      filesystem: {
        ...this.filesystem,
        read_paths: [...this.filesystem.read_paths],
        write_paths: [...this.filesystem.write_paths],
        denied_paths: [...this.filesystem.denied_paths],
      },
      tools: {
        allowed_categories: new Set(this.tools.allowed_categories),
        denied_tools: new Set(this.tools.denied_tools),
        allowed_tools: this.tools.allowed_tools
          ? new Set(this.tools.allowed_tools)
          : undefined,
        always_approve: new Set(this.tools.always_approve),
      },
      network: {
        ...this.network,
        allowed_domains: [...this.network.allowed_domains],
        denied_domains: [...this.network.denied_domains],
      },
      spawning: { ...this.spawning },
      git: {
        allowed_ops: new Set(this.git.allowed_ops),
        protected_branches: [...this.git.protected_branches],
        can_force_push: this.git.can_force_push,
        can_destructive: this.git.can_destructive,
        require_pr_branches: [...this.git.require_pr_branches],
      },
      quotas: { ...this.quotas },
    });

    if (child.spawning.max_depth > 0) child.spawning.max_depth -= 1;
    child.spawning.can_elevate = false;

    return child;
  }

  /**
   * Merge capabilities, taking the more restrictive option for each field.
   *
   * Rust equivalent: `AgentCapabilities::intersect()`
   */
  intersect(other: AgentCapabilities): AgentCapabilities {
    const intersectSets = <T>(a: Set<T>, b: Set<T>): Set<T> => {
      const result = new Set<T>();
      for (const item of a) if (b.has(item)) result.add(item);
      return result;
    };
    const unionSets = <T>(a: Set<T>, b: Set<T>): Set<T> => {
      const result = new Set<T>(a);
      for (const item of b) result.add(item);
      return result;
    };
    const minOpt = (
      a: number | undefined,
      b: number | undefined,
    ): number | undefined => {
      if (a !== undefined && b !== undefined) return Math.min(a, b);
      return a ?? b;
    };

    return new AgentCapabilities({
      filesystem: {
        read_paths: this.filesystem.read_paths.filter(
          (p) =>
            other.filesystem.read_paths.some((op) =>
              op.pattern() === p.pattern()
            ),
        ),
        write_paths: this.filesystem.write_paths.filter(
          (p) =>
            other.filesystem.write_paths.some((op) =>
              op.pattern() === p.pattern()
            ),
        ),
        denied_paths: (() => {
          const denied = [...this.filesystem.denied_paths];
          for (const p of other.filesystem.denied_paths) {
            if (!denied.some((dp) => dp.pattern() === p.pattern())) {
              denied.push(p);
            }
          }
          return denied;
        })(),
        follow_symlinks: this.filesystem.follow_symlinks &&
          other.filesystem.follow_symlinks,
        access_hidden: this.filesystem.access_hidden &&
          other.filesystem.access_hidden,
        can_delete: this.filesystem.can_delete && other.filesystem.can_delete,
        can_create_dirs: this.filesystem.can_create_dirs &&
          other.filesystem.can_create_dirs,
        max_write_size: minOpt(
          this.filesystem.max_write_size,
          other.filesystem.max_write_size,
        ),
      },
      tools: {
        allowed_categories: intersectSets(
          this.tools.allowed_categories,
          other.tools.allowed_categories,
        ),
        denied_tools: unionSets(
          this.tools.denied_tools,
          other.tools.denied_tools,
        ),
        allowed_tools: (() => {
          const a = this.tools.allowed_tools;
          const b = other.tools.allowed_tools;
          if (a && b) return intersectSets(a, b);
          return a ?? b;
        })(),
        always_approve: unionSets(
          this.tools.always_approve,
          other.tools.always_approve,
        ),
      },
      network: {
        allowed_domains: this.network.allowed_domains.filter(
          (d) =>
            other.network.allowed_domains.includes(d) ||
            other.network.allow_all,
        ),
        denied_domains: [
          ...new Set([
            ...this.network.denied_domains,
            ...other.network.denied_domains,
          ]),
        ].sort(),
        allow_all: this.network.allow_all && other.network.allow_all,
        rate_limit: minOpt(this.network.rate_limit, other.network.rate_limit),
        allow_api_calls: this.network.allow_api_calls &&
          other.network.allow_api_calls,
        max_response_size: minOpt(
          this.network.max_response_size,
          other.network.max_response_size,
        ),
      },
      spawning: {
        can_spawn: this.spawning.can_spawn && other.spawning.can_spawn,
        max_children: Math.min(
          this.spawning.max_children,
          other.spawning.max_children,
        ),
        max_depth: Math.min(this.spawning.max_depth, other.spawning.max_depth),
        can_elevate: this.spawning.can_elevate && other.spawning.can_elevate,
      },
      git: {
        allowed_ops: intersectSets(this.git.allowed_ops, other.git.allowed_ops),
        protected_branches: [
          ...new Set([
            ...this.git.protected_branches,
            ...other.git.protected_branches,
          ]),
        ].sort(),
        can_force_push: this.git.can_force_push && other.git.can_force_push,
        can_destructive: this.git.can_destructive && other.git.can_destructive,
        require_pr_branches: [
          ...new Set([
            ...this.git.require_pr_branches,
            ...other.git.require_pr_branches,
          ]),
        ].sort(),
      },
      quotas: {
        max_execution_time: minOpt(
          this.quotas.max_execution_time,
          other.quotas.max_execution_time,
        ),
        max_memory: minOpt(this.quotas.max_memory, other.quotas.max_memory),
        max_tokens: minOpt(this.quotas.max_tokens, other.quotas.max_tokens),
        max_tool_calls: minOpt(
          this.quotas.max_tool_calls,
          other.quotas.max_tool_calls,
        ),
        max_files_modified: minOpt(
          this.quotas.max_files_modified,
          other.quotas.max_files_modified,
        ),
      },
    });
  }
}

// ── Capability Profiles ─────────────────────────────────────────────

/**
 * Capability profile names.
 *
 * Rust equivalent: `CapabilityProfile` enum
 */
export type CapabilityProfile =
  | "read_only"
  | "standard_dev"
  | "full_access"
  | "custom";

/**
 * Parse a string into a CapabilityProfile.
 *
 * Rust equivalent: `CapabilityProfile::parse()`
 */
export function parseCapabilityProfile(
  s: string,
): CapabilityProfile | undefined {
  switch (s.toLowerCase().replace(/[-_]/g, "")) {
    case "readonly":
      return "read_only";
    case "standarddev":
    case "standard":
      return "standard_dev";
    case "fullaccess":
    case "full":
      return "full_access";
    case "custom":
      return "custom";
    default:
      return undefined;
  }
}

// ── Helpers ─────────────────────────────────────────────────────────

/** Simple domain matching with wildcard support. Rust equivalent of `AgentCapabilities::domain_matches()` */
function domainMatches(pattern: string, domain: string): boolean {
  if (pattern.startsWith("*.")) {
    const suffix = pattern.slice(1); // keep the dot
    return domain.endsWith(suffix) || domain === pattern.slice(2);
  }
  return pattern === domain;
}
