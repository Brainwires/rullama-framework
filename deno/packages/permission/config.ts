/**
 * Permission configuration loading.
 *
 * Handles loading and parsing of JSON-based permissions configuration.
 * Simplified from the Rust version (which uses TOML) to use JSON natively.
 *
 * Rust equivalent: `rullama-permissions/src/config.rs`
 * @module
 */

import {
  AgentCapabilities,
  type CapabilityProfile,
  type GitOperation,
  parseCapabilityProfile,
  PathPattern,
  type ToolCategory,
} from "./types.ts";

// ── Config Sections ─────────────────────────────────────────────────

/** Default configuration section. Rust equivalent: `DefaultConfig` */
export interface DefaultConfig {
  profile: string;
}

/** Filesystem configuration section. Rust equivalent: `FilesystemConfig` */
export interface FilesystemConfig {
  read_paths?: string[];
  write_paths?: string[];
  denied_paths?: string[];
  follow_symlinks?: boolean;
  access_hidden?: boolean;
  max_write_size?: string;
  can_delete?: boolean;
  can_create_dirs?: boolean;
}

/** Tools configuration section. Rust equivalent: `ToolsConfig` */
export interface ToolsConfig {
  allowed_categories?: string[];
  denied_tools?: string[];
  always_approve?: string[];
}

/** Network configuration section. Rust equivalent: `NetworkConfig` */
export interface NetworkConfig {
  allowed_domains?: string[];
  denied_domains?: string[];
  allow_all?: boolean;
  rate_limit?: number;
  allow_api_calls?: boolean;
}

/** Spawning configuration section. Rust equivalent: `SpawningConfig` */
export interface SpawningConfig {
  enabled?: boolean;
  max_children?: number;
  max_depth?: number;
  can_elevate?: boolean;
}

/** Git configuration section. Rust equivalent: `GitConfig` */
export interface GitConfig {
  allowed_ops?: string[];
  protected_branches?: string[];
  can_force_push?: boolean;
  can_destructive?: boolean;
  require_pr_branches?: string[];
}

/** Quotas configuration section. Rust equivalent: `QuotasConfig` */
export interface QuotasConfig {
  max_execution_time?: string;
  max_tool_calls?: number;
  max_files_modified?: number;
  max_tokens?: number;
}

/** Individual policy rule configuration. Rust equivalent: `PolicyRuleConfig` */
export interface PolicyRuleConfig {
  name: string;
  priority?: number;
  conditions?: PolicyConditionConfig[];
  action: string;
  enforcement?: string;
}

/** Policy condition configuration. Rust equivalent: `PolicyConditionConfig` */
export interface PolicyConditionConfig {
  tool?: string;
  tool_category?: string;
  file_path?: string;
  domain?: string;
  git_op?: string;
  min_trust_level?: number;
}

/** Policies configuration section. Rust equivalent: `PoliciesConfig` */
export interface PoliciesConfig {
  rules?: PolicyRuleConfig[];
}

// ── Root Config ─────────────────────────────────────────────────────

/**
 * Root configuration structure for permissions.
 *
 * Rust equivalent: `PermissionsConfig` struct
 */
export interface PermissionsConfig {
  default?: DefaultConfig;
  filesystem?: FilesystemConfig;
  tools?: ToolsConfig;
  network?: NetworkConfig;
  spawning?: SpawningConfig;
  git?: GitConfig;
  quotas?: QuotasConfig;
  policies?: PoliciesConfig;
}

/**
 * Load configuration from a JSON file.
 *
 * Rust equivalent: `PermissionsConfig::load()` (uses TOML in Rust)
 */
export function loadPermissionsConfig(path: string): PermissionsConfig {
  const content = Deno.readTextFileSync(path);
  return JSON.parse(content) as PermissionsConfig;
}

/**
 * Convert config to AgentCapabilities.
 *
 * Rust equivalent: `PermissionsConfig::to_capabilities()`
 */
export function configToCapabilities(
  config: PermissionsConfig,
): AgentCapabilities {
  const profileStr = config.default?.profile ?? "standard_dev";
  const profile: CapabilityProfile = parseCapabilityProfile(profileStr) ??
    "standard_dev";
  const caps = AgentCapabilities.fromProfile(profile);

  // Apply filesystem overrides
  if (config.filesystem?.read_paths) {
    caps.filesystem.read_paths = config.filesystem.read_paths.map((p) =>
      new PathPattern(p)
    );
  }
  if (config.filesystem?.write_paths) {
    caps.filesystem.write_paths = config.filesystem.write_paths.map((p) =>
      new PathPattern(p)
    );
  }
  if (config.filesystem?.denied_paths) {
    caps.filesystem.denied_paths = config.filesystem.denied_paths.map((p) =>
      new PathPattern(p)
    );
  }
  if (config.filesystem?.follow_symlinks !== undefined) {
    caps.filesystem.follow_symlinks = config.filesystem.follow_symlinks;
  }
  if (config.filesystem?.access_hidden !== undefined) {
    caps.filesystem.access_hidden = config.filesystem.access_hidden;
  }
  if (config.filesystem?.max_write_size !== undefined) {
    caps.filesystem.max_write_size = parseSize(
      config.filesystem.max_write_size,
    );
  }
  if (config.filesystem?.can_delete !== undefined) {
    caps.filesystem.can_delete = config.filesystem.can_delete;
  }
  if (config.filesystem?.can_create_dirs !== undefined) {
    caps.filesystem.can_create_dirs = config.filesystem.can_create_dirs;
  }

  // Apply tools overrides
  if (config.tools?.allowed_categories) {
    caps.tools.allowed_categories = new Set(
      config.tools.allowed_categories.filter(isToolCategory) as ToolCategory[],
    );
  }
  if (config.tools?.denied_tools) {
    caps.tools.denied_tools = new Set(config.tools.denied_tools);
  }
  if (config.tools?.always_approve) {
    caps.tools.always_approve = new Set(config.tools.always_approve);
  }

  // Apply network overrides
  if (config.network?.allowed_domains) {
    caps.network.allowed_domains = config.network.allowed_domains;
  }
  if (config.network?.denied_domains) {
    caps.network.denied_domains = config.network.denied_domains;
  }
  if (config.network?.allow_all !== undefined) {
    caps.network.allow_all = config.network.allow_all;
  }
  if (config.network?.rate_limit !== undefined) {
    caps.network.rate_limit = config.network.rate_limit;
  }
  if (config.network?.allow_api_calls !== undefined) {
    caps.network.allow_api_calls = config.network.allow_api_calls;
  }

  // Apply spawning overrides
  if (config.spawning?.enabled !== undefined) {
    caps.spawning.can_spawn = config.spawning.enabled;
  }
  if (config.spawning?.max_children !== undefined) {
    caps.spawning.max_children = config.spawning.max_children;
  }
  if (config.spawning?.max_depth !== undefined) {
    caps.spawning.max_depth = config.spawning.max_depth;
  }
  if (config.spawning?.can_elevate !== undefined) {
    caps.spawning.can_elevate = config.spawning.can_elevate;
  }

  // Apply git overrides
  if (config.git?.allowed_ops) {
    caps.git.allowed_ops = new Set(
      config.git.allowed_ops
        .map(parseGitOperation)
        .filter((op): op is GitOperation => op !== undefined),
    );
  }
  if (config.git?.protected_branches) {
    caps.git.protected_branches = config.git.protected_branches;
  }
  if (config.git?.can_force_push !== undefined) {
    caps.git.can_force_push = config.git.can_force_push;
  }
  if (config.git?.can_destructive !== undefined) {
    caps.git.can_destructive = config.git.can_destructive;
  }
  if (config.git?.require_pr_branches) {
    caps.git.require_pr_branches = config.git.require_pr_branches;
  }

  // Apply quota overrides
  if (config.quotas?.max_execution_time !== undefined) {
    caps.quotas.max_execution_time = parseDuration(
      config.quotas.max_execution_time,
    );
  }
  if (config.quotas?.max_tool_calls !== undefined) {
    caps.quotas.max_tool_calls = config.quotas.max_tool_calls;
  }
  if (config.quotas?.max_files_modified !== undefined) {
    caps.quotas.max_files_modified = config.quotas.max_files_modified;
  }
  if (config.quotas?.max_tokens !== undefined) {
    caps.quotas.max_tokens = config.quotas.max_tokens;
  }

  return caps;
}

// ── Parsing Helpers ─────────────────────────────────────────────────

/**
 * Parse a size string like "1MB" or "512KB" into bytes.
 *
 * Rust equivalent: `parse_size()`
 */
export function parseSize(s: string): number | undefined {
  const upper = s.trim().toUpperCase();
  let num: string;
  let unit: number;

  if (upper.endsWith("GB")) {
    num = upper.slice(0, -2).trim();
    unit = 1024 * 1024 * 1024;
  } else if (upper.endsWith("MB")) {
    num = upper.slice(0, -2).trim();
    unit = 1024 * 1024;
  } else if (upper.endsWith("KB")) {
    num = upper.slice(0, -2).trim();
    unit = 1024;
  } else if (upper.endsWith("B")) {
    num = upper.slice(0, -1).trim();
    unit = 1;
  } else {
    num = upper;
    unit = 1;
  }

  const parsed = parseInt(num, 10);
  return isNaN(parsed) ? undefined : parsed * unit;
}

/**
 * Parse a duration string like "30m" or "1h" into seconds.
 *
 * Rust equivalent: `parse_duration()`
 */
export function parseDuration(s: string): number | undefined {
  const lower = s.trim().toLowerCase();
  let num: string;
  let unit: number;

  if (lower.endsWith("h")) {
    num = lower.slice(0, -1).trim();
    unit = 3600;
  } else if (lower.endsWith("m")) {
    num = lower.slice(0, -1).trim();
    unit = 60;
  } else if (lower.endsWith("s")) {
    num = lower.slice(0, -1).trim();
    unit = 1;
  } else {
    num = lower;
    unit = 1;
  }

  const parsed = parseInt(num, 10);
  return isNaN(parsed) ? undefined : parsed * unit;
}

/** Parse a tool category string. Rust equivalent: `parse_tool_category()` */
export function parseToolCategory(s: string): ToolCategory | undefined {
  const normalized = s.toLowerCase().replace(/_/g, "");
  const map: Record<string, ToolCategory> = {
    fileread: "FileRead",
    filewrite: "FileWrite",
    search: "Search",
    git: "Git",
    gitdestructive: "GitDestructive",
    bash: "Bash",
    web: "Web",
    codeexecution: "CodeExecution",
    agentspawn: "AgentSpawn",
    planning: "Planning",
    system: "System",
  };
  return map[normalized];
}

function isToolCategory(s: string): boolean {
  return parseToolCategory(s) !== undefined;
}

/** Parse a git operation string. Rust equivalent: `parse_git_operation()` */
export function parseGitOperation(s: string): GitOperation | undefined {
  const lower = s.toLowerCase();
  const map: Record<string, GitOperation> = {
    status: "Status",
    diff: "Diff",
    log: "Log",
    add: "Add",
    commit: "Commit",
    push: "Push",
    pull: "Pull",
    fetch: "Fetch",
    branch: "Branch",
    checkout: "Checkout",
    merge: "Merge",
    rebase: "Rebase",
    reset: "Reset",
    stash: "Stash",
    tag: "Tag",
    forcepush: "ForcePush",
    force_push: "ForcePush",
    "force-push": "ForcePush",
  };
  return map[lower];
}
