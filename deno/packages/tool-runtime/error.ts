/**
 * Tool Error Taxonomy and Classification
 *
 * Based on AgentDebug paper (arxiv:2509.25370) - provides error classification
 * for intelligent retry strategies.
 */

const DEFAULT_MAX_RETRY_ATTEMPTS = 3;
const EXPONENTIAL_BACKOFF_BASE_MS = 500;

/** Resource types for Resource errors. */
export type ResourceType =
  | "FileNotFound"
  | "DirectoryNotFound"
  | "DiskSpace"
  | "Memory"
  | "ProcessLimit"
  | { Other: string };

/** Retry strategy for transient errors. */
export type RetryStrategy =
  | { type: "NoRetry" }
  | { type: "Immediate"; maxAttempts: number }
  | { type: "FixedDelay"; delayMs: number; maxAttempts: number }
  | { type: "ExponentialBackoff"; baseMs: number; maxAttempts: number };

/** Compute the delay (ms) for a given retry attempt, or undefined if exhausted. */
export function delayForAttempt(
  strategy: RetryStrategy,
  attempt: number,
): number | undefined {
  switch (strategy.type) {
    case "NoRetry":
      return undefined;
    case "Immediate":
      return attempt < strategy.maxAttempts ? 0 : undefined;
    case "FixedDelay":
      return attempt < strategy.maxAttempts ? strategy.delayMs : undefined;
    case "ExponentialBackoff":
      return attempt < strategy.maxAttempts
        ? strategy.baseMs * Math.pow(2, attempt)
        : undefined;
  }
}

/** Return the maximum number of retry attempts for a strategy. */
export function maxAttempts(strategy: RetryStrategy): number {
  switch (strategy.type) {
    case "NoRetry":
      return 0;
    case "Immediate":
      return strategy.maxAttempts;
    case "FixedDelay":
      return strategy.maxAttempts;
    case "ExponentialBackoff":
      return strategy.maxAttempts;
  }
}

/** Default retry strategy. */
export function defaultRetryStrategy(): RetryStrategy {
  return {
    type: "ExponentialBackoff",
    baseMs: EXPONENTIAL_BACKOFF_BASE_MS,
    maxAttempts: DEFAULT_MAX_RETRY_ATTEMPTS,
  };
}

/** Error taxonomy based on AgentDebug paper. */
export type ToolErrorCategory =
  | {
    type: "Transient";
    error: string;
    retryStrategy: RetryStrategy;
  }
  | {
    type: "InputValidation";
    error: string;
    suggestion?: string;
  }
  | {
    type: "ExternalService";
    error: string;
    service: string;
    retryAfterMs?: number;
  }
  | {
    type: "Permission";
    error: string;
    requiredPermission: string;
  }
  | {
    type: "Logic";
    error: string;
    context: string;
  }
  | {
    type: "Resource";
    error: string;
    resourceType: ResourceType;
  }
  | {
    type: "Unknown";
    error: string;
  };

/** Return the category name as a string. */
export function categoryName(cat: ToolErrorCategory): string {
  return cat.type.toLowerCase();
}

/** Return the error message string. */
export function errorMessage(cat: ToolErrorCategory): string {
  return cat.error;
}

/** Whether this error category is retryable. */
export function isRetryable(cat: ToolErrorCategory): boolean {
  return cat.type === "Transient" || cat.type === "ExternalService";
}

/** Return the retry strategy for this error. */
export function retryStrategy(cat: ToolErrorCategory): RetryStrategy {
  switch (cat.type) {
    case "Transient":
      return cat.retryStrategy;
    case "ExternalService":
      if (cat.retryAfterMs !== undefined) {
        return {
          type: "FixedDelay",
          delayMs: cat.retryAfterMs,
          maxAttempts: DEFAULT_MAX_RETRY_ATTEMPTS,
        };
      }
      return {
        type: "ExponentialBackoff",
        baseMs: 2000,
        maxAttempts: DEFAULT_MAX_RETRY_ATTEMPTS,
      };
    default:
      return { type: "NoRetry" };
  }
}

/** Get a suggestion for resolving this error, if available. */
export function getSuggestion(cat: ToolErrorCategory): string | undefined {
  switch (cat.type) {
    case "InputValidation":
      return cat.suggestion;
    case "Permission":
      return `Requires ${cat.requiredPermission} permission`;
    case "Resource":
      return `Resource issue: ${
        typeof cat.resourceType === "string"
          ? cat.resourceType
          : cat.resourceType.Other
      }`;
    default:
      return undefined;
  }
}

// ---- Error classification patterns ----

interface ErrorPattern {
  keywords: string[];
  builder: (error: string) => ToolErrorCategory;
}

const ERROR_PATTERNS: ErrorPattern[] = [
  {
    keywords: [
      "connection refused",
      "connection reset",
      "connection timed out",
    ],
    builder: (e) => ({
      type: "Transient",
      error: e,
      retryStrategy: {
        type: "ExponentialBackoff",
        baseMs: 1000,
        maxAttempts: 3,
      },
    }),
  },
  {
    keywords: ["timeout", "timed out", "deadline exceeded"],
    builder: (e) => ({
      type: "Transient",
      error: e,
      retryStrategy: {
        type: "ExponentialBackoff",
        baseMs: 2000,
        maxAttempts: 3,
      },
    }),
  },
  {
    keywords: ["network", "dns", "host unreachable", "no route"],
    builder: (e) => ({
      type: "Transient",
      error: e,
      retryStrategy: {
        type: "ExponentialBackoff",
        baseMs: 1000,
        maxAttempts: 3,
      },
    }),
  },
  {
    keywords: ["rate limit", "too many requests", "429", "quota exceeded"],
    builder: (e) => ({
      type: "ExternalService",
      error: e,
      service: "API",
      retryAfterMs: 5000,
    }),
  },
  {
    keywords: ["service unavailable", "503", "502", "bad gateway"],
    builder: (e) => ({
      type: "ExternalService",
      error: e,
      service: "external",
      retryAfterMs: 3000,
    }),
  },
  {
    keywords: ["internal server error", "500"],
    builder: (e) => ({
      type: "ExternalService",
      error: e,
      service: "external",
      retryAfterMs: 2000,
    }),
  },
  {
    keywords: ["permission denied", "access denied", "forbidden", "403"],
    builder: (e) => ({
      type: "Permission",
      error: e,
      requiredPermission: "access",
    }),
  },
  {
    keywords: ["unauthorized", "401", "authentication"],
    builder: (e) => ({
      type: "Permission",
      error: e,
      requiredPermission: "authentication",
    }),
  },
  {
    keywords: ["read-only", "cannot write", "not writable"],
    builder: (e) => ({
      type: "Permission",
      error: e,
      requiredPermission: "write",
    }),
  },
  {
    keywords: [
      "no such file",
      "file not found",
      "cannot find",
      "does not exist",
    ],
    builder: (e) => ({
      type: "Resource",
      error: e,
      resourceType: "FileNotFound",
    }),
  },
  {
    keywords: ["not a directory", "is a directory", "directory not found"],
    builder: (e) => ({
      type: "Resource",
      error: e,
      resourceType: "DirectoryNotFound",
    }),
  },
  {
    keywords: ["no space left", "disk full", "quota"],
    builder: (e) => ({
      type: "Resource",
      error: e,
      resourceType: "DiskSpace",
    }),
  },
  {
    keywords: ["out of memory", "cannot allocate", "memory"],
    builder: (e) => ({
      type: "Resource",
      error: e,
      resourceType: "Memory",
    }),
  },
  {
    keywords: ["invalid argument", "invalid parameter", "invalid input"],
    builder: (e) => ({
      type: "InputValidation",
      error: e,
      suggestion: "Check the input parameters",
    }),
  },
  {
    keywords: ["missing required", "required field", "missing argument"],
    builder: (e) => ({
      type: "InputValidation",
      error: e,
      suggestion: "Provide all required parameters",
    }),
  },
  {
    keywords: ["invalid path", "bad path", "malformed"],
    builder: (e) => ({
      type: "InputValidation",
      error: e,
      suggestion: "Check the path format",
    }),
  },
  {
    keywords: ["type error", "expected", "invalid type"],
    builder: (e) => ({
      type: "InputValidation",
      error: e,
      suggestion: "Check parameter types",
    }),
  },
];

function classifyBashError(error: string): ToolErrorCategory {
  const lower = error.toLowerCase();
  if (lower.includes("command not found")) {
    return {
      type: "InputValidation",
      error,
      suggestion:
        "Command does not exist. Check spelling or install the program.",
    };
  }
  if (lower.includes("exit code") || lower.includes("failed with")) {
    return { type: "Logic", error, context: "bash_execution" };
  }
  return { type: "Unknown", error };
}

function classifyFileError(error: string): ToolErrorCategory {
  const lower = error.toLowerCase();
  if (lower.includes("binary") || lower.includes("not valid utf-8")) {
    return {
      type: "InputValidation",
      error,
      suggestion: "File is binary or not valid text.",
    };
  }
  if (lower.includes("too large")) {
    return { type: "Resource", error, resourceType: "Memory" };
  }
  return { type: "Unknown", error };
}

function classifyWebError(error: string): ToolErrorCategory {
  const lower = error.toLowerCase();
  if (lower.includes("ssl") || lower.includes("certificate")) {
    return { type: "ExternalService", error, service: "SSL/TLS" };
  }
  if (lower.includes("redirect")) {
    return {
      type: "InputValidation",
      error,
      suggestion: "URL redirected. Follow the redirect or use the new URL.",
    };
  }
  return { type: "Unknown", error };
}

/** Classify an error from a tool result. */
export function classifyError(
  toolName: string,
  error: string,
): ToolErrorCategory {
  const errorLower = error.toLowerCase();

  for (const pattern of ERROR_PATTERNS) {
    if (pattern.keywords.some((kw) => errorLower.includes(kw))) {
      return pattern.builder(error);
    }
  }

  switch (toolName) {
    case "bash":
    case "Bash":
    case "execute_command":
      return classifyBashError(error);
    case "read_file":
    case "ReadFile":
    case "Read":
    case "write_file":
    case "WriteFile":
    case "Write":
      return classifyFileError(error);
    case "web_search":
    case "WebSearch":
    case "web_fetch":
    case "WebFetch":
    case "fetch_url":
      return classifyWebError(error);
    default:
      return { type: "Unknown", error };
  }
}

/** Outcome of a tool execution (for SEAL learning). */
export interface ToolOutcome {
  /** Name of the tool that ran. */
  toolName: string;
  /** Whether the final attempt succeeded. */
  success: boolean;
  /** Number of retries before the outcome. */
  retries: number;
  /** Category of the final error, if it failed. */
  errorCategory?: ToolErrorCategory;
  /** Total execution time in milliseconds. */
  executionTimeMs: number;
}

/** Create a successful tool outcome. */
export function successOutcome(
  toolName: string,
  retries: number,
  executionTimeMs: number,
): ToolOutcome {
  return { toolName, success: true, retries, executionTimeMs };
}

/** Create a failed tool outcome. */
export function failureOutcome(
  toolName: string,
  retries: number,
  errorCategory: ToolErrorCategory,
  executionTimeMs: number,
): ToolOutcome {
  return {
    toolName,
    success: false,
    retries,
    errorCategory,
    executionTimeMs,
  };
}
