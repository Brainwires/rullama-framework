/**
 * @module @rullama/tool-runtime
 *
 * Tool execution framework for the Brainwires Agent Framework.
 * Equivalent to Rust's `rullama-tool-runtime` crate (post-0.11.0 split).
 *
 * Provides the registry, executor trait, error taxonomy, sanitization,
 * smart routing, transaction manager, plus OpenAPI / OAuth / validation /
 * tool-search / tool-embedding building blocks. Concrete built-in tools
 * (Bash, FileOps, Git, Web, Search, SemanticSearch, Calendar) live in
 * `@rullama/tool-builtins`.
 */

// Error taxonomy and classification
export {
  categoryName,
  classifyError,
  defaultRetryStrategy,
  delayForAttempt,
  errorMessage,
  failureOutcome,
  getSuggestion,
  isRetryable,
  maxAttempts,
  type ResourceType,
  type RetryStrategy,
  retryStrategy,
  successOutcome,
  type ToolErrorCategory,
  type ToolOutcome,
} from "./error.ts";

// Tool executor interface
export {
  allow,
  type PreHookDecision,
  reject,
  type ToolExecutor,
  type ToolPreHook,
} from "./executor.ts";

// Tool registry
export { type ToolCategory, ToolRegistry } from "./registry.ts";

// Sanitization
export {
  containsSensitiveData,
  type ContentSource,
  filterToolOutput,
  isInjectionAttempt,
  redactSensitiveData,
  sanitizeExternalContent,
  wrapWithContentSource,
} from "./sanitization.ts";

// Smart router
export {
  analyzeMessages,
  analyzeQuery,
  getContextForAnalysis,
  getSmartTools,
  getSmartToolsWithMcp,
  getToolsForCategories,
} from "./smart_router.ts";

// Transaction manager
export { TransactionManager } from "./transaction.ts";

// Validation harness (was tools/tools/validation.ts pre-0.11.0)
export {
  extractExportName,
  isExportLine,
  ValidationTool,
} from "./validation.ts";

// OpenAPI tool generation
export {
  executeOpenApiTool,
  executeOpenApiToolWithEndpoint,
  type HttpMethod,
  type OpenApiEndpoint,
  type OpenApiParam,
  type OpenApiToolDef,
  openApiToToolDefs,
  openApiToTools,
} from "./openapi.ts";

// OAuth (PKCE, client-credentials, token refresh, pluggable token store)
export {
  authorizationCodePkceConfig,
  base64UrlEncode,
  clientCredentialsConfig,
  InMemoryTokenStore,
  isTokenExpired,
  newPkceChallenge,
  OAuthClient,
  type OAuthConfig,
  type OAuthFlow,
  type OAuthToken,
  type OAuthTokenStore,
  pkceAuthorizationUrl,
  type PkceChallenge,
} from "./oauth.ts";

// Tool search + embedding
export { cosineSimilarity, ToolEmbeddingIndex } from "./tool_embedding.ts";
export {
  DEFAULT_SEARCH_MODE,
  type SearchMode,
  ToolSearchTool,
} from "./tool_search.ts";
