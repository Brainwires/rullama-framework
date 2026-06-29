/**
 * @module @rullama/resilience
 *
 * Provider-layer resilience middleware for the Brainwires Agent Framework.
 *
 * Wraps any `@rullama/core` {@link Provider} with composable decorators:
 *
 * - {@link RetryProvider} — exponential backoff with jitter.
 * - {@link BudgetProvider} — token/USD/round caps with pre-flight rejection.
 * - {@link CircuitBreakerProvider} — half-open state machine, optional
 *   fallback provider.
 * - {@link CachedProvider} — content-addressed response cache for
 *   deterministic evals and local dev.
 *
 * Typical stacking (outermost first):
 *
 * ```text
 * CircuitBreaker → Retry → Budget → Cache → base Provider
 * ```
 *
 * Equivalent to Rust's `rullama-resilience` crate. The SQLite cache backend
 * from the Rust crate is intentionally omitted here — use any `CacheBackend`
 * implementation (Deno KV, Postgres, etc.) to get persistence.
 */

export {
  approxInputTokens,
  type BudgetConfig,
  BudgetGuard,
  BudgetProvider,
  defaultBudgetConfig,
} from "./budget.ts";
export {
  type CacheBackend,
  CachedProvider,
  type CachedResponse,
  type CacheKey,
  cacheKeyFor,
  MemoryCache,
} from "./cache.ts";
export {
  type CircuitBreakerConfig,
  CircuitBreakerProvider,
  type CircuitState,
  defaultCircuitBreakerConfig,
} from "./circuit.ts";
export {
  classifyError,
  type ErrorClass,
  isRetryable,
  parseRetryAfter,
} from "./classify.ts";
export { ResilienceError, type ResilienceErrorKind } from "./error.ts";
export {
  applyJitter,
  backoffFor,
  defaultRetryPolicy,
  noRetryPolicy,
  type RetryPolicy,
  RetryProvider,
} from "./retry.ts";
