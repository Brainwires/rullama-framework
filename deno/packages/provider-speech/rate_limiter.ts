/**
 * Token-bucket rate limiter for API request throttling.
 *
 * Provides a simple rate limiter that enforces a maximum number of
 * requests per minute using the token-bucket algorithm.
 *
 * Equivalent to Rust's `RateLimiter` in `rullama-providers`.
 */

// ---------------------------------------------------------------------------
// RateLimiter
// ---------------------------------------------------------------------------

/**
 * A token-bucket rate limiter.
 *
 * Tokens are refilled at a fixed rate. Each request consumes one token.
 * When no tokens are available, `acquire()` waits until one is refilled.
 */
export class RateLimiter {
  #tokens: number;
  #maxTokens: number;
  #refillIntervalMs: number;
  #lastRefill: number;

  /**
   * Create a new rate limiter with the given requests-per-minute limit.
   *
   * A limit of 0 means no requests are allowed.
   */
  constructor(requestsPerMinute: number) {
    this.#maxTokens = requestsPerMinute;
    this.#tokens = requestsPerMinute;
    this.#lastRefill = Date.now();

    if (requestsPerMinute > 0) {
      this.#refillIntervalMs = 60_000 / requestsPerMinute;
    } else {
      // Effectively infinite wait
      this.#refillIntervalMs = Number.MAX_SAFE_INTEGER;
    }
  }

  /**
   * Wait until a token is available, then consume it.
   *
   * Returns a promise that resolves when a token has been acquired.
   */
  async acquire(): Promise<void> {
    while (true) {
      // Try to consume a token
      if (this.#tokens > 0) {
        this.#tokens -= 1;
        return;
      }

      // No tokens available — refill and wait
      this.#refill();

      // If still no tokens, sleep for one refill interval
      if (this.#tokens === 0) {
        await sleep(this.#refillIntervalMs);
        this.#refill();
      }
    }
  }

  /**
   * Try to acquire a token without waiting.
   *
   * Returns `true` if a token was consumed, `false` if none were available.
   */
  tryAcquire(): boolean {
    this.#refill();
    if (this.#tokens > 0) {
      this.#tokens -= 1;
      return true;
    }
    return false;
  }

  /** Refill tokens based on elapsed time since last refill. */
  #refill(): void {
    const now = Date.now();
    const elapsed = now - this.#lastRefill;
    const maxInterval = Math.max(this.#refillIntervalMs, 1);
    const newTokens = Math.floor(elapsed / maxInterval);

    if (newTokens > 0) {
      this.#tokens = Math.min(this.#tokens + newTokens, this.#maxTokens);
      this.#lastRefill = now;
    }
  }

  /** Get the current number of available tokens (for diagnostics). */
  availableTokens(): number {
    return this.#tokens;
  }

  /** Get the configured requests-per-minute limit. */
  maxRequestsPerMinute(): number {
    return this.#maxTokens;
  }
}

// ---------------------------------------------------------------------------
// RateLimitedClient
// ---------------------------------------------------------------------------

/**
 * Options for creating a {@link RateLimitedClient}.
 */
export interface RateLimitedClientOptions {
  /** Maximum requests per minute. */
  requestsPerMinute: number;
}

/**
 * Wraps an async function with rate limiting.
 *
 * Every call to `execute()` acquires a token from the internal
 * {@link RateLimiter} before invoking the wrapped function.
 */
export class RateLimitedClient<TArgs extends unknown[], TResult> {
  readonly limiter: RateLimiter;
  readonly #fn: (...args: TArgs) => Promise<TResult>;

  /**
   * Create a rate-limited wrapper around `fn`.
   *
   * @param fn - The async function to rate-limit.
   * @param options - Rate limiting configuration.
   */
  constructor(
    fn: (...args: TArgs) => Promise<TResult>,
    options: RateLimitedClientOptions,
  ) {
    this.limiter = new RateLimiter(options.requestsPerMinute);
    this.#fn = fn;
  }

  /**
   * Acquire a token, then call the wrapped function.
   */
  async execute(...args: TArgs): Promise<TResult> {
    await this.limiter.acquire();
    return this.#fn(...args);
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
