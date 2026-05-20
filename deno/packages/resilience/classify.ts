/**
 * Error classification for retry decisions.
 *
 * Providers surface errors as plain `Error` objects. We classify by string
 * matching the error message against common transient-failure signatures —
 * pragmatic and stable enough for the upstream APIs in use.
 *
 * Equivalent to Rust's `brainwires_resilience::classify` module.
 */

/** Coarse error classification used by retry and circuit-breaker logic. */
export type ErrorClass =
  | "rate_limited"
  | "network"
  | "server_5xx"
  | "client_4xx"
  | "auth"
  | "unknown";

/** Whether errors in this class should be retried. */
export function isRetryable(cls: ErrorClass): boolean {
  return cls === "rate_limited" || cls === "network" || cls === "server_5xx";
}

/** Classify an Error by string inspection. */
export function classifyError(err: unknown): ErrorClass {
  const s = errorString(err).toLowerCase();

  if (s.includes("429") || s.includes("rate limit") || s.includes("too many requests")) {
    return "rate_limited";
  }
  if (s.includes(" 401") || s.includes("unauthorized") || s.includes("invalid api key")) {
    return "auth";
  }
  if (
    s.includes(" 500") ||
    s.includes(" 502") ||
    s.includes(" 503") ||
    s.includes(" 504") ||
    s.includes("internal server error") ||
    s.includes("bad gateway") ||
    s.includes("service unavailable") ||
    s.includes("gateway timeout")
  ) {
    return "server_5xx";
  }
  if (
    s.includes("connection reset") ||
    s.includes("connection refused") ||
    s.includes("connection closed") ||
    s.includes("broken pipe") ||
    s.includes("timed out") ||
    s.includes("timeout") ||
    s.includes("dns") ||
    s.includes("tls") ||
    s.includes("io error")
  ) {
    return "network";
  }
  if (
    s.includes(" 400") ||
    s.includes(" 403") ||
    s.includes(" 404") ||
    s.includes("bad request") ||
    s.includes("forbidden") ||
    s.includes("not found")
  ) {
    return "client_4xx";
  }

  return "unknown";
}

/**
 * Parse a `Retry-After` hint out of an error string. Looks for patterns like
 * `retry-after: 30`, `retry after 30s`, or `retry-after: 30 seconds`.
 * Returns milliseconds, or null if no hint can be extracted or the value
 * looks absurd (0 or > 3600 seconds).
 */
export function parseRetryAfter(err: unknown): number | null {
  const s = errorString(err).toLowerCase();
  const idx = (() => {
    const a = s.indexOf("retry-after");
    if (a !== -1) return a;
    return s.indexOf("retry after");
  })();
  if (idx === -1) return null;

  const tail = s.slice(idx);
  let num = "";
  let seenDigit = false;
  for (const c of tail) {
    if (c >= "0" && c <= "9") {
      num += c;
      seenDigit = true;
    } else if (seenDigit) {
      break;
    }
  }
  if (num.length === 0) return null;
  const secs = Number.parseInt(num, 10);
  if (!Number.isFinite(secs) || secs <= 0 || secs > 3600) return null;
  return secs * 1000;
}

function errorString(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}
