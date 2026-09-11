/**
 * PII redaction helpers.
 *
 * Narrower than the Rust crate — we port the two most-used helpers:
 * `hashSessionId` for consistent audit keys, and `redactSecrets` for removing
 * secrets from logs.
 *
 * Equivalent to selected parts of `rullama_telemetry::pii` (the heavier
 * PII tools — email / phone / SSN detectors — live in Rust until an explicit
 * request is made to port them).
 */

/** Deterministic, irreversible hash of a session id (lower 12 hex chars). */
import { redactSecrets as coreRedactSecrets } from "@rullama/core";

export async function hashSessionId(session_id: string): Promise<string> {
  const bytes = new TextEncoder().encode(`rullama-session:${session_id}`);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  let hex = "";
  for (let i = 0; i < 6; i++) {
    const b = digest[i];
    hex += b.toString(16).padStart(2, "0");
  }
  return hex;
}

/**
 * Strip secret patterns (API keys, tokens, JWTs, private keys, `password=…`)
 * from a free-form string. Delegates to `redactSecrets` in `@rullama/core`, the
 * same table `@rullama/tool-runtime` uses for tool output — so a log line and a
 * tool result redact identically.
 */
export function redactSecrets(text: string): string {
  return coreRedactSecrets(text);
}
