/**
 * PII redaction helpers.
 *
 * Narrower than the Rust crate — we port the two most-used helpers:
 * `hashSessionId` for consistent audit keys, and `redactPayload` for removing
 * obvious secrets from logs.
 *
 * Equivalent to selected parts of `brainwires_telemetry::pii` (the heavier
 * PII tools — email / phone / SSN detectors — live in Rust until an explicit
 * request is made to port them).
 */

/** Deterministic, irreversible hash of a session id (lower 12 hex chars). */
export async function hashSessionId(session_id: string): Promise<string> {
  const bytes = new TextEncoder().encode(`brainwires-session:${session_id}`);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  let hex = "";
  for (let i = 0; i < 6; i++) {
    const b = digest[i];
    hex += b.toString(16).padStart(2, "0");
  }
  return hex;
}

/** Strip common secret patterns from a free-form string. */
export function redactSecrets(text: string): string {
  return text
    .replace(/\b(sk|pk|api|bearer)[-_]?[A-Za-z0-9]{16,}/gi, "***REDACTED***")
    .replace(/\b[A-Za-z0-9+/]{32,}={0,2}\b/g, (m) => (m.length > 40 ? "***REDACTED***" : m));
}
