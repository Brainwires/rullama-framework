/** Trust level / origin of content injected into an agent's context.
 * Equivalent to Rust's `ContentSource` in rullama-core. */
export type ContentSource =
  | "system_prompt"
  | "user_input"
  | "agent_reasoning"
  | "external_content";

/** Trust level numeric values (lower = more trusted). */
const TRUST_ORDER: Record<ContentSource, number> = {
  system_prompt: 0,
  user_input: 1,
  agent_reasoning: 2,
  external_content: 3,
};

/** Returns true for sources that must be sanitized before injection. */
export function requiresSanitization(source: ContentSource): boolean {
  return source === "external_content";
}

/** Returns true if `source` is allowed to override `other`. */
export function canOverride(
  source: ContentSource,
  other: ContentSource,
): boolean {
  return TRUST_ORDER[source] < TRUST_ORDER[other];
}
