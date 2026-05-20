/**
 * Semantic response validator — provider-backed + heuristic fallback.
 *
 * Equivalent to Rust's `brainwires_reasoning::validator` module.
 */

import { ChatOptions, Message, type Provider } from "@brainwires/core";

/** Validation outcome. */
export type ValidationResult =
  | { kind: "valid"; confidence: number }
  | { kind: "invalid"; reason: string; severity: number; confidence: number }
  | { kind: "skipped" };

export function isValid(r: ValidationResult): boolean {
  return r.kind === "valid";
}

export function isInvalid(r: ValidationResult): boolean {
  return r.kind === "invalid";
}

const SYSTEM_PROMPT = `You are a response validator. Given a task and response, determine if the response is appropriate.

Check for:
1. Response addresses the task (not off-topic)
2. Response doesn't contain confusion or self-correction
3. Response isn't a refusal or "I can't do that"
4. Response isn't just repeating the task
5. Response has substance (not empty platitudes)

Output format:
- If valid: VALID
- If invalid: INVALID:<brief reason>

Be strict but fair. Only flag clear issues.`;

const REFUSAL_PATTERNS = [
  "i cannot",
  "i can't",
  "i'm unable",
  "i am unable",
  "sorry, i",
  "i don't have",
  "i do not have",
  "as an ai",
];

/** Parse the LLM verdict ("VALID" / "INVALID:reason" / ambiguous). */
export function parseValidation(output: string): ValidationResult {
  const up = output.trim().toUpperCase();
  if (up.startsWith("VALID") && !up.includes("INVALID")) {
    return { kind: "valid", confidence: 0.8 };
  }
  if (up.startsWith("INVALID")) {
    const idx = up.indexOf(":");
    const reason = idx >= 0 ? up.slice(idx + 1).trim() : "Unspecified validation failure";
    return { kind: "invalid", reason, severity: 0.6, confidence: 0.75 };
  }
  return { kind: "skipped" };
}

/** Pattern-based quick validation — no provider call. */
export function validateHeuristic(task: string, response: string): ValidationResult {
  const task_lower = task.toLowerCase();
  const resp_lower = response.toLowerCase();

  const task_words = new Set(task_lower.split(/\s+/).filter((w) => w.length > 3));
  const resp_words = new Set(resp_lower.split(/\s+/).filter((w) => w.length > 3));
  let overlap = 0;
  for (const w of task_words) if (resp_words.has(w)) overlap += 1;

  if (overlap === 0 && task_words.size > 3) {
    return {
      kind: "invalid",
      reason: "Response appears unrelated to task",
      severity: 0.6,
      confidence: 0.4,
    };
  }

  for (const p of REFUSAL_PATTERNS) {
    if (resp_lower.includes(p)) {
      return {
        kind: "invalid",
        reason: `Response contains refusal pattern: ${p}`,
        severity: 0.7,
        confidence: 0.6,
      };
    }
  }

  const task_trim = task_lower.trim();
  const resp_trim = resp_lower.trim();
  if (resp_trim.startsWith(task_trim) && response.length < task.length * 2) {
    return {
      kind: "invalid",
      reason: "Response appears to just repeat the task",
      severity: 0.5,
      confidence: 0.5,
    };
  }

  if (task.length > 100 && response.length < 20) {
    return {
      kind: "invalid",
      reason: "Response too short for complex task",
      severity: 0.4,
      confidence: 0.4,
    };
  }

  return { kind: "valid", confidence: 0.5 };
}

/** Provider-backed semantic response validator. */
export class LocalValidator {
  readonly provider: Provider;
  readonly model_id: string;

  constructor(provider: Provider, model_id: string) {
    this.provider = provider;
    this.model_id = model_id;
  }

  async validate(task: string, response: string): Promise<ValidationResult> {
    if (response.trim().length < 10) return { kind: "skipped" };

    const truncated = response.length > 500 ? response.slice(0, 500) : response;
    const user = `Validate if this response is appropriate for the task.

Task: ${task}

Response: ${truncated}

Output ONLY: VALID or INVALID:<reason>`;
    const options = ChatOptions.deterministic(50);
    options.setSystem(SYSTEM_PROMPT);
    try {
      const resp = await this.provider.chat([Message.user(user)], undefined, options);
      return parseValidation(resp.message.textOrSummary());
    } catch {
      return { kind: "skipped" };
    }
  }

  validateHeuristic(task: string, response: string): ValidationResult {
    return validateHeuristic(task, response);
  }
}
