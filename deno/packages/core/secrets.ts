/**
 * Secret detection and redaction shared by every package that handles text a
 * model may see or a log may keep: API keys, tokens, JWTs, private keys,
 * email addresses and `password=…` style assignments.
 *
 * `@rullama/tool-runtime`'s sanitizer and `@rullama/telemetry`'s PII helpers
 * both delegate here so there is exactly one pattern table to maintain.
 *
 * @module
 */

/** One redaction rule. */
export interface SensitivePattern {
  /** Global regex matching the secret; `lastIndex` is reset before each use. */
  regex: RegExp;
  /** Category name written into the `[REDACTED: <label>]` replacement. */
  label: string;
}

/** The patterns {@link redactSecrets} and {@link containsSecrets} apply, in order. */
export const SENSITIVE_PATTERNS: readonly SensitivePattern[] = [
  // OpenAI-style API keys: sk-..., sk-proj-...
  { regex: /sk-(?:proj-|org-)?[A-Za-z0-9_-]{20,}/g, label: "api-key" },
  // Anthropic API keys
  { regex: /sk-ant-[A-Za-z0-9_-]{20,}/g, label: "api-key" },
  // GitHub personal access tokens / fine-grained PATs
  { regex: /gh[pousr]_[A-Za-z0-9_]{20,}/g, label: "github-token" },
  // GitLab personal access tokens
  { regex: /glpat-[A-Za-z0-9_-]{20,}/g, label: "gitlab-token" },
  // AWS access key IDs
  { regex: /AKIA[0-9A-Z]{16}/g, label: "aws-access-key" },
  // AWS secret access keys (heuristic)
  {
    regex: /(?:aws[_-]?secret[_-]?access[_-]?key)\s*[=:]\s*[A-Za-z0-9/+]{40}/gi,
    label: "aws-secret",
  },
  // Generic Bearer tokens
  { regex: /(?:bearer)\s+[A-Za-z0-9\-._~+/]{20,}=*/gi, label: "bearer-token" },
  // JWTs (three base64url segments)
  {
    regex: /eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/g,
    label: "jwt",
  },
  // Private key PEM blocks
  {
    regex:
      /-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----/g,
    label: "private-key",
  },
  // Email addresses
  {
    regex: /\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b/g,
    label: "email",
  },
  // Generic patterns: password=VALUE or password: VALUE
  {
    regex:
      /(?:password|passwd|secret|credential|api[_-]?key|access[_-]?token)\s*[=:]\s*\S{4,}/gi,
    label: "credential",
  },
];

/** True when `text` matches any {@link SENSITIVE_PATTERNS} entry. */
export function containsSecrets(text: string): boolean {
  return SENSITIVE_PATTERNS.some(({ regex }) => {
    regex.lastIndex = 0;
    return regex.test(text);
  });
}

/** Replace every {@link SENSITIVE_PATTERNS} match with `[REDACTED: <label>]`. */
export function redactSecrets(text: string): string {
  let out = text;
  for (const { regex, label } of SENSITIVE_PATTERNS) {
    regex.lastIndex = 0;
    out = out.replace(regex, `[REDACTED: ${label}]`);
  }
  return out;
}
