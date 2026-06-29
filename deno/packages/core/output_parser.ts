/**
 * Structured output parsing for LLM responses.
 *
 * Provides parsers that extract structured data from raw LLM text output.
 * Supports JSON extraction, regex-based parsing, and JSON array parsing.
 *
 * Equivalent to Rust's `output_parser` module in `rullama-core`.
 *
 * @example
 * ```ts
 * const parser = new JsonOutputParser<{ sentiment: string; score: number }>();
 * const raw = 'Here is my analysis: {"sentiment": "positive", "score": 0.9}';
 * const review = parser.parse(raw);
 * // review.sentiment === "positive"
 * ```
 */

// ---------------------------------------------------------------------------
// OutputParser interface
// ---------------------------------------------------------------------------

/**
 * Interface for parsing structured output from LLM text responses.
 *
 * Equivalent to Rust's `OutputParser` trait.
 */
export interface OutputParser<T> {
  /**
   * Parse the raw LLM response text into structured output.
   *
   * @throws If the text cannot be parsed into the expected structure.
   */
  parse(text: string): T;

  /**
   * Return format instructions to inject into the prompt.
   *
   * These instructions tell the LLM how to format its response so this
   * parser can extract structured data from it.
   */
  formatInstructions(): string;
}

// ---------------------------------------------------------------------------
// JsonOutputParser
// ---------------------------------------------------------------------------

/**
 * Extracts JSON from LLM responses and deserializes into `T`.
 *
 * Handles common LLM quirks:
 * - JSON wrapped in markdown code fences
 * - JSON embedded in surrounding prose
 * - Leading/trailing whitespace
 *
 * Equivalent to Rust's `JsonOutputParser<T>`.
 */
export class JsonOutputParser<T> implements OutputParser<T> {
  /**
   * Parse the raw LLM response text, extracting and deserializing JSON.
   *
   * @throws If no valid JSON is found or deserialization fails.
   */
  parse(text: string): T {
    const jsonStr = extractJson(text);
    if (jsonStr === undefined) {
      throw new Error("No JSON found in LLM response");
    }
    try {
      return JSON.parse(jsonStr) as T;
    } catch (e) {
      throw new Error(
        `Failed to parse JSON from LLM response: ${(e as Error).message}`,
      );
    }
  }

  /** Returns instructions telling the LLM to respond with valid JSON only. */
  formatInstructions(): string {
    return "Respond with valid JSON only. Do not include any other text before or after the JSON.";
  }
}

// ---------------------------------------------------------------------------
// JsonListParser
// ---------------------------------------------------------------------------

/**
 * Extracts a list of items from a JSON array in the LLM response.
 *
 * Equivalent to Rust's `JsonListParser<T>`.
 */
export class JsonListParser<T> implements OutputParser<T[]> {
  /**
   * Parse the raw LLM response text, extracting and deserializing a JSON array.
   *
   * @throws If no valid JSON array is found or deserialization fails.
   */
  parse(text: string): T[] {
    const jsonStr = extractJson(text);
    if (jsonStr === undefined) {
      throw new Error("No JSON array found in LLM response");
    }
    try {
      const parsed = JSON.parse(jsonStr);
      if (!Array.isArray(parsed)) {
        throw new Error("Parsed JSON is not an array");
      }
      return parsed as T[];
    } catch (e) {
      throw new Error(
        `Failed to parse JSON array from LLM response: ${(e as Error).message}`,
      );
    }
  }

  /** Returns instructions telling the LLM to respond with a valid JSON array. */
  formatInstructions(): string {
    return "Respond with a valid JSON array only. Do not include any other text.";
  }
}

// ---------------------------------------------------------------------------
// RegexOutputParser
// ---------------------------------------------------------------------------

/**
 * Parses LLM output using a regex pattern with named capture groups.
 *
 * Equivalent to Rust's `RegexOutputParser`.
 */
export class RegexOutputParser implements OutputParser<Record<string, string>> {
  readonly #pattern: RegExp;
  readonly #groupNames: string[];

  /**
   * Create a new regex parser.
   *
   * The pattern should use named capture groups like `(?<name>...)`.
   *
   * @param pattern - A regex string or RegExp with named capture groups.
   * @throws If the pattern is an invalid regular expression.
   */
  constructor(pattern: string | RegExp) {
    try {
      this.#pattern = typeof pattern === "string"
        ? new RegExp(pattern)
        : pattern;
    } catch (e) {
      throw new Error(
        `Invalid regex pattern: ${(e as Error).message}`,
      );
    }

    // Extract named group names from the source string
    this.#groupNames = [];
    const groupRe = /\(\?<([^>]+)>/g;
    const source = this.#pattern.source;
    let m: RegExpExecArray | null;
    while ((m = groupRe.exec(source)) !== null) {
      this.#groupNames.push(m[1]);
    }
  }

  /**
   * Parse the raw LLM response text using the configured regex.
   *
   * @returns A record mapping named capture group names to their matched values.
   * @throws If the regex does not match the input text.
   */
  parse(text: string): Record<string, string> {
    const match = this.#pattern.exec(text);
    if (!match) {
      throw new Error("Regex pattern did not match LLM output");
    }

    const result: Record<string, string> = {};
    for (const name of this.#groupNames) {
      const value = match.groups?.[name];
      if (value !== undefined) {
        result[name] = value;
      }
    }
    return result;
  }

  /** Returns format instructions describing the expected pattern. */
  formatInstructions(): string {
    return `Format your response to match this pattern: ${this.#pattern.source}`;
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Extract JSON from text that may contain markdown fences or surrounding prose.
 *
 * Equivalent to Rust's `extract_json()`.
 */
export function extractJson(text: string): string | undefined {
  const trimmed = text.trim();

  // Try direct parse first
  if (
    (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
    (trimmed.startsWith("[") && trimmed.endsWith("]"))
  ) {
    return trimmed;
  }

  // Try markdown code fence: ```json ... ``` or ``` ... ```
  const fenceStart = trimmed.indexOf("```");
  if (fenceStart !== -1) {
    const afterFence = trimmed.slice(fenceStart + 3);
    // Skip optional language tag
    const newlineIdx = afterFence.indexOf("\n");
    const contentStart = newlineIdx !== -1 ? newlineIdx + 1 : 0;
    const content = afterFence.slice(contentStart);
    const endIdx = content.indexOf("```");
    if (endIdx !== -1) {
      const jsonStr = content.slice(0, endIdx).trim();
      if (jsonStr.length > 0) {
        return jsonStr;
      }
    }
  }

  // Try to find first { or [ and match to last } or ]
  const objStart = trimmed.indexOf("{");
  const arrStart = trimmed.indexOf("[");

  let startIdx: number;
  if (objStart !== -1 && arrStart !== -1) {
    startIdx = Math.min(objStart, arrStart);
  } else if (objStart !== -1) {
    startIdx = objStart;
  } else if (arrStart !== -1) {
    startIdx = arrStart;
  } else {
    return undefined;
  }

  const closeChar = trimmed[startIdx] === "{" ? "}" : "]";
  const endIndex = trimmed.lastIndexOf(closeChar);

  if (endIndex > startIdx) {
    return trimmed.slice(startIdx, endIndex + 1);
  }

  return undefined;
}
