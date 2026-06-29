/**
 * Response Confidence Extraction
 *
 * Based on the CISC paper (arxiv:2502.06233v1). Extracts confidence scores
 * from LLM responses using multiple heuristics, for use in decision-making
 * and SEAL learning loops.
 *
 * Equivalent to Rust's `rullama_core::confidence`.
 */

import type { ChatResponse, ContentBlock } from "./message.ts";

/** Individual factors that contribute to a confidence score. */
export interface ConfidenceFactors {
  /** Based on finish_reason (stop = high, truncated = low). */
  completion_confidence: number;
  /** Based on hedging/uncertainty patterns in text. */
  pattern_confidence: number;
  /** Based on response length (normalized). */
  length_confidence: number;
  /** Based on presence of tool use (structured = higher confidence). */
  structure_confidence: number;
}

/** Default-zero factors. */
export function defaultConfidenceFactors(): ConfidenceFactors {
  return {
    completion_confidence: 0,
    pattern_confidence: 0,
    length_confidence: 0,
    structure_confidence: 0,
  };
}

/** Return `(name, value)` for the lowest-scoring factor. */
export function weakestFactor(
  factors: ConfidenceFactors,
): [string, number] {
  const entries: [string, number][] = [
    ["completion", factors.completion_confidence],
    ["pattern", factors.pattern_confidence],
    ["length", factors.length_confidence],
    ["structure", factors.structure_confidence],
  ];
  return entries.reduce(
    (acc, cur) => (cur[1] < acc[1] ? cur : acc),
    ["unknown", 0.5] as [string, number],
  );
}

/** Confidence summary for a chat response. */
export interface ResponseConfidence {
  /** Overall confidence score (0.0–1.0). */
  score: number;
  /** Individual factors that produced the score. */
  factors: ConfidenceFactors;
}

/** Default-zero confidence. */
export function defaultResponseConfidence(): ResponseConfidence {
  return { score: 0, factors: defaultConfidenceFactors() };
}

/** A response is "high confidence" at score ≥ 0.8. */
export function isHighConfidence(c: ResponseConfidence): boolean {
  return c.score >= 0.8;
}

/** A response is "low confidence" below 0.6. */
export function isLowConfidence(c: ResponseConfidence): boolean {
  return c.score < 0.6;
}

/** Human-readable confidence band. */
export function confidenceLevel(c: ResponseConfidence): string {
  if (c.score >= 0.9) return "very_high";
  if (c.score >= 0.8) return "high";
  if (c.score >= 0.6) return "medium";
  if (c.score >= 0.4) return "low";
  return "very_low";
}

const LOW_CONFIDENCE_PATTERNS: readonly string[] = [
  "i'm not sure",
  "i think",
  "possibly",
  "might be",
  "could be",
  "i believe",
  "probably",
  "perhaps",
  "maybe",
  "not certain",
  "unclear",
  "i guess",
  "it seems",
  "apparently",
];

const SELF_CORRECTION_PATTERNS: readonly string[] = [
  "wait,",
  "actually,",
  "let me reconsider",
  "i made a mistake",
  "correction:",
  "i was wrong",
  "on second thought",
  "i need to revise",
  "let me correct",
  "that's not right",
];

const HIGH_CONFIDENCE_PATTERNS: readonly string[] = [
  "the answer is",
  "definitely",
  "certainly",
  "clearly",
  "without doubt",
  "the solution is",
  "this will work",
  "i can confirm",
];

function clamp(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

function getResponseText(response: ChatResponse): string {
  const content = response.message.content;
  if (typeof content === "string") return content;
  return content
    .filter((b): b is ContentBlock & { type: "text" } => b.type === "text")
    .map((b) => b.text)
    .join(" ");
}

function calculateCompletionConfidence(
  finishReason: string | undefined,
): number {
  switch (finishReason) {
    case "stop":
    case "end_turn":
      return 0.95;
    case "tool_use":
      return 0.90;
    case "length":
    case "max_tokens":
      return 0.50;
    case "content_filter":
      return 0.30;
    case undefined:
      return 0.70;
    default:
      return 0.60;
  }
}

function calculatePatternConfidence(text: string): number {
  const lower = text.toLowerCase();
  const lowCount = LOW_CONFIDENCE_PATTERNS.filter((p) => lower.includes(p))
    .length;
  const selfCorrectionCount =
    SELF_CORRECTION_PATTERNS.filter((p) => lower.includes(p)).length;
  const highCount = HIGH_CONFIDENCE_PATTERNS.filter((p) => lower.includes(p))
    .length;

  let confidence = 0.75;
  confidence -= Math.min(lowCount * 0.08, 0.35);
  confidence -= Math.min(selfCorrectionCount * 0.15, 0.30);
  confidence += Math.min(highCount * 0.05, 0.15);
  return clamp(confidence, 0.25, 0.98);
}

function calculateLengthConfidence(text: string): number {
  const tokens = Math.floor(text.length / 4);
  if (tokens < 10) return 0.40;
  if (tokens < 30) return 0.60;
  if (tokens < 50) return 0.75;
  if (tokens <= 500) return 0.90;
  if (tokens <= 1000) return 0.75;
  if (tokens <= 2000) return 0.60;
  return 0.50;
}

function calculateStructureConfidence(response: ChatResponse): number {
  const content = response.message.content;
  if (typeof content === "string") return 0.70;
  const hasToolUse = content.some((b) => b.type === "tool_use");
  return hasToolUse ? 0.90 : 0.75;
}

/**
 * Extract confidence from a chat response using completion/pattern/length/
 * structure heuristics. Weighted average favors pattern + completion.
 */
export function extractConfidence(response: ChatResponse): ResponseConfidence {
  const text = getResponseText(response);
  const completion = calculateCompletionConfidence(response.finish_reason);
  const pattern = calculatePatternConfidence(text);
  const length = calculateLengthConfidence(text);
  const structure = calculateStructureConfidence(response);

  const score = completion * 0.30 + pattern * 0.35 + length * 0.15 +
    structure * 0.20;

  return {
    score: clamp(score, 0, 1),
    factors: {
      completion_confidence: completion,
      pattern_confidence: pattern,
      length_confidence: length,
      structure_confidence: structure,
    },
  };
}

const OBVIOUS_LOW_CONFIDENCE: readonly string[] = [
  "i'm not sure",
  "i don't know",
  "i cannot",
  "i made a mistake",
  "that's not right",
];

/**
 * Fast confidence check without the full analysis pipeline.
 * Returns `false` when the response is obviously low-confidence.
 */
export function quickConfidenceCheck(response: ChatResponse): boolean {
  if (response.finish_reason === "length") return false;
  const lower = getResponseText(response).toLowerCase();
  return !OBVIOUS_LOW_CONFIDENCE.some((p) => lower.includes(p));
}
