/**
 * Retrieval-need classifier — decides whether a query likely needs earlier
 * conversation context.
 *
 * Equivalent to Rust's `brainwires_reasoning::retrieval_classifier` module.
 */

import { ChatOptions, Message, type Provider } from "@brainwires/core";

export type RetrievalNeed = "none" | "low" | "medium" | "high";

export function shouldRetrieve(need: RetrievalNeed): boolean {
  return need === "medium" || need === "high";
}

export function retrievalScore(need: RetrievalNeed): number {
  switch (need) {
    case "none":
      return 0;
    case "low":
      return 0.25;
    case "medium":
      return 0.6;
    case "high":
      return 0.9;
  }
}

export interface ClassificationResult {
  need: RetrievalNeed;
  confidence: number;
  used_local_llm: boolean;
  intent: string | null;
}

export function classificationFromLocal(
  need: RetrievalNeed,
  confidence: number,
  intent: string | null,
): ClassificationResult {
  return { need, confidence, used_local_llm: true, intent };
}

export function classificationFromFallback(
  need: RetrievalNeed,
  confidence: number,
): ClassificationResult {
  return { need, confidence, used_local_llm: false, intent: null };
}

const REFERENCE = [
  "earlier",
  "before",
  "we discussed",
  "remember when",
  "what was",
  "didn't we",
  "you mentioned",
  "as i said",
  "previously",
  "last time",
  "originally",
  "initially",
  "you said",
  "i said",
  "we talked",
  "back when",
  "recall",
  "mentioned earlier",
  "as mentioned",
];
const QUESTION = ["what did", "when did", "why did", "how did", "where was", "who was"];
const CONTINUATION = [
  "continue",
  "keep going",
  "and then",
  "what about",
  "more about",
  "tell me more",
  "go on",
];

export function classifyHeuristic(query: string, context_len: number): ClassificationResult {
  const lower = query.toLowerCase();
  let score = 0;
  let matches = 0;

  for (const p of REFERENCE) if (lower.includes(p)) { score += 0.4; matches += 1; }
  for (const p of QUESTION) if (lower.includes(p)) { score += 0.25; matches += 1; }
  for (const p of CONTINUATION) if (lower.includes(p)) { score += 0.15; matches += 1; }

  if (context_len < 3) score += 0.3;
  else if (context_len < 5) score += 0.2;
  else if (context_len < 10) score += 0.1;

  if (context_len < 10 && query.length < 100 && lower.includes("?")) {
    const pronouns = ["it", "they", "that", "those", "the one"];
    const words = lower.split(/\s+/);
    if (pronouns.some((p) => words.includes(p))) score += 0.2;
  }

  if (score > 1) score = 1;

  const need: RetrievalNeed = score >= 0.6
    ? "high"
    : score >= 0.35
    ? "medium"
    : score >= 0.15
    ? "low"
    : "none";

  const confidence = matches > 0 ? 0.7 + Math.min(0.2, matches * 0.05) : 0.5;
  return classificationFromFallback(need, confidence);
}

export function parseClassification(output: string): ClassificationResult {
  const trimmed = output.trim();
  const up = trimmed.toUpperCase();
  const colon = trimmed.indexOf(":");
  const intent = colon >= 0 ? trimmed.slice(colon + 1).trim() : null;

  const need: RetrievalNeed = up.startsWith("HIGH") || up.includes("HIGH:")
    ? "high"
    : up.startsWith("MEDIUM") || up.includes("MEDIUM:")
    ? "medium"
    : up.startsWith("LOW") || up.includes("LOW:")
    ? "low"
    : up.startsWith("NONE") || up.includes("NONE:")
    ? "none"
    : "low";

  return classificationFromLocal(need, 0.8, intent);
}

export class RetrievalClassifier {
  readonly provider: Provider;
  readonly model_id: string;

  constructor(provider: Provider, model_id: string) {
    this.provider = provider;
    this.model_id = model_id;
  }

  async classify(query: string, context_len: number): Promise<ClassificationResult | null> {
    const truncated = query.length > 200 ? query.slice(0, 200) : query;
    const prompt = `Classify if this query needs to retrieve earlier conversation context.

Query: "${truncated}"
Recent context messages: ${context_len}

Classify as:
- NONE: Query is self-contained, no prior context needed
- LOW: Might benefit from context but not required
- MEDIUM: Likely references earlier discussion
- HIGH: Definitely refers to prior conversation

Output format: LEVEL: brief reason
Example: HIGH: references "earlier" and asks about past discussion

Classification:`;
    const options = ChatOptions.deterministic(50);
    try {
      const resp = await this.provider.chat([Message.user(prompt)], undefined, options);
      return parseClassification(resp.message.textOrSummary());
    } catch {
      return null;
    }
  }

  classifyHeuristic(query: string, context_len: number): ClassificationResult {
    return classifyHeuristic(query, context_len);
  }
}
