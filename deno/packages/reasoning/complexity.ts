/**
 * Complexity scoring — provider-backed 0.0–1.0 assessment with heuristic fallback.
 *
 * Equivalent to Rust's `brainwires_reasoning::complexity` module.
 */

import { ChatOptions, Message, type Provider } from "@brainwires/core";

/** Result of complexity scoring. */
export interface ComplexityResult {
  /** 0.0 = trivial, 1.0 = very complex. */
  score: number;
  /** 0.0 – 1.0. */
  confidence: number;
  /** Whether an LLM was used (vs the heuristic fallback). */
  used_local_llm: boolean;
}

/** Neutral default: medium complexity, low confidence, no LLM. */
export function defaultComplexity(): ComplexityResult {
  return { score: 0.5, confidence: 0.3, used_local_llm: false };
}

/** Build a result from LLM output, clamping score and confidence. */
export function complexityFromLocal(score: number, confidence: number): ComplexityResult {
  return {
    score: clamp01(score),
    confidence: clamp01(confidence),
    used_local_llm: true,
  };
}

function clamp01(v: number): number {
  if (Number.isNaN(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

const SYSTEM_PROMPT = `You are a task complexity evaluator. Given a task description, output a complexity score.

Scoring guide:
- 0.0-0.2: Trivial (single step, no decisions)
- 0.2-0.4: Simple (few steps, straightforward)
- 0.4-0.6: Moderate (multiple steps, some decisions)
- 0.6-0.8: Complex (many steps, careful reasoning needed)
- 0.8-1.0: Very complex (intricate logic, multiple dependencies)

Consider:
- Number of steps or operations needed
- Required reasoning depth
- Ambiguity in requirements
- Dependencies between parts
- Potential for errors

Output ONLY a decimal number between 0.0 and 1.0.`;

const COMPLEX_KEYWORDS: Array<[string, number]> = [
  ["multiple", 0.1],
  ["several", 0.1],
  ["complex", 0.15],
  ["difficult", 0.15],
  ["careful", 0.1],
  ["ensure", 0.05],
  ["validate", 0.1],
  ["analyze", 0.1],
  ["refactor", 0.15],
  ["architecture", 0.2],
  ["design", 0.1],
  ["optimize", 0.15],
  ["performance", 0.1],
  ["security", 0.15],
  ["concurrent", 0.2],
  ["async", 0.1],
  ["parallel", 0.15],
  ["distributed", 0.2],
];

const SIMPLE_KEYWORDS: Array<[string, number]> = [
  ["simple", -0.1],
  ["trivial", -0.15],
  ["just", -0.05],
  ["only", -0.05],
  ["basic", -0.1],
  ["single", -0.05],
  ["one", -0.05],
  ["quick", -0.1],
  ["easy", -0.1],
];

/** Pure heuristic score based on keyword + length signals. */
export function scoreHeuristic(task_description: string): ComplexityResult {
  const lower = task_description.toLowerCase();
  let score = 0.3;

  for (const [kw, delta] of COMPLEX_KEYWORDS) {
    if (lower.includes(kw)) score += delta;
  }
  for (const [kw, delta] of SIMPLE_KEYWORDS) {
    if (lower.includes(kw)) score += delta;
  }

  const words = task_description.split(/\s+/).filter(Boolean).length;
  if (words > 50) score += 0.15;
  else if (words > 30) score += 0.1;
  else if (words < 10) score -= 0.1;

  return { score: clamp01(score), confidence: 0.4, used_local_llm: false };
}

/** Parse an LLM's decimal answer out of arbitrary text. */
export function parseScore(output: string): number | null {
  const trimmed = output.trim();
  const direct = Number.parseFloat(trimmed);
  if (!Number.isNaN(direct)) return clamp01(direct);

  const m = /\d+\.?\d*/.exec(trimmed);
  if (m) {
    const n = Number.parseFloat(m[0]);
    if (!Number.isNaN(n)) return clamp01(n);
  }
  return null;
}

/** Provider-backed complexity scorer with heuristic fallback. */
export class ComplexityScorer {
  readonly provider: Provider;
  readonly model_id: string;

  constructor(provider: Provider, model_id: string) {
    this.provider = provider;
    this.model_id = model_id;
  }

  /** LLM-backed scoring. Returns null on failure so callers can fall back. */
  async score(task_description: string): Promise<ComplexityResult | null> {
    const user = `Rate the complexity of this task from 0.0 (trivial) to 1.0 (very complex). Output ONLY a decimal number.

Task: ${task_description}`;
    const options = ChatOptions.deterministic(10);
    options.setSystem(SYSTEM_PROMPT);
    try {
      const resp = await this.provider.chat([Message.user(user)], undefined, options);
      const text = resp.message.textOrSummary();
      const score = parseScore(text);
      return score === null ? null : complexityFromLocal(score, 0.8);
    } catch {
      return null;
    }
  }

  /** Fast, pure-heuristic variant — no provider call. */
  scoreHeuristic(task_description: string): ComplexityResult {
    return scoreHeuristic(task_description);
  }
}
