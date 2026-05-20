/**
 * Ranking quality metrics for information retrieval evaluation.
 *
 * Pure functions operating on scores and ground-truth relevance labels.
 * No async, no external dependencies — safe to call from any eval case.
 *
 * | Function | What it measures |
 * |----------|-----------------|
 * | {@link ndcgAtK}        | Normalized Discounted Cumulative Gain |
 * | {@link mrr}            | Mean Reciprocal Rank |
 * | {@link precisionAtK}   | Fraction of top-K that are relevant |
 *
 * Equivalent to Rust's `brainwires_agents::eval::ranking_metrics` module.
 */

function zip(
  scores: readonly number[],
  relevance: readonly number[],
): Array<[number, number]> {
  if (scores.length !== relevance.length) {
    throw new Error("scores and relevance must have the same length");
  }
  return scores.map((s, i) => [s, relevance[i]] as [number, number]);
}

/**
 * Compute NDCG@K (Normalized Discounted Cumulative Gain).
 *
 * @param scores system-assigned scores; higher = more relevant per system
 * @param relevance ground-truth relevance labels (0 = irrelevant)
 * @param k cut-off depth; pass `0` to evaluate all items
 */
export function ndcgAtK(
  scores: readonly number[],
  relevance: readonly number[],
  k: number,
): number {
  if (scores.length !== relevance.length) {
    throw new Error("scores and relevance must have the same length");
  }
  if (scores.length === 0) return 0.0;

  const n = scores.length;
  const cut = k === 0 || k > n ? n : k;

  const ranked = zip(scores, relevance);
  ranked.sort((a, b) => b[0] - a[0]);

  const dcg = ranked.slice(0, cut).reduce((acc, [, rel], i) => {
    return acc + (Math.pow(2.0, rel) - 1.0) / Math.log2(i + 2.0);
  }, 0.0);

  const ideal = [...relevance].sort((a, b) => b - a);
  const idcg = ideal.slice(0, cut).reduce((acc, rel, i) => {
    return acc + (Math.pow(2.0, rel) - 1.0) / Math.log2(i + 2.0);
  }, 0.0);

  if (idcg === 0.0) return 0.0;
  const ratio = dcg / idcg;
  return Math.max(0.0, Math.min(1.0, ratio));
}

/**
 * Compute MRR (Mean Reciprocal Rank).
 *
 * Returns the reciprocal of the 1-based rank of the first relevant item.
 * Returns 0.0 if no relevant item exists.
 */
export function mrr(
  scores: readonly number[],
  relevance: readonly number[],
): number {
  if (scores.length !== relevance.length) {
    throw new Error("scores and relevance must have the same length");
  }
  if (scores.length === 0) return 0.0;

  const ranked = zip(scores, relevance);
  ranked.sort((a, b) => b[0] - a[0]);

  for (let i = 0; i < ranked.length; i++) {
    if (ranked[i][1] > 0) return 1.0 / (i + 1);
  }
  return 0.0;
}

/**
 * Compute Precision@K — the fraction of the top-K items with `relevance > 0`.
 *
 * @param k cut-off depth; pass `0` to evaluate all items
 */
export function precisionAtK(
  scores: readonly number[],
  relevance: readonly number[],
  k: number,
): number {
  if (scores.length !== relevance.length) {
    throw new Error("scores and relevance must have the same length");
  }
  if (scores.length === 0) return 0.0;

  const n = scores.length;
  const cut = k === 0 || k > n ? n : k;

  const ranked = zip(scores, relevance);
  ranked.sort((a, b) => b[0] - a[0]);

  const relevant = ranked.slice(0, cut).filter(([, r]) => r > 0).length;
  return relevant / cut;
}
