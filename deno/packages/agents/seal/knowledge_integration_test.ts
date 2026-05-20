import { assert } from "@std/assert";
import {
  type BehavioralKnowledgeCache,
  defaultIntegrationConfig,
  type IntegrationConfig,
  type PersonalKnowledgeCache,
  SealKnowledgeCoordinator,
  validateIntegrationConfig,
} from "./knowledge_integration.ts";

function emptyBks(): BehavioralKnowledgeCache {
  return {
    getMatchingTruthsWithScores: () => Promise.resolve([]),
    getReliableTruths: () => Promise.resolve([]),
    queueSubmission: () => Promise.resolve(),
  };
}

function emptyPks(): PersonalKnowledgeCache {
  return {
    getAllFacts: () => Promise.resolve([]),
    upsertFactSimple: () => Promise.resolve(),
  };
}

function makeCoord(config: IntegrationConfig = defaultIntegrationConfig()) {
  return new SealKnowledgeCoordinator(emptyBks(), emptyPks(), config);
}

Deno.test("integration config validation", () => {
  // Default is OK.
  validateIntegrationConfig(defaultIntegrationConfig());

  // Invalid quality threshold.
  let bad = defaultIntegrationConfig();
  bad.min_seal_quality_for_bks_boost = 1.5;
  let threw = false;
  try {
    validateIntegrationConfig(bad);
  } catch {
    threw = true;
  }
  assert(threw);

  // Invalid weight sum.
  bad = defaultIntegrationConfig();
  bad.seal_weight = 0.5;
  bad.bks_weight = 0.5;
  bad.pks_weight = 0.5;
  threw = false;
  try {
    validateIntegrationConfig(bad);
  } catch {
    threw = true;
  }
  assert(threw);
});

Deno.test("confidence harmonization", () => {
  const coord = makeCoord();
  // Only SEAL.
  const a = coord.harmonizeConfidence(0.8, undefined, undefined);
  assert(Math.abs(a - 0.8) < 0.01);

  // SEAL + BKS + PKS: 0.6*0.5 + 0.9*0.3 + 0.8*0.2 = 0.73.
  const b = coord.harmonizeConfidence(0.6, 0.9, 0.8);
  assert(Math.abs(b - 0.73) < 0.01);
});

Deno.test("retrieval threshold adjustment", () => {
  const coord = makeCoord();
  // Low quality → lower threshold: 0.75 * 0.7 = 0.525.
  let t = coord.adjustRetrievalThreshold(0.75, 0.0);
  assert(Math.abs(t - 0.525) < 0.01);

  // High quality → 0.75 * 1.0 = 0.75.
  t = coord.adjustRetrievalThreshold(0.75, 1.0);
  assert(Math.abs(t - 0.75) < 0.01);

  // Medium quality → 0.75 * 0.85 = 0.6375.
  t = coord.adjustRetrievalThreshold(0.75, 0.5);
  assert(Math.abs(t - 0.6375) < 0.01);
});
