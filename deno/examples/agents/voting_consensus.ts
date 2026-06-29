// Example: MAKER Voting Consensus Building Blocks
// Demonstrates creating sampled responses with metadata, configuring red-flag
// validation, and validating responses against reliability criteria.
// Run: deno run deno/examples/agents/voting_consensus.ts

import {
  type MdapRedFlagResult,
  type OutputFormat,
  relaxedRedFlagConfig,
  type ResponseMetadata,
  type SampledResponse,
  StandardRedFlagValidator,
  strictRedFlagConfig,
  type VoteResult,
  type VotingMethod,
} from "@rullama/agent";

async function main() {
  console.log("=== MAKER Voting Consensus Building Blocks ===\n");

  // 1. Create sampled responses with metadata
  console.log("--- 1. Creating Sampled Responses ---\n");

  const responses: SampledResponse<string>[] = [
    {
      value: "Paris",
      metadata: {
        tokenCount: 12,
        responseTimeMs: 150,
        formatValid: true,
        finishReason: "stop",
        model: "claude-sonnet",
      },
      rawResponse: "The capital of France is Paris.",
      confidence: 0.5,
    },
    {
      value: "Paris",
      metadata: {
        tokenCount: 8,
        responseTimeMs: 120,
        formatValid: true,
        finishReason: "stop",
        model: "claude-sonnet",
      },
      rawResponse: "Paris",
      confidence: 0.95,
    },
    {
      value: "Wait, actually Lyon",
      metadata: {
        tokenCount: 45,
        responseTimeMs: 300,
        formatValid: true,
        finishReason: "stop",
      },
      rawResponse:
        "Wait, I think it might be Lyon. Actually, let me reconsider...",
      confidence: 0.5,
    },
    {
      value: "Paris",
      metadata: {
        tokenCount: 900,
        responseTimeMs: 2000,
        formatValid: true,
        finishReason: "length",
        model: "gpt-4o",
      },
      rawResponse: "A very long response that was truncated...",
      confidence: 0.5,
    },
  ];

  for (let i = 0; i < responses.length; i++) {
    const resp = responses[i];
    console.log(
      `  Response ${
        i + 1
      }: value="${resp.value}", tokens=${resp.metadata.tokenCount}, confidence=${
        resp.confidence.toFixed(2)
      }, model=${resp.metadata.model ?? "unknown"}`,
    );
  }

  // 2. Compare RedFlagConfig presets
  console.log("\n--- 2. RedFlagConfig Presets ---\n");

  const strict = strictRedFlagConfig();
  const relaxed = relaxedRedFlagConfig();

  console.log("  Strict config:");
  console.log(`    maxResponseTokens: ${strict.maxResponseTokens}`);
  console.log(`    requireExactFormat: ${strict.requireExactFormat}`);
  console.log(`    flagSelfCorrection: ${strict.flagSelfCorrection}`);
  console.log(
    `    confusionPatterns: ${strict.confusionPatterns.length} patterns`,
  );
  console.log(`    minResponseLength: ${strict.minResponseLength}`);
  console.log(`    maxEmptyLineRatio: ${strict.maxEmptyLineRatio.toFixed(1)}`);

  console.log("  Relaxed config:");
  console.log(`    maxResponseTokens: ${relaxed.maxResponseTokens}`);
  console.log(`    requireExactFormat: ${relaxed.requireExactFormat}`);
  console.log(`    flagSelfCorrection: ${relaxed.flagSelfCorrection}`);
  console.log(
    `    confusionPatterns: ${relaxed.confusionPatterns.length} patterns`,
  );
  console.log(`    minResponseLength: ${relaxed.minResponseLength}`);
  console.log(`    maxEmptyLineRatio: ${relaxed.maxEmptyLineRatio.toFixed(1)}`);

  // 3. Validate responses with the strict validator
  console.log("\n--- 3. Red-Flag Validation (Strict) ---\n");

  const validator = StandardRedFlagValidator.strict();

  for (let i = 0; i < responses.length; i++) {
    const resp = responses[i];
    const result: MdapRedFlagResult = validator.validate(
      resp.rawResponse,
      resp.metadata,
    );
    const status = result.valid
      ? "VALID"
      : `FLAGGED: ${result.reason.kind} (severity: ${
        result.severity.toFixed(2)
      })`;
    console.log(`  Response ${i + 1}: ${status}`);
  }

  // 4. Validate with relaxed config
  console.log("\n--- 4. Red-Flag Validation (Relaxed) ---\n");

  const relaxedValidator = new StandardRedFlagValidator(
    relaxedRedFlagConfig(),
  );

  for (let i = 0; i < responses.length; i++) {
    const resp = responses[i];
    const result = relaxedValidator.validate(resp.rawResponse, resp.metadata);
    const status = result.valid ? "VALID" : "FLAGGED";
    console.log(`  Response ${i + 1}: ${status}`);
  }

  // 5. Format-aware validation
  console.log("\n--- 5. Format-Aware Validation ---\n");

  const jsonFormat: OutputFormat = { kind: "json" };
  const oneOfFormat: OutputFormat = {
    kind: "one_of",
    options: ["Paris", "London", "Berlin"],
  };

  const jsonValidator = StandardRedFlagValidator.withFormat(jsonFormat);
  const oneOfValidator = StandardRedFlagValidator.withFormat(oneOfFormat);

  const jsonResponse = '{"answer": "Paris"}';
  const plainResponse = "Paris";

  const jsonMeta: ResponseMetadata = {
    tokenCount: 20,
    responseTimeMs: 100,
    formatValid: true,
    finishReason: "stop",
  };

  console.log(
    `  JSON validator on '${jsonResponse}': ${
      jsonValidator.validate(jsonResponse, jsonMeta).valid ? "VALID" : "FLAGGED"
    }`,
  );
  console.log(
    `  JSON validator on '${plainResponse}': ${
      jsonValidator.validate(plainResponse, jsonMeta).valid
        ? "VALID"
        : "FLAGGED"
    }`,
  );
  console.log(
    `  OneOf validator on '${plainResponse}': ${
      oneOfValidator.validate(plainResponse, jsonMeta).valid
        ? "VALID"
        : "FLAGGED"
    }`,
  );
  console.log(
    `  OneOf validator on 'Tokyo': ${
      oneOfValidator.validate("Tokyo", jsonMeta).valid ? "VALID" : "FLAGGED"
    }`,
  );

  // 6. Show VoteResult structure
  console.log("\n--- 6. VoteResult Structure ---\n");

  const voteResult: VoteResult<string> = {
    winner: "Paris",
    winnerVotes: 4,
    totalVotes: 5,
    totalSamples: 7,
    redFlaggedCount: 2,
    voteDistribution: { Paris: 4, Lyon: 1 },
    confidence: 0.80,
    redFlagReasons: [
      "Response too long: 900 tokens > 750 limit",
      "Self-correction detected: 'Wait,'",
    ],
    earlyStopped: false,
    weightedConfidence: 0.85,
    votingMethod: "first_to_ahead_by_k",
  };

  console.log(`  Winner: "${voteResult.winner}"`);
  console.log(
    `  Winner votes: ${voteResult.winnerVotes}/${voteResult.totalVotes}`,
  );
  console.log(
    `  Total samples (incl. red-flagged): ${voteResult.totalSamples}`,
  );
  console.log(`  Red-flagged: ${voteResult.redFlaggedCount}`);
  console.log(`  Confidence: ${(voteResult.confidence * 100).toFixed(0)}%`);
  console.log(
    `  Weighted confidence: ${
      ((voteResult.weightedConfidence ?? 0) * 100).toFixed(0)
    }%`,
  );
  console.log(`  Early stopped: ${voteResult.earlyStopped}`);
  console.log("  Vote distribution:");
  for (
    const [candidate, votes] of Object.entries(voteResult.voteDistribution)
  ) {
    console.log(`    ${candidate}: ${votes} votes`);
  }
  console.log("  Red-flag reasons:");
  for (const reason of voteResult.redFlagReasons) {
    console.log(`    - ${reason}`);
  }

  console.log("\n=== Done ===");
}

await main();
