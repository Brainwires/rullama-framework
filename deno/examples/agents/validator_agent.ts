// Example: ValidatorAgent for Quality Gates
// Demonstrates how ValidatorAgent runs external validation checks
// (duplicate detection, syntax validity) on an agent's working set.
// A file with intentional issues is validated, fixed, then re-validated.
// Run: deno run deno/examples/agents/validator_agent.ts

import {
  defaultValidatorAgentConfig,
  formatValidationFeedback,
  formatValidatorStatus,
  runValidation,
  type ValidationCheck,
  type ValidationConfig,
  ValidatorAgent,
} from "@rullama/agent";

async function main() {
  console.log("=== Validation Loop Demo ===\n");

  // 1. Create a temp directory with a file that has issues
  const tmpDir = await Deno.makeTempDir({ prefix: "rullama-validation-" });

  const badFile = `${tmpDir}/utils.ts`;
  const badContent = `export function greet(name: string): string {
    return \`Hello, \${name}!\`;
}

export function greet(name: string): string {
    return \`Hi, \${name}!\`;
}
`;
  await Deno.writeTextFile(badFile, badContent);
  console.log(`Created file with duplicate function: ${badFile}`);

  // 2. Configure validation
  const checks: ValidationCheck[] = [
    { kind: "no_duplicates" },
    { kind: "syntax_valid" },
  ];

  const config: ValidationConfig = {
    checks,
    workingDirectory: tmpDir,
    maxRetries: 3,
    enabled: true,
    workingSetFiles: ["utils.ts"],
  };

  console.log("Validation config:");
  console.log(`  Checks:            [${checks.map((c) => c.kind).join(", ")}]`);
  console.log(`  Working directory:  ${config.workingDirectory}`);
  console.log(`  Max retries:       ${config.maxRetries}`);
  console.log(`  Working set files: [${config.workingSetFiles.join(", ")}]`);

  // 3. Run validation -- expect failure
  console.log("\n--- First validation run (expecting issues) ---");

  const result = await runValidation(config);
  console.log(`Passed: ${result.passed}`);
  console.log(`Issues found: ${result.issues.length}`);

  for (const issue of result.issues) {
    const severity = issue.severity.toUpperCase();
    console.log(`  [${severity}] ${issue.check}: ${issue.message}`);
    if (issue.file) {
      let location = `         File: ${issue.file}`;
      if (issue.line != null) location += ` (line ${issue.line})`;
      console.log(location);
    }
  }

  // Show the formatted feedback an agent would receive
  const feedback = formatValidationFeedback(result);
  console.log(`\nAgent feedback:\n${feedback}`);

  // 4. Fix the file
  console.log("--- Fixing the file ---");

  const goodContent = `export function greet(name: string): string {
    return \`Hello, \${name}!\`;
}

export function farewell(name: string): string {
    return \`Goodbye, \${name}!\`;
}
`;
  await Deno.writeTextFile(badFile, goodContent);
  console.log("Replaced duplicate function with unique 'farewell'");

  // 5. Re-run validation -- expect pass
  console.log("\n--- Second validation run (expecting pass) ---");

  const result2 = await runValidation(config);
  console.log(`Passed: ${result2.passed}`);
  console.log(`Issues found: ${result2.issues.length}`);

  const feedback2 = formatValidationFeedback(result2);
  console.log(`Agent feedback: ${feedback2}`);

  // 6. Demonstrate ValidatorAgent wrapper
  console.log("\n--- ValidatorAgent Wrapper ---");

  const validatorConfig = defaultValidatorAgentConfig(config);
  const validator = new ValidatorAgent("validator-1", validatorConfig);

  console.log(`  Agent ID: ${validator.id}`);
  console.log(`  Status: ${formatValidatorStatus(validator.status)}`);

  const agentResult = await validator.validate();
  console.log(`  Success: ${agentResult.success}`);
  console.log(`  Files checked: ${agentResult.filesChecked}`);
  console.log(`  Duration: ${agentResult.durationMs}ms`);
  console.log(`  Status after: ${formatValidatorStatus(validator.status)}`);

  // 7. Demonstrate file-existence check
  console.log("\n--- File existence check ---");

  const missingConfig: ValidationConfig = {
    checks: [],
    workingDirectory: tmpDir,
    maxRetries: 1,
    enabled: true,
    workingSetFiles: ["nonexistent.rs"],
  };

  const missingResult = await runValidation(missingConfig);
  console.log(`Passed (with missing file): ${missingResult.passed}`);
  for (const issue of missingResult.issues) {
    console.log(`  [${issue.check}] ${issue.message}`);
  }

  // 8. Summary
  console.log("\n--- Summary ---");
  console.log(
    "The validation loop prevents agents from reporting success when:",
  );
  console.log("  - Files in the working set do not exist on disk");
  console.log("  - Duplicate exports/functions/types are present");
  console.log("  - Basic syntax errors are detected");
  console.log("  - Build commands fail (when BuildSuccess check is enabled)");

  // Cleanup
  try {
    await Deno.remove(tmpDir, { recursive: true });
  } catch {
    // Ignore cleanup errors
  }

  console.log("\nValidation loop demo complete.");
}

await main();
