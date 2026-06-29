// Example: Policy Engine
// Demonstrates declarative policy rules with PolicyEngine — deny, allow-with-audit,
// and require-approval actions evaluated against various request types.
// Run: deno run deno/examples/permissions/policy_engine.ts

import {
  createPolicy,
  isDecisionAllowed,
  isDecisionRequiresApproval,
  PolicyActions,
  PolicyEngine,
  policyRequestForFile,
  policyRequestForGit,
  policyRequestForTool,
} from "@rullama/permission";

async function main() {
  console.log("=== Policy Engine Example ===\n");

  // 1. Create a fresh policy engine with custom rules
  const engine = new PolicyEngine();

  console.log("--- 1. Registering policies ---\n");

  engine.addPolicy(createPolicy("deny_env_files", {
    name: "Deny .env Files",
    description: "Block all access to .env files which may contain secrets",
    conditions: [{ type: "file_path", pattern: "**/.env*" }],
    action: PolicyActions.Deny,
    priority: 100,
  }));
  console.log("  Added: deny_env_files  (priority 100, Deny)");

  engine.addPolicy(createPolicy("approve_git_reset", {
    name: "Approve Git Reset",
    description: "Require human approval before performing git reset",
    conditions: [{ type: "git_op", operation: "Reset" }],
    action: PolicyActions.RequireApproval,
    priority: 90,
  }));
  console.log("  Added: approve_git_reset  (priority 90, RequireApproval)");

  engine.addPolicy(createPolicy("audit_bash", {
    name: "Audit Bash Tool",
    description: "Allow bash tool usage but log every invocation for audit",
    conditions: [{ type: "tool", name: "bash" }],
    action: PolicyActions.AllowWithAudit,
    priority: 50,
  }));
  console.log("  Added: audit_bash  (priority 50, AllowWithAudit)");

  // 2. Evaluate requests against the policy engine
  console.log("\n--- 2. Evaluating requests ---\n");

  const fileRequest = policyRequestForFile(".env.local", "read_file");
  const fileDecision = engine.evaluate(fileRequest);
  console.log(
    `  Request: read .env.local\n    Decision: ${fileDecision.action.type}\n    Matched policy: ${fileDecision.matched_policy}\n    Allowed: ${
      isDecisionAllowed(fileDecision)
    }\n`,
  );

  const gitRequest = policyRequestForGit("Reset");
  const gitDecision = engine.evaluate(gitRequest);
  console.log(
    `  Request: git reset\n    Decision: ${gitDecision.action.type}\n    Matched policy: ${gitDecision.matched_policy}\n    Requires approval: ${
      isDecisionRequiresApproval(gitDecision)
    }\n`,
  );

  const toolRequest = policyRequestForTool("bash");
  const toolDecision = engine.evaluate(toolRequest);
  console.log(
    `  Request: bash tool\n    Decision: ${toolDecision.action.type}\n    Matched policy: ${toolDecision.matched_policy}\n    Allowed: ${
      isDecisionAllowed(toolDecision)
    }\n    Audit: ${toolDecision.audit}\n`,
  );

  const safeRequest = policyRequestForFile("src/main.rs", "read_file");
  const safeDecision = engine.evaluate(safeRequest);
  console.log(
    `  Request: read src/main.rs\n    Decision: ${safeDecision.action.type}\n    Matched policy: ${safeDecision.matched_policy}\n    Allowed: ${
      isDecisionAllowed(safeDecision)
    }\n`,
  );

  // 3. List all registered policies
  console.log("--- 3. Registered policies ---\n");

  for (const policy of engine.policies()) {
    console.log(
      `  [${policy.id}] ${policy.name} (priority ${policy.priority}) -> ${policy.action.type}`,
    );
  }

  console.log("\n=== Done ===");
}

await main();
