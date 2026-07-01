/**
 * Cross-package integration test: Permission enforcement.
 *
 * Verifies that @rullama/permissions PolicyEngine correctly allows
 * and denies operations based on configured policies, and that
 * capability profiles enforce expected restrictions.
 */

import {
  assert,
  assertEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";
import {
  AgentCapabilities,
  createPolicy,
  createPolicyRequest,
  isDecisionAllowed,
  isDecisionRequiresApproval,
  parseCapabilityProfile,
  PolicyActions,
  PolicyEngine,
  policyRequestForFile,
  policyRequestForGit,
  policyRequestForNetwork,
  policyRequestForTool,
} from "@rullama/permission";

// ---------------------------------------------------------------------------
// PolicyEngine with custom policies
// ---------------------------------------------------------------------------

Deno.test("PolicyEngine allows when no policies match", () => {
  const engine = new PolicyEngine();
  const request = createPolicyRequest({ tool_name: "read_file" });
  const decision = engine.evaluate(request);

  assert(isDecisionAllowed(decision), "should allow when no policies match");
  assertEquals(decision.matched_policy, undefined);
});

Deno.test("PolicyEngine denies matching tool", () => {
  const engine = new PolicyEngine();
  engine.addPolicy(
    createPolicy("block_bash", {
      name: "Block Bash",
      description: "Deny all bash execution",
      conditions: [{ type: "tool", name: "execute_command" }],
      action: PolicyActions.Deny,
      priority: 100,
    }),
  );

  const request = policyRequestForTool("execute_command");
  const decision = engine.evaluate(request);

  assertEquals(decision.action.type, "deny");
  assertEquals(decision.matched_policy, "block_bash");
  assert(!isDecisionAllowed(decision));
});

Deno.test("PolicyEngine allows non-matching tool", () => {
  const engine = new PolicyEngine();
  engine.addPolicy(
    createPolicy("block_bash", {
      conditions: [{ type: "tool", name: "execute_command" }],
      action: PolicyActions.Deny,
      priority: 100,
    }),
  );

  const request = policyRequestForTool("read_file");
  const decision = engine.evaluate(request);

  assert(isDecisionAllowed(decision), "read_file should be allowed");
});

// ---------------------------------------------------------------------------
// File path policies
// ---------------------------------------------------------------------------

Deno.test("PolicyEngine denies access to .env files", () => {
  const engine = PolicyEngine.withDefaults();
  const request = policyRequestForFile("/project/.env", "read_file");
  const decision = engine.evaluate(request);

  assertEquals(decision.action.type, "deny");
  assertEquals(decision.matched_policy, "protect_env_files");
});

Deno.test("PolicyEngine denies access to secret files", () => {
  const engine = PolicyEngine.withDefaults();
  const request = policyRequestForFile(
    "/project/config/secret.json",
    "read_file",
  );
  const decision = engine.evaluate(request);

  assertEquals(decision.action.type, "deny_with_message");
  assertEquals(decision.matched_policy, "protect_secrets");
});

Deno.test("PolicyEngine allows normal file access", () => {
  const engine = PolicyEngine.withDefaults();
  const request = policyRequestForFile("/project/src/main.ts", "read_file");
  const decision = engine.evaluate(request);

  assert(isDecisionAllowed(decision), "normal file access should be allowed");
});

// ---------------------------------------------------------------------------
// Git operation policies
// ---------------------------------------------------------------------------

Deno.test("PolicyEngine requires approval for git reset", () => {
  const engine = PolicyEngine.withDefaults();
  const request = policyRequestForGit("Reset");
  const decision = engine.evaluate(request);

  assert(
    isDecisionRequiresApproval(decision),
    "git reset should require approval",
  );
  assertEquals(decision.matched_policy, "approve_git_reset");
});

Deno.test("PolicyEngine requires approval for git rebase", () => {
  const engine = PolicyEngine.withDefaults();
  const request = policyRequestForGit("Rebase");
  const decision = engine.evaluate(request);

  assert(
    isDecisionRequiresApproval(decision),
    "git rebase should require approval",
  );
  assertEquals(decision.matched_policy, "approve_git_rebase");
});

// ---------------------------------------------------------------------------
// Policy priority ordering
// ---------------------------------------------------------------------------

Deno.test("higher priority policy wins over lower priority", () => {
  const engine = new PolicyEngine();

  // Low priority: allow all tools
  engine.addPolicy(
    createPolicy("allow_all", {
      conditions: [{ type: "always" }],
      action: PolicyActions.Allow,
      priority: 10,
    }),
  );

  // High priority: deny specific tool
  engine.addPolicy(
    createPolicy("deny_dangerous", {
      conditions: [{ type: "tool", name: "delete_file" }],
      action: PolicyActions.Deny,
      priority: 100,
    }),
  );

  const request = policyRequestForTool("delete_file");
  const decision = engine.evaluate(request);

  assertEquals(decision.action.type, "deny", "high-priority deny should win");
  assertEquals(decision.matched_policy, "deny_dangerous");
});

// ---------------------------------------------------------------------------
// Capability profiles
// ---------------------------------------------------------------------------

Deno.test("read_only profile blocks writes", () => {
  const caps = AgentCapabilities.readOnly();

  // read_only should allow reads
  assert(caps.filesystem.read_paths.length > 0, "should have read paths");
  // read_only should block writes (empty write_paths)
  assertEquals(
    caps.filesystem.write_paths.length,
    0,
    "should have no write paths",
  );
});

Deno.test("full_access profile allows all", () => {
  const caps = AgentCapabilities.fullAccess();

  assert(caps.filesystem.read_paths.length > 0, "should have read paths");
  assert(caps.filesystem.write_paths.length > 0, "should have write paths");
  assert(caps.network.allow_all, "network allow_all should be true");
  assert(caps.spawning.can_spawn, "spawning can_spawn should be true");
});

Deno.test("standard_dev profile has balanced permissions", () => {
  const caps = AgentCapabilities.standardDev();

  assert(caps.filesystem.read_paths.length > 0, "should have read paths");
  assert(caps.filesystem.write_paths.length > 0, "should have write paths");
  // standard_dev should have some denied paths
  assert(
    caps.filesystem.denied_paths.length > 0,
    "should have denied paths for safety",
  );
});

Deno.test("fromProfile creates correct profiles", () => {
  const readOnly = AgentCapabilities.fromProfile("read_only");
  assertEquals(readOnly.filesystem.write_paths.length, 0);

  const fullAccess = AgentCapabilities.fromProfile("full_access");
  assert(fullAccess.filesystem.write_paths.length > 0);
});

Deno.test("parseCapabilityProfile parses valid profiles", () => {
  assertEquals(parseCapabilityProfile("read_only"), "read_only");
  assertEquals(parseCapabilityProfile("standard_dev"), "standard_dev");
  assertEquals(parseCapabilityProfile("full_access"), "full_access");
  assertEquals(parseCapabilityProfile("readonly"), "read_only"); // normalized
});

Deno.test("parseCapabilityProfile returns undefined for invalid", () => {
  assertEquals(parseCapabilityProfile("invalid"), undefined);
  assertEquals(parseCapabilityProfile(""), undefined);
});

// ---------------------------------------------------------------------------
// Network policy
// ---------------------------------------------------------------------------

Deno.test("custom network domain policy works", () => {
  const engine = new PolicyEngine();
  engine.addPolicy(
    createPolicy("block_evil_domain", {
      conditions: [{ type: "domain", pattern: "*.evil.com" }],
      action: PolicyActions.Deny,
      priority: 100,
    }),
  );

  const blockedReq = policyRequestForNetwork("api.evil.com");
  const blockedDecision = engine.evaluate(blockedReq);
  assertEquals(blockedDecision.action.type, "deny");

  const allowedReq = policyRequestForNetwork("api.good.com");
  const allowedDecision = engine.evaluate(allowedReq);
  assert(isDecisionAllowed(allowedDecision), "good domain should be allowed");
});

// ---------------------------------------------------------------------------
// Policy engine management
// ---------------------------------------------------------------------------

Deno.test("addPolicy and removePolicy work correctly", () => {
  const engine = new PolicyEngine();
  const policy = createPolicy("temp_policy", {
    conditions: [{ type: "tool", name: "test_tool" }],
    action: PolicyActions.Deny,
  });

  engine.addPolicy(policy);
  assertEquals(engine.policies().length, 1);

  const removed = engine.removePolicy("temp_policy");
  assert(removed !== undefined, "removed policy should be returned");
  assertEquals(engine.policies().length, 0);
});

Deno.test("removePolicy returns undefined for non-existent", () => {
  const engine = new PolicyEngine();
  const removed = engine.removePolicy("nope");
  assertEquals(removed, undefined);
});
