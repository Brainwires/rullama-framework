// Example: Trust & Audit
// Demonstrates trust level management with TrustManager and audit event
// logging with AuditLogger. Records successes and violations for an agent,
// observes trust level changes, then queries the audit log for statistics.
// Run: deno run --allow-read --allow-write --allow-env deno/examples/permissions/trust_audit.ts

import {
  AuditLogger,
  createAuditEvent,
  createAuditQuery,
  TrustManager,
  withAction,
  withAgent,
  withOutcome,
  withTarget,
} from "@rullama/permission";

async function main() {
  console.log("=== Trust & Audit Example ===\n");

  // 1. Setup
  console.log("--- 1. Setup ---\n");

  const tempDir = Deno.makeTempDirSync({ prefix: "rullama_trust_audit_" });
  const trustPath = `${tempDir}/trust_store.json`;
  const auditPath = `${tempDir}/audit.jsonl`;

  const trust = TrustManager.withPath(trustPath);
  const logger = AuditLogger.withPath(auditPath);

  console.log("  TrustManager created (persisted to temp dir)");
  console.log("  AuditLogger created (persisted to temp dir)");

  // 2. Record successes and build trust
  console.log("\n--- 2. Recording successes for agent-A ---\n");

  for (let i = 1; i <= 10; i++) {
    trust.recordSuccess("agent-A");

    let event = createAuditEvent("tool_execution");
    event = withAgent(event, "agent-A");
    event = withAction(event, "write_file");
    event = withTarget(event, `src/module_${i}.rs`);
    event = withOutcome(event, "success");
    logger.log(event);
  }

  const level = trust.getTrustLevel("agent-A");
  const factor = trust.get("agent-A")!;
  console.log(
    `  After 10 successes: level=${level}, score=${
      factor.score.toFixed(2)
    }, ops=${factor.total_ops}`,
  );

  // 3. Record a violation and observe trust decrease
  console.log("\n--- 3. Recording a major violation ---\n");

  const scoreBefore = trust.get("agent-A")!.score;

  trust.recordViolation("agent-A", "major");

  let violationEvent = createAuditEvent("policy_violation");
  violationEvent = withAgent(violationEvent, "agent-A");
  violationEvent = withAction(violationEvent, "write_file");
  violationEvent = withTarget(violationEvent, ".env");
  violationEvent = withOutcome(violationEvent, "denied");
  logger.log(violationEvent);

  const factorAfter = trust.get("agent-A")!;
  console.log(`  Score before violation: ${scoreBefore.toFixed(2)}`);
  console.log(
    `  Score after violation:  ${
      factorAfter.score.toFixed(2)
    }  (level: ${factorAfter.level})`,
  );

  // 4. Query audit log
  console.log("\n--- 4. Querying audit log ---\n");

  logger.flush();

  const allEvents = logger.query(createAuditQuery());
  console.log(`  Total events logged: ${allEvents.length}`);

  const violations = logger.query(
    createAuditQuery({ event_type: "policy_violation" }),
  );
  console.log(`  Policy violations:   ${violations.length}`);

  const agentAEvents = logger.query(
    createAuditQuery({ agent_id: "agent-A" }),
  );
  console.log(`  Events for agent-A:  ${agentAEvents.length}`);

  // 5. Audit statistics
  console.log("\n--- 5. Audit statistics ---\n");

  const stats = logger.statistics();
  console.log(`  Total events:        ${stats.total_events}`);
  console.log(`  Tool executions:     ${stats.tool_executions}`);
  console.log(`  Policy violations:   ${stats.policy_violations}`);
  console.log(`  Successful actions:  ${stats.successful_actions}`);
  console.log(`  Denied actions:      ${stats.denied_actions}`);

  // 6. Trust statistics
  console.log("\n--- 6. Trust statistics ---\n");

  const trustStats = trust.statistics();
  console.log(`  Total agents:        ${trustStats.total_agents}`);
  console.log(`  Total violations:    ${trustStats.total_violations}`);
  console.log(`  Total operations:    ${trustStats.total_operations}`);

  // Cleanup temp directory
  try {
    Deno.removeSync(tempDir, { recursive: true });
  } catch { /* ignore */ }

  console.log("\n=== Done ===");
}

await main();
