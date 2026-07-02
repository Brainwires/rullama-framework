// Example: Tool Transactions
// Demonstrates the TransactionManager for two-phase commit file write operations.
// Stage files, inspect pending state, then commit or rollback.
// Run: deno run --allow-read --allow-write --allow-env deno/examples/tool-system/tool_transactions.ts

import { TransactionManager } from "@rullama/tool-runtime";

async function main() {
  console.log("=== Tool Transactions Example ===\n");

  // 1. Create a transaction manager
  const txn = TransactionManager.create();
  console.log(`Transaction manager created.`);
  console.log(`Staging directory: ${txn.stagingDir}`);
  console.log(`Pending writes: ${txn.pendingCount()}\n`);

  // 2. Stage multiple file writes
  console.log("=== Staging Writes ===\n");

  const targetDir = `${
    Deno.env.get("TMPDIR") ?? "/tmp"
  }/rullama-txn-example`;
  Deno.mkdirSync(targetDir, { recursive: true });

  const writes = [
    {
      key: "config",
      target_path: `${targetDir}/config.json`,
      content: '{"version": "1.0", "debug": false}',
    },
    {
      key: "readme",
      target_path: `${targetDir}/README.md`,
      content: "# My Project\n\nA demonstration project.",
    },
    {
      key: "main",
      target_path: `${targetDir}/src/main.ts`,
      content: 'console.log("Hello from transactions!");\n',
    },
  ];

  for (const write of writes) {
    const staged = txn.stage(write);
    console.log(
      `  Staged '${write.key}' -> ${write.target_path}: ${
        staged ? "OK" : "already staged"
      }`,
    );
  }

  // Demonstrate duplicate key rejection (first write wins)
  const duplicate = txn.stage({
    key: "config",
    target_path: `${targetDir}/config-v2.json`,
    content: '{"version": "2.0"}',
  });
  console.log(
    `  Staged duplicate 'config': ${
      duplicate ? "OK" : "rejected (first write wins)"
    }`,
  );

  console.log(`\n  Pending writes: ${txn.pendingCount()}`);

  // 3. Commit the transaction
  console.log("\n=== Committing Transaction ===\n");

  const result = txn.commit();
  console.log(`  Committed: ${result.committed} file(s)`);
  console.log(`  Paths:`);
  for (const path of result.paths) {
    console.log(`    - ${path}`);
  }
  console.log(`  Pending after commit: ${txn.pendingCount()}`);

  // 4. Verify committed files exist
  console.log("\n=== Verifying Committed Files ===\n");

  for (const write of writes) {
    try {
      const content = Deno.readTextFileSync(write.target_path);
      console.log(
        `  ${write.key}: ${content.substring(0, 50)}${
          content.length > 50 ? "..." : ""
        }`,
      );
    } catch (e) {
      console.log(`  ${write.key}: ERROR - ${e}`);
    }
  }

  // 5. Demonstrate rollback with a new transaction
  console.log("\n=== Rollback Demonstration ===\n");

  const txn2 = TransactionManager.create();
  txn2.stage({
    key: "dangerous",
    target_path: `${targetDir}/should-not-exist.txt`,
    content: "This file should never be committed.",
  });
  txn2.stage({
    key: "also-dangerous",
    target_path: `${targetDir}/also-should-not-exist.txt`,
    content: "Neither should this one.",
  });

  console.log(`  Staged ${txn2.pendingCount()} write(s) for rollback demo.`);

  txn2.rollback();
  console.log(`  Rolled back. Pending: ${txn2.pendingCount()}`);

  // Verify the files do NOT exist
  let filesExist = false;
  try {
    Deno.statSync(`${targetDir}/should-not-exist.txt`);
    filesExist = true;
  } catch {
    // Expected: file does not exist
  }
  console.log(`  Rolled-back files exist: ${filesExist}`);

  // 6. Cleanup
  console.log("\n=== Cleanup ===\n");
  txn.dispose();
  txn2.dispose();

  try {
    Deno.removeSync(targetDir, { recursive: true });
    console.log(`  Removed demo directory: ${targetDir}`);
  } catch {
    console.log(`  Could not remove demo directory (may already be cleaned).`);
  }

  console.log("\nDone.");
}

await main();
