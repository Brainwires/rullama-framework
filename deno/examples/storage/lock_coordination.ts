// Example: Lock Coordination — in-memory resource locking for agents
// Demonstrates a simple lock coordination pattern using the storage layer's
// InMemoryStorageBackend to track resource locks across agents.
// Run: deno run deno/examples/storage/lock_coordination.ts

import {
  FieldTypes,
  fieldValueAsI64,
  fieldValueAsStr,
  FieldValues,
  Filters,
  InMemoryStorageBackend,
  optionalField,
  type Record,
  recordGet,
  requiredField,
} from "@rullama/storage";

// -- Lock helpers built on top of InMemoryStorageBackend ---------------------

const LOCKS_TABLE = "locks";
const LOCKS_SCHEMA = [
  requiredField("lock_id", FieldTypes.Utf8),
  requiredField("lock_type", FieldTypes.Utf8),
  requiredField("resource_path", FieldTypes.Utf8),
  requiredField("agent_id", FieldTypes.Utf8),
  requiredField("acquired_at", FieldTypes.Int64),
  optionalField("expires_at", FieldTypes.Int64),
];

interface LockInfo {
  lockId: string;
  lockType: string;
  resourcePath: string;
  agentId: string;
  acquiredAt: number;
  expiresAt?: number;
}

class SimpleLockStore {
  private backend = new InMemoryStorageBackend();
  private nextId = 1;

  async init(): Promise<void> {
    await this.backend.ensureTable(LOCKS_TABLE, LOCKS_SCHEMA);
  }

  async tryAcquire(
    lockType: string,
    resourcePath: string,
    agentId: string,
    ttlMs?: number,
  ): Promise<boolean> {
    // Check if resource is already locked
    const filter = Filters.And([
      Filters.Eq("lock_type", FieldValues.Utf8(lockType)),
      Filters.Eq("resource_path", FieldValues.Utf8(resourcePath)),
    ]);
    const existing = await this.backend.query(LOCKS_TABLE, filter);

    if (existing.length > 0) {
      const holderField = recordGet(existing[0], "agent_id");
      const holder = holderField ? fieldValueAsStr(holderField) : null;
      // Allow idempotent re-acquisition by same agent
      if (holder === agentId) return true;
      return false;
    }

    const now = Math.floor(Date.now() / 1000);
    const lockId = `lock-${this.nextId++}`;
    const expiresAt = ttlMs !== undefined
      ? now + Math.floor(ttlMs / 1000)
      : null;

    await this.backend.insert(LOCKS_TABLE, [
      [
        ["lock_id", FieldValues.Utf8(lockId)],
        ["lock_type", FieldValues.Utf8(lockType)],
        ["resource_path", FieldValues.Utf8(resourcePath)],
        ["agent_id", FieldValues.Utf8(agentId)],
        ["acquired_at", FieldValues.Int64(now)],
        ["expires_at", FieldValues.Int64(expiresAt)],
      ],
    ]);
    return true;
  }

  async isLocked(
    lockType: string,
    resourcePath: string,
  ): Promise<LockInfo | undefined> {
    const filter = Filters.And([
      Filters.Eq("lock_type", FieldValues.Utf8(lockType)),
      Filters.Eq("resource_path", FieldValues.Utf8(resourcePath)),
    ]);
    const records = await this.backend.query(LOCKS_TABLE, filter, 1);
    if (records.length === 0) return undefined;
    return parseLockRecord(records[0]);
  }

  async release(
    lockType: string,
    resourcePath: string,
    agentId: string,
  ): Promise<boolean> {
    const filter = Filters.And([
      Filters.Eq("lock_type", FieldValues.Utf8(lockType)),
      Filters.Eq("resource_path", FieldValues.Utf8(resourcePath)),
      Filters.Eq("agent_id", FieldValues.Utf8(agentId)),
    ]);
    const before = await this.backend.count(LOCKS_TABLE, filter);
    if (before === 0) return false;
    await this.backend.delete(LOCKS_TABLE, filter);
    return true;
  }

  async releaseAllForAgent(agentId: string): Promise<number> {
    const filter = Filters.Eq("agent_id", FieldValues.Utf8(agentId));
    const count = await this.backend.count(LOCKS_TABLE, filter);
    if (count > 0) {
      await this.backend.delete(LOCKS_TABLE, filter);
    }
    return count;
  }

  async listLocks(): Promise<LockInfo[]> {
    const records = await this.backend.query(LOCKS_TABLE);
    return records.map(parseLockRecord);
  }

  async cleanupExpired(): Promise<number> {
    const now = Math.floor(Date.now() / 1000);
    const filter = Filters.And([
      Filters.NotNull("expires_at"),
      Filters.Lte("expires_at", FieldValues.Int64(now)),
    ]);
    const count = await this.backend.count(LOCKS_TABLE, filter);
    if (count > 0) {
      await this.backend.delete(LOCKS_TABLE, filter);
    }
    return count;
  }
}

function parseLockRecord(r: Record): LockInfo {
  return {
    lockId: fieldValueAsStr(recordGet(r, "lock_id")!)!,
    lockType: fieldValueAsStr(recordGet(r, "lock_type")!)!,
    resourcePath: fieldValueAsStr(recordGet(r, "resource_path")!)!,
    agentId: fieldValueAsStr(recordGet(r, "agent_id")!)!,
    acquiredAt: fieldValueAsI64(recordGet(r, "acquired_at")!)!,
    expiresAt: recordGet(r, "expires_at")
      ? fieldValueAsI64(recordGet(r, "expires_at")!) ?? undefined
      : undefined,
  };
}

async function main() {
  console.log("=== Lock Coordination Example ===\n");

  // 1. Create a lock store
  const store = new SimpleLockStore();
  await store.init();
  console.log("SimpleLockStore created (backed by InMemoryStorageBackend)\n");

  // 2. Acquire a file-write lock
  const acquired = await store.tryAcquire(
    "file_write",
    "/src/main.ts",
    "agent-alpha",
  );
  console.log(
    `Acquire file_write on /src/main.ts (agent-alpha): ${
      acquired ? "OK" : "BLOCKED"
    }`,
  );

  // 3. Check lock status
  const lock = await store.isLocked("file_write", "/src/main.ts");
  if (lock) {
    console.log(`  Held by: ${lock.agentId} (lock_id=${lock.lockId})`);
  }
  console.log();

  // 4. Demonstrate idempotent re-acquisition (same agent)
  const reacquired = await store.tryAcquire(
    "file_write",
    "/src/main.ts",
    "agent-alpha",
  );
  console.log(
    `Re-acquire same lock (idempotent): ${reacquired ? "OK" : "BLOCKED"}`,
  );

  // 5. Demonstrate conflict — a different agent cannot take the same lock
  const conflict = await store.tryAcquire(
    "file_write",
    "/src/main.ts",
    "agent-beta",
  );
  console.log(
    `Acquire same resource as agent-beta: ${
      conflict ? "OK" : "BLOCKED (expected)"
    }`,
  );
  console.log();

  // 6. Acquire additional locks for different resources
  await store.tryAcquire("file_read", "/src/lib.ts", "agent-alpha");
  await store.tryAcquire("build", "/project/root", "agent-alpha");
  console.log("Acquired file_read on /src/lib.ts and build on /project/root");

  // 7. List all active locks
  const locks = await store.listLocks();
  console.log(`\nActive locks (${locks.length}):`);
  for (const l of locks) {
    console.log(
      `  [${l.lockType}] ${l.resourcePath} -> ${l.lockId} (agent=${l.agentId})`,
    );
  }
  console.log();

  // 8. Acquire a lock with a short TTL (will expire quickly)
  await store.tryAcquire("test", "/project/root", "agent-alpha", 50);
  console.log("Acquired test lock with 50ms TTL");

  // Wait for it to expire
  await new Promise((resolve) => setTimeout(resolve, 1100));

  // 9. Clean up expired locks
  const cleaned = await store.cleanupExpired();
  console.log(`Cleaned up ${cleaned} expired lock(s)`);

  // 10. Release a specific lock
  const released = await store.release(
    "file_write",
    "/src/main.ts",
    "agent-alpha",
  );
  console.log(
    `\nReleased file_write on /src/main.ts: ${released ? "OK" : "NOT FOUND"}`,
  );

  // 11. Release all locks for an agent
  const count = await store.releaseAllForAgent("agent-alpha");
  console.log(`Released ${count} remaining lock(s) for agent-alpha`);

  // Verify everything is cleaned up
  const remaining = await store.listLocks();
  console.log(`Remaining locks: ${remaining.length}`);

  console.log("\nDone.");
}

await main();
