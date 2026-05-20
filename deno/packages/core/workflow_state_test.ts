import { assert, assertEquals } from "@std/assert";
import { join } from "@std/path";
import {
  FsWorkflowStateStore,
  InMemoryWorkflowStateStore,
  isCompleted,
  newCheckpoint,
  newSideEffectRecord,
} from "./workflow_state.ts";

Deno.test("InMemory roundtrip", async () => {
  const store = new InMemoryWorkflowStateStore();
  assertEquals(await store.loadCheckpoint("t1"), null);

  const cp = newCheckpoint("t1", "agent-1");
  await store.saveCheckpoint(cp);

  const loaded = await store.loadCheckpoint("t1");
  assert(loaded);
  assertEquals(loaded.task_id, "t1");
  assertEquals(loaded.agent_id, "agent-1");
});

Deno.test("InMemory mark step and skip", async () => {
  const store = new InMemoryWorkflowStateStore();
  const effect = newSideEffectRecord("use-1", "write_file", "src/main.ts", true);
  await store.markStepComplete("t2", "use-1", effect);

  const cp = await store.loadCheckpoint("t2");
  assert(cp);
  assert(isCompleted(cp, "use-1"));
  assert(!isCompleted(cp, "use-2"));
  assertEquals(cp.step_index, 1);
});

Deno.test("InMemory delete removes checkpoint", async () => {
  const store = new InMemoryWorkflowStateStore();
  await store.saveCheckpoint(newCheckpoint("t3", "a"));
  await store.deleteCheckpoint("t3");
  assertEquals(await store.loadCheckpoint("t3"), null);
});

Deno.test("Fs save and load roundtrip", async () => {
  const dir = await Deno.makeTempDir();
  try {
    const store = new FsWorkflowStateStore(dir);
    assertEquals(await store.loadCheckpoint("task-a"), null);

    const cp = newCheckpoint("task-a", "agent-x");
    await store.saveCheckpoint(cp);

    const loaded = await store.loadCheckpoint("task-a");
    assert(loaded);
    assertEquals(loaded.task_id, "task-a");
    assertEquals(loaded.agent_id, "agent-x");
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("Fs atomic write leaves no tmp file", async () => {
  const dir = await Deno.makeTempDir();
  try {
    const store = new FsWorkflowStateStore(dir);
    await store.saveCheckpoint(newCheckpoint("atomic-task", "a"));

    // .tmp file must be gone after rename
    await assertDoesNotExist(join(dir, "atomic-task.json.tmp"));
    await assertExists(join(dir, "atomic-task.json"));
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("Fs markStepComplete creates checkpoint implicitly", async () => {
  const dir = await Deno.makeTempDir();
  try {
    const store = new FsWorkflowStateStore(dir);
    const effect = newSideEffectRecord("use-99", "write_file", "foo.ts", true);
    await store.markStepComplete("fresh-task", "use-99", effect);

    const cp = await store.loadCheckpoint("fresh-task");
    assert(cp);
    assert(isCompleted(cp, "use-99"));
    assertEquals(cp.step_index, 1);
    assertEquals(cp.side_effects_log.length, 1);
    assertEquals(cp.side_effects_log[0].tool_name, "write_file");
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("Fs delete is idempotent", async () => {
  const dir = await Deno.makeTempDir();
  try {
    const store = new FsWorkflowStateStore(dir);
    await store.saveCheckpoint(newCheckpoint("del-task", "a"));
    await store.deleteCheckpoint("del-task");
    await store.deleteCheckpoint("del-task");
    await store.deleteCheckpoint("never-existed");
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("Fs checkpoint path sanitises special chars", async () => {
  const dir = await Deno.makeTempDir();
  try {
    const store = new FsWorkflowStateStore(dir);
    const cp = newCheckpoint("proj/task.1 final", "a");
    await store.saveCheckpoint(cp);

    const loaded = await store.loadCheckpoint("proj/task.1 final");
    assert(loaded);
    assertEquals(loaded.task_id, "proj/task.1 final");

    // One file, no subdirectories.
    const entries: Deno.DirEntry[] = [];
    for await (const e of Deno.readDir(dir)) entries.push(e);
    assertEquals(entries.length, 1);
    assert(entries[0].isFile);
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

async function assertExists(path: string): Promise<void> {
  const stat = await Deno.stat(path).catch(() => null);
  assert(stat !== null, `expected ${path} to exist`);
}

async function assertDoesNotExist(path: string): Promise<void> {
  const stat = await Deno.stat(path).catch(() => null);
  assert(stat === null, `expected ${path} to NOT exist`);
}
