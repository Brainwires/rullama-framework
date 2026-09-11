import { assert, assertEquals, assertThrows } from "@std/assert";
import { TransactionManager } from "./transaction.ts";

// ── stage and commit ────────────────────────────────────────────────────────

Deno.test("TransactionManager - stage and commit", () => {
  const tmpDir = Deno.makeTempDirSync();
  const target = `${tmpDir}/output.txt`;
  const mgr = TransactionManager.create();

  const staged = mgr.stage({
    key: "k1",
    target_path: target,
    content: "hello world",
  });
  assertEquals(staged, true);
  assertEquals(mgr.pendingCount(), 1);

  // Target must not exist before commit
  let exists = false;
  try {
    Deno.statSync(target);
    exists = true;
  } catch { /* expected */ }
  assertEquals(exists, false);

  const result = mgr.commit();
  assertEquals(result.committed, 1);
  assertEquals(Deno.readTextFileSync(target), "hello world");
  assertEquals(mgr.pendingCount(), 0);

  mgr.dispose();
  Deno.removeSync(tmpDir, { recursive: true });
});

// ── rollback ────────────────────────────────────────────────────────────────

Deno.test("TransactionManager - rollback discards staged writes", () => {
  const tmpDir = Deno.makeTempDirSync();
  const target = `${tmpDir}/discard.txt`;
  const mgr = TransactionManager.create();

  mgr.stage({ key: "k1", target_path: target, content: "data" });
  assertEquals(mgr.pendingCount(), 1);

  mgr.rollback();
  assertEquals(mgr.pendingCount(), 0);

  let exists = false;
  try {
    Deno.statSync(target);
    exists = true;
  } catch { /* expected */ }
  assertEquals(exists, false);

  mgr.dispose();
  Deno.removeSync(tmpDir, { recursive: true });
});

// ── duplicate key ───────────────────────────────────────────────────────────

Deno.test("TransactionManager - duplicate key is idempotent", () => {
  const tmpDir = Deno.makeTempDirSync();
  const target = `${tmpDir}/idem.txt`;
  const mgr = TransactionManager.create();

  const first = mgr.stage({
    key: "same-key",
    target_path: target,
    content: "v1",
  });
  assertEquals(first, true);

  const second = mgr.stage({
    key: "same-key",
    target_path: target,
    content: "v2",
  });
  assertEquals(second, false);
  assertEquals(mgr.pendingCount(), 1);

  mgr.commit();
  // Only the first content should have been committed
  assertEquals(Deno.readTextFileSync(target), "v1");

  mgr.dispose();
  Deno.removeSync(tmpDir, { recursive: true });
});

// ── multiple files ──────────────────────────────────────────────────────────

Deno.test("TransactionManager - commit multiple files", () => {
  const tmpDir = Deno.makeTempDirSync();
  const f1 = `${tmpDir}/a.txt`;
  const f2 = `${tmpDir}/b.txt`;
  const mgr = TransactionManager.create();

  mgr.stage({ key: "k-a", target_path: f1, content: "alpha" });
  mgr.stage({ key: "k-b", target_path: f2, content: "beta" });
  assertEquals(mgr.pendingCount(), 2);

  const result = mgr.commit();
  assertEquals(result.committed, 2);
  assertEquals(Deno.readTextFileSync(f1), "alpha");
  assertEquals(Deno.readTextFileSync(f2), "beta");

  mgr.dispose();
  Deno.removeSync(tmpDir, { recursive: true });
});

// ── empty commit ────────────────────────────────────────────────────────────

Deno.test("TransactionManager - empty commit succeeds", () => {
  const mgr = TransactionManager.create();
  const result = mgr.commit();
  assertEquals(result.committed, 0);
  assertEquals(result.paths.length, 0);
  mgr.dispose();
});

// ── commit creates parent directories ───────────────────────────────────────

Deno.test("TransactionManager - commit creates parent directories", () => {
  const tmpDir = Deno.makeTempDirSync();
  const nested = `${tmpDir}/nested/deep/file.txt`;
  const mgr = TransactionManager.create();

  mgr.stage({ key: "k-nested", target_path: nested, content: "content" });
  mgr.commit();

  assertEquals(Deno.readTextFileSync(nested), "content");

  mgr.dispose();
  Deno.removeSync(tmpDir, { recursive: true });
});

// ── commit clears queue ─────────────────────────────────────────────────────

Deno.test("TransactionManager - commit clears queue", () => {
  const tmpDir = Deno.makeTempDirSync();
  const mgr = TransactionManager.create();

  mgr.stage({
    key: "k",
    target_path: `${tmpDir}/f.txt`,
    content: "x",
  });
  mgr.commit();
  assertEquals(mgr.pendingCount(), 0);

  // After commit, new stages are accepted
  mgr.stage({
    key: "k2",
    target_path: `${tmpDir}/g.txt`,
    content: "y",
  });
  assertEquals(mgr.pendingCount(), 1);

  mgr.dispose();
  Deno.removeSync(tmpDir, { recursive: true });
});

// ── rollback clears queue ───────────────────────────────────────────────────

Deno.test("TransactionManager - rollback clears queue", () => {
  const tmpDir = Deno.makeTempDirSync();
  const mgr = TransactionManager.create();

  mgr.stage({
    key: "k",
    target_path: `${tmpDir}/f.txt`,
    content: "x",
  });
  mgr.rollback();
  assertEquals(mgr.pendingCount(), 0);

  mgr.dispose();
  Deno.removeSync(tmpDir, { recursive: true });
});

// ── dispose cleans up staging dir ───────────────────────────────────────────

Deno.test("TransactionManager - dispose cleans staging dir", () => {
  const mgr = TransactionManager.create();
  const dir = mgr.stagingDir;

  // Staging dir should exist
  const stat = Deno.statSync(dir);
  assertEquals(stat.isDirectory, true);

  mgr.dispose();

  let exists = false;
  try {
    Deno.statSync(dir);
    exists = true;
  } catch { /* expected */ }
  assertEquals(exists, false);
});

Deno.test("TransactionManager - a traversal key stays inside the staging dir", () => {
  const staging = Deno.makeTempDirSync();
  const target = Deno.makeTempDirSync();
  try {
    const tm = TransactionManager.create(staging);
    assert(tm.stage({
      key: "../../escape",
      target_path: `${target}/out.txt`,
      content: "x",
    }));
    const staged = [...Deno.readDirSync(staging)].map((e) => e.name);
    assertEquals(staged.length, 1);
    assert(staged[0].startsWith("..%2F..%2Fescape"));
    assert(!staged[0].includes("/"));
    tm.rollback();
  } finally {
    Deno.removeSync(staging, { recursive: true });
    Deno.removeSync(target, { recursive: true });
  }
});

Deno.test("TransactionManager - projectRoot rejects targets outside it", () => {
  const staging = Deno.makeTempDirSync();
  const root = Deno.makeTempDirSync();
  try {
    const tm = TransactionManager.create(staging, root);
    assert(
      tm.stage({ key: "in", target_path: `${root}/ok.txt`, content: "x" }),
    );
    assertThrows(
      () => tm.stage({ key: "out", target_path: "/etc/evil", content: "x" }),
      Error,
      "outside",
    );
    tm.rollback();
  } finally {
    Deno.removeSync(staging, { recursive: true });
    Deno.removeSync(root, { recursive: true });
  }
});
