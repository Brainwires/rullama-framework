import { assert, assertEquals } from "@std/assert";
import { Message } from "@brainwires/core";
import { DenoKvSessionStore } from "./deno_kv.ts";
import { SessionId } from "./types.ts";

async function openStore(): Promise<{ store: DenoKvSessionStore; kv: Deno.Kv }> {
  const kv = await Deno.openKv(":memory:");
  return { store: new DenoKvSessionStore(kv), kv };
}

Deno.test("DenoKv roundtrip", async () => {
  const { store, kv } = await openStore();
  try {
    const id = SessionId.new("u1");
    await store.save(id, [Message.user("hi")]);
    const loaded = await store.load(id);
    assert(loaded);
    assertEquals(loaded.length, 1);
    assertEquals(loaded[0].text(), "hi");
  } finally {
    kv.close();
  }
});

Deno.test("DenoKv survives reopen (file-backed)", async () => {
  const dir = await Deno.makeTempDir();
  const path = `${dir}/sessions.kv`;
  try {
    {
      const kv = await Deno.openKv(path);
      const store = new DenoKvSessionStore(kv);
      await store.save(SessionId.new("persist"), [Message.user("keep me")]);
      kv.close();
    }
    const kv = await Deno.openKv(path);
    const store = new DenoKvSessionStore(kv);
    const loaded = await store.load(SessionId.new("persist"));
    assert(loaded);
    assertEquals(loaded.length, 1);
    assertEquals(loaded[0].text(), "keep me");
    kv.close();
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});

Deno.test("DenoKv list and delete", async () => {
  const { store, kv } = await openStore();
  try {
    await store.save(SessionId.new("a"), [Message.user("x")]);
    await store.save(SessionId.new("b"), [Message.user("y")]);
    const list = await store.list();
    assertEquals(list.length, 2);
    await store.delete(SessionId.new("a"));
    assertEquals((await store.list()).length, 1);
  } finally {
    kv.close();
  }
});

Deno.test("DenoKv load returns null for unknown", async () => {
  const { store, kv } = await openStore();
  try {
    assertEquals(await store.load(SessionId.new("missing")), null);
  } finally {
    kv.close();
  }
});

Deno.test("DenoKv save preserves created_at across overwrites", async () => {
  const { store, kv } = await openStore();
  try {
    const id = SessionId.new("stable");
    await store.save(id, [Message.user("first")]);
    const firstList = await store.list();
    const firstCreated = firstList[0].created_at;

    // A small delay so updated_at differs.
    await new Promise((r) => setTimeout(r, 5));
    await store.save(id, [Message.user("second")]);

    const secondList = await store.list();
    assertEquals(secondList[0].created_at, firstCreated, "created_at must survive overwrite");
    assert(secondList[0].updated_at >= firstCreated);
  } finally {
    kv.close();
  }
});
