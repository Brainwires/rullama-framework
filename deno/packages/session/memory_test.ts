import { assert, assertEquals } from "@std/assert";
import { Message } from "@brainwires/core";
import { InMemorySessionStore } from "./memory.ts";
import { SessionId } from "./types.ts";

Deno.test("roundtrip save load delete", async () => {
  const store = new InMemorySessionStore();
  const id = SessionId.new("alice");

  assertEquals(await store.load(id), null);

  const msgs = [Message.user("hi"), Message.assistant("hello")];
  await store.save(id, msgs);

  const loaded = await store.load(id);
  assert(loaded);
  assertEquals(loaded.length, 2);
  assertEquals(loaded[0].text(), "hi");

  await store.delete(id);
  assertEquals(await store.load(id), null);
});

Deno.test("save overwrites atomically", async () => {
  const store = new InMemorySessionStore();
  const id = SessionId.new("bob");
  await store.save(id, [Message.user("one")]);
  await store.save(id, [Message.user("two"), Message.user("three")]);
  const loaded = await store.load(id);
  assert(loaded);
  assertEquals(loaded.length, 2);
  assertEquals(loaded[0].text(), "two");
});

Deno.test("list returns known sessions", async () => {
  const store = new InMemorySessionStore();
  await store.save(SessionId.new("a"), [Message.user("x")]);
  await store.save(SessionId.new("b"), [Message.user("y")]);
  const list = await store.list();
  assertEquals(list.length, 2);
  const ids = list.map((r) => r.id.asStr());
  assert(ids.includes("a") && ids.includes("b"));
});

Deno.test("delete unknown is noop", async () => {
  const store = new InMemorySessionStore();
  await store.delete(SessionId.new("never-existed"));
});

Deno.test("load returns a defensive copy", async () => {
  const store = new InMemorySessionStore();
  const id = SessionId.new("defensive");
  await store.save(id, [Message.user("first")]);
  const first = await store.load(id);
  assert(first);
  first.push(Message.user("mutated"));
  const second = await store.load(id);
  assert(second);
  assertEquals(second.length, 1, "mutating the returned array must not affect store");
});

Deno.test("listPaginated slices results", async () => {
  const store = new InMemorySessionStore();
  for (const x of ["a", "b", "c", "d"]) {
    await store.save(SessionId.new(x), [Message.user(x)]);
    // Ensure distinct updated_at timestamps when tests run fast.
    await new Promise((r) => setTimeout(r, 2));
  }
  const page = await store.listPaginated({ offset: 1, limit: 2 });
  assertEquals(page.length, 2);
  // Unbounded limit via null:
  const all = await store.listPaginated({ offset: 0, limit: null });
  assertEquals(all.length, 4);
});
