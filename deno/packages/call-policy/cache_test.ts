import { assert, assertEquals } from "@std/assert";
import {
  ChatOptions,
  createUsage,
  defaultToolInputSchema,
  Message,
  type Tool,
} from "@rullama/core";
import {
  CachedProvider,
  type CachedResponse,
  cacheKeyFor,
  MemoryCache,
} from "./cache.ts";
import { EchoProvider } from "./test_util.ts";

Deno.test("miss populates cache then hits match", async () => {
  const inner = EchoProvider.ok("p");
  const { provider: cached, cache: mem } = CachedProvider.withMemoryCache(
    inner,
  );

  const msgs = [Message.user("hello")];
  const opts = new ChatOptions();

  const r1 = await cached.chat(msgs, undefined, opts);
  assertEquals(inner.calls(), 1);
  assertEquals(mem.size(), 1);

  const r2 = await cached.chat(msgs, undefined, opts);
  assertEquals(inner.calls(), 1, "cache hit must not call inner provider");
  assertEquals(r1.message.text(), r2.message.text());
});

Deno.test("different messages miss", async () => {
  const inner = EchoProvider.ok("p");
  const { provider: cached } = CachedProvider.withMemoryCache(inner);
  const opts = new ChatOptions();

  await cached.chat([Message.user("a")], undefined, opts);
  await cached.chat([Message.user("b")], undefined, opts);
  assertEquals(inner.calls(), 2);
});

Deno.test("key stable across tool reordering", async () => {
  const opts = new ChatOptions();
  const msgs = [Message.user("x")];
  const toolA: Tool = {
    name: "alpha",
    description: "",
    input_schema: defaultToolInputSchema(),
  };
  const toolB: Tool = {
    name: "beta",
    description: "",
    input_schema: defaultToolInputSchema(),
  };

  const k1 = await cacheKeyFor(msgs, [toolA, toolB], opts);
  const k2 = await cacheKeyFor(msgs, [toolB, toolA], opts);
  assertEquals(k1.value, k2.value);
  assert(k1.value.length > 0);
});

Deno.test("MemoryCache evicts least-recently-used entries past maxEntries", async () => {
  const cache = new MemoryCache(2);
  const resp = (text: string): CachedResponse => ({
    role: "assistant",
    text,
    usage: createUsage(1, 1),
    finish_reason: "end_turn",
  });
  await cache.put({ value: "a" }, resp("a"));
  await cache.put({ value: "b" }, resp("b"));
  await cache.get({ value: "a" }); // a is now the most recent
  await cache.put({ value: "c" }, resp("c")); // evicts b
  assertEquals(cache.size(), 2);
  assertEquals(await cache.get({ value: "b" }), null);
  assertEquals((await cache.get({ value: "a" }))?.text, "a");
  assertEquals((await cache.get({ value: "c" }))?.text, "c");
});
