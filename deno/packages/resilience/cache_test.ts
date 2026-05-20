import { assert, assertEquals } from "@std/assert";
import { ChatOptions, Message, defaultToolInputSchema, type Tool } from "@brainwires/core";
import { cacheKeyFor, CachedProvider } from "./cache.ts";
import { EchoProvider } from "./test_util.ts";

Deno.test("miss populates cache then hits match", async () => {
  const inner = EchoProvider.ok("p");
  const { provider: cached, cache: mem } = CachedProvider.withMemoryCache(inner);

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
  const toolA: Tool = { name: "alpha", description: "", input_schema: defaultToolInputSchema() };
  const toolB: Tool = { name: "beta", description: "", input_schema: defaultToolInputSchema() };

  const k1 = await cacheKeyFor(msgs, [toolA, toolB], opts);
  const k2 = await cacheKeyFor(msgs, [toolB, toolA], opts);
  assertEquals(k1.value, k2.value);
  assert(k1.value.length > 0);
});
