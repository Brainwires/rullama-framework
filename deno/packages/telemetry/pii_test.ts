import { assert, assertEquals } from "@std/assert";
import { hashSessionId, redactSecrets } from "./pii.ts";

Deno.test("hash is deterministic and short hex", async () => {
  const a = await hashSessionId("user-42");
  const b = await hashSessionId("user-42");
  assertEquals(a, b);
  assertEquals(a.length, 12);
  assert(/^[0-9a-f]+$/.test(a));
});

Deno.test("hash changes with input", async () => {
  const a = await hashSessionId("alice");
  const b = await hashSessionId("bob");
  assert(a !== b);
});

Deno.test("redact replaces obvious secrets", () => {
  const r = redactSecrets("api_key=sk-abcdef0123456789abcdef0123456789 plain-text");
  assert(r.includes("REDACTED"));
  assert(r.includes("plain-text"));
});
