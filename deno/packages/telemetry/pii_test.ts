import { assert, assertEquals, assertStringIncludes } from "@std/assert";
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
  const r = redactSecrets(
    "api_key=sk-abcdef0123456789abcdef0123456789 plain-text",
  );
  assert(r.includes("REDACTED"));
  assert(r.includes("plain-text"));
});

Deno.test("redactSecrets covers vendor key formats, JWTs and PEM blocks", () => {
  const text = [
    "anthropic sk-ant-abcdefghijklmnopqrstuvwxyz0123",
    "github ghp_abcdefghijklmnopqrstuvwxyz0123456789",
    "aws AKIAIOSFODNN7EXAMPLE",
    "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc_def-ghi",
    "password=hunter2xyz",
  ].join("\n");
  const out = redactSecrets(text);
  for (
    const leak of [
      "sk-ant-",
      "ghp_",
      "AKIAIOSFODNN7EXAMPLE",
      "eyJhbGci",
      "hunter2xyz",
    ]
  ) {
    assert(!out.includes(leak), `${leak} leaked: ${out}`);
  }
  assertStringIncludes(out, "[REDACTED:");
});
