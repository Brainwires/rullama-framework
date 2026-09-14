import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { join } from "@std/path";
import {
  compileBoundedRegex,
  confinePath,
  confinePathLexical,
  isWithin,
  PathEscapeError,
  safeFileName,
} from "./guards.ts";

Deno.test("isWithin: root itself, children, and nothing else", () => {
  assert(isWithin("/app", "/app"));
  assert(isWithin("/app", "/app/src/a.ts"));
  assert(!isWithin("/app", "/apple"));
  assert(!isWithin("/app", "/etc/passwd"));
  assert(!isWithin("/app", "/app/../etc"));
});

Deno.test("confinePathLexical: relative and in-root absolute paths resolve; escapes throw", () => {
  assertEquals(confinePathLexical("/app", "src/a.ts"), "/app/src/a.ts");
  assertEquals(confinePathLexical("/app", "/app/src/a.ts"), "/app/src/a.ts");
  assertEquals(confinePathLexical("/app", "."), "/app");
  assertThrows(() => confinePathLexical("/app", "../x"), PathEscapeError);
  assertThrows(
    () => confinePathLexical("/app", "src/../../x"),
    PathEscapeError,
  );
  assertThrows(
    () => confinePathLexical("/app", "/etc/passwd"),
    PathEscapeError,
  );
  assertThrows(() => confinePathLexical("/app", "/apple/x"), PathEscapeError);
});

Deno.test("confinePath: follows symlinks and rejects a link that leaves the root", async () => {
  const outside = await Deno.makeTempDir();
  const root = await Deno.makeTempDir();
  try {
    await Deno.writeTextFile(join(outside, "secret"), "s");
    await Deno.symlink(join(outside, "secret"), join(root, "link"));
    await Deno.mkdir(join(root, "sub"));
    await Deno.symlink(outside, join(root, "dirlink"));

    assertEquals(
      await confinePath(root, "sub/new.txt"),
      join(root, "sub/new.txt"),
    );
    assertEquals(
      await confinePath(root, "sub/missing/deeper.txt"),
      join(root, "sub/missing/deeper.txt"),
    );
    await assertRejects(() => confinePath(root, "link"), PathEscapeError);
    await assertRejects(
      () => confinePath(root, "dirlink/secret"),
      PathEscapeError,
    );
    await assertRejects(
      () => confinePath(root, "dirlink/new-file"),
      PathEscapeError,
    );
  } finally {
    await Deno.remove(outside, { recursive: true });
    await Deno.remove(root, { recursive: true });
  }
});

Deno.test("compileBoundedRegex: length budget and syntax errors", () => {
  assert(compileBoundedRegex("a+b").test("aab"));
  assertThrows(
    () => compileBoundedRegex("a".repeat(201)),
    Error,
    "maximum length",
  );
  assertThrows(() => compileBoundedRegex("("), Error, "Invalid regex");
});

Deno.test("safeFileName never contains a path separator", () => {
  assertEquals(safeFileName("../../etc/passwd"), "..%2F..%2Fetc%2Fpasswd");
  assert(!safeFileName("a/b\\c").includes("/"));
  assert(safeFileName("plain-key_1.txt") === "plain-key_1.txt");
});
