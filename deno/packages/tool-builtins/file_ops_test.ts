import {
  assert,
  assertEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { join } from "@std/path";
import { ToolContext } from "@rullama/core";
import { PathEscapeError } from "@rullama/tool-runtime";
import { FileOpsTool, MAX_READ_BYTES } from "./file_ops.ts";

async function withRoot(fn: (root: string, ctx: ToolContext) => Promise<void>) {
  const root = await Deno.realPath(await Deno.makeTempDir());
  try {
    await fn(root, new ToolContext({ working_directory: root }));
  } finally {
    await Deno.remove(root, { recursive: true });
  }
}

Deno.test("resolvePath confines to the working directory", () => {
  const ctx = new ToolContext({ working_directory: "/work" });
  assertEquals(FileOpsTool.resolvePath("a/b.txt", ctx), "/work/a/b.txt");
  assertEquals(FileOpsTool.resolvePath("/work/a", ctx), "/work/a");
  assertThrows(
    () => FileOpsTool.resolvePath("../etc/passwd", ctx),
    PathEscapeError,
  );
  assertThrows(
    () => FileOpsTool.resolvePath("/etc/passwd", ctx),
    PathEscapeError,
  );
});

Deno.test("read/write/delete refuse paths outside the root, including via symlink", async () => {
  await withRoot(async (root, ctx) => {
    const outside = await Deno.makeTempDir();
    try {
      await Deno.writeTextFile(join(outside, "secret.txt"), "top secret");
      await Deno.symlink(join(outside, "secret.txt"), join(root, "link.txt"));
      for (
        const [tool, input] of [
          ["read_file", { path: "../secret.txt" }],
          ["read_file", { path: "link.txt" }],
          ["write_file", { path: `${outside}/x.txt`, content: "x" }],
          ["delete_file", { path: "link.txt" }],
          ["list_directory", { path: ".." }],
        ] as const
      ) {
        const res = await FileOpsTool.execute("1", tool, input, ctx);
        assert(res.is_error, `${tool} should fail`);
        assertStringIncludes(res.content, "outside the working directory");
      }
      assertEquals(
        await Deno.readTextFile(join(outside, "secret.txt")),
        "top secret",
      );
    } finally {
      await Deno.remove(outside, { recursive: true });
    }
  });
});

Deno.test("write then read inside the root works; large reads are truncated", async () => {
  await withRoot(async (root, ctx) => {
    const w = await FileOpsTool.execute("1", "write_file", {
      path: "sub/a.txt",
      content: "hello",
    }, ctx);
    assert(!w.is_error, w.content);
    const r = await FileOpsTool.execute(
      "2",
      "read_file",
      { path: "sub/a.txt" },
      ctx,
    );
    assertStringIncludes(r.content, "hello");
    await Deno.writeTextFile(
      join(root, "big.txt"),
      "y".repeat(MAX_READ_BYTES + 10),
    );
    const big = await FileOpsTool.execute(
      "3",
      "read_file",
      { path: "big.txt" },
      ctx,
    );
    assert(!big.is_error);
    assertStringIncludes(big.content, "[truncated at");
    assert(big.content.length < MAX_READ_BYTES + 200);
  });
});

Deno.test("search_files treats the pattern as a glob, not a regex", async () => {
  await withRoot(async (root, ctx) => {
    await Deno.writeTextFile(join(root, "a+b.ts"), "");
    await Deno.writeTextFile(join(root, "ab.ts"), "");
    await Deno.writeTextFile(join(root, "aab.ts"), "");
    const res = await FileOpsTool.execute("1", "search_files", {
      path: ".",
      pattern: "a+b.ts",
    }, ctx);
    assertStringIncludes(res.content, "Matches: 1");
    assertStringIncludes(res.content, "a+b.ts");
    const star = await FileOpsTool.execute("2", "search_files", {
      path: ".",
      pattern: "a*b.ts",
    }, ctx);
    assertStringIncludes(star.content, "Matches: 3");
  });
});
