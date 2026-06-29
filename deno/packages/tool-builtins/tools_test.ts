import { assertEquals } from "@std/assert";
import { FileOpsTool } from "./file_ops.ts";
import { GitTool } from "./git.ts";
import { SearchTool } from "./search.ts";
import { WebTool } from "./web.ts";
import { ToolContext } from "@rullama/core";

// ---- FileOpsTool ----

Deno.test("FileOpsTool - getTools returns 7 tools", () => {
  const tools = FileOpsTool.getTools();
  assertEquals(tools.length, 7);
  const names = tools.map((t) => t.name);
  assertEquals(names.includes("read_file"), true);
  assertEquals(names.includes("write_file"), true);
  assertEquals(names.includes("edit_file"), true);
  assertEquals(names.includes("list_directory"), true);
});

Deno.test("FileOpsTool - read/write/edit roundtrip", async () => {
  const tmpDir = await Deno.makeTempDir();
  const context = new ToolContext({ working_directory: tmpDir });

  // Write
  const writeResult = await FileOpsTool.execute(
    "1",
    "write_file",
    { path: "test.txt", content: "Hello World! Hello World!" },
    context,
  );
  assertEquals(writeResult.is_error, false);

  // Read
  const readResult = await FileOpsTool.execute(
    "2",
    "read_file",
    { path: "test.txt" },
    context,
  );
  assertEquals(readResult.is_error, false);
  assertEquals(readResult.content.includes("Hello World!"), true);

  // Edit (first occurrence only)
  const editResult = await FileOpsTool.execute(
    "3",
    "edit_file",
    { path: "test.txt", old_text: "World", new_text: "Rust" },
    context,
  );
  assertEquals(editResult.is_error, false);

  // Verify edit
  const content = await Deno.readTextFile(`${tmpDir}/test.txt`);
  assertEquals(content, "Hello Rust! Hello World!");

  await Deno.remove(tmpDir, { recursive: true });
});

Deno.test("FileOpsTool - list directory", async () => {
  const tmpDir = await Deno.makeTempDir();
  await Deno.writeTextFile(`${tmpDir}/a.txt`, "");
  await Deno.writeTextFile(`${tmpDir}/b.txt`, "");
  const context = new ToolContext({ working_directory: tmpDir });

  const result = await FileOpsTool.execute(
    "4",
    "list_directory",
    { path: ".", recursive: false },
    context,
  );
  assertEquals(result.is_error, false);
  assertEquals(result.content.includes("a.txt"), true);
  assertEquals(result.content.includes("b.txt"), true);

  await Deno.remove(tmpDir, { recursive: true });
});

Deno.test("FileOpsTool - delete file", async () => {
  const tmpDir = await Deno.makeTempDir();
  await Deno.writeTextFile(`${tmpDir}/del.txt`, "");
  const context = new ToolContext({ working_directory: tmpDir });

  const result = await FileOpsTool.execute(
    "5",
    "delete_file",
    { path: "del.txt" },
    context,
  );
  assertEquals(result.is_error, false);

  try {
    await Deno.stat(`${tmpDir}/del.txt`);
    throw new Error("File should not exist");
  } catch (e) {
    assertEquals(e instanceof Deno.errors.NotFound, true);
  }

  await Deno.remove(tmpDir, { recursive: true });
});

Deno.test("FileOpsTool - create directory", async () => {
  const tmpDir = await Deno.makeTempDir();
  const context = new ToolContext({ working_directory: tmpDir });

  const result = await FileOpsTool.execute(
    "6",
    "create_directory",
    { path: "sub/dir" },
    context,
  );
  assertEquals(result.is_error, false);

  const stat = await Deno.stat(`${tmpDir}/sub/dir`);
  assertEquals(stat.isDirectory, true);

  await Deno.remove(tmpDir, { recursive: true });
});

Deno.test("FileOpsTool - unknown tool", async () => {
  const context = new ToolContext({ working_directory: Deno.cwd() });
  const result = await FileOpsTool.execute(
    "7",
    "nonexistent",
    {},
    context,
  );
  assertEquals(result.is_error, true);
});

// ---- GitTool ----

Deno.test("GitTool - getTools returns 11 tools", () => {
  const tools = GitTool.getTools();
  assertEquals(tools.length, 11);
  const names = tools.map((t) => t.name);
  assertEquals(names.includes("git_status"), true);
  assertEquals(names.includes("git_commit"), true);
  assertEquals(names.includes("git_branch"), true);
});

Deno.test("GitTool - unknown tool", async () => {
  const context = new ToolContext({ working_directory: Deno.cwd() });
  const result = await GitTool.execute("1", "unknown_tool", {}, context);
  assertEquals(result.is_error, true);
});

// ---- SearchTool ----

Deno.test("SearchTool - getTools returns 1 tool", () => {
  const tools = SearchTool.getTools();
  assertEquals(tools.length, 1);
  assertEquals(tools[0].name, "search_code");
});

Deno.test("SearchTool - unknown tool", async () => {
  const context = new ToolContext({ working_directory: Deno.cwd() });
  const result = await SearchTool.execute(
    "1",
    "unknown_tool",
    { pattern: "test" },
    context,
  );
  assertEquals(result.is_error, true);
});

// ---- WebTool ----

Deno.test("WebTool - getTools returns 1 tool", () => {
  const tools = WebTool.getTools();
  assertEquals(tools.length, 1);
  assertEquals(tools[0].name, "fetch_url");
});

Deno.test("WebTool - unknown tool", async () => {
  const context = new ToolContext({ working_directory: Deno.cwd() });
  const result = await WebTool.execute(
    "1",
    "unknown_tool",
    { url: "https://example.com" },
    context,
  );
  assertEquals(result.is_error, true);
});
