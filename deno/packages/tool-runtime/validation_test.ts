import { assertEquals } from "@std/assert";
import {
  extractExportName,
  isExportLine,
  ValidationTool,
} from "./validation.ts";
import { ToolContext } from "@rullama/core";

// ── Unit: isExportLine ──────────────────────────────────────────────────────

Deno.test("isExportLine - recognises export declarations", () => {
  assertEquals(isExportLine("export const FOO = 'bar'"), true);
  assertEquals(isExportLine("export function myFunc() {"), true);
  assertEquals(isExportLine("export interface MyInterface {"), true);
  assertEquals(isExportLine("export type MyType = string"), true);
  assertEquals(isExportLine("export class Foo {"), true);
  assertEquals(isExportLine("export enum Color {"), true);
  assertEquals(isExportLine("export async function run() {"), true);
  assertEquals(isExportLine("export default class Foo {"), true);
});

Deno.test("isExportLine - rejects non-export lines", () => {
  assertEquals(isExportLine("const FOO = 'bar'"), false);
  assertEquals(isExportLine("// export const X = 1"), false);
  assertEquals(isExportLine("import { Foo } from 'bar'"), false);
  assertEquals(isExportLine(""), false);
});

// ── Unit: extractExportName ─────────────────────────────────────────────────

Deno.test("extractExportName - extracts names from various exports", () => {
  assertEquals(extractExportName("export const FOO = 'bar'"), "FOO");
  assertEquals(extractExportName("export function myFunc() {"), "myFunc");
  assertEquals(
    extractExportName("export interface MyInterface {"),
    "MyInterface",
  );
  assertEquals(extractExportName("export type MyType = string"), "MyType");
  assertEquals(extractExportName("export class Widget {"), "Widget");
  assertEquals(extractExportName("export enum Color {"), "Color");
  assertEquals(
    extractExportName("export async function run() {"),
    "run",
  );
  assertEquals(extractExportName("export let count = 0"), "count");
  assertEquals(extractExportName("export var x = 1"), "x");
});

Deno.test("extractExportName - returns null for non-export lines", () => {
  assertEquals(extractExportName("const FOO = 'bar'"), null);
  assertEquals(extractExportName(""), null);
});

// ── ValidationTool.getTools ─────────────────────────────────────────────────

Deno.test("ValidationTool - getTools returns 3 tools", () => {
  const tools = ValidationTool.getTools();
  assertEquals(tools.length, 3);
  const names = tools.map((t) => t.name);
  assertEquals(names.includes("check_duplicates"), true);
  assertEquals(names.includes("verify_build"), true);
  assertEquals(names.includes("check_syntax"), true);
});

// ── check_duplicates ────────────────────────────────────────────────────────

Deno.test("check_duplicates - detects duplicate exports", async () => {
  const tmpDir = await Deno.makeTempDir();
  const filePath = `${tmpDir}/dup.ts`;
  await Deno.writeTextFile(
    filePath,
    [
      "export const FOO = 'bar'",
      "export const BAZ = 'qux'",
      "export const FOO = 'dup'",
    ].join("\n"),
  );

  const context = new ToolContext({ working_directory: tmpDir });
  const result = await ValidationTool.execute(
    "t1",
    "check_duplicates",
    { file_path: filePath },
    context,
  );

  assertEquals(result.is_error, false);
  const parsed = JSON.parse(result.content);
  assertEquals(parsed.has_duplicates, true);
  assertEquals(parsed.duplicate_count, 1);
  assertEquals(parsed.duplicates[0].name, "FOO");

  await Deno.remove(tmpDir, { recursive: true });
});

Deno.test("check_duplicates - no duplicates in clean file", async () => {
  const tmpDir = await Deno.makeTempDir();
  const filePath = `${tmpDir}/clean.ts`;
  await Deno.writeTextFile(
    filePath,
    [
      "export const A = 1",
      "export const B = 2",
      "export function doStuff() {}",
    ].join("\n"),
  );

  const context = new ToolContext({ working_directory: tmpDir });
  const result = await ValidationTool.execute(
    "t2",
    "check_duplicates",
    { file_path: filePath },
    context,
  );

  assertEquals(result.is_error, false);
  const parsed = JSON.parse(result.content);
  assertEquals(parsed.has_duplicates, false);
  assertEquals(parsed.total_exports, 3);

  await Deno.remove(tmpDir, { recursive: true });
});

Deno.test("check_duplicates - empty path returns error", async () => {
  const context = new ToolContext();
  const result = await ValidationTool.execute(
    "t3",
    "check_duplicates",
    { file_path: "" },
    context,
  );
  assertEquals(result.is_error, true);
});

// ── check_syntax ────────────────────────────────────────────────────────────

Deno.test("check_syntax - valid TypeScript file", async () => {
  const tmpDir = await Deno.makeTempDir();
  const filePath = `${tmpDir}/valid.ts`;
  await Deno.writeTextFile(filePath, "export const X = { a: 1 };\n");

  const context = new ToolContext({ working_directory: tmpDir });
  const result = await ValidationTool.execute(
    "t4",
    "check_syntax",
    { file_path: filePath },
    context,
  );

  assertEquals(result.is_error, false);
  const parsed = JSON.parse(result.content);
  assertEquals(parsed.valid_syntax, true);

  await Deno.remove(tmpDir, { recursive: true });
});

Deno.test("check_syntax - detects unmatched braces", async () => {
  const tmpDir = await Deno.makeTempDir();
  const filePath = `${tmpDir}/bad.ts`;
  await Deno.writeTextFile(filePath, "export function foo() {\n  return 1;\n");

  const context = new ToolContext({ working_directory: tmpDir });
  const result = await ValidationTool.execute(
    "t5",
    "check_syntax",
    { file_path: filePath },
    context,
  );

  assertEquals(result.is_error, true);
  const parsed = JSON.parse(result.content);
  assertEquals(parsed.valid_syntax, false);
  assertEquals(parsed.errors.length > 0, true);

  await Deno.remove(tmpDir, { recursive: true });
});

Deno.test("check_syntax - detects duplicate export keyword", async () => {
  const tmpDir = await Deno.makeTempDir();
  const filePath = `${tmpDir}/dup_kw.ts`;
  await Deno.writeTextFile(filePath, "export export const X = 1;\n");

  const context = new ToolContext({ working_directory: tmpDir });
  const result = await ValidationTool.execute(
    "t6",
    "check_syntax",
    { file_path: filePath },
    context,
  );

  assertEquals(result.is_error, true);
  const parsed = JSON.parse(result.content);
  assertEquals(parsed.valid_syntax, false);

  await Deno.remove(tmpDir, { recursive: true });
});

Deno.test("check_syntax - unsupported file type returns error", async () => {
  const tmpDir = await Deno.makeTempDir();
  const filePath = `${tmpDir}/foo.xyz`;
  await Deno.writeTextFile(filePath, "content");

  const context = new ToolContext({ working_directory: tmpDir });
  const result = await ValidationTool.execute(
    "t7",
    "check_syntax",
    { file_path: filePath },
    context,
  );

  assertEquals(result.is_error, true);

  await Deno.remove(tmpDir, { recursive: true });
});

// ── unknown tool ────────────────────────────────────────────────────────────

Deno.test("ValidationTool - unknown tool returns error", async () => {
  const context = new ToolContext();
  const result = await ValidationTool.execute(
    "t8",
    "not_a_tool",
    {},
    context,
  );
  assertEquals(result.is_error, true);
  assertEquals(result.content.includes("Unknown validation tool"), true);
});
