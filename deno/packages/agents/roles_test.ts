import { assert, assertEquals } from "@std/assert";
import type { Tool } from "@brainwires/core";
import { defaultToolInputSchema } from "@brainwires/core";
import {
  allowedTools,
  filterTools,
  roleDisplayName,
  systemPromptSuffix,
} from "./roles.ts";

function fakeTool(name: string): Tool {
  return {
    name,
    description: "",
    input_schema: defaultToolInputSchema(),
  };
}

Deno.test("exploration filters write tools", () => {
  const tools = [
    fakeTool("read_file"),
    fakeTool("write_file"),
    fakeTool("execute_command"),
    fakeTool("glob"),
  ];
  const names = filterTools("exploration", tools).map((t) => t.name);
  assert(names.includes("read_file"));
  assert(names.includes("glob"));
  assert(!names.includes("write_file"));
  assert(!names.includes("execute_command"));
});

Deno.test("execution passes all tools", () => {
  const tools = [fakeTool("read_file"), fakeTool("write_file")];
  assertEquals(filterTools("execution", tools).length, 2);
});

Deno.test("planning allows task tools but not write/execute", () => {
  const tools = [
    fakeTool("read_file"),
    fakeTool("task_create"),
    fakeTool("task_update"),
    fakeTool("plan_task"),
    fakeTool("write_file"),
    fakeTool("execute_command"),
  ];
  const names = filterTools("planning", tools).map((t) => t.name);
  assert(names.includes("task_create"));
  assert(names.includes("plan_task"));
  assert(!names.includes("write_file"));
  assert(!names.includes("execute_command"));
});

Deno.test("verification allows execute_command but not write", () => {
  const tools = [
    fakeTool("read_file"),
    fakeTool("execute_command"),
    fakeTool("verify_build"),
    fakeTool("write_file"),
    fakeTool("task_create"),
  ];
  const names = filterTools("verification", tools).map((t) => t.name);
  assert(names.includes("execute_command"));
  assert(names.includes("verify_build"));
  assert(!names.includes("write_file"));
  assert(!names.includes("task_create"));
});

Deno.test("system prompt suffix non-empty for constrained roles", () => {
  assert(systemPromptSuffix("exploration").length > 0);
  assert(systemPromptSuffix("planning").length > 0);
  assert(systemPromptSuffix("verification").length > 0);
  assertEquals(systemPromptSuffix("execution"), "");
});

Deno.test("allowedTools returns null for execution", () => {
  assertEquals(allowedTools("execution"), null);
});

Deno.test("role display names", () => {
  assertEquals(roleDisplayName("exploration"), "Exploration");
  assertEquals(roleDisplayName("execution"), "Execution");
});
