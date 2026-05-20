import { assert, assertEquals } from "@std/assert";
import { ToolContext } from "@brainwires/core";
import { CalendarTool, type CalendarConfig } from "./mod.ts";

Deno.test("getTools exposes the five calendar tools", () => {
  const tools = CalendarTool.getTools();
  assertEquals(tools.length, 5);
  const names = tools.map((t) => t.name);
  assert(names.includes("calendar_list_events"));
  assert(names.includes("calendar_create_event"));
  assert(names.includes("calendar_update_event"));
  assert(names.includes("calendar_delete_event"));
  assert(names.includes("calendar_find_free_time"));
});

Deno.test("create_event requires approval", () => {
  const tool = CalendarTool.getTools().find(
    (t) => t.name === "calendar_create_event",
  )!;
  assert(tool.requires_approval);
});

Deno.test("create_event required fields", () => {
  const tool = CalendarTool.getTools().find(
    (t) => t.name === "calendar_create_event",
  )!;
  const required = tool.input_schema.required ?? [];
  assert(required.includes("title"));
  assert(required.includes("start"));
  assert(required.includes("end"));
});

Deno.test("delete_event requires approval", () => {
  const tool = CalendarTool.getTools().find(
    (t) => t.name === "calendar_delete_event",
  )!;
  assert(tool.requires_approval);
});

Deno.test("find_free_time required fields", () => {
  const tool = CalendarTool.getTools().find(
    (t) => t.name === "calendar_find_free_time",
  )!;
  const required = tool.input_schema.required ?? [];
  assert(required.includes("time_min"));
  assert(required.includes("time_max"));
});

Deno.test("CalendarConfig JSON round-trip", () => {
  const config: CalendarConfig = {
    provider: {
      type: "google_calendar",
      client_id: "id",
      client_secret: "secret",
      refresh_token: "token",
    },
    default_calendar_id: "primary",
  };
  const json = JSON.stringify(config);
  const round = JSON.parse(json) as CalendarConfig;
  assertEquals(round.default_calendar_id, "primary");
});

Deno.test("execute unknown tool returns error", async () => {
  const ctx = new ToolContext();
  const result = await CalendarTool.execute(
    "1",
    "unknown_calendar_tool",
    {},
    ctx,
  );
  assert(result.is_error);
});
