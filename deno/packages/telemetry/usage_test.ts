import { assertEquals } from "@std/assert";
import {
  agentIdOf,
  costUsdOf,
  kindOf,
  tokensEvent,
  toolCallEvent,
} from "./usage.ts";

Deno.test("tokens constructor sets fields", () => {
  const e = tokensEvent("agent-1", "openai/gpt-4o", 500, 0.005);
  assertEquals(agentIdOf(e), "agent-1");
  assertEquals(costUsdOf(e), 0.005);
  assertEquals(kindOf(e), "tokens");
});

Deno.test("tool_call has zero cost by default", () => {
  const e = toolCallEvent("agent-1", "bash");
  assertEquals(costUsdOf(e), 0);
  assertEquals(kindOf(e), "tool_call");
});

Deno.test("serde roundtrip", () => {
  const e = tokensEvent("a", "model", 100, 0.001);
  const json = JSON.stringify(e);
  const back = JSON.parse(json);
  assertEquals(back.kind, "tokens");
  assertEquals(back.agent_id, "a");
});
