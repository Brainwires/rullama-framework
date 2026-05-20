import { assert } from "@std/assert";
import { buildAgentPrompt } from "./mod.ts";

Deno.test("all variants build without throwing", () => {
  buildAgentPrompt({ kind: "reasoning", agent_id: "a", working_directory: "/tmp" });
  buildAgentPrompt({
    kind: "planner",
    agent_id: "a",
    working_directory: "/tmp",
    goal: "do something",
    hints: [],
  });
  buildAgentPrompt({ kind: "judge", agent_id: "a", working_directory: "/tmp" });
  buildAgentPrompt({ kind: "simple", agent_id: "a", working_directory: "/tmp" });
  buildAgentPrompt({
    kind: "mdap_microagent",
    agent_id: "a",
    working_directory: "/tmp",
    vote_round: 1,
    peer_count: 3,
  });
});

Deno.test("no role does not append suffix", () => {
  const p = buildAgentPrompt({
    kind: "reasoning",
    agent_id: "a",
    working_directory: "/tmp",
  });
  assert(!p.includes("[ROLE:"));
});

Deno.test("role suffix is appended", () => {
  const p = buildAgentPrompt(
    { kind: "reasoning", agent_id: "a", working_directory: "/tmp" },
    "exploration",
  );
  assert(p.includes("[ROLE: Exploration]"));
});

Deno.test("planner embeds goal", () => {
  const p = buildAgentPrompt({
    kind: "planner",
    agent_id: "a",
    working_directory: "/tmp",
    goal: "implement LRU cache",
    hints: [],
  });
  assert(p.includes("implement LRU cache"));
});

Deno.test("planner with hints lists them", () => {
  const p = buildAgentPrompt({
    kind: "planner",
    agent_id: "a",
    working_directory: "/tmp",
    goal: "goal",
    hints: ["first hint", "second hint"],
  });
  assert(p.includes("HINTS FROM PREVIOUS CYCLES"));
  assert(p.includes("1. first hint"));
  assert(p.includes("2. second hint"));
});

Deno.test("mdap embeds vote round and peer count", () => {
  const p = buildAgentPrompt({
    kind: "mdap_microagent",
    agent_id: "a",
    working_directory: "/tmp",
    vote_round: 2,
    peer_count: 5,
  });
  assert(p.includes("round 2 of 5"));
});

Deno.test("simple is shorter than reasoning", () => {
  const simple = buildAgentPrompt({
    kind: "simple",
    agent_id: "a",
    working_directory: "/tmp",
  });
  const reasoning = buildAgentPrompt({
    kind: "reasoning",
    agent_id: "a",
    working_directory: "/tmp",
  });
  assert(simple.length < reasoning.length);
});
