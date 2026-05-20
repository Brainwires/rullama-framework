import { assert, assertEquals } from "@std/assert";
import { parsePlanSteps, stepsToTasks } from "./plan_parser.ts";

Deno.test("parse numbered steps", () => {
  const content = `
1. Create the user model
2. Add authentication endpoints
3. Implement JWT token handling
`;
  const steps = parsePlanSteps(content);
  assertEquals(steps.length, 3);
  assertEquals(steps[0].description, "Create the user model");
  assertEquals(steps[1].description, "Add authentication endpoints");
});

Deno.test("parse step: colon format", () => {
  const content = `
Step 1: Initialize the project
Step 2: Configure dependencies
`;
  const steps = parsePlanSteps(content);
  assertEquals(steps.length, 2);
  assertEquals(steps[0].description, "Initialize the project");
});

Deno.test("parse indented steps", () => {
  const content = `
1. Setup phase
  1. Install dependencies
  2. Configure environment
2. Implementation phase
`;
  const steps = parsePlanSteps(content);
  assertEquals(steps.length, 4);
});

Deno.test("stepsToTasks preserves plan_id + priorities", () => {
  const steps = parsePlanSteps("1. First step\n2. Second step");
  const tasks = stepsToTasks(steps, "plan-12345678");
  assertEquals(tasks.length, 2);
  assertEquals(tasks[0].plan_id, "plan-12345678");
  assert(tasks[1].depends_on.includes(tasks[0].id));
});

Deno.test("priority detection on ! important critical", () => {
  const steps = parsePlanSteps("1. Important: Fix critical bug!\n2. Normal task");
  assert(steps[0].is_priority);
  assert(!steps[1].is_priority);
});
