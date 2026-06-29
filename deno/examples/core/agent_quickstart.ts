// Example: Agent quickstart — tasks, lifecycle hooks, and working set
// Shows how to set up core agent infrastructure with tasks, hooks, and context management.
// Run: deno run deno/examples/core/agent_quickstart.ts

import {
  estimateTokens,
  HookRegistry,
  type HookResult,
  type LifecycleEvent,
  type LifecycleHook,
  Task,
  WorkingSet,
} from "@rullama/core";

async function main() {
  console.log("=== Agent Infrastructure Quickstart ===");

  // 1. Create and manage tasks
  console.log("\n=== Task Management ===");

  const parentTask = new Task("task-1", "Analyze authentication module");
  parentTask.setPriority("high");
  parentTask.start();
  console.log(`Created task: ${parentTask.id} — ${parentTask.description}`);
  console.log(
    `  Status: ${parentTask.status}, Priority: ${parentTask.priority}`,
  );

  // Create subtasks
  const subtask1 = Task.newSubtask(
    "task-1a",
    "Review auth middleware",
    "task-1",
  );
  const subtask2 = Task.newSubtask(
    "task-1b",
    "Check token validation",
    "task-1",
  );
  parentTask.addChild(subtask1.id);
  parentTask.addChild(subtask2.id);

  subtask1.start();
  subtask1.incrementIteration();
  subtask1.complete("Auth middleware uses JWT with proper expiry checks");
  console.log(
    `  Subtask ${subtask1.id}: ${subtask1.status} — ${subtask1.summary}`,
  );

  subtask2.start();
  subtask2.incrementIteration();
  subtask2.incrementIteration();
  subtask2.complete("Token validation handles refresh tokens correctly");
  console.log(
    `  Subtask ${subtask2.id}: ${subtask2.status} (${subtask2.iterations} iterations)`,
  );

  parentTask.complete("Authentication module analysis complete");
  console.log(`  Parent task: ${parentTask.status}`);
  console.log(
    `  Has children: ${parentTask.hasChildren()}, Is root: ${parentTask.isRoot()}`,
  );

  // 2. Create a plan-associated task with dependencies
  console.log("\n=== Task Dependencies ===");

  const planTask = Task.newForPlan(
    "task-2",
    "Refactor auth module",
    "plan-001",
  );
  const depTask = new Task("task-3", "Write tests for refactored auth");
  depTask.addDependency(planTask.id);

  console.log(
    `Task ${depTask.id} depends on: [${depTask.depends_on.join(", ")}]`,
  );
  console.log(
    `Task ${depTask.id} has dependencies: ${depTask.hasDependencies()}`,
  );

  // 3. Set up lifecycle hooks for event monitoring
  console.log("\n=== Lifecycle Hooks ===");

  const registry = new HookRegistry();

  // A logging hook that watches all events
  const loggingHook: LifecycleHook = {
    name: "logger",
    priority: () => 0,
    async onEvent(event: LifecycleEvent): Promise<HookResult> {
      console.log(`  [hook:logger] Event: ${event.type}`);
      return { type: "continue" };
    },
  };

  // A guard hook that can cancel dangerous tool executions
  const guardHook: LifecycleHook = {
    name: "safety-guard",
    priority: () => 10,
    async onEvent(event: LifecycleEvent): Promise<HookResult> {
      if (event.type === "tool_before_execute" && event.tool_name === "rm_rf") {
        console.log(
          `  [hook:guard] BLOCKED dangerous tool: ${event.tool_name}`,
        );
        return {
          type: "cancel",
          reason: "Dangerous tool blocked by safety guard",
        };
      }
      return { type: "continue" };
    },
  };

  registry.register(loggingHook);
  registry.register(guardHook);
  console.log(`Registered ${registry.length} hooks`);

  // Dispatch some events
  const startEvent: LifecycleEvent = {
    type: "agent_started",
    agent_id: "agent-1",
    task_description: "Analyze auth module",
  };
  const result1 = await registry.dispatch(startEvent);
  console.log(`  Dispatch result: ${result1.type}`);

  const toolEvent: LifecycleEvent = {
    type: "tool_before_execute",
    agent_id: "agent-1",
    tool_name: "read_file",
    args: { path: "src/auth.rs" },
  };
  const result2 = await registry.dispatch(toolEvent);
  console.log(`  Dispatch result: ${result2.type}`);

  const dangerEvent: LifecycleEvent = {
    type: "tool_before_execute",
    agent_id: "agent-1",
    tool_name: "rm_rf",
    args: { path: "/" },
  };
  const result3 = await registry.dispatch(dangerEvent);
  console.log(
    `  Dispatch result: ${result3.type}${
      result3.type === "cancel" ? ` (${result3.reason})` : ""
    }`,
  );

  // 4. Manage the working set (files in agent context)
  console.log("\n=== Working Set ===");

  const ws = new WorkingSet({
    max_files: 5,
    max_tokens: 5000,
    stale_after_turns: 3,
    auto_evict: true,
  });

  // Add files the agent is working with
  ws.add("src/auth.rs", estimateTokens("fn authenticate() { ... }".repeat(20)));
  ws.addPinned("Cargo.toml", estimateTokens('[package]\nname = "myapp"'));
  ws.addLabeled(
    "src/main.rs",
    estimateTokens("fn main() { ... }".repeat(10)),
    "entry-point",
  );

  console.log(`Files in working set: ${ws.length}`);
  console.log(`Total tokens: ${ws.totalTokens()}`);
  console.log(`Contains src/auth.rs: ${ws.contains("src/auth.rs")}`);

  // Advance turns — stale files get evicted automatically
  for (let i = 0; i < 4; i++) {
    ws.nextTurn();
  }
  // Touch auth.rs to keep it fresh
  ws.touch("src/auth.rs");

  console.log(`\nAfter 4 turns (touched auth.rs):`);
  console.log(`Files remaining: ${ws.length}`);
  for (const entry of ws.allEntries()) {
    console.log(
      `  ${entry.path} — ${entry.tokens} tokens, pinned=${entry.pinned}${
        entry.label ? `, label=${entry.label}` : ""
      }`,
    );
  }

  if (ws.lastEviction) {
    console.log(`Last eviction: ${ws.lastEviction}`);
  }

  console.log("\nAgent infrastructure ready!");
  console.log("  - Task: hierarchical task tracking with dependencies");
  console.log("  - HookRegistry: lifecycle event monitoring and guards");
  console.log("  - WorkingSet: automatic context management with LRU eviction");
}

await main();
