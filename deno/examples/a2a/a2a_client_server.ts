// Example: A2A Client/Server
// Demonstrates the A2A client API for sending messages, getting tasks,
// listing tasks, and canceling tasks using in-memory data (no running server).
// Run: deno run deno/examples/a2a/a2a_client_server.ts

import type {
  AgentCapabilities,
  AgentCard,
  AgentProvider,
  AgentSkill,
  Artifact,
  CancelTaskRequest,
  GetTaskRequest,
  ListTasksRequest,
  ListTasksResponse,
  Message,
  Part,
  SendMessageRequest,
  SendMessageResponse,
  Task,
  TaskState,
  TaskStatus,
} from "@rullama/a2a";
import {
  A2aClient,
  A2aError,
  createAgentMessage,
  createUserMessage,
} from "@rullama/a2a";
import type { A2aHandler } from "@rullama/a2a";

async function main(): Promise<void> {
  console.log("=== A2A Client/Server Example ===\n");

  // -----------------------------------------------------------------------
  // 1. Build an AgentCard for the demo handler
  // -----------------------------------------------------------------------
  console.log("--- 1. Agent Card ---");

  const card: AgentCard = {
    name: "demo-agent",
    description: "A demo A2A agent for the client/server example.",
    version: "0.1.0",
    supportedInterfaces: [],
    capabilities: {
      streaming: false,
      pushNotifications: false,
      extendedAgentCard: false,
    },
    skills: [
      {
        id: "echo",
        name: "Echo",
        description: "Echoes the user message back.",
        tags: ["demo"],
        examples: ["Say hello"],
      },
    ],
    defaultInputModes: ["text/plain"],
    defaultOutputModes: ["text/plain"],
    provider: {
      url: "https://example.com",
      organization: "Demo",
    },
  };

  console.log(`  Agent: ${card.name} v${card.version}`);
  console.log(
    `  Skills: ${card.skills.map((s: { id: string }) => s.id).join(", ")}`,
  );
  console.log(`  Streaming: ${card.capabilities.streaming}`);
  console.log();

  // -----------------------------------------------------------------------
  // 2. Simulate a handler processing a message
  // -----------------------------------------------------------------------
  console.log("--- 2. Simulate Handler ---");

  const tasks = new Map<string, Task>();

  // Simulate on_send_message handler
  const userMsg = createUserMessage("Hello from the A2A client!");
  console.log(`  User message: "${userMsg.parts[0]?.text}"`);

  const taskId = crypto.randomUUID();
  const contextId = userMsg.contextId ?? crypto.randomUUID();

  const agentReply: Message = createAgentMessage(
    `Echo: ${userMsg.parts[0]?.text}`,
  );
  agentReply.contextId = contextId;
  agentReply.taskId = taskId;

  const task: Task = {
    id: taskId,
    contextId,
    status: {
      state: "TASK_STATE_COMPLETED",
      message: agentReply,
      timestamp: new Date().toISOString(),
    },
    artifacts: [
      {
        artifactId: "artifact-1",
        name: "echo-result",
        description: "The echoed message",
        parts: [
          {
            text: `Echo: ${userMsg.parts[0]?.text}`,
            mediaType: "text/plain",
          },
        ],
      },
    ],
  };

  tasks.set(taskId, task);

  const response: SendMessageResponse = { task };

  console.log(`  Response task ID: ${response.task?.id}`);
  console.log(`  Task state:       ${response.task?.status.state}`);
  if (response.task?.status.message) {
    const text = response.task.status.message.parts[0]?.text ?? "(none)";
    console.log(`  Agent reply:      ${text}`);
  }
  if (response.task?.artifacts) {
    console.log(
      `  Artifacts:        ${response.task.artifacts.length} item(s)`,
    );
  }
  console.log();

  // -----------------------------------------------------------------------
  // 3. Get task by ID
  // -----------------------------------------------------------------------
  console.log("--- 3. Get Task ---");

  const fetched = tasks.get(taskId);
  if (fetched) {
    console.log(
      `  Fetched task: ${fetched.id} (state=${fetched.status.state})`,
    );
  } else {
    console.log("  Task not found");
  }
  console.log();

  // -----------------------------------------------------------------------
  // 4. List all tasks
  // -----------------------------------------------------------------------
  console.log("--- 4. List Tasks ---");

  const allTasks = [...tasks.values()];
  console.log(`  Total tasks: ${allTasks.length}`);
  for (const t of allTasks) {
    console.log(`    ${t.id} -- ${t.status.state}`);
  }
  console.log();

  // -----------------------------------------------------------------------
  // 5. Cancel a task
  // -----------------------------------------------------------------------
  console.log("--- 5. Cancel Task ---");

  const toCancel = tasks.get(taskId);
  if (toCancel) {
    toCancel.status.state = "TASK_STATE_CANCELED";
    toCancel.status.timestamp = new Date().toISOString();
    console.log(
      `  Canceled task: ${toCancel.id} (state=${toCancel.status.state})`,
    );
  }
  console.log();

  // -----------------------------------------------------------------------
  // 6. Demonstrate A2aClient construction
  // -----------------------------------------------------------------------
  console.log("--- 6. A2aClient API Overview ---");

  const client = new A2aClient({ baseUrl: "http://localhost:8080" });
  console.log("  Created A2aClient (JSON-RPC transport, no live server)");
  console.log();

  console.log("  Available client methods:");
  console.log("    client.sendMessage(req)           -- send a message");
  console.log(
    "    client.streamMessage(req)         -- stream a message (SSE)",
  );
  console.log("    client.getTask(req)               -- get task by ID");
  console.log("    client.listTasks(req)             -- list tasks");
  console.log("    client.cancelTask(req)            -- cancel a task");
  console.log(
    "    client.subscribeToTask(req)       -- subscribe to task updates",
  );
  console.log("    A2aClient.discover(baseUrl)       -- discover agent card");
  console.log();

  // -----------------------------------------------------------------------
  // 7. Demonstrate A2aHandler interface
  // -----------------------------------------------------------------------
  console.log("--- 7. A2aHandler Interface ---");

  console.log("  The A2aHandler interface defines:");
  console.log("    agentCard()                       -- return agent card");
  console.log("    onSendMessage(req)                -- handle messages");
  console.log("    onSendStreamingMessage(req)       -- handle streaming");
  console.log("    onGetTask(req)                    -- handle get task");
  console.log("    onListTasks(req)                  -- handle list tasks");
  console.log("    onCancelTask(req)                 -- handle cancel");
  console.log("    onSubscribeToTask(req)            -- handle subscriptions");
  console.log();

  // -----------------------------------------------------------------------
  // 8. Error handling
  // -----------------------------------------------------------------------
  console.log("--- 8. Error Handling ---");

  const notFoundErr = A2aError.taskNotFound("missing-task-id");
  console.log(
    `  Task not found error: code=${notFoundErr.code}, message="${notFoundErr.message}"`,
  );

  const unsupErr = A2aError.unsupportedOperation("streaming not enabled");
  console.log(
    `  Unsupported error:    code=${unsupErr.code}, message="${unsupErr.message}"`,
  );

  const jsonErr = notFoundErr.toJSON();
  console.log(`  Serialized to JSON:   ${JSON.stringify(jsonErr)}`);

  const roundTripped = A2aError.fromJSON(jsonErr);
  console.log(
    `  Round-tripped:        code=${roundTripped.code}, message="${roundTripped.message}"`,
  );

  console.log("\nDone.");
}

await main();
