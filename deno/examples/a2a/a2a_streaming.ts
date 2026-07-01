// Example: A2A Streaming Types
// Demonstrates building and serializing the core streaming types:
// TaskStatusUpdateEvent, TaskArtifactUpdateEvent, and StreamResponse.
// This example works entirely with in-memory data; no running server is needed.
// Run: deno run deno/examples/a2a/a2a_streaming.ts

import type {
  Artifact,
  Message,
  StreamResponse,
  Task,
  TaskArtifactUpdateEvent,
  TaskStatus,
  TaskStatusUpdateEvent,
} from "@rullama/a2a";
import {
  createAgentMessage,
  isArtifactUpdate,
  isMessageResponse,
  isStatusUpdate,
  isTaskResponse,
} from "@rullama/a2a";

async function main(): Promise<void> {
  console.log("=== A2A Streaming Types Example ===\n");

  const taskId = crypto.randomUUID();
  const contextId = crypto.randomUUID();

  // -----------------------------------------------------------------------
  // 1. TaskStatusUpdateEvent -- submitted
  // -----------------------------------------------------------------------
  console.log("--- TaskStatusUpdateEvent (submitted) ---");

  const submittedEvent: TaskStatusUpdateEvent = {
    taskId,
    contextId,
    status: {
      state: "TASK_STATE_SUBMITTED",
      timestamp: new Date().toISOString(),
    },
  };

  console.log(JSON.stringify(submittedEvent, null, 2));
  console.log();

  // -----------------------------------------------------------------------
  // 2. TaskStatusUpdateEvent -- working (with message)
  // -----------------------------------------------------------------------
  console.log("--- TaskStatusUpdateEvent (working) ---");

  const workingMessage = createAgentMessage("Processing your request...");

  const workingEvent: TaskStatusUpdateEvent = {
    taskId,
    contextId,
    status: {
      state: "TASK_STATE_WORKING",
      message: workingMessage,
      timestamp: new Date().toISOString(),
    },
  };

  console.log(JSON.stringify(workingEvent, null, 2));
  console.log();

  // -----------------------------------------------------------------------
  // 3. TaskArtifactUpdateEvent -- first chunk
  // -----------------------------------------------------------------------
  console.log("--- TaskArtifactUpdateEvent (chunk 1) ---");

  const artifactEvent1: TaskArtifactUpdateEvent = {
    taskId,
    contextId,
    artifact: {
      artifactId: "report-001",
      name: "analysis-report",
      description: "Code analysis report",
      parts: [
        {
          text: "## Code Analysis\n\nAnalyzing module structure...",
          mediaType: "text/markdown",
          filename: "report.md",
        },
      ],
    },
    index: 0,
    append: false,
    lastChunk: false,
  };

  console.log(JSON.stringify(artifactEvent1, null, 2));
  console.log();

  // -----------------------------------------------------------------------
  // 4. TaskArtifactUpdateEvent -- final chunk (append)
  // -----------------------------------------------------------------------
  console.log("--- TaskArtifactUpdateEvent (chunk 2, final) ---");

  const artifactEvent2: TaskArtifactUpdateEvent = {
    taskId,
    contextId,
    artifact: {
      artifactId: "report-001",
      parts: [
        {
          text: "\n### Summary\n\nAll checks passed. No issues found.",
          mediaType: "text/markdown",
        },
      ],
    },
    index: 0,
    append: true,
    lastChunk: true,
  };

  console.log(JSON.stringify(artifactEvent2, null, 2));
  console.log();

  // -----------------------------------------------------------------------
  // 5. StreamResponse -- wrapping a status update
  // -----------------------------------------------------------------------
  console.log("--- StreamResponse (status update) ---");

  const statusStream: StreamResponse = {
    statusUpdate: workingEvent,
  };

  console.log(JSON.stringify(statusStream, null, 2));
  console.log();

  // -----------------------------------------------------------------------
  // 6. StreamResponse -- wrapping an artifact update
  // -----------------------------------------------------------------------
  console.log("--- StreamResponse (artifact update) ---");

  const artifactStream: StreamResponse = {
    artifactUpdate: artifactEvent1,
  };

  console.log(JSON.stringify(artifactStream, null, 2));
  console.log();

  // -----------------------------------------------------------------------
  // 7. StreamResponse -- wrapping a standalone message
  // -----------------------------------------------------------------------
  console.log("--- StreamResponse (agent message) ---");

  const agentMsg: Message = {
    messageId: crypto.randomUUID(),
    role: "ROLE_AGENT",
    parts: [{ text: "Here is an intermediate status update." }],
    contextId,
    taskId,
  };

  const messageStream: StreamResponse = {
    message: agentMsg,
  };

  console.log(JSON.stringify(messageStream, null, 2));
  console.log();

  // -----------------------------------------------------------------------
  // 8. StreamResponse -- full task snapshot (final event)
  // -----------------------------------------------------------------------
  console.log("--- StreamResponse (full task snapshot) ---");

  const completedStatus: TaskStatus = {
    state: "TASK_STATE_COMPLETED",
    message: createAgentMessage("Analysis complete."),
    timestamp: new Date().toISOString(),
  };

  const fullTask: Task = {
    id: taskId,
    contextId,
    status: completedStatus,
    artifacts: [
      {
        artifactId: "report-001",
        name: "analysis-report",
        description: "Code analysis report",
        parts: [
          {
            text: "## Code Analysis\n\n...full report...",
            mediaType: "text/markdown",
            filename: "report.md",
          },
        ],
      },
    ],
  };

  const taskStream: StreamResponse = {
    task: fullTask,
  };

  console.log(JSON.stringify(taskStream, null, 2));
  console.log();

  // -----------------------------------------------------------------------
  // 9. Type guards and round-trip verification
  // -----------------------------------------------------------------------
  console.log("--- Type Guards & Round-Trip Verification ---");

  const events: StreamResponse[] = [
    statusStream,
    artifactStream,
    messageStream,
    taskStream,
  ];

  const labels = [
    "status update",
    "artifact update",
    "message",
    "task snapshot",
  ];

  for (let i = 0; i < events.length; i++) {
    const event = events[i];
    const serialized = JSON.stringify(event);
    const deserialized: StreamResponse = JSON.parse(serialized);

    // Verify type guards
    const guards = [
      isStatusUpdate(event) ? "statusUpdate" : null,
      isArtifactUpdate(event) ? "artifactUpdate" : null,
      isMessageResponse(event) ? "message" : null,
      isTaskResponse(event) ? "task" : null,
    ].filter(Boolean);

    // Verify round-trip
    const roundTripMatch =
      JSON.stringify(deserialized) === JSON.stringify(event);

    console.log(
      `  Event ${i} (${labels[i]}): round-trip ${
        roundTripMatch ? "OK" : "FAILED"
      }, guards=[${guards.join(", ")}]`,
    );
  }

  console.log("\nDone.");
}

await main();
