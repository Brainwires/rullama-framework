/**
 * Tests for the remote bridge subsystem.
 * Covers command queue, heartbeat, protocol types, and bridge state management.
 */

import { assert, assertEquals, assertThrows } from "@std/assert";

import {
  allSupportedCapabilities,
  assessConnectionQuality,
  type BackendCommand,
  type CommandPriority,
  // Command queue
  CommandQueue,
  defaultBridgeConfig,
  defaultProtocolAccept,
  defaultProtocolHello,
  displayBridgeStatus,
  // Heartbeat
  HeartbeatCollector,
  type MetricsSnapshot,
  NegotiatedProtocol,
  type PrioritizedCommand,
  PRIORITY_ORDER,
  // Protocol
  PROTOCOL_VERSION,
  type ProtocolCapability,
  ProtocolMetrics,
  QueueEntry,
  QueueError,
  type RemoteAgentInfo,
  // Bridge
  RemoteBridge,
  // Manager
  RemoteBridgeManager,
  type RemoteMessage,
  type RetryPolicy,
  SUPPORTED_VERSIONS,
} from "./mod.ts";

// ============================================================================
// Protocol Tests
// ============================================================================

Deno.test("protocol version constants", () => {
  assertEquals(PROTOCOL_VERSION, "1.1");
  assert(SUPPORTED_VERSIONS.includes("1.1"));
  assert(SUPPORTED_VERSIONS.includes("1.0"));
});

Deno.test("allSupportedCapabilities returns expected capabilities", () => {
  const caps = allSupportedCapabilities();
  assert(caps.includes("streaming"));
  assert(caps.includes("tools"));
  assert(caps.includes("attachments"));
  assert(caps.includes("priority"));
});

Deno.test("defaultProtocolHello", () => {
  const hello = defaultProtocolHello();
  assertEquals(hello.preferred_version, "1.1");
  assert(hello.supported_versions.length >= 2);
  assert(hello.capabilities.length > 0);
});

Deno.test("defaultProtocolAccept", () => {
  const accept = defaultProtocolAccept();
  assertEquals(accept.selected_version, "1.1");
  assert(accept.enabled_capabilities.includes("streaming"));
  assert(accept.enabled_capabilities.includes("tools"));
});

Deno.test("NegotiatedProtocol - hasCapability", () => {
  const proto = new NegotiatedProtocol("1.1", ["streaming", "compression"]);
  assert(proto.hasCapability("streaming"));
  assert(proto.hasCapability("compression"));
  assert(!proto.hasCapability("attachments"));
});

Deno.test("NegotiatedProtocol - fromAccept", () => {
  const accept = {
    selected_version: "1.1",
    enabled_capabilities: ["streaming", "tools"] as ProtocolCapability[],
  };
  const negotiated = NegotiatedProtocol.fromAccept(accept);
  assertEquals(negotiated.version, "1.1");
  assert(negotiated.hasCapability("streaming"));
  assert(negotiated.hasCapability("tools"));
  assert(!negotiated.hasCapability("attachments"));
});

Deno.test("NegotiatedProtocol - default", () => {
  const proto = NegotiatedProtocol.default();
  assertEquals(proto.version, PROTOCOL_VERSION);
  assert(proto.hasCapability("streaming"));
  assert(proto.hasCapability("tools"));
});

Deno.test("RemoteMessage serialization - register", () => {
  const msg: RemoteMessage = {
    type: "register",
    api_key: "bw_prod_test123",
    hostname: "my-laptop",
    os: "linux",
    version: "0.8.0",
  };
  const json = JSON.stringify(msg);
  assert(json.includes('"type":"register"'));
  assert(json.includes('"api_key":"bw_prod_test123"'));
});

Deno.test("BackendCommand deserialization - authenticated", () => {
  const json =
    '{"type":"authenticated","session_token":"abc123","user_id":"user-456","refresh_interval_secs":30}';
  const cmd = JSON.parse(json) as BackendCommand;
  assertEquals(cmd.type, "authenticated");
  if (cmd.type === "authenticated") {
    assertEquals(cmd.session_token, "abc123");
    assertEquals(cmd.user_id, "user-456");
    assertEquals(cmd.refresh_interval_secs, 30);
  }
});

Deno.test("BackendCommand deserialization - with protocol", () => {
  const json =
    '{"type":"authenticated","session_token":"abc123","user_id":"user-456","refresh_interval_secs":30,"protocol":{"selected_version":"1.1","enabled_capabilities":["streaming","tools"]}}';
  const cmd = JSON.parse(json) as BackendCommand;
  if (cmd.type === "authenticated" && cmd.protocol) {
    assertEquals(cmd.protocol.selected_version, "1.1");
    assert(cmd.protocol.enabled_capabilities.includes("streaming"));
    assert(cmd.protocol.enabled_capabilities.includes("tools"));
  }
});

Deno.test("RemoteAgentInfo serialization", () => {
  const info: RemoteAgentInfo = {
    session_id: "agent-123",
    model: "claude-3-5-sonnet",
    is_busy: false,
    working_directory: "/home/user/project",
    message_count: 5,
    last_activity: 1700000000,
    status: "idle",
    name: "main-agent",
  };
  const json = JSON.stringify(info);
  assert(json.includes('"session_id":"agent-123"'));
  assert(json.includes('"name":"main-agent"'));
});

Deno.test("PRIORITY_ORDER values", () => {
  assert(PRIORITY_ORDER.critical < PRIORITY_ORDER.high);
  assert(PRIORITY_ORDER.high < PRIORITY_ORDER.normal);
  assert(PRIORITY_ORDER.normal < PRIORITY_ORDER.low);
});

// ============================================================================
// Command Queue Tests
// ============================================================================

function makePingCommand(
  priority: CommandPriority,
  timestamp = 0,
): PrioritizedCommand {
  return {
    command: { type: "ping", timestamp } as BackendCommand,
    priority,
  };
}

Deno.test("CommandQueue - priority ordering", () => {
  const queue = new CommandQueue(100);

  queue.enqueue(makePingCommand("low"));
  queue.enqueue(makePingCommand("high"));
  queue.enqueue(makePingCommand("normal"));
  queue.enqueue(makePingCommand("critical"));

  assertEquals(queue.dequeue()!.command.priority, "critical");
  assertEquals(queue.dequeue()!.command.priority, "high");
  assertEquals(queue.dequeue()!.command.priority, "normal");
  assertEquals(queue.dequeue()!.command.priority, "low");
});

Deno.test("CommandQueue - FIFO within same priority", () => {
  const queue = new CommandQueue(100);

  for (let i = 0; i < 5; i++) {
    queue.enqueue(makePingCommand("normal", i));
  }

  for (let i = 0; i < 5; i++) {
    const entry = queue.dequeue()!;
    if (entry.command.command.type === "ping") {
      assertEquals(entry.command.command.timestamp, i);
    }
  }
});

Deno.test("CommandQueue - queue full", () => {
  const queue = new CommandQueue(2);

  queue.enqueue(makePingCommand("normal"));
  queue.enqueue(makePingCommand("normal"));

  // Third normal should fail
  assertThrows(
    () => queue.enqueue(makePingCommand("normal")),
    QueueError,
    "Queue is full",
  );

  // But critical should succeed even when full
  queue.enqueue(makePingCommand("critical"));
  assertEquals(queue.length, 3);
});

Deno.test("CommandQueue - enqueueSimple", () => {
  const queue = new CommandQueue();
  queue.enqueueSimple({ type: "ping", timestamp: 42 });
  assertEquals(queue.length, 1);
  const entry = queue.dequeue()!;
  assertEquals(entry.command.priority, "normal");
});

Deno.test("CommandQueue - peek", () => {
  const queue = new CommandQueue();
  assertEquals(queue.peek(), undefined);

  queue.enqueue(makePingCommand("normal", 1));
  queue.enqueue(makePingCommand("high", 2));

  const peeked = queue.peek()!;
  assertEquals(peeked.command.priority, "high");
  assertEquals(queue.length, 2); // peek doesn't remove
});

Deno.test("CommandQueue - isEmpty", () => {
  const queue = new CommandQueue();
  assert(queue.isEmpty);
  queue.enqueue(makePingCommand("normal"));
  assert(!queue.isEmpty);
});

Deno.test("CommandQueue - stats", () => {
  const queue = new CommandQueue();
  queue.enqueue(makePingCommand("critical"));
  queue.enqueue(makePingCommand("high"));
  queue.enqueue(makePingCommand("normal"));
  queue.enqueue(makePingCommand("normal"));
  queue.enqueue(makePingCommand("low"));

  const stats = queue.stats();
  assertEquals(stats.total, 5);
  assertEquals(stats.critical, 1);
  assertEquals(stats.high, 1);
  assertEquals(stats.normal, 2);
  assertEquals(stats.low, 1);
});

Deno.test("CommandQueue - retry logic", () => {
  const queue = new CommandQueue(100);

  const retryPolicy: RetryPolicy = {
    max_attempts: 3,
    backoff_multiplier: 2.0,
    initial_delay_ms: 100,
  };

  queue.enqueue({
    command: { type: "ping", timestamp: 42 },
    priority: "normal",
    retry_policy: retryPolicy,
  });

  let entry = queue.dequeue()!;
  assert(entry.shouldRetry());

  // Retry 1
  queue.requeueForRetry(entry);
  entry = queue.dequeue()!;
  assertEquals(entry.retryAttempt, 1);
  assert(entry.shouldRetry());

  // Retry 2
  queue.requeueForRetry(entry);
  entry = queue.dequeue()!;
  assertEquals(entry.retryAttempt, 2);
  assert(entry.shouldRetry());

  // Retry 3
  queue.requeueForRetry(entry);
  entry = queue.dequeue()!;
  assertEquals(entry.retryAttempt, 3);
  assert(!entry.shouldRetry()); // Max retries reached

  // Should fail to requeue
  assertThrows(
    () => queue.requeueForRetry(entry),
    QueueError,
    "Maximum retries exceeded",
  );
});

Deno.test("QueueEntry - nextRetryDelay", () => {
  const entry = new QueueEntry(
    {
      command: { type: "ping", timestamp: 0 },
      priority: "normal",
      retry_policy: {
        max_attempts: 3,
        backoff_multiplier: 2.0,
        initial_delay_ms: 100,
      },
    },
    0,
  );

  assertEquals(entry.nextRetryDelay(), 100); // 100 * 2^0

  entry.incrementRetry();
  assertEquals(entry.nextRetryDelay(), 200); // 100 * 2^1

  entry.incrementRetry();
  assertEquals(entry.nextRetryDelay(), 400); // 100 * 2^2

  entry.incrementRetry();
  assertEquals(entry.nextRetryDelay(), undefined); // max retries exceeded
});

Deno.test("QueueEntry - deadline tracking", () => {
  // Entry with no deadline
  const noDeadline = new QueueEntry(makePingCommand("normal"), 0);
  assert(!noDeadline.isExpired());
  assertEquals(noDeadline.timeUntilDeadline(), undefined);

  // Entry with a future deadline
  const future = new QueueEntry(
    { ...makePingCommand("normal"), deadline_ms: 60000 },
    1,
  );
  assert(!future.isExpired());
  const remaining = future.timeUntilDeadline();
  assert(remaining !== undefined && remaining > 0);

  // Entry with a past deadline (0ms = immediate)
  const past = new QueueEntry(
    { ...makePingCommand("normal"), deadline_ms: 0 },
    2,
  );
  // Might or might not be expired immediately due to timing
  // but deadline should be set
  assert(past.deadline !== undefined);
});

// ============================================================================
// Heartbeat Tests
// ============================================================================

Deno.test("HeartbeatCollector - initial state", () => {
  const collector = new HeartbeatCollector({
    version: "0.1.0-test",
    hostname: "test-host",
  });
  assert(!collector.hasAgents());
  assertEquals(collector.agentCount(), 0);
});

Deno.test("HeartbeatCollector - collect with provider", async () => {
  const agents: RemoteAgentInfo[] = [
    {
      session_id: "agent-1",
      model: "claude-3-5-sonnet",
      is_busy: false,
      working_directory: "/tmp",
      message_count: 5,
      last_activity: Date.now(),
      status: "idle",
    },
  ];

  const collector = new HeartbeatCollector({
    version: "0.5.0",
    hostname: "test-host",
    agentInfoProvider: () => agents,
  });

  const data = await collector.collect();
  assertEquals(data.agents.length, 1);
  assertEquals(data.agents[0].session_id, "agent-1");
  assertEquals(data.version, "0.5.0");
  assertEquals(data.hostname, "test-host");
  assert(collector.hasAgents());
  assertEquals(collector.agentCount(), 1);
});

Deno.test("HeartbeatCollector - detect spawned agents", async () => {
  let agents: RemoteAgentInfo[] = [];

  const collector = new HeartbeatCollector({
    version: "0.5.0",
    agentInfoProvider: () => agents,
  });

  // Initial collection (empty)
  await collector.collect();
  assertEquals(collector.agentCount(), 0);

  // Add an agent
  agents = [
    {
      session_id: "agent-new",
      model: "gpt-4",
      is_busy: false,
      working_directory: "/tmp",
      message_count: 0,
      last_activity: Date.now(),
      status: "idle",
    },
  ];

  const events = await collector.detectChanges();
  assert(
    events.some((e) =>
      e.event_type === "spawned" && e.agent_id === "agent-new"
    ),
  );
});

Deno.test("HeartbeatCollector - detect exited agents", async () => {
  const agents: RemoteAgentInfo[] = [
    {
      session_id: "agent-exit",
      model: "gpt-4",
      is_busy: false,
      working_directory: "/tmp",
      message_count: 0,
      last_activity: Date.now(),
      status: "idle",
    },
  ];

  let currentAgents = [...agents];

  const collector = new HeartbeatCollector({
    version: "0.5.0",
    agentInfoProvider: () => currentAgents,
  });

  // Initial collection
  await collector.collect();

  // Remove the agent
  currentAgents = [];

  const events = await collector.detectChanges();
  assert(
    events.some((e) =>
      e.event_type === "exited" && e.agent_id === "agent-exit"
    ),
  );
});

Deno.test("HeartbeatCollector - detect busy/idle state changes", async () => {
  const agent: RemoteAgentInfo = {
    session_id: "agent-state",
    model: "gpt-4",
    is_busy: false,
    working_directory: "/tmp",
    message_count: 0,
    last_activity: Date.now(),
    status: "idle",
  };

  let current = { ...agent };

  const collector = new HeartbeatCollector({
    version: "0.5.0",
    agentInfoProvider: () => [current],
  });

  // Initial collection
  await collector.collect();

  // Agent becomes busy
  current = { ...current, is_busy: true, status: "busy" };

  const events = await collector.detectChanges();
  assert(
    events.some((e) => e.event_type === "busy" && e.agent_id === "agent-state"),
  );
});

Deno.test("HeartbeatCollector - getCurrentAgents", async () => {
  const agents: RemoteAgentInfo[] = [
    {
      session_id: "a1",
      model: "m1",
      is_busy: false,
      working_directory: "/",
      message_count: 0,
      last_activity: 0,
      status: "idle",
    },
  ];

  const collector = new HeartbeatCollector({
    version: "0.1.0",
    agentInfoProvider: () => agents,
  });

  await collector.collect();
  const current = collector.getCurrentAgents();
  assertEquals(current.length, 1);
  assertEquals(current[0].session_id, "a1");
});

// ============================================================================
// Protocol Metrics Tests
// ============================================================================

Deno.test("ProtocolMetrics - recording and snapshot", () => {
  const metrics = new ProtocolMetrics();
  metrics.recordConnectionStart();

  metrics.recordMessageSent(100);
  metrics.recordMessageSent(200);
  metrics.recordMessageFailed();
  metrics.recordBytesReceived(150);

  const snapshot = metrics.snapshot();
  assertEquals(snapshot.messages_sent, 2);
  assertEquals(snapshot.messages_failed, 1);
  assertEquals(snapshot.bytes_sent, 300);
  assertEquals(snapshot.bytes_received, 150);
});

Deno.test("ProtocolMetrics - latency percentiles", () => {
  const metrics = new ProtocolMetrics();

  for (let i = 1; i <= 100; i++) {
    metrics.recordLatency(i);
  }

  const snapshot = metrics.snapshot();
  const p50 = snapshot.latency_p50!;
  const p95 = snapshot.latency_p95!;
  const p99 = snapshot.latency_p99!;
  assert(p50 >= 49 && p50 <= 51, `p50 should be around 50, got ${p50}`);
  assert(p95 >= 94 && p95 <= 96, `p95 should be around 95, got ${p95}`);
  assert(p99 >= 98 && p99 <= 100, `p99 should be around 99, got ${p99}`);
});

Deno.test("ProtocolMetrics - compression ratio", () => {
  const metrics = new ProtocolMetrics();
  metrics.recordCompression(1000, 400);

  const snapshot = metrics.snapshot();
  assert(Math.abs(snapshot.compression_ratio - 0.4) < 0.01);
});

Deno.test("ProtocolMetrics - reset", () => {
  const metrics = new ProtocolMetrics();
  metrics.recordConnectionStart();
  metrics.recordMessageSent(100);
  metrics.recordLatency(50);

  metrics.reset();

  const snapshot = metrics.snapshot();
  assertEquals(snapshot.messages_sent, 0);
  assertEquals(snapshot.bytes_sent, 0);
  assertEquals(snapshot.latency_p50, undefined);
});

Deno.test("assessConnectionQuality", () => {
  // Not enough data
  const unknown: MetricsSnapshot = {
    messages_sent: 5,
    messages_failed: 0,
    bytes_sent: 0,
    bytes_received: 0,
    compression_ratio: 1.0,
    uptime_secs: 0,
    idle_secs: 0,
  };
  assertEquals(assessConnectionQuality(unknown), "unknown");

  // Excellent
  const excellent: MetricsSnapshot = {
    ...unknown,
    messages_sent: 100,
    messages_failed: 0,
    latency_p95: 30,
  };
  assertEquals(assessConnectionQuality(excellent), "excellent");

  // Fair
  const fair: MetricsSnapshot = { ...excellent, latency_p95: 120 };
  assertEquals(assessConnectionQuality(fair), "fair");

  // Poor
  const poor: MetricsSnapshot = { ...excellent, messages_failed: 15 };
  assertEquals(assessConnectionQuality(poor), "poor");
});

// ============================================================================
// Bridge Tests
// ============================================================================

Deno.test("BridgeConfig defaults: no backend URL is baked in, and the bridge refuses to start without one", () => {
  const config = defaultBridgeConfig();
  assertEquals(config.backendUrl, "");
  assertEquals(config.heartbeatIntervalSecs, 5);
  assertEquals(config.version, "unknown");
  assertThrows(() => new RemoteBridge(config), Error, "backendUrl");
});

/** A config pointing at a relay of our own. */
const relayConfig = () => ({
  ...defaultBridgeConfig(),
  backendUrl: "https://relay.test",
});

Deno.test("RemoteBridge - initial state", () => {
  const bridge = new RemoteBridge(relayConfig());
  assertEquals(bridge.state, "disconnected");
  assert(!bridge.isReady);
  assertEquals(bridge.getUserId(), undefined);
});

Deno.test("RemoteBridge - protocol version", () => {
  const bridge = new RemoteBridge(relayConfig());
  assertEquals(bridge.protocolVersion(), PROTOCOL_VERSION);
});

Deno.test("RemoteBridge - capabilities", () => {
  const bridge = new RemoteBridge(relayConfig());
  // Default capabilities
  assert(bridge.hasCapability("streaming"));
  assert(bridge.hasCapability("tools"));
  assert(!bridge.hasCapability("compression"));
});

Deno.test("RemoteBridge - command result queue", () => {
  const bridge = new RemoteBridge(relayConfig());

  bridge.queueCommandResult({ type: "pong", timestamp: 12345 });
  // Internal queue is not directly accessible, but we verify it doesn't throw.
});

Deno.test("RemoteBridge - queueResult helper", () => {
  const bridge = new RemoteBridge(relayConfig());

  bridge.queueResult("cmd-1", { ok: true, value: { done: true } });
  bridge.queueResult("cmd-2", { ok: false, error: "something went wrong" });
  // No throw = success
});

Deno.test("RemoteBridge - shutdown sets state", () => {
  const bridge = new RemoteBridge(relayConfig());
  bridge.shutdown();
  assertEquals(bridge.state, "shutting_down");
});

Deno.test("RemoteBridge - state change handler", () => {
  const bridge = new RemoteBridge(relayConfig());
  const states: string[] = [];
  bridge.setStateChangeHandler((state) => states.push(state));

  bridge.shutdown();
  assertEquals(states, ["shutting_down"]);
});

// ============================================================================
// Manager Tests
// ============================================================================

Deno.test("RemoteBridgeManager - not running by default", () => {
  const manager = new RemoteBridgeManager({
    configProvider: {
      getRemoteConfig: () => undefined,
      getApiKey: () => undefined,
    },
    version: "0.1.0-test",
  });
  assert(!manager.isRunning());
});

Deno.test("RemoteBridgeManager - isEnabled", () => {
  const disabled = new RemoteBridgeManager({
    configProvider: {
      getRemoteConfig: () => undefined,
      getApiKey: () => undefined,
    },
    version: "0.1.0",
  });
  assert(!disabled.isEnabled());

  const enabled = new RemoteBridgeManager({
    configProvider: {
      getRemoteConfig: () => ({
        backendUrl: "https://test.example.com",
        apiKey: "test-key",
        heartbeatIntervalSecs: 5,
        reconnectDelaySecs: 5,
        maxReconnectAttempts: 3,
      }),
      getApiKey: () => "test-api-key",
    },
    version: "0.1.0",
  });
  assert(enabled.isEnabled());
});

Deno.test("RemoteBridgeManager - status when not running", () => {
  const manager = new RemoteBridgeManager({
    configProvider: {
      getRemoteConfig: () => undefined,
      getApiKey: () => undefined,
    },
    version: "0.1.0",
  });
  assertEquals(manager.status().kind, "disconnected");
});

Deno.test("displayBridgeStatus", () => {
  assertEquals(displayBridgeStatus({ kind: "disconnected" }), "Disconnected");
  assertEquals(displayBridgeStatus({ kind: "connected" }), "Connected");
  assertEquals(displayBridgeStatus({ kind: "authenticated" }), "Authenticated");
  assertEquals(
    displayBridgeStatus({ kind: "error", message: "timeout" }),
    "Error: timeout",
  );
});

Deno.test("RemoteBridgeManager - startFromConfig with no config returns false", async () => {
  const manager = new RemoteBridgeManager({
    configProvider: {
      getRemoteConfig: () => undefined,
      getApiKey: () => undefined,
    },
    version: "0.1.0",
  });
  const started = await manager.startFromConfig();
  assert(!started);
});

Deno.test("RemoteBridgeManager - stop when not running", async () => {
  const manager = new RemoteBridgeManager({
    configProvider: {
      getRemoteConfig: () => undefined,
      getApiKey: () => undefined,
    },
    version: "0.1.0",
  });
  await manager.stop(); // Should not throw
});
