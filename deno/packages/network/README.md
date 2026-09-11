# @rullama/network

Agent-to-agent networking layer for the rullama. Provides agent identity and
capability cards, message envelopes, routing strategies, a peer table, peer
discovery, an agent-management contract, a remote relay bridge (priority command
queue, heartbeat telemetry, protocol negotiation) and a stdio relay client.

Equivalent to the Rust `rullama-network` crate.

The MCP server framework (`McpServer`, tool registry, middleware) is **not** in
this package — it lives in `@rullama/mcp-server`. `AgentNetworkError` and
`ErrorCode` are re-exported here for convenience.

> **Transport security.** `@rullama/network` is an _unauthenticated_ transport
> layer: `MessageEnvelope.sender` is a self-asserted string and envelopes carry
> no signature, nonce or timestamp check. Use it only between processes you
> already trust (one host, one operator), or wrap it in an authenticated channel
> (mTLS, an SSH tunnel, an authenticated WebSocket). Signed envelopes with
> replay protection are planned for 0.13.

## Install

```sh
deno add @rullama/network
```

## Quick Example

Identity, peer table, envelopes and routing — the in-process core:

```ts
import {
  BroadcastRouter,
  ContentRouter,
  createAgentIdentity,
  directEnvelope,
  DirectRouter,
  displayTransportAddress,
  jsonPayload,
  ManualDiscovery,
  PeerTable,
  textPayload,
  topicEnvelope,
  withTtl,
} from "@rullama/network";

// Identities: a random UUID plus a capability card.
const me = createAgentIdentity("coordinator");
const reviewer = createAgentIdentity("code-review-agent");
reviewer.agentCard.capabilities.push("code-review");

// Peer table: who is reachable, and how.
const peers = new PeerTable();
peers.upsert(me, [{ type: "channel", channel: "coordinator" }]);
peers.upsert(reviewer, [{ type: "unix", path: "/tmp/reviewer.sock" }]);
peers.subscribe(reviewer.id, "reviews");

// Point-to-point: DirectRouter resolves the recipient's addresses.
const direct = directEnvelope(
  me.id,
  reviewer.id,
  jsonPayload({ task: "review PR #42" }),
);
const addrs = await new DirectRouter().route(direct, peers);
console.log(addrs.map(displayTransportAddress)); // ["unix:///tmp/reviewer.sock"]

// Topic: ContentRouter fans out to subscribers (excluding the sender).
const topic = withTtl(topicEnvelope(me.id, "reviews", textPayload("ping")), 3);
await new ContentRouter().route(topic, peers);

// Broadcast: every known peer except the sender.
new BroadcastRouter().strategy(); // { type: "broadcast" }

// Discovery: a static peer list you manage yourself.
const discovery = ManualDiscovery.withPeers([reviewer]);
await discovery.register(me);
const found = await discovery.lookup(reviewer.id);
console.log(found?.name); // "code-review-agent"
```

Remote bridge — connect local agents to a relay backend **you** run:

```ts
import {
  CommandQueue,
  defaultBridgeConfig,
  RemoteBridge,
  RemoteBridgeManager,
} from "@rullama/network";
import type { BackendCommand, RemoteAgentInfo } from "@rullama/network";

const bridge = new RemoteBridge({
  ...defaultBridgeConfig(),
  backendUrl: "https://relay.example.com", // must be http(s)://
  apiKey: Deno.env.get("RELAY_API_KEY") ?? "",
  version: "1.0.0",
  agentInfoProvider: (): RemoteAgentInfo[] => [],
});

const queue = new CommandQueue();
bridge.setCommandHandler((cmd: BackendCommand) => {
  queue.enqueue({ command: cmd, priority: "normal" });
});
bridge.setStateChangeHandler((state) => console.log("bridge:", state));

// bridge.run() registers, then heartbeats until bridge.shutdown().
// Or let a manager own the lifecycle:
const manager = new RemoteBridgeManager({
  version: "1.0.0",
  configProvider: {
    getRemoteConfig: () => ({
      backendUrl: "https://relay.example.com",
      apiKey: "",
      heartbeatIntervalSecs: 5,
      reconnectDelaySecs: 5,
      maxReconnectAttempts: 0,
    }),
    getApiKey: () => Deno.env.get("RELAY_API_KEY"),
  },
});
console.log(manager.isEnabled(), manager.status()); // true, { kind: "disconnected" }
```

The bridge registers via `POST {backendUrl}/api/remote/connect` and then polls
`POST {backendUrl}/api/remote/heartbeat` every `heartbeatIntervalSecs`, shipping
queued command results and executing any `BackendCommand`s the response carries.
Only the `"polling"` connection mode is implemented today; `"websocket"` is
declared for forward compatibility.

## Key Exports

| Export                                                                     | Description                                                                                |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `createAgentIdentity` / `AgentIdentity` / `AgentCard`                      | Agent identity (UUID + name) with a capability/protocol advertisement                      |
| `directEnvelope` / `broadcastEnvelope` / `topicEnvelope` / `replyEnvelope` | Build `MessageEnvelope`s; `withTtl` / `withCorrelation` add hop limits and correlation ids |
| `textPayload` / `jsonPayload` / `binaryPayload`                            | Envelope payload constructors                                                              |
| `DirectRouter` / `BroadcastRouter` / `ContentRouter`                       | Resolve an envelope to transport addresses (point-to-point, all peers, topic subscribers)  |
| `PeerTable` / `TransportAddress`                                           | Known peers, their addresses (`unix` / `tcp` / `url` / `channel`) and topic subscriptions  |
| `ManualDiscovery` / `Discovery`                                            | Static in-memory peer discovery; implement `Discovery` for registry/mDNS/gossip            |
| `AgentManager` / `SpawnConfig` / `AgentInfo` / `AgentResult`               | Contract for spawning and tracking task agents                                             |
| `AgentToolRegistry`                                                        | Pre-built MCP tool definitions (`agent_spawn`, `agent_status`, …) for agent ops            |
| `RemoteBridge` / `RemoteBridgeManager`                                     | HTTP-polling bridge to a relay backend, plus lifecycle/status management                   |
| `CommandQueue` / `QueueEntry` / `PrioritizedCommand`                       | Priority queue for backend commands with deadlines and retry policies                      |
| `HeartbeatCollector` / `ProtocolMetrics` / `assessConnectionQuality`       | Agent-roster diffing (`spawned` / `exited` / `busy` / …) and connection telemetry          |
| `PROTOCOL_VERSION` / `NegotiatedProtocol` / `ProtocolHello`                | Relay wire-protocol version and capability negotiation                                     |
| `AgentNetworkClient`                                                       | Spawns a relay subprocess and talks JSON-RPC to it over stdio                              |
| `AgentNetworkError` / `ErrorCode`                                          | Re-exported from `@rullama/mcp-server`                                                     |
