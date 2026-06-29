// Example: Network Manager — agent identities, messaging, and peer discovery
// Demonstrates creating agent identities with capability cards, discovering
// peers via ManualDiscovery, constructing message envelopes (direct, broadcast,
// topic, reply), and using the PeerTable for routing decisions.
// Run: deno run deno/examples/agent-network/network_manager.ts

import {
  type AgentIdentity,
  broadcastEnvelope,
  BroadcastRouter,
  ContentRouter,
  createAgentIdentity,
  defaultAgentCard,
  directEnvelope,
  DirectRouter,
  displayTransportAddress,
  hasCapability,
  jsonPayload,
  ManualDiscovery,
  PeerTable,
  replyEnvelope,
  supportsProtocol,
  textPayload,
  topicEnvelope,
  type TransportAddress,
  withCorrelation,
  withTtl,
} from "@rullama/network";

async function main(): Promise<void> {
  console.log("=== Network Manager Example ===\n");

  // ---------------------------------------------------------------------------
  // 1. Create agent identities with capability cards
  // ---------------------------------------------------------------------------
  console.log("--- Agent Identities ---");

  const orchestrator: AgentIdentity = {
    ...createAgentIdentity("orchestrator"),
    agentCard: {
      ...defaultAgentCard(),
      capabilities: ["task-routing", "load-balancing"],
      supportedProtocols: ["mcp", "ipc"],
      endpoint: "unix:///tmp/orchestrator.sock",
      maxConcurrentTasks: 50,
      computeCapacity: 1.0,
    },
  };
  console.log(
    `  ${orchestrator.name} (id=${orchestrator.id})` +
      `\n    capabilities: ${
        JSON.stringify(orchestrator.agentCard.capabilities)
      }` +
      `\n    protocols:    ${
        JSON.stringify(orchestrator.agentCard.supportedProtocols)
      }` +
      `\n    endpoint:     ${orchestrator.agentCard.endpoint}`,
  );

  const workerA: AgentIdentity = {
    ...createAgentIdentity("worker-alpha"),
    agentCard: {
      ...defaultAgentCard(),
      capabilities: ["code-generation"],
      supportedProtocols: ["mcp", "ipc"],
      endpoint: "unix:///tmp/worker-a.sock",
      maxConcurrentTasks: 10,
      computeCapacity: 0.6,
    },
  };
  console.log(
    `  ${workerA.name} (id=${workerA.id})` +
      `\n    capabilities: ${JSON.stringify(workerA.agentCard.capabilities)}`,
  );

  const workerB: AgentIdentity = {
    ...createAgentIdentity("worker-beta"),
    agentCard: {
      ...defaultAgentCard(),
      capabilities: ["code-review", "testing"],
      supportedProtocols: ["mcp", "a2a"],
      endpoint: "tcp://127.0.0.1:9091",
      maxConcurrentTasks: 5,
      computeCapacity: 0.8,
    },
  };
  console.log(
    `  ${workerB.name} (id=${workerB.id})` +
      `\n    capabilities: ${JSON.stringify(workerB.agentCard.capabilities)}`,
  );
  console.log();

  // ---------------------------------------------------------------------------
  // 2. Check capabilities and protocol support
  // ---------------------------------------------------------------------------
  console.log("--- Capability & Protocol Checks ---");

  console.log(
    `  orchestrator has "task-routing": ${
      hasCapability(orchestrator.agentCard, "task-routing")
    }`,
  );
  console.log(
    `  workerA supports "mcp": ${supportsProtocol(workerA.agentCard, "mcp")}`,
  );
  console.log(
    `  workerB supports "ipc": ${supportsProtocol(workerB.agentCard, "ipc")}`,
  );
  console.log(
    `  workerB supports "a2a": ${supportsProtocol(workerB.agentCard, "a2a")}`,
  );
  console.log();

  // ---------------------------------------------------------------------------
  // 3. Peer discovery with ManualDiscovery
  // ---------------------------------------------------------------------------
  console.log("--- Peer Discovery ---");

  const discovery = ManualDiscovery.withPeers([workerA, workerB]);

  // Register orchestrator too
  await discovery.register(orchestrator);

  const peers = await discovery.discover();
  console.log(`  Discovered ${peers.length} peer(s):`);
  for (const peer of peers) {
    console.log(
      `    ${peer.name} — protocols: ${
        JSON.stringify(peer.agentCard.supportedProtocols)
      }, ` +
        `endpoint: ${peer.agentCard.endpoint ?? "none"}`,
    );
  }

  // Look up a specific peer
  const found = await discovery.lookup(workerA.id);
  console.log(`  Lookup workerA: ${found ? found.name : "not found"}`);
  console.log(`  Discovery protocol: ${JSON.stringify(discovery.protocol())}`);
  console.log();

  // ---------------------------------------------------------------------------
  // 4. Message envelope construction
  // ---------------------------------------------------------------------------
  console.log("--- Message Envelopes ---");

  // Direct message to worker-alpha
  const direct = directEnvelope(
    orchestrator.id,
    workerA.id,
    textPayload("Please generate a Rust HTTP handler"),
  );
  console.log(
    `  Direct:    sender=${direct.sender.slice(0, 8)}..., ` +
      `recipient=${JSON.stringify(direct.recipient)}, payload=Text("...")`,
  );

  // Broadcast to all peers
  const broadcast = broadcastEnvelope(
    orchestrator.id,
    textPayload("System: reloading config"),
  );
  console.log(
    `  Broadcast: sender=${broadcast.sender.slice(0, 8)}..., ` +
      `recipient=${JSON.stringify(broadcast.recipient)}`,
  );

  // Topic-addressed message
  const topic = topicEnvelope(
    orchestrator.id,
    "build-events",
    jsonPayload({ status: "success", duration_ms: 1234 }),
  );
  console.log(
    `  Topic:     sender=${topic.sender.slice(0, 8)}..., ` +
      `recipient=${JSON.stringify(topic.recipient)}`,
  );

  // Reply to the direct message
  const reply = replyEnvelope(
    direct,
    workerA.id,
    textPayload("Handler generated!"),
  );
  console.log(
    `  Reply:     sender=${reply.sender.slice(0, 8)}..., ` +
      `correlationId=${reply.correlationId?.slice(0, 8)}...`,
  );

  // TTL-limited message
  const ttlMsg = withTtl(
    broadcastEnvelope(orchestrator.id, textPayload("heartbeat")),
    3,
  );
  console.log(`  TTL msg:   ttl=${ttlMsg.ttl}`);

  // Correlation ID
  const correlated = withCorrelation(direct, "req-001");
  console.log(`  Correlated: correlationId=${correlated.correlationId}`);
  console.log();

  // ---------------------------------------------------------------------------
  // 5. PeerTable and routing
  // ---------------------------------------------------------------------------
  console.log("--- PeerTable & Routing ---");

  const peerTable = new PeerTable();

  const workerAAddr: TransportAddress = {
    type: "unix",
    path: "/tmp/worker-a.sock",
  };
  const workerBAddr: TransportAddress = {
    type: "tcp",
    address: "127.0.0.1:9091",
  };

  peerTable.upsert(workerA, [workerAAddr]);
  peerTable.upsert(workerB, [workerBAddr]);
  console.log(`  PeerTable: ${peerTable.length} peers`);

  for (const peer of peerTable.allPeers()) {
    const addrs = peerTable.getAddresses(peer.id) ?? [];
    const addrStrs = addrs.map(displayTransportAddress);
    console.log(`    ${peer.name} -> ${addrStrs.join(", ")}`);
  }

  // Topic subscriptions
  peerTable.subscribe(workerA.id, "build-events");
  peerTable.subscribe(workerB.id, "build-events");
  peerTable.subscribe(workerB.id, "review-events");
  console.log(
    `  Subscribers to "build-events": ${
      peerTable.subscribers("build-events").length
    }`,
  );
  console.log(
    `  Subscribers to "review-events": ${
      peerTable.subscribers("review-events").length
    }`,
  );
  console.log();

  // ---------------------------------------------------------------------------
  // 6. Route messages using different strategies
  // ---------------------------------------------------------------------------
  console.log("--- Routing Strategies ---");

  // Direct routing
  const directRouter = new DirectRouter();
  console.log(
    `  DirectRouter strategy: ${JSON.stringify(directRouter.strategy())}`,
  );
  const directAddrs = await directRouter.route(direct, peerTable);
  console.log(
    `    Route direct msg -> ${
      directAddrs.map(displayTransportAddress).join(", ")
    }`,
  );

  // Broadcast routing
  const broadcastRouter = new BroadcastRouter();
  console.log(
    `  BroadcastRouter strategy: ${JSON.stringify(broadcastRouter.strategy())}`,
  );
  const broadcastAddrs = await broadcastRouter.route(broadcast, peerTable);
  console.log(
    `    Route broadcast -> ${
      broadcastAddrs.map(displayTransportAddress).join(", ")
    }`,
  );

  // Content-based routing
  const contentRouter = new ContentRouter();
  console.log(
    `  ContentRouter strategy: ${JSON.stringify(contentRouter.strategy())}`,
  );
  const topicAddrs = await contentRouter.route(topic, peerTable);
  console.log(
    `    Route topic "build-events" -> ${
      topicAddrs.map(displayTransportAddress).join(", ")
    }`,
  );

  console.log("\nDone.");
}

await main();
