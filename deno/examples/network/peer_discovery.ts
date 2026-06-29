// Example: Peer Discovery — ManualDiscovery, PeerTable, and routing strategies
// Demonstrates the discovery layer (register, deregister, lookup), PeerTable
// management (upsert, remove, topic subscriptions), and routing strategies
// (direct, broadcast, content-based) for message delivery.
// Run: deno run deno/examples/agent-network/peer_discovery.ts

import {
  type AgentIdentity,
  broadcastEnvelope,
  BroadcastRouter,
  ContentRouter,
  createAgentIdentity,
  defaultAgentCard,
  directEnvelope,
  DirectRouter,
  type Discovery,
  type DiscoveryProtocol,
  displayTransportAddress,
  hasCapability,
  jsonPayload,
  ManualDiscovery,
  PeerTable,
  supportsProtocol,
  textPayload,
  topicEnvelope,
  type TransportAddress,
} from "@rullama/network";

async function main(): Promise<void> {
  console.log("=== Peer Discovery Example ===\n");

  // ---------------------------------------------------------------------------
  // 1. Create a set of agents with diverse capabilities
  // ---------------------------------------------------------------------------
  console.log("--- Create Agents ---");

  const coordinator: AgentIdentity = {
    ...createAgentIdentity("coordinator"),
    agentCard: {
      ...defaultAgentCard(),
      capabilities: ["task-routing", "monitoring"],
      supportedProtocols: ["mcp", "ipc"],
      endpoint: "unix:///tmp/coordinator.sock",
      maxConcurrentTasks: 100,
      computeCapacity: 1.0,
    },
  };

  const coder: AgentIdentity = {
    ...createAgentIdentity("coder"),
    agentCard: {
      ...defaultAgentCard(),
      capabilities: ["code-generation", "refactoring"],
      supportedProtocols: ["mcp"],
      endpoint: "tcp://127.0.0.1:9001",
      maxConcurrentTasks: 10,
      computeCapacity: 0.7,
    },
  };

  const reviewer: AgentIdentity = {
    ...createAgentIdentity("reviewer"),
    agentCard: {
      ...defaultAgentCard(),
      capabilities: ["code-review", "testing"],
      supportedProtocols: ["mcp", "a2a"],
      endpoint: "tcp://127.0.0.1:9002",
      maxConcurrentTasks: 8,
      computeCapacity: 0.8,
    },
  };

  const deployer: AgentIdentity = {
    ...createAgentIdentity("deployer"),
    agentCard: {
      ...defaultAgentCard(),
      capabilities: ["deployment", "monitoring"],
      supportedProtocols: ["mcp", "ipc"],
      endpoint: "unix:///tmp/deployer.sock",
      maxConcurrentTasks: 5,
      computeCapacity: 0.5,
    },
  };

  const agents = [coordinator, coder, reviewer, deployer];
  for (const agent of agents) {
    console.log(
      `  ${agent.name}: capabilities=${
        JSON.stringify(agent.agentCard.capabilities)
      }, ` +
        `protocols=${JSON.stringify(agent.agentCard.supportedProtocols)}`,
    );
  }
  console.log();

  // ---------------------------------------------------------------------------
  // 2. ManualDiscovery — register, discover, lookup, deregister
  // ---------------------------------------------------------------------------
  console.log("--- ManualDiscovery Lifecycle ---");

  const discovery: Discovery = ManualDiscovery.withPeers([coder, reviewer]);
  console.log(`  Protocol: ${JSON.stringify(discovery.protocol())}`);

  // Initial discovery
  let found: AgentIdentity[] = await discovery.discover();
  console.log(
    `  Initial peers: ${found.map((p: AgentIdentity) => p.name).join(", ")}`,
  );

  // Register more agents
  await discovery.register(deployer);
  await discovery.register(coordinator);
  found = await discovery.discover();
  console.log(
    `  After registering deployer + coordinator: ${
      found.map((p: AgentIdentity) => p.name).join(", ")
    }`,
  );

  // Lookup by ID
  const lookedUp = await discovery.lookup(reviewer.id);
  console.log(
    `  Lookup reviewer by ID: ${lookedUp ? lookedUp.name : "not found"}`,
  );

  const missing = await discovery.lookup("nonexistent-id");
  console.log(
    `  Lookup nonexistent ID: ${missing ? missing.name : "not found"}`,
  );

  // Deregister
  await discovery.deregister(deployer.id);
  found = await discovery.discover();
  console.log(
    `  After deregistering deployer: ${
      found.map((p: AgentIdentity) => p.name).join(", ")
    }`,
  );
  console.log();

  // ---------------------------------------------------------------------------
  // 3. Capability-based filtering
  // ---------------------------------------------------------------------------
  console.log("--- Capability-Based Filtering ---");

  const allAgents: AgentIdentity[] = await discovery.discover();

  const coders = allAgents.filter((a: AgentIdentity) =>
    hasCapability(a.agentCard, "code-generation")
  );
  console.log(
    `  Agents with "code-generation": ${
      coders.map((a: AgentIdentity) => a.name).join(", ") || "none"
    }`,
  );

  const reviewers = allAgents.filter((a: AgentIdentity) =>
    hasCapability(a.agentCard, "code-review")
  );
  console.log(
    `  Agents with "code-review": ${
      reviewers.map((a: AgentIdentity) => a.name).join(", ") || "none"
    }`,
  );

  const monitors = allAgents.filter((a: AgentIdentity) =>
    hasCapability(a.agentCard, "monitoring")
  );
  console.log(
    `  Agents with "monitoring": ${
      monitors.map((a: AgentIdentity) => a.name).join(", ") || "none"
    }`,
  );

  const a2aAgents = allAgents.filter((a: AgentIdentity) =>
    supportsProtocol(a.agentCard, "a2a")
  );
  console.log(
    `  Agents supporting "a2a": ${
      a2aAgents.map((a: AgentIdentity) => a.name).join(", ") || "none"
    }`,
  );

  const ipcAgents = allAgents.filter((a: AgentIdentity) =>
    supportsProtocol(a.agentCard, "ipc")
  );
  console.log(
    `  Agents supporting "ipc": ${
      ipcAgents.map((a: AgentIdentity) => a.name).join(", ") || "none"
    }`,
  );
  console.log();

  // ---------------------------------------------------------------------------
  // 4. PeerTable — upsert, addresses, topic subscriptions
  // ---------------------------------------------------------------------------
  console.log("--- PeerTable Management ---");

  const peerTable = new PeerTable();

  // Map each agent to transport addresses
  const addressMap: [AgentIdentity, TransportAddress[]][] = [
    [coordinator, [{ type: "unix", path: "/tmp/coordinator.sock" }]],
    [coder, [{ type: "tcp", address: "127.0.0.1:9001" }]],
    [reviewer, [
      { type: "tcp", address: "127.0.0.1:9002" },
      { type: "url", url: "https://reviewer.internal:443" },
    ]],
  ];

  for (const [agent, addrs] of addressMap) {
    peerTable.upsert(agent, addrs);
  }

  console.log(`  Peers in table: ${peerTable.length}`);
  console.log(`  Table is empty: ${peerTable.isEmpty}`);

  for (const peer of peerTable.allPeers()) {
    const addrs = peerTable.getAddresses(peer.id) ?? [];
    console.log(
      `    ${peer.name} -> [${addrs.map(displayTransportAddress).join(", ")}]`,
    );
  }

  // Topic subscriptions
  peerTable.subscribe(coder.id, "build-events");
  peerTable.subscribe(reviewer.id, "build-events");
  peerTable.subscribe(reviewer.id, "review-requests");
  peerTable.subscribe(coordinator.id, "build-events");
  peerTable.subscribe(coordinator.id, "review-requests");

  console.log(
    `  "build-events" subscribers: ${
      peerTable.subscribers("build-events").length
    }`,
  );
  console.log(
    `  "review-requests" subscribers: ${
      peerTable.subscribers("review-requests").length
    }`,
  );

  // Unsubscribe and check
  peerTable.unsubscribe(coordinator.id, "build-events");
  console.log(
    `  "build-events" after unsubscribe coordinator: ${
      peerTable.subscribers("build-events").length
    }`,
  );

  // Remove a peer entirely
  const removed = peerTable.remove(coder.id);
  console.log(`  Removed peer: ${removed?.name ?? "none"}`);
  console.log(`  Peers remaining: ${peerTable.length}`);
  console.log(
    `  "build-events" after removing coder: ${
      peerTable.subscribers("build-events").length
    }`,
  );
  console.log();

  // ---------------------------------------------------------------------------
  // 5. Re-populate and demonstrate routing strategies
  // ---------------------------------------------------------------------------
  console.log("--- Routing Strategies ---");

  // Re-add coder for routing demo
  peerTable.upsert(coder, [{ type: "tcp", address: "127.0.0.1:9001" }]);
  peerTable.subscribe(coder.id, "build-events");

  // Direct routing
  const directRouter = new DirectRouter();
  const directMsg = directEnvelope(
    coordinator.id,
    coder.id,
    textPayload("Please implement the API handler"),
  );
  const directAddrs = await directRouter.route(directMsg, peerTable);
  console.log(`  DirectRouter (${JSON.stringify(directRouter.strategy())}):`);
  console.log(`    -> ${directAddrs.map(displayTransportAddress).join(", ")}`);

  // Broadcast routing
  const broadcastRouter = new BroadcastRouter();
  const broadcastMsg = broadcastEnvelope(
    coordinator.id,
    textPayload("System: maintenance window starting"),
  );
  const broadcastAddrs = await broadcastRouter.route(broadcastMsg, peerTable);
  console.log(
    `  BroadcastRouter (${JSON.stringify(broadcastRouter.strategy())}):`,
  );
  console.log(
    `    -> ${broadcastAddrs.map(displayTransportAddress).join(", ")}`,
  );

  // Content-based routing
  const contentRouter = new ContentRouter();
  const topicMsg = topicEnvelope(
    coordinator.id,
    "build-events",
    jsonPayload({ status: "success", commit: "abc123", duration_ms: 4500 }),
  );
  const topicAddrs = await contentRouter.route(topicMsg, peerTable);
  console.log(`  ContentRouter (${JSON.stringify(contentRouter.strategy())}):`);
  console.log(`    -> ${topicAddrs.map(displayTransportAddress).join(", ")}`);

  // Content-based routing for review-requests topic
  const reviewMsg = topicEnvelope(
    coder.id,
    "review-requests",
    jsonPayload({ pr: 42, title: "Add caching layer" }),
  );
  const reviewAddrs = await contentRouter.route(reviewMsg, peerTable);
  console.log(`  ContentRouter for "review-requests":`);
  console.log(`    -> ${reviewAddrs.map(displayTransportAddress).join(", ")}`);

  // ---------------------------------------------------------------------------
  // 6. Error handling for routing mismatches
  // ---------------------------------------------------------------------------
  console.log();
  console.log("--- Routing Error Handling ---");

  try {
    // DirectRouter cannot handle broadcast messages
    await directRouter.route(broadcastMsg, peerTable);
    console.log("  DirectRouter + broadcast -> OK (unexpected)");
  } catch (e) {
    console.log(
      `  DirectRouter + broadcast -> Error: ${
        e instanceof Error ? e.message : e
      }`,
    );
  }

  try {
    // ContentRouter cannot handle direct messages
    await contentRouter.route(directMsg, peerTable);
    console.log("  ContentRouter + direct -> OK (unexpected)");
  } catch (e) {
    console.log(
      `  ContentRouter + direct -> Error: ${
        e instanceof Error ? e.message : e
      }`,
    );
  }

  console.log("\nDone.");
}

await main();
