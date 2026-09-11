/**
 * @module @rullama/network
 *
 * Agent-to-agent networking layer: agent identity + capability cards,
 * message envelopes, routing strategies, a peer table, peer discovery,
 * an agent-management contract, the remote relay bridge (command queue,
 * heartbeat telemetry, protocol negotiation) and a stdio relay client.
 *
 * The MCP server framework lives in `@rullama/mcp-server` — import from
 * there directly. `AgentNetworkError` + `ErrorCode` are re-exported here for
 * convenience (the underlying class lives in `@rullama/mcp-server`).
 */

export { AgentNetworkError, ErrorCode } from "./error.ts";

// =============================================================================
// Identity Layer
// =============================================================================

export {
  type AgentCard,
  type AgentIdentity,
  createAgentIdentity,
  createAgentIdentityWithId,
  defaultAgentCard,
  hasCapability,
  type ProtocolId,
  supportsProtocol,
} from "./identity.ts";

// =============================================================================
// Network Core Types
// =============================================================================

export {
  binaryPayload,
  broadcastEnvelope,
  directEnvelope,
  jsonPayload,
  type MessageEnvelope,
  type MessageTarget,
  type Payload,
  replyEnvelope,
  textPayload,
  topicEnvelope,
  withCorrelation,
  withTtl,
} from "./envelope.ts";

// =============================================================================
// Routing Layer
// =============================================================================

export {
  BroadcastRouter,
  ContentRouter,
  DirectRouter,
  type Router,
  type RoutingStrategy,
} from "./routing.ts";

export {
  displayTransportAddress,
  PeerTable,
  type TransportAddress,
} from "./peer_table.ts";

// =============================================================================
// Discovery Layer
// =============================================================================

export {
  type Discovery,
  type DiscoveryProtocol,
  ManualDiscovery,
} from "./discovery.ts";

// =============================================================================
// Agent Management
// =============================================================================

export {
  type AgentInfo,
  type AgentManager,
  type AgentResult,
  type SpawnConfig,
} from "./agent_manager.ts";

export { AgentToolRegistry } from "./agent_tools.ts";

// =============================================================================
// Client
// =============================================================================

export {
  type AgentConfig,
  AgentNetworkClient,
  AgentNetworkClientError,
} from "./client.ts";

// =============================================================================
// Remote Bridge
// =============================================================================

export {
  allSupportedCapabilities,
  assessConnectionQuality,
  // Command queue
  CommandQueue,
  defaultBridgeConfig,
  defaultProtocolAccept,
  defaultProtocolHello,
  defaultRetryPolicy,
  displayBridgeStatus,
  // Heartbeat & telemetry
  HeartbeatCollector,
  MIN_PROTOCOL_VERSION,
  NegotiatedProtocol,
  PRIORITY_ORDER,
  // Protocol
  PROTOCOL_VERSION,
  ProtocolMetrics,
  QueueEntry,
  QueueError,
  // Bridge
  RemoteBridge,
  // Manager
  RemoteBridgeManager,
  SUPPORTED_VERSIONS,
} from "./remote/mod.ts";

export type {
  AgentEvent,
  AgentEventType,
  AgentInfoProvider,
  BackendCommand,
  BackendCommand_AttachmentChunk,
  BackendCommand_AttachmentComplete,
  BackendCommand_AttachmentUpload,
  BackendCommand_Authenticated,
  BackendCommand_AuthenticationFailed,
  BackendCommand_CancelOperation,
  BackendCommand_Disconnect,
  BackendCommand_Ping,
  BackendCommand_RequestSync,
  BackendCommand_SendInput,
  BackendCommand_SlashCommand,
  BackendCommand_SpawnAgent,
  BackendCommand_Subscribe,
  BackendCommand_Unsubscribe,
  // Bridge types
  BridgeConfig,
  BridgeConfigProvider,
  BridgeState,
  CommandHandler,
  CommandPriority,
  CompressionAlgorithm,
  ConnectionMode,
  ConnectionQuality,
  // Heartbeat types
  HeartbeatData,
  MetricsSnapshot,
  PrioritizedCommand,
  ProtocolAccept,
  // Protocol types
  ProtocolCapability,
  ProtocolHello,
  QueueStats,
  RemoteAgentInfo,
  // Manager types
  RemoteBridgeConfig,
  RemoteBridgeStatus,
  RemoteMessage,
  RemoteMessage_AgentEvent,
  RemoteMessage_AgentStream,
  RemoteMessage_AttachmentReceived,
  RemoteMessage_CommandResult,
  RemoteMessage_Heartbeat,
  RemoteMessage_Pong,
  RemoteMessage_Register,
  RetryPolicy,
  StateChangeHandler,
  StreamChunkType,
} from "./remote/mod.ts";
